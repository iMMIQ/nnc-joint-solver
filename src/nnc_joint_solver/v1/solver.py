"""Version 1 solver with recipe search and critical-path scheduling."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
import math

from nnc_joint_solver.base import JointScheduleSolver
from nnc_joint_solver.ir.joint_tiling_schedule import (
    JointAction,
    JointActionKind,
    JointFailure,
    JointFailureCategory,
    JointFailureStatus,
    JointProblem,
    JointRecipe,
    JointSolution,
)
from nnc_joint_solver.solve_utils import (
    action_cost,
    build_successor_ids,
    recipe_cost,
    solve_recipe_selection,
    solver_failure,
    topological_action_order,
)
from nnc_joint_solver.validation import validate_joint_problem


@dataclass(frozen=True)
class _SearchState:
    recipe_by_region: dict[str, str]
    lower_bound: int
    partial_cost: int


class V1JointScheduleSolver(JointScheduleSolver):
    """Searches recipe combinations and refines them with local objective search."""

    def __init__(
        self,
        *,
        beam_width: int = 64,
        exhaustive_limit: int = 4096,
        max_local_passes: int = 4,
    ) -> None:
        self.beam_width = max(int(beam_width), 1)
        self.exhaustive_limit = max(int(exhaustive_limit), 1)
        self.max_local_passes = max(int(max_local_passes), 0)

    def solve(self, problem: JointProblem) -> JointSolution | JointFailure:
        problem_failure = validate_joint_problem(problem)
        if problem_failure is not None:
            return problem_failure

        recipes_by_region = _recipes_by_region(problem)
        missing_regions = [
            region.region_id
            for region in problem.regions
            if region.region_id not in recipes_by_region
        ]
        if missing_regions:
            return solver_failure(
                JointFailureStatus.INVALID_PROBLEM,
                JointFailureCategory.INVALID_SOLUTION,
                f"regions without recipes: {missing_regions}",
            )

        region_order = _topological_region_order(problem)
        boundary_pairs = _compatible_recipe_pairs(problem)
        min_cost_by_region = {
            region_id: min(recipe_cost(recipe) for recipe in region_recipes)
            for region_id, region_recipes in recipes_by_region.items()
        }
        region_successors = {
            region.region_id: tuple(region.successor_region_ids)
            for region in problem.regions
        }
        boundary_neighbors = _boundary_neighbor_pairs(problem)

        best_assignment: dict[str, str] | None = None
        best_objective = math.inf
        evaluated_assignments = 0
        search_mode = "beam"

        total_recipe_combinations = 1
        for region in problem.regions:
            total_recipe_combinations *= len(recipes_by_region[region.region_id])
            if total_recipe_combinations > self.exhaustive_limit:
                break

        if total_recipe_combinations <= self.exhaustive_limit:
            search_mode = "exhaustive"

            def dfs(index: int, selected: dict[str, str]) -> None:
                nonlocal best_assignment, best_objective, evaluated_assignments
                lower_bound = _estimate_lower_bound(
                    region_order=region_order,
                    region_successors=region_successors,
                    selected=selected,
                    min_cost_by_region=min_cost_by_region,
                    recipes_by_region=recipes_by_region,
                )
                if lower_bound >= best_objective:
                    return
                if index >= len(region_order):
                    evaluated_assignments += 1
                    result = _evaluate_assignment(problem, selected)
                    if isinstance(result, JointSolution) and result.objective_value < best_objective:
                        best_objective = result.objective_value
                        best_assignment = dict(selected)
                    return

                region_id = region_order[index]
                for recipe in _ordered_feasible_recipes(
                    region_id=region_id,
                    selected=selected,
                    recipes_by_region=recipes_by_region,
                    boundary_pairs=boundary_pairs,
                ):
                    selected[region_id] = recipe.recipe_id
                    dfs(index + 1, selected)
                    del selected[region_id]

            dfs(0, {})
        else:
            states = [
                _SearchState(
                    recipe_by_region={},
                    lower_bound=_estimate_lower_bound(
                        region_order=region_order,
                        region_successors=region_successors,
                        selected={},
                        min_cost_by_region=min_cost_by_region,
                        recipes_by_region=recipes_by_region,
                    ),
                    partial_cost=0,
                )
            ]
            for region_id in region_order:
                expanded: list[_SearchState] = []
                for state in states:
                    for recipe in _ordered_feasible_recipes(
                        region_id=region_id,
                        selected=state.recipe_by_region,
                        recipes_by_region=recipes_by_region,
                        boundary_pairs=boundary_pairs,
                    ):
                        selected = dict(state.recipe_by_region)
                        selected[region_id] = recipe.recipe_id
                        lower_bound = _estimate_lower_bound(
                            region_order=region_order,
                            region_successors=region_successors,
                            selected=selected,
                            min_cost_by_region=min_cost_by_region,
                            recipes_by_region=recipes_by_region,
                        )
                        if lower_bound >= best_objective:
                            continue
                        expanded.append(
                            _SearchState(
                                recipe_by_region=selected,
                                lower_bound=lower_bound,
                                partial_cost=state.partial_cost + recipe_cost(recipe),
                            )
                        )
                if not expanded:
                    states = []
                    break
                expanded.sort(
                    key=lambda state: (
                        state.lower_bound,
                        state.partial_cost,
                        tuple(
                            state.recipe_by_region[region]
                            for region in sorted(state.recipe_by_region)
                        ),
                    )
                )
                states = expanded[: self.beam_width]

            for state in states:
                if len(state.recipe_by_region) != len(region_order):
                    continue
                evaluated_assignments += 1
                result = _evaluate_assignment(problem, state.recipe_by_region)
                if isinstance(result, JointSolution) and result.objective_value < best_objective:
                    best_objective = result.objective_value
                    best_assignment = dict(state.recipe_by_region)

        if best_assignment is None:
            return solver_failure(
                JointFailureStatus.ERROR,
                JointFailureCategory.SOLVER_REPORTED_INFEASIBLE,
                "v1 could not find a valid mandatory-action recipe assignment",
            )

        result = _solve_assignment_best_effort(
            problem,
            best_assignment,
            diagnostics={
                "solver": "v1",
                "search_mode": search_mode,
                "beam_width": self.beam_width,
                "evaluated_assignments": evaluated_assignments,
            },
        )
        if not isinstance(result, JointSolution):
            return result

        return _refine_solution_locally(
            problem,
            result,
            recipes_by_region=recipes_by_region,
            boundary_pairs=boundary_pairs,
            boundary_neighbors=boundary_neighbors,
            max_local_passes=self.max_local_passes,
        )


def _evaluate_assignment(
    problem: JointProblem,
    recipe_by_region: Mapping[str, str],
) -> JointSolution | JointFailure:
    return _solve_assignment_best_effort(
        problem,
        recipe_by_region,
        diagnostics={"solver": "v1.candidate"},
    )


def _solve_assignment_best_effort(
    problem: JointProblem,
    recipe_by_region: Mapping[str, str],
    *,
    diagnostics: Mapping[str, object],
) -> JointSolution | JointFailure:
    baseline_result = solve_recipe_selection(
        problem,
        recipe_by_region,
        diagnostics={**diagnostics, "schedule_strategy": "baseline_ready"},
    )
    critical_path_result = solve_recipe_selection(
        problem,
        recipe_by_region,
        diagnostics={**diagnostics, "schedule_strategy": "critical_path_ready"},
        ready_priority_factory=_critical_path_ready_priority_factory,
    )
    return _prefer_better_result(baseline_result, critical_path_result)


def _prefer_better_result(
    left: JointSolution | JointFailure,
    right: JointSolution | JointFailure,
) -> JointSolution | JointFailure:
    if isinstance(left, JointSolution) and isinstance(right, JointSolution):
        if right.objective_value < left.objective_value:
            return right
        return left
    if isinstance(left, JointSolution):
        return left
    if isinstance(right, JointSolution):
        return right
    return left


def _critical_path_ready_priority_factory(
    problem: JointProblem,
    active_actions: Mapping[str, JointAction],
):
    active_action_ids = set(active_actions)
    topo_order = topological_action_order(problem, active_action_ids)
    if topo_order is None:
        return lambda action_id: (0, action_id)

    successors = build_successor_ids(problem, active_action_ids)
    criticality: dict[str, int] = {}
    for action_id in reversed(topo_order):
        tail_cost = max([0] + [criticality[succ] for succ in successors[action_id]])
        criticality[action_id] = action_cost(active_actions[action_id]) + tail_cost

    order_index = {action.action_id: index for index, action in enumerate(problem.actions)}
    value_size_by_id = {value.value_id: value.size_bytes for value in problem.values}

    def priority(action_id: str) -> tuple[object, ...]:
        action = active_actions[action_id]
        kind_rank = {
            JointActionKind.COMPUTE: 0,
            JointActionKind.DMA_OUT: 1,
            JointActionKind.DMA_IN: 2,
        }.get(action.kind, 3)
        write_bytes = sum(value_size_by_id.get(value_id, 0) for value_id in action.writes)
        read_bytes = sum(value_size_by_id.get(value_id, 0) for value_id in action.reads)
        return (
            -criticality[action_id],
            kind_rank,
            -write_bytes,
            -read_bytes,
            order_index[action_id],
        )

    return priority


def _refine_solution_locally(
    problem: JointProblem,
    solution: JointSolution,
    *,
    recipes_by_region: Mapping[str, tuple[JointRecipe, ...]],
    boundary_pairs: Mapping[tuple[str, str], set[tuple[str, str]]],
    boundary_neighbors: tuple[tuple[str, str], ...],
    max_local_passes: int,
) -> JointSolution:
    if max_local_passes <= 0:
        return solution

    mutable_regions = tuple(
        region_id
        for region_id, region_recipes in recipes_by_region.items()
        if len(region_recipes) > 1
    )
    if not mutable_regions:
        return solution

    current_solution = solution
    current_assignment = {
        item.region_id: item.recipe_id for item in solution.selected_recipes
    }
    improvement_count = 0
    completed_passes = 0

    while completed_passes < max_local_passes:
        completed_passes += 1
        next_solution = _best_single_region_improvement(
            problem,
            current_solution=current_solution,
            current_assignment=current_assignment,
            mutable_regions=mutable_regions,
            recipes_by_region=recipes_by_region,
            boundary_pairs=boundary_pairs,
        )
        if next_solution is not None:
            current_solution = next_solution
            current_assignment = {
                item.region_id: item.recipe_id for item in current_solution.selected_recipes
            }
            improvement_count += 1
            continue

        next_solution = _best_pair_region_improvement(
            problem,
            current_solution=current_solution,
            current_assignment=current_assignment,
            boundary_neighbors=boundary_neighbors,
            recipes_by_region=recipes_by_region,
            boundary_pairs=boundary_pairs,
        )
        if next_solution is None:
            break
        current_solution = next_solution
        current_assignment = {
            item.region_id: item.recipe_id for item in current_solution.selected_recipes
        }
        improvement_count += 1

    if improvement_count == 0:
        return solution

    diagnostics = dict(current_solution.diagnostics)
    diagnostics["local_search_passes"] = completed_passes
    diagnostics["local_search_improvements"] = improvement_count
    return replace(current_solution, diagnostics=diagnostics)


def _best_single_region_improvement(
    problem: JointProblem,
    *,
    current_solution: JointSolution,
    current_assignment: Mapping[str, str],
    mutable_regions: tuple[str, ...],
    recipes_by_region: Mapping[str, tuple[JointRecipe, ...]],
    boundary_pairs: Mapping[tuple[str, str], set[tuple[str, str]]],
) -> JointSolution | None:
    best_result: JointSolution | None = None
    best_objective = current_solution.objective_value

    for region_id in mutable_regions:
        selected_without_region = {
            other_region_id: recipe_id
            for other_region_id, recipe_id in current_assignment.items()
            if other_region_id != region_id
        }
        for recipe in recipes_by_region[region_id]:
            if recipe.recipe_id == current_assignment[region_id]:
                continue
            if not _is_compatible_with_selected(
                region_id=region_id,
                recipe_id=recipe.recipe_id,
                selected=selected_without_region,
                boundary_pairs=boundary_pairs,
            ):
                continue
            candidate_assignment = dict(selected_without_region)
            candidate_assignment[region_id] = recipe.recipe_id
            result = _evaluate_assignment(problem, candidate_assignment)
            if (
                isinstance(result, JointSolution)
                and result.objective_value < best_objective
            ):
                best_result = result
                best_objective = result.objective_value

    return best_result


def _best_pair_region_improvement(
    problem: JointProblem,
    *,
    current_solution: JointSolution,
    current_assignment: Mapping[str, str],
    boundary_neighbors: tuple[tuple[str, str], ...],
    recipes_by_region: Mapping[str, tuple[JointRecipe, ...]],
    boundary_pairs: Mapping[tuple[str, str], set[tuple[str, str]]],
) -> JointSolution | None:
    best_result: JointSolution | None = None
    best_objective = current_solution.objective_value

    for left_region_id, right_region_id in boundary_neighbors:
        if (
            len(recipes_by_region[left_region_id]) <= 1
            or len(recipes_by_region[right_region_id]) <= 1
        ):
            continue
        selected_without_pair = {
            region_id: recipe_id
            for region_id, recipe_id in current_assignment.items()
            if region_id not in {left_region_id, right_region_id}
        }
        for left_recipe in recipes_by_region[left_region_id]:
            if not _is_compatible_with_selected(
                region_id=left_region_id,
                recipe_id=left_recipe.recipe_id,
                selected=selected_without_pair,
                boundary_pairs=boundary_pairs,
            ):
                continue
            partial_assignment = dict(selected_without_pair)
            partial_assignment[left_region_id] = left_recipe.recipe_id
            for right_recipe in recipes_by_region[right_region_id]:
                if (
                    left_recipe.recipe_id == current_assignment[left_region_id]
                    and right_recipe.recipe_id == current_assignment[right_region_id]
                ):
                    continue
                if not _is_compatible_with_selected(
                    region_id=right_region_id,
                    recipe_id=right_recipe.recipe_id,
                    selected=partial_assignment,
                    boundary_pairs=boundary_pairs,
                ):
                    continue
                candidate_assignment = dict(partial_assignment)
                candidate_assignment[right_region_id] = right_recipe.recipe_id
                result = _evaluate_assignment(problem, candidate_assignment)
                if (
                    isinstance(result, JointSolution)
                    and result.objective_value < best_objective
                ):
                    best_result = result
                    best_objective = result.objective_value

    return best_result


def _recipes_by_region(problem: JointProblem) -> dict[str, tuple[JointRecipe, ...]]:
    grouped: dict[str, list[JointRecipe]] = defaultdict(list)
    for recipe in problem.recipes:
        grouped[recipe.region_id].append(recipe)
    return {
        region_id: tuple(
            sorted(
                region_recipes,
                key=lambda recipe: (
                    recipe_cost(recipe),
                    recipe.value_footprint.resident_bytes,
                    recipe.value_footprint.transfer_bytes,
                    recipe.recipe_id,
                ),
            )
        )
        for region_id, region_recipes in grouped.items()
    }


def _topological_region_order(problem: JointProblem) -> tuple[str, ...]:
    order_index = {region.region_id: index for index, region in enumerate(problem.regions)}
    predecessor_counts = {
        region.region_id: len(region.predecessor_region_ids)
        for region in problem.regions
    }
    successors = {
        region.region_id: list(region.successor_region_ids)
        for region in problem.regions
    }
    ready = sorted(
        [region_id for region_id, count in predecessor_counts.items() if count == 0],
        key=order_index.__getitem__,
    )
    order: list[str] = []
    while ready:
        region_id = ready.pop(0)
        order.append(region_id)
        for successor_id in sorted(successors[region_id], key=order_index.__getitem__):
            predecessor_counts[successor_id] -= 1
            if predecessor_counts[successor_id] == 0:
                ready.append(successor_id)
                ready.sort(key=order_index.__getitem__)
    if len(order) != len(problem.regions):
        return tuple(region.region_id for region in problem.regions)
    return tuple(order)


def _compatible_recipe_pairs(
    problem: JointProblem,
) -> dict[tuple[str, str], set[tuple[str, str]]]:
    boundary_pairs: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for boundary in problem.boundary_constraints:
        boundary_pairs[(boundary.src_region_id, boundary.dst_region_id)] = {
            (pair.src_recipe_id, pair.dst_recipe_id)
            for pair in boundary.compatible_recipe_pairs
        }
    return boundary_pairs


def _boundary_neighbor_pairs(problem: JointProblem) -> tuple[tuple[str, str], ...]:
    recipe_counts_by_region = _recipes_by_region(problem)
    return tuple(
        sorted(
            {
                tuple(sorted((boundary.src_region_id, boundary.dst_region_id)))
                for boundary in problem.boundary_constraints
                if len(boundary.compatible_recipe_pairs)
                < (
                    len(recipe_counts_by_region.get(boundary.src_region_id, ()))
                    * len(recipe_counts_by_region.get(boundary.dst_region_id, ()))
                )
            }
        )
    )


def _ordered_feasible_recipes(
    *,
    region_id: str,
    selected: Mapping[str, str],
    recipes_by_region: Mapping[str, tuple[JointRecipe, ...]],
    boundary_pairs: Mapping[tuple[str, str], set[tuple[str, str]]],
) -> tuple[JointRecipe, ...]:
    feasible: list[JointRecipe] = []
    for recipe in recipes_by_region[region_id]:
        if _is_compatible_with_selected(
            region_id=region_id,
            recipe_id=recipe.recipe_id,
            selected=selected,
            boundary_pairs=boundary_pairs,
        ):
            feasible.append(recipe)
    return tuple(feasible)


def _is_compatible_with_selected(
    *,
    region_id: str,
    recipe_id: str,
    selected: Mapping[str, str],
    boundary_pairs: Mapping[tuple[str, str], set[tuple[str, str]]],
) -> bool:
    for other_region_id, other_recipe_id in selected.items():
        forward_pairs = boundary_pairs.get((region_id, other_region_id))
        if forward_pairs is not None and (recipe_id, other_recipe_id) not in forward_pairs:
            return False
        backward_pairs = boundary_pairs.get((other_region_id, region_id))
        if backward_pairs is not None and (other_recipe_id, recipe_id) not in backward_pairs:
            return False
    return True


def _estimate_lower_bound(
    *,
    region_order: tuple[str, ...],
    region_successors: Mapping[str, tuple[str, ...] | list[str]],
    selected: Mapping[str, str],
    min_cost_by_region: Mapping[str, int],
    recipes_by_region: Mapping[str, tuple[JointRecipe, ...]],
) -> int:
    assigned_costs = {
        region_id: next(
            recipe_cost(recipe)
            for recipe in recipes_by_region[region_id]
            if recipe.recipe_id == recipe_id
        )
        for region_id, recipe_id in selected.items()
    }
    downstream_cost: dict[str, int] = {}
    for region_id in reversed(region_order):
        local_cost = assigned_costs.get(region_id, min_cost_by_region[region_id])
        succ_cost = max(
            [0] + [downstream_cost[succ] for succ in region_successors.get(region_id, ())]
        )
        downstream_cost[region_id] = local_cost + succ_cost
    return max(downstream_cost.values(), default=0)


__all__ = ["V1JointScheduleSolver"]
