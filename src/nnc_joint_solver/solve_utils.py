"""Reusable scheduling and solution-construction helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from nnc_joint_solver.ir.joint_tiling_schedule import (
    JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION,
    JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
    JointAction,
    JointActionKind,
    JointFailure,
    JointFailureCategory,
    JointFailureStatus,
    JointProblem,
    JointRecipe,
    JointResidencyWindow,
    JointScheduledAction,
    JointSelectedRecipe,
    JointSolution,
    JointSramAllocation,
    JointSramItem,
    JointSramItemKind,
)
from nnc_joint_solver.validation import validate_joint_solution


ReadyPriority = Callable[[str], tuple[object, ...]]
ReadyPriorityFactory = Callable[[JointProblem, Mapping[str, JointAction]], ReadyPriority]


@dataclass(frozen=True)
class _LiveSramItem:
    order_index: int
    item: JointSramItem
    start_time: int
    end_time: int


def build_selected_recipes(
    problem: JointProblem,
    recipe_by_region: Mapping[str, str],
) -> tuple[JointSelectedRecipe, ...]:
    return tuple(
        JointSelectedRecipe(
            region_id=region.region_id,
            recipe_id=recipe_by_region[region.region_id],
        )
        for region in problem.regions
    )


def active_actions_for_recipes(
    problem: JointProblem,
    recipe_by_region: Mapping[str, str],
) -> dict[str, JointAction]:
    recipes_by_id = {recipe.recipe_id: recipe for recipe in problem.recipes}
    actions_by_id = {action.action_id: action for action in problem.actions}
    mandatory_action_ids = {
        action_id
        for recipe_id in recipe_by_region.values()
        for action_id in recipes_by_id[recipe_id].activates_action_ids
    }
    return {
        action_id: actions_by_id[action_id]
        for action_id in mandatory_action_ids
    }


def build_predecessor_ids(
    problem: JointProblem,
    active_action_ids: set[str],
) -> dict[str, list[str]]:
    predecessor_ids: dict[str, list[str]] = defaultdict(list)
    for edge in problem.dependency_edges:
        if edge.src_action_id in active_action_ids and edge.dst_action_id in active_action_ids:
            predecessor_ids[edge.dst_action_id].append(edge.src_action_id)
    return predecessor_ids


def build_successor_ids(
    problem: JointProblem,
    active_action_ids: set[str],
) -> dict[str, list[str]]:
    successors: dict[str, list[str]] = {action_id: [] for action_id in active_action_ids}
    for edge in problem.dependency_edges:
        if edge.src_action_id in active_action_ids and edge.dst_action_id in active_action_ids:
            successors[edge.src_action_id].append(edge.dst_action_id)
    return successors


def action_cost(action: JointAction) -> int:
    return action.duration + action.launch_overhead


def recipe_cost(recipe: JointRecipe) -> int:
    return recipe.cost_parameters.latency + recipe.cost_parameters.launch_overhead


def default_ready_priority(
    problem: JointProblem,
    active_actions: Mapping[str, JointAction],
) -> ReadyPriority:
    order_index = {action.action_id: index for index, action in enumerate(problem.actions)}
    value_size_by_id = {value.value_id: value.size_bytes for value in problem.values}

    def priority(action_id: str) -> tuple[object, ...]:
        action = active_actions[action_id]
        write_bytes = sum(value_size_by_id.get(value_id, 0) for value_id in action.writes)
        read_bytes = sum(value_size_by_id.get(value_id, 0) for value_id in action.reads)
        if action.kind is JointActionKind.COMPUTE:
            return (0, -write_bytes, -read_bytes, order_index[action_id])
        if action.kind is JointActionKind.DMA_OUT:
            return (1, 0, 0, order_index[action_id])
        if action.kind is JointActionKind.DMA_IN:
            return (2, 0, 0, order_index[action_id])
        return (3, 0, 0, order_index[action_id])

    return priority


def topological_action_order(
    problem: JointProblem,
    active_action_ids: set[str],
    *,
    ready_priority: ReadyPriority | None = None,
) -> tuple[str, ...] | None:
    successors = build_successor_ids(problem, active_action_ids)
    predecessor_counts: dict[str, int] = {action_id: 0 for action_id in active_action_ids}
    active_actions = {
        action.action_id: action
        for action in problem.actions
        if action.action_id in active_action_ids
    }
    priority = ready_priority or default_ready_priority(problem, active_actions)

    for edge in problem.dependency_edges:
        if edge.src_action_id not in active_action_ids or edge.dst_action_id not in active_action_ids:
            continue
        predecessor_counts[edge.dst_action_id] += 1

    ready = sorted(
        [action_id for action_id, count in predecessor_counts.items() if count == 0],
        key=priority,
    )
    order: list[str] = []
    while ready:
        action_id = ready.pop(0)
        order.append(action_id)
        for successor_id in sorted(successors[action_id], key=priority):
            predecessor_counts[successor_id] -= 1
            if predecessor_counts[successor_id] == 0:
                ready.append(successor_id)
                ready.sort(key=priority)
    if len(order) != len(active_action_ids):
        return None
    return tuple(order)


def solve_recipe_selection(
    problem: JointProblem,
    recipe_by_region: Mapping[str, str],
    *,
    diagnostics: Mapping[str, object],
    ready_priority_factory: ReadyPriorityFactory | None = None,
) -> JointSolution | JointFailure:
    selected_recipes = build_selected_recipes(problem, recipe_by_region)
    active_actions = active_actions_for_recipes(problem, recipe_by_region)
    active_action_ids = set(active_actions)
    ready_priority = None
    if ready_priority_factory is not None:
        ready_priority = ready_priority_factory(problem, active_actions)
    schedule_order = topological_action_order(
        problem,
        active_action_ids,
        ready_priority=ready_priority,
    )
    if schedule_order is None:
        return solver_failure(
            JointFailureStatus.ERROR,
            JointFailureCategory.DEPENDENCY_VIOLATION,
            "mandatory action graph is cyclic",
        )

    start_by_action, end_by_action = schedule_ordered_actions(
        problem,
        active_actions=active_actions,
        schedule_order=schedule_order,
    )
    objective_value = max(end_by_action.values(), default=0)
    residency_windows = tuple(
        _minimal_residency_windows(problem, active_actions, end_by_action, objective_value)
    )
    generated_sram_items = _generated_residency_items(problem, residency_windows)
    sram_allocations = _pack_sram_allocations(
        problem,
        fixed_items=problem.sram_items,
        generated_items=generated_sram_items,
        start_by_action=start_by_action,
        end_by_action=end_by_action,
        residency_windows=residency_windows,
        objective_value=objective_value,
    )
    if sram_allocations is None:
        return solver_failure(
            JointFailureStatus.ERROR,
            JointFailureCategory.SRAM_CAPACITY_EXCEEDED,
            "SRAM allocator could not place active items within capacity",
        )

    solution = JointSolution(
        schema_version=JOINT_TILING_SCHEDULE_SOLUTION_SCHEMA_VERSION,
        selected_recipes=selected_recipes,
        scheduled_actions=tuple(
            JointScheduledAction(action_id=action_id, start_time=start_by_action[action_id])
            for action_id in schedule_order
        ),
        residency_windows=residency_windows,
        objective_value=objective_value,
        generated_sram_items=generated_sram_items,
        sram_allocations=sram_allocations,
        diagnostics=dict(diagnostics),
    )
    solution_failure = validate_joint_solution(problem, solution)
    if solution_failure is not None:
        return solution_failure
    return solution


def schedule_ordered_actions(
    problem: JointProblem,
    *,
    active_actions: Mapping[str, JointAction],
    schedule_order: tuple[str, ...],
) -> tuple[dict[str, int], dict[str, int]]:
    start_by_action: dict[str, int] = {}
    end_by_action: dict[str, int] = {}
    resource_available: dict[str, int] = defaultdict(int)
    predecessor_ids = build_predecessor_ids(problem, set(active_actions))
    action_order = {
        action.action_id: index for index, action in enumerate(problem.actions)
    }

    for action_id in schedule_order:
        action = active_actions[action_id]
        earliest = _earliest_action_start(
            action_id=action_id,
            action=action,
            active_actions=active_actions,
            predecessor_ids=predecessor_ids,
            start_by_action=start_by_action,
            end_by_action=end_by_action,
            resource_available=resource_available,
            action_order=action_order,
        )
        start_by_action[action_id] = earliest
        end_by_action[action_id] = earliest + action_cost(action)
        resource_available[action.resource_kind.value] = end_by_action[action_id]
    return start_by_action, end_by_action


def solver_failure(
    status: JointFailureStatus,
    error_category: JointFailureCategory,
    reason: str,
) -> JointFailure:
    return JointFailure(
        schema_version=JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION,
        status=status,
        error_category=error_category,
        diagnostics={"reason": reason},
    )


def _earliest_action_start(
    *,
    action_id: str,
    action: JointAction,
    active_actions: Mapping[str, JointAction],
    predecessor_ids: Mapping[str, list[str]],
    start_by_action: Mapping[str, int],
    end_by_action: Mapping[str, int],
    resource_available: Mapping[str, int],
    action_order: Mapping[str, int],
) -> int:
    predecessor_end = max(
        [0] + [end_by_action[pred] for pred in predecessor_ids.get(action_id, ())]
    )
    earliest = max(resource_available[action.resource_kind.value], predecessor_end)
    if action.kind is not JointActionKind.DMA_IN:
        return earliest

    consumer_id = _paired_consumer_compute_action_id(
        action_id=action_id,
        action=action,
        active_actions=active_actions,
        predecessor_ids=predecessor_ids,
        action_order=action_order,
    )
    if consumer_id is None:
        return earliest

    consumer = active_actions[consumer_id]
    non_dma_predecessor_end = max(
        [
            0,
            *[
                end_by_action[pred]
                for pred in predecessor_ids.get(consumer_id, ())
                if pred != action_id
                and active_actions[pred].kind is not JointActionKind.DMA_IN
                and pred in end_by_action
            ],
        ]
    )
    consumer_ready = max(
        non_dma_predecessor_end,
        resource_available[consumer.resource_kind.value],
    )
    sibling_dma_ids = sorted(
        [
            pred
            for pred in predecessor_ids.get(consumer_id, ())
            if active_actions[pred].kind is JointActionKind.DMA_IN
        ],
        key=action_order.__getitem__,
    )
    trailing_dma_cost = 0
    seen_self = False
    for sibling_id in reversed(sibling_dma_ids):
        if sibling_id == action_id:
            seen_self = True
            break
        trailing_dma_cost += action_cost(active_actions[sibling_id])
    if not seen_self:
        return earliest

    latest_start_without_stall = (
        consumer_ready
        - trailing_dma_cost
        - action_cost(action)
    )
    return max(earliest, latest_start_without_stall)


def _paired_consumer_compute_action_id(
    *,
    action_id: str,
    action: JointAction,
    active_actions: Mapping[str, JointAction],
    predecessor_ids: Mapping[str, list[str]],
    action_order: Mapping[str, int],
) -> str | None:
    if action.region_id is None or action.recipe_id is None:
        return None

    candidates = [
        other.action_id
        for other in active_actions.values()
        if other.kind is JointActionKind.COMPUTE
        and other.region_id == action.region_id
        and other.recipe_id == action.recipe_id
        and action_id in predecessor_ids.get(other.action_id, ())
    ]
    if not candidates:
        return None
    return min(candidates, key=action_order.__getitem__)


def _generated_residency_items(
    problem: JointProblem,
    residency_windows: tuple[JointResidencyWindow, ...],
) -> tuple[JointSramItem, ...]:
    size_by_value = {value.value_id: value.size_bytes for value in problem.values}
    return tuple(
        JointSramItem(
            item_id=f"{window.residency_id}.item",
            kind=JointSramItemKind.RESIDENT_WINDOW,
            size_bytes=size_by_value[window.value_id],
            alignment_bytes=problem.default_alignment_bytes,
            is_optional=False,
            owner_action_id=None,
            owner_value_id=window.value_id,
            owner_residency_id=window.residency_id,
        )
        for window in residency_windows
    )


def _pack_sram_allocations(
    problem: JointProblem,
    *,
    fixed_items: tuple[JointSramItem, ...],
    generated_items: tuple[JointSramItem, ...],
    start_by_action: Mapping[str, int],
    end_by_action: Mapping[str, int],
    residency_windows: tuple[JointResidencyWindow, ...],
    objective_value: int,
) -> tuple[JointSramAllocation, ...] | None:
    live_items = _collect_live_sram_items(
        fixed_items=fixed_items,
        generated_items=generated_items,
        start_by_action=start_by_action,
        end_by_action=end_by_action,
        residency_windows=residency_windows,
        objective_value=objective_value,
    )
    allocations: list[JointSramAllocation] = []
    active: list[tuple[int, int, int, int]] = []

    for live_item in live_items:
        active = [
            (start_time, end_time, offset, end_offset)
            for start_time, end_time, offset, end_offset in active
            if end_time > live_item.start_time
        ]
        candidate_offset = 0
        placed_offset: int | None = None
        for _, _, offset, end_offset in sorted(active, key=lambda interval: interval[2]):
            aligned_offset = _align(candidate_offset, live_item.item.alignment_bytes)
            if aligned_offset + live_item.item.size_bytes <= offset:
                placed_offset = aligned_offset
                break
            candidate_offset = max(candidate_offset, end_offset)
        if placed_offset is None:
            placed_offset = _align(candidate_offset, live_item.item.alignment_bytes)
        if placed_offset + live_item.item.size_bytes > problem.sram_capacity_bytes:
            return None
        allocations.append(
            JointSramAllocation(item_id=live_item.item.item_id, offset=placed_offset)
        )
        active.append(
            (
                live_item.start_time,
                live_item.end_time,
                placed_offset,
                placed_offset + live_item.item.size_bytes,
            )
        )
    return tuple(allocations)


def _collect_live_sram_items(
    *,
    fixed_items: tuple[JointSramItem, ...],
    generated_items: tuple[JointSramItem, ...],
    start_by_action: Mapping[str, int],
    end_by_action: Mapping[str, int],
    residency_windows: tuple[JointResidencyWindow, ...],
    objective_value: int,
) -> tuple[_LiveSramItem, ...]:
    windows_by_residency = {
        window.residency_id: window for window in residency_windows
    }
    live_items: list[_LiveSramItem] = []
    for order_index, item in enumerate((*fixed_items, *generated_items)):
        lifetime = _item_lifetime(
            item,
            start_by_action=start_by_action,
            end_by_action=end_by_action,
            windows_by_residency=windows_by_residency,
            objective_value=objective_value,
        )
        if lifetime is None:
            continue
        start_time, end_time = lifetime
        live_items.append(
            _LiveSramItem(
                order_index=order_index,
                item=item,
                start_time=start_time,
                end_time=end_time,
            )
        )
    return tuple(
        sorted(
            live_items,
            key=lambda live_item: (
                live_item.start_time,
                live_item.end_time,
                live_item.order_index,
            ),
        )
    )


def _item_lifetime(
    item: JointSramItem,
    *,
    start_by_action: Mapping[str, int],
    end_by_action: Mapping[str, int],
    windows_by_residency: Mapping[str, JointResidencyWindow],
    objective_value: int,
) -> tuple[int, int] | None:
    if item.owner_residency_id is not None:
        window = windows_by_residency.get(item.owner_residency_id)
        if window is None:
            return None
        return window.start_time, window.end_time
    if item.owner_action_id is not None:
        start_time = start_by_action.get(item.owner_action_id)
        end_time = end_by_action.get(item.owner_action_id)
        if start_time is None or end_time is None:
            return None
        return start_time, end_time
    return 0, objective_value


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _minimal_residency_windows(
    problem: JointProblem,
    active_actions: Mapping[str, JointAction],
    end_by_action: Mapping[str, int],
    objective_value: int,
) -> list[JointResidencyWindow]:
    windows: list[JointResidencyWindow] = []
    active_action_ids = set(active_actions)
    for value in problem.values:
        active_consumers = [
            consumer.action_id
            for consumer in value.consumers
            if consumer.action_id in active_action_ids
        ]
        if not active_consumers:
            continue
        if value.producer is None:
            if value.initial_tier.value == "sram":
                open_end = 0
            else:
                open_end = next(
                    (
                        end_by_action[action_id]
                        for action_id, action in active_actions.items()
                        if action.kind is JointActionKind.DMA_IN and value.value_id in action.writes
                    ),
                    None,
                )
                if open_end is None:
                    continue
        else:
            writer_ids = [
                action_id
                for action_id, action in active_actions.items()
                if value.value_id in action.writes
            ]
            if not writer_ids:
                continue
            preferred_writer = value.producer.action_id
            writer_id = preferred_writer if preferred_writer in writer_ids else min(
                writer_ids,
                key=end_by_action.__getitem__,
            )
            open_end = end_by_action[writer_id]
        close_end = max(end_by_action[action_id] for action_id in active_consumers)
        if value.required_final_tier.value == "sram":
            close_end = objective_value
        if close_end > open_end:
            windows.append(
                JointResidencyWindow(
                    residency_id=f"{value.value_id}@{open_end}",
                    value_id=value.value_id,
                    start_time=open_end,
                    end_time=close_end,
                )
            )
    return windows


__all__ = [
    "ReadyPriorityFactory",
    "action_cost",
    "active_actions_for_recipes",
    "build_predecessor_ids",
    "build_selected_recipes",
    "build_successor_ids",
    "default_ready_priority",
    "recipe_cost",
    "solve_recipe_selection",
    "solver_failure",
    "topological_action_order",
]
