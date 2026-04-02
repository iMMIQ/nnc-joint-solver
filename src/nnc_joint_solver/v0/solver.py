"""Version 0 baseline solver."""

from __future__ import annotations

from nnc_joint_solver.base import JointScheduleSolver
from nnc_joint_solver.ir.joint_tiling_schedule import (
    JointFailure,
    JointFailureCategory,
    JointFailureStatus,
    JointProblem,
    JointSolution,
)
from nnc_joint_solver.solve_utils import solve_recipe_selection, solver_failure
from nnc_joint_solver.validation import validate_joint_problem


class V0JointScheduleSolver(JointScheduleSolver):
    """Deterministic v0 baseline for the external joint contract."""

    def solve(self, problem: JointProblem) -> JointSolution | JointFailure:
        problem_failure = validate_joint_problem(problem)
        if problem_failure is not None:
            return problem_failure

        recipe_by_region: dict[str, str] = {}
        for region in problem.regions:
            region_recipes = [
                recipe for recipe in problem.recipes if recipe.region_id == region.region_id
            ]
            if not region_recipes:
                return solver_failure(
                    JointFailureStatus.INVALID_PROBLEM,
                    JointFailureCategory.INVALID_SOLUTION,
                    f"region {region.region_id!r} has no recipes",
                )
            recipe_by_region[region.region_id] = region_recipes[0].recipe_id

        return solve_recipe_selection(
            problem,
            recipe_by_region,
            diagnostics={"solver": "v0"},
        )


BaselineJointScheduleSolver = V0JointScheduleSolver


__all__ = ["BaselineJointScheduleSolver", "V0JointScheduleSolver"]
