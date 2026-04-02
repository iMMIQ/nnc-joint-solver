"""CLI entrypoint for the standalone joint solver."""

from __future__ import annotations

import argparse
import json
import sys

from nnc_joint_solver.ir.joint_tiling_schedule import (
    JOINT_TILING_SCHEDULE_FAILURE_SCHEMA_VERSION,
    JointFailure,
    JointFailureCategory,
    JointFailureStatus,
    JointProblem,
)
from nnc_joint_solver.solver import V0JointScheduleSolver, V1JointScheduleSolver


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        failure = _failure(
            JointFailureStatus.INVALID_PROBLEM,
            JointFailureCategory.INVALID_SOLUTION,
            f"stdin must be valid JSON: {exc.msg}",
        )
        json.dump(failure.to_json(), sys.stdout)
        return 0

    try:
        problem = JointProblem.from_json(payload)
    except (TypeError, ValueError) as exc:
        failure = _failure(
            JointFailureStatus.INVALID_PROBLEM,
            JointFailureCategory.INVALID_SOLUTION,
            str(exc),
        )
        json.dump(failure.to_json(), sys.stdout)
        return 0

    solver = V0JointScheduleSolver() if args.solver_version == "v0" else V1JointScheduleSolver()
    result = solver.solve(problem)
    json.dump(result.to_json(), sys.stdout)
    return 0


def run() -> None:
    raise SystemExit(main(sys.argv[1:]))


def _failure(
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


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--solver-version", choices=("v0", "v1"), default="v1")
    return parser.parse_args(argv)
