"""Benchmark helpers for evaluating solver makespan on fixed joint problems."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from time import perf_counter

from nnc_joint_solver.ir.joint_tiling_schedule import JointFailure, JointProblem, JointSolution
from nnc_joint_solver.solver import CliJointScheduleSolver, DEFAULT_SOLVER_TIMEOUT_SECONDS


@dataclass(frozen=True)
class SolverBenchmarkRun:
    status: str
    elapsed_seconds: float
    makespan: int | None
    failure_status: str | None
    error_category: str | None
    diagnostics: dict[str, object] | None


def default_problem_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "benchmarks" / "problems" / "resnet18_o3_1m.problem.json"


def default_solver_command() -> list[str]:
    root = Path(__file__).resolve().parents[2]
    return [sys.executable, str(root / "bin" / "nnc-joint-solver")]


def load_problem(path: str | Path) -> JointProblem:
    payload = json.loads(Path(path).read_text())
    return JointProblem.from_json(payload)


def run_solver_benchmark(
    *,
    problem_path: str | Path | None = None,
    solver_command: list[str] | tuple[str, ...] | None = None,
    repeats: int = 1,
    timeout_seconds: float = DEFAULT_SOLVER_TIMEOUT_SECONDS,
) -> dict[str, object]:
    if repeats <= 0:
        raise ValueError("repeats must be >= 1")

    resolved_problem_path = Path(problem_path or default_problem_path()).resolve()
    problem = load_problem(resolved_problem_path)
    resolved_solver_command = list(solver_command or default_solver_command())
    solver = CliJointScheduleSolver(
        resolved_solver_command,
        timeout_seconds=timeout_seconds,
    )

    runs: list[SolverBenchmarkRun] = []
    for _ in range(repeats):
        start = perf_counter()
        result = solver.solve(problem)
        elapsed_seconds = perf_counter() - start
        runs.append(_run_from_result(result, elapsed_seconds))

    successful_runs = [run for run in runs if run.makespan is not None]
    payload: dict[str, object] = {
        "problem_path": str(resolved_problem_path),
        "solver_command": resolved_solver_command,
        "repeats": repeats,
        "runs": [
            {
                "status": run.status,
                "elapsed_seconds": run.elapsed_seconds,
                "makespan": run.makespan,
                "failure_status": run.failure_status,
                "error_category": run.error_category,
                "diagnostics": run.diagnostics,
            }
            for run in runs
        ],
    }
    if successful_runs:
        makespans = [run.makespan for run in successful_runs if run.makespan is not None]
        payload["status"] = "ok"
        payload["score"] = min(makespans)
        payload["best_makespan"] = min(makespans)
        payload["mean_makespan"] = sum(makespans) / len(makespans)
    else:
        first_failure = runs[0]
        payload["status"] = "failure"
        payload["score"] = None
        payload["failure_status"] = first_failure.failure_status
        payload["error_category"] = first_failure.error_category
    return payload


def _run_from_result(
    result: JointSolution | JointFailure,
    elapsed_seconds: float,
) -> SolverBenchmarkRun:
    if isinstance(result, JointSolution):
        diagnostics = dict(result.diagnostics) if isinstance(result.diagnostics, dict) else None
        return SolverBenchmarkRun(
            status="ok",
            elapsed_seconds=elapsed_seconds,
            makespan=result.objective_value,
            failure_status=None,
            error_category=None,
            diagnostics=diagnostics,
        )

    diagnostics = dict(result.diagnostics) if isinstance(result.diagnostics, dict) else None
    return SolverBenchmarkRun(
        status="failure",
        elapsed_seconds=elapsed_seconds,
        makespan=None,
        failure_status=result.status.value,
        error_category=result.error_category.value,
        diagnostics=diagnostics,
    )


__all__ = [
    "default_problem_path",
    "default_solver_command",
    "load_problem",
    "run_solver_benchmark",
]
