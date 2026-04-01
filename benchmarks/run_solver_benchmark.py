#!/usr/bin/env python3
"""Run the standalone solver benchmark on a fixed joint problem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nnc_joint_solver.benchmark import default_problem_path, run_solver_benchmark


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark solver makespan against a fixed joint_tiling_schedule_problem_v1 payload."
    )
    parser.add_argument(
        "--problem",
        default=str(default_problem_path()),
        help="Path to the problem JSON to benchmark",
    )
    parser.add_argument(
        "--solver-command",
        nargs="+",
        default=None,
        help="Override solver command. Defaults to the local bin/nnc-joint-solver wrapper.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of solver invocations to run",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=5.0,
        help="Timeout per solver invocation",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the JSON benchmark result",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    payload = run_solver_benchmark(
        problem_path=args.problem,
        solver_command=args.solver_command,
        repeats=args.repeats,
        timeout_seconds=args.timeout_seconds,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
