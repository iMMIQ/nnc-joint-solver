# nnc-joint-solver

Standalone external solver for `nnc-py` joint tiling/schedule problems.

## Scope

This repository owns:

- the external `joint_tiling_schedule_*_v1` contract used by the solver
- contract validation
- a deterministic baseline solver
- a stdin/stdout CLI entrypoint

It does not contain compiler passes, materialization, or code generation.

## CLI

The checked-in executable is:

```bash
bin/nnc-joint-solver
```

It reads a `joint_tiling_schedule_problem_v1` JSON object from stdin and writes
either:

- `joint_tiling_schedule_solution_v1`
- `joint_tiling_schedule_failure_v1`

to stdout.

Structured infeasible/error payloads exit `0`. Transport or protocol failures
exit non-zero.

## Development

Run tests from the parent checkout or inside this repository:

```bash
pytest tests/test_solver.py -v
```

## Benchmark

You can benchmark solver quality independently from the compiler once a fixed
`joint_tiling_schedule_problem_v1` JSON has been exported:

```bash
python benchmarks/run_solver_benchmark.py \
  --problem benchmarks/problems/resnet18_o3_1m.problem.json
```

The benchmark score is the solver's returned `objective_value` (makespan). A
smaller score is better. If the solver fails to produce a valid solution, the
benchmark reports `score: null` plus the structured failure fields.
