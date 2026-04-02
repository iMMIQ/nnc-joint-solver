# nnc-joint-solver

Standalone external solver for `nnc-py` joint tiling/schedule problems.

## Scope

This repository owns:

- the external `joint_tiling_schedule_*_v1` contract used by the solver
- contract validation
- a deterministic baseline solver
- a stdin/stdout CLI entrypoint

It does not contain compiler passes, materialization, or code generation.

## Problem Class

`joint_tiling_schedule_problem_v1` defines a discrete optimization problem over:

- one recipe choice per region
- one start time per active action
- zero or more SRAM residency windows per value
- one SRAM offset per active SRAM item

The contract objective is `min_makespan`: minimize the completion time of the
last scheduled action while satisfying dataflow, boundary, resource, residency,
and SRAM-capacity constraints.

At the contract level, the solver is choosing jointly:

- a recipe `p(r)` for each region `r`
- an active action set induced by those recipe choices, plus any optional
  spill/reload actions it decides to schedule
- an integer start time `s_a` for every scheduled action `a`
- SRAM residency intervals for values that must be present in fast memory
- aligned byte offsets for all active SRAM items

## Formal Statement

Let:

- `R` be the set of regions
- `P_r` be the legal recipes for region `r`
- `A` be the set of declared actions
- `V` be the set of logical values
- `E` be the set of dependency edges
- `B` be the set of boundary constraints
- `K` be the set of resource kinds, with `slot_count = 1` for every resource in
  v1
- `I_fixed` be the fixed SRAM items declared in the problem
- `C` be `sram_capacity_bytes`

Decision variables:

- `x_{r,p} in {0,1}`: recipe `p` is selected for region `r`
- `y_a in {0,1}`: action `a` is scheduled
- `s_a in Z_{>=0}`: start time of scheduled action `a`
- `T in Z_{>=0}`: makespan
- for each value `v`, a sequence of SRAM windows
  `W_v = {[b_{v,k}, e_{v,k})}_k`
- for each active SRAM item `i`, an aligned offset `o_i in Z_{>=0}`

Derived quantities:

- `end(a) = s_a + duration(a) + launch_overhead(a)`
- mandatory actions are the union of `activates_action_ids` across the selected
  recipes
- generated resident-window SRAM items are in one-to-one correspondence with
  `residency_windows`

Minimize:

```text
minimize T
subject to T = max_a end(a)
```

Subject to the validator-enforced constraints:

1. Recipe selection:
   `sum_{p in P_r} x_{r,p} = 1` for every region `r`.
2. Boundary compatibility:
   for each boundary `(r_src, r_dst) in B`, the selected pair
   `(p(r_src), p(r_dst))` must appear in `compatible_recipe_pairs`.
3. Action coverage:
   every mandatory action must be scheduled exactly once; optional actions may
   be scheduled at most once; no other actions may appear.
4. Precedence:
   for every active dependency edge `(u, v) in E`,
   `end(u) <= s_v`.
5. Resource exclusivity:
   if two scheduled actions use the same `resource_kind`, their execution
   intervals may not overlap. In v1 this is a unary-resource model.
6. Residency legality:
   values must enter SRAM only at legal anchors, satisfy
   `initial_tier` / `required_final_tier`, obey
   `must_keep` / `allows_multiple_sram_windows`, and use matching
   spill/reload actions when leaving and re-entering SRAM.
7. Read legality:
   compute, `dma_out`, and spill actions may read a value only while that value
   is resident in SRAM for the full action interval.
8. SRAM placement:
   every active fixed or generated SRAM item must have one allocation;
   allocations must satisfy alignment; items whose lifetimes overlap in time may
   not overlap in address space; every item must fit within `[0, C)`.
9. Objective consistency:
   the returned `objective_value` must equal the actual makespan induced by the
   schedule.

The exact legality check is implemented in
`src/nnc_joint_solver/validation.py`, so that file is the normative source for
accepted and rejected solutions.

## Current Baseline Solver

`BaselineJointScheduleSolver` is intentionally much narrower than the full
contract problem above. Its current behavior is:

- select the first declared recipe for every region
- schedule only the mandatory actions induced by those recipe choices
- never schedule optional spill/reload actions
- compute a topological order, then do a deterministic earliest-start list
  schedule subject to dependency and unary-resource constraints
- delay `dma_in` actions when possible so they finish just before their paired
  compute action, reducing unnecessary SRAM overlap
- create minimal residency windows from producer completion to last active
  consumer completion, extending to the makespan when a value must end in SRAM
- generate one resident-window SRAM item per residency window
- pack fixed and generated SRAM items with a greedy first-fit allocator ordered
  by lifetime start time

So the current baseline solver does **not** search recipe combinations, does
not insert optional transfer actions, and does not optimize globally over the
entire contract decision space. It is a deterministic reference implementation
that returns a valid solution for the no-optional-action subset when one fits.

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
