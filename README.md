# nnc-joint-solver

A joint solver for the Neural Network Compiler that solves the combined **tiling selection, scheduling, and SRAM placement** optimization problem for neural network inference on hardware accelerators (NPUs).

## Problem Definition

Given a `JointProblem`, the solver must simultaneously make the following decisions:

1. **Recipe Selection** — Pick exactly one tiling strategy (recipe) per compute region.
2. **Action Scheduling** — Assign a start time to each active action.
3. **SRAM Residency Windows** — Determine SRAM residency time intervals for values that must reside in SRAM.
4. **SRAM Offset Allocation** — Assign memory offsets to all SRAM items.

**Objective: minimize makespan** (the maximum end time across all scheduled actions).

The solution must satisfy the following constraints (the normative constraint-checking implementation is in `validation.py`):

### Recipe Selection Constraints

- Each region must have exactly one selected recipe, and that recipe must belong to the region.
- For every adjacent region pair (regions sharing output/input values), the chosen recipe pair must appear in the corresponding `JointBoundaryConstraint`'s `compatible_recipe_pairs`.

### Action Scheduling Constraints

- All mandatory actions (induced by the `activates_action_ids` of selected recipes) must be scheduled.
- Optional actions (spill/reload) may be scheduled, but no action outside the mandatory + optional set may appear.
- An action's execution interval is `[start_time, start_time + duration + launch_overhead)`.
- **Dependency edges**: for each dependency edge `src -> dst` where both actions are scheduled, `end(src) <= start(dst)`.
- **Resource exclusivity**: two actions on the same `resource_kind` may not have overlapping execution intervals.

### SRAM Residency Constraints

- Residency windows for the same value may not overlap in time.
- A value with `initial_tier == sram` must have its first window start at time 0.
- A value with `required_final_tier == sram` must have its last window end at the makespan (`objective_value`).
- A value with `allows_multiple_sram_windows == false` may have at most one residency window.
- Each window's start time must be anchored to a valid "open time": for the first window, time 0 (when `initial_tier == sram`) or the end time of a compute/dma_in/reload action that writes this value; for subsequent windows, exactly the end time of a compute/dma_in/reload action writing this value.
- A value with `must_keep == true` must have exactly one continuous window starting from the earliest valid open time and covering at least through the last active consumer's end time.
- When a value leaves SRAM (previous window ends), a matching spill action must complete at exactly that time; when it re-enters SRAM (next window starts), a matching reload action must complete at exactly that time.

### Read Legality

- When a compute, dma_out, or spill action reads a value, that value must have a residency window covering the entire action execution interval `[start, end)`.

### Transfer Legality

- Spill/reload actions may only target values with `spillable == true`.
- During a spill action's execution, the target value must be resident in SRAM.
- A reload action must have a preceding completed spill action for the same value.

### SRAM Placement Constraints

- Every active SRAM item (problem-declared temp_interval/transfer_buffer + resident_window items generated from residency windows) must have exactly one offset allocation.
- Offsets must be non-negative and satisfy `alignment_bytes`.
- `offset + size_bytes` must not exceed `sram_capacity_bytes`.
- Two items whose time lifetimes overlap must not have overlapping address ranges `[offset, offset + size_bytes)`.
- At any point in time, the sum of all resident value sizes + all executing actions' `temp_bytes` must not exceed `sram_capacity_bytes`.

### Objective Consistency

- `objective_value` must equal the maximum end time across all scheduled actions.
- The producer action of any value with `required_final_tier == sram` must complete by `objective_value`.

## Installation and Usage

### As a Python Library

```python
from nnc_joint_solver import (
    V0JointScheduleSolver,
    V1JointScheduleSolver,
    LatestJointScheduleSolver,
    JointScheduleSolver,
    JointProblem,
    JointSolution,
    JointFailure,
)

problem = JointProblem.from_json(payload)
solver = LatestJointScheduleSolver()  # currently points to V1
result = solver.solve(problem)

if isinstance(result, JointSolution):
    print(f"makespan = {result.objective_value}")
else:
    print(f"failed: {result.status} / {result.error_category}")
```

### As a CLI Tool

The CLI receives a `JointProblem` JSON on stdin and outputs a `JointSolution` or `JointFailure` JSON on stdout:

```bash
# Use the default latest version (V1)
echo '{"schema_version": "joint_tiling_schedule_problem_v1", ...}' | nnc-joint-solver

# Specify V0
echo '...' | nnc-joint-solver --solver-version v0
```

### Subprocess Invocation

Use `CliJointScheduleSolver` to invoke the solver as a subprocess:

```python
from nnc_joint_solver import CliJointScheduleSolver

transport = CliJointScheduleSolver(
    command=["nnc-joint-solver"],
    timeout_seconds=10.0,
)
result = transport.solve(problem)  # JointSolution | JointFailure
```

## API Reference

### Abstract Interface

All solvers implement the `JointScheduleSolver` abstract base class:

```python
class JointScheduleSolver(ABC):
    @abstractmethod
    def solve(self, problem: JointProblem) -> JointSolution | JointFailure:
        ...
```

**Input:** `JointProblem` — the full description of the joint optimization problem.
**Output:** a discriminated union: `JointSolution` on success, `JointFailure` on error.

### Core Data Models

All data models are defined in `ir/joint_tiling_schedule.py` as frozen dataclasses with `to_json()` / `from_json()` bidirectional JSON serialization.

#### Enumerations

| Enum | Values | Description |
|------|--------|-------------|
| `JointRegionKind` | `single_op`, `fused_group` | Compute region type |
| `JointValueTier` | `unmaterialized`, `input`, `const`, `slow`, `sram` | Value storage tier |
| `JointSramItemKind` | `temp_interval`, `transfer_buffer`, `resident_window` | SRAM item type |
| `JointActionKind` | `compute`, `dma_in`, `dma_out`, `spill`, `reload` | Action type |
| `JointDependencyEdgeKind` | `data`, `order` | Dependency edge type |
| `JointResourceKind` | `DMA`, `MATMUL`, `SHAPE`, `OTHER` | Hardware resource type |
| `JointFailureStatus` | `infeasible`, `timeout`, `invalid_problem`, `error` | Failure status code |
| `JointFailureCategory` | 8 categories (`dependency_violation`, `resource_overlap`, `sram_capacity_exceeded`, etc.) | Detailed error classification |

#### Problem Side

```python
@dataclass(frozen=True)
class JointProblem:
    schema_version: str              # "joint_tiling_schedule_problem_v1"
    regions: tuple[JointRegion, ...]
    recipes: tuple[JointRecipe, ...]
    values: tuple[JointValue, ...]
    actions: tuple[JointAction, ...]
    boundary_constraints: tuple[JointBoundaryConstraint, ...]
    dependency_edges: tuple[JointDependencyEdge, ...]
    resources: tuple[JointResource, ...]        # must include DMA/MATMUL/SHAPE/OTHER
    sram_capacity_bytes: int                    # SRAM capacity limit
    sram_items: tuple[JointSramItem, ...]       # fixed SRAM allocation requirements
    default_alignment_bytes: int                # default alignment in bytes
    objective: str                              # "minimize_makespan"
```

- `JointRegion` — a compute region with input/output value IDs, predecessor/successor region IDs.
- `JointRecipe` — a tiling strategy for a region, including tile_spec, layout_spec, activated action IDs, value footprint, and cost parameters.
- `JointValue` — a tensor/value with size, initial/final tier, producer/consumer, spillability, and SRAM residency constraints.
- `JointAction` — an operation (compute/DMA/spill/reload) with resource kind, duration, launch overhead, read/write value IDs, and temp bytes.
- `JointBoundaryConstraint` — constraints between adjacent regions: which recipe pairs are compatible.
- `JointDependencyEdge` — a directed dependency edge between actions.

#### Solution Side

```python
@dataclass(frozen=True)
class JointSolution:
    schema_version: str                          # "joint_tiling_schedule_solution_v1"
    selected_recipes: tuple[JointSelectedRecipe, ...]    # chosen recipe per region
    scheduled_actions: tuple[JointScheduledAction, ...]  # start time per action
    residency_windows: tuple[JointResidencyWindow, ...]  # SRAM residency windows for values
    objective_value: int                         # makespan
    generated_sram_items: tuple[JointSramItem, ...]      # SRAM items generated from residency windows
    sram_allocations: tuple[JointSramAllocation, ...]    # offset per SRAM item
    diagnostics: object                          # optional diagnostic info
```

#### Failure Side

```python
@dataclass(frozen=True)
class JointFailure:
    schema_version: str                          # "joint_tiling_schedule_failure_v1"
    status: JointFailureStatus                   # failure status
    error_category: JointFailureCategory         # error category
    diagnostics: object                          # diagnostic details
```

### CLI Transport Layer

`CliJointScheduleSolver` wraps the subprocess JSON communication protocol:

- Serializes `JointProblem` to JSON and writes it to the subprocess stdin
- Reads JSON from subprocess stdout, deserializes into `JointSolution` or `JointFailure` based on `schema_version`
- Handles timeouts (`DEFAULT_SOLVER_TIMEOUT_SECONDS = 5.0`), process errors, and format errors

### Schema Versions

The wire protocol uses explicit version strings:

| Scenario | schema_version |
|----------|---------------|
| Input | `joint_tiling_schedule_problem_v1` |
| Success output | `joint_tiling_schedule_solution_v1` |
| Failure output | `joint_tiling_schedule_failure_v1` |

### Solver Versions

| Class | Description |
|-------|-------------|
| `V0JointScheduleSolver` | Baseline solver, picks the first recipe per region |
| `V1JointScheduleSolver` | Heuristic solver with beam search + local search |
| `LatestJointScheduleSolver` | Points to the current latest version (V1) |
| `BaselineJointScheduleSolver` | Alias for `V0JointScheduleSolver` |

## Developing a New Algorithm

### 1. Create a Solver Implementation

Create a new version directory under `src/nnc_joint_solver/` and implement the `JointScheduleSolver` interface:

```
src/nnc_joint_solver/v2/
  __init__.py      # export V2JointScheduleSolver
  solver.py        # implementation
```

Minimal skeleton for `v2/solver.py`:

```python
from nnc_joint_solver.base import JointScheduleSolver
from nnc_joint_solver.ir.joint_tiling_schedule import (
    JointProblem,
    JointSolution,
    JointFailure,
)


class V2JointScheduleSolver(JointScheduleSolver):
    def solve(self, problem: JointProblem) -> JointSolution | JointFailure:
        # 1. (optional) Validate the problem
        # 2. Select recipes
        # 3. Schedule actions
        # 4. Compute SRAM residency windows and allocations
        # 5. Construct and return JointSolution or JointFailure
        ...
```

### 2. Leverage Shared Utilities

`solve_utils.py` provides reusable scheduling infrastructure:

| Function | Purpose |
|----------|---------|
| `solve_recipe_selection()` | Full pipeline: build selected recipes → determine active actions → topological sort → schedule → residency windows → SRAM allocation → validate → return solution |
| `topological_action_order()` | Kahn's topological sort on dependency edges with customizable tie-breaking priority |
| `schedule_ordered_actions()` | Assign start times in topological order, handling resource exclusivity and JIT DMA delay heuristic |
| `_minimal_residency_windows()` | Compute minimal SRAM residency time windows |
| `_pack_sram_allocations()` | First-fit decreasing interval packing algorithm |

Most new algorithms only need to focus on the **recipe selection strategy**, then call `solve_recipe_selection()` to get a complete solution:

```python
from nnc_joint_solver.solve_utils import solve_recipe_selection

class V2JointScheduleSolver(JointScheduleSolver):
    def solve(self, problem: JointProblem) -> JointSolution | JointFailure:
        # Custom recipe selection logic
        selected = {...}  # region_id -> recipe_id
        return solve_recipe_selection(problem, selected)
```

### 3. Register the Solver

Update the following files to make the new solver available:

**`src/nnc_joint_solver/v2/__init__.py`:**

```python
from nnc_joint_solver.v2.solver import V2JointScheduleSolver

__all__ = ["V2JointScheduleSolver"]
```

**`src/nnc_joint_solver/solver.py` — add export and update latest:**

```python
from nnc_joint_solver.v2.solver import V2JointScheduleSolver

LatestJointScheduleSolver = V2JointScheduleSolver  # update to point here
```

**`src/nnc_joint_solver/cli.py` — add CLI version option:**

```python
parser.add_argument("--solver-version", choices=("v0", "v1", "v2"), default="v2")
```

### 4. Validation

`validation.py` provides comprehensive constraint checking:

- `validate_joint_problem()` — validates problem structural integrity
- `validate_joint_solution()` — validates that the solution satisfies all constraints (recipe coverage, action scheduling legality, no resource overlaps, dependency edges satisfied, boundary compatibility, SRAM capacity and overlap checks)

It is recommended to call validation before returning from the solver to ensure a valid solution.
