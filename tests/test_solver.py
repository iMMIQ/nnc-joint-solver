from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nnc_joint_solver.ir.joint_tiling_schedule import (  # noqa: E402
    JointFailure,
    JointProblem,
    JointSolution,
)
from nnc_joint_solver.solver import BaselineJointScheduleSolver  # noqa: E402
from nnc_joint_solver.validation import (  # noqa: E402
    validate_joint_problem,
    validate_joint_solution,
)


def _allocatable_problem_payload() -> dict[str, object]:
    return {
        "schema_version": "joint_tiling_schedule_problem_v1",
        "regions": [
            {
                "region_id": "region0",
                "kind": "single_op",
                "member_nodes": ["region0"],
                "input_value_ids": ["input0"],
                "output_value_ids": ["output0"],
                "predecessor_region_ids": [],
                "successor_region_ids": [],
            }
        ],
        "recipes": [
            {
                "recipe_id": "region0.recipe0",
                "region_id": "region0",
                "tile_spec": {"axes": ["h", "w"], "shape": [8, 8]},
                "layout_spec": {"layout_tags": ["nchw"]},
                "activates_action_ids": [
                    "region0.recipe0.dma_in.input0",
                    "region0.recipe0.compute",
                    "region0.recipe0.dma_out.output0",
                ],
                "value_footprint": {
                    "resident_bytes": 160,
                    "scratch_bytes": 64,
                    "transfer_bytes": 160,
                },
                "cost_parameters": {"latency": 9, "launch_overhead": 3},
            }
        ],
        "values": [
            {
                "value_id": "input0",
                "size_bytes": 64,
                "initial_tier": "input",
                "required_final_tier": "input",
                "must_keep": False,
                "spillable": False,
                "allows_multiple_sram_windows": False,
                "producer": None,
                "consumers": [
                    {"action_id": "region0.recipe0.dma_in.input0"},
                    {"action_id": "region0.recipe0.compute"},
                ],
            },
            {
                "value_id": "output0",
                "size_bytes": 96,
                "initial_tier": "unmaterialized",
                "required_final_tier": "slow",
                "must_keep": False,
                "spillable": False,
                "allows_multiple_sram_windows": False,
                "producer": {"action_id": "region0.recipe0.compute"},
                "consumers": [{"action_id": "region0.recipe0.dma_out.output0"}],
            },
        ],
        "actions": [
            {
                "action_id": "region0.recipe0.dma_in.input0",
                "kind": "dma_in",
                "resource_kind": "DMA",
                "duration": 2,
                "launch_overhead": 1,
                "reads": ["input0"],
                "writes": ["input0"],
                "temp_bytes": 0,
                "is_optional": False,
                "region_id": "region0",
                "recipe_id": "region0.recipe0",
                "optional_value_id": None,
            },
            {
                "action_id": "region0.recipe0.compute",
                "kind": "compute",
                "resource_kind": "OTHER",
                "duration": 5,
                "launch_overhead": 1,
                "reads": ["input0"],
                "writes": ["output0"],
                "temp_bytes": 64,
                "is_optional": False,
                "region_id": "region0",
                "recipe_id": "region0.recipe0",
                "optional_value_id": None,
            },
            {
                "action_id": "region0.recipe0.dma_out.output0",
                "kind": "dma_out",
                "resource_kind": "DMA",
                "duration": 2,
                "launch_overhead": 1,
                "reads": ["output0"],
                "writes": ["output0"],
                "temp_bytes": 0,
                "is_optional": False,
                "region_id": "region0",
                "recipe_id": "region0.recipe0",
                "optional_value_id": None,
            },
        ],
        "boundary_constraints": [],
        "dependency_edges": [
            {
                "src_action_id": "region0.recipe0.dma_in.input0",
                "dst_action_id": "region0.recipe0.compute",
                "kind": "data",
            },
            {
                "src_action_id": "region0.recipe0.compute",
                "dst_action_id": "region0.recipe0.dma_out.output0",
                "kind": "data",
            },
        ],
        "resources": [
            {"resource_kind": "DMA", "slot_count": 1},
            {"resource_kind": "MATMUL", "slot_count": 1},
            {"resource_kind": "SHAPE", "slot_count": 1},
            {"resource_kind": "OTHER", "slot_count": 1},
        ],
        "sram_capacity_bytes": 192,
        "sram_items": [
            {
                "item_id": "region0.recipe0.compute.temp",
                "kind": "temp_interval",
                "size_bytes": 64,
                "alignment_bytes": 16,
                "is_optional": False,
                "owner_action_id": "region0.recipe0.compute",
                "owner_value_id": None,
                "owner_residency_id": None,
            },
            {
                "item_id": "region0.recipe0.compute.pack",
                "kind": "transfer_buffer",
                "size_bytes": 32,
                "alignment_bytes": 16,
                "is_optional": False,
                "owner_action_id": "region0.recipe0.dma_in.input0",
                "owner_value_id": None,
                "owner_residency_id": None,
            },
        ],
        "default_alignment_bytes": 16,
        "objective": "min_makespan",
    }


def _solution_payload() -> dict[str, object]:
    return {
        "schema_version": "joint_tiling_schedule_solution_v1",
        "selected_recipes": [{"region_id": "region0", "recipe_id": "region0.recipe0"}],
        "scheduled_actions": [
            {"action_id": "region0.recipe0.dma_in.input0", "start_time": 0},
            {"action_id": "region0.recipe0.compute", "start_time": 3},
            {"action_id": "region0.recipe0.dma_out.output0", "start_time": 9},
        ],
        "residency_windows": [
            {
                "residency_id": "input0@3",
                "value_id": "input0",
                "start_time": 3,
                "end_time": 9,
            },
            {
                "residency_id": "output0@9",
                "value_id": "output0",
                "start_time": 9,
                "end_time": 12,
            },
        ],
        "objective_value": 12,
        "generated_sram_items": [
            {
                "item_id": "input0@3.item",
                "kind": "resident_window",
                "size_bytes": 64,
                "alignment_bytes": 16,
                "is_optional": False,
                "owner_action_id": None,
                "owner_value_id": "input0",
                "owner_residency_id": "input0@3",
            },
            {
                "item_id": "output0@9.item",
                "kind": "resident_window",
                "size_bytes": 96,
                "alignment_bytes": 16,
                "is_optional": False,
                "owner_action_id": None,
                "owner_value_id": "output0",
                "owner_residency_id": "output0@9",
            },
        ],
        "sram_allocations": [
            {"item_id": "region0.recipe0.compute.temp", "offset": 0},
            {"item_id": "region0.recipe0.compute.pack", "offset": 0},
            {"item_id": "input0@3.item", "offset": 64},
            {"item_id": "output0@9.item", "offset": 0},
        ],
        "diagnostics": {"solver": "baseline"},
    }


def _jit_dma_problem_payload() -> dict[str, object]:
    return {
        "schema_version": "joint_tiling_schedule_problem_v1",
        "regions": [
            {
                "region_id": "region0",
                "kind": "single_op",
                "member_nodes": ["region0"],
                "input_value_ids": ["input0", "weight0"],
                "output_value_ids": ["mid"],
                "predecessor_region_ids": [],
                "successor_region_ids": ["region1"],
            },
            {
                "region_id": "region1",
                "kind": "single_op",
                "member_nodes": ["region1"],
                "input_value_ids": ["mid", "weight1"],
                "output_value_ids": ["out"],
                "predecessor_region_ids": ["region0"],
                "successor_region_ids": [],
            },
        ],
        "recipes": [
            {
                "recipe_id": "region0.recipe0",
                "region_id": "region0",
                "tile_spec": {"axes": ["h", "w"], "shape": [8, 8]},
                "layout_spec": {"layout_tags": ["nchw"]},
                "activates_action_ids": [
                    "region0.recipe0.dma_in.input0",
                    "region0.recipe0.dma_in.weight0",
                    "region0.recipe0.compute",
                ],
                "value_footprint": {
                    "resident_bytes": 320,
                    "scratch_bytes": 64,
                    "transfer_bytes": 320,
                },
                "cost_parameters": {"latency": 40, "launch_overhead": 1},
            },
            {
                "recipe_id": "region1.recipe0",
                "region_id": "region1",
                "tile_spec": {"axes": ["h", "w"], "shape": [8, 8]},
                "layout_spec": {"layout_tags": ["nchw"]},
                "activates_action_ids": [
                    "region1.recipe0.dma_in.weight1",
                    "region1.recipe0.compute",
                    "region1.recipe0.dma_out.out",
                ],
                "value_footprint": {
                    "resident_bytes": 320,
                    "scratch_bytes": 64,
                    "transfer_bytes": 320,
                },
                "cost_parameters": {"latency": 40, "launch_overhead": 1},
            },
        ],
        "values": [
            {
                "value_id": "input0",
                "size_bytes": 64,
                "initial_tier": "input",
                "required_final_tier": "input",
                "must_keep": False,
                "spillable": False,
                "allows_multiple_sram_windows": False,
                "producer": None,
                "consumers": [
                    {"action_id": "region0.recipe0.dma_in.input0"},
                    {"action_id": "region0.recipe0.compute"},
                ],
            },
            {
                "value_id": "weight0",
                "size_bytes": 256,
                "initial_tier": "const",
                "required_final_tier": "const",
                "must_keep": False,
                "spillable": False,
                "allows_multiple_sram_windows": False,
                "producer": None,
                "consumers": [
                    {"action_id": "region0.recipe0.dma_in.weight0"},
                    {"action_id": "region0.recipe0.compute"},
                ],
            },
            {
                "value_id": "mid",
                "size_bytes": 96,
                "initial_tier": "unmaterialized",
                "required_final_tier": "slow",
                "must_keep": False,
                "spillable": False,
                "allows_multiple_sram_windows": False,
                "producer": {"action_id": "region0.recipe0.compute"},
                "consumers": [{"action_id": "region1.recipe0.compute"}],
            },
            {
                "value_id": "weight1",
                "size_bytes": 256,
                "initial_tier": "const",
                "required_final_tier": "const",
                "must_keep": False,
                "spillable": False,
                "allows_multiple_sram_windows": False,
                "producer": None,
                "consumers": [
                    {"action_id": "region1.recipe0.dma_in.weight1"},
                    {"action_id": "region1.recipe0.compute"},
                ],
            },
            {
                "value_id": "out",
                "size_bytes": 64,
                "initial_tier": "unmaterialized",
                "required_final_tier": "slow",
                "must_keep": False,
                "spillable": False,
                "allows_multiple_sram_windows": False,
                "producer": {"action_id": "region1.recipe0.compute"},
                "consumers": [{"action_id": "region1.recipe0.dma_out.out"}],
            },
        ],
        "actions": [
            {
                "action_id": "region0.recipe0.dma_in.input0",
                "kind": "dma_in",
                "resource_kind": "DMA",
                "duration": 2,
                "launch_overhead": 1,
                "reads": ["input0"],
                "writes": ["input0"],
                "temp_bytes": 0,
                "is_optional": False,
                "region_id": "region0",
                "recipe_id": "region0.recipe0",
                "optional_value_id": None,
            },
            {
                "action_id": "region0.recipe0.dma_in.weight0",
                "kind": "dma_in",
                "resource_kind": "DMA",
                "duration": 2,
                "launch_overhead": 1,
                "reads": ["weight0"],
                "writes": ["weight0"],
                "temp_bytes": 0,
                "is_optional": False,
                "region_id": "region0",
                "recipe_id": "region0.recipe0",
                "optional_value_id": None,
            },
            {
                "action_id": "region0.recipe0.compute",
                "kind": "compute",
                "resource_kind": "OTHER",
                "duration": 40,
                "launch_overhead": 1,
                "reads": ["input0", "weight0"],
                "writes": ["mid"],
                "temp_bytes": 64,
                "is_optional": False,
                "region_id": "region0",
                "recipe_id": "region0.recipe0",
                "optional_value_id": None,
            },
            {
                "action_id": "region1.recipe0.dma_in.weight1",
                "kind": "dma_in",
                "resource_kind": "DMA",
                "duration": 2,
                "launch_overhead": 1,
                "reads": ["weight1"],
                "writes": ["weight1"],
                "temp_bytes": 0,
                "is_optional": False,
                "region_id": "region1",
                "recipe_id": "region1.recipe0",
                "optional_value_id": None,
            },
            {
                "action_id": "region1.recipe0.compute",
                "kind": "compute",
                "resource_kind": "OTHER",
                "duration": 40,
                "launch_overhead": 1,
                "reads": ["mid", "weight1"],
                "writes": ["out"],
                "temp_bytes": 64,
                "is_optional": False,
                "region_id": "region1",
                "recipe_id": "region1.recipe0",
                "optional_value_id": None,
            },
            {
                "action_id": "region1.recipe0.dma_out.out",
                "kind": "dma_out",
                "resource_kind": "DMA",
                "duration": 2,
                "launch_overhead": 1,
                "reads": ["out"],
                "writes": ["out"],
                "temp_bytes": 0,
                "is_optional": False,
                "region_id": "region1",
                "recipe_id": "region1.recipe0",
                "optional_value_id": None,
            },
        ],
        "boundary_constraints": [
            {
                "boundary_id": "region0->region1",
                "src_region_id": "region0",
                "dst_region_id": "region1",
                "compatible_recipe_pairs": [
                    {
                        "src_recipe_id": "region0.recipe0",
                        "dst_recipe_id": "region1.recipe0",
                    }
                ],
            }
        ],
        "dependency_edges": [
            {
                "src_action_id": "region0.recipe0.dma_in.input0",
                "dst_action_id": "region0.recipe0.compute",
                "kind": "data",
            },
            {
                "src_action_id": "region0.recipe0.dma_in.weight0",
                "dst_action_id": "region0.recipe0.compute",
                "kind": "data",
            },
            {
                "src_action_id": "region0.recipe0.compute",
                "dst_action_id": "region1.recipe0.compute",
                "kind": "data",
            },
            {
                "src_action_id": "region1.recipe0.dma_in.weight1",
                "dst_action_id": "region1.recipe0.compute",
                "kind": "data",
            },
            {
                "src_action_id": "region1.recipe0.compute",
                "dst_action_id": "region1.recipe0.dma_out.out",
                "kind": "data",
            },
        ],
        "resources": [
            {"resource_kind": "DMA", "slot_count": 1},
            {"resource_kind": "MATMUL", "slot_count": 1},
            {"resource_kind": "SHAPE", "slot_count": 1},
            {"resource_kind": "OTHER", "slot_count": 1},
        ],
        "sram_capacity_bytes": 512,
        "sram_items": [
            {
                "item_id": "region0.recipe0.compute.temp",
                "kind": "temp_interval",
                "size_bytes": 64,
                "alignment_bytes": 16,
                "is_optional": False,
                "owner_action_id": "region0.recipe0.compute",
                "owner_value_id": None,
                "owner_residency_id": None,
            },
            {
                "item_id": "region1.recipe0.compute.temp",
                "kind": "temp_interval",
                "size_bytes": 64,
                "alignment_bytes": 16,
                "is_optional": False,
                "owner_action_id": "region1.recipe0.compute",
                "owner_value_id": None,
                "owner_residency_id": None,
            },
        ],
        "default_alignment_bytes": 16,
        "objective": "min_makespan",
    }


def test_joint_problem_round_trips_required_sram_fields():
    problem = JointProblem.from_json(_allocatable_problem_payload())

    assert problem.to_json()["schema_version"] == "joint_tiling_schedule_problem_v1"
    assert problem.sram_items
    assert problem.default_alignment_bytes == 16


def test_invalid_problem_returns_structured_failure():
    payload = _allocatable_problem_payload()
    payload["recipes"] = []
    problem = JointProblem.from_json(payload)

    failure = validate_joint_problem(problem)

    assert isinstance(failure, JointFailure)


def test_baseline_solver_delays_dma_inputs_to_avoid_unnecessary_sram_overlap():
    problem = JointProblem.from_json(_jit_dma_problem_payload())

    result = BaselineJointScheduleSolver().solve(problem)

    assert isinstance(result, JointSolution)
    starts = {item.action_id: item.start_time for item in result.scheduled_actions}
    assert starts["region1.recipe0.dma_in.weight1"] >= starts["region0.recipe0.compute"]


def test_joint_solution_round_trips_required_sram_fields():
    solution = JointSolution.from_json(_solution_payload())

    assert solution.residency_windows[0].residency_id == "input0@3"
    assert solution.generated_sram_items[0].owner_residency_id == "input0@3"
    assert solution.sram_allocations[0].offset == 0


def test_upgraded_v1_contract_rejects_missing_required_fields():
    bad_problem_payload = _allocatable_problem_payload()
    bad_problem_payload.pop("sram_items")
    bad_solution_payload = _solution_payload()
    bad_solution_payload.pop("sram_allocations")

    try:
        JointProblem.from_json(bad_problem_payload)
    except ValueError as exc:
        assert "sram_items" in str(exc)
    else:
        raise AssertionError("expected JointProblem.from_json() to reject missing sram_items")

    try:
        JointSolution.from_json(bad_solution_payload)
    except ValueError as exc:
        assert "sram_allocations" in str(exc)
    else:
        raise AssertionError(
            "expected JointSolution.from_json() to reject missing sram_allocations"
        )


def test_baseline_solver_returns_valid_solution_for_allocatable_problem():
    problem = JointProblem.from_json(_allocatable_problem_payload())

    result = BaselineJointScheduleSolver().solve(problem)

    assert isinstance(result, JointSolution)
    assert result.generated_sram_items
    assert result.sram_allocations
    assert validate_joint_solution(problem, result) is None


def test_cli_emits_solution_json_for_valid_problem():
    cli = ROOT / "bin" / "nnc-joint-solver"
    result = subprocess.run(
        [sys.executable, str(cli)],
        input=json.dumps(_allocatable_problem_payload()),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "joint_tiling_schedule_solution_v1"
