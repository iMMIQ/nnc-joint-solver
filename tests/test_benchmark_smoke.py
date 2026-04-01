from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nnc_joint_solver.benchmark import run_solver_benchmark  # noqa: E402


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


def test_solver_benchmark_reports_baseline_makespan(tmp_path):
    problem_path = tmp_path / "problem.json"
    problem_path.write_text(json.dumps(_allocatable_problem_payload()))

    payload = run_solver_benchmark(problem_path=problem_path)

    assert payload["status"] == "ok"
    assert payload["score"] == 12
    assert payload["best_makespan"] == 12
    assert payload["runs"][0]["makespan"] == 12
