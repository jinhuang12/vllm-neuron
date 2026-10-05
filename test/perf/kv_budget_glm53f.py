# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash KV/state budget per rank at TP=64: before (5938748) and after.

CPU only; no device, no weights. The "after" figures come from the real code path:
the model's own ``get_kv_spec`` at world size 64, ``NeuronModelRunner.get_kv_cache_spec``,
vLLM's grouping and block-pool sizing, ``NeuronWorker._kv_cache_need_bytes`` /
``_kv_cache_footprint_bytes`` / ``_compute_kv_budget``, and the runner's own indexer
side-cache builder. The "before" figure is the 5938748 pricing kept as
``before_fix_footprint_bytes`` in ``test/vllm_neuron/worker/test_kv_budget_glm53f.py``,
which this script imports so the two cannot drift apart.

Usage (from the worktree root):

    PYTHONPATH=$PWD python test/perf/kv_budget_glm53f.py \\
        --max-num-seqs 64 --max-model-len 8192 --output reports/kv_budget.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test.vllm_neuron.worker import test_kv_budget_glm53f as kv  # noqa: E402

GIB = 1024**3
LENGTHS = (4096, 8192, 16384)

#: Per-rank residency scenarios the budget is computed against.
RESIDENCY = {
    # Measured at 5938748 on every rank (server_decode.log): parameters and
    # buffers 2.79 GiB, 7.96 GiB resident with the prepared kernel operands.
    "as_built_5938748": {
        "param_bytes": kv.MEASURED_PARAM_BYTES,
        "resident_bytes": kv.MEASURED_RESIDENT_BYTES,
    },
    # The planner's per-rank weight entitlement (PLANNER_REPORT.md section 2a).
    "planner_weights_4.99GB": {
        "param_bytes": int(4.99e9),
        "resident_bytes": int(4.99e9),
    },
}


@contextlib.contextmanager
def knob_env(knobs: dict):
    """Set the KV budget env knobs for the duration of a block, then restore them."""
    saved = {name: os.environ.get(name) for name in knobs}
    for name, value in knobs.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def budget_bytes(model_specs, scenario: dict, knobs: dict) -> int:
    """The worker's KV budget (``_compute_kv_budget``) for one residency and knob set."""
    layers, _ = model_specs
    runner = kv.fake_runner(layers, max_num_seqs=1, max_model_len=4096)
    worker = kv.fake_worker(
        runner,
        param_bytes=scenario["param_bytes"],
        resident_bytes=scenario["resident_bytes"],
    )
    with knob_env(knobs):
        return int(
            worker._compute_kv_budget(
                kv.TOTAL_HBM_BYTES, scenario["param_bytes"], kv.GPU_MEMORY_UTILIZATION
            )
        )


def after_bytes(model_specs, seqs: int, length: int, gate_knobs: bool) -> dict:
    return kv.after_fix_bytes(
        max_num_seqs=seqs,
        max_model_len=length,
        gate_knobs=gate_knobs,
        model_specs=model_specs,
    )


def largest_fitting(fits, ceiling: int = 4096) -> int:
    """Largest ``seqs`` in ``[0, ceiling]`` with ``fits(seqs)``; ``fits`` is monotone."""
    low, high = 0, ceiling
    while low < high:
        middle = (low + high + 1) // 2
        if fits(middle):
            low = middle
        else:
            high = middle - 1
    return low


def max_sequences(model_specs, budget: int, length: int) -> dict:
    """How many sequences of ``length`` tokens fit ``budget`` under each pricing."""

    def after_fits(gate_knobs: bool, with_side: bool):
        def fits(seqs: int) -> bool:
            if seqs == 0:
                return True
            part = after_bytes(model_specs, seqs, length, gate_knobs)
            total = part["total_bytes"] if with_side else part["footprint_bytes"]
            return total <= budget

        return fits

    return {
        "before_5938748": largest_fitting(
            lambda s: s == 0
            or kv.before_fix_footprint_bytes(max_num_seqs=s, max_model_len=length)
            <= budget
        ),
        "after_default_serve_worker_check": largest_fitting(after_fits(False, False)),
        "after_gate_serve_worker_check": largest_fitting(after_fits(True, False)),
        "after_gate_serve_incl_side_caches": largest_fitting(after_fits(True, True)),
    }


def gib(value: int) -> float:
    return round(value / GIB, 4)


def summary_lines(result: dict) -> list[str]:
    """A short plain-text reading of ``result``, printed after the JSON."""
    point, per_rank = result["point"], result["per_rank_gib"]
    lines = [
        f"per rank at {point['max_num_seqs']} x {point['max_model_len']} tokens: "
        f"before (5938748) {per_rank['before_5938748_footprint']} GiB; "
        f"after, gate line {per_rank['after_gate_serve_footprint']} GiB "
        f"({per_rank['after_gate_serve_total_incl_side_caches']} GiB incl. side caches); "
        f"after, default line {per_rank['after_default_serve_footprint']} GiB",
    ]
    for name, entry in result["budgets"].items():
        lines.append(f"{name}: budget {entry['budget_gib']} GiB, max sequences:")
        for length, fit in entry["max_sequences"].items():
            lines.append(
                f"  {int(length):>5} tokens: before {fit['before_5938748']}, "
                f"after default line {fit['after_default_serve_worker_check']}, "
                f"after gate line {fit['after_gate_serve_worker_check']} "
                f"({fit['after_gate_serve_incl_side_caches']} incl. side caches)"
            )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--graph-reserve-gib",
        type=float,
        default=None,
        help="VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB to evaluate besides the default",
    )
    parser.add_argument(
        "--cap-fraction",
        type=float,
        default=None,
        help="VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION to evaluate besides the default",
    )
    args = parser.parse_args()

    from vllm_neuron import envs

    model_specs = kv.glm53f_layer_specs()
    seqs, length = args.max_num_seqs, args.max_model_len

    defaults = {
        "VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION": None,
        "VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB": None,
    }
    with knob_env(defaults):
        default_values = {
            "VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION": envs.VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION,
            "VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB": envs.VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB,
        }
    knob_sets = {"defaults": defaults}
    if args.graph_reserve_gib is not None or args.cap_fraction is not None:
        knob_sets["evaluated"] = {
            "VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION": args.cap_fraction,
            "VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB": args.graph_reserve_gib,
        }

    before = kv.before_fix_footprint_bytes(max_num_seqs=seqs, max_model_len=length)
    after_gate = after_bytes(model_specs, seqs, length, gate_knobs=True)
    after_default = after_bytes(model_specs, seqs, length, gate_knobs=False)

    result = {
        "point": {
            "max_num_seqs": seqs,
            "max_model_len": length,
            "tensor_parallel_size": kv.TP_WORLD_SIZE,
            "hybrid_kv_block_size": kv.BLOCK_SIZE_TOKENS,
        },
        "per_rank_bytes": {
            "before_5938748_footprint": before,
            "after_gate_serve": after_gate,
            "after_default_serve": after_default,
        },
        "per_rank_gib": {
            "before_5938748_footprint": gib(before),
            "after_gate_serve_footprint": gib(after_gate["footprint_bytes"]),
            "after_gate_serve_total_incl_side_caches": gib(after_gate["total_bytes"]),
            "after_default_serve_footprint": gib(after_default["footprint_bytes"]),
        },
        "serve_lines": {
            "gate_serve": "--no-enable-prefix-caching --mamba-block-size <max_model_len>",
            "default_serve": "prefix caching on, no --mamba-block-size (5938748 line)",
        },
        "default_knob_values": default_values,
        "budgets": {},
    }
    for scenario_name, scenario in RESIDENCY.items():
        for knob_name, knobs in knob_sets.items():
            budget = budget_bytes(model_specs, scenario, knobs)
            result["budgets"][f"{scenario_name}/{knob_name}"] = {
                "knobs": knobs,
                "budget_bytes": budget,
                "budget_gib": gib(budget),
                "point_fits_worker_check": after_gate["footprint_bytes"] <= budget,
                "point_fits_incl_side_caches": after_gate["total_bytes"] <= budget,
                "max_sequences": {
                    str(n): max_sequences(model_specs, budget, n) for n in LENGTHS
                },
            }

    text = json.dumps(result, indent=2)
    print(text)
    print("\n".join(summary_lines(result)))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
