# SPDX-License-Identifier: Apache-2.0
"""Error table of the dense-window kernel against the as-built sparse path (not a test).

Runs every case of ``test_mla_dense_window.CASES`` in the NKI simulator and writes, per
case and per reference, the max abs error, the max element error relative to the
reference's peak, the per-row relative L2 (max, and the share of rows above 1e-4 and
1e-3), in fp32 and after a bf16 round of both outputs (the cast the layer applies before
``o_proj``).

    export VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NEURON_LOGICAL_NC_CONFIG=2 \\
        NEURON_PLATFORM_TARGET_OVERRIDE=trn2
    python -m test.vllm_neuron.functional.attention.mla_dense_window_errors \\
        out.json [case-id ...]

CPU only. Both kernels run in the NKI simulator, and the as-built index rows come from
the selection chain's torch routes. The script also applies the environment
``test/conftest.py`` gives a pytest run: without a platform target the simulator asks
the Neuron runtime for one, and that opens every ``/dev/neuron*`` node. It refuses a
simulator switch other than ``1``, and it exits 1 without a table if the process holds a
Neuron device node open at the end.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

from test.vllm_neuron.functional.attention.neuron_device_nodes import (
    open_neuron_device_nodes,
)


def _metrics(got: torch.Tensor, want: torch.Tensor) -> dict:
    got64, want64 = got.double(), want.double()
    diff = (got64 - want64).abs()
    rows_got = got64.reshape(got64.shape[0], -1)
    rows_want = want64.reshape(want64.shape[0], -1)
    row_rel = (rows_got - rows_want).norm(dim=1) / rows_want.norm(dim=1).clamp_min(1e-30)
    return {
        "max_abs": float(diff.max()),
        "max_abs_over_peak": float(diff.max() / want64.abs().max().clamp_min(1e-30)),
        "rel_l2": float((got64 - want64).norm() / want64.norm().clamp_min(1e-30)),
        "row_rel_max": float(row_rel.max()),
        "row_rel_median": float(row_rel.median()),
        "rows": int(row_rel.numel()),
        "rows_above_1e-4": float((row_rel > 1e-4).double().mean()),
        "rows_above_1e-3": float((row_rel > 1e-3).double().mean()),
    }


def main(path: str, case_ids: list[str]) -> int:
    from test.conftest import DEFAULTED_ENV

    for name, value in DEFAULTED_ENV.items():
        os.environ.setdefault(name, value)
    os.environ.setdefault("NKI_SIMULATOR", "1")
    os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
    for name in ("VLLM_NEURON_CPU_MODE", "NKI_SIMULATOR"):
        if os.environ[name] != "1":
            raise SystemExit(f"{name} must be 1: this script runs in the CPU simulator only")
    from test.vllm_neuron.functional.attention import test_mla_dense_window as T

    cases = [param for param in T.CASES if not case_ids or param.id in case_ids]
    unknown = sorted(set(case_ids) - {param.id for param in T.CASES})
    if unknown:
        raise SystemExit(f"unknown case ids {unknown}; known: {[p.id for p in T.CASES]}")
    table = []
    for param in cases:
        tokens, start, pages, heads = param.values
        seed = tokens * 7 + start * 3 + heads
        q, bank, table_row, written, offset, seq_lens, window = T._case(tokens, start, pages,
                                                                         heads, seed)
        began = time.monotonic()
        dense = T._dense(q, bank, seq_lens, table_row, written, offset)
        dense_s = time.monotonic() - began
        indices = T._asbuilt_indices(seq_lens, seed=7)
        refs = {}
        for name, fp32 in (("sparse_lowp", False), ("sparse_fp32", True)):
            if fp32:
                os.environ["VLLM_NEURON_MLA_SPARSE_FP32"] = "1"
            else:
                os.environ.pop("VLLM_NEURON_MLA_SPARSE_FP32", None)
            refs[name] = T._sparse(q, bank, indices, table_row, written, offset)
        os.environ.pop("VLLM_NEURON_MLA_SPARSE_FP32", None)
        oracle = T._oracle(q, window, seq_lens)
        row = {"case": param.id, "tokens": tokens, "start": start, "window_rows": pages * T.PAGE,
               "heads": heads, "dense_sim_seconds": round(dense_s, 1), "fp32": {}, "bf16": {}}
        for name, ref in refs.items():
            row["fp32"][name] = _metrics(dense, ref)
            row["bf16"][name] = _metrics(dense.to(torch.bfloat16), ref.to(torch.bfloat16))
        row["fp32"]["oracle_f64"] = _metrics(dense, oracle)
        row["fp32"]["sparse_lowp_vs_oracle_f64"] = _metrics(refs["sparse_lowp"], oracle)
        row["fp32"]["sparse_fp32_vs_oracle_f64"] = _metrics(refs["sparse_fp32"], oracle)
        row["bf16"]["oracle_f64"] = _metrics(dense.to(torch.bfloat16), oracle.to(torch.bfloat16))
        table.append(row)
        print(json.dumps({"case": param.id, "fp32_lowp": row["fp32"]["sparse_lowp"]}), flush=True)
    held = open_neuron_device_nodes()
    if held:
        print(f"Neuron device nodes open in a CPU-only run: {held}; no table written",
              file=sys.stderr)
        return 1
    with open(path, "w") as handle:
        json.dump(table, handle, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:]))
