# SPDX-License-Identifier: Apache-2.0
"""Device repro and check of the bank-form KDA graph inputs.

Run through the device lease on slice ``host`` (one trn2 chip, 4 logical cores):

    PATH=<venv>/bin:$PATH NEURON_LIBTORCH_CACHE_ROOT=<compile cache> \\
    python3 <devlease.py> slice host -- \\
        <venv>/bin/python test/hardware/bank_input_contiguity.py \\
        --tree <worktree> --json <out> --form both --batches 2 64

A bs=64 server failed its first bank-form decode warmup (b2/s2048) in
``libtorch_neuronx_lite/compile/backend.py:552 executor.execute`` with "Detected
non-contiguous slicing for requested Device Tensor". The bank form hands each KDA layer its
two WHOLE state banks as graph inputs; at 3098da3 the KV allocation built them as
slot-strided ``torch.as_strided`` views of one raw buffer (both states side by side in
every slot), and the executor accepts only a contiguous slice of a storage.

What runs here, on one KDA layer's raw buffer at the bs=64 line's real geometry (64 slots of
``recurrent_state_slot_bytes`` = 67840 bytes: conv ``[3, 384]`` bf16 beside recurrent
``[1, 128, 128]`` fp32, one head per rank at TP=64), with the production gather/scatter
(:mod:`vllm_neuron.functional.state_banks`) as the captured graph:

* ``--form old``: the banks are built exactly as 3098da3's allocation built them (the
  loop is copied below, same strides and offsets). Expected: the graph compiles and its
  first execution raises the executor's non-contiguous-slicing error (the repro);
* ``--form new``: the banks are built by the fixed allocation
  (:func:`vllm_neuron.vllm.worker.glm5next_state_banks.state_bank_regions`: the same raw
  buffer, each state one contiguous region). Expected: for every ``B`` in ``--batches``
  the graph executes ``--steps`` times, every execution's output and the whole raw buffer
  after it are bitwise equal to a CPU reference stepping host copies of the same banks;
  the view form's row write (``bank[slot].copy_(host row)``, the B=1 and prefill path)
  between two executions lands where the next execution reads it.

The JSON report holds the strides of both forms, the error text of the repro, and every
check. Exit status 0 when every expected outcome was observed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--tree", type=pathlib.Path, required=True)
parser.add_argument("--json", type=pathlib.Path, required=True)
parser.add_argument("--form", choices=("old", "new", "both"), default="both")
parser.add_argument("--batches", type=int, nargs="+", default=[2, 64])
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--slots", type=int, default=64)
args = parser.parse_args()
sys.path.insert(0, str(args.tree.resolve()))
sys.dont_write_bytecode = True

import torch  # noqa: E402

import vllm_neuron  # noqa: E402,F401 -- registers the Neuron compilation backend
from vllm_neuron.functional.state_banks import gather_bank_rows, scatter_bank_rows  # noqa: E402

DEVICE = "neuron:0"
#: The bs=64 line's KDA state geometry at TP=64 (``MambaStateShapeCalculator.kda_state_shape``
#: with one head per rank; dtypes from ``kda_state_dtype(bf16, "auto")``).
SHAPES = ((3, 384), (1, 128, 128))
DTYPES = (torch.bfloat16, torch.float32)
ALIGN = 256  # kv_spec_patch.RECURRENT_SLOT_ALIGN_BYTES
STATE_BYTES = sum(
    int(torch.empty(shape, device="meta").numel()) * dtype.itemsize
    for shape, dtype in zip(SHAPES, DTYPES)
)
SLOT_BYTES = -(-STATE_BYTES // ALIGN) * ALIGN
assert SLOT_BYTES == 67840, SLOT_BYTES  # test_kv_spec_patch pins this number
SLOTS = int(args.slots)

report: dict = {
    "tree": str(args.tree.resolve()),
    "vllm_neuron": vllm_neuron.__file__,
    "torch": torch.__version__,
    "visible_cores": os.environ.get("NEURON_RT_VISIBLE_CORES"),
    "cache_root": os.environ.get("NEURON_LIBTORCH_CACHE_ROOT"),
    "geometry": {
        "slots": SLOTS, "shapes": [list(s) for s in SHAPES], "dtypes": [str(d) for d in DTYPES],
        "slot_bytes": SLOT_BYTES, "raw_bytes": SLOTS * SLOT_BYTES,
    },
    "forms": {},
}
expected_ok = True


def say(text: str) -> None:
    print(f"[bank_input_contiguity] {text}", flush=True)


def strided_slot_banks(raw: torch.Tensor) -> list[torch.Tensor]:
    """3098da3's allocation (neuron_model_runner.py:10581-10616): slot-strided views."""
    num_slots = raw.numel() // SLOT_BYTES
    state_tensors = []
    state_offset_bytes = 0
    for shape, dtype in zip(SHAPES, DTYPES, strict=True):
        dtype_size = dtype.itemsize
        assert SLOT_BYTES % dtype_size == 0
        target_shape = (num_slots, *shape)
        contiguous = torch.empty(target_shape, device="meta").stride()
        assert state_offset_bytes % dtype_size == 0
        state_tensors.append(
            torch.as_strided(
                raw.view(dtype),
                size=target_shape,
                stride=(SLOT_BYTES // dtype_size, *contiguous[1:]),
                storage_offset=state_offset_bytes // dtype_size,
            )
        )
        state_offset_bytes += contiguous[0] * dtype_size
    return state_tensors


def region_banks(raw: torch.Tensor) -> list[torch.Tensor]:
    """The fixed allocation: the runner's own helper, one contiguous region per state."""
    from vllm_neuron.vllm.worker.glm5next_state_banks import state_bank_regions

    return state_bank_regions(raw, SHAPES, DTYPES, slot_bytes=SLOT_BYTES)


BUILDERS = {"old": strided_slot_banks, "new": region_banks}


def fill(banks: list[torch.Tensor]) -> None:
    """Deterministic, exactly representable values: slot and element both readable."""
    for index, bank in enumerate(banks):
        rows = bank.shape[0]
        per_row = bank[0].numel()
        values = (torch.arange(rows).reshape(rows, 1) * 2.0 + index
                  + (torch.arange(per_row) % 7).reshape(1, per_row) * 0.125)
        bank.copy_(values.reshape(bank.shape).to(bank.dtype))


def step(conv, recurrent, slots):
    """The bank form's own graph: gather by slot, advance, write back onto the banks."""
    conv_rows = gather_bank_rows(conv, slots)
    recurrent_rows = gather_bank_rows(recurrent, slots)
    scatter_bank_rows(conv, slots, conv_rows + 1.0)
    scatter_bank_rows(recurrent, slots, recurrent_rows + 1.0)
    # A per-row checksum of what this execution read, before its own write.
    return conv_rows.to(torch.float32).sum(dim=(1, 2)) + recurrent_rows.sum(dim=(1, 2, 3))


def slot_order(batch: int) -> list[int]:
    if batch == 2:
        return [37, 5]
    gen = torch.Generator().manual_seed(20261007)
    return torch.randperm(SLOTS, generator=gen).tolist()[:batch]


def describe(banks) -> list[dict]:
    return [
        {"shape": list(b.shape), "stride": list(b.stride()), "storage_offset": b.storage_offset(),
         "dtype": str(b.dtype), "is_contiguous": bool(b.is_contiguous()),
         "row_is_contiguous": bool(b[0].is_contiguous())}
        for b in banks
    ]


def run_form(form: str) -> None:
    global expected_ok
    entry: dict = {"batches": {}}
    report["forms"][form] = entry
    build = BUILDERS[form]
    raw_host = torch.zeros(SLOTS * SLOT_BYTES, dtype=torch.int8)
    host_banks = build(raw_host)
    fill(host_banks)
    entry["banks"] = describe(host_banks)
    say(f"form={form} banks={entry['banks']}")
    raw_device = raw_host.to(DEVICE)
    device_banks = build(raw_device)
    entry["device_banks_contiguous"] = [bool(b.is_contiguous()) for b in device_banks]
    compiled = torch.compile(step, backend="neuron_libtorch", fullgraph=True, dynamic=False)

    for batch in args.batches:
        case: dict = {"slots": slot_order(batch), "steps": []}
        entry["batches"][str(batch)] = case
        slots_host = torch.tensor(case["slots"], dtype=torch.int64)
        slots_device = slots_host.to(DEVICE)
        try:
            for index in range(args.steps):
                started = time.perf_counter()
                out = compiled(*device_banks, slots_device)
                out_host = out.cpu()
                elapsed = time.perf_counter() - started
                expect = step(*host_banks, slots_host)
                raw_after = raw_device.cpu()
                case["steps"].append({
                    "output_bitwise": bool(torch.equal(out_host, expect)),
                    "raw_buffer_bitwise": bool(torch.equal(raw_after, raw_host)),
                    "wall_s": elapsed,
                })
                say(f"form={form} B={batch} step={index} {case['steps'][-1]}")
            case["ok"] = all(s["output_bitwise"] and s["raw_buffer_bitwise"] for s in case["steps"])
            case["executed"] = True
        except Exception as caught:  # noqa: BLE001 -- the refusal is what is read
            case["executed"] = False
            case["error"] = f"{type(caught).__name__}: {caught}"
            case["error_is_non_contiguous_slicing"] = "non-contiguous slicing" in str(caught)
            say(f"form={form} B={batch} raised {case['error'][:200]}")
        if form == "old":
            # The repro: the executor refuses the strided bank.
            case["expected"] = (not case["executed"]) and case.get("error_is_non_contiguous_slicing", False)
        else:
            case["expected"] = case.get("ok", False)
        expected_ok = expected_ok and case["expected"]

    if form == "new" and all(c["executed"] for c in entry["batches"].values()):
        # The view form's write between two executions: one row, a contiguous slice.
        conv, recurrent = device_banks
        slot = 37
        row = torch.full(SHAPES[0], 9.5, dtype=DTYPES[0])
        conv[slot].copy_(row)
        host_banks[0][slot].copy_(row)
        slots_host = torch.tensor([slot, 5], dtype=torch.int64)
        out_host = compiled(conv, recurrent, slots_host.to(DEVICE)).cpu()
        expect = step(*host_banks, slots_host)
        entry["row_write_between_executions"] = {
            "output_bitwise": bool(torch.equal(out_host, expect)),
            "raw_buffer_bitwise": bool(torch.equal(raw_device.cpu(), raw_host)),
        }
        say(f"form=new row write: {entry['row_write_between_executions']}")
        expected_ok = expected_ok and all(entry["row_write_between_executions"].values())


try:
    for form in (("old", "new") if args.form == "both" else (args.form,)):
        run_form(form)
except Exception:  # noqa: BLE001
    report["error"] = traceback.format_exc()
    print(report["error"], flush=True)
    expected_ok = False

report["expected_outcomes_observed"] = expected_ok
args.json.parent.mkdir(parents=True, exist_ok=True)
args.json.write_text(json.dumps(report, indent=2) + "\n")
say(f"expected_outcomes_observed={expected_ok} -> {args.json}")
sys.exit(0 if expected_ok else 1)
