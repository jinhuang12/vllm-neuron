# SPDX-License-Identifier: Apache-2.0
"""The backend writes a KDA carrier back to the device only when the carrier is a graph input.

The KDA forward's device contract: the state carriers of a compiled decode step are
inputs of that step -- the whole banks with ``state_slots`` (the bank form), or the
``bank[slot]`` views sliced eagerly OUTSIDE the compiled region and passed in. The
``neuron_libtorch`` backend bypasses functionalization and keeps an in-place write
only as a mutation of one of its input placeholders: ``AliasingOutputRewritePass``
maps each mutated input to an aliased output (``io_map``), and that map is what the
device writes back. A view sliced INSIDE the traced region is an intermediate; its
``copy_`` is rewritten out of place, the map stays empty, and the device keeps the old
rows while the step's output is still right -- a silent stale state. Eager CPU and the
simulator write all three forms, so this test runs the backend's own FX passes on the
graphs dynamo captures for the production write-back (``scatter_bank_rows`` and the
whole-view ``copy_`` of ``_fused_decode_requests``) and pins the ``io_map`` of each
form, which is the property the device behaviour follows.
"""

from __future__ import annotations

import copy

import pytest
import torch

from libtorch_neuronx_lite.fx_passes import get_default_pass_manager

from vllm_neuron.functional.state_banks import scatter_bank_rows

HEADS, KDIM, TAPS = 1, 128, 4
CHECKPOINTS = 4
BANK_SLOTS = 6
SLOT_IDS = (4, 1)
BATCH = len(SLOT_IDS)
CONV_ROW = (TAPS - 1, 3 * HEADS * KDIM)
REC_ROW = (HEADS, KDIM, KDIM)


def _write_rows(carrier: torch.Tensor, new_rows: torch.Tensor) -> None:
    """The view form's write of ``_fused_decode_requests``: the whole view, in place."""
    carrier[0:CHECKPOINTS].copy_(new_rows[0:CHECKPOINTS])


def _step_bank_form(conv_bank, rec_bank, slots, new_conv, new_rec):
    scatter_bank_rows(conv_bank, slots, new_conv)
    scatter_bank_rows(rec_bank, slots, new_rec)
    return new_rec[:, -1, 0, 0]  # stands in for the step's ``core``


def _step_eager_views(conv_0, conv_1, rec_0, rec_1, new_conv, new_rec):
    for index, (conv, rec) in enumerate(((conv_0, rec_0), (conv_1, rec_1))):
        _write_rows(conv, new_conv[index])
        _write_rows(rec, new_rec[index])
    return new_rec[:, -1, 0, 0]


def _step_views_inside(conv_bank, rec_bank, new_conv, new_rec):
    for index, slot in enumerate(SLOT_IDS):
        _write_rows(conv_bank[slot], new_conv[index])
        _write_rows(rec_bank[slot], new_rec[index])
    return new_rec[:, -1, 0, 0]


def _compile_with_probe(fn, mutated: list[set[str]]):
    """``torch.compile`` with a backend that runs the Neuron FX passes on a copy of the
    captured graph, records which of ``fn``'s arguments the aliasing ``io_map`` maps
    (``{output index: input placeholder index}``; dynamo orders placeholders by first
    use, so the index is resolved to the placeholder's name) and executes the graph
    eagerly, so the CPU result is checked too."""

    def backend(gm: torch.fx.GraphModule, example_inputs):
        placeholders = [str(node.target) for node in gm.graph.nodes if node.op == "placeholder"]
        _, metadata = get_default_pass_manager().run_passes(
            copy.deepcopy(gm), target_device="xla", compiler_workdir=None
        )
        io_map = metadata["aliasing_output_rewrite"]["io_map"]
        names = {placeholders[index] for index in io_map.values()}
        mutated.append({arg for arg in fn.__code__.co_varnames[: fn.__code__.co_argcount]
                        if any(arg in name for name in names)})
        assert len(names) == len(io_map) and len(mutated[-1]) == len(names), (io_map, placeholders)
        return gm.forward

    return torch.compile(fn, backend=backend, fullgraph=True, dynamic=False)


def _banks_and_rows():
    gen = torch.Generator().manual_seed(9700)
    conv_bank = torch.randn((BANK_SLOTS, CHECKPOINTS, *CONV_ROW), generator=gen).to(torch.bfloat16)
    rec_bank = torch.randn((BANK_SLOTS, CHECKPOINTS, *REC_ROW), generator=gen)
    new_conv = torch.randn((BATCH, CHECKPOINTS, *CONV_ROW), generator=gen).to(torch.bfloat16)
    new_rec = torch.randn((BATCH, CHECKPOINTS, *REC_ROW), generator=gen)
    return conv_bank, rec_bank, new_conv, new_rec


def _assert_written(conv_bank, rec_bank, new_conv, new_rec, before, label):
    for index, slot in enumerate(SLOT_IDS):
        assert torch.equal(conv_bank[slot], new_conv[index]), label
        assert torch.equal(rec_bank[slot], new_rec[index]), label
    untouched = [slot for slot in range(BANK_SLOTS) if slot not in SLOT_IDS]
    assert torch.equal(conv_bank[untouched], before[0][untouched]), label
    assert torch.equal(rec_bank[untouched], before[1][untouched]), label


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def test_bank_form_aliases_both_banks():
    """Whole banks + ``state_slots``: the ``index_copy_`` mutates two inputs, and both
    are aliased outputs -- the production write the device keeps."""
    conv_bank, rec_bank, new_conv, new_rec = _banks_and_rows()
    before = (conv_bank.clone(), rec_bank.clone())
    mutated: list[set[str]] = []
    slots = torch.tensor(SLOT_IDS, dtype=torch.int64)
    _compile_with_probe(_step_bank_form, mutated)(conv_bank, rec_bank, slots, new_conv, new_rec)
    assert mutated == [{"conv_bank", "rec_bank"}], mutated
    _assert_written(conv_bank, rec_bank, new_conv, new_rec, before, "bank form")


def test_eager_views_passed_as_inputs_alias_every_view():
    """``bank[slot]`` views sliced outside the compiled region and passed in: each
    view is an input placeholder, each ``copy_`` is kept as an aliased output."""
    conv_bank, rec_bank, new_conv, new_rec = _banks_and_rows()
    before = (conv_bank.clone(), rec_bank.clone())
    mutated: list[set[str]] = []
    views = (*(conv_bank[slot] for slot in SLOT_IDS), *(rec_bank[slot] for slot in SLOT_IDS))
    _compile_with_probe(_step_eager_views, mutated)(*views, new_conv, new_rec)
    assert mutated == [{"conv_0", "conv_1", "rec_0", "rec_1"}], mutated
    _assert_written(conv_bank, rec_bank, new_conv, new_rec, before, "eager views")


def test_views_sliced_inside_the_region_alias_nothing():
    """The trap: ``bank[slot]`` taken inside the traced step is an intermediate, the
    ``copy_`` through it mutates no input, the ``io_map`` is empty -- on device the
    rows are not written back. Eager CPU writes them, so the map is the only witness."""
    conv_bank, rec_bank, new_conv, new_rec = _banks_and_rows()
    before = (conv_bank.clone(), rec_bank.clone())
    mutated: list[set[str]] = []
    _compile_with_probe(_step_views_inside, mutated)(conv_bank, rec_bank, new_conv, new_rec)
    assert mutated == [set()], mutated
    _assert_written(conv_bank, rec_bank, new_conv, new_rec, before, "views inside (CPU writes)")
