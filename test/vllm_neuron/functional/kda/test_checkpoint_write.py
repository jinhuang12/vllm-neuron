# SPDX-License-Identifier: Apache-2.0
"""``write_page_rows``: rows of state into pages of a pool tensor, at the KDA checkpoint geometry.

Read here, on CPU tensors (no device):

1. the torch path writes each carrier's bytes at its offset of its page and no other byte,
   with int32 or int64 block ids, for rows of any bit pattern, and returns the pages;
2. the kernel on the NKI simulator writes the same row values, on one and on two 128-row
   tiles, and returns the operand it wrote. The simulator casts a kernel's results back by
   value, so this is a value-level check; whether every bit pattern survives is a
   property of the device's DMA;
3. traced on meta tensors, the write is one kernel call on the pages placeholder, the
   graph returns its result, and no node is a dtype view (the backend refuses one when it
   lowers the graph);
4. refusals by name.

The compile through the backend is ``test_checkpoint_write_compile``.
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.kda import checkpoint_write as cw
from vllm_neuron.functional.kda.checkpoint_write import CheckpointWriteError, write_page_rows

pytestmark = [pytest.mark.fast]

#: The served KDA checkpoint: conv bf16 [3, 384] at byte 0 and recurrent f32 [1, 128, 128]
#: at byte 2304 of a 131072-byte page.
PAGE = 131072
SHAPES = ((3, 384), (1, 128, 128))
DTYPES = (torch.bfloat16, torch.float32)
OFFSETS = (0, 2304)
USED = 2304 + 65536


def _bits(shape, dtype, gen) -> torch.Tensor:
    """A tensor of ``shape`` whose bytes are random: every bit pattern, NaNs included."""
    count = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
    raw = torch.randint(-128, 128, (count,), dtype=torch.int8, generator=gen)
    return raw.view(dtype).view(shape)


def _rows(count: int, gen, *, bits: bool) -> list[torch.Tensor]:
    if bits:
        return [_bits((count, *shape), dtype, gen) for shape, dtype in zip(SHAPES, DTYPES)]
    return [torch.randn(count, *shape, generator=gen).to(dtype)
            for shape, dtype in zip(SHAPES, DTYPES)]


def _expected(before: torch.Tensor, block_ids, rows) -> torch.Tensor:
    want = before.clone().view(-1, PAGE)
    for i, block in enumerate(block_ids):
        for carrier, offset in zip(rows, OFFSETS):
            data = carrier[i].reshape(-1).view(torch.int8)
            want[block, offset:offset + data.numel()] = data
    return want.view(-1)


@pytest.fixture
def torch_path(monkeypatch):
    monkeypatch.setattr(cw, "can_run_kernel", lambda tensor: False)


# ── 1. the torch path ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_the_torch_path_writes_each_carriers_bytes_and_no_other_byte(torch_path, id_dtype):
    gen = torch.Generator().manual_seed(1)
    raw = torch.randint(-128, 128, (6 * PAGE,), dtype=torch.int8, generator=gen)
    before = raw.clone()
    rows = _rows(2, gen, bits=True)
    block_ids = [5, 2]

    written = write_page_rows(raw.view(6, PAGE), torch.tensor(block_ids, dtype=id_dtype), rows,
                              OFFSETS, PAGE)

    assert torch.equal(raw, _expected(before, block_ids, rows))
    assert written.data_ptr() == raw.data_ptr()


def test_the_torch_path_writes_through_a_cache_view_of_another_dtype(torch_path):
    """The latent layer's cache view of the pool tensor is a page operand as the int8 view is."""
    gen = torch.Generator().manual_seed(2)
    raw = torch.randint(-128, 128, (4 * PAGE,), dtype=torch.int8, generator=gen)
    before = raw.clone()
    rows = _rows(1, gen, bits=True)

    write_page_rows(raw.view(torch.bfloat16).view(4 * 128, 1, 512), torch.tensor([3]), rows,
                    OFFSETS, PAGE)

    assert torch.equal(raw, _expected(before, [3], rows))


# ── 2. the kernel on the simulator ──────────────────────────────────────────────────


@pytest.mark.parametrize("count", [2, 130], ids=["one-tile", "two-tiles"])
def test_the_kernel_writes_the_rows_values_on_the_simulator(count):
    if not cw.can_run_kernel(torch.empty(1)):
        pytest.skip("the NKI simulator is off (NKI_SIMULATOR=1 with VLLM_NEURON_CPU_MODE=1)")
    gen = torch.Generator().manual_seed(3)
    blocks = 160
    raw = torch.randint(-128, 128, (blocks * PAGE,), dtype=torch.int8, generator=gen)
    before = raw.clone()
    block_ids = torch.randperm(blocks, generator=gen)[:count].tolist()
    rows = _rows(count, gen, bits=False)
    pages = raw.view(blocks, PAGE)

    written = write_page_rows(pages, torch.tensor(block_ids, dtype=torch.int32), rows, OFFSETS,
                              PAGE)

    got = pages[block_ids]
    for carrier, shape, dtype, offset in zip(rows, SHAPES, DTYPES, OFFSETS):
        nbytes = carrier[0].numel() * carrier.element_size()
        values = got[:, offset:offset + nbytes].contiguous().view(dtype).view(count, *shape)
        assert torch.equal(values, carrier)
    others = sorted(set(range(blocks)) - set(block_ids))
    assert torch.equal(pages[others], before.view(blocks, PAGE)[others])
    assert torch.equal(got[:, USED:], before.view(blocks, PAGE)[block_ids][:, USED:])
    assert written is pages


# ── 3. traced ───────────────────────────────────────────────────────────────────────


def test_traced_on_meta_tensors_the_write_is_one_kernel_call_on_the_pages():
    seen = []

    def record(gm, example_inputs):
        seen.append(gm)
        return gm.forward

    def step(pages, block_ids, conv, recurrent):
        return write_page_rows(pages, block_ids, [conv, recurrent], OFFSETS, PAGE)

    meta = torch.device("meta")
    torch._dynamo.reset()
    torch.compile(step, backend=record, fullgraph=True)(
        torch.empty(8 * 128, 1, 512, dtype=torch.bfloat16, device=meta),
        torch.empty(2, dtype=torch.int32, device=meta),
        *(torch.empty(2, *shape, dtype=dtype, device=meta) for shape, dtype in zip(SHAPES, DTYPES)))

    (graph,) = [gm.graph for gm in seen]
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    kernels = [node for node in graph.nodes if "nki_kernel_wrapper" in str(node.target)]
    assert len(kernels) == 1
    pages = next(node for node in placeholders if tuple(node.meta["example_value"].shape)
                 == (8 * 128, 1, 512))
    assert kernels[0].kwargs["args"][0] is pages
    (output,) = [node for node in graph.nodes if node.op == "output"]
    assert kernels[0] in output.args[0]
    assert not [node for node in graph.nodes if "view" in str(node.target)
                and any(isinstance(a, torch.dtype) for a in (*node.args, *node.kwargs.values()))]
    torch._dynamo.reset()


# ── 4. refusals ─────────────────────────────────────────────────────────────────────


def _good():
    gen = torch.Generator().manual_seed(4)
    return torch.zeros(4, PAGE, dtype=torch.int8), torch.tensor([1, 2]), _rows(2, gen, bits=False)


@pytest.mark.parametrize("pages, match", [
    (torch.zeros(4, PAGE - 2, dtype=torch.int8), "whole 131072-byte pages"),
    (torch.zeros(PAGE, 4, dtype=torch.int8).t(), "not contiguous"),
])
def test_pages_that_are_not_whole_contiguous_pages_are_refused(pages, match):
    _, block_ids, rows = _good()
    with pytest.raises(CheckpointWriteError, match=match):
        write_page_rows(pages, block_ids, rows, OFFSETS, PAGE)


@pytest.mark.parametrize("block_ids", [torch.tensor([1.0, 2.0]), torch.tensor([[1, 2]])])
def test_block_ids_that_are_not_an_integer_vector_are_refused(block_ids):
    pages, _, rows = _good()
    with pytest.raises(CheckpointWriteError, match="int32 or int64"):
        write_page_rows(pages, block_ids, rows, OFFSETS, PAGE)


def test_rows_of_another_count_are_refused():
    pages, block_ids, rows = _good()
    with pytest.raises(CheckpointWriteError, match="carrier 1"):
        write_page_rows(pages, block_ids, [rows[0], rows[1][:1]], OFFSETS, PAGE)


def test_a_carrier_that_does_not_fit_its_page_is_refused():
    pages, block_ids, rows = _good()
    with pytest.raises(CheckpointWriteError, match="carrier 1: 65536 byte"):
        write_page_rows(pages, block_ids, rows, (0, PAGE - 4), PAGE)


def test_more_carriers_than_one_launch_writes_are_refused_on_the_kernel_route(monkeypatch):
    monkeypatch.setattr(cw, "can_run_kernel", lambda tensor: True)
    pages, block_ids, rows = _good()
    small = [torch.zeros(2, 4, dtype=torch.int8)] * (cw.CARRIERS_PER_LAUNCH + 1)
    with pytest.raises(CheckpointWriteError, match="up to 3 carriers"):
        write_page_rows(pages, block_ids, small, [16 * k for k in range(len(small))], PAGE)


def test_a_device_tensor_with_nki_kernels_off_is_refused(torch_path):
    meta = torch.device("meta")
    with pytest.raises(CheckpointWriteError, match="needs its NKI kernel"):
        write_page_rows(torch.empty(4, PAGE, dtype=torch.int8, device=meta),
                        torch.empty(2, dtype=torch.int32, device=meta),
                        [torch.empty(2, *s, dtype=d, device=meta) for s, d in zip(SHAPES, DTYPES)],
                        OFFSETS, PAGE)
