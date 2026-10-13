# SPDX-License-Identifier: Apache-2.0
"""The KDA checkpoint store: one recurrent state row per KV block, in the block's own page.

With prefix caching on, block ``b`` of a request's KDA group stands for the state that
group's layers reach at the block's last token. vLLM gives every KV cache group its
blocks from one pool, and each pool tensor is shared by one layer of every group; the
runner allocates each pool tensor whole, for the latent layer on it. So page ``b`` of
the tensor a KDA layer shares is that layer's while its group holds block ``b``, and the
checkpoint is kept there: one committed state row of every carrier at fixed byte
offsets. The prefill's checkpoint write and the runner's restore are coded against the
functions here, so their names and signatures are pinned.

Read here, with the served TP 64 specs (no model weights):

1. names and signatures;
2. the layout: carrier offsets and sizes from the ``MambaSpec``, on a plain and on a
   speculative server, refused when a checkpoint does not fit a page;
3. the pool arithmetic: each KDA layer shares a tensor with one latent layer, the layers
   of one group sit on distinct tensors, and one boundary's checkpoints fill 34 of the
   44 pages (36 with the draft layer) that one block id per group spans;
4. the page view: no copy, contiguous, whole pages;
5. aliasing: the page a write fills is the latent layer's page of the same block id,
   byte for byte, and a latent write to another block leaves it as it was;
6. a write then a read returns what was written and moves no other page; padding rows
   may name the null block; the write finds its pages by bytes, whatever the tensor's
   row size;
7. a restore writes only the slot's entering row;
8. traced on meta tensors (the device route), a write is one NKI kernel call on the page
   view itself with no dtype view in the graph, and the graph returns its result; a read
   traces to one ``index_select`` and dtype views, so it runs eager;
9. refusals by name.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/vllm/worker/test_kda_checkpoint_store.py
"""

from __future__ import annotations

import dataclasses
import inspect
import math

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from test.vllm_neuron.worker import test_kv_budget_glm53f as kv
from test.vllm_neuron.worker import test_kv_budget_kda_default as kda_default
from vllm_neuron.vllm.patches.kv_spec_patch import recurrent_state_slot_bytes
from vllm_neuron.vllm.worker import glm5next_state_banks as banks_module
from vllm_neuron.vllm.worker.glm5next_state_banks import (
    KdaCheckpointLayout,
    kda_checkpoint_layout,
    kda_checkpoint_pages,
    read_kda_checkpoints,
    restore_kda_slot,
    state_bank_regions,
    write_kda_checkpoints,
)

pytestmark = [pytest.mark.fast]

BLOCKS = 6
SLOTS = 3
EMPTY = inspect.Parameter.empty
POSITIONAL = inspect.Parameter.POSITIONAL_OR_KEYWORD
#: The served checkpoint: conv (3, 384) bf16 + recurrent (1, 128, 128) fp32 at TP 64.
CONV_BYTES, RECURRENT_BYTES = 3 * 384 * 2, 128 * 128 * 4
CHECKPOINT_BYTES = CONV_BYTES + RECURRENT_BYTES
PAGE_BYTES = 131_072


@pytest.fixture(scope="module")
def served_specs():
    """The runner's KV cache specs for the served model at TP 64, by layer name."""
    layers, text_config = kv.glm53f_layer_specs()
    runner = kv.fake_runner(layers, max_num_seqs=1, max_model_len=8192,
                            text_config=text_config)
    return runner.get_kv_cache_spec()


@pytest.fixture(scope="module")
def kda_spec(served_specs):
    from vllm.v1.kv_cache_interface import MambaSpec

    specs = [spec for spec in served_specs.values() if isinstance(spec, MambaSpec)]
    assert len(specs) == kv.KDA_LAYERS
    assert len({(spec.shapes, spec.dtypes, spec.page_size_bytes) for spec in specs}) == 1
    return specs[0]


@pytest.fixture(scope="module")
def latent_spec(served_specs):
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    specs = [spec for spec in served_specs.values() if isinstance(spec, MLAAttentionSpec)]
    assert len(specs) == kv.MLA_LAYERS
    return specs[0]


def _raw(blocks=BLOCKS, page=PAGE_BYTES) -> torch.Tensor:
    return torch.zeros(blocks * page, dtype=torch.int8)


def _rows(layout, count: int, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn((count, *shape), generator=generator).to(dtype)
            for shape, dtype in zip(layout.shapes, layout.dtypes, strict=True)]


def _noise(tensor: torch.Tensor, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    tensor.copy_(torch.randint(-128, 128, tuple(tensor.shape), generator=generator,
                               dtype=torch.int8).view(tensor.dtype)
                 if tensor.dtype == torch.int8 else
                 torch.randn(tuple(tensor.shape), generator=generator).to(tensor.dtype))


def _same_bytes(got: torch.Tensor, want: torch.Tensor) -> bool:
    """Byte equality: noise bytes read as floats hold NaNs, which never compare equal."""
    return torch.equal(got.contiguous().view(torch.int8), want.contiguous().view(torch.int8))


def _ids(*ids: int) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.int64)


# ── 1. names and signatures ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("function, parameters", [
    (kda_checkpoint_layout, ["spec"]),
    (kda_checkpoint_pages, ["raw", "layout"]),
    (write_kda_checkpoints, ["pages", "block_ids", "rows", "layout"]),
    (read_kda_checkpoints, ["pages", "block_ids", "layout"]),
    (restore_kda_slot, ["banks", "slot", "pages", "block_id", "layout"]),
], ids=lambda value: getattr(value, "__name__", ""))
def test_the_store_functions_keep_their_names_and_signatures(function, parameters):
    """The prefill's checkpoint write and the runner's restore call these by name."""
    assert function.__name__ in banks_module.__all__
    signature = inspect.signature(function)
    assert [(p.name, p.kind, p.default) for p in signature.parameters.values()] == [
        (name, POSITIONAL, EMPTY) for name in parameters
    ]


def test_the_layout_keeps_its_fields():
    assert "KdaCheckpointLayout" in banks_module.__all__
    assert [field.name for field in dataclasses.fields(KdaCheckpointLayout)] == [
        "shapes", "dtypes", "offsets", "page_bytes",
    ]


# ── 2. the layout ───────────────────────────────────────────────────────────────────


def test_the_served_checkpoint_is_the_conv_row_then_the_recurrent_row(kda_spec):
    layout = kda_checkpoint_layout(kda_spec)
    assert layout.shapes == ((3, 384), (1, 128, 128))
    assert layout.dtypes == (torch.bfloat16, torch.float32)
    assert layout.offsets == (0, CONV_BYTES)
    assert layout.nbytes == (CONV_BYTES, RECURRENT_BYTES)
    assert layout.used_bytes == CHECKPOINT_BYTES == 67_840
    assert layout.page_bytes == kda_spec.page_size_bytes == PAGE_BYTES


def test_a_speculative_server_keeps_the_same_checkpoint(kda_spec):
    """A slot holds 1 + k rows, one per verify token; a checkpoint is the committed row."""
    speculative = dataclasses.replace(kda_spec, num_speculative_blocks=3)
    assert recurrent_state_slot_bytes(speculative) == 4 * CHECKPOINT_BYTES
    assert kda_checkpoint_layout(speculative) == kda_checkpoint_layout(kda_spec)


def test_the_layout_follows_the_spec(kda_spec):
    """Each carrier starts at a multiple of its own element size."""
    other = dataclasses.replace(kda_spec, shapes=((3, 3), (1, 4, 4)),
                                dtypes=(torch.bfloat16, torch.float32))
    layout = kda_checkpoint_layout(other)
    assert layout.offsets == (0, 20)  # 18 bytes of conv, rounded up to 4
    assert layout.used_bytes == 20 + 64


def test_a_checkpoint_larger_than_a_page_is_refused(kda_spec):
    """An unpadded page of a bf16 carrier then an fp32 one: 6 + 4 bytes, and the fp32
    row starts at byte 8, so the checkpoint takes 12."""
    tight = dataclasses.replace(kda_spec, shapes=((3,), (1,)), page_size_padded=None,
                                dtypes=(torch.bfloat16, torch.float32))
    assert tight.page_size_bytes == 10
    with pytest.raises(ValueError, match="takes 12 bytes and a KV page holds 10"):
        kda_checkpoint_layout(tight)


# ── 3. the pool arithmetic ──────────────────────────────────────────────────────────


def _prefix_caching_worker(monkeypatch):
    """The worker fake for one 8K sequence with ``--enable-prefix-caching`` resolved."""
    cfg, _ = kda_default._resolve_past_the_refusal(
        monkeypatch, [kda_default.PC_FLAG_ON], max_model_len=8192)
    return kda_default._priced_worker(kv.glm53f_layer_specs(), cfg, seqs=1, length=8192)


def _pool(worker, specs):
    """vLLM's KV cache config for ``specs`` at the budget the worker returns."""
    from vllm.v1.core.kv_cache_utils import (
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
    )

    groups = get_kv_cache_groups(worker.vllm_config, specs)
    need = worker._kv_cache_need_bytes(log=False)
    return get_kv_cache_config_from_groups(worker.vllm_config, groups, need)


@pytest.mark.parametrize("draft_layer", [False, True], ids=["plain", "with-draft-layer"])
def test_one_boundarys_checkpoints_fit_the_pages_its_block_ids_span(monkeypatch, draft_layer):
    """Without drafting: 4 KDA groups x 11 pool tensors = 44 pages, 34 hold a checkpoint
    (40.0 % of their bytes, 51.8 % of each used page). With the draft layer (one more
    latent layer of the same spec): 3 groups x 12 tensors = 36 pages (48.9 %)."""
    from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec

    worker = _prefix_caching_worker(monkeypatch)
    specs = worker.model_runner.get_kv_cache_spec()
    assert {spec.block_size for spec in specs.values()} == {128}
    if draft_layer:
        latent_spec = next(s for s in specs.values() if isinstance(s, MLAAttentionSpec))
        specs = {name: (dataclasses.replace(spec, num_speculative_blocks=3)
                        if isinstance(spec, MambaSpec) else spec)
                 for name, spec in specs.items()}
        specs["layers.45.self_attn"] = latent_spec
    config = _pool(worker, specs)
    kda_groups = [group for group in config.kv_cache_groups
                  if isinstance(group.kv_cache_spec, MambaSpec)]
    group_of = {name: index for index, group in enumerate(config.kv_cache_groups)
                for name in group.layer_names}

    for tensor in config.kv_cache_tensors:
        # One latent layer per tensor, and at most one layer of each group.
        latent = [n for n in tensor.shared_by if isinstance(specs[n], MLAAttentionSpec)]
        assert len(latent) == 1, tensor.shared_by
        groups = [group_of[name] for name in tensor.shared_by]
        assert len(groups) == len(set(groups)), tensor.shared_by
        assert tensor.size // config.num_blocks == PAGE_BYTES
    layout = kda_checkpoint_layout(kda_groups[0].kv_cache_spec)
    spans = len(kda_groups) * len(config.kv_cache_tensors)
    held = sum(len(group.layer_names) for group in kda_groups)
    assert held == kv.KDA_LAYERS
    assert (len(kda_groups), len(config.kv_cache_tensors), spans) == (
        (3, 12, 36) if draft_layer else (4, 11, 44))
    assert held * layout.used_bytes <= spans * PAGE_BYTES
    assert held * layout.used_bytes == 2_306_560
    assert round(100 * held * layout.used_bytes / (spans * PAGE_BYTES), 1) == (
        48.9 if draft_layer else 40.0)
    assert round(100 * layout.used_bytes / PAGE_BYTES, 1) == 51.8


# ── 4. the page view ────────────────────────────────────────────────────────────────


def test_the_page_view_is_the_pool_tensor_itself_cut_into_whole_pages(kda_spec):
    layout = kda_checkpoint_layout(kda_spec)
    raw = _raw()
    pages = kda_checkpoint_pages(raw, layout)
    assert tuple(pages.shape) == (BLOCKS, PAGE_BYTES)
    assert pages.dtype == torch.int8
    assert pages.is_contiguous()
    assert pages.data_ptr() == raw.data_ptr()


def test_a_buffer_of_partial_pages_is_refused(kda_spec):
    layout = kda_checkpoint_layout(kda_spec)
    with pytest.raises(ValueError, match="whole pages of 131072"):
        kda_checkpoint_pages(torch.zeros(BLOCKS * PAGE_BYTES + 2, dtype=torch.int8), layout)


# ── 5. aliasing with the latent page ────────────────────────────────────────────────


def _latent_bank(raw, latent_spec) -> torch.Tensor:
    """The latent layer's view of the same buffer, as the runner builds it."""
    shape = (int(raw.numel()) // latent_spec.page_size_bytes, latent_spec.num_kv_heads,
             latent_spec.block_size, latent_spec.head_size)
    return raw.view(latent_spec.dtype).view(shape)


def test_a_checkpoint_fills_the_latent_layers_page_of_the_same_block_id(kda_spec,
                                                                          latent_spec):
    """The aliasing is the layout: vLLM hands block 4 to one group at a time, so while the
    KDA group holds it the latent layer never writes its page."""
    assert latent_spec.page_size_bytes == PAGE_BYTES
    layout = kda_checkpoint_layout(kda_spec)
    raw = _raw()
    _noise(raw, seed=1)
    latent = _latent_bank(raw, latent_spec)
    before = latent.clone()
    rows = _rows(layout, 1, seed=2)

    write_kda_checkpoints(kda_checkpoint_pages(raw, layout), _ids(4), rows, layout)

    page = latent[4].reshape(-1).view(torch.int8)
    assert torch.equal(page[:CONV_BYTES], rows[0].reshape(-1).view(torch.int8))
    assert torch.equal(page[CONV_BYTES:CHECKPOINT_BYTES], rows[1].reshape(-1).view(torch.int8))
    assert _same_bytes(page[CHECKPOINT_BYTES:],
                       before[4].reshape(-1).view(torch.int8)[CHECKPOINT_BYTES:])
    others = [0, 1, 2, 3, 5]
    assert torch.equal(latent[others].view(torch.int8), before[others].view(torch.int8))


def test_a_latent_write_to_another_block_leaves_the_checkpoint(kda_spec, latent_spec):
    """The latent layer's own write, ``index_copy_`` of token rows into its bank."""
    layout = kda_checkpoint_layout(kda_spec)
    raw = _raw()
    pages = kda_checkpoint_pages(raw, layout)
    rows = _rows(layout, 1, seed=3)
    write_kda_checkpoints(pages, _ids(2), rows, layout)

    latent = _latent_bank(raw, latent_spec)
    flat = latent.view(-1, latent_spec.num_kv_heads, latent_spec.head_size)
    tokens = torch.arange(3 * latent_spec.block_size, 4 * latent_spec.block_size)
    flat.index_copy_(0, tokens, torch.randn(len(tokens), 1, latent_spec.head_size,
                                            generator=torch.Generator().manual_seed(4))
                     .to(latent_spec.dtype))

    for got, want in zip(read_kda_checkpoints(pages, _ids(2), layout), rows, strict=True):
        assert torch.equal(got, want)


# ── 6. write and read ───────────────────────────────────────────────────────────────


def test_a_write_then_a_read_returns_the_rows_and_no_other_page_moves(kda_spec):
    layout = kda_checkpoint_layout(kda_spec)
    raw = _raw()
    _noise(raw, seed=5)
    pages = kda_checkpoint_pages(raw, layout)
    before = pages.clone()
    rows = _rows(layout, 2, seed=6)

    write_kda_checkpoints(pages, _ids(4, 2), rows, layout)

    for got, want in zip(read_kda_checkpoints(pages, _ids(4, 2), layout), rows, strict=True):
        assert torch.equal(got, want)
    assert torch.equal(pages[[0, 1, 3, 5]], before[[0, 1, 3, 5]])
    assert pages.data_ptr() == raw.data_ptr()


def test_a_padding_row_may_name_the_null_block(kda_spec):
    """Block 0 is vLLM's null block: the pool never hands it out as a real block and it
    is never the last entry of a hit, so no restore reads it."""
    from vllm.v1.core.block_pool import BlockPool

    from vllm_neuron.vllm.worker.neuron_model_runner import NULL_BLOCK_ID

    pool = BlockPool(num_gpu_blocks=4, enable_caching=True, hash_block_size=128)
    assert pool.null_block.block_id == NULL_BLOCK_ID == 0
    layout = kda_checkpoint_layout(kda_spec)
    pages = kda_checkpoint_pages(_raw(), layout)
    rows = _rows(layout, 3, seed=7)
    write_kda_checkpoints(pages, _ids(3, 0, 0), rows, layout)
    got = read_kda_checkpoints(pages, _ids(3), layout)
    assert all(torch.equal(g[0], r[0]) for g, r in zip(got, rows, strict=True))


def test_a_write_finds_its_pages_by_bytes_whatever_the_row_size(kda_spec):
    """The write takes any contiguous tensor of whole pages (the latent layer's cache view
    is one): rows of half a page are two rows per page."""
    layout = kda_checkpoint_layout(kda_spec)
    raw = _raw()
    _noise(raw, seed=14)
    before = kda_checkpoint_pages(raw, layout).clone()
    rows = _rows(layout, 1, seed=15)

    write_kda_checkpoints(raw.view(2 * BLOCKS, PAGE_BYTES // 2), _ids(4), rows, layout)

    pages = kda_checkpoint_pages(raw, layout)
    for got, want in zip(read_kda_checkpoints(pages, _ids(4), layout), rows, strict=True):
        assert torch.equal(got, want)
    assert torch.equal(pages[4, CHECKPOINT_BYTES:], before[4, CHECKPOINT_BYTES:])
    assert torch.equal(pages[[0, 1, 2, 3, 5]], before[[0, 1, 2, 3, 5]])


# ── 7. restore ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("draft_rows", [0, 3], ids=["plain", "speculative"])
def test_a_restore_writes_only_the_slots_entering_row(kda_spec, draft_rows):
    """Row 0 of the slot (the row a fresh slot reads); other slots and rows stay."""
    spec = dataclasses.replace(kda_spec, num_speculative_blocks=draft_rows)
    layout = kda_checkpoint_layout(spec)
    slot_bytes = recurrent_state_slot_bytes(spec)
    bank_raw = torch.zeros(SLOTS * slot_bytes, dtype=torch.int8)
    _noise(bank_raw, seed=8)
    banks = state_bank_regions(bank_raw, spec.shapes, spec.dtypes, slot_bytes=slot_bytes,
                               checkpoints=1 + draft_rows)
    before = [bank.clone() for bank in banks]
    pages = kda_checkpoint_pages(_raw(), layout)
    rows = _rows(layout, 1, seed=9)
    write_kda_checkpoints(pages, _ids(4), rows, layout)

    restore_kda_slot(banks, 1, pages, 4, layout)

    for bank, old, want in zip(banks, before, rows, strict=True):
        entering = bank[1, 0] if draft_rows else bank[1]
        assert torch.equal(entering, want[0])
        assert _same_bytes(bank[[0, 2]], old[[0, 2]])
        if draft_rows:
            assert _same_bytes(bank[1, 1:], old[1, 1:])


# ── 8. traced ───────────────────────────────────────────────────────────────────────


def test_traced_a_write_is_one_kernel_call_on_the_page_view_itself(kda_spec):
    """On the device route (meta tensors), the write is one NKI kernel call whose page
    operand is the graph input itself, and the graph returns the kernel's result, which
    the backend aliases to that input (``test_checkpoint_write_compile``)."""
    from vllm_neuron.functional.kda import checkpoint_write

    layout = kda_checkpoint_layout(kda_spec)
    if not checkpoint_write.can_run_kernel(torch.empty(1, device="meta")):
        pytest.skip("NKI kernels are off on this route (NKI_SIMULATOR=1 with CPU mode)")
    seen = []

    def record(gm, example_inputs):
        seen.append(gm)
        return gm.forward

    def step(pages, block_ids, conv_rows, recurrent_rows):
        return write_kda_checkpoints(pages, block_ids, [conv_rows, recurrent_rows], layout)

    meta = torch.device("meta")
    rows = [torch.empty(2, *shape, dtype=dtype, device=meta)
            for shape, dtype in zip(layout.shapes, layout.dtypes, strict=True)]
    torch._dynamo.reset()
    torch.compile(step, backend=record, fullgraph=True)(
        torch.empty(BLOCKS, PAGE_BYTES, dtype=torch.int8, device=meta),
        torch.empty(2, dtype=torch.int64, device=meta), *rows)
    torch._dynamo.reset()

    (graph,) = [gm.graph for gm in seen]
    pages = next(node for node in graph.nodes if node.op == "placeholder"
                 and tuple(node.meta["example_value"].shape) == (BLOCKS, PAGE_BYTES))
    kernels = [node for node in graph.nodes if "nki_kernel_wrapper" in str(node.target)]
    assert len(kernels) == 1
    assert kernels[0].kwargs["args"][0] is pages
    (output,) = [node for node in graph.nodes if node.op == "output"]
    assert kernels[0] in output.args[0]
    assert not [node for node in graph.nodes if "view" in str(node.target)
                and any(isinstance(a, torch.dtype) for a in (*node.args, *node.kwargs.values()))]


def test_traced_a_read_is_one_index_select_of_whole_pages(kda_spec):
    """The carriers are dtype views of the gathered pages, which the device backend
    refuses when it lowers a graph, so a read runs eager (``read_kda_checkpoints``)."""
    layout = kda_checkpoint_layout(kda_spec)
    pages = kda_checkpoint_pages(_raw(), layout)

    graph = make_fx(lambda pages, ids: read_kda_checkpoints(pages, ids, layout))(
        pages, _ids(3)).graph
    selects = [node for node in graph.nodes
               if node.op == "call_function" and "index_select" in str(node.target)]
    assert len(selects) == 1


# ── 9. refusals ─────────────────────────────────────────────────────────────────────


def test_a_write_of_rows_of_another_shape_is_refused_by_carrier(kda_spec):
    layout = kda_checkpoint_layout(kda_spec)
    rows = _rows(layout, 1, seed=11)
    rows[1] = rows[1][:, :, :64]
    with pytest.raises(ValueError, match="carrier 1"):
        write_kda_checkpoints(kda_checkpoint_pages(_raw(), layout), _ids(1), rows, layout)


def test_block_ids_must_be_an_int64_or_int32_vector(kda_spec):
    layout = kda_checkpoint_layout(kda_spec)
    with pytest.raises(ValueError, match="int64 or int32"):
        write_kda_checkpoints(kda_checkpoint_pages(_raw(), layout),
                              torch.tensor([1.0]), _rows(layout, 1, seed=12), layout)


def test_a_write_with_a_missing_carrier_is_refused(kda_spec):
    layout = kda_checkpoint_layout(kda_spec)
    with pytest.raises(ValueError, match="2 carrier"):
        write_kda_checkpoints(kda_checkpoint_pages(_raw(), layout), _ids(1),
                              _rows(layout, 1, seed=13)[:1], layout)


def test_pages_of_another_size_are_refused(kda_spec):
    layout = kda_checkpoint_layout(kda_spec)
    pages = torch.zeros(BLOCKS, PAGE_BYTES // 2, dtype=torch.int8)
    with pytest.raises(ValueError, match="131072-byte rows"):
        read_kda_checkpoints(pages, _ids(1), layout)


def test_a_restore_into_banks_of_another_spec_is_refused(kda_spec):
    other = dataclasses.replace(kda_spec, shapes=((2, 8), (1, 4, 4)))
    slot_bytes = recurrent_state_slot_bytes(other)
    banks = state_bank_regions(torch.zeros(SLOTS * slot_bytes, dtype=torch.int8),
                               other.shapes, other.dtypes, slot_bytes=slot_bytes)
    layout = kda_checkpoint_layout(kda_spec)
    with pytest.raises(ValueError, match="carrier 0"):
        restore_kda_slot(banks, 0, kda_checkpoint_pages(_raw(), layout), 1, layout)


def test_the_checkpoint_bytes_follow_from_the_shapes(kda_spec):
    layout = kda_checkpoint_layout(kda_spec)
    assert layout.used_bytes == sum(
        math.prod(shape) * dtype.itemsize
        for shape, dtype in zip(kda_spec.shapes, kda_spec.dtypes, strict=True))
