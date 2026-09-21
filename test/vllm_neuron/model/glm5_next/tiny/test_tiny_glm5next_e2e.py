"""End-to-end tests for the tiny GLM-5.3-Flash model through the Neuron model runner.

``bind_kv_cache``, the runner-built per-layer carriers and side caches, an
eight-token generation against the torch reference, the per-request indexer
ring and its cursor, and scattered KV pages. The model fixture comes from
``test_tiny_glm5next_forward``.
"""

from __future__ import annotations

import inspect
import math
import os
from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as item

pytestmark = [pytest.mark.fast, pytest.mark.forked]

GENERATED_TOKENS = 8


def _blocks_for(slots: int) -> int:
    """The blocks a sequence of ``slots`` slots occupies, the runner's own rounding."""
    return -(-int(slots) // item.MLA_PAGE_SIZE)


PROMPT_BLOCKS = _blocks_for(item.STACK_TOKENS)

E2E_BLOCKS = _blocks_for(item.STACK_TOKENS + GENERATED_TOKENS)

E2E_MAX_SEQ_LEN = item.STACK_TOKENS + GENERATED_TOKENS


def _aligned_blocks(slots: int) -> int:
    """The block-table width the runner's warmup builder reports for ``slots``."""
    alignment = 128 // item.MLA_PAGE_SIZE if item.MLA_PAGE_SIZE <= 128 else 1
    return -(-_blocks_for(slots) // alignment) * alignment


E2E_WINDOW_BLOCKS = _aligned_blocks(E2E_MAX_SEQ_LEN)

E2E_BANK_BLOCKS = E2E_BLOCKS + E2E_WINDOW_BLOCKS

E2E_CONV_STATE_SHAPE = (2, 3)
E2E_RECURRENT_STATE_SHAPE = (2, 4, 4)
E2E_STATE_SLOTS = 8

E2E_MAX_NUM_SEQS = 1

# Two request slots, so a row this request does not own exists to read against.
E2E_SCOPE_MAX_NUM_SEQS = 2


def _fixture(**overrides):
    """The root fixture from the forward module, at two KV heads."""
    return item._root_fixture(num_key_value_heads=2, **overrides)


def _require_cpu_mode() -> None:
    """Both CPU-mode flags must already be set in the process environment."""
    missing = [
        name
        for name in ("VLLM_NEURON_CPU_MODE", "NKI_SIMULATOR")
        if os.environ.get(name) != "1"
    ]
    if missing:
        raise item.VacuousControlError(
            f"{missing} must be 1 in the process environment; the kernels read them "
            f"at import time, so a fixture that set them here would measure the wrong "
            f"backend"
        )


def _runner_shaped_caches(root) -> dict[str, list[torch.Tensor]]:
    """The dict ``initialize_kv_cache`` hands ``bind_kv_cache``, built the runner's own way."""
    caches: dict[str, list[torch.Tensor]] = {}
    for layer_spec in root.get_kv_spec().layers:
        recurrent = (
            layer_spec.kda_conv_state_shape,
            layer_spec.kda_recurrent_state_shape,
        )
        if all(value is not None for value in recurrent):
            caches[layer_spec.name] = [
                torch.zeros(
                    (E2E_STATE_SLOTS, *layer_spec.kda_conv_state_shape),
                    dtype=layer_spec.kda_conv_state_dtype,
                ),
                torch.zeros(
                    (E2E_STATE_SLOTS, *layer_spec.kda_recurrent_state_shape),
                    dtype=layer_spec.kda_recurrent_state_dtype,
                ),
            ]
            continue
        shape = (
            E2E_BANK_BLOCKS,
            int(layer_spec.num_kv_heads),
            item.MLA_PAGE_SIZE,
            int(layer_spec.head_size),
        )
        banks = 1 if layer_spec.latent_kv else 2
        caches[layer_spec.name] = [
            torch.zeros(shape, dtype=layer_spec.dtype) for _ in range(banks)
        ]
    return caches


def _assert_config_matches_index_dials(text_config) -> None:
    """The built config must carry the two indexer dials the fixtures were built from."""
    pairs = {
        "index_kpool": (int(text_config.index_kpool), item.MLA_INDEX_KPOOL),
        "index_head_dim": (int(text_config.index_head_dim), item.MLA_INDEX_HEAD_DIM),
    }
    wrong = {name: values for name, values in pairs.items() if values[0] != values[1]}
    if wrong:
        raise item.VacuousControlError(
            f"the built config and this file's indexer dials disagree on {wrong} "
            f"(config value first), so the operands here are written for another geometry"
        )


def _recurrent_spec(root) -> KVSpec:
    """The stack's own spec with every layer reporting recurrent geometry instead of a pair."""
    return KVSpec(
        layers=[
            LayerSpec(
                name=layer_spec.name,
                num_kv_heads=layer_spec.num_kv_heads,
                head_size=layer_spec.head_size,
                dtype=layer_spec.dtype,
                sliding_window_size=None,
                chunk_size=None,
                kda_conv_state_shape=E2E_CONV_STATE_SHAPE,
                kda_recurrent_state_shape=E2E_RECURRENT_STATE_SHAPE,
                kda_conv_state_dtype=torch.float32,
                kda_recurrent_state_dtype=torch.float32,
            )
            for layer_spec in root.get_kv_spec().layers
        ]
    )


def _geometries(banks, *, block_ids, state_slot: int, page_size: int | None = None,
                window_blocks: int | None = None):
    """One geometry per bank, the shape the carrier builder pairs positionally with its banks."""
    page = item.MLA_PAGE_SIZE if page_size is None else page_size
    ids = [int(value) for value in block_ids]
    return [
        {
            "block_ids": ids,
            "state_slot": int(state_slot),
            "page_size": int(page),
            "window_blocks": len(ids) if window_blocks is None else int(window_blocks),
        }
        for _ in banks
    ]


def _mixed_spec(root) -> KVSpec:
    """The stack's own spec with alternate layers reporting recurrent geometry instead of a pair."""
    layers = []
    for index, layer_spec in enumerate(root.get_kv_spec().layers):
        if index % 2 == 0:
            layers.append(layer_spec)
            continue
        layers.append(
            LayerSpec(
                name=layer_spec.name,
                num_kv_heads=layer_spec.num_kv_heads,
                head_size=layer_spec.head_size,
                dtype=layer_spec.dtype,
                sliding_window_size=None,
                chunk_size=None,
                kda_conv_state_shape=E2E_CONV_STATE_SHAPE,
                kda_recurrent_state_shape=E2E_RECURRENT_STATE_SHAPE,
                kda_conv_state_dtype=torch.float32,
                kda_recurrent_state_dtype=torch.float32,
            )
        )
    return KVSpec(layers=layers)


def test_bind_kv_cache_maps_every_sparse_layer_onto_its_own_slots():
    """Every layer the spec reports gets its own bank, and the bank is the runner's storage."""
    _require_cpu_mode()
    root = _fixture()["root"]
    caches = _runner_shaped_caches(root)
    spec_layers = root.get_kv_spec().layers

    assert not hasattr(root, "glm5next_layer_banks"), (
        "the stack already carries glm5next_layer_banks before anything bound it, so the "
        "test cannot tell the method from the fixture"
    )
    root.bind_kv_cache(caches)
    banks = root.glm5next_layer_banks

    assert len(banks) == len(spec_layers) == item.STACK_LAYERS
    assert [bank["name"] for bank in banks] == [s.name for s in spec_layers]
    assert [bank["layer_index"] for bank in banks] == list(range(len(spec_layers)))
    assert {bank["family"] for bank in banks} == {"self_attn"}

    pointers = set()
    for bank, layer_spec in zip(banks, spec_layers):
        allocated = caches[layer_spec.name][0]
        view = bank["latent_cache"]
        assert tuple(view.shape) == (
            E2E_BANK_BLOCKS * item.MLA_PAGE_SIZE,
            1,
            int(layer_spec.head_size),
        )
        assert int(bank["slots"]) == E2E_BANK_BLOCKS * item.MLA_PAGE_SIZE
        assert view.data_ptr() == allocated.data_ptr(), (
            f"{bank['name']}'s latent view does not start at the runner's own storage, so "
            f"the layers would write a copy the runner never reads"
        )
        assert allocated.data_ptr() not in pointers, (
            f"{bank['name']} shares storage with an earlier layer; two layers writing one "
            f"latent cache is not detected anywhere below this point"
        )
        pointers.add(allocated.data_ptr())

        view[0, 0, 0] = 1.5
        assert float(allocated.reshape(-1)[0]) == 1.5, (
            f"a write through {bank['name']}'s view did not land in the runner's tensor"
        )
        view[0, 0, 0] = 0.0


def test_bind_kv_cache_maps_recurrent_layers_by_the_fields_the_spec_carries(monkeypatch):
    """A layer that reports ``kda_*`` geometry gets its two state banks, paired positionally."""
    _require_cpu_mode()
    root = _fixture()["root"]
    spec = _recurrent_spec(root)
    monkeypatch.setattr(root, "get_kv_spec", lambda: spec)
    caches = _runner_shaped_caches(root)

    root.bind_kv_cache(caches)
    banks = root.glm5next_layer_banks
    assert {bank["family"] for bank in banks} == {"linear_attn"}
    for bank, layer_spec in zip(banks, spec.layers):
        conv, recurrent = caches[layer_spec.name]
        assert bank["conv_state"] is conv
        assert bank["recurrent_state"] is recurrent
        assert int(bank["state_slots"]) == E2E_STATE_SLOTS
        assert tuple(bank["conv_state"].shape[1:]) == E2E_CONV_STATE_SHAPE
        assert tuple(bank["recurrent_state"].shape[1:]) == E2E_RECURRENT_STATE_SHAPE


def test_bind_kv_cache_refuses_a_missing_layer_by_name():
    """A layer the dict does not hold is refused by name."""
    _require_cpu_mode()
    root = _fixture()["root"]
    caches = _runner_shaped_caches(root)
    absent = root.get_kv_spec().layers[0].name
    caches.pop(absent)
    with pytest.raises(ValueError, match=f"no entry for KV layer '{absent}'"):
        root.bind_kv_cache(caches)


def test_bind_kv_cache_refuses_a_bank_of_the_wrong_rank():
    """A bank that is not the runner's four-axis allocation is refused rather than viewed."""
    _require_cpu_mode()
    root = _fixture()["root"]
    caches = _runner_shaped_caches(root)
    name = root.get_kv_spec().layers[0].name
    flat = caches[name][0]
    caches[name] = [flat.reshape(-1), *caches[name][1:]]
    with pytest.raises(ValueError, match="the runner allocates"):
        root.bind_kv_cache(caches)


def test_bind_kv_cache_refuses_a_bank_whose_geometry_is_not_the_specs():
    """A head count or width the model did not ask for is refused, naming both numbers."""
    _require_cpu_mode()
    root = _fixture()["root"]
    caches = _runner_shaped_caches(root)
    layer_spec = root.get_kv_spec().layers[0]
    caches[layer_spec.name] = [
        torch.zeros(
            (E2E_BANK_BLOCKS, 2, item.MLA_PAGE_SIZE, int(layer_spec.head_size)),
            dtype=layer_spec.dtype,
        ),
        *caches[layer_spec.name][1:],
    ]
    with pytest.raises(ValueError, match="head count and width"):
        root.bind_kv_cache(caches)


def test_bind_kv_cache_refuses_a_partial_recurrent_geometry(monkeypatch):
    """Half a recurrent declaration is refused: the two states are paired positionally."""
    _require_cpu_mode()
    root = _fixture()["root"]
    full = _recurrent_spec(root)
    partial = KVSpec(
        layers=[
            LayerSpec(
                name=layer_spec.name,
                num_kv_heads=layer_spec.num_kv_heads,
                head_size=layer_spec.head_size,
                dtype=layer_spec.dtype,
                sliding_window_size=None,
                chunk_size=None,
                kda_conv_state_shape=E2E_CONV_STATE_SHAPE,
                kda_recurrent_state_shape=None,
                kda_conv_state_dtype=torch.float32,
                kda_recurrent_state_dtype=torch.float32,
            )
            for layer_spec in full.layers
        ]
    )
    monkeypatch.setattr(root, "get_kv_spec", lambda: partial)
    caches = {
        layer_spec.name: [
            torch.zeros((E2E_STATE_SLOTS, *E2E_CONV_STATE_SHAPE)),
            torch.zeros((E2E_STATE_SLOTS, *E2E_RECURRENT_STATE_SHAPE)),
        ]
        for layer_spec in partial.layers
    }
    with pytest.raises(ValueError, match="part of its recurrent geometry"):
        root.bind_kv_cache(caches)


def test_bind_kv_cache_refuses_a_spec_that_disagrees_with_the_stack(monkeypatch):
    """A spec shorter than the stack is refused: the carriers are paired positionally."""
    _require_cpu_mode()
    root = _fixture()["root"]
    short = KVSpec(layers=root.get_kv_spec().layers[:-1])
    caches = _runner_shaped_caches(root)
    monkeypatch.setattr(root, "get_kv_spec", lambda: short)
    with pytest.raises(ValueError, match="the stack holds"):
        root.bind_kv_cache(caches)


def test_runner_derived_operands_equal_the_fixtures_prefill_operands():
    """The runner's ``slot_mapping`` and ``seq_lens`` equal the forward module's, value for value."""
    _require_cpu_mode()
    selection = item._mla_selection_operands(
        tokens=item.STACK_TOKENS, pages=item.STACK_PAGES
    )
    device = torch.device("cpu")
    slots = NeuronModelRunner._glm5next_pool_slot_mapping(
        tokens=item.STACK_TOKENS,
        start_position=0,
        index_kpool=item.MLA_INDEX_KPOOL,
        device=device,
    )
    lengths = NeuronModelRunner._glm5next_row_seq_lens(
        tokens=item.STACK_TOKENS, start_position=0, device=device
    )
    assert slots.dtype == selection["slot_mapping"].dtype
    assert torch.equal(slots, selection["slot_mapping"])
    assert lengths.dtype == selection["seq_lens"].dtype
    assert torch.equal(lengths, selection["seq_lens"])

    shifted_slots = NeuronModelRunner._glm5next_pool_slot_mapping(
        tokens=item.STACK_TOKENS,
        start_position=1,
        index_kpool=item.MLA_INDEX_KPOOL,
        device=device,
    )
    shifted_lengths = NeuronModelRunner._glm5next_row_seq_lens(
        tokens=item.STACK_TOKENS, start_position=1, device=device
    )
    assert not torch.equal(shifted_slots, selection["slot_mapping"]), (
        "the pool derivation ignores start_position, so a second prefill chunk would pool "
        "on the chunk's boundaries instead of the sequence's"
    )
    assert not torch.equal(shifted_lengths, selection["seq_lens"]), (
        "the causal-length derivation ignores start_position, so a second prefill chunk "
        "would bound every row at the chunk's own length"
    )


def test_side_caches_meet_the_indexers_own_stated_minimum():
    """The pooled store leaves one trash row above every addressable pool, and the ring is sized."""
    _require_cpu_mode()
    root = _fixture()["root"]
    caches = _runner_shaped_caches(root)
    root.bind_kv_cache(caches)
    banks = root.glm5next_layer_banks
    text_config = root.text_config
    _assert_config_matches_index_dials(text_config)
    side = NeuronModelRunner._glm5next_side_caches(
        banks,
        index_kpool=int(text_config.index_kpool),
        index_head_dim=int(text_config.index_head_dim),
        max_seq_len=item.STACK_TOKENS,
        request_slots=E2E_STATE_SLOTS,
    )
    candidates = item.STACK_TOKENS // int(text_config.index_kpool)
    assert len(side) == len(banks)
    for entry, bank in zip(side, banks):
        assert int(entry["pool_cache"].shape[0]) == E2E_STATE_SLOTS
        assert int(entry["pool_cache"].shape[1]) >= candidates + 1
        assert int(entry["pool_cache"].shape[2]) == int(text_config.index_head_dim)
        assert tuple(entry["tail"].shape) == (
            E2E_STATE_SLOTS,
            2,
            int(text_config.index_kpool),
            int(text_config.index_head_dim),
        )
        assert entry["pool_cache"].dtype == bank["latent_cache"].dtype
    assert len({entry["pool_cache"].data_ptr() for entry in side}) == len(side), (
        "two layers share one pooled-key store, so the second would read the first's pools"
    )


def test_runner_built_carriers_drive_the_root_and_write_the_runners_own_cache():
    """One prefill through carriers the runner built, ending in the runner's tensors changing."""
    _require_cpu_mode()
    fixture = _fixture()
    root, layers = fixture["root"], fixture["layers"]
    caches = _runner_shaped_caches(root)
    root.bind_kv_cache(caches)
    banks = root.glm5next_layer_banks
    text_config = root.text_config
    _assert_config_matches_index_dials(text_config)

    side = NeuronModelRunner._glm5next_side_caches(
        banks,
        index_kpool=int(text_config.index_kpool),
        index_head_dim=int(text_config.index_head_dim),
        max_seq_len=item.STACK_TOKENS,
        request_slots=E2E_STATE_SLOTS,
    )
    carriers = NeuronModelRunner._glm5next_layer_carriers(
        banks,
        side,
        geometries=_geometries(banks, block_ids=range(PROMPT_BLOCKS), state_slot=0),
        is_prefill=True,
        tokens=item.STACK_TOKENS,
        start_position=0,
        softmax_scale=item.MLA_SOFTMAX_SCALE,
        max_seq_len=item.STACK_TOKENS,
        index_kpool=int(text_config.index_kpool),
    )

    reference = item._stack_carriers(
        layers,
        item._mla_selection_operands(
            tokens=item.STACK_TOKENS, pages=item.STACK_PAGES
        ),
    )
    assert len(carriers) == len(reference) == item.STACK_LAYERS
    for index, (got, expected) in enumerate(zip(carriers, reference)):
        assert set(got) - set(expected) == {"prefill_tail", "prefill_end_position"}, (
            f"layer {index}'s runner-built carrier carries {sorted(set(got))} and the "
            f"reference prefill carrier carries {sorted(set(expected))}; the runner adds "
            f"only the ring and its end position to the prefill leg"
        )
        assert set(expected) - set(got) == set(), (
            f"layer {index}'s runner-built carrier lost {sorted(set(expected) - set(got))}, "
            f"which the reference prefill carrier declares"
        )
        assert tuple(got["latent_cache"].shape[1:]) == tuple(
            expected["latent_cache"].shape[1:]), (
            f"layer {index}'s runner-built cache has rows shaped "
            f"{tuple(got['latent_cache'].shape[1:])} where the reference carrier's rows are "
            f"{tuple(expected['latent_cache'].shape[1:])}"
        )
        assert int(got["latent_cache"].shape[0]) == E2E_BANK_BLOCKS * item.MLA_PAGE_SIZE, (
            f"layer {index}'s runner-built cache holds {int(got['latent_cache'].shape[0])} "
            f"row(s) where this file's bank holds "
            f"{E2E_BANK_BLOCKS * item.MLA_PAGE_SIZE}; the carrier hands the whole bank"
        )
        position = got["start_position"]
        assert int(position.reshape(-1)[0]) == 0
        assert float(got["softmax_scale"]) == float(item.MLA_SOFTMAX_SCALE)

    input_ids = torch.randint(
        0,
        item.STACK_VOCAB_SIZE,
        (item.STACK_TOKENS,),
        generator=torch.Generator().manual_seed(item.SEED_STACK_IDS),
        dtype=torch.int64,
    )
    positions = torch.tensor(item.ROOT_SAMPLING_POSITIONS, dtype=torch.long)

    before_banks = [caches[bank["name"]][0].clone() for bank in banks]
    before_pools = [entry["pool_cache"].clone() for entry in side]

    logits = root.forward(
        input_ids, layer_carriers=carriers, sampling_positions=positions
    )

    assert logits.shape[0] == len(item.ROOT_SAMPLING_POSITIONS)
    assert logits.shape[-1] == item.STACK_VOCAB_SIZE
    assert torch.isfinite(logits.to(torch.float32)).all()

    for bank, before in zip(banks, before_banks):
        allocated = caches[bank["name"]][0]
        assert not torch.equal(allocated, before), (
            f"the forward left the runner's own latent bank for {bank['name']} unchanged, "
            f"so the layer wrote something that is not the runner's cache"
        )
    for index, (entry, before) in enumerate(zip(side, before_pools)):
        assert not torch.equal(entry["pool_cache"], before), (
            f"layer {index}'s pooled-key store is unchanged after a {item.STACK_TOKENS}-token "
            f"prefill, so the indexer wrote no pool through this carrier"
        )


def test_the_tiny_config_stays_inside_the_cpu_run_bounds():
    """The tiny config's size bounds, read off the config that ran."""
    _require_cpu_mode()
    fixture = _fixture()
    root, cfg = fixture["root"], fixture["cfg"]
    missing = [
        name
        for name in ("hidden_size", "intermediate_size", "num_key_value_heads",
                     "qk_nope_head_dim", "v_head_dim", "index_head_dim")
        if not hasattr(cfg, name)
    ]
    if missing:
        raise item.VacuousControlError(
            f"the built config declares none of {missing}, so the bounds cannot be read "
            f"off the config that ran"
        )
    parameters = sum(int(p.numel()) for p in root.parameters() if p is not None)
    head_dims = {
        name: int(getattr(cfg, name))
        for name in ("qk_nope_head_dim", "v_head_dim", "index_head_dim")
    }
    assert parameters < 32_000_000
    assert int(cfg.hidden_size) % 256 == 0
    assert int(cfg.intermediate_size) >= 512
    assert max(head_dims.values()) <= 128
    assert int(cfg.num_key_value_heads) == 2


def _slot_position(runner, slot=None):
    """The ring cursor for one request slot; ``req-0``'s slot when ``slot`` is None."""
    positions = getattr(runner, "_glm5next_side_cache_positions", None) or {}
    if slot is None:
        table = getattr(runner, "_glm5next_request_slot_table", None) or {}
        slot = table.get("req-0")
    return None if slot is None else positions.get(int(slot))


def _own_slot(runner, key: str = "req-0") -> int:
    """The state slot the runner assigned to request ``key``."""
    table = getattr(runner, "_glm5next_request_slot_table", None) or {}
    if key not in table:
        raise item.VacuousControlError(
            f"no state slot is assigned to {key!r} yet, so a per-request row cannot "
            f"be read; a real step must claim the slot before this is called"
        )
    return int(table[key])


def _entry(*, row, tokens: int, cached: int, threshold: int, block_size: int) -> dict:
    """One KV-cache group's attention-metadata entry, at this step's geometry."""
    table = torch.tensor([[int(value) for value in row]], dtype=torch.int32)
    return {
        "block_table_tensor": table,
        "full_block_table_tensor": table,
        "slot_mapping": torch.arange(tokens, dtype=torch.int32) + cached,
        "max_query_len": int(tokens),
        "block_size": int(block_size),
        "max_blocks_per_seq": int(table.shape[1]),
        "decode_token_threshold": int(threshold),
        "cached_seq_len": torch.tensor([cached], dtype=torch.int32),
        "host_block_table": table,
        "host_num_computed_tokens": [int(cached)],
        "kv_segment_size": int(table.shape[1]) * int(block_size),
    }


def _metadata(names, *, blocks: int, tokens: int, cached: int, threshold: int = 1) -> dict:
    """The runner's attention-metadata mapping: one entry per layer name, one group's geometry."""
    entry = _entry(row=range(blocks), tokens=tokens, cached=cached, threshold=threshold,
                   block_size=item.MLA_PAGE_SIZE)
    return {str(name): entry for name in names}


def _model_kwargs(runner, *, input_ids, cached: int, sampling_row: int) -> dict:
    """One step's generic runner kwargs, translated by the converter under test."""
    tokens = int(input_ids.shape[0])
    blocks = _blocks_for(cached + tokens)
    generic = {
        "input_ids": input_ids,
        "positions": torch.arange(tokens, dtype=torch.long) + cached,
        "attn_metadata": _metadata(
            [bank["name"] for bank in runner.model.glm5next_layer_banks],
            blocks=blocks, tokens=tokens, cached=cached,
        ),
        "sampling_positions": torch.tensor([sampling_row], dtype=torch.long),
        "sampling_params": None,
        "spec_decode_metadata": None,
        "rank": None,
        "logit_mask": None,
    }
    return runner._glm5next_model_kwargs(generic)


def _reference_attention_half(layer, raw, gains, hidden, cfg, *, tokens: int):
    """The forward module's attention-half reference at a given token count."""
    selection = item._mla_selection_operands(tokens=tokens, pages=item.STACK_PAGES)
    normed = item._ffn_norm(
        hidden, layer.input_layernorm_weight, float(cfg.rms_norm_eps)
    )
    attention = layer.self_attn
    q_latent = attention.project_query_latent(normed)
    topk_indices = attention.indexer(
        normed,
        q_latent,
        item._mla_pool_cache(pages=item.STACK_PAGES),
        selection["seq_lens"],
        max_seq_len=tokens,
        page_size=item.MLA_PAGE_SIZE,
        slot_mapping=selection["slot_mapping"],
    )
    attended = item._mla_dense_reference(
        attention,
        raw,
        gains,
        normed.float(),
        item._mla_latent_cache(attention, tokens=tokens),
        topk_indices,
        softmax_scale=item.MLA_SOFTMAX_SCALE,
    )
    return attended


def _reference_logits(fixture, token_ids) -> torch.Tensor:
    """The torch reference model's logits for the last row of this token sequence."""
    cfg, layers = fixture["cfg"], fixture["layers"]
    tokens = int(token_ids.shape[0])
    hidden = fixture["table"][token_ids]
    for index, layer in enumerate(layers):
        raw, gains = fixture["attention_operands"][index]
        attended = _reference_attention_half(
            layer, raw, gains, hidden, cfg, tokens=tokens
        )
        hidden = hidden.float() + attended.float()
        half = item._stack_ffn_half(
            layer,
            hidden,
            cfg,
            fixture["mlp_operands"][index],
            routed=index in fixture["moe_at"],
        )
        hidden = hidden.float() + half["out"].float()
    final = item._ffn_norm(hidden, fixture["final_gain"], float(cfg.rms_norm_eps))
    return item._root_reference(final, fixture["head"], [tokens - 1])


def _assert_route_predicate_r3(label: str, before: dict, after: dict) -> None:
    """The kernel route was live, no family fell back to torch, and at least one kernel fired."""
    from vllm_neuron.utils.neuron_utils import can_run_kernel

    item._assert_every_counter_family_is_registered()
    item._assert_no_unregistered_counter_family()
    runnable = bool(can_run_kernel())
    fired = sorted(
        name
        for name, (dispatch, _fallback) in after.items()
        if dispatch - before[name][0] > 0
    )
    fallbacks = {
        name: after[name][1] - before[name][1]
        for name in after
        if after[name][1] - before[name][1]
    }
    assert runnable, (
        f"{label}: can_run_kernel() is False, so this run took torch and nothing below "
        f"measured the NKI kernels"
    )
    assert not fallbacks, (
        f"{label}: the torch-fallback counters moved on {sorted(fallbacks)}; expected 0 "
        f"across every registered family"
    )
    assert fired, (
        f"{label}: no kernel counter moved, so the run completed with no NKI kernel "
        f"dispatched at all"
    )


def test_the_generation_is_eight_tokens_and_every_step_matches_the_reference():
    """Eight tokens through the runner's converter, each step matching the torch reference."""
    _require_cpu_mode()
    fixture = _fixture()
    root = fixture["root"]
    _assert_config_matches_index_dials(root.text_config)
    caches = _runner_shaped_caches(root)
    root.bind_kv_cache(caches)

    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["req-0"])
    runner.model = root
    runner.max_model_len = E2E_MAX_SEQ_LEN
    runner.max_num_reqs = E2E_MAX_NUM_SEQS

    prompt = torch.randint(
        0,
        item.STACK_VOCAB_SIZE,
        (item.STACK_TOKENS,),
        generator=torch.Generator().manual_seed(item.SEED_STACK_IDS),
        dtype=torch.int64,
    )

    produced: list[torch.Tensor] = []
    generated: list[int] = []
    steps: list[tuple[str, int, torch.Tensor]] = []

    item._reset_seam_counters()
    before = item._read_seam_counters()

    converted = _model_kwargs(
        runner, input_ids=prompt, cached=0, sampling_row=item.STACK_TOKENS - 1
    )
    assert sorted(converted) == [
        "expert_parallel_rank", "input_ids", "layer_carriers", "moe_group",
        "sampling_positions", "tp_degree",
    ], f"the converter handed the model {sorted(converted)}"
    assert "slot_mapping" in converted["layer_carriers"][0], (
        "the prompt step built a decode carrier; the prefill leg is the one that carries "
        "slot_mapping"
    )
    logits = root.forward(**converted)
    produced.append(logits[-1].float())
    steps.append(("prefill", int(prompt.shape[0]), prompt.clone()))
    generated.append(int(logits[-1].argmax()))

    for step in range(GENERATED_TOKENS - 1):
        cached = item.STACK_TOKENS + step
        fed = torch.tensor([generated[-1]], dtype=torch.int64)
        converted = _model_kwargs(runner, input_ids=fed, cached=cached, sampling_row=0)
        assert "tail" in converted["layer_carriers"][0], (
            f"decode step {step} built a prefill carrier; the decode leg is the one that "
            f"carries tail and position"
        )
        assert int(converted["layer_carriers"][0]["position"]) == cached
        logits = root.forward(**converted)
        produced.append(logits[-1].float())
        sequence = torch.cat([prompt, torch.tensor(generated, dtype=torch.int64)])
        steps.append((f"decode{step}", int(sequence.shape[0]), sequence))
        generated.append(int(logits[-1].argmax()))

    after = item._read_seam_counters()

    assert len(generated) == GENERATED_TOKENS, (
        f"the generation appended {len(generated)} token(s); expected {GENERATED_TOKENS}"
    )
    assert len(produced) == GENERATED_TOKENS

    _assert_route_predicate_r3("the 8-token generation", before, after)

    for (label, length, sequence), got in zip(steps, produced):
        expected = _reference_logits(fixture, sequence[:length])[0].float()
        assert torch.isfinite(got).all(), f"{label} produced a non-finite logit"
        torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-5)

    written = int((caches[root.glm5next_layer_banks[0]["name"]][0] != 0).sum())
    assert written > 0


def _grouped_metadata(banks, *, tokens: int, sparse_row, state_row,
                      sparse_block_size: int | None = None, sparse_cached: int = 0,
                      state_cached: int = 0, threshold: int = 1) -> dict:
    """Two KV-cache groups, each with its own block table, keyed by every layer name."""
    sparse = _entry(
        row=sparse_row, tokens=tokens, cached=sparse_cached, threshold=threshold,
        block_size=item.MLA_PAGE_SIZE if sparse_block_size is None else sparse_block_size,
    )
    state = _entry(row=state_row, tokens=tokens, cached=state_cached, threshold=threshold,
                   block_size=item.MLA_PAGE_SIZE)
    return {
        bank["name"]: (sparse if bank["family"] == "self_attn" else state)
        for bank in banks
    }


def _generic(*, tokens: int, metadata: dict, sampling_row: int) -> dict:
    """One step's generic runner kwargs at a given metadata mapping."""
    return {
        "input_ids": torch.zeros(int(tokens), dtype=torch.long),
        "positions": torch.arange(int(tokens), dtype=torch.long),
        "attn_metadata": metadata,
        "sampling_positions": torch.tensor([int(sampling_row)], dtype=torch.long),
        "sampling_params": None,
        "spec_decode_metadata": None,
        "rank": None,
        "logit_mask": None,
    }


def test_the_converter_reads_each_layers_own_kv_cache_group(monkeypatch):
    """A hybrid stack has two block tables, and every layer's carrier comes from its own."""
    _require_cpu_mode()
    root = _fixture()["root"]
    spec = _mixed_spec(root)
    monkeypatch.setattr(root, "get_kv_spec", lambda: spec)
    caches = _runner_shaped_caches(root)
    root.bind_kv_cache(caches)
    banks = root.glm5next_layer_banks
    families = sorted({bank["family"] for bank in banks})
    if families != ["linear_attn", "self_attn"]:
        raise item.VacuousControlError(
            f"the test needs both families in one stack for two groups to exist at all, "
            f"and the spec it built reports {families}"
        )

    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["req-0"])
    runner.model = root
    runner.max_model_len = E2E_MAX_SEQ_LEN
    runner.max_num_reqs = E2E_MAX_NUM_SEQS
    tokens = GENERATED_TOKENS
    state_slot = E2E_STATE_SLOTS - 1
    sparse_row = [0, 1]
    grouped = _grouped_metadata(banks, tokens=tokens, sparse_row=sparse_row,
                                state_row=[state_slot])
    translated = runner._glm5next_model_kwargs(
        _generic(tokens=tokens, metadata=grouped, sampling_row=tokens - 1)
    )
    carriers = translated["layer_carriers"]
    assert len(carriers) == len(banks)
    for index, (bank, carrier) in enumerate(zip(banks, carriers)):
        if bank["family"] == "self_attn":
            page = int(bank["block_size"])
            assert carrier["latent_cache"].data_ptr() == bank["latent_cache"].data_ptr(), (
                f"layer {index} is sparse and its carrier is not a view of its own "
                f"group's bank"
            )
            assert int(carrier["latent_cache"].shape[0]) == int(
                bank["latent_cache"].shape[0]
            )
            assert carrier["block_table_row"].flatten().tolist() == list(sparse_row), (
                f"layer {index} is sparse and its block table does not name its own "
                f"group's blocks"
            )
            assert carrier["latent_slots"].tolist()[0] == sparse_row[0] * page
        else:
            assigned = runner._glm5next_request_slot_table["req-0"]
            assert (carrier["conv_state"][0].data_ptr()
                    == bank["conv_state"][assigned].data_ptr()), (
                f"layer {index} is recurrent and its conv state is not the slot the "
                f"runner's request table assigned"
            )
            assert (carrier["recurrent_state"][0].data_ptr()
                    == bank["recurrent_state"][assigned].data_ptr())

    short = dict(grouped)
    short.pop(banks[0]["name"])
    with pytest.raises(ValueError, match="has no attention-metadata entry"):
        runner._glm5next_model_kwargs(
            _generic(tokens=tokens, metadata=short, sampling_row=tokens - 1)
        )

    wrong_page = _grouped_metadata(banks, tokens=tokens, sparse_row=sparse_row,
                                   state_row=[state_slot],
                                   sparse_block_size=int(item.MLA_PAGE_SIZE) * 2)
    with pytest.raises(ValueError, match="reports page"):
        runner._glm5next_model_kwargs(
            _generic(tokens=tokens, metadata=wrong_page, sampling_row=tokens - 1)
        )

    disagreeing = _grouped_metadata(banks, tokens=tokens, sparse_row=sparse_row,
                                    state_row=[state_slot], state_cached=1)
    with pytest.raises(ValueError, match="must agree on the leg and the cached length"):
        runner._glm5next_model_kwargs(
            _generic(tokens=tokens, metadata=disagreeing, sampling_row=tokens - 1)
        )

    shared_row = [state_slot + offset for offset in range(_blocks_for(tokens))]
    single = {
        bank["name"]: _entry(row=shared_row, tokens=tokens, cached=0, threshold=1,
                             block_size=item.MLA_PAGE_SIZE)
        for bank in banks
    }
    control = runner._glm5next_model_kwargs(
        _generic(tokens=tokens, metadata=single, sampling_row=tokens - 1)
    )
    sparse_index = next(i for i, bank in enumerate(banks) if bank["family"] == "self_attn")
    per_layer_row = carriers[sparse_index]["block_table_row"].flatten().tolist()
    single_row = control["layer_carriers"][sparse_index]["block_table_row"].flatten().tolist()
    assert per_layer_row != single_row, (
        "the single-table mapping named the same sparse pages as the per-layer one, so "
        "the test is not measuring the per-layer lookup at all"
    )


def test_the_converter_does_not_hand_the_root_a_kv_page_as_its_quant_block():
    """Two different numbers share the name ``block_size``; only the KV page reaches the layers."""
    _require_cpu_mode()
    root = _fixture()["root"]
    caches = _runner_shaped_caches(root)
    root.bind_kv_cache(caches)
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["req-0"])
    runner.model = root
    runner.max_model_len = E2E_MAX_SEQ_LEN
    runner.max_num_reqs = E2E_MAX_NUM_SEQS

    translated = _model_kwargs(
        runner,
        input_ids=torch.zeros(item.STACK_TOKENS, dtype=torch.long),
        cached=0,
        sampling_row=item.STACK_TOKENS - 1,
    )
    declared = inspect.signature(type(root).forward).parameters
    assert set(translated) <= set(declared), (
        f"the converter returned {sorted(set(translated) - set(declared))}, which the root "
        f"forward does not declare"
    )
    assert "block_size" not in translated
    assert declared["block_size"].default is None

    page = int(item.MLA_PAGE_SIZE)
    quant = int(item.BLOCK_QUANT_SIZE)
    if page % quant == 0:
        raise item.VacuousControlError(
            f"the KV page {page} is a multiple of the quant block {quant}, so passing the "
            f"page as the root's block_size would not be refused and this check would be "
            f"vacuous"
        )

    for index, (bank, carrier) in enumerate(
        zip(root.glm5next_layer_banks, translated["layer_carriers"])
    ):
        assert int(carrier["page_size"]) == int(bank["block_size"]), (
            f"layer {index}'s carrier carries page {int(carrier['page_size'])} and its bank "
            f"is paged {int(bank['block_size'])}; the KV page must still reach the layers"
        )


def test_the_side_caches_live_across_steps_and_a_fresh_sequence_clears_the_ring():
    """One set of side caches per process, and a new sequence must not inherit the old ring."""
    _require_cpu_mode()
    root = _fixture()["root"]
    caches = _runner_shaped_caches(root)
    root.bind_kv_cache(caches)
    banks = root.glm5next_layer_banks
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["req-0"])
    runner.model = root
    runner.max_model_len = E2E_MAX_SEQ_LEN
    runner.max_num_reqs = E2E_SCOPE_MAX_NUM_SEQS

    live = runner._glm5next_live_side_caches(banks)
    again = runner._glm5next_live_side_caches(banks)
    rings = [side for side in live if "tail" in side]
    assert live is again, "the side caches must be one set per process, not one per step"
    assert all(a["tail"] is b["tail"] for a, b in zip(rings, [s for s in again if "tail" in s]))
    if not rings:
        raise item.VacuousControlError(
            "no bank in this stack carries a ring, so the test measures nothing"
        )

    _model_kwargs(
        runner,
        input_ids=torch.zeros(item.STACK_TOKENS, dtype=torch.long),
        cached=0,
        sampling_row=item.STACK_TOKENS - 1,
    )
    own = _own_slot(runner)
    above_the_bound = int(rings[0]["pool_cache"].shape[1]) - 1
    candidates = item.STACK_TOKENS // int(root.text_config.index_kpool)
    if above_the_bound <= candidates:
        raise item.VacuousControlError(
            f"row {above_the_bound} is not above this sequence's candidate bound "
            f"{candidates}, so planting there would be reachable and the check below "
            f"would measure the wrong thing"
        )
    for side in rings:
        side["tail"].fill_(3.0)
        side["pool_cache"][:, above_the_bound].fill_(5.0)

    _model_kwargs(
        runner,
        input_ids=torch.zeros(item.STACK_TOKENS, dtype=torch.long),
        cached=0,
        sampling_row=item.STACK_TOKENS - 1,
    )
    cleared = max(float(side["tail"][own].abs().max()) for side in rings)
    stale = min(
        float(side["pool_cache"][own][above_the_bound].abs().max()) for side in rings
    )
    others = [
        float(side["tail"][index].abs().max())
        for side in rings
        for index in range(int(side["tail"].shape[0]))
        if index != own
    ]
    if not others:
        raise item.VacuousControlError(
            f"the test runs at a bound of {E2E_SCOPE_MAX_NUM_SEQS} sequence(s) so that a row "
            f"this request does not own exists to read the clearing's scope against, and the "
            f"ring it was handed carries {int(rings[0]['tail'].shape[0])} row(s)"
        )
    untouched = min(others)
    assert cleared == 0.0, (
        "a prefill at position 0 is a new sequence and must start on an empty ring; this one "
        "inherited the planted state"
    )
    assert untouched == 3.0, (
        "the fresh prefill cleared a ring row this request does not own; the rows are per "
        "request, so clearing wider than one slot is a cross-request write"
    )
    assert stale == 5.0, (
        "the pooled store must not be blanket-cleared: the planted row is above this "
        "sequence's candidate bound and therefore unreachable, and clearing it would hide a "
        "bound defect instead of exposing one"
    )

    for side in rings:
        side["tail"].fill_(7.0)
    _model_kwargs(
        runner,
        input_ids=torch.zeros(1, dtype=torch.long),
        cached=item.STACK_TOKENS,
        sampling_row=0,
    )
    kept = min(float(side["tail"][own].abs().max()) for side in rings)
    assert kept == 7.0, (
        "a decode step must not clear the ring; the ring is the decode leg's own state and "
        "clearing it would lose the partial pool this step is meant to advance"
    )


DSA_TOKENS_PER_CALL = 128
REMAINDER_PROMPT = item.STACK_TOKENS - 2
EVEN_PROMPT = item.STACK_TOKENS


def _the_row_scale_bound(reference: torch.Tensor) -> float:
    """atol plus rtol times the row's largest value."""
    return 1e-5 + 1e-2 * float(reference.abs().max())


def test_the_prefill_remainder_is_seeded_and_the_next_pool_completes_whole():
    """The tokens after a prompt's last complete pool must reach the ring the decode leg pools."""
    _require_cpu_mode()
    root = _fixture()["root"]
    caches = _runner_shaped_caches(root)
    root.bind_kv_cache(caches)
    banks = root.glm5next_layer_banks
    pool = int(root.text_config.index_kpool)
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["req-0"])
    runner.model = root
    runner.max_model_len = E2E_MAX_SEQ_LEN
    runner.max_num_reqs = E2E_MAX_NUM_SEQS

    remainder = REMAINDER_PROMPT % pool
    completed_pool = (EVEN_PROMPT - 1) // pool
    if remainder == 0 or EVEN_PROMPT % pool != 0 or EVEN_PROMPT > E2E_MAX_SEQ_LEN:
        raise item.VacuousControlError(
            f"the test needs a prompt that leaves a remainder ({REMAINDER_PROMPT} % {pool} "
            f"= {remainder}), an extension that divides evenly ({EVEN_PROMPT} % {pool} = "
            f"{EVEN_PROMPT % pool}) and both inside {E2E_MAX_SEQ_LEN} slots"
        )
    if max(REMAINDER_PROMPT, EVEN_PROMPT) > DSA_TOKENS_PER_CALL:
        raise item.VacuousControlError(
            f"the test prefills {REMAINDER_PROMPT} and {EVEN_PROMPT} tokens in one leg, and "
            f"the DSA path takes at most {DSA_TOKENS_PER_CALL} per call; a longer leg raises "
            f"inside the causal-bound kernel instead of measuring the seeding"
        )

    ids = torch.arange(EVEN_PROMPT, dtype=torch.long) % int(root.text_config.vocab_size)
    side = runner._glm5next_live_side_caches(banks)
    rings = [entry for entry in side if "tail" in entry]
    if not rings:
        raise item.VacuousControlError("no bank in this stack carries a ring")

    _model_kwargs(runner, input_ids=ids[:REMAINDER_PROMPT], cached=0,
                  sampling_row=REMAINDER_PROMPT - 1)
    own = _own_slot(runner)

    def _rows() -> list[torch.Tensor]:
        """The completed pool's row in each ring, for this request's slot."""
        return [entry["pool_cache"][own][completed_pool] for entry in rings]

    def _clear_the_row() -> None:
        for row in _rows():
            row.zero_()

    _clear_the_row()
    root(**_model_kwargs(runner, input_ids=ids, cached=0, sampling_row=EVEN_PROMPT - 1))
    expected_rows = [row.clone().float() for row in _rows()]
    reference_scale = max(float(row.abs().max()) for row in expected_rows)
    if reference_scale == 0.0:
        raise item.VacuousControlError(
            "the all-prefill route wrote nothing into the pool row the test compares, so "
            "every comparison below would pass against zeros"
        )

    _clear_the_row()
    root(**_model_kwargs(runner, input_ids=ids[:REMAINDER_PROMPT], cached=0,
                         sampling_row=REMAINDER_PROMPT - 1))
    low = min(float(entry["tail"][own][0, :remainder].abs().max()) for entry in rings)
    high = max(float(entry["tail"][own][0, remainder:].abs().max()) for entry in rings)
    assert low > 0.0, (
        "the prefill left the ring's remainder slots empty, so the next pool would complete "
        "from zeros in their place"
    )
    assert high == 0.0, (
        "the prefill wrote ring slots belonging to positions it never saw, which would put "
        "the wrong keys in the next completion"
    )
    for step in range(remainder):
        position = REMAINDER_PROMPT + step
        root(**_model_kwargs(runner, input_ids=ids[position:position + 1], cached=position,
                             sampling_row=0))
    for row, reference in zip(_rows(), expected_rows):
        delta = float((row.float() - reference).abs().max())
        bound = _the_row_scale_bound(reference)
        assert delta <= bound, (
            "the pool the decode leg completed misses the all-prefill reference by more than "
            "the tolerance allows at this row's scale, which is a wrong pool rather than the "
            "one extra bf16 round trip the decode kernel faithfully performs"
        )

    # With the ring emptied after the prefill, the completed pool must miss the
    # reference by at least the share of the pool the emptied ring removes.
    _clear_the_row()
    root(**_model_kwargs(runner, input_ids=ids[:REMAINDER_PROMPT], cached=0,
                         sampling_row=REMAINDER_PROMPT - 1))
    for entry in rings:
        entry["tail"][own].zero_()
    for step in range(remainder):
        position = REMAINDER_PROMPT + step
        root(**_model_kwargs(runner, input_ids=ids[position:position + 1], cached=position,
                             sampling_row=0))
    for row, reference in zip(_rows(), expected_rows):
        delta = float((row.float() - reference).abs().max())
        bound = _the_row_scale_bound(reference)
        over_by = (delta / bound) if bound > 0.0 else float("inf")
        share = remainder / pool
        floor = share * float(reference.abs().max()) / math.sqrt(reference.numel()) / bound
        assert delta > bound, (
            "with the ring emptied after the prefill, the completed pool still matched the "
            "all-prefill reference inside tolerance, so the test is not measuring the "
            "seeding at all"
        )
        assert over_by >= floor, (
            "with the ring emptied the completed pool did miss the reference, but by less than "
            "the share of the pool an emptied ring removes can account for, so the check no "
            "longer separates a missing pool member from a rounding"
        )


MULTI_TOKEN_DECODE = 3
STALE_DECODE_GAP = 3
PLANTED_RING = 9.0


def _sparse_indexer(layers):
    """The first layer's DSA indexer."""
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        indexer = getattr(attention, "indexer", None)
        if indexer is not None:
            return indexer
    raise item.VacuousControlError(
        "this fixture exposes no DSA indexer, so the two prefill_tail refusals cannot be "
        "reached and the checks below would pass without measuring anything"
    )


def _live_rings(runner, banks):
    return [side for side in runner._glm5next_live_side_caches(banks) if "tail" in side]


def test_the_decode_leg_refuses_a_step_carrying_more_than_one_token():
    """A decode step with more than one token is refused by name."""
    _require_cpu_mode()
    fixture = _fixture()
    root = fixture["root"]
    root.bind_kv_cache(_runner_shaped_caches(root))
    banks = root.glm5next_layer_banks
    text_config = root.text_config
    if MULTI_TOKEN_DECODE <= 1:
        raise item.VacuousControlError(
            f"this check needs a decode carrying more than one token; it carries "
            f"{MULTI_TOKEN_DECODE}, which is what the refusal allows"
        )
    side = NeuronModelRunner._glm5next_side_caches(
        banks,
        index_kpool=int(text_config.index_kpool),
        index_head_dim=int(text_config.index_head_dim),
        max_seq_len=item.STACK_TOKENS,
        request_slots=E2E_STATE_SLOTS,
    )

    def build(tokens: int):
        return NeuronModelRunner._glm5next_layer_carriers(
            banks,
            side,
            geometries=_geometries(banks, block_ids=range(PROMPT_BLOCKS), state_slot=0),
            is_prefill=False,
            tokens=tokens,
            start_position=item.STACK_TOKENS,
            softmax_scale=item.MLA_SOFTMAX_SCALE,
            max_seq_len=item.STACK_TOKENS + tokens,
            index_kpool=int(text_config.index_kpool),
        )

    with pytest.raises(ValueError) as caught:
        build(MULTI_TOKEN_DECODE)
    message = str(caught.value)
    assert "threading a multi-token decode" in message, (
        f"a {MULTI_TOKEN_DECODE}-token decode raised, but not the refusal this check names: "
        f"{message}"
    )
    assert f"{MULTI_TOKEN_DECODE} token(s)" in message, (
        "the refusal does not report the count it actually saw, so a reader cannot tell "
        "which step was refused"
    )

    control = None
    try:
        build(1)
    except Exception as exc:
        control = f"{type(exc).__name__}: {exc}"
    if control is not None and "threading a multi-token decode" in control:
        raise item.VacuousControlError(
            "a single-token decode raised the multi-token refusal too, so this check is not "
            "measuring the token count and would pass with the guard removed"
        )


def test_the_indexer_refuses_a_prefill_ring_handed_to_a_decode_step():
    """``prefill_tail`` on a decode step is refused by name."""
    _require_cpu_mode()
    fixture = _fixture()
    root, layers = fixture["root"], fixture["layers"]
    root.bind_kv_cache(_runner_shaped_caches(root))
    banks = root.glm5next_layer_banks
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["req-0"])
    runner.model = root
    runner.max_model_len = E2E_MAX_SEQ_LEN
    runner.max_num_reqs = E2E_MAX_NUM_SEQS
    indexer = _sparse_indexer(layers)
    rings = _live_rings(runner, banks)
    if not rings:
        raise item.VacuousControlError(
            "the live side caches carry no ring, so there is no prefill_tail to hand a "
            "decode step and this check cannot reach its refusal"
        )
    ring = rings[0]["tail"][0]
    pool_cache = item._mla_pool_cache(pages=item.STACK_PAGES)
    selection = item._mla_selection_operands(
        tokens=item.STACK_TOKENS, pages=item.STACK_PAGES
    )
    minimal = torch.zeros(1, 1, dtype=pool_cache.dtype)
    refusal = "prefill_tail is the prefill leg's ring"

    def call(**extra):
        return indexer(
            minimal,
            minimal,
            pool_cache,
            selection["seq_lens"],
            max_seq_len=item.STACK_TOKENS,
            page_size=item.MLA_PAGE_SIZE,
            tail=ring,
            position=0,
            **extra,
        )

    with pytest.raises(Exception) as caught:
        call(prefill_tail=ring)
    message = str(caught.value)
    assert refusal in message.lower(), (
        f"the decode step carrying a prefill ring raised, but not the refusal this check "
        f"names: {message}"
    )

    control = None
    try:
        call()
    except Exception as exc:
        control = f"{type(exc).__name__}: {exc}"
    if control is not None and refusal in control.lower():
        raise item.VacuousControlError(
            "the decode step raised the prefill-ring refusal even without a prefill ring, "
            "so this check is not measuring the argument it names"
        )


def test_the_indexer_refuses_a_prefill_ring_with_no_end_position():
    """``prefill_tail`` without ``prefill_end_position`` is refused by name."""
    _require_cpu_mode()
    fixture = _fixture()
    root, layers = fixture["root"], fixture["layers"]
    root.bind_kv_cache(_runner_shaped_caches(root))
    banks = root.glm5next_layer_banks
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["req-0"])
    runner.model = root
    runner.max_model_len = E2E_MAX_SEQ_LEN
    runner.max_num_reqs = E2E_MAX_NUM_SEQS
    indexer = _sparse_indexer(layers)
    rings = _live_rings(runner, banks)
    if not rings:
        raise item.VacuousControlError(
            "the live side caches carry no ring, so there is no prefill_tail to seed and "
            "this check cannot reach its refusal"
        )
    ring = rings[0]["tail"][0]
    pool_cache = item._mla_pool_cache(pages=item.STACK_PAGES)
    selection = item._mla_selection_operands(
        tokens=item.STACK_TOKENS, pages=item.STACK_PAGES
    )
    minimal = torch.zeros(1, 1, dtype=pool_cache.dtype)
    refusal = "prefill_tail with no prefill_end_position"

    def call(**extra):
        return indexer(
            minimal,
            minimal,
            pool_cache,
            selection["seq_lens"],
            max_seq_len=item.STACK_TOKENS,
            page_size=item.MLA_PAGE_SIZE,
            slot_mapping=selection["slot_mapping"],
            prefill_tail=ring,
            **extra,
        )

    with pytest.raises(Exception) as caught:
        call()
    message = str(caught.value)
    assert refusal in message.lower(), (
        f"the seeding call raised, but not the refusal this check names: {message}"
    )

    control = None
    try:
        call(prefill_end_position=int(item.STACK_TOKENS))
    except Exception as exc:
        control = f"{type(exc).__name__}: {exc}"
    if control is not None and refusal in control.lower():
        raise item.VacuousControlError(
            "the seeding call still reported a missing end position after one was supplied, "
            "so this check is not measuring that argument"
        )


def test_a_fresh_sequence_resets_the_cursor_so_two_requests_never_share_the_ring():
    """The live ring carries a position cursor, and a fresh sequence resets it."""
    _require_cpu_mode()
    root = _fixture()["root"]
    root.bind_kv_cache(_runner_shaped_caches(root))
    banks = root.glm5next_layer_banks
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["req-0"])
    runner.model = root
    runner.max_model_len = E2E_MAX_SEQ_LEN
    runner.max_num_reqs = E2E_MAX_NUM_SEQS
    prompt = int(item.STACK_TOKENS)
    stale = prompt + STALE_DECODE_GAP
    if STALE_DECODE_GAP == 0:
        raise item.VacuousControlError(
            "the stale decode must sit at a position the cursor does not name; a gap of 0 "
            "is the position the cursor does name"
        )
    if stale + 1 > E2E_MAX_SEQ_LEN:
        raise item.VacuousControlError(
            f"the stale decode at {stale} plus its token exceeds the {E2E_MAX_SEQ_LEN} "
            f"slots this fixture allocates, so it would be refused for length instead"
        )

    def step(cached: int, tokens: int):
        return _model_kwargs(
            runner,
            input_ids=torch.zeros(tokens, dtype=torch.long),
            cached=cached,
            sampling_row=tokens - 1 if tokens > 1 else 0,
        )

    step(0, prompt)
    opened = int(_slot_position(runner))
    step(prompt, 1)
    advanced = int(_slot_position(runner))
    assert opened == prompt, (
        f"a fresh prefill of {prompt} tokens left the cursor at {opened}; the cursor must "
        f"name the position the ring has been advanced to"
    )
    assert advanced == prompt + 1, (
        f"a decode that continued the sequence left the cursor at {advanced} rather than "
        f"{prompt + 1}, so the ring's recorded position no longer tracks the work done"
    )

    with pytest.raises(ValueError) as caught:
        step(stale, 1)
    message = str(caught.value)
    assert "stands at position" in message, (
        f"a decode at {stale} raised, but not the cursor refusal this check names: {message}"
    )
    assert int(_slot_position(runner)) == advanced, (
        "the refused step moved the cursor; a refusal must leave the ring's recorded "
        "position exactly as it was, or the next legitimate step is refused for a "
        "mismatch the refused one caused"
    )

    # A position-0 prefill the carriers builder refuses must still leave the
    # ring unowned, so the previous sequence's next step is refused too.
    for side in _live_rings(runner, banks):
        side["tail"].fill_(PLANTED_RING)
    wrong_page = {
        bank["name"]: _entry(
            row=range(PROMPT_BLOCKS),
            tokens=prompt,
            cached=0,
            threshold=1,
            block_size=int(item.MLA_PAGE_SIZE) * 2,
        )
        for bank in banks
    }
    with pytest.raises(ValueError) as opening:
        runner._glm5next_model_kwargs(
            _generic(tokens=prompt, metadata=wrong_page, sampling_row=prompt - 1)
        )
    refused_opening = str(opening.value)
    if ("stands at position" in refused_opening
            or "holds no sequence cursor" in refused_opening):
        raise item.VacuousControlError(
            "the position-0 prefill was refused by the cursor rather than by the carriers "
            "builder, so this check never reaches the window between emptying the ring "
            "and opening its cursor"
        )
    assert _slot_position(runner) is None, (
        f"a refused opening left the cursor at {_slot_position(runner)}, while "
        f"the ring had already been emptied; the previous sequence's next step would then "
        f"be served from blanks instead of refused"
    )
    with pytest.raises(ValueError) as orphan:
        step(advanced, 1)
    orphaned = str(orphan.value)
    assert "holds no sequence cursor" in orphaned, (
        f"after a refused opening, the previous sequence's next step at {advanced} was not "
        f"refused for having no owner: {orphaned}"
    )

    for side in _live_rings(runner, banks):
        side["tail"].fill_(PLANTED_RING)
    step(0, prompt)
    reset = int(_slot_position(runner))
    own = _own_slot(runner)
    planted = max(
        float(side["tail"][own].abs().max()) for side in _live_rings(runner, banks)
    )
    assert reset == prompt, (
        f"the second fresh prefill left the cursor at {reset}; a prefill at position 0 is "
        f"a new sequence and must reopen the ring at its own length"
    )
    assert planted == 0.0, (
        "the second sequence started on the first sequence's planted ring, which is a "
        "silent cross-sequence read"
    )

    with pytest.raises(ValueError) as after:
        step(advanced, 1)
    control = str(after.value)
    assert "stands at position" in control, (
        f"the first sequence's next position {advanced} was still served after a second "
        f"sequence opened, so the cursor did not reset: {control}"
    )
    assert str(reset) in control, (
        "the refusal does not report the position the ring actually stands at, so a reader "
        "cannot tell which sequence owns the ring"
    )


def test_a_synthetic_decode_at_position_zero_is_served_and_leaves_the_cursor_alone():
    """A synthetic (warmup) decode at position zero is served without moving a real sequence's cursor."""
    _require_cpu_mode()
    root = _fixture()["root"]
    root.bind_kv_cache(_runner_shaped_caches(root))
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["req-0"])
    runner.model = root
    runner.max_model_len = E2E_MAX_SEQ_LEN
    runner.max_num_reqs = E2E_MAX_NUM_SEQS
    prompt = int(item.STACK_TOKENS)

    def step(cached: int, tokens: int):
        return _model_kwargs(
            runner,
            input_ids=torch.zeros(tokens, dtype=torch.long),
            cached=cached,
            sampling_row=tokens - 1 if tokens > 1 else 0,
        )

    step(0, prompt)
    step(prompt, 1)
    before = int(_slot_position(runner))

    scheduled = runner.input_batch
    runner.input_batch = SimpleNamespace(req_ids=[])
    synthetic = step(0, 1)
    runner.input_batch = scheduled
    after = _slot_position(runner)
    carrier = synthetic["layer_carriers"][0]
    if "slot_mapping" in carrier:
        raise item.VacuousControlError(
            "the synthetic call at cached length 0 built a prefill carrier, so it went down "
            "the opening path and this check is not measuring the synthetic decode at all"
        )
    assert "tail" in carrier and "position" in carrier, (
        f"the synthetic call did not build a decode carrier; its keys are {sorted(carrier)} "
        f"and the decode leg's own keywords are tail and position"
    )
    assert after is not None and int(after) == before, (
        f"the synthetic decode moved the cursor from {before} to {after}; a warmup between "
        f"two real steps must be invisible to the sequence that owns the ring"
    )

    step(before, 1)
    resumed = int(_slot_position(runner))
    assert resumed == before + 1, (
        f"after the synthetic step the real sequence resumed to {resumed} rather than "
        f"{before + 1}; if the synthetic step had taken or cleared the claim, this step "
        f"would have been refused instead"
    )


def test_a_real_decode_with_no_open_sequence_is_still_refused_by_name():
    """A real decode with no open sequence is refused by name; only position zero is synthetic."""
    _require_cpu_mode()
    root = _fixture()["root"]
    root.bind_kv_cache(_runner_shaped_caches(root))
    banks = root.glm5next_layer_banks
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["req-0"])
    runner.model = root
    runner.max_model_len = E2E_MAX_SEQ_LEN
    runner.max_num_reqs = E2E_MAX_NUM_SEQS

    def step(cached: int, tokens: int):
        return _model_kwargs(
            runner,
            input_ids=torch.zeros(tokens, dtype=torch.long),
            cached=cached,
            sampling_row=0,
        )

    _live_rings(runner, banks)
    opened = _slot_position(runner)
    if opened is not None:
        raise item.VacuousControlError(
            f"this runner already carries a cursor at {opened}, so 'no open sequence' is "
            f"not the state being measured"
        )

    with pytest.raises(ValueError) as caught:
        step(1, 1)
    message = str(caught.value)
    assert "holds no sequence cursor" in message, (
        f"a decode at position 1 with no sequence open raised, but not the refusal this "
        f"check names: {message}"
    )

    runner.input_batch = SimpleNamespace(req_ids=[])
    step(0, 1)
    after = _slot_position(runner)
    assert after is None, (
        f"the synthetic step at position 0 opened a cursor at {after}; a step that is not a "
        f"real sequence step must leave the ring unclaimed"
    )


def test_scattered_pages_give_the_same_logits_as_pages_in_one_run():
    """Where a request's pages sit cannot matter: scattered pages give the same logits, bit for bit."""
    _require_cpu_mode()

    contiguous = list(range(PROMPT_BLOCKS))
    scattered = [PROMPT_BLOCKS + offset for offset in reversed(range(PROMPT_BLOCKS))]
    assert contiguous != scattered
    assert len(set(scattered)) == len(scattered)
    steps = [later - earlier for earlier, later in zip(scattered, scattered[1:])]
    assert any(step != 1 for step in steps), (
        "the scattered pages are one ascending run, so the test is not reading the "
        "scattered case at all"
    )

    input_ids = torch.randint(
        0,
        item.STACK_VOCAB_SIZE,
        (item.STACK_TOKENS,),
        generator=torch.Generator().manual_seed(item.SEED_STACK_IDS),
        dtype=torch.int64,
    )
    positions = torch.tensor(item.ROOT_SAMPLING_POSITIONS, dtype=torch.long)

    root = _fixture()["root"]

    def run(block_ids):
        """One prefill through runner-built carriers naming ``block_ids``."""
        caches = _runner_shaped_caches(root)
        root.bind_kv_cache(caches)
        banks = root.glm5next_layer_banks
        text_config = root.text_config
        side = NeuronModelRunner._glm5next_side_caches(
            banks,
            index_kpool=int(text_config.index_kpool),
            index_head_dim=int(text_config.index_head_dim),
            max_seq_len=item.STACK_TOKENS,
            request_slots=E2E_STATE_SLOTS,
        )
        carriers = NeuronModelRunner._glm5next_layer_carriers(
            banks,
            side,
            geometries=_geometries(banks, block_ids=block_ids, state_slot=0),
            is_prefill=True,
            tokens=item.STACK_TOKENS,
            start_position=0,
            softmax_scale=item.MLA_SOFTMAX_SCALE,
            max_seq_len=item.STACK_TOKENS,
            index_kpool=int(text_config.index_kpool),
        )
        logits = root.forward(
            input_ids, layer_carriers=carriers, sampling_positions=positions
        )
        sparse = next(
            carrier for bank, carrier in zip(banks, carriers)
            if bank["family"] == "self_attn"
        )
        return logits, sparse

    run_logits, run_sparse = run(contiguous)
    scatter_logits, scatter_sparse = run(scattered)
    assert tuple(run_logits.shape) == tuple(scatter_logits.shape)

    assert not torch.equal(
        run_sparse["block_table_row"], scatter_sparse["block_table_row"]
    )
    assert not torch.equal(run_sparse["latent_slots"], scatter_sparse["latent_slots"])

    page = int(item.MLA_PAGE_SIZE)
    written = sorted({int(slot) // page for slot in scatter_sparse["latent_slots"]})
    assert written and not set(written) & set(contiguous)

    assert torch.equal(run_logits, scatter_logits), (
        "the same prompt produced different logits from pages held in a different order, "
        "so the gather is reading rows by where they sit rather than by what the table names"
    )
