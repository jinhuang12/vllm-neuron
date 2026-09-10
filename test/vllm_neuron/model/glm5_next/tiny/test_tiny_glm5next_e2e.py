"""``inc-glm53f-054b``: the runner's caches reach the layers, and eight tokens match a reference.

WHAT THIS FILE MEASURES. This half of ``inc-glm53f-054`` threads the runner's allocated caches
into the per-layer carriers this model family takes as forward ARGUMENTS, and then measures the
block's registered acceptance over that thread:

  1. ``Glm5NextForConditionalGeneration.bind_kv_cache`` -- the method the runner calls on every
     start-up (``neuron_model_runner.py:9142``) and that this package did not have until plan
     revision 276. It must map every layer the spec reports onto that layer's OWN slots, keep
     views rather than copies, and refuse by name anything it cannot map.
  2. ``NeuronModelRunner._glm5next_layer_carriers`` and its two operand derivations -- the runner
     side that turns those banks into one mapping per layer.
  3. ``NeuronModelRunner._glm5next_model_kwargs`` -- the converter the five production call
     sites go through. It must read EACH layer's own KV-cache group and must not hand the root
     a KV page size where the root declares an FP8 weight-quant block.
  4. The registered acceptance itself: eight generated tokens whose logits match a torch
     reference composed from ``-054a``'s landed oracles, at the registered tolerance.

THE BIND IS EXERCISED, NOT ASSUMED. The last item runs the root's forward with carriers built by
the runner's own helpers out of a runner-shaped cache dict, and then requires the RUNNER'S OWN
TENSORS to have changed. A cache that arrived as a copy, or a bank handed to the wrong layer,
fails that conjunct. An item that built its own caches and never bound anything would pass while
the production path stayed broken, which is the reason this file exists in this shape.

WHAT THIS FILE DOES NOT MEASURE, stated so the gap is not read as coverage. One sequence per
forward and one token per decode step: a batch of more than one request and a multi-token decode
(speculative decoding's verify step) both refuse by name rather than being threaded, so this file
measures the refusal and not the feature. One rank: the reference is single-rank and reads the
fixture's own per-rank operands, so it neither reduces FFN partial sums across ranks nor
compensates a scale grid -- the two defects ``inc-glm53f-054c`` and ``-054d`` own stay visible
through it. No hardware: the CPU lane is the whole scope here, and a real Neuron compile is
stage 7's. The recurrent (KDA) arms are measured on specs this file substitutes rather than on a
KDA layer, because the landed tiny stack is sparse-attention on every layer
(``test_tiny_glm5next_forward.py:3822-3842`` reads ``layer.self_attn`` for all of them); the
mapper and the converter read no layer module, so the substitution exercises the same code the
runner drives.

HOW TO RUN IT, and both variables must be in the environment rather than set from a fixture
(``inc-glm53f-051``'s obligation 4):

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 \\
        python -m pytest -s -rA test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_e2e.py
"""

from __future__ import annotations

import inspect
import math
import os

import pytest
import torch

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

# `-054a`'s OWN fixture, dials and landed prefill operands. Imported, never re-implemented:
# this file measures the runner against what that item's acceptance ran, and the import order
# follows the landed convention (`test/vllm_neuron/functional/dsa/test_causal_bound.py:99-110`).
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as item

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The acceptance's own token count: "generates **8/8** tokens" (plan `:1183`). The first
#: comes from the prompt's last row and the other seven from one decode step each, so the
#: generation exercises both legs and appends exactly eight tokens.
GENERATED_TOKENS = 8

#: Whole blocks, always DERIVED from a slot count so a change to any landed dial moves them.
def _blocks_for(slots: int) -> int:
    """The blocks a sequence of ``slots`` slots occupies, the runner's own rounding."""
    return -(-int(slots) // item.MLA_PAGE_SIZE)


#: The prompt occupies this many blocks, which is the run the converter slices for a prefill.
PROMPT_BLOCKS = _blocks_for(item.STACK_TOKENS)

#: The bank must hold the prompt AND everything the generation appends, or the last decode
#: step would write past the end -- which the layer refuses (`model_fp8.py:6505-6509`).
E2E_BLOCKS = _blocks_for(item.STACK_TOKENS + GENERATED_TOKENS)

#: The longest sequence this file admits, the number the side caches are sized from exactly as
#: the runner sizes them from ``max_model_len``.
E2E_MAX_SEQ_LEN = item.STACK_TOKENS + GENERATED_TOKENS

#: The recurrent-state geometry the substituted spec reports. Small and arbitrary: the mapper
#: under test reads shapes off the spec and never off a layer, so these numbers only have to
#: be self-consistent between the spec and the banks this file allocates for it.
E2E_CONV_STATE_SHAPE = (2, 3)
E2E_RECURRENT_STATE_SHAPE = (2, 4, 4)
E2E_STATE_SLOTS = 8


def _fixture(**overrides):
    """`-054a`'s root fixture, with the ONE dial the registered constraint set names.

    THE OVERRIDE IS `num_key_value_heads = 2`, and it is here rather than in `-054a`'s file
    for two reasons that both bind: the acceptance this block carries names that value in its
    constraint set (plan `:1183`), and the plan's Tests bullet says the two halves write
    different files in one directory and neither edits the other's. `_root_config` takes
    overrides and they win over its dials (`test_tiny_glm5next_forward.py:2623`), so the
    constraint is met by asking for it here.

    IT IS THE SAME OVERRIDE FOR EVERY ITEM IN THIS FILE, so one config runs everywhere and no
    item measures a tree another item did not.
    """
    return item._root_fixture(num_key_value_heads=2, **overrides)


def _require_cpu_mode() -> None:
    """Both flags come from the process environment, or this file is not measuring the CPU lane."""
    missing = [
        name
        for name in ("VLLM_NEURON_CPU_MODE", "NKI_SIMULATOR")
        if os.environ.get(name) != "1"
    ]
    if missing:
        raise item.VacuousControlError(
            f"{missing} must be 1 in the process environment; the seams read them at "
            f"import time, so a fixture that set them here would measure the wrong backend"
        )


def _runner_shaped_caches(root) -> dict[str, list[torch.Tensor]]:
    """The dict ``initialize_kv_cache`` hands ``bind_kv_cache``, built the runner's own way.

    Every shape here is the runner's: a sparse layer gets a key/value PAIR of
    ``[blocks, num_kv_heads, block_size, head_size]`` (``neuron_model_runner.py:9002-9038``)
    and a recurrent layer gets two ``[slots, *state shape]`` banks in conv-then-recurrent
    order (``:9100-9130``). The geometry is read off the spec the model itself produced, so
    this helper cannot disagree with the model about what was asked for.
    """
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
            E2E_BLOCKS,
            int(layer_spec.num_kv_heads),
            item.MLA_PAGE_SIZE,
            int(layer_spec.head_size),
        )
        caches[layer_spec.name] = [
            torch.zeros(shape, dtype=layer_spec.dtype),
            torch.zeros(shape, dtype=layer_spec.dtype),
        ]
    return caches


def _assert_config_matches_landed_dials(text_config) -> None:
    """The tree that ran must carry the two index dials the landed operands were built from.

    ``inc-glm53f-054a``'s fixture puts them into the config it builds
    (``test_tiny_glm5next_forward.py:2653-2654``) and this file's derivations read them back
    off the built config, so a fixture that stopped passing them would make every operand
    comparison below compare two different geometries and still look green.
    """
    pairs = {
        "index_kpool": (int(text_config.index_kpool), item.MLA_INDEX_KPOOL),
        "index_head_dim": (int(text_config.index_head_dim), item.MLA_INDEX_HEAD_DIM),
    }
    wrong = {name: values for name, values in pairs.items() if values[0] != values[1]}
    if wrong:
        raise item.VacuousControlError(
            f"the built config and this file's landed dials disagree on {wrong} "
            f"(config value first), so the operands here are written for another geometry"
        )


def _recurrent_spec(root) -> KVSpec:
    """The stack's own spec with every layer REPORTING recurrent geometry instead of a pair.

    The mapper recognises a family by the fields the spec carries, never by a layer name
    (``model_fp8.py``'s ``bind_kv_cache``, the same test the runner makes at
    ``neuron_model_runner.py:9193-9199``), and it reads no layer module at all. So a spec
    with the stack's own length and names, reporting the four ``kda_*`` fields, drives the
    recurrent branch of exactly the code the runner drives.
    """
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


def _geometries(banks, *, block_ids, state_slot: int, page_size: int | None = None):
    """One geometry per bank, the shape the carrier builder pairs positionally with its banks.

    THE PAGE COMES FROM THE LANDED DIAL, not from the bank, so the builder's cross-check
    between the group's page and the bank's own paging compares two independently sourced
    numbers instead of one number twice.
    """
    page = item.MLA_PAGE_SIZE if page_size is None else page_size
    return [
        {
            "block_ids": [int(value) for value in block_ids],
            "state_slot": int(state_slot),
            "page_size": int(page),
        }
        for _ in banks
    ]


def _mixed_spec(root) -> KVSpec:
    """The stack's own spec with ALTERNATE layers reporting recurrent geometry instead of a pair.

    THIS IS THE HYBRID SHAPE THAT MAKES TWO KV-CACHE GROUPS EXIST. A sparse layer's spec
    becomes a ``FullAttentionSpec`` and a recurrent layer's a ``MambaSpec``
    (``neuron_model_runner.py:9193-9245``), and the KV-cache manager gives each class its own
    group with its own block table. The landed tiny stack is sparse on every layer, so one
    group is all it would ever have; this builds the two-group case out of the stack's own
    names and geometry, reading no layer module -- and neither the mapper nor the converter
    reads one either.
    """
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


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 1. bind_kv_cache maps every sparse layer onto its own slots, as views.
# ══════════════════════════════════════════════════════════════════════════════════════


def test_bind_kv_cache_maps_every_sparse_layer_onto_its_own_slots():
    """Every layer the spec reports gets its OWN bank, and the bank is the runner's storage.

    FOUR CONJUNCTS, each of which fails on a different real defect:

    * the attribute does not exist before the call and does after it -- so the item cannot
      pass against a tree that never ran the method;
    * one bank per spec layer, in spec order, keyed by the spec's own name -- so a mapper
      that skipped a layer or reordered them fails;
    * the latent view's storage IS the runner's tensor and no two layers share one -- so a
      mapper that copied, or that handed one bank to two layers, fails;
    * a write through the view lands in the runner's tensor -- the property the whole paged
      cache depends on, checked directly rather than inferred from the shapes.
    """
    _require_cpu_mode()
    root = _fixture()["root"]
    caches = _runner_shaped_caches(root)
    spec_layers = root.get_kv_spec().layers

    assert not hasattr(root, "glm5next_layer_banks"), (
        "the stack already carries glm5next_layer_banks before anything bound it, so this "
        "item cannot tell the method from the fixture"
    )
    root.bind_kv_cache(caches)
    banks = root.glm5next_layer_banks

    print(f"TINYE2E|bind_sparse|layers={len(banks)}|spec={len(spec_layers)}"
          f"|blocks={E2E_BLOCKS}|page={item.MLA_PAGE_SIZE}")
    assert len(banks) == len(spec_layers) == item.STACK_LAYERS
    assert [bank["name"] for bank in banks] == [s.name for s in spec_layers]
    assert [bank["layer_index"] for bank in banks] == list(range(len(spec_layers)))
    assert {bank["family"] for bank in banks} == {"self_attn"}

    pointers = set()
    for bank, layer_spec in zip(banks, spec_layers):
        allocated = caches[layer_spec.name][0]
        view = bank["latent_cache"]
        print(f"TINYE2E|bank|{bank['name']}|bank={tuple(allocated.shape)}"
              f"|view={tuple(view.shape)}|slots={bank['slots']}")
        assert tuple(view.shape) == (
            E2E_BLOCKS * item.MLA_PAGE_SIZE,
            1,
            int(layer_spec.head_size),
        )
        assert int(bank["slots"]) == E2E_BLOCKS * item.MLA_PAGE_SIZE
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


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 2. bind_kv_cache maps recurrent layers by the fields the spec carries.
# ══════════════════════════════════════════════════════════════════════════════════════


def test_bind_kv_cache_maps_recurrent_layers_by_the_fields_the_spec_carries(monkeypatch):
    """A layer that reports ``kda_*`` geometry gets its two state banks, paired positionally.

    THE CONTROL IS THE FAMILY LABEL: the same stack, the same names and the same mapper,
    reporting recurrent geometry instead of a key/value pair, must come back
    ``linear_attn`` on every layer. Item 1 requires ``self_attn`` on the same stack, so the
    pair of items shows the branch is taken from the spec and not from the stack.
    """
    _require_cpu_mode()
    root = _fixture()["root"]
    spec = _recurrent_spec(root)
    monkeypatch.setattr(root, "get_kv_spec", lambda: spec)
    caches = _runner_shaped_caches(root)

    root.bind_kv_cache(caches)
    banks = root.glm5next_layer_banks
    print(f"TINYE2E|bind_recurrent|layers={len(banks)}"
          f"|families={sorted({b['family'] for b in banks})}")
    assert {bank["family"] for bank in banks} == {"linear_attn"}
    for bank, layer_spec in zip(banks, spec.layers):
        conv, recurrent = caches[layer_spec.name]
        assert bank["conv_state"] is conv
        assert bank["recurrent_state"] is recurrent
        assert int(bank["state_slots"]) == E2E_STATE_SLOTS
        assert tuple(bank["conv_state"].shape[1:]) == E2E_CONV_STATE_SHAPE
        assert tuple(bank["recurrent_state"].shape[1:]) == E2E_RECURRENT_STATE_SHAPE


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 3. bind_kv_cache refuses, by name, every dict it cannot map.
# ══════════════════════════════════════════════════════════════════════════════════════


def test_bind_kv_cache_refuses_a_missing_layer_by_name():
    """A layer the dict does not hold refuses and prints what the dict does hold."""
    _require_cpu_mode()
    root = _fixture()["root"]
    caches = _runner_shaped_caches(root)
    absent = root.get_kv_spec().layers[0].name
    caches.pop(absent)
    with pytest.raises(ValueError, match=f"no entry for KV layer '{absent}'"):
        root.bind_kv_cache(caches)


def test_bind_kv_cache_refuses_a_bank_of_the_wrong_rank():
    """A bank that is not the runner's four-axis allocation refuses rather than being viewed."""
    _require_cpu_mode()
    root = _fixture()["root"]
    caches = _runner_shaped_caches(root)
    name = root.get_kv_spec().layers[0].name
    flat = caches[name][0]
    caches[name] = [flat.reshape(-1), caches[name][1]]
    with pytest.raises(ValueError, match="the runner allocates"):
        root.bind_kv_cache(caches)


def test_bind_kv_cache_refuses_a_bank_whose_geometry_is_not_the_specs():
    """A head count or width the model did not ask for refuses, and says both numbers."""
    _require_cpu_mode()
    root = _fixture()["root"]
    caches = _runner_shaped_caches(root)
    layer_spec = root.get_kv_spec().layers[0]
    caches[layer_spec.name] = [
        torch.zeros(
            (E2E_BLOCKS, 2, item.MLA_PAGE_SIZE, int(layer_spec.head_size)),
            dtype=layer_spec.dtype,
        ),
        caches[layer_spec.name][1],
    ]
    with pytest.raises(ValueError, match="head count and width"):
        root.bind_kv_cache(caches)


def test_bind_kv_cache_refuses_a_partial_recurrent_geometry(monkeypatch):
    """Half a recurrent declaration refuses: the two states are paired positionally."""
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
    """A spec shorter than the stack refuses: the carriers are paired positionally."""
    _require_cpu_mode()
    root = _fixture()["root"]
    short = KVSpec(layers=root.get_kv_spec().layers[:-1])
    caches = _runner_shaped_caches(root)
    monkeypatch.setattr(root, "get_kv_spec", lambda: short)
    with pytest.raises(ValueError, match="the stack holds"):
        root.bind_kv_cache(caches)


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 4. the two operand derivations equal the landed tiny operands.
# ══════════════════════════════════════════════════════════════════════════════════════


def test_derived_carrier_operands_equal_the_landed_tiny_operands():
    """The runner's ``slot_mapping`` and ``seq_lens`` are the landed item's, value for value.

    The landed stack item builds both operands from the indexer's stated rules
    (``test_tiny_glm5next_forward.py:2853-2872``) and ``inc-glm53f-054a``'s acceptance ran
    against them. The runner now derives the same two, so equality here is a cross-check
    against a landed reference rather than against this file's own arithmetic.

    THE CONTROL IS THE SAME DERIVATION AT A NON-ZERO START POSITION, which must NOT equal
    the landed operand: both derivations are absolute in the sequence, so shifting the
    chunk's start moves every pool boundary and every causal length. A function that
    ignored ``start_position`` -- the chunked-prefill defect this cross-check exists to
    catch -- passes the equality above and fails this.
    """
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
    print(f"TINYE2E|operands|slots={tuple(slots.shape)}:{slots.dtype}"
          f"|seq_lens={tuple(lengths.shape)}:{lengths.dtype}"
          f"|pooled_rows={int((slots >= 0).sum())}")
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
    """The pooled store leaves one trash row above every addressable pool, and the ring is sized.

    The minimum is the indexer's own, in its own words: ``allocate at least candidates + 1``
    where ``candidates = max_seq_len // index_kpool`` (``model_fp8.py:5288-5295``). The
    allocator sits ON that boundary rather than over-allocating, so the assertions below
    are equalities in the derived row count; the guard itself is the instrument in item 5,
    where a store one row short raises inside the indexer instead of being asserted here.
    """
    _require_cpu_mode()
    root = _fixture()["root"]
    caches = _runner_shaped_caches(root)
    root.bind_kv_cache(caches)
    banks = root.glm5next_layer_banks
    text_config = root.text_config
    _assert_config_matches_landed_dials(text_config)
    side = NeuronModelRunner._glm5next_side_caches(
        banks,
        index_kpool=int(text_config.index_kpool),
        index_head_dim=int(text_config.index_head_dim),
        max_seq_len=item.STACK_TOKENS,
    )
    candidates = item.STACK_TOKENS // int(text_config.index_kpool)
    print(f"TINYE2E|side_caches|sets={len(side)}|candidates={candidates}"
          f"|pool_rows={tuple(side[0]['pool_cache'].shape)}"
          f"|tail={tuple(side[0]['tail'].shape)}")
    assert len(side) == len(banks)
    for entry, bank in zip(side, banks):
        assert int(entry["pool_cache"].shape[0]) >= candidates + 1
        assert int(entry["pool_cache"].shape[1]) == int(text_config.index_head_dim)
        assert tuple(entry["tail"].shape) == (
            2,
            int(text_config.index_kpool),
            int(text_config.index_head_dim),
        )
        assert entry["pool_cache"].dtype == bank["latent_cache"].dtype
    assert len({entry["pool_cache"].data_ptr() for entry in side}) == len(side), (
        "two layers share one pooled-key store, so the second would read the first's pools"
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 5. the runner's own carriers drive the root, and the root writes the runner's cache.
# ══════════════════════════════════════════════════════════════════════════════════════


def test_runner_built_carriers_drive_the_root_and_write_the_runners_own_cache():
    """One prefill through carriers the RUNNER built, ending in the runner's tensors changing.

    THIS IS THE ITEM THE INCREMENT EXISTS FOR. Nothing in the carrier chain is built by this
    file: ``bind_kv_cache`` maps a runner-shaped dict, the runner's own helpers allocate the
    two un-specced side caches and assemble the mappings, and the root is called with those.

    THE KEY SETS ARE CHECKED AGAINST THE LANDED PREFILL CARRIER, not against a list typed
    here: ``inc-glm53f-054a``'s stack item builds the carrier its forward accepted
    (``test_tiny_glm5next_forward.py:3831-3843``), so a runner mapping that grew or lost a
    key fails against the shape the landed acceptance ran.

    THE LAST CONJUNCT IS THE WRITE-THROUGH: every sparse layer's latent bank -- the tensor
    the RUNNER allocated, not the view -- and every pooled store must differ from its
    pre-call copy. A bind that handed out copies, a slice that pointed at the wrong blocks,
    or a carrier that reached the wrong layer all fail here, and no tolerance is involved.
    """
    _require_cpu_mode()
    fixture = _fixture()
    root, layers = fixture["root"], fixture["layers"]
    caches = _runner_shaped_caches(root)
    root.bind_kv_cache(caches)
    banks = root.glm5next_layer_banks
    text_config = root.text_config
    _assert_config_matches_landed_dials(text_config)

    side = NeuronModelRunner._glm5next_side_caches(
        banks,
        index_kpool=int(text_config.index_kpool),
        index_head_dim=int(text_config.index_head_dim),
        max_seq_len=item.STACK_TOKENS,
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

    landed = item._stack_carriers(
        layers,
        item._mla_selection_operands(
            tokens=item.STACK_TOKENS, pages=item.STACK_PAGES
        ),
    )
    assert len(carriers) == len(landed) == item.STACK_LAYERS
    for index, (got, want) in enumerate(zip(carriers, landed)):
        print(f"TINYE2E|carrier|{index}|keys={sorted(got)}")
        assert set(got) - set(want) == {"prefill_tail", "prefill_end_position"}, (
            f"layer {index}'s runner-built carrier carries {sorted(set(got))} and the "
            f"landed prefill carrier carries {sorted(set(want))}; commit 6 adds the ring "
            f"and its end position to the prefill leg and nothing else may move"
        )
        assert set(want) - set(got) == set(), (
            f"layer {index}'s runner-built carrier LOST {sorted(set(want) - set(got))}, "
            f"which the landed prefill carrier declares"
        )
        assert tuple(got["latent_cache"].shape) == tuple(want["latent_cache"].shape)
        assert int(got["start_position"]) == 0
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

    print(f"TINYE2E|forward|logits={tuple(logits.shape)}:{logits.dtype}"
          f"|rows={len(item.ROOT_SAMPLING_POSITIONS)}|vocab={item.STACK_VOCAB_SIZE}")
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
# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 6. the config that ran IS the constraint set the acceptance names.
# ══════════════════════════════════════════════════════════════════════════════════════


def test_the_tiny_config_is_the_registered_constraint_set():
    """The five constraints the criterion names, read off the config that ran.

    The acceptance names them as one set: fewer than 32 M parameters, `hidden_size % 256 == 0`,
    `intermediate_size >= 512`, `head_dim <= 128` and `num_key_value_heads = 2` (plan `:1183`,
    "the constraint set `model_bringup.md` states for CPU/NKI runs"). An acceptance that
    asserted the logits and never the fixture would pass on a tree the criterion does not
    describe, so every constraint is READ here and printed before it is asserted.

    EVERY VALUE COMES OFF THE BUILT CONFIG AND THE BUILT MODULE, never from this file's
    constants: a fixture that stopped setting one of them must redden this item rather than
    agree with a copy of the number.
    """
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
            f"the built config declares none of {missing}, so this item cannot read the "
            f"constraint set the acceptance names off the tree that ran"
        )
    parameters = sum(int(p.numel()) for p in root.parameters() if p is not None)
    head_dims = {
        name: int(getattr(cfg, name))
        for name in ("qk_nope_head_dim", "v_head_dim", "index_head_dim")
    }
    print(f"TINYE2E|constraint_set|parameters={parameters}|bound=32000000"
          f"|hidden_size={int(cfg.hidden_size)}|mod256={int(cfg.hidden_size) % 256}"
          f"|intermediate_size={int(cfg.intermediate_size)}"
          f"|num_key_value_heads={int(cfg.num_key_value_heads)}"
          f"|head_dims={sorted(head_dims.items())}")
    assert parameters < 32_000_000
    assert int(cfg.hidden_size) % 256 == 0
    assert int(cfg.intermediate_size) >= 512
    assert max(head_dims.values()) <= 128
    assert int(cfg.num_key_value_heads) == 2


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 7. THE REGISTERED ACCEPTANCE: eight tokens, the route predicate, and the reference.
# ══════════════════════════════════════════════════════════════════════════════════════


def _entry(*, row, tokens: int, cached: int, threshold: int, block_size: int) -> dict:
    """ONE KV-cache group's attention-metadata entry, at this step's geometry.

    THE KEYS AND THEIR SHAPES ARE THE RUNNER'S OWN, read off the mapping it builds at
    `neuron_model_runner.py:4417-4429`; the converter under test reads five of them --
    `block_table_tensor`, `block_size`, `max_query_len`, `decode_token_threshold` and
    `cached_seq_len` -- and the rest are present so that this is the runner's mapping and not
    a five-key stand-in.
    """
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
        "kv_segment_size": int(table.shape[1]) * int(block_size),
    }


def _metadata(names, *, blocks: int, tokens: int, cached: int, threshold: int = 1) -> dict:
    """The runner's attention-metadata mapping: ONE ENTRY PER LAYER NAME, one group's geometry.

    THE KEYING IS THE RUNNER'S. It builds a block table per KV-cache group and then writes
    that group's entry under every layer name in the group
    (`neuron_model_runner.py:4256-4257`), and the converter looks each layer's own name up. A
    mapping under one made-up key would not reach the code under test at all.
    """
    entry = _entry(row=range(blocks), tokens=tokens, cached=cached, threshold=threshold,
                   block_size=item.MLA_PAGE_SIZE)
    return {str(name): entry for name in names}


def _model_kwargs(runner, *, input_ids, cached: int, sampling_row: int) -> dict:
    """One step's generic runner kwargs, translated by the converter under test.

    THE GENERIC KEYS ARE THE ONES THE RUNNER SENDS (`neuron_model_runner.py:7504-7515`),
    including the six this model implements nowhere, so the converter is measured dropping
    exactly what it says it drops rather than being handed a pre-cleaned mapping.
    """
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
    """`-054a`'s `_stack_attention_half` with its length taken from an argument.

    IT IS THAT FUNCTION, five lines of it re-composed here for ONE reason: the landed one
    fixes its three cache operands at `STACK_TOKENS` (`test_tiny_glm5next_forward.py:3875-3886`)
    and this file compares a growing sequence, while the plan's Tests bullet forbids either
    half from editing the other's file. Every callee is the landed one -- the norm, the
    projection, the indexer, the pooled store, the latent cache and the dense reference -- so
    nothing about the reference's arithmetic is re-derived here.

    THE INDEXER IS EXECUTED, which is the landed function's own reason: a pool selection is a
    discontinuous function of its input, so running the model's own indexer on the reference's
    own states makes the selection the reference's by construction.
    """
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
    """The torch reference model's logits for the LAST row of this token sequence.

    THE COMPOSITION IS `-054a` ITEM 6's, EQUATION FOR EQUATION: the embedding is the table
    index, each layer adds its attention half to the tensor it received
    (`test_tiny_glm5next_forward.py:4644`), then adds its FFN half to that
    (`:4706`), and the stack ends in the final norm (`:4796`); the head projection is item
    7's `_root_reference` (`:5235`), whose gather is a python loop for its own reason.

    EVERY EPSILON COMES FROM THE CONFIG, which the criterion requires in those words --
    `text_config.rms_norm_eps` included: `_ffn_norm` is called with `float(cfg.rms_norm_eps)`
    here and the attention half passes the same value down, so a wrong call-site literal in
    the module under test reddens this comparison instead of cancelling out.

    IT IS SINGLE-RANK BY CONSTRUCTION. `_root_fixture` refuses a world size other than 1
    (`:5178`), and this reference reads the fixture's own per-rank operands, so it neither
    reduces across ranks nor compensates a scale grid -- the two defects `-054c` and `-054d`
    own are left visible rather than papered over.
    """
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
    """The registered predicate, form R-3, in the words it was registered in.

    Three conjuncts, from plan `:1183`'s route bullet and §4b.2:

    1. `can_run_kernel()` is True. Under `VLLM_NEURON_CPU_MODE=1` this reads the
       `NKI_SIMULATOR` flag, so a run launched without the simulator is refused here instead
       of passing on the torch oracle.
    2. the aggregate torch-fallback counter across every seam this campaign owns reads exactly
       0 over the generation. "Every seam" is made complete by `-054a`'s two registry checks,
       which are called below rather than re-implemented: one requires every counter family in
       every registered module to be claimed by a row, the other requires no counter family
       anywhere in `vllm_neuron.functional` to be unregistered.
    3. the SET of seam counters that fired is reported and asserted non-empty -- the conjunct
       that tells this campaign's kernels from `torch` composed end to end.

    IT IS NOT `-054a`'s HELPER. That one takes an expected dispatch count per seam, which is
    form R-1; this block registered R-3, whose value is which path was taken and not how many
    times, so predicting counts here would assert something the register does not.
    """
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
    print(f"TINYE2E|route_predicate|{label}|can_run_kernel={runnable}"
          f"|fired={fired}|fallback_total={sum(fallbacks.values())}"
          f"|fallbacks={sorted(fallbacks.items())}")
    assert runnable, (
        "can_run_kernel() is False, so this run took the torch oracle and no reading below "
        "is a reading of this campaign's kernels"
    )
    assert not fallbacks, (
        f"the torch-fallback counters moved on {sorted(fallbacks)}; the registered predicate "
        f"is exactly 0 across every seam this campaign owns"
    )
    assert fired, (
        "no seam counter moved over the generation, so an 8-token run completed with no NKI "
        "kernel dispatched at all"
    )


def test_the_generation_is_eight_tokens_and_every_step_matches_the_reference():
    """THE REGISTERED ACCEPTANCE. Eight tokens, the route predicate, and eight comparisons.

    The criterion, quoted: a tiny config "generates **8/8** tokens without exception, and its
    logits match a torch reference model built from the same weights at
    `assert_close(rtol=1e-2, atol=1e-5)`" (plan `:1183`). The tolerance and the token count are
    the registered ones and this file holds no other; the constraint set is item 6's.

    HOW THE EIGHT TOKENS ARE PRODUCED. One prefill over the prompt gives the first token from
    its last row; seven decode steps, each one token with the cache growing between them, give
    the other seven. Both legs are therefore exercised, and the eight appended tokens are the
    criterion's `8/8`.

    EVERY STEP GOES THROUGH THE CONVERTER, not through the carrier builder directly: the
    generic runner kwargs are assembled at each step and `_glm5next_model_kwargs` translates
    them, so this item measures the production path from the runner's own mapping down to the
    layers. The runner instance is allocated without running `__init__` -- the converter reads
    two attributes and this file sets both -- so no engine is constructed to measure a
    translation.

    THE FIRST COMPARISON IS THE CONTROL FOR THE OTHER SEVEN. Step 0 is the 128-token prompt,
    which is the composition `-054a`'s items 6 and 7 already measured and whose acceptance
    passed on hardware. A defect in this file's reference reddens that comparison before any
    decode-leg claim rests on it.
    """
    _require_cpu_mode()
    fixture = _fixture()
    root = fixture["root"]
    _assert_config_matches_landed_dials(root.text_config)
    caches = _runner_shaped_caches(root)
    root.bind_kv_cache(caches)

    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = root
    runner.max_model_len = E2E_MAX_SEQ_LEN

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
        "input_ids", "layer_carriers", "sampling_positions"
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

    print(f"TINYE2E|generation|tokens={len(generated)}|expected={GENERATED_TOKENS}"
          f"|decode_steps={GENERATED_TOKENS - 1}|ids={generated}")
    assert len(generated) == GENERATED_TOKENS, (
        f"the generation appended {len(generated)} token(s) and the criterion is "
        f"{GENERATED_TOKENS}/{GENERATED_TOKENS}"
    )
    assert len(produced) == GENERATED_TOKENS

    _assert_route_predicate_r3("the 8-token generation", before, after)

    # ---- THE COMPARISON. One reference per step, over the tokens that step was given.
    for (label, length, sequence), got in zip(steps, produced):
        want = _reference_logits(fixture, sequence[:length])[0].float()
        spread = float((got - want).abs().max())
        print(f"TINYE2E|logits|{label}|tokens={length}|rows={tuple(got.shape)}"
              f"|max_abs_delta={spread:.6g}|rtol=1e-2|atol=1e-5")
        assert torch.isfinite(got).all(), f"{label} produced a non-finite logit"
        torch.testing.assert_close(got, want, rtol=1e-2, atol=1e-5)

    # ---- The caches the runner allocated carry the whole generation, not just the prompt.
    written = int((caches[root.glm5next_layer_banks[0]["name"]][0] != 0).sum())
    print(f"TINYE2E|cache_written|nonzero_elements={written}"
          f"|slots={root.glm5next_layer_banks[0]['slots']}")
    assert written > 0


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 8. the converter reads EACH layer's own KV-cache group, not one group's for all.
# ══════════════════════════════════════════════════════════════════════════════════════


def _grouped_metadata(banks, *, tokens: int, sparse_row, state_row,
                      sparse_block_size: int | None = None, sparse_cached: int = 0,
                      state_cached: int = 0, threshold: int = 1) -> dict:
    """TWO KV-cache groups, each with its own block table, keyed by every layer name.

    This is the mapping a hybrid stack produces: one entry per group, written under every
    layer name of that group (`neuron_model_runner.py:4256-4257`). The two rows differ on
    purpose, so a converter that read one entry for the whole stack would slice the sparse
    layers out of the recurrent group's table -- the defect this item is about.
    """
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
    """One step's generic runner kwargs at a GIVEN metadata mapping.

    The ids are zeros of the right length: the converter reads `input_ids.shape[0]` and
    nothing else off them, so a real prompt would add nothing this item can check.
    """
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
    """A hybrid stack has TWO block tables, and every layer's carrier comes from its own.

    WHY THIS ITEM EXISTS. Commit 1 read one entry -- `next(iter(metadata_map.values()))` --
    and applied its block size, its block run and its first block id to every layer. The
    runner builds one table per KV-CACHE GROUP (`neuron_model_runner.py:4115-4121`) and a
    GLM-5.3-Flash stack is hybrid, so the sparse layers and the recurrent layers land in
    different groups with different tables. One entry for the whole stack therefore slices one
    family out of the other family's table, and `inc-glm53f-051`'s interface record says
    nothing below this point detects it. Review r1 of commit 1 found it.

    THE CONTROL IS THE OLD BEHAVIOUR, RUN. The last block calls the same converter with the
    single-table mapping commit 1 effectively used and requires the sparse slice to land
    somewhere ELSE. Remove the per-layer lookup and the two calls agree, and this item fails.
    """
    _require_cpu_mode()
    root = _fixture()["root"]
    spec = _mixed_spec(root)
    monkeypatch.setattr(root, "get_kv_spec", lambda: spec)
    caches = _runner_shaped_caches(root)
    root.bind_kv_cache(caches)
    banks = root.glm5next_layer_banks
    families = sorted({bank["family"] for bank in banks})
    print(f"TINYE2E|groups|layers={len(banks)}|families={families}")
    if families != ["linear_attn", "self_attn"]:
        raise item.VacuousControlError(
            f"this item needs BOTH families in one stack for two groups to exist at all, "
            f"and the spec it built reports {families}"
        )

    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = root
    runner.max_model_len = E2E_MAX_SEQ_LEN
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
            want = bank["latent_cache"][sparse_row[0] * page]
            print(f"TINYE2E|group_slice|{index}|sparse|first_block={sparse_row[0]}"
                  f"|slots={int(carrier['latent_cache'].shape[0])}")
            assert carrier["latent_cache"].data_ptr() == want.data_ptr(), (
                f"layer {index} is sparse and its latent slice does not start at its own "
                f"group's first block"
            )
            assert int(carrier["latent_cache"].shape[0]) == len(sparse_row) * page
        else:
            print(f"TINYE2E|group_slice|{index}|recurrent|state_slot={state_slot}")
            assert (carrier["conv_state"].data_ptr()
                    == bank["conv_state"][state_slot].data_ptr()), (
                f"layer {index} is recurrent and its conv state is not the slot its own "
                f"group's table names"
            )
            assert (carrier["recurrent_state"].data_ptr()
                    == bank["recurrent_state"][state_slot].data_ptr())

    # ---- A layer with no entry of its own refuses by name rather than borrowing one.
    short = dict(grouped)
    short.pop(banks[0]["name"])
    with pytest.raises(ValueError, match="has no attention-metadata entry"):
        runner._glm5next_model_kwargs(
            _generic(tokens=tokens, metadata=short, sampling_row=tokens - 1)
        )

    # ---- A group whose page disagrees with the bank's own paging refuses rather than slicing.
    wrong_page = _grouped_metadata(banks, tokens=tokens, sparse_row=sparse_row,
                                   state_row=[state_slot],
                                   sparse_block_size=int(item.MLA_PAGE_SIZE) * 2)
    with pytest.raises(ValueError, match="reports page"):
        runner._glm5next_model_kwargs(
            _generic(tokens=tokens, metadata=wrong_page, sampling_row=tokens - 1)
        )

    # ---- Groups that disagree about the step refuse: the layers are stepped together.
    disagreeing = _grouped_metadata(banks, tokens=tokens, sparse_row=sparse_row,
                                    state_row=[state_slot], state_cached=1)
    with pytest.raises(ValueError, match="must agree on the leg and the cached length"):
        runner._glm5next_model_kwargs(
            _generic(tokens=tokens, metadata=disagreeing, sampling_row=tokens - 1)
        )

    # ---- THE CONTROL: one table for the whole stack lands the sparse slice elsewhere.
    single = {
        bank["name"]: _entry(row=[state_slot], tokens=tokens, cached=0, threshold=1,
                             block_size=item.MLA_PAGE_SIZE)
        for bank in banks
    }
    control = runner._glm5next_model_kwargs(
        _generic(tokens=tokens, metadata=single, sampling_row=tokens - 1)
    )
    sparse_index = next(i for i, bank in enumerate(banks) if bank["family"] == "self_attn")
    per_layer_ptr = carriers[sparse_index]["latent_cache"].data_ptr()
    single_ptr = control["layer_carriers"][sparse_index]["latent_cache"].data_ptr()
    print(f"TINYE2E|group_control|per_layer={per_layer_ptr}|single_table={single_ptr}"
          f"|differ={per_layer_ptr != single_ptr}")
    assert per_layer_ptr != single_ptr, (
        "the single-table mapping produced the same sparse slice as the per-layer one, so "
        "this item is not measuring the per-layer lookup at all"
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 9. the root's `block_size` is the weight-quant block, and the converter leaves it unset.
# ══════════════════════════════════════════════════════════════════════════════════════


def test_the_converter_does_not_hand_the_root_a_kv_page_as_its_quant_block():
    """Two different numbers share one name, and only one of them belongs to the root.

    THE ROOT'S `block_size` IS THE FP8 WEIGHT-QUANT BLOCK, "tokens per block, forwarded to the
    expert bank unread" (`model_fp8.py:8386`), and the bank refuses it unless it is a positive
    multiple of `BLOCK_QUANT_SIZE` (`model_fp8.py:1775-1780`). The KV page size is a different
    number -- 4 in this fixture -- so passing it would raise `Glm5NextBlockQuantRouteError` on
    the first routed layer. Unset, the bank uses its own declared block, which is the value
    `-054a`'s landed acceptance asserts the root forwards to its stack
    (`test_tiny_glm5next_forward.py:5392-5393`).

    THE CONTROL IS THE ARITHMETIC, and it is the same constant the model's own guard imports:
    `item.BLOCK_QUANT_SIZE` comes from `blockwise_fp8_retile`, which is where
    `model_fp8.py:1676-1680` gets it. This item requires the KV page NOT to be a multiple of
    it, so the wrong value would have been refused rather than silently tolerated; if a later
    dial made the two coincide, this item fails and says so instead of going quiet.
    """
    _require_cpu_mode()
    root = _fixture()["root"]
    caches = _runner_shaped_caches(root)
    root.bind_kv_cache(caches)
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = root
    runner.max_model_len = E2E_MAX_SEQ_LEN

    translated = _model_kwargs(
        runner,
        input_ids=torch.zeros(item.STACK_TOKENS, dtype=torch.long),
        cached=0,
        sampling_row=item.STACK_TOKENS - 1,
    )
    declared = inspect.signature(type(root).forward).parameters
    print(f"TINYE2E|root_params|declared={sorted(declared)}|translated={sorted(translated)}")
    assert set(translated) <= set(declared), (
        f"the converter returned {sorted(set(translated) - set(declared))}, which the root "
        f"forward does not declare"
    )
    assert "block_size" not in translated
    assert declared["block_size"].default is None

    page = int(item.MLA_PAGE_SIZE)
    quant = int(item.BLOCK_QUANT_SIZE)
    print(f"TINYE2E|quant_block|kv_page={page}|quant_block={quant}"
          f"|page_is_a_multiple={page % quant == 0}")
    if page % quant == 0:
        raise item.VacuousControlError(
            f"the KV page {page} IS a multiple of the quant block {quant}, so passing the "
            f"page as the root's block_size would not be refused and this item's control is "
            f"vacuous"
        )

    for index, (bank, carrier) in enumerate(
        zip(root.glm5next_layer_banks, translated["layer_carriers"])
    ):
        assert int(carrier["page_size"]) == int(bank["block_size"]), (
            f"layer {index}'s carrier carries page {int(carrier['page_size'])} and its bank "
            f"is paged {int(bank['block_size'])}; the KV page must still reach the layers"
        )


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 10. the side caches live across steps, and a fresh sequence starts on an empty ring.
# ══════════════════════════════════════════════════════════════════════════════════════


def test_the_side_caches_live_across_steps_and_a_fresh_sequence_clears_the_ring():
    """One set of side caches per process, and a new sequence must not inherit the old ring.

    WHAT IS MEASURED, AND WHY IT IS NOT A STYLE POINT. `_glm5next_live_side_caches` allocates
    the pooled store and the decode ring ONCE and hands the same objects to every step
    (`neuron_model_runner.py:4870-4889`); that identity is what makes a decode step's ring
    survive into the next step. Its cost is that a NEW sequence would start on the previous
    sequence's partial pool -- every real token stashes into the ring
    (`vllm_neuron/functional/dsa/decode_tail_update.py`) -- and its next completion would pool
    those stale members with no tolerance and no refusal anywhere below. A prefill at position
    0 is a new sequence, so the converter clears the ring there.

    THE POOLED STORE IS NOT CLEARED, and this item requires that too. The candidate gather is
    bounded by this sequence's own `max_seq_len` (`model_fp8.py:5288-5295`) and every complete
    pool below that bound is written by the prefill, so a row above the bound is unreachable.
    The planted row above the bound must SURVIVE, so a future blanket clear of both caches
    fails here and has to argue for itself.

    THE CONTROL IS THE DECODE LEG: the same planted ring, threaded through a decode step, must
    come back UNCLEARED. A converter that zeroed the ring on every call would satisfy the first
    conjunct and fail this one, so the item cannot pass by clearing too much.

    WHAT ITEM 11 MEASURES INSTEAD, and this item deliberately does not: the prefill's own
    remainder. The rows past the last complete pool exist only inside the model's prefill
    branch (`:5486-5500`), which now seeds them into the ring the converter binds with
    `seed_tail` (`model_fp8.py:4631-4638`); a `tail` passed to that forward would select the
    decode leg (`:5449-5470`), so the ring arrives as `prefill_tail` instead. This item's
    prompt divides evenly -- `STACK_TOKENS` is 128 and `MLA_INDEX_KPOOL` is 4 -- so the
    seeding is invisible here by construction; item 11 uses a prompt two rows past a pool
    boundary and pools those rows on its first decode step.
    """
    _require_cpu_mode()
    root = _fixture()["root"]
    caches = _runner_shaped_caches(root)
    root.bind_kv_cache(caches)
    banks = root.glm5next_layer_banks
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = root
    runner.max_model_len = E2E_MAX_SEQ_LEN

    live = runner._glm5next_live_side_caches(banks)
    again = runner._glm5next_live_side_caches(banks)
    rings = [side for side in live if "tail" in side]
    print(f"TINYE2E|side_caches|same_set={live is again}|entries={len(live)}"
          f"|with_a_ring={len(rings)}")
    assert live is again, "the side caches must be one set per process, not one per step"
    assert all(a["tail"] is b["tail"] for a, b in zip(rings, [s for s in again if "tail" in s]))
    if not rings:
        raise item.VacuousControlError(
            "no bank in this stack carries a ring, so this item measures nothing"
        )

    above_the_bound = int(rings[0]["pool_cache"].shape[0]) - 1
    candidates = item.STACK_TOKENS // int(root.text_config.index_kpool)
    print(f"TINYE2E|pool_rows|planted_row={above_the_bound}|this_sequences_candidates="
          f"{candidates}")
    if above_the_bound <= candidates:
        raise item.VacuousControlError(
            f"row {above_the_bound} is not above this sequence's candidate bound "
            f"{candidates}, so planting there would be reachable and the second conjunct "
            f"would be measuring the wrong thing"
        )
    for side in rings:
        side["tail"].fill_(3.0)
        side["pool_cache"][above_the_bound].fill_(5.0)

    _model_kwargs(
        runner,
        input_ids=torch.zeros(item.STACK_TOKENS, dtype=torch.long),
        cached=0,
        sampling_row=item.STACK_TOKENS - 1,
    )
    cleared = max(float(side["tail"].abs().max()) for side in rings)
    stale = min(float(side["pool_cache"][above_the_bound].abs().max()) for side in rings)
    print(f"TINYE2E|fresh_prefill|ring_max={cleared}|planted_pool_row_min={stale}")
    assert cleared == 0.0, (
        "a prefill at position 0 is a new sequence and must start on an empty ring; this one "
        "inherited the planted state"
    )
    assert stale == 5.0, (
        "the pooled store must not be blanket-cleared: the planted row is above this "
        "sequence's candidate bound and therefore unreachable, and clearing it would hide a "
        "bound defect instead of exposing one"
    )

    # ---- THE CONTROL: the decode leg keeps the ring it was handed. The ring IS the state.
    for side in rings:
        side["tail"].fill_(7.0)
    _model_kwargs(
        runner,
        input_ids=torch.zeros(1, dtype=torch.long),
        cached=item.STACK_TOKENS,
        sampling_row=0,
    )
    kept = min(float(side["tail"].abs().max()) for side in rings)
    print(f"TINYE2E|decode_step|ring_min={kept}")
    assert kept == 7.0, (
        "a decode step must not clear the ring; the ring is the decode leg's own state and "
        "clearing it would lose the partial pool this step is meant to advance"
    )


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 11. a prompt that leaves a remainder: the ring is seeded and the next pool is whole.
# ══════════════════════════════════════════════════════════════════════════════════════

#: A prompt that does NOT divide into pools, and the same prompt extended to the next pool
#: boundary. Both are DERIVED from the landed dials, so a change to either moves them.
#:
#: BOTH FIT THROUGH THE DSA PATH IN ONE LEG, which is why they sit BELOW the stack's token
#: count rather than above it. The `-103` causal-bound kernel binds the query-token axis to one
#: partition tile, so a prefill of more than `DSA_TOKENS_PER_CALL` tokens raises inside the
#: kernel (`dma_copy dst partition dimension 132 exceeds maximum 128`, measured on the host and
#: read back in `054b-r4-accept-20260909T210355Z.out:623`). Widening either dial is not this
#: item's to do: the kernel increment is the DSA lane's, and until it lands every counted item
#: keeps prompt and extension inside the ceiling.
DSA_TOKENS_PER_CALL = 128
REMAINDER_PROMPT = item.STACK_TOKENS - 2
EVEN_PROMPT = item.STACK_TOKENS


def _the_row_scale_bound(reference: torch.Tensor) -> float:
    """The registered pair read at the row's scale: atol plus rtol times the row's largest value.

    ONE definition. The pass side of the remainder item, its must-fail control and the synthetic
    proof below all call this, so none of the three can drift from the others. The expression is
    the one the control already carried before this increment; neither constant changed.
    """
    return 1e-5 + 1e-2 * float(reference.abs().max())


def _the_bf16_step(scale: float) -> float:
    """The gap between neighbouring bf16 values at `scale`, derived rather than written down.

    bfloat16 stores seven mantissa bits, so `torch.finfo(torch.bfloat16).eps` IS the gap at 1.0,
    and the gap at any other magnitude is that eps shifted by the magnitude's exponent. Reading it
    out of `torch.finfo` keeps the number a property of the dtype: if the ring's dtype ever
    changes, this follows it instead of asserting a dtype from memory. Writing the power of two by
    hand here would also have hidden the mistake this increment corrects.
    """
    eps = float(torch.finfo(torch.bfloat16).eps)
    if scale == 0.0:
        return eps
    return math.ldexp(eps, math.frexp(scale)[1] - 1)


def _an_orthonormal_hadamard(width: int) -> torch.Tensor:
    """A real Hadamard matrix of `width`, normalised so it is orthonormal.

    Built by the Sylvester doubling, which is the same construction the kernel's butterfly walks.
    It is here so the synthetic proof can put a planted difference through an ACTUAL rotation
    instead of reasoning about one.
    """
    h = torch.ones(1, 1)
    while h.shape[0] < width:
        top = torch.cat([h, h], dim=1)
        bottom = torch.cat([h, -h], dim=1)
        h = torch.cat([top, bottom], dim=0)
    if h.shape[0] != width:
        raise item.VacuousControlError(
            f"a Hadamard of width {width} does not exist by doubling; the proof below would "
            f"rotate by the wrong matrix"
        )
    return h / math.sqrt(width)


def test_the_row_scale_bound_admits_one_rounding_and_refuses_a_pool_built_short():
    """The bound the remainder item uses must admit one bf16 rounding and refuse a short pool.

    WHY THIS ITEM EXISTS. The remainder item compares two routes that differ by one extra rounding
    to bf16, and it therefore reads its allowance at the row's scale rather than cell by cell. A
    comparison shaped that way is only worth having if it still refuses the thing the item guards
    against -- a pool completed with zeros where the prefill should have seeded real keys. Both
    directions are proved here on planted rows, where the answer is known before the run, instead
    of being argued from the wording of an assertion.

    THE FLOOR IS PROVED HERE TOO, INCLUDING THROUGH A ROTATION. The control applies a floor
    derived from the share of pool members an emptied ring removes, and that derivation leans on
    the rotation after the pooling being orthogonal -- it preserves the loss's length but may
    spread it across the row, and the least a spread-out vector's largest cell can be is its length
    over the square root of its width. The last block plants a loss, rotates it by a real Hadamard
    and checks the floor survives, because an argument about a rotation is not a measurement of
    one.

    NO MODEL RUNS HERE. Planted tensors only, so this item is fast and cannot pass by accident on a
    stack that is not exercising the seam.
    """
    reference = torch.arange(1, 129, dtype=torch.bfloat16).float()
    width = int(reference.numel())
    scale = float(reference.abs().max())
    bound = _the_row_scale_bound(reference)
    step = _the_bf16_step(scale)
    print(f"TINYE2E|bound_proof|width={width}|scale={scale:.6g}|bound={bound:.6g}"
          f"|bf16_step={step:.6g}")
    if scale == 0.0 or bound == 0.0 or step == 0.0:
        raise item.VacuousControlError(
            f"this proof needs a non-zero row, bound and step; it has scale={scale}, "
            f"bound={bound}, step={step}, so every comparison below would be vacuous"
        )

    # ONE ROUNDING MUST PASS. Half a step is exactly the worst a round-to-nearest can do, and one
    # extra bf16 quantisation of the pooled vector is exactly one round-to-nearest.
    rounded = reference + step / 2
    delta = float((rounded - reference).abs().max())
    print(f"TINYE2E|bound_proof|one_rounding|max_abs_delta={delta:.6g}|bound={bound:.6g}"
          f"|bf16_steps={delta / step:.6g}|under_by={bound / delta:.6g}x")
    assert delta <= bound, (
        "the row-scale bound refuses a row that differs from its reference by the worst a single "
        "bf16 rounding can do, so the remainder item would redden on a faithful kernel"
    )

    # A POOL BUILT SHORT MUST FAIL, at every share of missing members this fixture could produce.
    for missing, total in ((1, 4), (2, 4), (3, 4)):
        share = missing / total
        delta_vector = reference * share
        delta = float(delta_vector.abs().max())
        floor = share * scale / math.sqrt(width) / bound
        over_by = delta / bound
        print(f"TINYE2E|bound_proof|missing_{missing}_of_{total}|max_abs_delta={delta:.6g}"
              f"|bound={bound:.6g}|over_by={over_by:.6g}x|floor={floor:.6g}x"
              f"|bf16_steps={delta / step:.6g}")
        assert delta > bound, (
            "a pool missing a share of its members matched its reference inside the row-scale "
            "bound, so the remainder item's control would pass with the seeding removed"
        )
        assert over_by >= floor, (
            "a pool missing a share of its members missed the bound by less than that share "
            "predicts, so the floor the control applies is not derived correctly"
        )

        # AND IT MUST STILL FAIL AFTER THE ROTATION, which is where the floor's square-root term
        # comes from. The rotation is orthonormal, so the loss keeps its length and can only
        # spread; the floor allows for the worst spreading and must hold anyway.
        rotated = delta_vector @ _an_orthonormal_hadamard(width)
        rotated_max = float(rotated.abs().max())
        print(f"TINYE2E|bound_proof|missing_{missing}_of_{total}_rotated"
              f"|max_abs_delta={rotated_max:.6g}|bound={bound:.6g}"
              f"|over_by={rotated_max / bound:.6g}x|floor={floor:.6g}x"
              f"|length_before={float(delta_vector.norm()):.6g}"
              f"|length_after={float(rotated.norm()):.6g}")
        assert rotated_max / bound >= floor, (
            "after an orthonormal rotation the planted loss no longer missed the bound by the "
            "floor the control applies, so that floor would redden a faithful run"
        )


def test_the_prefill_remainder_is_seeded_and_the_next_pool_completes_whole():
    """The tokens after a prompt's last complete pool must reach the ring the decode leg pools.

    THE ROW THIS ITEM MEASURES. A prompt of {REMAINDER} tokens completes pools 0..{LAST}
    and leaves two positions over. The decode leg pools from the ring
    (`vllm_neuron/functional/dsa/decode_tail_update.py`), so unless the prefill stashed those
    two positions there, the pool that completes on the second decode step is built from zeros
    in their place -- wrong candidate keys for every later token, with no tolerance and no
    refusal below to catch it.

    HOW IT IS CHECKED WITHOUT A NEW ORACLE. Two routes over the SAME token ids: one prefill of
    {EVEN} tokens, which pools that block inside the prefill seam, against a prefill of
    {REMAINDER} plus two decode steps, which pools it out of the ring. The pooled row they both
    write is compared at the registered pair, `rtol=1e-2` / `atol=1e-5` (plan `:1183`); no new
    constant is introduced here. The two routes do not use one kernel -- `decode_tail_update`
    records that its completion is not bit-identical to the prefill kernel's on the same pool,
    two bf16 round trips against one -- which is why the registered tolerance and not equality.

    THE PAIR IS READ AT THE ROW'S SCALE, NOT CELL BY CELL. The control below already read it that
    way and the pass side did not, so the two sides were not measuring one predicate; both now
    call `_the_bound`. The reason is measured rather than preferred: the extra round trip
    quantises the pooled vector BEFORE the 128-point Hadamard rotation
    (`decode_tail_update.py:370`, then `:379`), and that rotation mixes every input cell into
    every output cell, so one rounding at the pooled vector's own magnitude lands in all 128
    output cells at the size of the WHOLE ROW. A cell that cancels to near zero therefore cannot
    meet a per-cell relative tolerance however correct the kernel is. On the granted run the row
    agreed to HALF a bf16 step at its own magnitude -- 0.0078125 where the step at 2.65625 is
    0.015625, against an allowance of 0.0265725, so 3.4 times under -- while ten cells sitting 618
    times below the row's scale missed a per-cell allowance of 5.3e-05
    (`increments/investigate-054b-red2-item11-r1.md`). Half a step is exactly the worst a single
    round-to-nearest can do, which is the tightest signature one extra quantisation can leave;
    bfloat16 stores seven mantissa bits, and both printed rows below derive the step from
    `torch.finfo` rather than restating a power of two.

    THE CONTROL IS A MUST-FAIL. The last block runs route B again and empties the ring after
    the prefill, which is exactly "seeding removed". The completed pool must then MISS the
    reference by more than the tolerance. If the seeding stopped happening, that block would
    pass quietly and this item would fail, which is the direction an acceptance row has to
    fail in.

    THE DIRECT CONJUNCT is in the middle: after the prefill, the ring's low slots must be
    non-zero and its high slots -- the positions the prompt never reached -- must still be
    zero. That is the seeding itself, not its consequence.
    """
    _require_cpu_mode()
    root = _fixture()["root"]
    caches = _runner_shaped_caches(root)
    root.bind_kv_cache(caches)
    banks = root.glm5next_layer_banks
    pool = int(root.text_config.index_kpool)
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = root
    runner.max_model_len = E2E_MAX_SEQ_LEN

    remainder = REMAINDER_PROMPT % pool
    completed_pool = (EVEN_PROMPT - 1) // pool
    print(f"TINYE2E|remainder_case|prompt={REMAINDER_PROMPT}|even={EVEN_PROMPT}|pool={pool}"
          f"|remainder={remainder}|completing_position={EVEN_PROMPT - 1}"
          f"|pool_id={completed_pool}|dsa_ceiling={DSA_TOKENS_PER_CALL}"
          f"|margin={DSA_TOKENS_PER_CALL - EVEN_PROMPT}")
    if remainder == 0 or EVEN_PROMPT % pool != 0 or EVEN_PROMPT > E2E_MAX_SEQ_LEN:
        raise item.VacuousControlError(
            f"this item needs a prompt that leaves a remainder ({REMAINDER_PROMPT} % {pool} "
            f"= {remainder}), an extension that divides evenly ({EVEN_PROMPT} % {pool} = "
            f"{EVEN_PROMPT % pool}) and both inside {E2E_MAX_SEQ_LEN} slots"
        )
    if max(REMAINDER_PROMPT, EVEN_PROMPT) > DSA_TOKENS_PER_CALL:
        # The kernel's limit, refused here rather than 300 lines down inside NKI: the DSA path
        # takes the query-token axis as one partition tile, so a longer leg cannot run at all.
        raise item.VacuousControlError(
            f"this item prefills {REMAINDER_PROMPT} and {EVEN_PROMPT} tokens in one leg, and "
            f"the DSA path takes at most {DSA_TOKENS_PER_CALL} per call; a longer leg raises "
            f"inside the causal-bound kernel instead of measuring the seeding"
        )

    ids = torch.arange(EVEN_PROMPT, dtype=torch.long) % int(root.text_config.vocab_size)
    side = runner._glm5next_live_side_caches(banks)
    rings = [entry for entry in side if "tail" in entry]
    if not rings:
        raise item.VacuousControlError("no bank in this stack carries a ring")

    def _rows() -> list[torch.Tensor]:
        return [entry["pool_cache"][completed_pool] for entry in rings]

    def _clear_the_row() -> None:
        for row in _rows():
            row.zero_()

    def _the_bound(reference: torch.Tensor) -> float:
        """The registered pair read at the row's scale, from the module-level definition.

        The control at the bottom of this item was already written this way; the pass side was
        not, and that gap is what this closes. A control that tests a different predicate from the
        thing it controls proves nothing, so the two sides now cannot drift apart: the pass side
        asserts the miss is at most this bound, the control asserts it exceeds the same bound, and
        the synthetic proof above checks both directions on planted rows -- all three reading one
        expression, written once, in `_the_row_scale_bound`.
        """
        return _the_row_scale_bound(reference)

    # ---- Route A: one prefill over the whole block, pooled inside the prefill seam.
    _clear_the_row()
    root(**_model_kwargs(runner, input_ids=ids, cached=0, sampling_row=EVEN_PROMPT - 1))
    want = [row.clone().float() for row in _rows()]
    reference_scale = max(float(row.abs().max()) for row in want)
    print(f"TINYE2E|route_a|pool_id={completed_pool}|max_abs={reference_scale:.6g}")
    if reference_scale == 0.0:
        raise item.VacuousControlError(
            "the all-prefill route wrote nothing into the pool row this item compares, so "
            "every comparison below would pass against zeros"
        )

    # ---- Route B: prefill the odd prompt, then step through the remainder.
    _clear_the_row()
    root(**_model_kwargs(runner, input_ids=ids[:REMAINDER_PROMPT], cached=0,
                         sampling_row=REMAINDER_PROMPT - 1))
    low = min(float(entry["tail"][0, :remainder].abs().max()) for entry in rings)
    high = max(float(entry["tail"][0, remainder:].abs().max()) for entry in rings)
    print(f"TINYE2E|seeded_ring|low_slots_min={low:.6g}|high_slots_max={high:.6g}"
          f"|seeded_slots={remainder}")
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
    passing_deltas = []
    for index, (row, reference) in enumerate(zip(_rows(), want)):
        delta = float((row.float() - reference).abs().max())
        bound = _the_bound(reference)
        passing_deltas.append(delta)
        under_by = (bound / delta) if delta > 0.0 else float("inf")
        step = _the_bf16_step(float(reference.abs().max()))
        print(f"TINYE2E|pool_row|{index}|max_abs_delta={delta:.6g}|tolerance_bound={bound:.6g}"
              f"|under_by={under_by:.6g}x|bf16_step={step:.6g}"
              f"|bf16_steps={delta / step:.6g}|rtol=1e-2|atol=1e-5")
        assert delta <= bound, (
            "the pool the decode leg completed misses the all-prefill reference by more than the "
            "registered pair allows at this row's scale, which is a wrong pool rather than the "
            "one extra bf16 round trip the decode kernel faithfully performs"
        )

    # ---- THE CONTROL, must fail: the same route with the seeding taken back out.
    _clear_the_row()
    root(**_model_kwargs(runner, input_ids=ids[:REMAINDER_PROMPT], cached=0,
                         sampling_row=REMAINDER_PROMPT - 1))
    for entry in rings:
        entry["tail"].zero_()
    for step in range(remainder):
        position = REMAINDER_PROMPT + step
        root(**_model_kwargs(runner, input_ids=ids[position:position + 1], cached=position,
                             sampling_row=0))
    for index, (row, reference) in enumerate(zip(_rows(), want)):
        delta = float((row.float() - reference).abs().max())
        bound = _the_bound(reference)
        passing = passing_deltas[index]
        over_by = (delta / bound) if bound > 0.0 else float("inf")
        against_the_passing_route = (delta / passing) if passing > 0.0 else float("inf")
        # THE FLOOR IS DERIVED, NOT CHOSEN, AND THAT IS WHY IT CANNOT BE TUNED. Emptying the ring
        # turns `remainder` of the pool's `pool` members into zeros, so the pooled vector loses
        # that share of its mass. The rotation that follows is orthonormal: it keeps the loss's
        # length and can only spread it across the row, and the least a spread-out vector's
        # largest cell can be is its length over the square root of the row's width. Dividing by
        # the bound already in hand turns that into "how many times past the bound". Every term is
        # a model dial or a quantity already computed here -- no number is written on this line --
        # so the floor moves when the fixture moves, and there is no knob to turn when a run comes
        # back red. `test_the_row_scale_bound_admits_one_rounding_and_refuses_a_pool_built_short`
        # checks the formula at three shares and through a real Hadamard rotation.
        share = remainder / pool
        floor = share * float(reference.abs().max()) / math.sqrt(reference.numel()) / bound
        print(f"TINYE2E|control|{index}|max_abs_delta={delta:.6g}|tolerance_bound={bound:.6g}"
              f"|over_by={over_by:.6g}x|discrimination_floor={floor:.6g}x"
              f"|share_of_the_pool_the_emptied_ring_removes={share:.6g}"
              f"|times_the_passing_route={against_the_passing_route:.6g}x")
        assert delta > bound, (
            "with the ring emptied after the prefill, the completed pool still matched the "
            "all-prefill reference inside the registered tolerance, so this item is not "
            "measuring the seeding at all"
        )
        assert over_by >= floor, (
            "with the ring emptied the completed pool did miss the reference, but by less than "
            "the share of the pool an emptied ring removes can account for; the control fires, "
            "yet by so little that it no longer separates a missing pool member from a rounding. "
            "That is a finding to report, not a bound to widen"
        )


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEMS 12-15 (`inc-glm53f-054f`). THE THREE REFUSALS THAT HAD NO TEST ARM, AND THE
# POSITION CURSOR THAT GIVES THE LIVE RING A SEQUENCE IDENTITY.
#
# WHY THESE FOUR ARE HERE AND NOT IN `-054b`. The three refusals were written by `-054b`
# and nothing measured them, so nothing would have noticed if one stopped firing. The
# cursor is new: the side caches are keyed by absolute position and carry no sequence
# identity, so a fresh request admitted at a non-zero cached length -- an automatic
# prefix-cache hit -- would read the previous request's rows with no error at all.
#
# EVERY ARM CARRIES ITS OWN MUST-FAIL CONTROL, and for a refusal the control is the
# NEGATIVE one: the same call with the offending argument removed must NOT produce that
# refusal. Without it an arm passes whenever the call fails for any reason, which is the
# way a refusal test rots. `item.VacuousControlError` fires when a control cannot
# separate the two cases at all.
# ══════════════════════════════════════════════════════════════════════════════════════

MULTI_TOKEN_DECODE = 3
STALE_DECODE_GAP = 3
PLANTED_RING = 9.0


def _sparse_indexer(layers):
    """The first layer's DSA indexer, reached the way this file already reaches it."""
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        indexer = getattr(attention, "indexer", None)
        if indexer is not None:
            return indexer
    raise item.VacuousControlError(
        "this fixture exposes no DSA indexer, so the two prefill_tail refusals cannot be "
        "reached and the arms below would pass without measuring anything"
    )


def _live_rings(runner, banks):
    return [side for side in runner._glm5next_live_side_caches(banks) if "tail" in side]


def test_the_decode_leg_refuses_a_step_carrying_more_than_one_token():
    """ITEM 12. A decode step with more than one token is refused BY NAME.

    THE REFUSAL EXISTS BECAUSE THE SEAM IS SINGULAR: the decode leg advances the ring one
    position per call, taking one key row and one position. A multi-token decode is
    speculative decoding's verify step, which this half does not implement, and threading
    one would advance the ring once for several tokens.

    THE CONTROL IS THE NEGATIVE ONE. The same call carrying a single token must not raise
    this refusal. Without that half, the arm would pass on any failure of the builder --
    a wrong geometry, a bad slot -- and would stop measuring the token count entirely.
    """
    _require_cpu_mode()
    fixture = _fixture()
    root = fixture["root"]
    root.bind_kv_cache(_runner_shaped_caches(root))
    banks = root.glm5next_layer_banks
    text_config = root.text_config
    if MULTI_TOKEN_DECODE <= 1:
        raise item.VacuousControlError(
            f"this arm needs a decode carrying more than one token; it carries "
            f"{MULTI_TOKEN_DECODE}, which is what the refusal allows"
        )
    side = NeuronModelRunner._glm5next_side_caches(
        banks,
        index_kpool=int(text_config.index_kpool),
        index_head_dim=int(text_config.index_head_dim),
        max_seq_len=item.STACK_TOKENS,
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
    print(f"TINYE2E|refusal_multi_token_decode|tokens={MULTI_TOKEN_DECODE}|{message}")
    assert "threading a multi-token decode" in message, (
        f"a {MULTI_TOKEN_DECODE}-token decode raised, but not the refusal this arm names: "
        f"{message}"
    )
    assert f"{MULTI_TOKEN_DECODE} token(s)" in message, (
        "the refusal does not report the count it actually saw, so a reader cannot tell "
        "which step was refused"
    )

    # ---- THE CONTROL: one token on the same path must not raise THIS refusal.
    control = None
    try:
        build(1)
    except Exception as exc:  # noqa: BLE001 - the control reports whatever it gets
        control = f"{type(exc).__name__}: {exc}"
    print(f"TINYE2E|refusal_multi_token_decode_control|tokens=1|raised={control}")
    if control is not None and "threading a multi-token decode" in control:
        raise item.VacuousControlError(
            "a single-token decode raised the multi-token refusal too, so this arm is not "
            "measuring the token count and would pass with the guard removed"
        )


def test_the_indexer_refuses_a_prefill_ring_handed_to_a_decode_step():
    """ITEM 13. ``prefill_tail`` on a decode step is refused BY NAME.

    ONE CALL ADVANCES THE RING OR SEEDS IT, NEVER BOTH. The decode leg's ring is ``tail``
    and the prefill leg's is ``prefill_tail``; a call carrying both is asking the indexer
    to do two different things to one buffer.

    THE OPERANDS ARE DELIBERATELY MINIMAL, AND THAT IS SAFE HERE FOR A STATED REASON: the
    refusal is raised before the hidden states or the query latent are read at all, so
    passing real ones would add a shape this arm does not measure. The control is what
    keeps that honest -- with ``prefill_tail`` dropped, this refusal must not appear.
    """
    _require_cpu_mode()
    fixture = _fixture()
    root, layers = fixture["root"], fixture["layers"]
    root.bind_kv_cache(_runner_shaped_caches(root))
    banks = root.glm5next_layer_banks
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = root
    runner.max_model_len = E2E_MAX_SEQ_LEN
    indexer = _sparse_indexer(layers)
    rings = _live_rings(runner, banks)
    if not rings:
        raise item.VacuousControlError(
            "the live side caches carry no ring, so there is no prefill_tail to hand a "
            "decode step and this arm cannot reach its refusal"
        )
    ring = rings[0]["tail"]
    pool_cache = item._mla_pool_cache(pages=item.STACK_PAGES)
    selection = item._mla_selection_operands(tokens=1, pages=item.STACK_PAGES)
    minimal = torch.zeros(1, 1, dtype=pool_cache.dtype)

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
    print(f"TINYE2E|refusal_prefill_ring_on_decode|{type(caught.value).__name__}|{message}")
    assert "prefill_tail is the PREFILL leg's ring" in message, (
        f"the decode step carrying a prefill ring raised, but not the refusal this arm "
        f"names: {message}"
    )

    # ---- THE CONTROL: the same decode step without the prefill ring.
    control = None
    try:
        call()
    except Exception as exc:  # noqa: BLE001
        control = f"{type(exc).__name__}: {exc}"
    print(f"TINYE2E|refusal_prefill_ring_on_decode_control|raised={control}")
    if control is not None and "prefill_tail is the PREFILL leg's ring" in control:
        raise item.VacuousControlError(
            "the decode step raised the prefill-ring refusal even without a prefill ring, "
            "so this arm is not measuring the argument it names"
        )


def test_the_indexer_refuses_a_prefill_ring_with_no_end_position():
    """ITEM 14. ``prefill_tail`` without ``prefill_end_position`` is refused BY NAME.

    SEEDING THE RING NEEDS THE SEQUENCE LENGTH AFTER THIS CHUNK. The remainder positions
    a prefill seeds are the ones past its last complete pool, so the indexer cannot know
    WHICH slots to write without the end position. Seeding without it would write the
    wrong slots and the next completion would pool the wrong keys.

    THE CONTROL SUPPLIES THE END POSITION and requires this refusal to disappear, which is
    what separates "the end position was missing" from "the call failed".
    """
    _require_cpu_mode()
    fixture = _fixture()
    root, layers = fixture["root"], fixture["layers"]
    root.bind_kv_cache(_runner_shaped_caches(root))
    banks = root.glm5next_layer_banks
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = root
    runner.max_model_len = E2E_MAX_SEQ_LEN
    indexer = _sparse_indexer(layers)
    rings = _live_rings(runner, banks)
    if not rings:
        raise item.VacuousControlError(
            "the live side caches carry no ring, so there is no prefill_tail to seed and "
            "this arm cannot reach its refusal"
        )
    ring = rings[0]["tail"]
    pool_cache = item._mla_pool_cache(pages=item.STACK_PAGES)
    selection = item._mla_selection_operands(
        tokens=item.STACK_TOKENS, pages=item.STACK_PAGES
    )
    minimal = torch.zeros(1, 1, dtype=pool_cache.dtype)

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
    print(f"TINYE2E|refusal_seed_without_end_position|{type(caught.value).__name__}"
          f"|{message}")
    assert "a prefill_tail with no prefill_end_position" in message, (
        f"the seeding call raised, but not the refusal this arm names: {message}"
    )

    # ---- THE CONTROL: the same call with the end position supplied.
    control = None
    try:
        call(prefill_end_position=int(item.STACK_TOKENS))
    except Exception as exc:  # noqa: BLE001
        control = f"{type(exc).__name__}: {exc}"
    print(f"TINYE2E|refusal_seed_without_end_position_control|raised={control}")
    if control is not None and "a prefill_tail with no prefill_end_position" in control:
        raise item.VacuousControlError(
            "the seeding call still reported a missing end position after one was supplied, "
            "so this arm is not measuring that argument"
        )


def test_a_fresh_sequence_resets_the_cursor_so_two_requests_never_share_the_ring():
    """ITEM 15. The live ring carries a POSITION CURSOR, and a fresh sequence resets it.

    THE DEFECT THIS CLOSES. The side caches live for the process and are keyed by absolute
    position with no sequence identity. A fresh request that enters at a non-zero cached
    length -- what an automatic prefix-cache hit produces -- would be served from the last
    request's rows with nothing raised and nothing shaped wrongly. The cursor supplies the
    identity: it is the position the live ring has been advanced to.

    FOUR THINGS ARE MEASURED, and the last two are the control:
      1. a fresh prefill sets the cursor to the tokens it consumed, and a decode that
         continues it advances the cursor by one;
      2. a decode at a position the cursor does not name is REFUSED, and the refused step
         leaves the cursor untouched -- a refusal must leave no trace;
      3. a second fresh prefill resets the cursor and empties the ring, so the second
         request reads its own rows and not the planted ones;
      4. THE CONTROL, which is the must-fail direction: after that reset, the FIRST
         request's next position is refused. If it were still served, the ring would have
         kept the first sequence's identity and item 3 would be passing on a stale cursor.
    """
    _require_cpu_mode()
    root = _fixture()["root"]
    root.bind_kv_cache(_runner_shaped_caches(root))
    banks = root.glm5next_layer_banks
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = root
    runner.max_model_len = E2E_MAX_SEQ_LEN
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

    # ---- 1. the first sequence opens and advances.
    step(0, prompt)
    opened = int(runner._glm5next_side_cache_cursor)
    step(prompt, 1)
    advanced = int(runner._glm5next_side_cache_cursor)
    print(f"TINYE2E|cursor_first_sequence|opened={opened}|advanced={advanced}"
          f"|prompt={prompt}")
    assert opened == prompt, (
        f"a fresh prefill of {prompt} tokens left the cursor at {opened}; the cursor must "
        f"name the position the ring has been advanced to"
    )
    assert advanced == prompt + 1, (
        f"a decode that continued the sequence left the cursor at {advanced} rather than "
        f"{prompt + 1}, so the ring's recorded position no longer tracks the work done"
    )

    # ---- 2. a step the cursor does not name is refused, and leaves no trace.
    with pytest.raises(ValueError) as caught:
        step(stale, 1)
    message = str(caught.value)
    print(f"TINYE2E|cursor_refuses_a_stale_position|at={stale}|{message}")
    assert "stands at position" in message, (
        f"a decode at {stale} raised, but not the cursor refusal this arm names: {message}"
    )
    assert int(runner._glm5next_side_cache_cursor) == advanced, (
        "the refused step moved the cursor; a refusal must leave the ring's recorded "
        "position exactly as it was, or the next legitimate step is refused for a "
        "mismatch the refused one caused"
    )

    # ---- 3. a second fresh sequence resets the cursor and empties the ring.
    for side in _live_rings(runner, banks):
        side["tail"].fill_(PLANTED_RING)
    step(0, prompt)
    reset = int(runner._glm5next_side_cache_cursor)
    planted = max(float(side["tail"].abs().max()) for side in _live_rings(runner, banks))
    print(f"TINYE2E|cursor_second_sequence|reset={reset}|ring_max={planted}"
          f"|planted={PLANTED_RING}")
    assert reset == prompt, (
        f"the second fresh prefill left the cursor at {reset}; a prefill at position 0 is "
        f"a new sequence and must reopen the ring at its own length"
    )
    assert planted == 0.0, (
        "the second sequence started on the first sequence's planted ring, which is the "
        "silent cross-sequence read this item exists to refuse"
    )

    # ---- 4. THE CONTROL: the first sequence's next position is now refused.
    with pytest.raises(ValueError) as after:
        step(advanced, 1)
    control = str(after.value)
    print(f"TINYE2E|cursor_reset_control|first_sequence_next={advanced}|{control}")
    assert "stands at position" in control, (
        f"the first sequence's next position {advanced} was still served after a second "
        f"sequence opened, so the cursor did not reset and item 3 above would pass on a "
        f"stale identity: {control}"
    )
    assert str(reset) in control, (
        "the refusal does not report the position the ring actually stands at, so a reader "
        "cannot tell which sequence owns the ring"
    )
