"""``inc-glm53f-054b`` commit 1: the runner's caches reach the layers, and the layers write them.

WHAT THIS FILE MEASURES. This half of ``inc-glm53f-054`` threads the runner's allocated caches
into the per-layer carriers this model family takes as forward ARGUMENTS. Commit 1 lands the two
halves of that thread and this file measures both:

  1. ``Glm5NextForConditionalGeneration.bind_kv_cache`` -- the method the runner calls on every
     start-up (``neuron_model_runner.py:8669``) and that this package did not have until plan
     revision 276. It must map every layer the spec reports onto that layer's OWN slots, keep
     views rather than copies, and refuse by name anything it cannot map.
  2. ``NeuronModelRunner._glm5next_layer_carriers`` and its two operand derivations -- the runner
     side that turns those banks into one mapping per layer.

THE BIND IS EXERCISED, NOT ASSUMED. The last item runs the root's forward with carriers built by
the runner's own helpers out of a runner-shaped cache dict, and then requires the RUNNER'S OWN
TENSORS to have changed. A cache that arrived as a copy, or a bank handed to the wrong layer,
fails that conjunct. An item that built its own caches and never bound anything would pass while
the production path stayed broken, which is the reason this file exists in this shape.

WHAT THIS FILE DOES NOT MEASURE, stated so the gap is not read as coverage. The registered
acceptance for ``inc-glm53f-054b`` is eight generated tokens compared against a torch reference
built from the same weights (plan ``:1183``). That reference needs the decode leg, which commit 2
lands; commit 1's items below assert shape, finiteness and write-through, never a token value, and
this file carries no tolerance of its own. The recurrent (KDA) arm of the bind is measured on a
spec this file substitutes rather than on a KDA layer, because the landed tiny stack is
sparse-attention on every layer (``test_tiny_glm5next_forward.py:3822-3842`` reads
``layer.self_attn`` for all of them); the mapper reads no layer module, so the substitution
exercises the same code the runner drives.

HOW TO RUN IT, and both variables must be in the environment rather than set from a fixture
(``inc-glm53f-051``'s obligation 4):

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 \\
        python -m pytest -s -rA test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_e2e.py
"""

from __future__ import annotations

import os

import pytest
import torch

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

# `-054a`'s OWN fixture, dials and landed prefill operands. Imported, never re-implemented:
# this file measures the runner against what that item's acceptance ran, and the import order
# follows the landed convention (`test/vllm_neuron/functional/dsa/test_causal_bound.py:99-110`).
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as item

pytestmark = [pytest.mark.fast]

#: Pages of ``MLA_PAGE_SIZE`` slots, enough that one ascending run of them covers the stack
#: item's token count exactly. The number is DERIVED from the two landed dials rather than
#: typed, so a change to either moves this with it.
E2E_BLOCKS = item.STACK_TOKENS // item.MLA_PAGE_SIZE

#: The recurrent-state geometry the substituted spec reports. Small and arbitrary: the mapper
#: under test reads shapes off the spec and never off a layer, so these numbers only have to
#: be self-consistent between the spec and the banks this file allocates for it.
E2E_CONV_STATE_SHAPE = (2, 3)
E2E_RECURRENT_STATE_SHAPE = (2, 4, 4)
E2E_STATE_SLOTS = 8


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
    ``[blocks, num_kv_heads, block_size, head_size]`` (``neuron_model_runner.py:8529-8565``)
    and a recurrent layer gets two ``[slots, *state shape]`` banks in conv-then-recurrent
    order (``:8627-8657``). The geometry is read off the spec the model itself produced, so
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
    ``neuron_model_runner.py:8720-8726``), and it reads no layer module at all. So a spec
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
    root = item._root_fixture()["root"]
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
    root = item._root_fixture()["root"]
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
    root = item._root_fixture()["root"]
    caches = _runner_shaped_caches(root)
    absent = root.get_kv_spec().layers[0].name
    caches.pop(absent)
    with pytest.raises(ValueError, match=f"no entry for KV layer '{absent}'"):
        root.bind_kv_cache(caches)


def test_bind_kv_cache_refuses_a_bank_of_the_wrong_rank():
    """A bank that is not the runner's four-axis allocation refuses rather than being viewed."""
    _require_cpu_mode()
    root = item._root_fixture()["root"]
    caches = _runner_shaped_caches(root)
    name = root.get_kv_spec().layers[0].name
    flat = caches[name][0]
    caches[name] = [flat.reshape(-1), caches[name][1]]
    with pytest.raises(ValueError, match="the runner allocates"):
        root.bind_kv_cache(caches)


def test_bind_kv_cache_refuses_a_bank_whose_geometry_is_not_the_specs():
    """A head count or width the model did not ask for refuses, and says both numbers."""
    _require_cpu_mode()
    root = item._root_fixture()["root"]
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
    root = item._root_fixture()["root"]
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
    root = item._root_fixture()["root"]
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
    where ``candidates = max_seq_len // index_kpool`` (``model_fp8.py:5211-5218``). The
    allocator sits ON that boundary rather than over-allocating, so the assertions below
    are equalities in the derived row count; the guard itself is the instrument in item 5,
    where a store one row short raises inside the indexer instead of being asserted here.
    """
    _require_cpu_mode()
    root = item._root_fixture()["root"]
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
    fixture = item._root_fixture()
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
        block_ids=range(E2E_BLOCKS),
        state_slot=0,
        is_prefill=True,
        tokens=item.STACK_TOKENS,
        start_position=0,
        softmax_scale=item.MLA_SOFTMAX_SCALE,
        max_seq_len=item.STACK_TOKENS,
        page_size=item.MLA_PAGE_SIZE,
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
        assert set(got) == set(want), (
            f"layer {index}'s runner-built carrier carries {sorted(set(got))} and the "
            f"landed prefill carrier carries {sorted(set(want))}"
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
