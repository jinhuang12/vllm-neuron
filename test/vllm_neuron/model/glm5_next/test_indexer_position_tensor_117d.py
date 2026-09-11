# SPDX-License-Identifier: Apache-2.0
"""Acceptance -- the sparse indexer's decode step serves every position from one graph.

THE DECLARED ACCEPTANCE, Tier N, CPU mode:

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 python -m pytest \\
      test/vllm_neuron/model/glm5_next/test_indexer_position_tensor_117d.py \\
      -s -rA -p no:randomly -p no:cacheprovider

Three items, one test each, no ``parametrize`` and no skip.

WHAT THE BLOCK IS ABOUT. A decode graph is captured once and replayed at every
position. The indexer's two ring seams took the position as a python int, so the ring
row each one addressed was the row the capture happened at -- written again at every
later step, which is silent corruption rather than a recompile. Both entry points now
take the position as a tensor, the pool write is addressed on device, and the kernel's
own slot stays the compile-time constant it has to be, reached by rotating the ring.

WHAT THE ITEMS OBSERVE.

  A01  The two seam entry points, driven on SHAPE-ONLY tensors at position 0 and at
       position N: one shape each, and no value read. The shape-only device is the
       instrument -- there is no data behind it, so a host read of the position raises
       instead of returning a number that happens to be right.
  A02  The tensor route against the LANDED int route, at the five declared positions,
       as an equality and not a tolerance. The criterion's named reading is the emitted
       indices; the ring and the pooled store are the same claim's other two faces and
       are read the same way.
  A03  The decode leg's own source: no host read of the position is spelled there.

TWO OF THE THREE FAIL AT THE BASE, AND THE THIRD IS A REGRESSION EQUALITY. A01 fails
because the base's seams reach ``int(position)`` on a shape-only tensor, which has no
value to give; A03 fails because the base's decode leg spells two such reads. A02 PASSES
at the base, and that is not a gap: it hands the seams CPU 0-d tensors, and ``int()`` on
one RETURNS the number rather than raising, so the base coerces the tensor operand into
its own int route and A02's two sides become the same run. What A02 measures at the base
is therefore that the base agrees with itself; what it measures HERE is that the new
route agrees with the landed one. The block's must-fail arm names items (1) and (3) only.

WHAT THIS FILE DOES NOT DO. Nothing here captures a real graph or serves a real
request. The reading that one captured graph replays across positions is a host-side
end-to-end one on the serving run, and this file must not be read as making it.
"""

from __future__ import annotations

import ast
import inspect
import os
import textwrap
from pathlib import Path

import pytest
import torch

from test.vllm_neuron.model.glm5_next import test_dsa_layer as layer_half

# --------------------------------------------------------------------------- #
# The declared fixture. Every width is the landed sibling's, read from it rather
# than typed again, so this file cannot disagree with the geometry the landed
# items already run at.
# --------------------------------------------------------------------------- #
#: Tokens the prefill leg lays down before the decode steps are compared.
PREFILL_TOKENS = layer_half.PREFILL_TOKENS
#: Slots per pool. The ring's length and the modulus every address below uses.
POOL = layer_half.POOL_SIZE
#: Rows per page of the pooled-key store, and the store's own row count.
PAGE_SIZE = layer_half.PAGE_SIZE
POOL_ROWS = layer_half.PAGES * layer_half.PAGE_SIZE
#: The batch's longest sequence, held CONSTANT across every position compared below.
#: It is what sizes the candidate axis, so a per-position value would change the
#: regime under the comparison instead of changing only the position.
MAX_SEQ_LEN = PREFILL_TOKENS + 1
#: The positions item A02 compares: the first step, the step after it, the last slot
#: of a pool, the first slot of the next one, and the step that follows the prefill.
#: Two of them end a pool (``POOL - 1`` and ``PREFILL_TOKENS``) and three do not.
DECLARED_POSITIONS = (0, 1, POOL - 1, POOL, PREFILL_TOKENS)
#: The two positions item A01 derives shapes at.
LOW_POSITION = 0
HIGH_POSITION = PREFILL_TOKENS


def say(name: str, *values) -> None:
    """One reading per line, tagged so a launcher can anchor on it."""
    print("POSTENSOR|" + name + "|" + "|".join(str(value) for value in values))


def _require_cpu_mode() -> None:
    """The declared acceptance runs under VLLM_NEURON_CPU_MODE=1, so read it, not set it."""
    assert os.environ.get("VLLM_NEURON_CPU_MODE") == "1", (
        "the declared acceptance runs under VLLM_NEURON_CPU_MODE=1 and this process "
        "does not carry it, so nothing below would be measuring the declared mode"
    )


def _seam_module():
    """The ring seam's module, imported in a test body for the sibling file's reason."""
    from vllm_neuron.functional.dsa import decode_tail_update

    return decode_tail_update


def _shape_only_indexer():
    """An indexer with NO weight but its per-slot bias, on the shape-only device.

    ``tail_step`` and ``seed_tail`` read the bias and nothing else off the module, so
    materialising the four projections would be furniture.

    THE BIAS MUST BE A PARAMETER AND NOT A PLAIN TENSOR. Its name is reserved on the
    module by ``register_parameter(name, None)``, so it lives in the parameter registry
    from construction; ``nn.Module.__setattr__`` refuses a plain tensor on such a name
    and raises before this fixture returns. The landed sibling assigns the same bias the
    same way for the same reason.
    """
    indexer = layer_half._bare_indexer()
    indexer.index_kpool_compress_ape = torch.nn.Parameter(
        torch.zeros(POOL, int(indexer.index_head_dim), dtype=torch.bfloat16, device="meta"),
        requires_grad=False,
    )
    return indexer


def _shape_only_operands(indexer, *, tokens: int) -> dict:
    """One ring, one chunk of keys and one of gate scores, all shape-only."""
    head_dim = int(indexer.index_head_dim)
    return {
        "tail": torch.zeros(2, POOL, head_dim, dtype=torch.bfloat16, device="meta"),
        "key": torch.zeros(tokens, head_dim, dtype=torch.bfloat16, device="meta"),
        "gate": torch.zeros(tokens, head_dim, dtype=torch.bfloat16, device="meta"),
    }


def _full_indexer():
    """The landed fixture's own indexer, every leaf materialised, and its config."""
    stack, cfg, _gen = layer_half.build_layer_stack(layers=1)
    return stack[0].attention.indexer, cfg


def _candidate_count() -> int:
    """The addressable pool count the indexer will derive, recomputed from its closed form."""
    return MAX_SEQ_LEN // POOL


def _state(head_dim: int) -> dict:
    """A zeroed pooled-key store and a zeroed ring, the pair the legs write in place."""
    return {
        "pool_cache": torch.zeros(POOL_ROWS, head_dim, dtype=torch.bfloat16),
        "tail": torch.zeros(2, POOL, head_dim, dtype=torch.bfloat16),
    }


def _prefill(indexer, cfg, state: dict) -> None:
    """Lay down the prefill leg's pooled rows and its seeded remainder, once.

    Driven through the LANDED int route on purpose: both decode clones below have to
    start from a state neither of the two routes under comparison produced, or a
    difference in the seeding would be present on both sides and invisible.
    """
    gen = torch.Generator().manual_seed(700_101)
    hidden = torch.randn(
        PREFILL_TOKENS, int(cfg.hidden_size), generator=gen, dtype=torch.float32
    )
    q_latent = torch.randn(
        PREFILL_TOKENS, int(cfg.q_lora_rank), generator=gen, dtype=torch.float32
    )
    indexer(
        hidden,
        q_latent,
        state["pool_cache"],
        torch.arange(1, PREFILL_TOKENS + 1, dtype=torch.int32),
        max_seq_len=MAX_SEQ_LEN,
        page_size=PAGE_SIZE,
        slot_mapping=layer_half.prefill_slot_mapping(PREFILL_TOKENS, POOL),
        prefill_tail=state["tail"],
        prefill_end_position=PREFILL_TOKENS,
    )


def _decode_step(indexer, cfg, state: dict, position) -> torch.Tensor:
    """One decode step on ``state``, which it writes in place. Returns the emitted indices.

    ``position`` is handed over UNCHANGED -- a python int takes the landed route and a
    tensor takes the new one, and choosing between them is the whole comparison. The
    value is read here to build this row's own context length, which is legitimate: a
    test runs outside any traced region and the two runs must be handed the same length.
    """
    gen = torch.Generator().manual_seed(700_102)
    hidden = torch.randn(1, int(cfg.hidden_size), generator=gen, dtype=torch.float32)
    q_latent = torch.randn(1, int(cfg.q_lora_rank), generator=gen, dtype=torch.float32)
    at = int(position)
    return indexer(
        hidden,
        q_latent,
        state["pool_cache"],
        torch.tensor([at + 1], dtype=torch.int32),
        max_seq_len=MAX_SEQ_LEN,
        page_size=PAGE_SIZE,
        tail=state["tail"],
        position=position,
    )


def _worst(got: torch.Tensor, want: torch.Tensor) -> tuple[float, int]:
    """``(max_abs_diff, differing_elements)`` between two tensors of one shape."""
    assert tuple(got.shape) == tuple(want.shape), (
        f"{tuple(got.shape)} against {tuple(want.shape)}"
    )
    diff = (got.to(torch.float64) - want.to(torch.float64)).abs()
    return float(diff.max()) if diff.numel() else 0.0, int((diff != 0).sum())


# ══════════════════════════════════════════════════════════════════════════════
# A01. One shape at two positions, and no value read on the way there.
# ══════════════════════════════════════════════════════════════════════════════
def test_a01_the_two_ring_seams_derive_one_shape_at_two_positions_without_reading_a_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both entry points, at position 0 and at position N, on shape-only tensors.

    THE INSTRUMENT IS THE DEVICE. A shape-only tensor carries no data, so ``int()`` on
    one raises: reaching the end of both calls IS the reading that no value was read,
    and the shapes that come back are the reading that neither derived an extent from
    the position. Nothing here asserts a numeric answer -- there is none to assert.

    THE KERNEL GATE IS FORCED OFF FOR THIS ITEM, and the reason is not a shortcut: the
    simulator needs values and this item deliberately supplies none. What it observes
    sits AHEAD of the route split -- the rotation, the addresses and the masked write
    are computed before either route is chosen -- so the route the call ends up taking
    is not part of the reading. Item A02 exercises the route the product ships.
    """
    _require_cpu_mode()
    seam = _seam_module()
    monkeypatch.setattr(seam, "can_run_kernel", lambda *args, **kwargs: False)

    indexer = _shape_only_indexer()
    head_dim = int(indexer.index_head_dim)
    shapes: dict[str, list[tuple[int, ...]]] = {"pooled": [], "ring": [], "seed": []}

    for position in (LOW_POSITION, HIGH_POSITION):
        at = torch.tensor(position, dtype=torch.int32, device="meta")

        step = _shape_only_operands(indexer, tokens=1)
        pooled, new_tail = indexer.tail_step(
            step["tail"], step["key"], step["gate"], at
        )
        shapes["pooled"].append(tuple(pooled.shape))
        shapes["ring"].append(tuple(new_tail.shape))

        chunk = _shape_only_operands(indexer, tokens=PREFILL_TOKENS)
        written = indexer.seed_tail(
            chunk["tail"], chunk["key"], chunk["gate"], at
        )
        shapes["seed"].append(tuple(written.shape))

        say("A01_AT_POSITION", position, tuple(pooled.shape), tuple(new_tail.shape),
            tuple(written.shape), str(pooled.device), str(new_tail.device))

    say("A01_POOLED_SHAPES", *shapes["pooled"])
    say("A01_RING_SHAPES", *shapes["ring"])
    say("A01_COUNT_SHAPES", *shapes["seed"])

    assert shapes["pooled"][0] == shapes["pooled"][1], (
        "the pooled key's shape moved with the position, so the decode step's own "
        "output is not the same shape at two positions"
    )
    assert shapes["ring"][0] == shapes["ring"][1], (
        "the ring's shape moved with the position, which no captured graph survives"
    )
    assert shapes["seed"][0] == shapes["seed"][1]
    assert shapes["pooled"][0] == (1, head_dim)
    assert shapes["ring"][0] == (2, POOL, head_dim)
    assert shapes["seed"][0] == (), (
        "the written-row count came back with an extent, so it is not the 0-d tensor "
        "a caller can carry without reading it"
    )

    # The write address is the third thing the position used to fix, so it is read here
    # as a pair of shape-only tensors rather than left to A02's values alone.
    index, completes = _seam_module().decode_pool_address(
        torch.tensor(HIGH_POSITION, dtype=torch.int32, device="meta"),
        POOL,
        torch.device("meta"),
    )
    say("A01_ADDRESS", tuple(index.shape), index.dtype, tuple(completes.shape),
        completes.dtype)
    assert tuple(index.shape) == () and tuple(completes.shape) == ()
    assert completes.dtype == torch.bool


# ══════════════════════════════════════════════════════════════════════════════
# A02. The tensor route equals the landed int route, exactly.
# ══════════════════════════════════════════════════════════════════════════════
def test_a02_the_tensor_route_and_the_landed_int_route_agree_bit_for_bit() -> None:
    """Five declared positions, one prefill state, two decode runs each, no tolerance.

    THE NAMED READING IS THE EMITTED INDICES and it is asserted as ``0.0``. The ring and
    the pooled store are read the same way and for the same claim: they are what the
    position ADDRESSES, so an equality on the indices alone would pass a route that
    stashed the token in the wrong ring row on a step whose selection is bounded away
    from it.

    THE TRASH ROW IS EXCLUDED FROM THE STORE'S COMPARISON, and that is a declared
    difference rather than a blind spot. A step that ends no pool has no pooled key; the
    int route decides not to write and the tensor route cannot, so it writes to the row
    the prefill leg already sends its non-completions to. That row is above the
    addressable candidates, so nothing gathers it, and the difference is DISCLOSED below
    as a value at every position.

    THE PREFILL SEAM IS COMPARED AT THE SAME FIVE VALUES, in the second half. The chunk
    is one token there, so the landed route refuses the values below its own length --
    that refusal is asymmetric by design (the tensor route cannot read the value to
    refuse it) and the item records which values it covered rather than passing over the
    difference in silence.
    """
    _require_cpu_mode()
    say("A02_KERNEL_GATE_LIVE", layer_half.gate_live())
    indexer, cfg = _full_indexer()
    head_dim = int(indexer.index_head_dim)
    candidates = _candidate_count()
    assert candidates > int(indexer.select_k()), (
        f"{candidates} candidate pool(s) is not more than the {int(indexer.select_k())} "
        f"selected, so the indexer would take its causal bypass and this item would "
        f"compare two answers that never touched the ring"
    )

    seeded = _state(head_dim)
    _prefill(indexer, cfg, seeded)
    say("A02_PREFILL_STATE", "pool_rows_written",
        int((seeded["pool_cache"].to(torch.float32).abs().sum(dim=1) != 0).sum()),
        "ring_rows_written",
        int((seeded["tail"][0].to(torch.float32).abs().sum(dim=1) != 0).sum()))

    for position in DECLARED_POSITIONS:
        by_int = {name: value.clone() for name, value in seeded.items()}
        by_tensor = {name: value.clone() for name, value in seeded.items()}

        want = _decode_step(indexer, cfg, by_int, position)
        got = _decode_step(
            indexer, cfg, by_tensor, torch.tensor(position, dtype=torch.int32)
        )

        index_abs, index_n = _worst(got, want)
        ring_abs, ring_n = _worst(by_tensor["tail"], by_int["tail"])
        store_abs, store_n = _worst(
            by_tensor["pool_cache"][:candidates], by_int["pool_cache"][:candidates]
        )
        trash_abs, _trash_n = _worst(
            by_tensor["pool_cache"][POOL_ROWS - 1], by_int["pool_cache"][POOL_ROWS - 1]
        )
        say("A02_DECODE", f"position={position}", f"slot={position % POOL}",
            f"ends_pool={position % POOL == POOL - 1}",
            f"indices_max_abs_diff={index_abs:.6e}", f"indices_differing={index_n}",
            f"ring_max_abs_diff={ring_abs:.6e}", f"ring_differing={ring_n}",
            f"store_max_abs_diff={store_abs:.6e}", f"store_differing={store_n}",
            f"trash_row_max_abs_diff={trash_abs:.6e}",
            f"emitted={tuple(got.shape)}",
            f"non_sentinel_columns={int((got >= 0).sum())}")

        assert index_abs == 0.0 and index_n == 0, (
            f"the two routes emitted different indices at position {position}: "
            f"{index_n} column(s) differ, worst {index_abs}"
        )
        assert ring_abs == 0.0 and ring_n == 0, (
            f"the ring differs at position {position} after one step, so the tensor "
            f"route stashed this token in a row the landed route did not"
        )
        assert store_abs == 0.0 and store_n == 0, (
            f"an addressable pooled row differs at position {position}"
        )

    # THE PREFILL SEAM, at the same five values. One token, so the ring's remainder
    # window is at most one row wide and the seeded slot is the one the value picks.
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextDSAIndexerError

    gen = torch.Generator().manual_seed(700_103)
    key = torch.randn(1, head_dim, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    gate = torch.randn(1, head_dim, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    chunk_tokens = int(key.shape[0])

    # THE ONE ASYMMETRY, NAMED RATHER THAN STEPPED OVER: a chunk cannot end before its
    # own tokens, and only the route that may read the value can say so.
    for end_position in [v for v in DECLARED_POSITIONS if v < chunk_tokens]:
        ring = torch.zeros(2, POOL, head_dim, dtype=torch.bfloat16)
        with pytest.raises(Glm5NextDSAIndexerError) as caught:
            indexer.seed_tail(ring, key, gate, end_position)
        say("A02_SEED_REFUSED_BY_THE_INT_ROUTE", f"end_position={end_position}",
            " ".join(str(caught.value).split())[:110])

    covered = []
    for end_position in [v for v in DECLARED_POSITIONS if v >= chunk_tokens]:
        ring_int = torch.full((2, POOL, head_dim), -1.0, dtype=torch.bfloat16)
        ring_tensor = ring_int.clone()
        by_int_rows = indexer.seed_tail(ring_int, key, gate, end_position)
        by_tensor_rows = indexer.seed_tail(
            ring_tensor, key, gate, torch.tensor(end_position, dtype=torch.int32)
        )
        ring_abs, ring_n = _worst(ring_tensor, ring_int)
        say("A02_SEED", f"end_position={end_position}",
            f"remainder={end_position % POOL}", f"int_rows={int(by_int_rows)}",
            f"tensor_rows={int(by_tensor_rows)}",
            f"ring_max_abs_diff={ring_abs:.6e}", f"ring_differing={ring_n}")
        assert int(by_tensor_rows) == int(by_int_rows), (
            f"the two routes wrote a different number of ring rows at "
            f"end_position={end_position}"
        )
        assert ring_abs == 0.0 and ring_n == 0, (
            f"the seeded ring differs at end_position={end_position}, so the tensor "
            f"route wrote a different slot or a different token"
        )
        covered.append(end_position)

    say("A02_SEED_VALUES_COVERED", *covered)
    assert covered == [v for v in DECLARED_POSITIONS if v >= chunk_tokens], (
        f"the prefill seam's comparison covered {covered}, which is not every declared "
        f"value a chunk of {chunk_tokens} token(s) admits"
    )


# ══════════════════════════════════════════════════════════════════════════════
# A03. The decode leg spells no host read of the position (STRUCTURAL).
# ══════════════════════════════════════════════════════════════════════════════
def test_a03_the_decode_leg_spells_no_host_read_of_the_position() -> None:
    """An ``ast`` count over the decode branch of the indexer's own forward.

    THE SCOPE IS NAMED AND NARROW, because a count over a whole file would answer a
    different question. It is the ``if is_decode:`` branch of
    ``Glm5NextDSAIndexer.forward``: the leg a captured decode graph runs, where the base
    spends the real position twice -- once handing it to the ring seam and once dividing
    it into a pool address. The count is of ``int(...)`` applied to the position by NAME,
    which is what a host read of it looks like in source.

    A COUNT IS NOT A PROOF THAT NO READ CAN HAPPEN, and this item does not claim to be
    one: it reads the leg's own source. A01 is the reading that no value is reached at
    run time, on a device that cannot supply one.
    """
    _require_cpu_mode()
    from vllm_neuron.model.glm5_next import model_fp8

    source = inspect.getsource(model_fp8.Glm5NextDSAIndexer.forward)
    tree = ast.parse(textwrap.dedent(source))
    names = ("position", "end_position", "prefill_end_position")

    decode_branch = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "is_decode"
        ):
            decode_branch = node
            break
    assert decode_branch is not None, (
        "no `if is_decode:` branch was found in the indexer's forward, so this item "
        "could not have been counting the decode leg"
    )

    def host_reads(nodes) -> list[str]:
        found = []
        for node in nodes:
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id == "int"
                    and len(inner.args) == 1
                    and isinstance(inner.args[0], ast.Name)
                    and inner.args[0].id in names
                ):
                    found.append(f"int({inner.args[0].id})")
        return found

    on_the_leg = host_reads(decode_branch.body)
    whole_forward = host_reads([tree])
    say("A03_DECODE_LEG_HOST_READS", len(on_the_leg), *on_the_leg)
    say("A03_WHOLE_FORWARD_HOST_READS", len(whole_forward), *whole_forward)
    say("A03_THE_LEG_USES", "torch.is_tensor(position)" in source,
        "decode_pool_address(" in source)

    assert len(on_the_leg) == 0, (
        f"the decode leg still reads the position on the host: {on_the_leg}. The base "
        f"spends it twice here and both are what this block removed"
    )
    assert len(whole_forward) == 0, (
        f"the forward reads a position value on the host outside the decode leg: "
        f"{whole_forward}"
    )
    assert "torch.is_tensor(position)" in source
    assert "decode_pool_address(" in source

    # The seam files are read as bytes for the one name that must NOT come back: a
    # host read spelled as a method rather than as a cast. Both paths come from the
    # loaded modules, so neither can point at a file this run did not import.
    for path in (
        Path(model_fp8.__file__).resolve(),
        Path(_seam_module().__file__).resolve(),
    ):
        text = path.read_text()
        say("A03_ITEM_CALLS", path.name, text.count(".item()"))
        assert ".item()" not in text, (
            f"{path.name} spells `.item()`, which is the same host read under another "
            f"name"
        )
