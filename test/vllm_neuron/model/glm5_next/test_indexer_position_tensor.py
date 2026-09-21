# SPDX-License-Identifier: Apache-2.0
"""The sparse indexer's decode step serves every position from one graph.

Both ring writes derive one shape at two positions without reading a value, and the
tensor route agrees with the python-int route bit for bit.
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
# the declared fixture. Every width is the sibling's, read from it rather
# than typed again, so this file cannot disagree with the geometry the
# tests already run at.
# --------------------------------------------------------------------------- #
#: tokens the prefill leg lays down before the decode steps are compared.
PREFILL_TOKENS = layer_half.PREFILL_TOKENS
#: Slots per pool. The ring's length and the modulus every address below uses.
POOL = layer_half.POOL_SIZE
#: Rows per page of the pooled-key store, and the store's own row count.
PAGE_SIZE = layer_half.PAGE_SIZE
POOL_ROWS = layer_half.PAGES * layer_half.PAGE_SIZE
#: The batch's longest sequence, held constant across every position compared below.
#: It is what sizes the candidate axis, so a per-position value would change the
#: regime under the comparison instead of changing only the position.
MAX_SEQ_LEN = PREFILL_TOKENS + 1
#: The positions the route-equality test compares: the first step, the step after it, the last slot
#: of a pool, the first slot of the next one, and the step that follows the prefill.
#: Two of them end a pool (``POOL - 1`` and ``PREFILL_TOKENS``) and three do not.
DECLARED_POSITIONS = (0, 1, POOL - 1, POOL, PREFILL_TOKENS)
#: The two positions the shape test derives shapes at.
LOW_POSITION = 0
HIGH_POSITION = PREFILL_TOKENS


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
    """An indexer with no weight but its per-slot bias, on the shape-only device. """
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
    """The fixture's own indexer, every leaf materialised, and its config."""
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
    """Lay down the prefill leg's pooled rows and its seeded remainder, once. """
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
    """One decode step on ``state``, which it writes in place. """
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
# One shape at two positions, and no value read on the way there.
# ══════════════════════════════════════════════════════════════════════════════
def test_the_two_ring_seams_derive_one_shape_at_two_positions_without_reading_a_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both entry points, at position 0 and at position N, on shape-only tensors. """
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
    # as a pair of shape-only tensors rather than left to the value comparison alone.
    index, completes = _seam_module().decode_pool_address(
        torch.tensor(HIGH_POSITION, dtype=torch.int32, device="meta"),
        POOL,
        torch.device("meta"),
    )
    assert tuple(index.shape) == () and tuple(completes.shape) == ()
    assert completes.dtype == torch.bool


# ══════════════════════════════════════════════════════════════════════════════
# The tensor route equals the int route, exactly.
# ══════════════════════════════════════════════════════════════════════════════
def test_the_tensor_route_and_the_int_route_agree_bit_for_bit() -> None:
    """Five declared positions, one prefill state, two decode runs each, no tolerance.
    """
    _require_cpu_mode()
    indexer, cfg = _full_indexer()
    head_dim = int(indexer.index_head_dim)
    candidates = _candidate_count()
    assert candidates > int(indexer.select_k()), (
        f"{candidates} candidate pool(s) is not more than the {int(indexer.select_k())} "
        f"selected, so the indexer would take its causal bypass and this test would "
        f"compare two answers that never touched the ring"
    )

    seeded = _state(head_dim)
    _prefill(indexer, cfg, seeded)

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
        _trash_abs, _trash_n = _worst(
            by_tensor["pool_cache"][POOL_ROWS - 1], by_int["pool_cache"][POOL_ROWS - 1]
        )

        assert index_abs == 0.0 and index_n == 0, (
            f"the two routes emitted different indices at position {position}: "
            f"{index_n} column(s) differ, worst {index_abs}"
        )
        assert ring_abs == 0.0 and ring_n == 0, (
            f"the ring differs at position {position} after one step, so the tensor "
            f"route stashed this token in a row the route did not"
        )
        assert store_abs == 0.0 and store_n == 0, (
            f"an addressable pooled row differs at position {position}"
        )

    # The prefill seam, at the same five values. One token, so the ring's remainder
    # window is at most one row wide and the seeded slot is the one the value picks.
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextDSAIndexerError

    gen = torch.Generator().manual_seed(700_103)
    key = torch.randn(1, head_dim, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    gate = torch.randn(1, head_dim, generator=gen, dtype=torch.float32).to(torch.bfloat16)
    chunk_tokens = int(key.shape[0])

    # The one asymmetry, named rather than stepped over: a chunk cannot end before its
    # own tokens, and only the route that may read the value can say so.
    for end_position in [v for v in DECLARED_POSITIONS if v < chunk_tokens]:
        ring = torch.zeros(2, POOL, head_dim, dtype=torch.bfloat16)
        with pytest.raises(Glm5NextDSAIndexerError) as caught:
            indexer.seed_tail(ring, key, gate, end_position)

    covered = []
    for end_position in [v for v in DECLARED_POSITIONS if v >= chunk_tokens]:
        ring_int = torch.full((2, POOL, head_dim), -1.0, dtype=torch.bfloat16)
        ring_tensor = ring_int.clone()
        by_int_rows = indexer.seed_tail(ring_int, key, gate, end_position)
        by_tensor_rows = indexer.seed_tail(
            ring_tensor, key, gate, torch.tensor(end_position, dtype=torch.int32)
        )
        ring_abs, ring_n = _worst(ring_tensor, ring_int)
        assert int(by_tensor_rows) == int(by_int_rows), (
            f"the two routes wrote a different number of ring rows at "
            f"end_position={end_position}"
        )
        assert ring_abs == 0.0 and ring_n == 0, (
            f"the seeded ring differs at end_position={end_position}, so the tensor "
            f"route wrote a different slot or a different token"
        )
        covered.append(end_position)

    assert covered == [v for v in DECLARED_POSITIONS if v >= chunk_tokens], (
        f"the prefill seam's comparison covered {covered}, which is not every declared "
        f"value a chunk of {chunk_tokens} token(s) admits"
    )


# ══════════════════════════════════════════════════════════════════════════════
# The decode leg spells no host read of the position (structural).
# ══════════════════════════════════════════════════════════════════════════════
def test_the_decode_leg_spells_no_host_read_of_the_position() -> None:
    """An ``ast`` count over the decode branch of the indexer's own forward. """
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
        "no `if is_decode:` branch was found in the indexer's forward, so this test "
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

    # The seam files are read for the one name that must not come back: a host read
    # spelled as a method rather than as a cast. Both paths come from the loaded
    # modules, so neither can point at a file this run did not import.
    #
    # The reading is of the code, not of the text. A comment or a docstring naming
    # `.item()` -- to say why the call is not there -- is not a call, and a scan over
    # the bytes cannot tell the two apart. So the count is of call sites: an
    # attribute access named `test` that is being called. Prose cannot enter it, and
    # a call cannot hide from it behind a line break or spacing either.
    for path in (
        Path(model_fp8.__file__).resolve(),
        Path(_seam_module().__file__).resolve(),
    ):
        calls = [
            node
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "test"
        ]
        assert not calls, (
            f"{path.name} calls `.item()` at line(s) "
            f"{[node.lineno for node in calls]}, which is the same host read under "
            f"another name"
        )
