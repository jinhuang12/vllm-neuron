# SPDX-License-Identifier: Apache-2.0
"""``inc-glm53f-117c`` acceptance -- one captured graph serves every position.

THE DECLARED ACCEPTANCE, Tier N, CPU mode:

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 python -m pytest \\
      test/vllm_neuron/model/glm5_next/test_capture_shape_identity_117c.py \\
      -s -rA -p no:randomly -p no:cacheprovider

Six items, one test each, no ``parametrize`` and no skip.

WHAT THE BLOCK IS ABOUT. A graph is captured once and replayed at every position, so
anything whose SHAPE or whose CONTROL FLOW comes from a position value pins the graph
to the position it was captured at. Two such things existed: the MLA layer read its
cache as ``[: start + tokens]``, whose length grows with every decode step, and the KDA
layer chose its entering state with a python branch on the same value.

WHAT THESE ITEMS OBSERVE, AND WHAT THEY DO NOT. Items 1 to 4 are BEHAVIOURAL: they
drive the runner's own carrier builder and read the window it hands a layer. Items 5
and 6 are STRUCTURAL, in the form this campaign already uses for the state hook
(``test_kda_runner_state.py`` B01 and B02): they read the source of the two methods and
assert the position no longer reaches a python int there. Nothing here captures a real
dynamo graph -- that reading is a host-side end-to-end one on the serving run, and this
file must not be read as making it.

EVERY ITEM FAILS AT THE BASE. Items 1 and 2 because the base's window is the request's
own blocks, so its length moves with the position; item 3 because the base has no
headroom refusal to raise; item 4 because the base hands a python int; items 5 and 6
because the base's source carries the two host reads this block removed.
"""

from __future__ import annotations

import inspect
import os

import pytest
import torch

from vllm_neuron.model.glm5_next import model_fp8
from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

# --------------------------------------------------------------------------- #
# The declared geometry. Every number here is this file's own fixture, not a
# registered dial: the block's subject is the RELATION between the window's length
# and the position, and that relation has to hold at any legal geometry.
# --------------------------------------------------------------------------- #
#: Slots per block in the bank and in the group's page.
DECLARED_PAGE_SIZE = 8
#: The latent width of the fixture's bank. Nothing under test reads it -- the carrier
#: builder slices rows and never inspects the width -- so it is this file's own number
#: and deliberately not the checkpoint's, which would suggest a dial was registered here.
DECLARED_HEAD_SIZE = 16
#: The bucket's block-table width: the window's length in blocks.
DECLARED_WINDOW_BLOCKS = 4
#: Blocks the bank holds. It carries a spare window past the last block a request is
#: given here, which is the headroom item 3 reads.
DECLARED_BANK_BLOCKS = 12
#: The first block this request is given, deliberately not block 0, so a window that
#: started at the bank's own base rather than the request's would be visible.
DECLARED_FIRST_BLOCK = 2
#: The two positions items 1 and 2 compare. They differ by enough to change how many
#: BLOCKS the request occupies, which is what moved the base's window length.
DECLARED_LOW_POSITION = 3
DECLARED_HIGH_POSITION = 25
DECLARED_TOKENS = 1
#: The carrier keys a sparse layer is handed, as the landed builder writes them.
DECLARED_SPARSE_CARRIER_KEYS = {
    "latent_cache",
    "pool_cache",
    "seq_lens",
    "start_position",
    "softmax_scale",
    "max_seq_len",
    "page_size",
    "tail",
    "position",
}


def say(name: str, *values) -> None:
    """One reading per line, tagged so a launcher can anchor on it."""
    print("CAPSHAPE|" + name + "|" + "|".join(str(value) for value in values))


def _require_cpu_mode() -> None:
    """The declared acceptance runs under VLLM_NEURON_CPU_MODE=1, so read it, not set it."""
    assert os.environ.get("VLLM_NEURON_CPU_MODE") == "1", (
        "the declared acceptance runs under VLLM_NEURON_CPU_MODE=1 and this process "
        "does not carry it, so nothing below would be measuring the declared mode"
    )


def _bank(head_size: int, *, blocks: int = DECLARED_BANK_BLOCKS) -> dict:
    """One sparse bank, carrying only the four keys the carrier builder reads.

    Built by hand for the reason ``test_kda_runner_state.py`` gives for its own banks: a
    builder that started reading a fifth key would raise here rather than quietly find a
    stand-in value.
    """
    return {
        "name": "model.layers.0.self_attn",
        "family": "self_attn",
        "block_size": DECLARED_PAGE_SIZE,
        "latent_cache": torch.zeros(
            (blocks * DECLARED_PAGE_SIZE, 1, head_size), dtype=torch.bfloat16
        ),
    }


def _geometry(*, position: int, window_blocks: int = DECLARED_WINDOW_BLOCKS) -> dict:
    """This request's pages at ``position``, plus the window's length in blocks.

    The block run is the ascending run the builder requires, and it is exactly as long
    as the request needs at this position -- which is the number the base's window used
    and the number this block's window no longer uses.
    """
    used = max(1, -(-(position + DECLARED_TOKENS) // DECLARED_PAGE_SIZE))
    return {
        "block_ids": [DECLARED_FIRST_BLOCK + offset for offset in range(used)],
        "state_slot": DECLARED_FIRST_BLOCK,
        "page_size": DECLARED_PAGE_SIZE,
        "window_blocks": int(window_blocks),
    }


def _carrier(bank: dict, text_config, *, position: int, geometry: dict) -> dict:
    """The one carrier the builder hands this bank's layer at ``position``."""
    side = NeuronModelRunner._glm5next_side_caches(
        [bank],
        index_kpool=int(text_config.index_kpool),
        index_head_dim=int(text_config.index_head_dim),
        max_seq_len=DECLARED_BANK_BLOCKS * DECLARED_PAGE_SIZE,
    )
    carriers = NeuronModelRunner._glm5next_layer_carriers(
        [bank],
        side,
        geometries=[geometry],
        is_prefill=False,
        tokens=DECLARED_TOKENS,
        start_position=position,
        softmax_scale=1.0,
        max_seq_len=position + DECLARED_TOKENS,
        index_kpool=int(text_config.index_kpool),
    )
    assert len(carriers) == 1, f"one bank was handed {len(carriers)} carrier(s)"
    return carriers[0]


def _world():
    """A text config and one sparse bank, the pair every behavioural item drives."""
    text_config = Glm5NextTextConfig()
    return text_config, _bank(DECLARED_HEAD_SIZE)


# ══════════════════════════════════════════════════════════════════════════════
# ITEM 1. The window has ONE length at two positions.
# ══════════════════════════════════════════════════════════════════════════════
def test_the_window_the_layer_is_handed_has_one_length_at_two_positions() -> None:
    """The defect, read directly: the length the layer sees must not move with the position.

    The two positions are chosen so the request occupies a DIFFERENT number of blocks at
    each -- that difference is exactly what the base's window length followed, and it is
    what a captured graph cannot survive.
    """
    _require_cpu_mode()
    text_config, bank = _world()
    head_size = int(bank["latent_cache"].shape[2])

    low = _carrier(bank, text_config, position=DECLARED_LOW_POSITION,
                   geometry=_geometry(position=DECLARED_LOW_POSITION))
    high = _carrier(bank, text_config, position=DECLARED_HIGH_POSITION,
                    geometry=_geometry(position=DECLARED_HIGH_POSITION))

    low_blocks = len(_geometry(position=DECLARED_LOW_POSITION)["block_ids"])
    high_blocks = len(_geometry(position=DECLARED_HIGH_POSITION)["block_ids"])
    say("I1_BLOCKS_THE_REQUEST_OCCUPIES", f"low={low_blocks}", f"high={high_blocks}")
    assert low_blocks != high_blocks, (
        "the two positions occupy the same number of blocks, so this item could not "
        "tell a constant window from a request-shaped one"
    )

    say("I1_WINDOW_SHAPES", tuple(low["latent_cache"].shape),
        tuple(high["latent_cache"].shape))
    assert tuple(low["latent_cache"].shape) == tuple(high["latent_cache"].shape)
    assert tuple(low["latent_cache"].shape) == (
        DECLARED_WINDOW_BLOCKS * DECLARED_PAGE_SIZE,
        1,
        head_size,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ITEM 2. The window is the BUCKET's block count, and it starts at the request's page.
# ══════════════════════════════════════════════════════════════════════════════
def test_the_window_is_the_buckets_block_count_and_starts_at_the_requests_page() -> None:
    """Two conjuncts: the length is the bucket's, and the base is still the request's.

    A window of the right length that started at the bank's own base would put every
    write on another sequence's rows, so the length alone is not the property.
    """
    _require_cpu_mode()
    text_config, bank = _world()
    geometry = _geometry(position=DECLARED_LOW_POSITION)
    carrier = _carrier(bank, text_config, position=DECLARED_LOW_POSITION,
                       geometry=geometry)

    window = DECLARED_WINDOW_BLOCKS * DECLARED_PAGE_SIZE
    say("I2_WINDOW_SLOTS", int(carrier["latent_cache"].shape[0]), f"want={window}")
    assert int(carrier["latent_cache"].shape[0]) == window
    assert len(geometry["block_ids"]) < DECLARED_WINDOW_BLOCKS, (
        "this request already occupies the whole window, so the item could not tell "
        "the bucket's length from the request's"
    )

    first_slot = bank["latent_cache"][DECLARED_FIRST_BLOCK * DECLARED_PAGE_SIZE]
    say("I2_WINDOW_STARTS_AT", DECLARED_FIRST_BLOCK * DECLARED_PAGE_SIZE)
    assert carrier["latent_cache"].data_ptr() == first_slot.data_ptr(), (
        "the window does not start at this request's own first page, so its writes "
        "would land on another sequence's rows"
    )


# ══════════════════════════════════════════════════════════════════════════════
# ITEM 3. A bank without the spare window refuses BY NAME.
# ══════════════════════════════════════════════════════════════════════════════
def test_a_bank_without_the_spare_window_refuses_by_name() -> None:
    """A slice past the end of a bank returns a SHORTER view instead of raising.

    That is the whole reason this refusal exists: a shortened view turns the constant
    length back into a per-request one exactly where a captured graph cannot see it. The
    control below reads the silent truncation directly, so the refusal is measured
    against the behaviour it replaces rather than asserted on its own.
    """
    _require_cpu_mode()
    text_config, _ = _world()
    short = _bank(DECLARED_HEAD_SIZE, blocks=DECLARED_FIRST_BLOCK + 1)

    base = DECLARED_FIRST_BLOCK * DECLARED_PAGE_SIZE
    window = DECLARED_WINDOW_BLOCKS * DECLARED_PAGE_SIZE
    truncated = short["latent_cache"][base : base + window]
    say("I3_CONTROL_THE_SLICE_TRUNCATES_SILENTLY",
        int(truncated.shape[0]), f"asked={window}")
    assert int(truncated.shape[0]) < window, (
        "this bank is long enough for the window, so the item is not reading the "
        "short-bank case at all"
    )

    with pytest.raises(ValueError) as caught:
        _carrier(short, text_config, position=DECLARED_LOW_POSITION,
                 geometry=_geometry(position=DECLARED_LOW_POSITION))
    message = " ".join(str(caught.value).split())
    say("I3_MESSAGE", message[:190])
    assert "spare" in message
    assert "clamped base" in message


# ══════════════════════════════════════════════════════════════════════════════
# ITEM 4. The position reaches the layer as a TENSOR.
# ══════════════════════════════════════════════════════════════════════════════
def test_the_position_reaches_the_layer_as_a_tensor() -> None:
    """The traced boundary takes a tensor; a python int there is baked into the graph.

    The value is read back host-side, which is legitimate outside a traced region and is
    how this item tells a tensor carrying the right position from a tensor carrying any
    position at all.
    """
    _require_cpu_mode()
    text_config, bank = _world()
    carrier = _carrier(bank, text_config, position=DECLARED_HIGH_POSITION,
                       geometry=_geometry(position=DECLARED_HIGH_POSITION))

    assert set(carrier) == DECLARED_SPARSE_CARRIER_KEYS, (
        f"a sparse carrier holds {sorted(carrier)}, not "
        f"{sorted(DECLARED_SPARSE_CARRIER_KEYS)}"
    )
    position = carrier["start_position"]
    say("I4_START_POSITION", type(position).__name__,
        getattr(position, "dtype", None), int(position))
    assert torch.is_tensor(position), (
        "the position reaches the layer as a python int, which a captured graph turns "
        "into the constant it was captured with"
    )
    assert int(position) == DECLARED_HIGH_POSITION


# ══════════════════════════════════════════════════════════════════════════════
# ITEM 5. The MLA read no longer depends on the position (STRUCTURAL).
# ══════════════════════════════════════════════════════════════════════════════
def test_the_mla_read_no_longer_depends_on_the_position() -> None:
    """``attend()``'s own source: the read is the whole window and no host read remains.

    A structural reading, in the form this campaign uses for the state hook. It observes
    the three lines the block changed; it does not observe an attention output.
    """
    _require_cpu_mode()
    source = inspect.getsource(model_fp8.Glm5NextMLAAttention.attend)

    say("I5_HOST_READS_OF_THE_POSITION", source.count("int(start_position)"))
    assert source.count("int(start_position)") == 0

    say("I5_READS_THE_WHOLE_WINDOW", "c_kv = latent_cache[:, 0, :]" in source)
    assert "c_kv = latent_cache[:, 0, :]" in source
    assert "[: start + tokens" not in source, (
        "the read is still cut at the position, so its length still moves with every "
        "decode step"
    )

    say("I5_WRITES_BY_INDEX", "index_copy_" in source)
    assert "index_copy_" in source
    assert "torch.as_tensor(" in source


# ══════════════════════════════════════════════════════════════════════════════
# ITEM 6. The KDA entering state is chosen without reading the position (STRUCTURAL).
# ══════════════════════════════════════════════════════════════════════════════
def test_the_kda_entering_state_is_chosen_without_reading_the_position() -> None:
    """The branch became a device choice, and the helper that makes it returns a tensor.

    Two conjuncts, one structural and one a value reading: the forward carries no host
    read of the position and selects with ``torch.where``, and the helper it selects
    with answers in a 0-d bool tensor rather than a python bool.
    """
    _require_cpu_mode()
    source = inspect.getsource(model_fp8.Glm5NextKDAAttention.forward)

    say("I6_HOST_READS_OF_THE_POSITION", source.count("int(start_position)"))
    assert source.count("int(start_position)") == 0
    say("I6_CHOOSES_ON_DEVICE", "torch.where(" in source, "_start_is_zero(" in source)
    assert "torch.where(" in source
    assert "_start_is_zero(" in source

    device = torch.device("cpu")
    at_zero = model_fp8._start_is_zero(torch.tensor(0, dtype=torch.int32), device)
    above = model_fp8._start_is_zero(torch.tensor(DECLARED_HIGH_POSITION), device)
    say("I6_HELPER", type(at_zero).__name__, at_zero.dtype, at_zero.dim(),
        bool(at_zero), bool(above))
    assert torch.is_tensor(at_zero) and torch.is_tensor(above)
    assert at_zero.dtype == torch.bool and at_zero.dim() == 0
    assert bool(at_zero) is True
    assert bool(above) is False
