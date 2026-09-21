# SPDX-License-Identifier: Apache-2.0
"""A step's carrier geometry comes from the host, not from a device tensor.

The request count, the page span and the row a slice follows are all host reads, so
a device tensor under a host key has to refuse by name.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

# ---------------------------------------------------------------------------
# Declared values. Small, and each one named where it is used.
# ---------------------------------------------------------------------------

#: Slots per page. Four is the tiny fixture's KV page (``test_tiny_glm5next_e2e.py``).
PAGE = 4

#: The latent bank: pages, and the width one slot holds.
BANK_PAGES = 32
LATENT_WIDTH = 8

#: The bank's own row count, which is what a sparse carrier hands a layer: the whole bank,
#: no slice and no copy. It is the same number at every position and for every request,
#: which is why the readings about a request are made on its block table instead.
BANK_SLOTS = BANK_PAGES * PAGE

#: Recurrent state slots in the linear-attention bank. The converter does not read a state
#: slot off the block row -- it hands each request a slot of its own table -- so this width
#: is the bank's paging geometry and the bound below is the request axis.
STATE_SLOTS = 8

#: How many sequences the modelled engine admits at once. One, and the bank above holds
#: more slots than that on purpose: the two numbers are not the same axis.
DECLARED_MAX_NUM_SEQS = 1
#: The id of that one request. The converter keys a sequence's state by its id and refuses a
#: real step served without one, so a shell with no id can only ever be served the opening
#: bucket -- which no test here reads, and one of them steps past.
DECLARED_REQUEST = "host-geometry-request"

#: The slot a step carrying no request id is served from, which is every step this file
#: drives: the converter takes no claim for such a step and reads slot 0.
SYNTHETIC_SLOT = 0

#: A single-token step is a decode and anything longer is a prefill.
DECODE_THRESHOLD = 1

#: The longest sequence the harness admits, which is what the side-cache allocator bounds
#: its pooled store by (``neuron_model_runner.py``).
MAX_MODEL_LEN = BANK_PAGES * PAGE


def _table_blocks(row) -> int:
    """The block table's width for an entry ``_entry`` built from ``row``. """
    return len(row)


def _padded_table(row, pages: int) -> list[int]:
    """The block ids a carrier built from ``row`` names, at ``pages`` pages of its own.
    """
    return [int(value) for value in row[:pages]] + [-1] * (_table_blocks(row) - pages)


def _physical_rows(row, *, start: int, tokens: int) -> list[int]:
    """Each of this step's tokens as a physical bank row, by the runner's own formula.
    """
    rows: list[int] = []
    for offset in range(int(tokens)):
        position = int(start) + offset
        rows.append(int(row[position // PAGE]) * PAGE + position % PAGE)
    return rows


def _text_config() -> SimpleNamespace:
    """The four config fields the converter reads, and nothing else."""
    return SimpleNamespace(
        index_kpool=4,
        index_head_dim=LATENT_WIDTH,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
    )


def _sparse_bank(device: torch.device) -> dict:
    """One sparse-attention bank, in the shape ``bind_kv_cache`` leaves on the model.
    """
    latent = torch.zeros(
        (BANK_PAGES * PAGE, 1, LATENT_WIDTH), dtype=torch.bfloat16, device=device
    )
    return {
        "name": "model.layers.0.self_attn",
        "family": "self_attn",
        "block_size": PAGE,
        "latent_cache": latent,
    }


def _linear_bank(device: torch.device) -> dict:
    """One linear-attention bank, whose carrier is a slot of each state tensor."""
    return {
        "name": "model.layers.1.attention",
        "family": "linear_attn",
        "state_slots": STATE_SLOTS,
        "conv_state": torch.zeros(
            (STATE_SLOTS, 4, 6), dtype=torch.bfloat16, device=device
        ),
        "recurrent_state": torch.zeros(
            (STATE_SLOTS, 2, 6, 6), dtype=torch.float32, device=device
        ),
    }


def _runner(banks) -> NeuronModelRunner:
    """A runner carrying only what the converter reads. """
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = SimpleNamespace(
        text_config=_text_config(), glm5next_layer_banks=tuple(banks)
    )
    runner.max_model_len = MAX_MODEL_LEN
    # The converter sizes its per-sequence caches by the engine's concurrent-sequence
    # bound, so a runner shell must model that bound too.
    runner.max_num_reqs = DECLARED_MAX_NUM_SEQS
    # It also keys each sequence's state by its request id, and a step served without one
    # is the opening bucket's. Every step here is the one request's.
    runner.input_batch = SimpleNamespace(req_ids=[DECLARED_REQUEST])
    return runner


def _entry(
    *,
    host_row,
    host_cached: int,
    device_row,
    device_cached: int,
    tokens: int,
    device: torch.device,
) -> dict:
    """One KV-cache group's metadata entry, with its host and device halves set apart.
    """
    device_table = torch.tensor(
        [[int(value) for value in row] for row in device_row],
        dtype=torch.int32,
        device=device,
    )
    return {
        "block_table_tensor": device_table,
        "full_block_table_tensor": device_table,
        "slot_mapping": torch.arange(tokens, dtype=torch.int32, device=device),
        "max_query_len": int(tokens),
        "block_size": PAGE,
        "max_blocks_per_seq": int(device_table.shape[1]),
        "decode_token_threshold": DECODE_THRESHOLD,
        "cached_seq_len": torch.tensor(
            [[int(device_cached)]], dtype=torch.int32, device=device
        ),
        "host_block_table": host_row,
        "host_num_computed_tokens": host_cached,
        "kv_segment_size": int(device_table.shape[1]) * PAGE,
    }


def _kwargs(banks, entry, *, tokens: int, device: torch.device) -> dict:
    """The generic mapping a call site hands the converter, at this step's token count."""
    return {
        "input_ids": torch.ones(tokens, dtype=torch.int32, device=device),
        "attn_metadata": {str(bank["name"]): entry for bank in banks},
        "sampling_positions": torch.zeros(1, dtype=torch.int32, device=device),
    }


def _open_ring_at(runner, banks, position: int) -> None:
    """Stand the live indexer ring up and declare which position it holds. """
    runner._glm5next_side_cache_set = NeuronModelRunner._glm5next_side_caches(
        banks,
        index_kpool=int(_text_config().index_kpool),
        index_head_dim=int(_text_config().index_head_dim),
        max_seq_len=MAX_MODEL_LEN,
        request_slots=DECLARED_MAX_NUM_SEQS,
    )
    runner._glm5next_request_slot_table = {DECLARED_REQUEST: SYNTHETIC_SLOT}
    runner._glm5next_side_cache_positions = {SYNTHETIC_SLOT: int(position)}


# ══════════════════════════════════════════════════════════════════════════════
# A captured step: every device tensor and every bank on ``meta``.
# ══════════════════════════════════════════════════════════════════════════════
def test_a_capture_on_meta_builds_its_carriers() -> None:
    """The graph-extraction world, which is where the hardware run stopped. """
    meta = torch.device("meta")
    banks = [_sparse_bank(meta), _linear_bank(meta)]
    runner = _runner(banks)
    tokens = 8
    host_row = [0, 1, 2, 3]
    entry = _entry(
        host_row=[host_row],
        host_cached=[0],
        device_row=[[0, 1, 2, 3]],
        device_cached=0,
        tokens=tokens,
        device=meta,
    )

    translated = runner._glm5next_model_kwargs(_kwargs(banks, entry, tokens=tokens, device=meta))

    carriers = translated["layer_carriers"]
    assert len(carriers) == 2
    sparse, linear = carriers
    # The carrier is the bank itself, no slice and no copy, so its length is the bank's at
    # every step. Eight tokens at position 0 still occupy two pages of four, and those two
    # pages are named by the two keys below rather than by a length.
    assert sparse["latent_cache"] is banks[0]["latent_cache"]
    assert int(sparse["latent_cache"].shape[0]) == BANK_SLOTS
    # The pages, as form only. This test's carrier is on `meta`, where a tensor has a shape
    # and no values, so what is read here is the shape a captured graph is compiled for: the
    # table is the bucket's width and the rows are one per token of the step. The values are
    # read where they exist, on the CPU steps below.
    assert tuple(sparse["block_table_row"].shape) == (_table_blocks(host_row), 1)
    assert sparse["block_table_row"].dtype == torch.int32
    assert sparse["block_table_row"].device.type == "meta"
    assert tuple(sparse["latent_slots"].shape) == (tokens,)
    assert sparse["latent_slots"].dtype == torch.int64
    assert sparse["latent_slots"].device.type == "meta"
    # The position arrives as a tensor, because a host number here is baked into the graph
    # it was captured with, and this test's carrier is on `meta`, where a value does not
    # exist to be read. What `meta` does answer is the operand's form, which is what a
    # captured graph depends on; the value is read where it is readable, on the continuing
    # CPU step below.
    assert tuple(sparse["start_position"].shape) == ()
    assert sparse["start_position"].dtype == torch.int32
    assert sparse["start_position"].device.type == "meta"
    assert sparse["latent_cache"].device.type == "meta"
    # The linear layer's carrier is the bank's slot, which this request owns.
    # The carrier holds one view per request rather than one tensor, because two requests'
    # states are two rows of one bank, so the shape claim is made on this step's one
    # request's own view.
    assert len(linear["conv_state"]) == 1
    assert tuple(linear["conv_state"][0].shape) == (4, 6)
    # A host number at this boundary would be a captured constant, and the linear carrier
    # carries one row per request, so its position is a 1-D int32 tensor as long as the
    # batch rather than a 0-d one. The three lines below claim that the position reaches
    # the layer as a tensor whose value no captured graph holds, at this step's one
    # request.
    assert tuple(linear["start_position"].shape) == (1,)
    assert linear["start_position"].dtype == torch.int32
    assert linear["start_position"].device.type == "meta"


# ══════════════════════════════════════════════════════════════════════════════
# Which half the converter read, made visible by making the halves disagree.
# ══════════════════════════════════════════════════════════════════════════════
def test_the_host_row_is_the_one_the_slice_follows() -> None:
    """A serving-shaped step whose two halves name different pages. """
    cpu = torch.device("cpu")
    banks = [_sparse_bank(cpu)]
    runner = _runner(banks)
    tokens = 6
    host_row = [4, 5, 6, 7]
    entry = _entry(
        host_row=[host_row],
        host_cached=[0],
        device_row=[[500, 501, 502, 503]],
        device_cached=0,
        tokens=tokens,
        device=cpu,
    )

    translated = runner._glm5next_model_kwargs(_kwargs(banks, entry, tokens=tokens, device=cpu))

    carrier = translated["layer_carriers"][0]
    bank = banks[0]["latent_cache"]
    # The carrier is the whole bank, so its length and its pointer are the bank's whatever
    # row the converter read, and neither can tell the two halves of the bank apart. Which
    # half the converter read is measured below instead, on the rows the layer writes.
    assert int(carrier["latent_cache"].shape[0]) == BANK_SLOTS
    assert carrier["latent_cache"].data_ptr() == bank.data_ptr()
    # Where that measurement lives now. Six tokens at position 0 occupy two pages, and they
    # are the host row's first two: the table names pages 4 and 5 and pads the bucket's
    # remaining two entries, and the rows the layer writes are those pages' own. The device
    # row's pages 500 and 501 satisfy neither reading -- their rows are outside this bank.
    pages = -(-tokens // PAGE)
    assert pages == 2
    assert carrier["block_table_row"].flatten().tolist() == _padded_table(host_row, pages)
    assert carrier["latent_slots"].tolist() == _physical_rows(
        host_row, start=0, tokens=tokens
    )


# ══════════════════════════════════════════════════════════════════════════════
# The batch's own row count, not the padded one.
# ══════════════════════════════════════════════════════════════════════════════
def test_a_padded_device_table_does_not_decide_the_request_count() -> None:
    """One request, in a device table padded to four rows. """
    cpu = torch.device("cpu")
    banks = [_sparse_bank(cpu)]
    runner = _runner(banks)
    tokens = 4
    host_row = [0, 1]
    entry = _entry(
        host_row=[host_row],
        host_cached=[0],
        device_row=[[0, 1], [0, 0], [0, 0], [0, 0]],
        device_cached=0,
        tokens=tokens,
        device=cpu,
    )

    translated = runner._glm5next_model_kwargs(_kwargs(banks, entry, tokens=tokens, device=cpu))

    # The length is the bank's, at every position and for every request, so what this test
    # measures is that the step was not refused as a four-request batch: a carrier that
    # exists at all is that reading. The two lines after it say what the carrier holds --
    # this request's one page, and the bucket's second entry padded.
    carrier = translated["layer_carriers"][0]
    assert int(carrier["latent_cache"].shape[0]) == BANK_SLOTS
    assert carrier["block_table_row"].flatten().tolist() == _padded_table(host_row, 1)
    assert carrier["latent_slots"].tolist() == _physical_rows(
        host_row, start=0, tokens=tokens
    )


# ══════════════════════════════════════════════════════════════════════════════
# The rule, mechanical: a device tensor under a host key is refused.
# ══════════════════════════════════════════════════════════════════════════════
def test_a_device_tensor_under_a_host_key_is_refused_by_name() -> None:
    """The refusal that keeps the rule true for a call site written later. """
    cpu = torch.device("cpu")
    banks = [_sparse_bank(cpu)]
    runner = _runner(banks)
    tokens = 4
    entry = _entry(
        host_row=[[0, 1]],
        host_cached=[0],
        device_row=[[0, 1]],
        device_cached=0,
        tokens=tokens,
        device=cpu,
    )
    entry["host_block_table"] = torch.zeros(
        (1, 2), dtype=torch.int32, device=torch.device("meta")
    )

    with pytest.raises(ValueError) as refused:
        runner._glm5next_model_kwargs(_kwargs(banks, entry, tokens=tokens, device=cpu))

    message = str(refused.value)
    assert "host_block_table" in message
    assert "meta" in message


# ══════════════════════════════════════════════════════════════════════════════
# The view rule: the bank itself, and the rows this step may write.
# ══════════════════════════════════════════════════════════════════════════════
def test_the_carrier_view_spans_whole_pages_and_aliases_the_bank() -> None:
    """The rule the converter implements, read back off one continuing step. """
    cpu = torch.device("cpu")
    banks = [_sparse_bank(cpu)]
    runner = _runner(banks)
    cached, tokens = 5, 6
    _open_ring_at(runner, banks, cached)
    entry = _entry(
        host_row=[[4, 5, 6, 7, 8]],
        host_cached=[cached],
        device_row=[[4, 5, 6, 7, 8]],
        device_cached=cached,
        tokens=tokens,
        device=cpu,
    )

    translated = runner._glm5next_model_kwargs(_kwargs(banks, entry, tokens=tokens, device=cpu))

    carrier = translated["layer_carriers"][0]
    view = carrier["latent_cache"]
    host_row = [4, 5, 6, 7, 8]
    pages = -(-(cached + tokens) // PAGE)
    assert pages == 3
    # A length says nothing about a request, so each clause a window-shaped carrier once
    # supported is read where its risk moved instead: the whole-pages clause is the table's
    # own entries, which are this request's first `pages` and then the bucket's padding, and
    # the coverage clause is one bank row per token of the step, each one inside the page
    # its own position falls in.
    assert int(view.shape[0]) == BANK_SLOTS
    assert carrier["block_table_row"].flatten().tolist() == _padded_table(host_row, pages)
    rows = _physical_rows(host_row, start=cached, tokens=tokens)
    assert carrier["latent_slots"].tolist() == rows
    # The position's value, read here because this test's carrier is on CPU; the captured step's is on
    # `meta`, where the tensor has a form but no value.
    assert int(carrier["start_position"]) == cached
    # The carrier is the bank's own storage, from its first slot, and the write the layer
    # makes goes to the rows it was handed rather than to a window position.
    bank = banks[0]["latent_cache"]
    assert view.data_ptr() == bank.data_ptr()
    view[rows, 0, :] = 1.0
    assert bool(bank[4 * PAGE + cached].eq(1.0).all())
    assert bool(bank[4 * PAGE + cached - 1].eq(0.0).all())


# ══════════════════════════════════════════════════════════════════════════════
# The attention seam reaches its dispatch on a captured step's carrier.
# ══════════════════════════════════════════════════════════════════════════════
def test_the_seam_reaches_its_dispatch_on_meta_tensors() -> None:
    """A precondition that reads values cannot run where values do not exist. """
    from vllm_neuron.functional.attention import mla_sparse as seam

    meta = torch.device("meta")
    banks = [_sparse_bank(meta)]
    runner = _runner(banks)
    tokens = 2 * PAGE
    entry = _entry(
        host_row=[[0, 1]],
        host_cached=[0],
        device_row=[[0, 1]],
        device_cached=0,
        tokens=tokens,
        device=meta,
    )

    carrier = runner._glm5next_model_kwargs(
        _kwargs(banks, entry, tokens=tokens, device=meta)
    )["layer_carriers"][0]
    # The cache side is the bank the carrier hands over, taken whole, because the kernel
    # stages the pages the block table names rather than reading a run off a position. A
    # position cannot be read as a host number on a `meta` carrier, and the paged table
    # removed the need to: no length is derived from it here.
    c_kv = carrier["latent_cache"][:, 0, :]
    q_lift = torch.zeros((1, 1, LATENT_WIDTH), dtype=torch.float32, device=meta)
    selected = torch.zeros((1, seam.KEY_CHUNK), dtype=torch.int32, device=meta)

    seam.reset_mla_sparse_dispatch_counters()
    before = seam.mla_sparse_dispatch_counters()
    raised: Exception | None = None
    try:
        seam.mla_sparse_attention(
            q_lift, c_kv, selected, float(carrier["softmax_scale"])
        )
    except Exception as caught:  # noqa: BLE001 -- the kernel boundary is not measured here
        raised = caught
    after = seam.mla_sparse_dispatch_counters()

    assert after[0] == before[0] + 1, (
        f"the seam's dispatch counter did not move: {before} -> {after}. The call did not "
        f"get past the range refusal"
        + (f"; it raised {type(raised).__name__}: {raised}" if raised else "")
    )
    if raised is not None:
        message = str(raised)
        assert "cannot be called on meta tensors" not in message, (
            f"the seam still read a value off a meta tensor: {message}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# A recurrent row at the table's own width, built the way the runner builds it.
# ══════════════════════════════════════════════════════════════════════════════
def test_a_recurrent_row_at_the_tables_width_is_accepted() -> None:
    """A recurrent bank is handed its KV group's full padded row, and must be served.
    """
    cpu = torch.device("cpu")
    banks = [_linear_bank(cpu)]
    tokens = PAGE
    width = -(-MAX_MODEL_LEN // PAGE)
    slot = STATE_SLOTS - 1
    shapes = {
        "the warmup builder's ascending row": list(range(width)),
        "a served row, its slot first and the table's padding after": (
            [slot] + [0] * (width - 1)
        ),
    }

    served = {}
    for what, row in shapes.items():
        runner = _runner(banks)
        entry = _entry(
            host_row=[row],
            host_cached=[0],
            device_row=[row],
            device_cached=0,
            tokens=tokens,
            device=cpu,
        )

        try:
            translated = runner._glm5next_model_kwargs(
                _kwargs(banks, entry, tokens=tokens, device=cpu)
            )
        except ValueError as refused:
            raise AssertionError(
                f"{what} was refused, and it is what the runner hands every recurrent "
                f"bank: {refused}"
            ) from refused

        carrier = translated["layer_carriers"][0]
        assert len(carrier["recurrent_state"]) == 1, (
            f"{what} names one request, and its carrier holds "
            f"{len(carrier['recurrent_state'])} state view(s)"
        )
        served[what] = carrier["recurrent_state"][0].data_ptr()

    state = banks[0]["recurrent_state"]
    assert len(set(served.values())) == 1, (
        f"the two rows were served from different slots, so the slot still follows the "
        f"block table: {served}"
    )
    assert set(served.values()) == {state[0].data_ptr()}, (
        f"neither row was served from the slot the request table hands a step with no "
        f"request of its own, which is slot 0: {served}"
    )
