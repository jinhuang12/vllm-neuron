# SPDX-License-Identifier: Apache-2.0
"""The carrier geometry of a GLM-5.3-Flash step comes from the host, not from a device tensor.

THE DECLARED ACCEPTANCE:

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 \\
      NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python -m pytest \\
      test/vllm_neuron/model/glm5_next/test_host_geometry_119.py -s -rA \\
      -p no:randomly -p no:cacheprovider

Seven items, one test each, no ``parametrize``.

WHAT THIS FILE IS ABOUT. ``NeuronModelRunner._glm5next_model_kwargs`` decides which cache
pages a step occupies before the traced call runs. It used to read those numbers off
``attn_metadata``'s device tensors with ``int()``, which is ``Tensor.item()``. Under
``VLLM_NEURON_CPU_COMPILE`` the whole batch lives on ``meta``
(``neuron_worker.py:500-501``), where a value does not exist, so every prefill graph
extraction raised. The numbers now come from the entry's ``host_num_computed_tokens`` and
``host_block_table``, which both metadata builders fill from the runner's own host arrays.

THE SIX HOST-GEOMETRY ITEMS

* A01 -- a captured step. The metadata is built the way ``_build_warmup_attention_metadata``
  builds it, with every device tensor and every bank on ``meta``, and the converter must
  produce carriers. This item is the one the failed hardware run reproduces: at the base it
  raises ``Tensor.item() cannot be called on meta tensors``.
* A02 -- the source, made visible. The host row and the device row name DIFFERENT pages, and
  the slice the layer receives must be the host row's. At the base the slice follows the
  device row.
* A03 -- the request count. The device block table carries the padded row count and the host
  arrays carry the batch's own. At the base the padded count is read as the batch and a
  single request is refused as four.
* A04 -- the rule made mechanical. A device tensor placed under a host key is refused BY NAME
  rather than converted, so a call site written later cannot reintroduce the defect quietly.
* A05 -- the view rule. The layer is handed the bank ITSELF, the pages it may write are the
  ones its own block table names, and a write through the rows the carrier hands it lands in
  the bank rather than in a copy. This item used to read the same at the base and no longer
  can: the pages now arrive under two keys the base does not carry.
* A06 -- the second host read on the same path. Every MLA layer of every step calls
  ``mla_sparse_attention`` (``model_fp8.py:6873``), whose range refusal read the selected-row
  range with ``int(...)``. The seam now reaches its dispatch on a captured step's own
  carrier; at the base it raises the same meta read, one seam further along than A01.

THE WIDTH ITEM, ON THE SAME CONVERTER

* B01 -- the width a recurrent row arrives at. Both metadata builders hand every cache group
  the group's own padded row, so a recurrent bank's row is as wide as the table and the one
  slot the scheduler gave it is the row's first entry. The converter must serve such a row
  from that entry. This item reads the same at the base, because the base checks no width
  either; it is here to hold this candidate's own width rule to the shape the runner builds.

THE BASE ARM, DECLARED: A01, A02, A03, A04, A05 and A06 FAIL; B01 PASSES.

RE-PINNED AFTER THE PAGED-WINDOW WORK. The converter hands a layer a window whose length is
the leg's bucket span, so the view is WIDER than the step's own pages and the position
arrives as a tensor rather than an int. Five readings moved: the three view lengths in A01,
A02 and A03, and the position reads in A01 and A06, which asked a ``meta`` tensor for a value
it does not hold. Each original is quoted verbatim where it was replaced. A05 keeps its
base-passing arm on purpose -- it pins whole pages and coverage rather than the exact length,
which the window's own acceptance file measures.

RE-PINNED AGAIN, NOW THAT THE BANK TRAVELS WHOLE. The window slice is gone: a layer is handed
the bank itself, its request's block table as a ``[pages, 1]`` int32 column padded with ``-1``,
and ``latent_slots``, each token's physical bank row. So a view LENGTH no longer says anything
about a request -- every carrier's is the bank's -- and the readings that measured a request
through that length are made on the two new keys instead, where the same risk sits: a table
that named another sequence's pages, or rows that addressed them, is what those readings now
catch. Five items moved: A01, A02, A03 and A05 read the bank plus the pages, and A06 slices
the bank it is handed. A05 loses the base-passing arm this costs, which is recorded above.

CONVENTIONS. The runner is stood up with ``__new__`` and given only the attributes the
converter reads, which is this campaign's landed harness shape
(``test_kda_runner_state.py:209-228``). No layer is driven here: every carrier under
assertion is the one the runner decided to build.
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

#: Recurrent state slots in the linear-attention bank. RE-PINNED: the converter no longer
#: reads a state slot off the block row -- it hands each request a slot of its own table --
#: so this width is the bank's paging geometry and the bound below is the request axis.
#: The original reading, verbatim: "The converter reads a state slot as the first block id
#: of the row, so the bank must be wide enough for the rows below."
STATE_SLOTS = 8

#: How many sequences the modelled engine admits at once. One, and the bank above holds
#: more slots than that on purpose: the two numbers are not the same axis.
DECLARED_MAX_NUM_SEQS = 1
#: The id of that one request. The converter keys a sequence's state by its id and refuses a
#: real step served without one, so a shell with no id can only ever be served the opening
#: bucket -- which no item here reads, and one of them steps past.
DECLARED_REQUEST = "host-geometry-request"

#: The slot a step carrying no request id is served from, which is every step this file
#: drives: the converter takes no claim for such a step and reads slot 0.
SYNTHETIC_SLOT = 0

#: A single-token step is a decode and anything longer is a prefill.
DECODE_THRESHOLD = 1

#: The longest sequence the harness admits, which is what the side-cache allocator bounds
#: its pooled store by (``neuron_model_runner.py:4917-4921``).
MAX_MODEL_LEN = BANK_PAGES * PAGE


def _table_blocks(row) -> int:
    """The block table's width for an entry ``_entry`` built from ``row``.

    The table the converter hands a layer is as wide as the LEG's bucket span capped by the
    declared blocks per sequence, and never as wide as this step's own pages: a width that
    followed the step would be a new graph at every step. Every entry in this file declares
    both numbers off the same row -- the blocks per sequence is the row's width and the
    segment is that width in slots -- so the span always reaches the cap and the table is
    the row's whole width. That is one number per item and it does not move with the
    position, which is the property the padded table exists to give.
    """
    return len(row)


def _padded_table(row, pages: int) -> list[int]:
    """The block ids a carrier built from ``row`` names, at ``pages`` pages of its own.

    The request's pages come first, in the order the row gives them, and every entry the
    bucket pads is ``-1``. Both halves are read: a table that dropped the order would page
    another sequence's rows, and one that padded with a block number would address page 0.
    """
    return [int(value) for value in row[:pages]] + [-1] * (_table_blocks(row) - pages)


def _physical_rows(row, *, start: int, tokens: int) -> list[int]:
    """Each of this step's tokens as a PHYSICAL bank row, by the runner's own formula.

    At one context-parallel rank a slot is ``block_number * block_size + block_offset``
    (``neuron_model_runner.py:330-334``), and that number is the row index of the bank the
    carrier hands over. It is restated here because it is the address the layer's write
    consumes now that the rows of a request are not a run.
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

    The four keys are the ones the carrier builder reads for this family
    (``neuron_model_runner.py:5016-5026``): the name it is looked up by, the family that
    selects the branch, the page its bank is cut into, and the bank flattened over pages and
    slots, which is the sequence view the layers take.
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
    """A runner carrying only what the converter reads.

    Building it with ``__new__`` keeps every other attribute absent, so a converter that
    started reading something new would raise here rather than quietly find a stand-in.
    """
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = SimpleNamespace(
        text_config=_text_config(), glm5next_layer_banks=tuple(banks)
    )
    runner.max_model_len = MAX_MODEL_LEN
    # RE-PINNED: the converter now sizes its per-sequence caches by the engine's
    # concurrent-sequence bound, so a runner shell must model that bound too.
    runner.max_num_reqs = DECLARED_MAX_NUM_SEQS
    # RE-PINNED AGAIN: it also keys each sequence's state by its request id, and a step
    # served without one is the opening bucket's. Every step here is the one request's.
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

    THE KEYS ARE THE RUNNER'S OWN, read off the mapping it builds at
    ``neuron_model_runner.py:4418-4442``. The two halves are given separately here because
    that is the whole measurement: in the runner they carry the same numbers, and an item
    that sets them apart shows which half the converter read.
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
    """Stand the live indexer ring up and declare which position it holds.

    A prefill at position 0 opens the ring inside the converter. A step that continues a
    sequence is refused unless the ring already stands at its position, so an item at a
    non-zero position hands the runner every record the previous step would have left.
    RE-PINNED: the ring is one position PER REQUEST SLOT and this shell names one request,
    so the records are THREE and the slot the request owns is one of them. Handing the
    position alone is not enough: a request absent from the table is a new one, and
    claiming a slot empties whatever position that slot stood at, which would take this
    ring back down before the step reads it. The original reading, verbatim:
    ``runner._glm5next_side_cache_cursor = int(position)``.
    """
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
# A01. A captured step: every device tensor and every bank on ``meta``.
# ══════════════════════════════════════════════════════════════════════════════
def test_a01_a_capture_on_meta_builds_its_carriers() -> None:
    """The graph-extraction world, which is where the hardware run stopped.

    Warmup declares a cached length of 0, one row, and this bucket's own pages
    (``neuron_model_runner.py:4373-4380``, ``neuron_model_runner.py:4431-4442``). The device
    half is on ``meta``, where reading a value is impossible; the host half is the array the
    runner already holds.
    """
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
    # RE-PINNED AGAIN: the carrier is the bank ITSELF, no slice and no copy, so its length
    # is the bank's at every step. The two readings this replaces, verbatim:
    # `assert int(sparse["latent_cache"].shape[0]) == _window_slots(host_row)` and
    # `assert int(sparse["latent_cache"].shape[0]) >= 2 * PAGE`, the second of which kept
    # the original "eight tokens at position 0 occupy two pages of four". Those two pages
    # are still the ones this step writes, and they are named by the two keys below.
    assert sparse["latent_cache"] is banks[0]["latent_cache"]
    assert int(sparse["latent_cache"].shape[0]) == BANK_SLOTS
    # THE PAGES, AS FORM ONLY. This item's carrier is on `meta`, where a tensor has a shape
    # and no values, so what is read here is the shape a captured graph is compiled for: the
    # table is the bucket's width and the rows are one per token of the step. The VALUES are
    # read where they exist, on the CPU steps in A02 and A05.
    assert tuple(sparse["block_table_row"].shape) == (_table_blocks(host_row), 1)
    assert sparse["block_table_row"].dtype == torch.int32
    assert sparse["block_table_row"].device.type == "meta"
    assert tuple(sparse["latent_slots"].shape) == (tokens,)
    assert sparse["latent_slots"].dtype == torch.int64
    assert sparse["latent_slots"].device.type == "meta"
    # RE-PINNED: the position arrives as a tensor -- for the window's own reason, that a
    # host number here is baked into the graph it was captured with -- and this item's
    # carrier is on `meta`, where a value does not exist to be read. The reading this
    # replaces, verbatim:
    # `assert int(sparse["start_position"]) == 0`. What `meta` does answer is the operand's
    # form, which is what a captured graph depends on; the VALUE is read where it is
    # readable, on the continuing CPU step in A05.
    assert tuple(sparse["start_position"].shape) == ()
    assert sparse["start_position"].dtype == torch.int32
    assert sparse["start_position"].device.type == "meta"
    assert sparse["latent_cache"].device.type == "meta"
    # The linear layer's carrier is the bank's slot, which this request owns.
    # RE-PINNED: the carrier holds ONE VIEW PER REQUEST rather than one tensor, because
    # two requests' states are two rows of one bank. The reading this replaces, verbatim:
    # `assert tuple(linear["conv_state"].shape) == (4, 6)`. The same claim is made here on
    # this step's one request's own view.
    assert len(linear["conv_state"]) == 1
    assert tuple(linear["conv_state"][0].shape) == (4, 6)
    # RE-PINNED for the same reason -- a host number at this boundary is a captured
    # constant -- replacing `assert int(linear["start_position"]) == 0`.
    # RE-PINNED AGAIN: the linear carrier carries ONE ROW PER REQUEST, so its position is
    # a 1-D int32 tensor as long as the batch rather than a 0-d one. The reading this
    # replaces, verbatim:
    # `assert tuple(linear["start_position"].shape) == ()`. What that line claimed --
    # that the position reaches the layer as a tensor whose value no captured graph
    # holds -- is what these three lines claim, at this step's one request.
    assert tuple(linear["start_position"].shape) == (1,)
    assert linear["start_position"].dtype == torch.int32
    assert linear["start_position"].device.type == "meta"


# ══════════════════════════════════════════════════════════════════════════════
# A02. Which half the converter read, made visible by making the halves disagree.
# ══════════════════════════════════════════════════════════════════════════════
def test_a02_the_host_row_is_the_one_the_slice_follows() -> None:
    """A serving-shaped step whose two halves name different pages.

    The host row names pages 4 and 5 of the bank; the device row names pages 500 and 501,
    which the bank does not have. So the two halves are told apart by the pages the carrier
    names and by the bank rows it hands the layer, which is a difference read rather than
    inferred: a carrier built from the device row names page 500 and addresses row 2000 of a
    bank that holds 128.
    """
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
    # RE-PINNED AGAIN: the carrier is the whole bank, so its length and its pointer are the
    # bank's whatever row the converter read, and neither can tell the two halves apart any
    # more. The two readings this replaces, verbatim:
    # `assert int(carrier["latent_cache"].shape[0]) == _window_slots(host_row)` and
    # `assert carrier["latent_cache"].data_ptr() == bank[4 * PAGE].data_ptr()`, the second
    # of which carried this item's whole measurement -- which half the converter read.
    assert int(carrier["latent_cache"].shape[0]) == BANK_SLOTS
    assert carrier["latent_cache"].data_ptr() == bank.data_ptr()
    # WHERE THAT MEASUREMENT LIVES NOW. Six tokens at position 0 occupy two pages, and they
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
# A03. The batch's own row count, not the padded one.
# ══════════════════════════════════════════════════════════════════════════════
def test_a03_a_padded_device_table_does_not_decide_the_request_count() -> None:
    """One request, in a device table padded to four rows.

    ``_build_attention_metadata`` sizes its device tensors by ``padded_num_reqs`` and its host
    arrays by ``num_reqs`` (``neuron_model_runner.py:4237-4244``). A padding row names no
    request, so reading the padded height as the batch refuses a step that is single.
    """
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

    # RE-PINNED AGAIN: the length is the bank's, at every position and for every request.
    # The reading this replaces, verbatim:
    # `assert int(carrier["latent_cache"].shape[0]) == _window_slots(host_row)`, which
    # replaced "four tokens at position 0 occupy one page" --
    # `assert int(translated["layer_carriers"][0]["latent_cache"].shape[0]) == PAGE`. What
    # this item measures is that the step was NOT refused as a four-request batch, and a
    # carrier that exists at all is that reading; the two lines after it say what that
    # carrier holds -- this request's one page, and the bucket's second entry padded.
    carrier = translated["layer_carriers"][0]
    assert int(carrier["latent_cache"].shape[0]) == BANK_SLOTS
    assert carrier["block_table_row"].flatten().tolist() == _padded_table(host_row, 1)
    assert carrier["latent_slots"].tolist() == _physical_rows(
        host_row, start=0, tokens=tokens
    )


# ══════════════════════════════════════════════════════════════════════════════
# A04. The rule, mechanical: a device tensor under a host key is refused.
# ══════════════════════════════════════════════════════════════════════════════
def test_a04_a_device_tensor_under_a_host_key_is_refused_by_name() -> None:
    """The refusal that keeps the rule true for a call site written later.

    Converting the tensor instead would be the defect again, one lap removed: it reads on
    CPU and raises inside a capture.
    """
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
# A05. The view rule: the bank itself, and the rows this step may write.
# ══════════════════════════════════════════════════════════════════════════════
def test_a05_the_carrier_view_spans_whole_pages_and_aliases_the_bank() -> None:
    """The rule the converter implements, read back off one continuing step.

    THE RULE. The layer is handed the bank, and the pages the request's own tokens occupy
    are named beside it: ``ceil((start_position + tokens) / page)`` entries of its block
    table, from the request's first page, and one physical bank row per token of the step.

    WHY THE ALIAS MATTERS. The layer writes this step's latents THROUGH what it is handed
    (``model_fp8.py:6855``) at the rows it is handed, and a later step reads them back out
    of the bank. The carrier is the bank itself, so both land there. A copy would be
    discarded where the next step reads.

    RE-PINNED. The length is now the bucket's window and the read is the whole of it,
    because the old length was derived from the position and therefore held only for the
    step it was captured at. The
    rule this replaces, verbatim: "The slice spans the whole pages the request's own tokens
    occupy, counted from the request's first page: ``ceil((start_position + tokens) / page)``
    pages. It therefore covers ``start_position + tokens`` slots, which is the bound the
    layer checks before it writes". Every clause of it that this item can still read is
    still read below: the view starts at the request's first page, it COVERS the step's own
    slots, and it aliases the bank. What changed is that covering is no longer exactness --
    a length that tracked the position is what pinned a captured graph to one position.

    RE-PINNED AGAIN, AND THE BASE-PASSING ARM GOES WITH IT. There is no slice left to read a
    request out of: the carrier is the bank, so the clauses above are read on the two keys
    the pages arrive under, which the base does not carry. The whole-pages clause is the
    table's entries, the coverage clause is the physical row of every token, and the alias
    clause is the write below, which now goes through those rows.
    """
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
    # RE-PINNED AGAIN: the three readings this replaces, verbatim --
    # `assert int(view.shape[0]) % PAGE == 0`, `assert int(view.shape[0]) >= pages * PAGE`
    # and `assert int(view.shape[0]) >= cached + tokens` -- were what survived of
    # `assert int(view.shape[0]) == pages * PAGE` while the carrier was a window. A length
    # says nothing about a request now, so each clause is read where its risk moved: the
    # whole-pages clause is the table's own entries, which are this request's first `pages`
    # and then the bucket's padding, and the coverage clause is one bank row per token of
    # the step, each one inside the page its own position falls in.
    assert int(view.shape[0]) == BANK_SLOTS
    assert carrier["block_table_row"].flatten().tolist() == _padded_table(host_row, pages)
    rows = _physical_rows(host_row, start=cached, tokens=tokens)
    assert carrier["latent_slots"].tolist() == rows
    # The position's VALUE, read here because this item's carrier is on CPU: A01's is on
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
# A06. The attention seam reaches its dispatch on a captured step's carrier.
# ══════════════════════════════════════════════════════════════════════════════
def test_a06_the_seam_reaches_its_dispatch_on_meta_tensors() -> None:
    """A precondition that reads values cannot run where values do not exist.

    ``mla_sparse_attention`` refused an out-of-range selected row by reading the row range
    with ``int(...)`` (``mla_sparse.py:1411``), which is the same call the converter used to
    make and which a ``meta`` tensor cannot answer. Every MLA layer of every step goes
    through that seam (``model_fp8.py:6873``, unconditionally), so a captured prefill reached
    it and stopped there.

    WHAT THIS ITEM MEASURES AND WHAT IT DOES NOT. It measures that the call gets PAST the
    precondition: the seam's own dispatch counter, one line beyond it, moves. Whatever the
    kernel boundary does with ``meta`` inputs after that is NOT measured here -- that is
    a vendor question and ``D01`` in ``test_meta_forward_119.py`` reports it.

    THE GEOMETRY IS THE CONVERTER'S. The cache side is the carrier the runner built, taken
    the way the layer takes it (``model_fp8.py:6855``, ``model_fp8.py:6859``) -- the bank
    whole -- and the scale is the carrier's own. The selected-row width is the seam's
    declared tile, ``KEY_CHUNK``, imported rather than typed: the admissibility clause
    requires a positive multiple of it (``mla_sparse.py:1293-1299``).
    """
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
    # RE-PINNED: the cache side is the bank the carrier hands over, taken whole, because the
    # kernel stages the pages the block table names rather than reading a run off a position.
    # The reading this replaces, verbatim:
    # `start = int(carrier["start_position"])` then
    # `c_kv = carrier["latent_cache"][: start + tokens, 0, :]`. That `int()` cannot answer on
    # a `meta` carrier, and reading a length off the position is what the paged table removed.
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
        f"get past the range refusal, which is the line this increment repairs"
        + (f"; it raised {type(raised).__name__}: {raised}" if raised else "")
    )
    if raised is not None:
        message = str(raised)
        assert "cannot be called on meta tensors" not in message, (
            f"the seam still read a value off a meta tensor: {message}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# B01. A recurrent row at the table's own width, built the way the runner builds it.
# ══════════════════════════════════════════════════════════════════════════════
def test_b01_a_recurrent_row_at_the_tables_width_is_accepted() -> None:
    """A recurrent bank is handed its KV group's full padded row, and must be served.

    NEITHER BUILDER NARROWS A ROW TO THE ONE SLOT A RECURRENT BANK USES. The warmup builder
    writes ``torch.arange(max_num_blocks_per_req)`` (``neuron_model_runner.py:4437-4442``) and
    the serving builder slices the group's own table (``neuron_model_runner.py:4246``), whose
    width is ``max_model_len`` over the page size, so a converter that asked a recurrent row to
    be one entry wide would refuse every hybrid step. Both shapes are driven here, each named in
    its own failure message.

    RE-PINNED. The state slot no longer comes from the block table at all: it is the request's
    own, handed out by the slot table the runner keys on request id, and the carrier holds one
    view per request rather than one tensor. The reading this replaces, verbatim: "The slot the
    scheduler allocated is the row's first entry and the rest is the table's padding" -- graded
    as ``carrier["recurrent_state"].data_ptr() == state[row[0]].data_ptr()``. What survives is
    the acceptance this item is named for: the full-width row is served rather than refused. What
    replaces the slot clause is the property that made the change worth making -- the two rows
    name different first entries, and both are now served from ONE slot, which is the request's.
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
