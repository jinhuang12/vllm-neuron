# SPDX-License-Identifier: Apache-2.0
"""The carrier geometry of a GLM-5.3-Flash step comes from the host, not from a device tensor.

THE DECLARED ACCEPTANCE:

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 \\
      NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python -m pytest \\
      test/vllm_neuron/model/glm5_next/test_host_geometry_119.py -s -rA \\
      -p no:randomly -p no:cacheprovider

Six items, one test each, no ``parametrize``.

WHAT THIS FILE IS ABOUT. ``NeuronModelRunner._glm5next_model_kwargs`` decides which cache
pages a step occupies before the traced call runs. It used to read those numbers off
``attn_metadata``'s device tensors with ``int()``, which is ``Tensor.item()``. Under
``VLLM_NEURON_CPU_COMPILE`` the whole batch lives on ``meta``
(``neuron_worker.py:500-501``), where a value does not exist, so every prefill graph
extraction raised. The numbers now come from the entry's ``host_num_computed_tokens`` and
``host_block_table``, which both metadata builders fill from the runner's own host arrays.

THE SIX ITEMS

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
* A05 -- the view rule, and the base-passing control. The slice handed to a layer spans whole
  pages from the request's first page, covers ``start_position + tokens``, and is an ALIAS of
  the bank rather than a copy. This item reads the same at the base, because it is about the
  rule the converter has always implemented rather than about the source it reads.
* A06 -- the second host read on the same path. Every MLA layer of every step calls
  ``mla_sparse_attention`` (``model_fp8.py:6895``), whose range refusal read the selected-row
  range with ``int(...)``. The seam now reaches its dispatch on a captured step's own
  carrier; at the base it raises the same meta read, one seam further along than A01.

THE BASE ARM, DECLARED: A01, A02, A03, A04 and A06 FAIL and A05 PASSES.

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

#: Recurrent state slots in the linear-attention bank. The converter reads a state slot as
#: the first block id of the row, so the bank must be wide enough for the rows below.
STATE_SLOTS = 8

#: A single-token step is a decode and anything longer is a prefill.
DECODE_THRESHOLD = 1

#: The longest sequence the harness admits, which is what the side-cache allocator bounds
#: its pooled store by (``neuron_model_runner.py:4916-4920``).
MAX_MODEL_LEN = BANK_PAGES * PAGE


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
    (``neuron_model_runner.py:5015-5025``): the name it is looked up by, the family that
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
    ``neuron_model_runner.py:4417-4441``. The two halves are given separately here because
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
    sequence is refused unless the ring already stands at its position
    (``neuron_model_runner.py:5338-5345``), so an item at a non-zero position hands the
    runner the same two attributes the previous step would have left.
    """
    runner._glm5next_side_cache_set = NeuronModelRunner._glm5next_side_caches(
        banks,
        index_kpool=int(_text_config().index_kpool),
        index_head_dim=int(_text_config().index_head_dim),
        max_seq_len=MAX_MODEL_LEN,
    )
    runner._glm5next_side_cache_cursor = int(position)


# ══════════════════════════════════════════════════════════════════════════════
# A01. A captured step: every device tensor and every bank on ``meta``.
# ══════════════════════════════════════════════════════════════════════════════
def test_a01_a_capture_on_meta_builds_its_carriers() -> None:
    """The graph-extraction world, which is where the hardware run stopped.

    Warmup declares a cached length of 0, one row, and this bucket's own pages
    (``neuron_model_runner.py:4372-4379``, ``:4430-4441``). The device half is on ``meta``,
    where reading a value is impossible; the host half is the array the runner already holds.
    """
    meta = torch.device("meta")
    banks = [_sparse_bank(meta), _linear_bank(meta)]
    runner = _runner(banks)
    tokens = 8
    entry = _entry(
        host_row=[[0, 1, 2, 3]],
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
    # Eight tokens at position 0 occupy two pages of four.
    assert int(sparse["latent_cache"].shape[0]) == 2 * PAGE
    assert int(sparse["start_position"]) == 0
    assert sparse["latent_cache"].device.type == "meta"
    # The linear layer's carrier is the bank's slot, which the row's first id names.
    assert tuple(linear["conv_state"].shape) == (4, 6)
    assert int(linear["start_position"]) == 0


# ══════════════════════════════════════════════════════════════════════════════
# A02. Which half the converter read, made visible by making the halves disagree.
# ══════════════════════════════════════════════════════════════════════════════
def test_a02_the_host_row_is_the_one_the_slice_follows() -> None:
    """A serving-shaped step whose two halves name different pages.

    The host row names pages 4 and 5 of the bank; the device row names pages 500 and 501,
    which the bank does not have. A slice taken from the device row is empty, so the item
    reads the difference rather than inferring it.
    """
    cpu = torch.device("cpu")
    banks = [_sparse_bank(cpu)]
    runner = _runner(banks)
    tokens = 6
    entry = _entry(
        host_row=[[4, 5, 6, 7]],
        host_cached=[0],
        device_row=[[500, 501, 502, 503]],
        device_cached=0,
        tokens=tokens,
        device=cpu,
    )

    translated = runner._glm5next_model_kwargs(_kwargs(banks, entry, tokens=tokens, device=cpu))

    carrier = translated["layer_carriers"][0]
    # Six tokens at position 0 occupy two pages, and they are the host row's first two.
    assert int(carrier["latent_cache"].shape[0]) == 2 * PAGE
    bank = banks[0]["latent_cache"]
    assert carrier["latent_cache"].data_ptr() == bank[4 * PAGE].data_ptr()


# ══════════════════════════════════════════════════════════════════════════════
# A03. The batch's own row count, not the padded one.
# ══════════════════════════════════════════════════════════════════════════════
def test_a03_a_padded_device_table_does_not_decide_the_request_count() -> None:
    """One request, in a device table padded to four rows.

    ``_build_attention_metadata`` sizes its device tensors by ``padded_num_reqs`` and its host
    arrays by ``num_reqs`` (``neuron_model_runner.py:4236-4243``). A padding row names no
    request, so reading the padded height as the batch refuses a step that is single.
    """
    cpu = torch.device("cpu")
    banks = [_sparse_bank(cpu)]
    runner = _runner(banks)
    tokens = 4
    entry = _entry(
        host_row=[[0, 1]],
        host_cached=[0],
        device_row=[[0, 1], [0, 0], [0, 0], [0, 0]],
        device_cached=0,
        tokens=tokens,
        device=cpu,
    )

    translated = runner._glm5next_model_kwargs(_kwargs(banks, entry, tokens=tokens, device=cpu))

    assert int(translated["layer_carriers"][0]["latent_cache"].shape[0]) == PAGE


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
# A05. The view rule, and the control that passes at the base.
# ══════════════════════════════════════════════════════════════════════════════
def test_a05_the_carrier_view_spans_whole_pages_and_aliases_the_bank() -> None:
    """The rule the converter implements, read back off one continuing step.

    THE RULE. The slice spans the whole pages the request's own tokens occupy, counted from
    the request's first page: ``ceil((start_position + tokens) / page)`` pages. It therefore
    covers ``start_position + tokens`` slots, which is the bound the layer checks before it
    writes (``model_fp8.py:6862-6867``), and its rows are the request's own.

    WHY THE ALIAS MATTERS. The layer writes this step's latents THROUGH the slice
    (``model_fp8.py:6877``) and reads slot 0 to the last written slot back out of it
    (``model_fp8.py:6881``). A basic slice is a view, so both land in the bank. A gather
    would return a copy, and the write would be discarded where the next step reads.
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
    pages = -(-(cached + tokens) // PAGE)
    assert pages == 3
    assert int(view.shape[0]) == pages * PAGE
    # The layer's own bound: the step's last slot is inside the slice it was handed.
    assert int(view.shape[0]) >= cached + tokens
    # The slice starts at the request's first page, and it is the bank's own storage.
    bank = banks[0]["latent_cache"]
    assert view.data_ptr() == bank[4 * PAGE].data_ptr()
    view[cached : cached + tokens, 0, :] = 1.0
    assert bool(bank[4 * PAGE + cached].eq(1.0).all())
    assert bool(bank[4 * PAGE + cached - 1].eq(0.0).all())


# ══════════════════════════════════════════════════════════════════════════════
# A06. The attention seam reaches its dispatch on a captured step's carrier.
# ══════════════════════════════════════════════════════════════════════════════
def test_a06_the_seam_reaches_its_dispatch_on_meta_tensors() -> None:
    """A precondition that reads values cannot run where values do not exist.

    ``mla_sparse_attention`` refused an out-of-range selected row by reading the row range
    with ``int(...)`` (``mla_sparse.py:1400``), which is the same call the converter used to
    make and which a ``meta`` tensor cannot answer. Every MLA layer of every step goes
    through that seam (``model_fp8.py:6895``, unconditionally), so a captured prefill reached
    it and stopped there.

    WHAT THIS ITEM MEASURES AND WHAT IT DOES NOT. It measures that the call gets PAST the
    precondition: the seam's own dispatch counter, one line beyond it, moves. Whatever the
    kernel boundary does with ``meta`` inputs after that is NOT measured here -- that is
    a vendor question and ``D01`` in ``test_meta_forward_119.py`` reports it.

    THE GEOMETRY IS THE CONVERTER'S. The cache side is the carrier the runner built, sliced
    the way the layer slices it (``model_fp8.py:6877``, ``:6881``), and the scale is the
    carrier's own. The selected-row width is the seam's declared tile, ``KEY_CHUNK``,
    imported rather than typed: the admissibility clause requires a positive multiple of it
    (``mla_sparse.py:1379-1383``).
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
    start = int(carrier["start_position"])
    c_kv = carrier["latent_cache"][: start + tokens, 0, :]
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
