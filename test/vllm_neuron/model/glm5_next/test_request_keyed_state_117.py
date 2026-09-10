# SPDX-License-Identifier: Apache-2.0
"""Acceptance for the request-keyed cache state: the runner half.

THE DECLARED ACCEPTANCE, the Tier N harness this campaign uses:

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
      NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python -m pytest \\
      test/vllm_neuron/model/glm5_next/test_request_keyed_state_117.py \\
      -s -rA -p no:randomly -p no:cacheprovider --timeout 60

EVERY ITEM IN THIS FILE FAILS AT THE BASE COMMIT, AND THAT IS THE POINT. The
converter refuses a block table carrying more than one row
(``neuron_model_runner.py:5136-5141``), so at base each item below raises that
refusal instead of reading a value. An item that passed at base would be
measuring nothing.

THE FOUR ITEMS HERE, and each names the tripwire it must fail on.

* A1 -- two admitted requests get DIFFERENT recurrent state slots, and each
  layer's carrier is a VIEW of its own slot row rather than a copy. A copy would
  send the layer's in-place advance somewhere the next step never reads.
  Tripwire: a slot table that returns one slot for both requests.
* A3 -- one request's pooled store and tail ring are byte-unchanged by a step of
  the other request. Tripwire: the process-wide allocation, which shares one set
  across every request.
* A4 -- the per-token operands for a two-request batch equal the two
  per-request derivations concatenated, exactly. Tripwire: a derivation from a
  single start position, which is what the base does.
* A7 -- a padded decode row writes NOWHERE. The padding value is zero
  (``neuron_model_runner.py:95``) and zero is a real cache slot, not a sentinel,
  so an unmasked write lands a padded row's latent in a slot a live request can
  own. Tripwire: dropping the mask. The item also asserts both real writes
  landed, so a write path disabled altogether fails it too.

THE THREE REMAINING ITEMS (slot reuse, the physical scattered write, and the
refusal that still holds for speculative decoding) live in the next commit of
this increment, not in another file.

WHY THE HARNESS IS RE-AUTHORED HERE rather than imported. The two landed files
with a converter harness -- ``test_kda_runner_state.py`` and
``tiny/test_tiny_glm5next_e2e.py`` -- each state that their harness keeps their
own file's single writer. Importing either would make that file a dependency of
this one and take the property away from it. The shapes below are theirs; the
code is this file's own.

CONVENTIONS. ``model_fp8`` is never imported at module level, because
``test_factory.py:318-319`` is a landed assertion that it stays out of
``sys.modules``. The runner is stood up with ``__new__`` and given only the
attributes the converter reads, so a converter that started reading something
new raises here instead of quietly finding a stand-in.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.vllm.worker.neuron_model_runner import (
    NULL_BLOCK_ID,
    NeuronModelRunner,
)

# ---------------------------------------------------------------------------
# Declared values. Each is either read off the runner or declared here because
# nothing landed binds it; the difference is stated per value.
# ---------------------------------------------------------------------------

#: Two requests is the smallest batch that can show request keying at all: one
#: request cannot collide with itself.
DECLARED_REQUESTS = 2

#: The recurrent-state bank's slot count. Four rather than two so that a slot
#: number can never be confused with a block id or a request index by accident.
DECLARED_STATE_SLOTS = 4

#: The page. Nothing landed binds a page for a synthetic bank, so this file
#: declares one and the bank below is built to it; the converter cross-checks the
#: two against each other, which is the property A4 relies on.
DECLARED_PAGE_SIZE = 32

#: The latent bank's block count and its width. The width is the only shape the
#: sparse branch of the carrier builder reads off the bank.
DECLARED_BLOCKS = 8
DECLARED_HEAD_SIZE = 64
DECLARED_KV_HEADS = 1

#: The indexer's two widths. The side-cache allocator reads both and derives its
#: row count from them, so they are named rather than inlined.
DECLARED_INDEX_KPOOL = 8
DECLARED_INDEX_HEAD_DIM = 32

#: The leg test the converter makes is ``max_query_len > decode_token_threshold``,
#: so one means a single-token step reads as a decode.
DECLARED_DECODE_THRESHOLD = 1

#: The two requests' cached lengths. They DIFFER on purpose: equal lengths would
#: let a single-start-position derivation pass A4 by coincidence.
DECLARED_CACHED_LENGTHS = (7, 19)

#: The two requests' block rows. Neither is a prefix of the other and the second
#: is deliberately not one ascending run continuing the first, so a stack-wide
#: contiguous slice cannot serve both.
DECLARED_SPARSE_ROWS = ((0, 1), (4, 5))

#: The decode bucket A7 pads up to. Two real requests in a bucket of four leaves
#: two padded rows, which is the case the padding writer produces.
DECLARED_DECODE_BUCKET = 4

#: The state-carrier keys, in the model's own declared spelling
#: (``model_fp8.py:4149``). The carrier is splatted into the layer, so an extra
#: key is a TypeError and a missing one is served as a default.
DECLARED_STATE_CARRIER_KEYS = {"conv_state", "recurrent_state", "is_prefill"}

#: The conv and recurrent state shapes per slot. Small, and their only
#: requirement is that the two differ so a mixed-up carrier is visible.
DECLARED_CONV_SHAPE = (4, 16)
DECLARED_RECURRENT_SHAPE = (2, 8, 8)


class VacuousControlError(AssertionError):
    """A control that cannot discriminate is a failure, never a pass."""


def _require_cpu_mode() -> None:
    """The environment is part of the acceptance command, never a fixture.

    The declared command sets ``VLLM_NEURON_CPU_MODE=1``. Reading it rather than
    setting it is what keeps the command the instrument: a file that set the
    variable itself would pass under a command that did not.
    """
    if os.environ.get("VLLM_NEURON_CPU_MODE") != "1":
        pytest.fail(
            "this file's declared acceptance runs under VLLM_NEURON_CPU_MODE=1; "
            f"the environment carries {os.environ.get('VLLM_NEURON_CPU_MODE')!r}"
        )


# ---------------------------------------------------------------------------
# The world: a two-layer hybrid stack, one sparse layer and one linear layer.
# ---------------------------------------------------------------------------
def _text_config() -> SimpleNamespace:
    """Only the fields the converter reads off ``model.text_config``.

    Five of them: the two indexer widths, and the two head widths whose sum's
    inverse square root is the softmax scale the converter passes down. Anything
    else absent means a converter that grew a new read raises here.
    """
    return SimpleNamespace(
        index_kpool=DECLARED_INDEX_KPOOL,
        index_head_dim=DECLARED_INDEX_HEAD_DIM,
        qk_nope_head_dim=DECLARED_HEAD_SIZE,
        qk_rope_head_dim=0,
    )


def _banks() -> list[dict]:
    """The bank mapping ``bind_kv_cache`` leaves on the model, built its way.

    ONE SPARSE BANK AND ONE LINEAR BANK, because the stack this campaign ports is
    hybrid and the two families take different carriers. The sparse bank keeps
    both views the model keeps (``model_fp8.py:9129-9141``): the paged bank and
    its flattened sequence view, which at one KV head is the same slot order the
    runner's own slot mapping produces.
    """
    latent_bank = torch.zeros(
        (DECLARED_BLOCKS, DECLARED_KV_HEADS, DECLARED_PAGE_SIZE, DECLARED_HEAD_SIZE),
        dtype=torch.bfloat16,
    )
    sparse = {
        "name": "model.layers.0.self_attn",
        "layer_index": 0,
        "family": "self_attn",
        "latent_bank": latent_bank,
        "latent_cache": latent_bank.view(
            DECLARED_BLOCKS * DECLARED_PAGE_SIZE, DECLARED_KV_HEADS, DECLARED_HEAD_SIZE
        ),
        "blocks": DECLARED_BLOCKS,
        "block_size": DECLARED_PAGE_SIZE,
        "slots": DECLARED_BLOCKS * DECLARED_PAGE_SIZE,
        "head_size": DECLARED_HEAD_SIZE,
    }
    linear = {
        "name": "model.layers.1.attention",
        "layer_index": 1,
        "family": "linear_attn",
        "state_slots": DECLARED_STATE_SLOTS,
        "conv_state": torch.zeros(
            (DECLARED_STATE_SLOTS, *DECLARED_CONV_SHAPE), dtype=torch.bfloat16
        ),
        "recurrent_state": torch.zeros(
            (DECLARED_STATE_SLOTS, *DECLARED_RECURRENT_SHAPE), dtype=torch.bfloat16
        ),
    }
    return [sparse, linear]


def _runner(banks) -> NeuronModelRunner:
    """A runner carrying ONLY what the converter reads.

    ``input_batch.req_ids`` is here because the request identity the slot table
    keys on is the runner's own batch-ordered list, paired with block-table rows
    by request index (``neuron_model_runner.py:1126``, ``:2108``). Building the
    object with ``__new__`` keeps every other attribute absent.
    """
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = SimpleNamespace(
        text_config=_text_config(), glm5next_layer_banks=tuple(banks)
    )
    runner.max_model_len = DECLARED_BLOCKS * DECLARED_PAGE_SIZE
    runner.input_batch = SimpleNamespace(
        req_ids=[f"req-{index}" for index in range(DECLARED_REQUESTS)]
    )
    return runner


def _entry(*, rows, tokens: int, cached, block_size: int) -> dict:
    """ONE KV-cache group's attention-metadata entry, at this step's geometry.

    THE KEYS ARE THE RUNNER'S OWN, read off the mapping it builds
    (``neuron_model_runner.py:4417-4429``). ``rows`` carries one block row PER
    REQUEST and ``cached`` one cached length per request, which is the only
    difference from the landed single-request helpers this shape comes from.
    """
    table = torch.tensor([[int(value) for value in row] for row in rows], dtype=torch.int32)
    lengths = torch.tensor([int(value) for value in cached], dtype=torch.int32)
    return {
        "block_table_tensor": table,
        "full_block_table_tensor": table,
        "slot_mapping": _physical_slots(rows=rows, cached=cached, tokens=tokens,
                                        block_size=block_size),
        "max_query_len": int(tokens),
        "block_size": int(block_size),
        "max_blocks_per_seq": int(table.shape[1]),
        "decode_token_threshold": DECLARED_DECODE_THRESHOLD,
        "cached_seq_len": lengths,
        "kv_segment_size": int(table.shape[1]) * int(block_size),
    }


def _physical_slots(*, rows, cached, tokens: int, block_size: int,
                    padded_rows: int = 0) -> torch.Tensor:
    """The physical slot per token, derived the way the runner derives it.

    THE FORMULA IS THE RUNNER'S, not this file's: at one context-parallel rank it
    is ``block_number * block_size + block_offset``
    (``neuron_model_runner.py:330-334``). Padded rows carry the padding value the
    runner writes (``:3571-3574``), which is zero and therefore a real slot.
    """
    slots: list[int] = []
    for row, start in zip(rows, cached):
        for offset in range(int(tokens)):
            position = int(start) + offset
            block = int(row[position // int(block_size)])
            slots.append(block * int(block_size) + position % int(block_size))
    slots.extend([NULL_BLOCK_ID] * int(padded_rows))
    return torch.tensor(slots, dtype=torch.int32)


def _metadata(banks, *, tokens: int, sparse_rows, state_rows, cached) -> dict:
    """TWO KV-cache groups, each with its own table, keyed by every layer name.

    This is the mapping a hybrid stack produces: one entry per group written under
    every layer name of that group (``neuron_model_runner.py:4256-4257``). The two
    groups carry different rows on purpose, because a converter reading one entry
    for the whole stack would slice one family out of the other's table.
    """
    sparse = _entry(rows=sparse_rows, tokens=tokens, cached=cached,
                    block_size=DECLARED_PAGE_SIZE)
    state = _entry(rows=state_rows, tokens=tokens, cached=cached,
                   block_size=DECLARED_PAGE_SIZE)
    return {
        str(bank["name"]): (sparse if bank["family"] == "self_attn" else state)
        for bank in banks
    }


def _generic(*, tokens: int, metadata: dict) -> dict:
    """One step's generic runner kwargs at a given metadata mapping.

    The ids are zeros of the right length: the converter reads
    ``input_ids.shape[0]`` and nothing else off them.
    """
    return {
        "input_ids": torch.zeros(int(tokens), dtype=torch.long),
        "positions": torch.arange(int(tokens), dtype=torch.long),
        "attn_metadata": metadata,
        "sampling_positions": torch.tensor([int(tokens) - 1], dtype=torch.long),
        "sampling_params": None,
        "spec_decode_metadata": None,
        "rank": None,
        "logit_mask": None,
    }


def _two_request_decode(runner, banks):
    """One decode step carrying one token for each of the two requests.

    Returned as the converter's own output, because every item below reads what
    the runner decided to hand the layers and never builds a carrier itself.
    """
    metadata = _metadata(
        banks,
        tokens=DECLARED_REQUESTS,
        sparse_rows=DECLARED_SPARSE_ROWS,
        state_rows=DECLARED_SPARSE_ROWS,
        cached=DECLARED_CACHED_LENGTHS,
    )
    converted = runner._glm5next_model_kwargs(_generic(
        tokens=DECLARED_REQUESTS, metadata=metadata
    ))
    assert sorted(converted) == ["input_ids", "layer_carriers", "sampling_positions"], (
        f"the converter returned {sorted(converted)}"
    )
    return converted["layer_carriers"]
