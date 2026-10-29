# SPDX-License-Identifier: Apache-2.0
"""In-place write of one KDA slot's conv rows and recurrent state into its page.

A prefill chunk ends by storing its last conv rows and its final state in the
sequence's slot, one page of the KV cache's int8 page buffer, laid out as the decode
kernel reads it: the conv rows from the page's first byte, then the state. The
kernel copies each part to its place in the slot's page, the slot read at run time,
and writes no other byte, so the buffer is updated in place and returned. Without a
Neuron device or the NKI simulator, the same bytes go in by an ``index_copy_`` of
the slot's whole page.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.kda.fused_decode import (
    BY_SLOT,
    CONV_BYTES,
    CONV_WIDTH,
    DIM,
    HEADS,
    ROWS,
    STATE_BYTES,
    STATIC,
    _conv_rows,
    _state_rows,
)
from vllm_neuron.utils.hardware_config import get_logical_nc_config
from vllm_neuron.utils.neuron_utils import can_run_kernel


def kda_page_row_write(
    pages: torch.Tensor, slot: torch.Tensor, conv: torch.Tensor, state: torch.Tensor
) -> torch.Tensor:
    """Write one slot's conv rows and state into its page in place; returns the buffer.

    Args:
        pages: [slots, page bytes] contiguous int8 page buffer, updated in place.
        slot: [1] integer, the page to write.
        conv: [1, 3, 1152] bf16 conv rows, oldest first, channels q | k | v; they go to
            the page's first 6,912 bytes.
        state: [1, 3, 128, 128] fp32 recurrent state per head, [K, V]; it goes right
            after the conv rows.

    Returns:
        The page buffer, with the slot's first 203,520 bytes replaced and no other byte
        written.
    """
    assert pages.dim() == 2 and pages.dtype == torch.int8, f"pages {tuple(pages.shape)} {pages.dtype}"
    assert pages.is_contiguous(), f"pages strides {pages.stride()}: pass the raw page buffer, not a view"
    assert pages.shape[1] >= CONV_BYTES + STATE_BYTES and pages.shape[1] % 4 == 0, \
        f"a {pages.shape[1]}-byte page cannot hold the states at fp32 alignment"
    assert slot.numel() == 1, f"slot {tuple(slot.shape)}: one page per call"
    assert conv.shape == (1, ROWS, CONV_WIDTH) and conv.dtype == torch.bfloat16, \
        f"conv {tuple(conv.shape)} {conv.dtype}"
    assert state.shape == (1, HEADS, DIM, DIM) and state.dtype == torch.float32, \
        f"state {tuple(state.shape)} {state.dtype}"
    if not can_run_kernel(pages):
        return _index_copy_write(pages, slot, conv, state)
    kernel = wrap_nki(kda_page_row_write_kernel)[get_logical_nc_config()]
    return kernel(pages=pages, slot=slot.reshape(1, 1).int(), conv=conv.contiguous(),
                  state=state.contiguous())


@nki.jit
def kda_page_row_write_kernel(pages, slot, conv, state):
    """The write of :func:`kda_page_row_write`; the page buffer is written in place."""
    # One page's write is one program's work; the other programs of the logical core idle.
    if nl.program_id(0) == 0:
        idx = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx, src=slot, dge_mode=STATIC)
        rows = nl.ndarray((ROWS, CONV_WIDTH), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=rows, src=conv.ap(pattern=[[CONV_WIDTH, ROWS], [1, CONV_WIDTH]], offset=0),
                      dge_mode=STATIC)
        held = nl.ndarray((DIM, HEADS, DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=held, src=state.ap(pattern=[[DIM, DIM], [DIM * DIM, HEADS], [1, DIM]], offset=0),
                      dge_mode=STATIC)
        at = idx.ap(pattern=[[1, 1], [1, 1]], offset=0)
        nisa.dma_copy(dst=_conv_rows(pages.view(nl.bfloat16), at), src=rows, dge_mode=BY_SLOT)
        nisa.dma_copy(dst=_state_rows(pages.view(nl.float32), at), src=held, dge_mode=BY_SLOT)
    return pages


def _index_copy_write(pages, slot, conv, state):
    """The kernel's bytes by an ``index_copy_`` of the slot's whole page."""
    index = slot.reshape(1).long()
    parts = [part.contiguous().view(torch.int8).reshape(1, -1) for part in (conv, state)]
    tail = pages.index_select(0, index)[:, CONV_BYTES + STATE_BYTES:]
    return pages.index_copy_(0, index, torch.cat([*parts, tail], dim=1))
