# SPDX-License-Identifier: Apache-2.0
"""Write rows of state into pages of a KV pool tensor, one page per block id, in NKI.

``write_page_rows(pages, block_ids, rows, offsets, page_bytes)`` writes the bytes of row
``i`` of each carrier ``rows[c]`` into page ``block_ids[i]`` of ``pages``, from byte
``offsets[c]`` of the page, and returns the written pages. The page's other bytes are left
as they are. The KDA checkpoint store (``vllm_neuron.vllm.worker.glm5next_state_banks``)
keeps one committed state row of every carrier per block this way, in the block's page of
the pool tensor the layer shares with its latent layer.

``pages`` is any contiguous tensor whose bytes are whole pages, page ``b`` at bytes
``b * page_bytes``: the store's int8 page view, or the latent layer's cache view of the
same pool tensor itself, so that a graph that writes both has one written input per
storage.

On the device one kernel launch moves every carrier. Per 128-row tile it reads the block
ids into SBUF; per free-dim tile of each carrier it moves the rows into SBUF with a plain
DMA and out to their pages with one indirect DMA on the page axis. An indirect DMA needs
its other side in SBUF (neuronx-cc refuses an HBM-to-HBM one, ``NCC_IBIR231``). The
kernel views each operand as bytes inside the kernel, so no dtype view or bitcast runs in
torch and no instruction but a DMA touches the bytes. It returns ``pages``: the backend's
NKI pass records the operand alias, and its aliasing pass makes ``pages`` an aliased
output of the graph, the whole tensor written in place. The aliasing pass also hands the
kernel's result to every later operation on the same graph input, so the writes to one
input run in order, each on the one before, whether or not a caller uses the tensor
:func:`write_page_rows` returns. Another view of the same buffer is another graph input,
outside that chain.

Every block id must name a page of ``pages``. Under LNC2 an out-of-bounds dynamic-offset
access drops the whole tile (``vllm_neuron.functional.dsa.ragged_pack``). The ids are
device data, so the checks here cannot read them; vLLM's block ids, the null block 0
included, are pages of the pool. Ids that name one page more than once leave one of the
rows.

Where NKI cannot run on CPU tensors, the torch path writes the same bytes through byte
views of the pages. It runs eager, on the host: the backend refuses an in-graph dtype
view when it lowers a graph to HLO, so a device tensor with NKI kernels off is refused
here by name. The kernel route has no torch dtype view.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.utils.neuron_utils import can_run_kernel

#: Carriers one launch writes: the kernel's row parameters (``first_hbm`` .. ``third_hbm``).
CARRIERS_PER_LAUNCH = 3

#: Bytes per partition one DMA tile carries: a contiguous run of at least 2 KiB saturates a
#: DMA queue, and 32 KiB keeps the staging tile a small share of an SBUF partition.
FREE_TILE_BYTES = 32 * 1024


class CheckpointWriteError(ValueError):
    """A page write this module refuses: pages that are not whole pages, block ids that are
    not an int32 or int64 vector, rows of another count or out of their page, too many
    carriers, or a device tensor with NKI kernels off."""


@nki.jit
def _write_page_rows_nki(pages_hbm, ids_hbm, num_blocks, page_elems, first_hbm, first_offset,
                         second_hbm=None, second_offset=0, third_hbm=None, third_offset=0):
    """Write row ``i`` of each carrier into page ``ids[i]`` of ``pages``, at its offset.

    Args:
        pages_hbm: the pages, contiguous, ``num_blocks * page_elems`` elements of any dtype.
        ids_hbm: ``[n, 1]`` int32 block ids.
        num_blocks, page_elems: pages and elements per page (trace-time ints).
        first_hbm, second_hbm, third_hbm: ``[n, F_j]`` rows of any dtype, contiguous; the
            last two may be None.
        first_offset, second_offset, third_offset: each carrier's byte offset in its page.

    Returns:
        ``pages_hbm``, written in place.
    """
    # The kernel frontend takes plain ``for`` loops over simple names (no comprehension,
    # no tuple target), so the carriers and their offsets are listed by index.
    candidates = (first_hbm, second_hbm, third_hbm)
    candidate_offsets = (first_offset, second_offset, third_offset)
    carriers = []
    offsets = []
    for k in range(len(candidates)):
        if candidates[k] is not None:
            carriers.append(candidates[k].view(nl.int8))
            offsets.append(candidate_offsets[k])
    pages8 = pages_hbm.reshape((num_blocks, page_elems)).view(nl.int8)
    page_bytes = pages8.shape[1]
    count = ids_hbm.shape[0]
    pmax = nl.tile_size.pmax
    for t in range((count + pmax - 1) // pmax):
        held = min(pmax, count - t * pmax)
        index = nl.ndarray((held, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=index, src=ids_hbm.ap(pattern=[[1, held], [1, 1]], offset=t * pmax))
        for j in range(len(carriers)):
            rows8 = carriers[j]
            width = rows8.shape[1]
            for f in range((width + FREE_TILE_BYTES - 1) // FREE_TILE_BYTES):
                f0 = f * FREE_TILE_BYTES
                span = min(FREE_TILE_BYTES, width - f0)
                tile = nl.ndarray((held, span), dtype=nl.int8, buffer=nl.sbuf)
                nisa.dma_copy(dst=tile, src=rows8.ap(pattern=[[width, held], [1, span]],
                                                     offset=t * pmax * width + f0))
                nisa.dma_copy(
                    dst=pages8.ap(pattern=[[page_bytes, held], [1, span]],
                                  offset=offsets[j] + f0, vector_offset=index,
                                  indirect_dim=0),
                    src=tile,
                )
    return pages_hbm


def _check(pages: Tensor, block_ids: Tensor, rows: Sequence[Tensor], offsets: Sequence[int],
           page_bytes: int) -> int:
    """The number of pages of ``pages``; raises :class:`CheckpointWriteError` otherwise."""
    page_bytes = int(page_bytes)
    nbytes = int(pages.numel()) * pages.element_size()
    if page_bytes <= 0 or nbytes == 0 or nbytes % page_bytes or not pages.is_contiguous():
        raise CheckpointWriteError(
            f"the pages are a contiguous tensor of whole {page_bytes}-byte pages; got "
            f"{tuple(pages.shape)} {pages.dtype} ({nbytes} bytes"
            f"{'' if pages.is_contiguous() else ', not contiguous'})"
        )
    if page_bytes % pages.element_size():
        raise CheckpointWriteError(
            f"a {page_bytes}-byte page is not a whole number of {pages.dtype} elements"
        )
    if not torch.is_tensor(block_ids) or block_ids.dim() != 1 or block_ids.dtype not in (
            torch.int32, torch.int64):
        raise CheckpointWriteError(
            "block ids are a [n] int32 or int64 tensor, one id per page row; got "
            f"{tuple(block_ids.shape) if torch.is_tensor(block_ids) else block_ids!r}"
            f"{' ' + str(block_ids.dtype) if torch.is_tensor(block_ids) else ''}"
        )
    if len(rows) != len(offsets) or not rows:
        raise CheckpointWriteError(
            f"one byte offset per carrier: {len(rows)} carrier(s), {len(offsets)} offset(s)"
        )
    count = int(block_ids.shape[0])
    for carrier, (carrier_rows, offset) in enumerate(zip(rows, offsets)):
        width = int(carrier_rows[0].numel()) * carrier_rows.element_size() if count else 0
        if carrier_rows.dim() < 1 or int(carrier_rows.shape[0]) != count:
            raise CheckpointWriteError(
                f"carrier {carrier} has {tuple(carrier_rows.shape)} rows for {count} block id(s)"
            )
        if int(offset) < 0 or int(offset) + width > page_bytes:
            raise CheckpointWriteError(
                f"carrier {carrier}: {width} byte(s) from byte {int(offset)} do not fit a "
                f"{page_bytes}-byte page"
            )
    return nbytes // page_bytes


def write_page_rows(pages: Tensor, block_ids: Tensor, rows: Sequence[Tensor],
                    offsets: Sequence[int], page_bytes: int) -> Tensor:
    """Write row ``i`` of each ``rows[c]`` into page ``block_ids[i]`` of ``pages``.

    Args:
        pages: a contiguous tensor of whole ``page_bytes``-byte pages, of any dtype and
            shape; written in place.
        block_ids: ``[n]`` int32 or int64 page indices. int32 ids built on the host take
            no conversion in the graph.
        rows: one ``[n, ...]`` tensor per carrier, any dtype; row ``i``'s bytes go to page
            ``block_ids[i]``.
        offsets: each carrier's byte offset in its page.
        page_bytes: bytes per page.

    Returns:
        The written pages. In a traced graph this is the kernel's result, which the
        backend hands to every later operation on the same graph input. Eager on CPU
        tensors it is
        ``pages`` itself, on the torch path and on the NKI simulator (which writes the
        operand and returns a copy of it).

    Raises:
        CheckpointWriteError: as the class says.
    """
    rows = tuple(rows)
    offsets = tuple(int(offset) for offset in offsets)
    num_blocks = _check(pages, block_ids, rows, offsets, page_bytes)
    count = int(block_ids.shape[0])
    if count == 0:
        return pages
    page_bytes = int(page_bytes)
    if can_run_kernel(pages):
        if len(rows) > CARRIERS_PER_LAUNCH:
            raise CheckpointWriteError(
                f"one launch writes up to {CARRIERS_PER_LAUNCH} carriers; got {len(rows)}"
            )
        flat = []
        for carrier_rows, offset in zip(rows, offsets):
            flat.extend((carrier_rows.contiguous().reshape(count, -1), offset))
        ids = block_ids.reshape(count, 1).to(torch.int32)
        written = wrap_nki(_write_page_rows_nki)(pages, ids, num_blocks,
                                                 page_bytes // pages.element_size(), *flat)
        # Eager on CPU tensors the NKI simulator writes the operand in place and returns a
        # copy of it, so the operand is what a caller keeps writing. Traced, the kernel's
        # result carries the write.
        if pages.device.type == "cpu" and not torch.compiler.is_compiling():
            return pages
        return written
    if pages.device.type != "cpu":
        raise CheckpointWriteError(
            f"a page write on {pages.device} needs its NKI kernel: the torch path cuts the "
            "pages with dtype views, which the device backend does not lower, and NKI "
            "kernels are off (VLLM_NEURON_DISABLE_NKI_KERNELS)"
        )
    page_view = pages.reshape(-1).view(torch.int8).view(num_blocks, page_bytes)
    index = block_ids.to(torch.int64)
    for carrier_rows, offset in zip(rows, offsets):
        data = carrier_rows.contiguous().reshape(count, -1).view(torch.int8)
        page_view[index, offset:offset + int(data.shape[1])] = data
    return pages
