# SPDX-License-Identifier: Apache-2.0
"""Per-request state kept in whole banks: gather rows by slot, write them back in place.

A decode step of ``B`` requests reaches a layer with the layer's whole state bank
(``[slots, ...]``) and a ``[B]`` slot tensor rather than ``B`` per-request views of the
bank. This module is the model's side of that form:

* :func:`gather_bank_rows` reads the ``B`` rows a kernel consumes (one device gather);
* :func:`scatter_bank_rows` writes the advanced rows back **onto the bank itself**
  (``index_copy_`` on the bank tensor). The compile backend's aliasing pass maps an
  in-place op whose first operand is a graph input to a whole-tensor aliased output,
  so the bank the runner holds is updated on device. A write through a view of the
  bank (``bank[slot].copy_(...)``) is what the backend drops, and must not be used
  here;
* :func:`bank_rows_problem` and :func:`shared_store_problem` are the two refusals a
  consumer states by name: a bank whose rows are not the shape the kernel expects,
  and a slot tensor that names one store twice. The one exception is the scratch
  store -- the last slot of the sparse family's side caches, past the engine's
  concurrency bound -- which every padding row of a step names, so its ring write
  lands where no request reads.

Nothing here reads a value off a tensor, so every function is safe inside a traced
region; the refusals read shapes and dtypes only, except :func:`shared_store_problem`,
whose caller guards it with ``values_are_readable``.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "bank_rows_problem",
    "gather_bank_rows",
    "scatter_bank_rows",
    "shared_store_problem",
]


def gather_bank_rows(bank: Tensor, slots: Tensor) -> Tensor:
    """Return ``bank[slots]`` as ``[B, ...]``: the rows a kernel consumes, in step order."""
    return bank.index_select(0, slots)


def scatter_bank_rows(bank: Tensor, slots: Tensor, rows: Tensor) -> None:
    """Write ``rows`` (``[B, ...]``) onto ``bank`` at ``slots``, in place on the bank itself.

    The cast makes the write exact for the bank's dtype (the kernels return the conv
    carrier in the bank's dtype already and the recurrent one in float32, so for those
    it is the identity). ``index_copy_`` on the bank -- never on a view of it -- is what
    the backend keeps as an aliased whole-bank output.
    """
    bank.index_copy_(0, slots, rows.to(bank.dtype))


def bank_rows_problem(bank: Tensor, slots: Tensor, row_shape: tuple[int, ...], *, name: str) -> str | None:
    """Return why ``bank`` and ``slots`` cannot serve rows of ``row_shape``, or ``None``."""
    want = tuple(int(value) for value in row_shape)
    if bank.dim() != len(want) + 1 or tuple(bank.shape[1:]) != want:
        return (
            f"{name} must be a bank [slots, *{want}] in the bank form; got "
            f"{tuple(bank.shape)}"
        )
    if not torch.is_tensor(slots) or slots.dim() != 1 or int(slots.numel()) == 0:
        return (
            f"{name} takes one slot per request as a [B] tensor with B >= 1; got "
            f"{slots if not torch.is_tensor(slots) else tuple(slots.shape)}"
        )
    if slots.dtype not in (torch.int32, torch.int64):
        return f"{name}'s slots must be int32 or int64; got {slots.dtype}"
    return None


def shared_store_problem(slots: Tensor, batch: int, *, scratch: int) -> str | None:
    """Return why ``slots`` names one store twice, or ``None``.

    Rows naming ``scratch`` (the padding rows) may repeat; every other slot is a
    request's own store and appears once. Reads the values: eager only.
    """
    if tuple(slots.shape) != (int(batch),):
        return f"slots must be a [{int(batch)}] tensor, one entry per request; got {tuple(slots.shape)}"
    owned = slots[slots != int(scratch)]
    if int(torch.unique(owned).numel()) != int(owned.numel()):
        return (
            f"slots must be distinct, one store per request (only the scratch store "
            f"{int(scratch)} may be named by several padding rows); got {slots.tolist()}"
        )
    return None
