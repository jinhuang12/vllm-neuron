# SPDX-License-Identifier: Apache-2.0
"""The dispatch flatten of the blockwise MoE mapping as one row gather per program.

``indexed_flatten`` (nkilib) scatters each expert's ``T // f_len`` rows of ``f_len`` token
indices to the output, at the expert's row offset: one dependent scatter DMA per expert of
the program's share, each into a private partial with an out-of-bounds skip, then a
reload, a ``sendrecv`` exchange of the partials and a max merge. This kernel gives
the same output as a gather: each output row finds its source row from the offsets, so a
program reads its rows' sources in one indirect DMA and writes its own rows once.

For output row ``r`` and nondecreasing offsets ``o``::

    e*    = #{e >= 1 : o[e] <= r}                    # the last expert whose window starts at or before r
    local = r - o[0] - sum_{e >= 1, o[e] <= r} (o[e] - o[e - 1])     # = r - o[e*]
    valid = 0 <= local < T // f_len
    out[r] = input_rows[e* * (T // f_len) + local] if valid else padding_val

``indexed_flatten``'s row ``r`` holds the last expert (in expert order) whose window
``[o[e], o[e] + T // f_len)`` holds ``r``, or the padding value. For nondecreasing offsets that is
the largest ``e`` with ``o[e] <= r`` when ``r - o[e] < T // f_len``; no window holds ``r`` otherwise
(a smaller ``e`` has ``o[e]`` no larger, so ``r - o[e]`` is no smaller). Its max merge of the two
programs' partials gives the same row when a row that one program writes into the other
program's rows holds only padding: the ``find_nonzero_indices`` contract (each input row
holds its expert's token indices from position 0, then ``padding_val``; the caller's
offsets give expert ``e`` at least the rows its tokens fill). Both forms copy the same
input words, so under that contract the output is bit-equal.

The index steps run in fp32 on the vector engine, exact: every value is an integer below
2^24. Invalid rows gather row 0 (a finite row) and the vector engine writes
``row * valid + padding_val * (1 - valid)``, exact in fp32 for the int32 token indices. No
DMA here has an out-of-bounds skip, and no program reads another program's rows.

A program takes ``ceil(R / programs)`` of the ``R = output_len // f_len`` output rows, in
tiles of at most ``nl.tile_size.pmax`` rows. The tiles are unrolled (``nl.static_range``);
the kernel has no device loop.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl

from nkilib.core.utils.kernel_assert import kernel_assert
from nkilib.core.utils.kernel_helpers import div_ceil


@nki.jit
def gather_flatten(
    input_tensor: nl.NkiTensor,
    f_len: int,
    output_len: int,
    row_offsets: nl.NkiTensor,
    padding_val: int = -1,
) -> nl.NkiTensor:
    """``indexed_flatten``'s output for nondecreasing ``row_offsets`` from 0, as a row gather.

    Args:
        input_tensor: ``[E, T]`` int32 on HBM, rows under the ``find_nonzero_indices``
            contract (module docstring).
        f_len: elements per output row; ``T % f_len == 0``.
        output_len: output elements; ``output_len % f_len == 0``.
        row_offsets: ``[E]`` int32 on HBM, nondecreasing, the first at least 0: each
            expert's first output row (``indexed_flatten`` with ``row_offsets_start`` 0).
        padding_val: the value of rows no expert's window holds.

    Returns:
        ``[output_len]`` int32 on shared HBM.
    """
    num_experts, tokens = input_tensor.shape
    kernel_assert(row_offsets.shape[0] == num_experts,
                  f"row_offsets must hold one offset per expert: {row_offsets.shape[0]} != {num_experts}")
    kernel_assert(tokens % f_len == 0, f"T must be divisible by f_len, got {tokens=}, {f_len=}")
    kernel_assert(output_len % f_len == 0, f"output_len must be divisible by f_len, got {output_len=}, {f_len=}")
    kernel_assert(num_experts >= 2, f"the index steps need two experts or more, got {num_experts}")
    rows_per_expert = tokens // f_len
    out_rows = output_len // f_len
    pmax = nl.tile_size.pmax
    prg_rows = div_ceil(out_rows, nl.num_programs(0))
    row_start = nl.program_id(0) * prg_rows
    row_end = min(out_rows, row_start + prg_rows)

    output = nl.ndarray((output_len,), dtype=input_tensor.dtype, buffer=nl.shared_hbm)
    output_rows = output.reshape((out_rows, f_len))
    input_rows = input_tensor.reshape((num_experts * rows_per_expert, f_len))

    for tile in nl.static_range(div_ceil(max(row_end - row_start, 0), pmax)):
        t0 = row_start + tile * pmax
        n = min(pmax, row_end - t0)
        _gather_tile(output_rows, input_rows, row_offsets, t0, n, num_experts, rows_per_expert,
                     float(padding_val))
    return output


def _gather_tile(output_rows, input_rows, row_offsets, t0: int, n: int, num_experts: int,
                 rows_per_expert: int, padding: float):
    """Output rows ``[t0, t0 + n)``: the index steps on ``[n, E]``, one gather, one store."""
    f_len = output_rows.shape[1]
    tail = num_experts - 1
    f32 = nl.float32

    # Every row's partition gets all offsets (partition step 0), in fp32.
    offsets_i = nl.ndarray((n, num_experts), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=offsets_i, src=row_offsets.ap(pattern=[[0, n], [1, num_experts]], offset=0))
    offsets = nl.ndarray((n, num_experts), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=offsets, src=offsets_i)
    row = nl.ndarray((n, 1), dtype=f32, buffer=nl.sbuf)
    nisa.iota(dst=row, pattern=[[0, 1]], offset=t0, channel_multiplier=1)

    # started[e - 1] = o[e] <= r; step[e - 1] = o[e] - o[e - 1]; their sums give e* and o[e*] - o[0].
    started = nl.ndarray((n, tail), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=started, data=offsets[0:n, 1:num_experts], op0=nl.less_equal, operand0=row)
    step = nl.ndarray((n, tail), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=step, data1=offsets[0:n, 1:num_experts], data2=offsets[0:n, 0:tail],
                       op=nl.subtract)
    nisa.tensor_tensor(dst=step, data1=step, data2=started, op=nl.multiply)
    expert = nl.ndarray((n, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=expert, data=started, op=nl.add, axis=1)
    rise = nl.ndarray((n, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=rise, data=step, op=nl.add, axis=1)

    # local = r - o[0] - rise; source = e* x rows_per_expert + local; valid = 0 <= local < rows_per_expert.
    local = nl.ndarray((n, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=local, data=row, op0=nl.subtract, operand0=offsets[0:n, 0:1],
                       op1=nl.subtract, operand1=rise)
    source = nl.ndarray((n, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=source, data=expert, op0=nl.multiply, operand0=float(rows_per_expert),
                       op1=nl.add, operand1=local)
    below = nl.ndarray((n, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=below, data=local, op0=nl.less, operand0=float(rows_per_expert))
    valid = nl.ndarray((n, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=valid, data=local, op0=nl.greater_equal, operand0=0.0,
                       op1=nl.multiply, operand1=below)
    # An invalid row reads row 0, a finite row the select below replaces.
    source_i = nl.ndarray((n, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=source_i, data1=source, data2=valid, op=nl.multiply)
    fill = nl.ndarray((n, 1), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=fill, data=valid, op0=nl.multiply, operand0=-padding, op1=nl.add,
                       operand1=padding)

    gathered = nl.ndarray((n, f_len), dtype=input_rows.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gathered, src=input_rows.ap(pattern=[[f_len, n], [1, f_len]], offset=0,
                                                  vector_offset=source_i, indirect_dim=0))
    result = nl.ndarray((n, f_len), dtype=input_rows.dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=result, data=gathered, op0=nl.multiply, operand0=valid, op1=nl.add,
                       operand1=fill)
    nisa.dma_copy(dst=output_rows[t0:t0 + n, 0:f_len], src=result)
