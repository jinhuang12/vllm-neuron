# SPDX-License-Identifier: Apache-2.0
"""Tracing accepts a sparse-attention step that fills its window, and refuses it unpadded.

Graph extraction traces the model with ``torch.compile``, and dynamo propagates fake
tensors through the kernel wrapper before one tile is placed. That propagation reads the
overlay's destination END as an index into the staged tile, so a step whose rows reach the
window's last row lands one element past the tile and is refused -- a 128-row step into a
128-row window of 128 latents with ``index 16384 is out of bounds for axis 0 with size
16384``. Chunking the transfers does not move that end, which is why ``_staged_window``
carries ``STAGE_PAD`` rows beyond the window instead. No other CPU path sees the bound: the
simulator runs the body as plain Python and a direct fake-tensor call accepts the same
pattern, so this file is its only reading in the tree.

The pad is load-bearing at the served block size and not only in the fixtures: at a block of
128 a 2,048-token request's second chunk writes rows 1,024 to 2,047 of a 2,048-row window,
which does reach that window's last row.
"""

from __future__ import annotations

import pytest
import torch
import torch._dynamo as dynamo

from vllm_neuron.functional.attention import mla_sparse as MS

pytestmark = [pytest.mark.fast, pytest.mark.forked]

# The traced widths. The latent rank and the selected-row count are the kernel's own tile
# width; the geometries below carry their own page size.
LATENT = 128
HEADS = 4
QUERIES = 128
TOPK = 128
SCALE = 0.125

# ``(window rows, page size, step rows)``. The first three are steps that FILL their window:
# the served block, then the two block sizes the model fixtures stage. The last is a step a
# quarter of its window, which ends well inside the tile.
FILLED = ((4096, 4096, 4096), (128, 128, 128), (256, 4, 256), (4096, 4096, 1024))

# The geometry the pad is removed under: one page, filled, the smallest window that refuses.
REFUSES_AT = (128, 128, 128)

# The two texts the refusal carries: the bound it reached, and the call it reached it in.
# Read as substrings, because the sizes in the message are the traced widths and its wording
# is the toolchain's, while the bound itself is what this file pins.
BOUND = "out of bounds"
WRAPPER = "nki_kernel_wrapper"


class _Captured(Exception):
    """Raised in place of running a kept graph, exactly as the capture backend raises."""


class _Keep:
    """A compile backend that keeps every graph and runs none, so no kernel body executes."""

    def __init__(self) -> None:
        self.graphs: list = []

    def __call__(self, graph_module, _example_inputs):
        self.graphs.append(graph_module)
        return self._stop

    @staticmethod
    def _stop(*_args, **_kwargs):
        raise _Captured()

    def kernel_calls(self) -> int:
        """How many kernel calls the kept graphs carry."""
        return sum(1 for held in self.graphs for node in held.graph.nodes
                   if node.op == "call_function" and WRAPPER in str(node.target))


def _operands(window_rows: int, page: int, tokens: int) -> tuple:
    """One paged call's operands: the bank, its table, and the step.

    The bank is two pages wider than the window, as every bank in this tree is: a bank the
    size of its window puts the last page read's own end on the bank's last element, which
    is a second reading of the same bound and would hide this one.
    """
    return (torch.zeros(window_rows + 2 * page, LATENT, dtype=torch.bfloat16),
            torch.zeros(QUERIES, HEADS, LATENT, dtype=torch.bfloat16),
            torch.zeros(QUERIES, TOPK, dtype=torch.int32),
            torch.zeros(window_rows // page, 1, dtype=torch.int32),
            torch.zeros(tokens, LATENT, dtype=torch.bfloat16),
            torch.zeros(1, 1, dtype=torch.int32))


def _call(bank, queries, selected, table, written, at, page):
    """The paged seam and nothing beside it, so one graph holds the call this file reads."""
    return MS.mla_sparse_attention(queries, bank, selected, SCALE, block_table_row=table,
                                   written=written, write_offset=at, page_size=page)


def _traced(window_rows: int, page: int, tokens: int) -> tuple[_Keep, str]:
    """Trace one paged call; return the kept graphs and the refusal's message, or an empty one."""
    keep = _Keep()
    dynamo.reset()
    compiled = torch.compile(lambda *held: _call(*held, page), backend=keep,
                             fullgraph=True, dynamic=False)
    try:
        compiled(*_operands(window_rows, page, tokens))
    except _Captured:
        return keep, ""
    except Exception as error:  # a refusal is this file's reading, not a failure of it
        return keep, " ".join(str(error).split())
    return keep, ""


def test_a_step_that_fills_its_window_traces_as_the_kernel_ships() -> None:
    """Four geometries trace: three filled windows, and one step a quarter of its window.

    The filled arms are the ones the staged pad exists for -- their rows reach the window's
    last row. The served arm fills 4,096 rows in 32 transfers, the 256-row arm in two, the
    128-row arm in one, so a pad that only helped a single-transfer overlay fails here. A
    refusal on the quarter arm would belong to the paged operands rather than to a filled
    window.
    """
    for window_rows, page, tokens in FILLED:
        keep, refusal = _traced(window_rows, page, tokens)
        arm = f"{window_rows}_{page}_{tokens}"
        assert refusal == "", (
            f"a step of {tokens} rows into a {window_rows}-row window of {page}-row pages is "
            f"refused while tracing: {refusal[:400]}. The extracted graph is built this way, "
            f"so a refusal here is a serving path that never compiles"
        )
        assert keep.kernel_calls() >= 1, (
            f"the graphs kept for the {arm} arm hold no kernel call, so it read no kernel: a "
            f"trace that routed around the entry would pass the assertion above unchanged"
        )


def test_tracing_refuses_a_filled_window_once_the_staged_pad_is_removed() -> None:
    """With ``STAGE_PAD`` at 0 the filled window is refused, naming the bound and the call.

    The sizes and the wording of the refusal belong to the toolchain, so neither is read
    here. The shipped pad is put back whether the assertions pass or fail.
    """
    window_rows, page, tokens = REFUSES_AT
    shipped = MS.STAGE_PAD
    try:
        MS.STAGE_PAD = 0
        _keep, refusal = _traced(window_rows, page, tokens)
    finally:
        MS.STAGE_PAD = shipped
    assert MS.STAGE_PAD == shipped and shipped > 0, (
        f"the pad was not removed and restored: it reads {MS.STAGE_PAD} rows afterwards and "
        f"the kernel ships {shipped}"
    )
    assert BOUND in refusal and WRAPPER in refusal, (
        f"a step filling its window is no longer refused with the pad removed, and the "
        f"refusal read was {refusal[:400] or 'none'}. The pad in _staged_window exists only "
        f"for that refusal, so a toolchain that takes the unpadded tile makes the pad "
        f"removable"
    )
