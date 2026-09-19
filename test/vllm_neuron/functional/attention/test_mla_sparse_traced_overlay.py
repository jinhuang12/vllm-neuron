# SPDX-License-Identifier: Apache-2.0
"""The traced venue takes a step that FILLS its window, and refuses it once the staged pad is removed.

Graph extraction traces the model with ``torch.compile``, and dynamo propagates FAKE tensors through
the kernel wrapper before one tile is placed. That propagation reads the overlay's destination END as an
INDEX into the staged tile, so a step whose rows reach the window's last row lands one element past the
tile and is refused: a 128-row step into a 128-row window of 128 latents is refused with ``index 16384
is out of bounds for axis 0 with size 16384``, and a 4,096-row step into the served window with ``index
524288 ... with size 524288``. Chunking the transfers does not move that end, which is why the staged
tile carries ``STAGE_PAD`` rows beyond the window instead. NEITHER OTHER CPU VENUE SEES THE BOUND -- the
simulator runs the body as plain Python and a direct fake-tensor call accepts the same pattern -- so
this file is its only reading in the tree.

TWO tests, one per conjunct, and NO ``parametrize``:

  1. a step whose rows FILL its window traces as the kernel ships, at the served block size and at the
     two small block sizes the model fixtures use, and a step a quarter of its window is the control
     that this venue takes a paged call at all;
  2. the same call with the pad removed is REFUSED, which is what makes test 1 a reading rather than a
     habit: a venue that had lost the bound would take the kernel with the pad or without it.

BOTH ITEMS READ A TRACE AND NO VALUE, so the latent rank here is the kernel's own tile width and the
values of the staged window are read at the served rank by ``test_mla_sparse_paged.py``. Every bank is
two pages WIDER than its window, as every bank in this tree is: a bank the size of its window puts the
last page read's own end on the bank's last element, which is a second reading of the same bound and
would hide this one. The compile backend keeps each graph and runs none, which is the runner's own
capture contract, so no kernel body reaches the simulator here.

The declared command for this file::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest test/vllm_neuron/functional/attention/test_mla_sparse_traced_overlay.py -v -s \
        -p no:cacheprovider
"""

from __future__ import annotations

import pytest
import torch
import torch._dynamo as dynamo

from vllm_neuron.functional.attention import mla_sparse as MS

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The traced widths. The latent rank and the selected-row count are the kernel's own tile width; the
#: geometries below carry their own page size, and every window is a whole number of pages.
LATENT = 128
HEADS = 4
QUERIES = 128
TOPK = 128
SCALE = 0.125

#: ``(window rows, page size, step rows)``, every one a step that FILLS its window: the served block,
#: then the two block sizes the model fixtures stage, and last a step a QUARTER of its window, the
#: control that this venue takes a paged call whose step ends well inside the tile.
FILLED = ((4096, 4096, 4096), (128, 128, 128), (256, 4, 256), (4096, 4096, 1024))

#: The geometry test 2 removes the pad under: one page, filled, the smallest window that refuses.
REFUSES_AT = (128, 128, 128)

#: The two texts the refusal carries: the bound it reached, and the call it reached it in. Read as
#: substrings, because the sizes in the message are the traced widths and its wording is the
#: toolchain's, while the bound itself is what this file pins.
BOUND = "out of bounds"
WRAPPER = "nki_kernel_wrapper"


class _Captured(Exception):
    """Raised in place of running a kept graph, exactly as the runner's capture backend raises."""


class _Keep:
    """A compile backend that keeps every graph and runs none, so no kernel body is executed here."""

    def __init__(self) -> None:
        self.graphs: list = []

    def __call__(self, graph_module, _example_inputs):
        self.graphs.append(graph_module)
        return self._stop

    @staticmethod
    def _stop(*_args, **_kwargs):
        raise _Captured()

    def kernel_calls(self) -> int:
        """How many kernel calls the kept graphs carry: the call whose operands this file traces."""
        return sum(1 for held in self.graphs for node in held.graph.nodes
                   if node.op == "call_function" and WRAPPER in str(node.target))


def _operands(window_rows: int, page: int, tokens: int) -> tuple:
    """One paged call's operands: a bank two pages wider than the window, its table, and the step."""
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


def _say(label: str, value: object) -> None:
    """One reading per line, so the transcript carries the value and not only a pass."""
    print(f"TRACED_OVERLAY_{label}={value}", flush=True)


def test_a_step_whose_rows_fill_the_window_traces_as_the_kernel_ships() -> None:
    """Four geometries trace: the served block filled, both fixture blocks filled, and a quarter step.

    The filled arms are the ones the staged pad exists for -- their rows reach the window's last row,
    which is where this venue reads one element past the tile without the pad. The served arm fills
    4,096 rows in 32 transfers, the 256-row arm in two, the 128-row arm in one, so a pad that only
    helped a single-transfer overlay fails here. The quarter arm ends well inside the window and is the
    control: a refusal there would belong to this venue or to the paged operands, not to a filled one.
    """
    for window_rows, page, tokens in FILLED:
        keep, refusal = _traced(window_rows, page, tokens)
        arm = f"{window_rows}_{page}_{tokens}"
        _say(f"KERNEL_CALLS_{arm}", keep.kernel_calls())
        _say(f"REFUSAL_{arm}", refusal[:200] or "none")
        assert refusal == "", (
            f"a step of {tokens} rows into a {window_rows}-row window of {page}-row pages is refused "
            f"by the traced venue: {refusal[:400]}. The extracted graph is built in this venue, so a "
            f"refusal here is a serving path that never compiles"
        )
        assert keep.kernel_calls() >= 1, (
            f"the graphs kept for the {arm} arm hold no kernel call, so it read no kernel: a trace "
            f"that routed around the entry would pass the reading above unchanged"
        )


def test_the_traced_venue_refuses_the_same_call_with_the_staged_pad_removed() -> None:
    """With the pad removed, the filled window is refused -- the bound the pad answers.

    ``STAGE_PAD`` is set to 0, which is the tile the kernel staged before this bound was read, and test
    1's smallest filled arm is traced again. The refusal has to name the bound it reached and the call
    it reached it in; its sizes and its wording belong to the toolchain, so neither is read here. The
    shipped pad is put back whether the reading passes or fails.
    """
    window_rows, page, tokens = REFUSES_AT
    shipped = MS.STAGE_PAD
    try:
        MS.STAGE_PAD = 0
        keep, refusal = _traced(window_rows, page, tokens)
    finally:
        MS.STAGE_PAD = shipped
    _say("UNPADDED_PAD", f"0.{MS.STAGE_PAD}")
    _say("UNPADDED_KERNEL_CALLS", keep.kernel_calls())
    _say("UNPADDED_REFUSAL", refusal[:200] or "none")
    assert MS.STAGE_PAD == shipped and shipped > 0, (
        f"the pad was not removed and restored as this item requires: it reads {MS.STAGE_PAD} rows "
        f"after the reading and the kernel ships {shipped}"
    )
    assert BOUND in refusal and WRAPPER in refusal, (
        f"a step filling its window is NOT refused by this venue any more with the pad removed, and "
        f"the refusal read was {refusal[:400] or 'none'}. The pad in _staged_window exists only for "
        f"that refusal, so a venue that takes the unpadded tile makes the pad removable -- and makes "
        f"the item above evidence of nothing"
    )
