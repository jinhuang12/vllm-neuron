# SPDX-License-Identifier: Apache-2.0
"""The traced venue takes the staged overlay as the kernel writes it, and refuses it written whole.

Graph extraction traces the model with ``torch.compile``, and dynamo propagates FAKE tensors through
the kernel wrapper before one tile is placed. That propagation bounds a destination pattern by the
element count of the tile it writes into, and a pattern covering EVERY row of the staged window from a
runtime destination row reaches one element past that count: a window of 128 rows by 128 latents is
refused with ``index 16384 is out of bounds for axis 0 with size 16384``. NEITHER OTHER CPU VENUE SEES
IT -- the simulator runs the kernel body as plain Python, and a direct fake-tensor call of the same
entry accepts the same pattern -- so this file is the only reading of that bound in the tree, and the
bound is the whole reason ``_overlay_chunk`` caps a chunk one row short of the window.

TWO tests, one per conjunct, and NO ``parametrize``:

  1. a step whose rows FILL its window traces as the kernel ships, and so does a step half its
     window's width, which is the control that this venue takes a paged call at all;
  2. the same call with the cap defeated is REFUSED, which is what makes test 1 a reading rather than a
     habit: a venue that had lost the bound would take the kernel with the cap or without it.

BOTH ITEMS READ A TRACE AND NO VALUE, so the widths here are the smallest that keep the window a whole
number of staging pieces; the staged window's contents are read at the served latent rank by
``test_mla_sparse_paged.py``. The compile backend keeps each graph and runs none, which is the runner's
own capture contract, so no kernel body reaches the simulator here.

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

#: The traced widths. The latent rank and the selected-row count are the kernel's own tile width, the
#: page is the narrowest admissible block, and the window row counts below are whole pages of it.
LATENT = 128
HEADS = 4
QUERIES = 128
TOPK = 128
PAGE = 4
SCALE = 0.125

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


def _operands(window_rows: int, tokens: int) -> tuple:
    """One paged call's operands: a bank the size of its own window, its table, and this step's rows."""
    return (torch.zeros(window_rows, LATENT, dtype=torch.bfloat16),
            torch.zeros(QUERIES, HEADS, LATENT, dtype=torch.bfloat16),
            torch.zeros(QUERIES, TOPK, dtype=torch.int32),
            torch.zeros(window_rows // PAGE, 1, dtype=torch.int32),
            torch.zeros(tokens, LATENT, dtype=torch.bfloat16),
            torch.zeros(1, 1, dtype=torch.int32))


def _call(bank, queries, selected, table, written, at):
    """The paged seam and nothing beside it, so one graph holds the call this file reads."""
    return MS.mla_sparse_attention(queries, bank, selected, SCALE, block_table_row=table,
                                   written=written, write_offset=at, page_size=PAGE)


def _traced(window_rows: int, tokens: int) -> tuple[_Keep, str]:
    """Trace one paged call; return the kept graphs and the refusal's message, or an empty one."""
    keep = _Keep()
    dynamo.reset()
    compiled = torch.compile(_call, backend=keep, fullgraph=True, dynamic=False)
    try:
        compiled(*_operands(window_rows, tokens))
    except _Captured:
        return keep, ""
    except Exception as error:  # a refusal is this file's reading, not a failure of it
        return keep, " ".join(str(error).split())
    return keep, ""


def _say(label: str, value: object) -> None:
    """One reading per line, so the transcript carries the value and not only a pass."""
    print(f"TRACED_OVERLAY_{label}={value}", flush=True)


def test_a_step_whose_rows_fill_the_window_traces_as_the_kernel_ships() -> None:
    """A 128-row step into a 128-row window traces, and into a 256-row window it traces too.

    The first arm is the case the cap exists for: the rows fill the window, so the shipped rule writes
    them in 127 rows and 1 instead of 128, and the trace carries no refusal. The second arm is the
    control: the same step into twice the window is never capped, so a refusal there would belong to
    this venue or to the paged operands and not to the filled window.
    """
    for window_rows, tokens in ((128, 128), (256, 128)):
        keep, refusal = _traced(window_rows, tokens)
        _say(f"CHUNK_{window_rows}_{tokens}", MS._overlay_chunk(window_rows))
        _say(f"KERNEL_CALLS_{window_rows}_{tokens}", keep.kernel_calls())
        _say(f"REFUSAL_{window_rows}_{tokens}", refusal[:200] or "none")
        assert refusal == "", (
            f"a step of {tokens} rows into a {window_rows}-row window is refused by the traced venue: "
            f"{refusal[:400]}. The extracted graph is built in this venue, so a refusal here is a "
            f"serving path that never compiles"
        )
        assert keep.kernel_calls() >= 1, (
            f"the graphs kept for a {window_rows}-row window hold no kernel call, so this arm read no "
            f"kernel: a trace that routed around the entry would pass the reading above unchanged"
        )


def test_the_traced_venue_refuses_an_overlay_written_in_one_whole_pattern() -> None:
    """With the cap defeated, the filled window is refused, which is the bound the cap answers.

    The chunk rule is replaced by one that returns the whole window, the form the kernel emitted before
    the cap, and the same call of test 1's first arm is traced again. The refusal has to name the bound
    it reached and the call it reached it in; its sizes and its wording belong to the toolchain, so
    neither is read here. The shipped rule is put back whether the reading passes or fails.
    """
    shipped = MS._overlay_chunk
    try:
        MS._overlay_chunk = lambda window: window
        uncapped = MS._overlay_chunk(128)
        keep, refusal = _traced(128, 128)
    finally:
        MS._overlay_chunk = shipped
    _say("UNCAPPED_CHUNK", f"{uncapped}.{MS._overlay_chunk(128)}")
    _say("UNCAPPED_KERNEL_CALLS", keep.kernel_calls())
    _say("UNCAPPED_REFUSAL", refusal[:200] or "none")
    assert uncapped == 128 and MS._overlay_chunk(128) == 127, (
        f"the chunk rule was not defeated and restored as this item requires: it returned {uncapped} "
        f"rows under the replacement and {MS._overlay_chunk(128)} rows after it"
    )
    assert BOUND in refusal and WRAPPER in refusal, (
        f"an overlay written in one whole-window pattern is NOT refused by this venue any more, and "
        f"the refusal read was {refusal[:400] or 'none'}. The cap in _overlay_chunk exists only for "
        f"that refusal, so a venue that takes the whole pattern makes the cap removable -- and makes "
        f"the item above evidence of nothing"
    )
