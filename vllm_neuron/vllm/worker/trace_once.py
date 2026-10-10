# SPDX-License-Identifier: Apache-2.0
"""One Dynamo trace per graph of the target model, for graph extraction and warm-up.

Graph extraction calls the target model to write the HLO of each graph for the
parallel compile. The warm-up then calls it again with the same inputs to load the
graph's NEFF. With a ``torch.compile`` wrapper for each call, Dynamo traces each
graph two times: the guards of a traced frame hold the backend that compiled it, so
the second wrapper does not find the frame of the first.

Here one wrapper serves both calls. Its backend, ``TraceOnceBackend``, gives the
capture backend a copy of each graph that Dynamo traces during extraction, and keeps
the graph. The warm-up call passes the frame's guards, and its first run hands the
kept graph to the served backend, which looks the graph up in the compile cache and
loads its NEFF. The served backend thus gets the graph that it would have traced
itself, and the cache keys do not change. A graph whose guards fail is traced again
and compiled at once, as before.

``trace_once_refusal`` names the starts that keep a wrapper for each call.
"""

from __future__ import annotations

import contextlib
import copy
import logging
from typing import Any, Callable

import torch
from torch._dynamo.backends.registry import lookup_backend

from vllm_neuron import envs

logger = logging.getLogger(__name__)

CAPTURE_BACKEND = "neuron_libtorch_graph_capture"


def _rank_zero() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _info(message: str, *args: Any) -> None:
    if _rank_zero():
        logger.info(message, *args)


def trace_once_refusal(*, tensor_replacement: bool) -> str | None:
    """Return why extraction and the warm-up keep a wrapper each, or None."""
    if envs.is_native_backend():
        return (
            "the native backend compiles each graph from its warm-up and captures none"
        )
    if not envs.VLLM_NEURON_DISABLE_PARALLEL_TRACE:
        return (
            "graph extraction runs in the parallel trace pool, in processes of its own "
            "(VLLM_NEURON_DISABLE_PARALLEL_TRACE is not set)"
        )
    if tensor_replacement:
        return (
            "tensor replacement is configured: the warm-up traces with replacement "
            "tensors that graph extraction does not set"
        )
    return None


def compile_target_model(
    model: torch.nn.Module,
    *,
    capture: bool,
    fullgraph: bool,
    options: dict,
    tensor_replacement: bool,
) -> tuple[torch.nn.Module, Callable[..., Any] | None]:
    """Return the served wrapper of ``model``, and what graph extraction calls.

    The second item is None when ``capture`` is False. When ``trace_once_refusal``
    names the start, it is a ``torch.compile`` wrapper of its own, as the parallel
    trace pool needs.
    """
    served_backend = envs.get_compile_backend_name()
    if not capture:
        served = torch.compile(
            model, backend=served_backend, fullgraph=fullgraph, options=options
        )
        return served, None
    refusal = trace_once_refusal(tensor_replacement=tensor_replacement)
    if refusal is not None:
        _info(
            "Graph extraction and the warm-up trace each graph separately: %s", refusal
        )
        extraction = torch.compile(
            model, backend=CAPTURE_BACKEND, fullgraph=fullgraph, options=options
        )
        served = torch.compile(
            model, backend=served_backend, fullgraph=fullgraph, options=options
        )
        return served, extraction
    backend = TraceOnceBackend(served_backend, CAPTURE_BACKEND)
    served = torch.compile(model, backend=backend, fullgraph=fullgraph, options=options)
    _info("Graph extraction and the warm-up share one Dynamo trace per graph")
    return served, ExtractionEntry(served, backend)


class TraceOnceBackend:
    """The served wrapper's backend; in graph extraction it also captures each graph.

    The two backends are looked up when the wrapper is built, as ``torch.compile``
    does for a backend name.
    """

    def __init__(self, served_backend: str, capture_backend: str) -> None:
        self._served = lookup_backend(served_backend)
        self._capture = lookup_backend(capture_backend)
        self.extracting = False
        self.kept = 0
        self.used = 0

    @contextlib.contextmanager
    def extraction(self):
        """Mark the calls inside as graph extraction."""
        self.extracting = True
        try:
            yield
        finally:
            self.extracting = False

    def __call__(self, gm: torch.fx.GraphModule, example_inputs, **kwargs):
        if not self.extracting:
            if self.kept > self.used:
                _info(
                    "Dynamo traced a graph outside graph extraction while %d graph(s) "
                    "that extraction traced are not yet used",
                    self.kept - self.used,
                )
            return self._served(gm, example_inputs, **kwargs)
        # The capture backend's passes edit the graph that they get, and the served
        # backend must get the graph as Dynamo traced it.
        copied = torch.fx.GraphModule(gm, copy.deepcopy(gm.graph))
        bail = self._capture(copied, list(example_inputs), **kwargs)
        self.kept += 1
        return _KeptGraph(self, gm, list(example_inputs), kwargs, bail)

    def _serve(self, kept: "_KeptGraph") -> Callable[..., Any]:
        compiled = self._served(kept.gm, kept.example_inputs, **kept.kwargs)
        self.used += 1
        _info(
            "Warm-up uses the trace of graph extraction (%d of %d graphs)",
            self.used,
            self.kept,
        )
        return compiled


class _KeptGraph:
    """The callable that Dynamo holds for a graph traced in extraction.

    In extraction it runs the capture backend's result, which raises
    ``CaptureComplete``. Its first run outside extraction compiles the graph with
    the served backend; then it runs the compiled graph.
    """

    def __init__(self, backend: TraceOnceBackend, gm, example_inputs, kwargs, bail):
        self._backend = backend
        self.gm = gm
        self.example_inputs = example_inputs
        self.kwargs = kwargs
        self._bail = bail
        self._compiled = None

    def __call__(self, *args):
        if self._backend.extracting:
            return self._bail(*args)
        if self._compiled is None:
            self._compiled = self._backend._serve(self)
            self.gm = self.example_inputs = None
        return self._compiled(*args)


class ExtractionEntry:
    """What graph extraction calls in place of a capture wrapper.

    It calls the served wrapper in extraction, which traces and captures the graph
    and raises ``CaptureComplete``, as a capture wrapper does.
    """

    def __init__(self, served: torch.nn.Module, backend: TraceOnceBackend) -> None:
        self._served = served
        self._backend = backend

    def __call__(self, *args, **kwargs):
        with self._backend.extraction():
            self._served(*args, **kwargs)
        raise RuntimeError(
            "graph extraction ran a compiled graph instead of capturing it: the graph "
            "was traced and compiled outside graph extraction before it"
        )
