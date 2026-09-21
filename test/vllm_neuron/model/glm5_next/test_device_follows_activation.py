# SPDX-License-Identifier: Apache-2.0
"""An allocation on the KDA attention's traced path lives where its activation lives."""

from __future__ import annotations

import ast
import os
import pathlib
import traceback

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from test.vllm_neuron.model.glm5_next import test_kda_layer as kda_half

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: Each of these allocates rather than deriving, so each decides a device.
FACTORIES = frozenset({"empty", "zeros", "ones", "full", "arange", "tensor"})

#: A prefill from position 0, then a decode step opening where it left off, so the branch
#: at ``model_fp8.py`` is read on the side a served sequence takes as well.
LEGS = (
    ("prefill", kda_half.DECLARED_PREFILL_TOKENS, True, 0),
    ("decode", 1, False, kda_half.DECLARED_PREFILL_TOKENS),
)


def _require_cpu_mode() -> None:
    """The flag comes from the process environment; the seams read it at import."""
    if os.environ.get("VLLM_NEURON_CPU_MODE") != "1":
        raise RuntimeError("VLLM_NEURON_CPU_MODE must be 1 in the process environment")


def _census(path: pathlib.Path) -> tuple[int, list[tuple[int, str, str]]]:
    """``(calls read, offenders)``; an offender is ``(line, factory, enclosing def)``.
    """
    calls, offenders = [], []

    def walk(node: ast.AST, where: str) -> None:
        for child in ast.iter_child_nodes(node):
            func = getattr(child, "func", None)
            if (
                isinstance(func, ast.Attribute)
                and func.attr in FACTORIES
                and getattr(func.value, "id", None) == "torch"
            ):
                calls.append(child.lineno)
                keywords = {keyword.arg for keyword in child.keywords}
                if where != "__init__" and "device" not in keywords:
                    offenders.append((child.lineno, func.attr, where))
            inner = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            walk(child, child.name if inner else where)

    walk(ast.parse(path.read_text()), "<module>")
    return len(calls), offenders


def test_every_allocation_outside_init_names_its_device() -> None:
    """No allocation outside ``__init__`` takes the default device."""
    _require_cpu_mode()
    from vllm_neuron.model.glm5_next import model_fp8

    path = pathlib.Path(model_fp8.__file__)
    _calls, offenders = _census(path)
    named = ",".join(f"{line}:torch.{factory}" for line, factory, _ in offenders)
    assert not offenders, (
        f"{len(offenders)} allocation(s) outside __init__ name no device= and take "
        f"the default device under capture: {named} in {path}"
    )


def _held(shape, device) -> torch.Tensor:
    return torch.empty(tuple(shape), dtype=torch.float32, device=device)


def _hold_the_kernel_seams(monkeypatch, entries: dict[str, int]) -> None:
    """Hold the five kernel dispatches; none of them is measured here. """
    from vllm_neuron.functional.kda import chunked_recurrence as chunked
    from vllm_neuron.functional.kda import decode_state, depthwise_conv1d, gate_clamp

    def conv(img, filt, *rest, **options):
        assert not rest and not options, "the hold serves the seam's plain call only"
        entries["conv"] += 1
        # ``[N, C, 1, Q]``, with ``Q`` from the seam's own formula.
        end = depthwise_conv1d.output_width(int(img.shape[-1]), int(filt.shape[-1]))
        return _held((img.shape[0], img.shape[1], 1, end), img.device)

    def gate(g, a_log, *, lower, bias=None):
        entries["gate"] += 1
        return _held(g.shape, g.device)  # ``[T, D]``

    def intra(q, k, v, beta, gk):
        entries["intra"] += 1
        chunks, chunk, kdim = (int(x) for x in q.shape)
        key, value = (chunks, chunk, kdim), (chunks, chunk, int(v.shape[2]))
        square = (chunks, chunk, chunk)
        return chunked.IntraChunkOutputs(
            w=_held(key, q.device), u=_held(value, q.device), kg=_held(key, q.device),
            a_inv=_held(square, q.device), aqk=_held(square, q.device),
        )

    def inter(kg, w, u, gk, q, aqk, *, state=None):
        entries["inter"] += 1
        chunks, chunk, kdim = (int(x) for x in q.shape)
        value = (chunks, chunk, int(u.shape[2]))
        return chunked.InterChunkOutputs(
            o=_held(value, q.device),
            final_state=_held((int(u.shape[2]), kdim), q.device),
            v_new=_held(value, q.device),
        )

    def decode(state, q, k, v, beta, gk):
        entries["decode"] += 1
        vdim, kdim = (int(x) for x in state.shape)
        return decode_state.DecodeStepOutputs(
            o=_held((1, vdim), q.device), state=_held((vdim, kdim), q.device)
        )

    monkeypatch.setattr(depthwise_conv1d, "depthwise_conv1d", conv)
    monkeypatch.setattr(gate_clamp, "kda_gate_clamp", gate)
    monkeypatch.setattr(chunked, "kda_intra_chunk", intra)
    monkeypatch.setattr(chunked, "kda_inter_chunk", inter)
    monkeypatch.setattr(decode_state, "kda_decode_step", decode)


def _drive(config, hidden: int, heads: int, tokens: int, is_prefill: bool, start: int):
    """One capture-shaped call: the module and every input on ``meta``, fake tensor mode.
    """
    from vllm_neuron.model.glm5_next import model_fp8

    module = model_fp8.Glm5NextKDAAttention(config, kda_half.TP_WORLD_SIZE)
    weights = kda_half._make_weights(
        hidden, heads, kda_half.KDA_HEAD_SIZE,
        kda_half.KDA_CONV_KERNEL_SIZE,
    )
    for name, tensor in weights.items():
        if name != "input_layernorm_weight":  # that leaf belongs to the layer
            setattr(module, name, nn.Parameter(tensor, requires_grad=False))
    module.to("meta")
    meta = {"device": "meta"}
    with FakeTensorMode(allow_non_fake_inputs=True):
        return module(
            torch.empty(tokens, hidden, dtype=torch.float32, **meta),
            conv_state=torch.zeros(
                module.kda_conv_state_shape, dtype=module.kda_conv_state_dtype, **meta
            ),
            recurrent_state=torch.zeros(
                module.kda_recurrent_state_shape,
                dtype=module.kda_recurrent_state_dtype,
                **meta,
            ),
            is_prefill=is_prefill,
            start_position=start,
            chunk_size=kda_half.DECLARED_CHUNK,
        )


def _site_of(error: BaseException) -> str:
    """The last frame inside the plugin as ``file:line``, or ``unknown``."""
    inside = [frame for frame in traceback.extract_tb(error.__traceback__)
              if "/vllm_neuron/" in frame.filename]
    if not inside:
        return "unknown"
    return f"{inside[-1].filename.split('/vllm_neuron/')[-1]}:{inside[-1].lineno}"


def test_the_forward_runs_where_its_inputs_live(monkeypatch) -> None:
    """Both legs, on ``meta``, with every kernel dispatch held."""
    _require_cpu_mode()
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    config = Glm5NextTextConfig()
    hidden = int(config.hidden_size)
    heads = kda_half.KDA_NUM_HEADS // kda_half.TP_WORLD_SIZE
    for leg, tokens, is_prefill, start in LEGS:
        entries = dict.fromkeys(("conv", "gate", "intra", "inter", "decode"), 0)
        _hold_the_kernel_seams(monkeypatch, entries)
        returned, error = None, None
        try:
            returned = _drive(config, hidden, heads, tokens, is_prefill, start)
        except Exception as caught:  # noqa: BLE001 -- classified below, not swallowed
            error = caught
        text = "none" if error is None else str(error).splitlines()[0]
        stopped = f"; it stopped at {_site_of(error)}: {text}" if error else ""
        # Read before the outcome: the drive reached the recurrence. A run that stopped
        # earlier proves nothing about either allocation.
        for seam in ("conv", "gate") + (
            ("intra", "inter") if is_prefill else ("decode",)
        ):
            assert entries[seam] >= 1, (
                f"the {leg} drive never reached the {seam} seam, so it never reached the "
                f"allocations under test{stopped}"
            )
        assert error is None, f"the {leg} drive did not complete on meta{stopped}"
        assert returned.device.type == "meta", (
            f"the {leg} drive returned on {returned.device}, not its input's device"
        )
        assert tuple(returned.shape) == (tokens, hidden), (
            f"the {leg} drive returned {tuple(returned.shape)}, not {(tokens, hidden)}"
        )
