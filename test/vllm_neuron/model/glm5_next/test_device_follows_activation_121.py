# SPDX-License-Identifier: Apache-2.0
"""An allocation on the KDA attention's traced path lives where its activation lives.

    VLLM_NEURON_CPU_MODE=1 python -m pytest -s -rA \\
      test/vllm_neuron/model/glm5_next/test_device_follows_activation_121.py

Graph extraction holds the module and every input on ``meta``
(``neuron_worker.py:500-501``), so a factory call naming no ``device=`` takes the default
device instead and the first arithmetic against a parameter meets two devices. A01 is an
AST census of that mistake; A02 drives the forward where the capture drives it. With
``GLM53F_121_EXPECT_BASE=1`` both items assert the PRE-REPAIR reading instead -- A01 the
two device-less allocations at ``model_fp8.py:4182`` and ``:4192``, A02 the two-device
failure 64 workers reported out of graph extraction -- which is this file's control arm.
"""

from __future__ import annotations

import ast
import os
import pathlib
import traceback

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from test.vllm_neuron.model.glm5_next import test_kda_layer as landed

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: Each of these allocates rather than deriving, so each decides a device.
FACTORIES = frozenset({"empty", "zeros", "ones", "full", "arange", "tensor"})

MODEL_FILE_VAR = "GLM53F_121_MODEL_FILE"
EXPECT_BASE_VAR = "GLM53F_121_EXPECT_BASE"

#: The two lines the pre-repair tree carries the device-less allocations on.
BASE_OFFENDER_LINES = (4182, 4192)

#: A prefill from position 0, then a decode step opening where it left off, so the branch
#: at ``model_fp8.py:4191`` is read on the side a served sequence takes as well.
LEGS = (
    ("prefill", landed.DECLARED_PREFILL_TOKENS, True, 0),
    ("decode", 1, False, landed.DECLARED_PREFILL_TOKENS),
)


def _expect_base() -> bool:
    return os.environ.get(EXPECT_BASE_VAR) == "1"


def _require_cpu_mode() -> None:
    """The flag comes from the process environment; the seams read it at import."""
    if os.environ.get("VLLM_NEURON_CPU_MODE") != "1":
        raise RuntimeError("VLLM_NEURON_CPU_MODE must be 1 in the process environment")


def _census(path: pathlib.Path) -> tuple[int, list[tuple[int, str, str]]]:
    """``(calls read, offenders)``; an offender is ``(line, factory, enclosing def)``.

    A call inside ``__init__`` is exempt because what it builds is registered on the
    module, so ``Module.to(device)`` carries it; a tensor built inside any other method
    belongs to no module and nothing moves it later.
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


def test_a01_every_allocation_outside_init_names_its_device() -> None:
    """The census, over the driven module's own file or the one asked for."""
    _require_cpu_mode()
    from vllm_neuron.model.glm5_next import model_fp8

    path = pathlib.Path(os.environ.get(MODEL_FILE_VAR) or model_fp8.__file__)
    calls, offenders = _census(path)
    named = ",".join(f"{line}:torch.{factory}" for line, factory, _ in offenders)
    print(
        f"A01|census|file={path}|factory_calls={calls}"
        f"|offenders={len(offenders)}|lines={named or 'none'}"
    )
    if _expect_base():
        assert tuple(line for line, _f, _w in offenders) == BASE_OFFENDER_LINES, (
            f"expected device-less allocations at {BASE_OFFENDER_LINES}; read "
            f"{named or 'none'} in {path}"
        )
        assert all(where == "forward" for _l, _f, where in offenders), offenders
    else:
        assert not offenders, (
            f"{len(offenders)} allocation(s) outside __init__ name no device= and take "
            f"the default device under capture: {named} in {path}"
        )


def _held(shape, device) -> torch.Tensor:
    return torch.empty(tuple(shape), dtype=torch.float32, device=device)


def _hold_the_kernel_seams(monkeypatch, entries: dict[str, int]) -> None:
    """Hold the five kernel dispatches; none of them is measured here.

    A held seam returns tensors at the shapes its own docstring declares, derived from the
    operands it was handed, on the device of its activation operand -- a fake tensor
    carries nothing to hand a kernel. Four would take a torch oracle if their route were
    stood down instead and the gate seam ships none, so holding reaches all five.
    """
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

    Fake tensor mode is the capture regime and eager is not: an eager ``meta`` write out of
    a host buffer refuses for having no data, reporting the copy and not the device rule.
    """
    from vllm_neuron.model.glm5_next import model_fp8

    module = model_fp8.Glm5NextKDAAttention(config, landed.REGISTERED_TP_WORLD_SIZE)
    weights = landed._make_weights(
        hidden, heads, landed.DECLARED_KDA_HEAD_SIZE,
        landed.DECLARED_KDA_CONV_KERNEL_SIZE,
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
            chunk_size=landed.DECLARED_CHUNK,
        )


def _site_of(error: BaseException) -> str:
    """The last frame inside the plugin as ``file:line``, or ``unknown``."""
    inside = [frame for frame in traceback.extract_tb(error.__traceback__)
              if "/vllm_neuron/" in frame.filename]
    if not inside:
        return "unknown"
    return f"{inside[-1].filename.split('/vllm_neuron/')[-1]}:{inside[-1].lineno}"


def test_a02_the_forward_runs_where_its_inputs_live(monkeypatch) -> None:
    """Both legs, on ``meta``, with every kernel dispatch held."""
    _require_cpu_mode()
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    config = Glm5NextTextConfig()
    hidden = int(config.hidden_size)
    heads = landed.DECLARED_KDA_NUM_HEADS // landed.REGISTERED_TP_WORLD_SIZE
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
        shape = "none" if returned is None else str(tuple(returned.shape))
        print(
            f"A02|{leg}|completed={'no' if error else 'yes'}"
            f"|held={','.join(f'{s}:{c}' for s, c in entries.items())}"
            f"|returned={shape}@{'none' if returned is None else returned.device.type}"
            f"|error={text}|site={'none' if error is None else _site_of(error)}"
        )
        # THE PREMISE, READ BEFORE THE OUTCOME: the drive reached the recurrence. A leg
        # that stopped earlier proves nothing about either allocation.
        for seam in ("conv", "gate") + (
            ("intra", "inter") if is_prefill else ("decode",)
        ):
            assert entries[seam] >= 1, (
                f"the {leg} drive never reached the {seam} seam, so it never reached the "
                f"allocations under test{stopped}"
            )
        if _expect_base():
            assert error is not None and "two different devices" in text, (
                f"expected the pre-repair {leg} drive to fail on two devices{stopped}"
            )
            assert "cpu" in text and "meta" in text, text
            continue
        assert error is None, f"the {leg} drive did not complete on meta{stopped}"
        assert returned.device.type == "meta", (
            f"the {leg} drive returned on {returned.device}, not its input's device"
        )
        assert tuple(returned.shape) == (tokens, hidden), (
            f"the {leg} drive returned {tuple(returned.shape)}, not {(tokens, hidden)}"
        )
