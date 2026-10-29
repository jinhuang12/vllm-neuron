# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's load-time layouts read each loaded weight whole and place each result whole.

The Neuron device's copy refuses a source of another dtype, other sizes or a non-contiguous
layout, so the prepare step may touch a loaded weight only to read it whole to the host and to
place a result. A torch function mode stands in for that rule on CPU: it marks the loaded
parameters by their storages, and on a marked tensor allows only the metadata reads, ``detach``
and a host read of a whole tensor, which returns an unmarked clone; a placement of a contiguous
host tensor returns a marked clone. The model has test_forward.py's widths, built on meta from
the checkpoint's config at one rank, with seeded random weights for every parameter but the
experts', which stay on meta. Set ``KIMI_K3_CHECKPOINT`` to the checkpoint directory (its
config only); run with ``VLLM_NEURON_CPU_MODE=1``.
"""

import ast
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.overrides import TorchFunctionMode
import test_forward as forward
from vllm_neuron.functional.attention.attn_res_decode import attn_res_score_weight
from vllm_neuron.functional.attention.mla_paged import prepare_mla_weights
from vllm_neuron.model.kimi_k3 import model as m

pytestmark = pytest.mark.skipif(
    not (forward.CHECKPOINT / "config.json").is_file(),
    reason=f"Kimi K3 checkpoint not found at {forward.CHECKPOINT}",
)

METADATA = {"shape", "dtype", "device", "is_meta", "grad_fn", "size", "stride", "dim", "numel",
            "is_contiguous", "storage_offset"}
KDA = ("input_weight", "conv_weight", "forget_weight", "output_weight")
MLA = ("input_weight", "gate_weight", "latent_weight", "output_weight", "w_qb_t", "w_uk", "w_uv_t")
MOE = ("down_router_weight", "latent_norm_weight", "latent_up_weight", "shared_down_weight")
MLP = ("gate_up_weight", "down_weight")
PROJECTIONS = ("q", "k", "v", "g", "f_a", "b")
HOST_FUNCTIONS = ("attn_res_score_weight", "prepare_mla_weights")
SCOPES = ("_stack", *(f"{c}.prepare_after_load" for c in (
    "KimiK3KDA", "KimiK3MLA", "KimiK3MLP", "KimiK3MoE", "KimiK3DecoderLayer", "KimiK3Model",
    "KimiK3ForConditionalGeneration")), "KimiK3ForConditionalGeneration.load_weights",
    "KimiK3ForConditionalGeneration.load_weights_lite")
CALLEES = {"derive_on_host", "nn.Parameter", "nn.ParameterList", "_stack", "_tp_rank_and_size",
           "self._restore_unloaded", "_tp_group", "weight_loaders_mxfp4.load_weights",
           "self.local_expert_ids", "dist.is_initialized", "self.named_parameters",
           "self.named_buffers", "self._unloaded.update", "dict", "len", "ValueError"}


def _tensors(x) -> list:
    """The tensors in a call's arguments, through lists, tuples and dicts."""
    if isinstance(x, torch.Tensor):
        return [x]
    items = x.values() if isinstance(x, dict) else x if isinstance(x, (list, tuple)) else ()
    return [t for item in items for t in _tensors(item)]


def _name(func) -> str:
    """A function's name, or the attribute a property getter reads."""
    return func.__self__.__name__ if type(func).__name__ == "method-wrapper" else func.__name__


def _device_only(args, kwargs) -> bool:
    """True for ``.to`` given a device and nothing else."""
    rest = [*args[1:], *kwargs.values()]
    return len(rest) == 1 and isinstance(rest[0], (str, torch.device)) and set(kwargs) <= {"device"}


def _whole(t: torch.Tensor) -> bool:
    """True for a contiguous tensor that starts its storage and spans all of it."""
    return (t.is_contiguous() and t.storage_offset() == 0
            and t.numel() * t.element_size() == t.untyped_storage().nbytes())


class DeviceRule(TorchFunctionMode):
    """Refuses every call on a marked tensor that the device does not allow; marks placements."""

    def __init__(self, marked=()):
        super().__init__()
        self.held = list(marked)  # keeps marked storages alive, so no address is reused
        self.marked = {t.untyped_storage().data_ptr() for t in self.held}
        self.refused = []

    def is_marked(self, t) -> bool:
        """True for a tensor on a marked storage."""
        return (isinstance(t, torch.Tensor) and not t.is_meta
                and t.untyped_storage().data_ptr() in self.marked)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = _name(func)
        marked = [t for t in _tensors((args, kwargs)) if self.is_marked(t)]
        if not marked:
            out = func(*args, **kwargs)
            if name == "to" and _device_only(args, kwargs) and args[0].is_contiguous():
                out = out.clone()
                self.held.append(out)
                self.marked.add(out.untyped_storage().data_ptr())
            return out
        if name in METADATA or name == "detach":
            return func(*args, **kwargs)
        host_read = (name == "cpu" and len(args) == 1 and not kwargs) or (
            name == "to" and _device_only(args, kwargs))
        if host_read and _whole(args[0]):
            return func(*args, **kwargs).clone()
        self.refused.append(f"{name} on a loaded {tuple(marked[0].shape)} {marked[0].dtype}")
        raise RuntimeError(f"the device refuses {name} on a loaded weight")


def _loaded():
    """test_forward.py's model with seeded random weights for all but the experts, and copies."""
    model = forward._model()
    weights = forward._weights(model)
    for name, value in weights.items():
        path, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(path), leaf, nn.Parameter(value, requires_grad=False))
    return model, {name: value.clone() for name, value in weights.items()}


def _derived(model) -> dict:
    """Every weight the prepare step derives, by tree name."""
    out = {"model.output_score_w": model.model.output_score_w}
    for i, half in enumerate(model.head_weights or (None, None)):
        out[f"head_weights.{i}"] = half
    for i, layer in enumerate(model.model.layers):
        p = f"model.layers.{i}."
        out[p + "attention_score_w"] = layer.attention_score_w
        out[p + "mlp_score_w"] = layer.mlp_score_w
        attn = KDA if layer.is_kda else MLA
        out.update({f"{p}self_attn.{a}": getattr(layer.self_attn, a) for a in attn})
        ffn, q = layer.feed_forward, p + "mlp."
        if isinstance(ffn, m.KimiK3MoE):
            out.update({f"{p}block_sparse_moe.{a}": getattr(ffn, a) for a in MOE})
            ffn, q = ffn.shared_experts, p + "block_sparse_moe.shared_experts."
        out.update({q + a: getattr(ffn, a) for a in MLP})
    return out


def _direct(w: dict, rank: int, size: int) -> dict:
    """Each derived weight by its site's expression applied directly to the loaded weights."""
    def t(name):
        return w[name].t().contiguous()

    def stack(*names):
        return torch.cat([w[n] for n in names]).t().contiguous()

    out = {"model.output_score_w": attn_res_score_weight(
        w["model.output_attn_res_norm.weight"], w["model.output_attn_res_proj.weight"])}
    for i, h in enumerate(w["lm_head.weight"].chunk(2, dim=0)):
        out[f"head_weights.{i}"] = h.t().contiguous()
    for i in range(forward.LAYERS):
        p = f"model.layers.{i}."
        a, f = p + "self_attn.", p + "block_sparse_moe."
        out[p + "attention_score_w"] = attn_res_score_weight(
            w[p + "self_attention_res_norm.weight"], w[p + "self_attention_res_proj.weight"])
        out[p + "mlp_score_w"] = attn_res_score_weight(
            w[p + "mlp_res_norm.weight"], w[p + "mlp_res_proj.weight"])
        if a + "q_proj.weight" in w:
            out[a + "input_weight"] = stack(*(f"{a}{x}_proj.weight" for x in PROJECTIONS))
            taps = [w[f"{a}{x}_conv1d.weight"].squeeze(1) for x in "qkv"]
            out[a + "conv_weight"] = torch.cat(taps).float()
            out[a + "forget_weight"] = t(a + "f_b_proj.weight")
        else:
            latent = torch.cat([w[a + "q_a_proj.weight"], w[a + "kv_a_proj_with_mqa.weight"]])
            rows = latent.shape[1] // size
            k_rows = latent[:, rank * rows : (rank + 1) * rows]
            out[a + "latent_weight"] = k_rows.float().t().contiguous()
            out[a + "input_weight"] = stack(a + "q_a_proj.weight", a + "kv_a_proj_with_mqa.weight")
            out[a + "gate_weight"] = stack(a + "g_proj.weight")
            out[a + "w_qb_t"], out[a + "w_uk"], out[a + "w_uv_t"] = prepare_mla_weights(
                w[a + "q_b_proj.weight"], w[a + "kv_b_proj.weight"])
        out[a + "output_weight"] = t(a + "o_proj.weight")
        q = p + "mlp."
        if f + "gate.weight" in w:
            down_router = torch.cat([w[f + "routed_expert_down_proj.weight"], w[f + "gate.weight"]])
            out[f + "down_router_weight"] = down_router.float().t().contiguous()
            norm = w[f + "routed_expert_norm.weight"].float().chunk(size)[rank]
            out[f + "latent_norm_weight"] = norm.reshape(-1, 1).contiguous()
            out[f + "latent_up_weight"] = t(f + "routed_expert_up_proj.weight")
            q = f + "shared_experts."
            out[f + "shared_down_weight"] = t(q + "down_proj.weight")
        out[q + "gate_up_weight"] = stack(q + "gate_proj.weight", q + "up_proj.weight")
        out[q + "down_weight"] = t(q + "down_proj.weight")
    return out


def _differs(got, want) -> str | None:
    """What differs between two tensors in bits, dtype, shape or strides; None when nothing."""
    if got is None:
        return "missing"
    if (got.dtype, got.shape, got.stride()) != (want.dtype, want.shape, want.stride()):
        return f"{got.dtype} {tuple(got.shape)} {got.stride()}, want {want.dtype} " \
               f"{tuple(want.shape)} {want.stride()}"
    same = torch.equal(got.contiguous().view(torch.uint8), want.contiguous().view(torch.uint8))
    return None if same else "bits"


def test_prepare_step_reads_loaded_weights_whole_and_places_every_derived_weight():
    """Under the device rule the prepare step and the lite load run with 0 refusals."""
    model, saved = _loaded()
    rule, errors = DeviceRule(p for _, p in model.named_parameters() if not p.is_meta), []
    with rule:
        try:
            model.prepare_after_load()
        except Exception as e:  # a refusal, or any other error, fails the item below
            errors.append(repr(e))
    assert not rule.refused and not errors, f"refused {rule.refused}, errors {errors}"
    unplaced = [name for name, t in _derived(model).items() if not rule.is_marked(t)]
    assert not unplaced, f"derived weights not placed: {unplaced}"

    lite, want, errors = forward._model(), _direct(saved, 0, 1), []
    with DeviceRule() as rule:
        try:
            lite.load_weights_lite(str(forward.CHECKPOINT), torch.device("cpu"), None)
        except Exception as e:
            errors.append(repr(e))
    assert not rule.refused and not errors, f"lite load: refused {rule.refused}, errors {errors}"
    wrong = {name: t if t is None else (t.device.type, tuple(t.shape), t.dtype)
             for name, t in _derived(lite).items() if t is None or not t.is_meta
             or (t.shape, t.dtype) != (want[name].shape, want[name].dtype)}
    assert not wrong, f"lite layouts not on meta at the derived shape and dtype: {wrong}"


def _outside_helper(node):
    """``node`` and the nodes under it, but not the arguments of a ``derive_on_host`` call."""
    yield node
    if isinstance(node, ast.Call) and ast.unparse(node.func) == "derive_on_host":
        return
    for child in ast.iter_child_nodes(node):
        yield from _outside_helper(child)


def test_load_scopes_touch_loaded_weights_only_through_the_helper():
    """The load scopes call only the listed callees, and the host functions only in the helper."""
    tree = ast.parse(Path(m.__file__).read_text())
    found = {node.name: node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in SCOPES}
    found.update({f"{c.name}.{f.name}": f for c in tree.body if isinstance(c, ast.ClassDef)
                  for f in c.body
                  if isinstance(f, ast.FunctionDef) and f"{c.name}.{f.name}" in SCOPES})
    assert set(found) == set(SCOPES), f"scopes not found: {set(SCOPES) - set(found)}"
    hits = []
    for scope, fn in found.items():
        for node in (n for stmt in fn.body for n in _outside_helper(stmt)):
            if isinstance(node, ast.Call):
                callee = ast.unparse(node.func)
                if callee not in CALLEES and not callee.endswith(".prepare_after_load"):
                    hits.append(f"{scope}: call {callee} at line {node.lineno}")
            elif isinstance(node, ast.Subscript):
                if (scope, ast.unparse(node)) != (SCOPES[-2], "unloaded[0]"):
                    hits.append(f"{scope}: subscript {ast.unparse(node)} at line {node.lineno}")
    inside = {id(n) for call in ast.walk(tree) if isinstance(call, ast.Call)
              and ast.unparse(call.func) == "derive_on_host"
              for arg in (*call.args, *call.keywords) for n in ast.walk(arg)}
    hits += [f"{node.id} outside the helper at line {node.lineno}" for node in ast.walk(tree)
             if isinstance(node, ast.Name) and node.id in HOST_FUNCTIONS
             and id(node) not in inside]
    assert not hits, f"{len(found)} scopes; layout work outside the helper: {hits}"


def test_derived_weights_equal_their_direct_expressions_in_bits(monkeypatch):
    """At rank 0 of 1 and rank 1 of 4, each derived weight equals its site's direct expression."""
    differ = {}
    for rank, size in ((0, 1), (1, 4)):
        monkeypatch.setattr(m, "_tp_rank_and_size", lambda r=rank, s=size: (r, s))
        model, saved = _loaded()
        model.prepare_after_load()
        got, want = _derived(model), _direct(saved, rank, size)
        assert set(got) == set(want)
        differ.update({(rank, name): d for name in want if (d := _differs(got[name], want[name]))})
        monkeypatch.undo()
    assert not differ, f"derived weights not equal to their direct expressions: {differ}"
