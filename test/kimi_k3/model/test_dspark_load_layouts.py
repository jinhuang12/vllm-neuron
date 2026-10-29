# SPDX-License-Identifier: Apache-2.0
"""The DSpark drafter's load-time layouts read each loaded weight whole and place each result whole.

The drafter is built on meta from its checkpoint's config, at rank 1 of a 4-way tensor-parallel
group, with seeded random CPU values for every parameter. test_load_time_layouts.py's torch
function mode stands in for the Neuron device's copy rule: on a marked tensor it allows only the
metadata reads, ``detach`` and a host read of a whole tensor, and it marks each placement. Set
``KIMI_K3_DSPARK_CHECKPOINT`` to the draft directory (its config only; the default is
``~/models/Inferact/Kimi-K3-DSpark-Block5``); run with ``VLLM_NEURON_CPU_MODE=1``.
"""

import ast
import json
import os
import types
from pathlib import Path

import pytest
import torch
from torch import nn
from test_load_time_layouts import DeviceRule, _differs, _outside_helper
from vllm_neuron.functional.attention.mla_paged import prepare_mla_weights
from vllm_neuron.model.kimi_k3 import dspark

DRAFT = Path(
    os.environ.get(
        "KIMI_K3_DSPARK_CHECKPOINT", Path.home() / "models" / "Inferact" / "Kimi-K3-DSpark-Block5"
    )
)

pytestmark = pytest.mark.skipif(
    not (DRAFT / "config.json").is_file(), reason=f"DSpark config not found at {DRAFT}"
)

RANK, TP, LAYERS, POSITIONS = 1, 4, 3, 64
MLA = ("latent_weight", "output_weight", "w_qb_t", "w_uk", "w_uv_t")
MLP = ("gate_up_weight", "down_weight")
SCOPES = ("K3DSparkMLA.prepare_after_load", "K3DSparkMLP.prepare_after_load",
          "K3DSparkForCausalLM.prepare_after_load", "K3DSparkForCausalLM.load_weights",
          "K3DSparkForCausalLM.load_weights_lite")
CALLEES = {"derive_on_host", "yarn_rope_table", "table.to", "self._restore_unloaded", "_tp_group",
           "SafetensorsCheckpoint", "str", "dist.is_initialized", "load", "k.startswith",
           "ValueError", "sorted", "self.load_state_dict", "dict", "self.named_parameters",
           "self._unloaded.update", "self.named_buffers"}


@pytest.fixture
def rank_1_of_4(monkeypatch):
    """The drafter sees a 4-way tensor-parallel group, as its rank 1."""
    group = types.SimpleNamespace(world_size=TP, rank_in_group=RANK, device_group=None)
    monkeypatch.setattr(dspark, "_tp_group", lambda: group)


def _draft() -> dspark.K3DSparkForCausalLM:
    """The drafter on meta from the checkpoint's config."""
    with open(DRAFT / "config.json") as f:
        config = json.load(f)
    with torch.device("meta"):
        return dspark.K3DSparkForCausalLM.from_configs(
            config, start_layer_idx=93, max_positions=POSITIONS
        )


def _loaded():
    """The drafter with seeded random CPU values for every parameter, and copies of them."""
    draft, gen, saved = _draft(), torch.Generator().manual_seed(0), {}
    for name, p in list(draft.named_parameters()):
        value = (0.02 * torch.randn(p.shape, generator=gen)).to(p.dtype)
        path, _, leaf = name.rpartition(".")
        setattr(draft.get_submodule(path), leaf, nn.Parameter(value, requires_grad=False))
        saved[name] = value.clone()
    return draft, saved


def _derived(draft) -> dict:
    """Every tensor the prepare step derives from loaded weights, by tree name."""
    out = {"context_kv_weight": draft.context_kv_weight}
    for i, layer in enumerate(draft.layers):
        out.update({f"layers.{i}.self_attn.{a}": getattr(layer.self_attn, a) for a in MLA})
        out.update({f"layers.{i}.mlp.{a}": getattr(layer.mlp, a) for a in MLP})
    return out


def _direct(w: dict) -> dict:
    """Each derived tensor by its site's expression applied directly to the loaded weights."""
    kv_a = [w[f"layers.{i}.self_attn.kv_a_proj_with_mqa.weight"] for i in range(LAYERS)]
    out = {"context_kv_weight": torch.cat(kv_a).float()}
    for i in range(LAYERS):
        a, f = f"layers.{i}.self_attn.", f"layers.{i}.mlp."
        latent = torch.cat([w[a + "q_a_proj.weight"], w[a + "kv_a_proj_with_mqa.weight"]])
        out[a + "latent_weight"] = latent.float().t().contiguous()
        out[a + "output_weight"] = w[a + "o_proj.weight"].t().contiguous()
        out[a + "w_qb_t"], out[a + "w_uk"], out[a + "w_uv_t"] = prepare_mla_weights(
            w[a + "q_b_proj.weight"], w[a + "kv_b_proj.weight"])
        gate_up = torch.cat([w[f + "gate_proj.weight"], w[f + "up_proj.weight"]])
        out[f + "gate_up_weight"] = gate_up.t().contiguous()
        out[f + "down_weight"] = w[f + "down_proj.weight"].t().contiguous()
    return out


def test_prepare_step_reads_loaded_weights_whole_and_places_every_derived_tensor(rank_1_of_4):
    """Under the device rule the prepare step runs with 0 refusals and places all 22 tensors."""
    draft, _ = _loaded()
    rule, errors = DeviceRule(draft.parameters()), []
    with rule:
        try:
            draft.prepare_after_load()
        except Exception as e:  # a refusal, or any other error, fails the item below
            errors.append(repr(e))
    assert not rule.refused and not errors, f"refused {rule.refused}, errors {errors}"
    derived = _derived(draft)
    unplaced = [name for name, t in derived.items() if not rule.is_marked(t)]
    assert (len(derived), unplaced) == (22, []), f"derived tensors not placed: {unplaced}"

    lite, errors = _draft(), []
    with DeviceRule() as rule:
        try:
            lite.load_weights_lite(str(DRAFT), torch.device("cpu"), None)
        except Exception as e:
            errors.append(repr(e))
    assert not rule.refused and not errors, f"lite load: refused {rule.refused}, errors {errors}"
    wrong = {name: t if t is None else (t.device.type, tuple(t.shape), t.dtype)
             for name, t in _derived(lite).items() if t is None or not t.is_meta
             or (t.shape, t.dtype) != (derived[name].shape, derived[name].dtype)}
    assert not wrong, f"lite layouts not on meta at the derived shape and dtype: {wrong}"


def test_load_scopes_touch_loaded_weights_only_through_the_helper():
    """The load scopes call only the listed callees, and the MLA layout only in the helper."""
    tree = ast.parse(Path(dspark.__file__).read_text())
    found = {f"{c.name}.{f.name}": f for c in tree.body if isinstance(c, ast.ClassDef)
             for f in c.body if isinstance(f, ast.FunctionDef) and f"{c.name}.{f.name}" in SCOPES}
    assert set(found) == set(SCOPES), f"scopes not found: {set(SCOPES) - set(found)}"
    hits = []
    for scope, fn in found.items():
        for node in (n for stmt in fn.body for n in _outside_helper(stmt)):
            if isinstance(node, ast.Call):
                callee = ast.unparse(node.func)
                if callee not in CALLEES and not callee.endswith(".prepare_after_load"):
                    hits.append(f"{scope}: call {callee} at line {node.lineno}")
            elif isinstance(node, ast.Subscript):
                hits.append(f"{scope}: subscript {ast.unparse(node)} at line {node.lineno}")
    inside = {id(n) for call in ast.walk(tree) if isinstance(call, ast.Call)
              and ast.unparse(call.func) == "derive_on_host"
              for arg in (*call.args, *call.keywords) for n in ast.walk(arg)}
    hits += [f"prepare_mla_weights outside the helper at line {node.lineno}"
             for node in ast.walk(tree) if isinstance(node, ast.Name)
             and node.id == "prepare_mla_weights" and id(node) not in inside]
    assert not hits, f"{len(found)} scopes; layout work outside the helper: {hits}"


def test_derived_tensors_equal_their_direct_expressions_in_bits(rank_1_of_4):
    """At rank 1 of 4, each derived tensor equals its site's direct expression."""
    draft, saved = _loaded()
    draft.prepare_after_load()
    got, want = _derived(draft), _direct(saved)
    assert set(got) == set(want) and len(want) == 22
    differ = {name: d for name in want if (d := _differs(got[name], want[name]))}
    assert not differ, f"derived tensors not equal to their direct expressions: {differ}"
