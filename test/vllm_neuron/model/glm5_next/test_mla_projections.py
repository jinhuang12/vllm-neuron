# SPDX-License-Identifier: Apache-2.0
"""The MLA projection call site: shapes, allocations and numeric agreement.

Shapes are checked against the config's closed form at five widths, and the values
against a torch oracle.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import os
import re
import sys
import threading
import warnings
from pathlib import Path

import pytest
import torch

#: The five projection sites and their widths, transcribed from this file's
#: geometry bullets as they are carried in `### 3L.1`. They are
#: transcribed so that the closed form the code computes is compared against a
#: number from this file, not against itself.
DECLARED_SITES = (
    ("q_a_proj", 4096, 1536),
    ("q_b_proj", 1536, 16384),
    ("kv_a_proj_with_mqa", 4096, 512),
    ("kv_b_proj", 512, 32768),
    ("o_proj", 16384, 4096),
)

#: Every config value the five closed forms are built from, read off the config class
DECLARED_CONFIG_FIELDS = (
    ("hidden_size", 4096),
    ("num_attention_heads", 64),
    ("kv_lora_rank", 512),
    ("q_lora_rank", 1536),
    ("qk_nope_head_dim", 256),
    ("qk_rope_head_dim", 0),
    ("v_head_dim", 256),
    ("mla_use_nope", True),
    ("rms_norm_eps", 1e-05),
)

#: The sequence length every case runs at. Above the sub-kernel-selection
#: threshold the kernel's check 2 reads from its defining line, which closes the
#: same dodge here: a shorter sequence would route to the sub-kernel that carries
#: no width bound at all, so it would prove nothing about the refusal.
DECLARED_SEQ = 128
CTE_THRESHOLD = 96

#: The declared section 3 threshold for a bf16 module comparison. Quoted, never
#: authored here and never widened to reach green.
RTOL = 1e-2
ATOL = 1e-5

#: A reduced geometry for the numeric case. Only the widths shrink — the sequence
#: length, the rotary width and the head-count structure are the checkpoint's.
TINY_OVERRIDES = {
    "hidden_size": 256,
    "num_attention_heads": 4,
    "q_lora_rank": 64,
    "kv_lora_rank": 32,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 0,
    "v_head_dim": 16,
}

#: Path fragments naming the two members measured as refusing.
#: Resolved from the installed distribution rather than guessed.
VENDOR_PATH_FRAGMENTS = ("core/qkv/", "core/output_projection/")

F1_CHAIN_FRAGMENTS = ("libtorch_neuronx_lite/nki/nki_hop.py", "nki/_simulator.py")

#: The torch forms check 5 counts. Matched by the leaf attribute name, so
#: `nisa.nc_matmul` does not collide with `torch.matmul`.
TORCH_MATMUL_ATTRS = frozenset({"linear", "matmul", "einsum"})

OWNED_CLASS = "Glm5NextMLAAttention"

ROTARY_NAME = re.compile(r"rope|rotary", re.IGNORECASE)


def _impl():
    """Import the implementation module inside a test body, never at import."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _config_module():
    from vllm_neuron.model.glm5_next import config

    return config


def real_config():
    """The checkpoint geometry, which is this dataclass's own default set."""
    return _config_module().Glm5NextTextConfig()


def tiny_config():
    """The same class with the widths reduced, for the numeric case."""
    return dataclasses.replace(real_config(), **TINY_OVERRIDES)


def closed_form_widths(cfg) -> tuple[tuple[str, int, int], ...]:
    """The five sites, computed here from config values, independently. """
    heads = int(cfg.num_attention_heads)
    qk_head_dim = int(cfg.qk_nope_head_dim) + int(cfg.qk_rope_head_dim)
    return (
        ("q_a_proj", int(cfg.hidden_size), int(cfg.q_lora_rank)),
        ("q_b_proj", int(cfg.q_lora_rank), heads * qk_head_dim),
        (
            "kv_a_proj_with_mqa",
            int(cfg.hidden_size),
            int(cfg.kv_lora_rank) + int(cfg.qk_rope_head_dim),
        ),
        (
            "kv_b_proj",
            int(cfg.kv_lora_rank),
            heads * (int(cfg.qk_nope_head_dim) + int(cfg.v_head_dim)),
        ),
        ("o_proj", heads * int(cfg.v_head_dim), int(cfg.hidden_size)),
    )


def build_attention(cfg, seed: int = 20390):
    """An MLA attention module with its five projections and two norms loaded. """
    module = _impl().Glm5NextMLAAttention(cfg)
    gen = torch.Generator().manual_seed(seed)
    for name, idim, odim in closed_form_widths(cfg):
        scale = float(idim) ** -0.5
        weight = torch.randn(odim, idim, generator=gen, dtype=torch.float32) * scale
        setattr(module, f"{name}_weight", torch.nn.Parameter(weight))
    for name, width in (
        ("q_a_layernorm_weight", int(cfg.q_lora_rank)),
        ("kv_a_layernorm_weight", int(cfg.kv_lora_rank) + int(cfg.qk_rope_head_dim)),
    ):
        gain = 1.0 + torch.randn(width, generator=gen, dtype=torch.float32) * 0.05
        setattr(module, name, torch.nn.Parameter(gain))
    return module


def owned_class_node() -> tuple[ast.ClassDef, str]:
    """The owned class as a syntax tree, plus the source it was parsed from."""
    source = inspect.getsource(_impl())
    tree = ast.parse(source)
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == OWNED_CLASS
    )
    return node, source


def _dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def trace_projection(module, hidden, attn_out):
    """Run both projection entry points under a frame trace. """
    counts = {"vendor": [], "chain": {f: 0 for f in F1_CHAIN_FRAGMENTS}, "events": 0}
    threads: set[str] = set()

    def profile(frame, event, arg):  # noqa: ANN001, ARG001 -- CPython's signature
        counts["events"] += 1
        filename = frame.f_code.co_filename.replace(os.sep, "/")
        for fragment in VENDOR_PATH_FRAGMENTS:
            if fragment in filename:
                counts["vendor"].append(f"{filename}::{frame.f_code.co_name}")
                return
        for fragment in F1_CHAIN_FRAGMENTS:
            if fragment in filename:
                counts["chain"][fragment] += 1
                threads.add(threading.current_thread().name)

    threading.setprofile(profile)
    sys.setprofile(profile)
    try:
        query, key_nope, value = module.project_qkv(hidden)
        projected = module.project_output(attn_out)
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
    counts["threads"] = sorted(threads)
    counts["out"] = (query, key_nope, value, projected)
    return counts


def test_projected_shapes_match_the_closed_form_at_five_widths() -> None:
    """check 1 of 5 — 5/5 sites: projected shapes equal the closed form. """

    config_path = Path(inspect.getsourcefile(_config_module()))
    config_tree = ast.parse(config_path.read_text())
    class_node = next(
        n for n in config_tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Glm5NextTextConfig"
    )
    defined: dict[str, tuple[object, int]] = {}
    for node in class_node.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                try:
                    defined[node.target.id] = (
                        ast.literal_eval(node.value),
                        node.lineno,
                    )
                except ValueError:
                    continue
    for name, expected_value in DECLARED_CONFIG_FIELDS:
        got_value, _ = defined[name]
        assert got_value == expected_value, (
            f"{name} is {got_value!r}, the declared geometry says "
            f"{expected_value!r}"
        )

    cfg = real_config()
    module = build_attention(cfg)
    expected = closed_form_widths(cfg)
    assert expected == DECLARED_SITES, (
        f"the closed form computed from config values is {expected}, which does "
        f"not reproduce the declared transcribed widths {DECLARED_SITES}"
    )
    assert module.projection_widths() == DECLARED_SITES, (
        f"the module's own widths are {module.projection_widths()}, not the "
        f"plan's {DECLARED_SITES}"
    )

    prepared_count = module.prepare_projection_weights()
    assert prepared_count == 5, f"expected 5 prepared weights, got {prepared_count}"

    heads = int(cfg.num_attention_heads)
    hidden = torch.zeros(DECLARED_SEQ, int(cfg.hidden_size), dtype=torch.float32)
    attn_out = torch.zeros(
        DECLARED_SEQ, heads, int(cfg.v_head_dim), dtype=torch.float32
    )
    query, key_nope, value = module.project_qkv(hidden)
    projected = module.project_output(attn_out)

    prepared = getattr(module, module.PREPARED_WEIGHTS_ATTR)
    sites_exact = 0
    for name, idim, odim in DECLARED_SITES:
        got = tuple(prepared[name].shape)
        assert got == (idim, odim), (
            f"{name}: the prepared weight is {got}, contraction-major closed "
            f"form is {(idim, odim)}"
        )
        sites_exact += 1

    observed = {
        "query": tuple(query.shape),
        "key_nope": tuple(key_nope.shape),
        "value": tuple(value.shape),
        "project_output": tuple(projected.shape),
    }
    expected_observed = {
        "query": (DECLARED_SEQ, heads, int(cfg.qk_nope_head_dim)),
        "key_nope": (DECLARED_SEQ, heads, int(cfg.qk_nope_head_dim)),
        "value": (DECLARED_SEQ, heads, int(cfg.v_head_dim)),
        "project_output": (DECLARED_SEQ, int(cfg.hidden_size)),
    }
    assert observed == expected_observed, (
        f"end-to-end shapes {observed} do not match the closed-form "
        f"expectation {expected_observed}"
    )

    assert sites_exact == 5, f"expected all 5 sites exact, got {sites_exact}"


def test_zero_rotary_parameters_allocated() -> None:
    """check 2 of 5 — a counted zero: 0 rotary parameters allocated. """

    cfg = real_config()
    module = build_attention(cfg)

    assert int(cfg.qk_rope_head_dim) == 0, (
        f"this check is scoped to a NoPE checkpoint; qk_rope_head_dim is "
        f"{cfg.qk_rope_head_dim}"
    )
    assert bool(cfg.mla_use_nope) is True, "mla_use_nope is not set"

    declared = tuple(getattr(module, "declared_param_names", ()))
    declared_rotary = [n for n in declared if ROTARY_NAME.search(n)]

    live = [n for n, p in module.named_parameters() if p is not None]
    live_rotary = [n for n in live if ROTARY_NAME.search(n)]

    node, _source = owned_class_node()
    declared_in_source: list[str] = []
    scalar_widths: list[str] = []
    tensor_under_rotary_name: list[str] = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            leaf = _dotted(n.func).rsplit(".", 1)[-1]
            if leaf in {"_declare_parameters", "register_parameter"}:
                for a in n.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        if ROTARY_NAME.search(a.value):
                            declared_in_source.append(f"line {n.lineno}: {a.value}")
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Attribute) and ROTARY_NAME.search(t.attr):
                    coercion = (
                        isinstance(n.value, ast.Call)
                        and _dotted(n.value.func) in {"int", "float", "bool"}
                    )
                    entry = f"line {n.lineno}: self.{t.attr}"
                    (scalar_widths if coercion else tensor_under_rotary_name).append(
                        entry
                    )
    assert scalar_widths, (
        "no rotary-named scalar width is assigned in the class, so this "
        "check's premise is unverifiable here; the instrument is broken"
    )

    widths = dict((n, (i, o)) for n, i, o in module.projection_widths())
    heads = int(cfg.num_attention_heads)
    assert widths["q_b_proj"][1] // heads == int(cfg.qk_nope_head_dim), (
        "the query head width carries a rotary slice on a NoPE checkpoint"
    )
    assert widths["kv_a_proj_with_mqa"][1] == int(cfg.kv_lora_rank), (
        "the latent width carries a rotary slice on a NoPE checkpoint"
    )

    total_rotary = (
        len(declared_rotary)
        + len(live_rotary)
        + len(declared_in_source)
        + len(tensor_under_rotary_name)
    )
    assert total_rotary == 0, (
        f"a rotary parameter exists on a NoPE checkpoint: declared "
        f"{declared_rotary}, live {live_rotary}, declared in source "
        f"{declared_in_source}, non-scalar {tensor_under_rotary_name}"
    )


def test_numeric_agreement_against_a_torch_oracle() -> None:
    """check 3 of 5 — 1/1 case at S = 128 matches a torch projection oracle. """

    cfg = tiny_config()
    assert DECLARED_SEQ > CTE_THRESHOLD, (
        f"S={DECLARED_SEQ} does not exceed {CTE_THRESHOLD}; this case would "
        f"exercise the sub-kernel that carries no width bound"
    )
    assert int(cfg.qk_rope_head_dim) == 0, "the reduction changed the rotary width"
    assert int(cfg.num_attention_heads) >= 2, "the reduction removed the head axis"

    module = build_attention(cfg, seed=30390)
    module.prepare_projection_weights()

    heads = int(cfg.num_attention_heads)
    nope = int(cfg.qk_nope_head_dim)
    vdim = int(cfg.v_head_dim)
    gen = torch.Generator().manual_seed(40390)
    hidden = torch.randn(
        DECLARED_SEQ, int(cfg.hidden_size), generator=gen, dtype=torch.float32
    ) * 0.05
    attn_out = torch.randn(
        DECLARED_SEQ, heads, vdim, generator=gen, dtype=torch.float32
    ) * 0.05

    def norm(x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + float(cfg.rms_norm_eps)) * gain

    def linear(x: torch.Tensor, name: str) -> torch.Tensor:
        weight = getattr(module, f"{name}_weight").to(torch.float32)
        return torch.matmul(x, weight.t())

    q_latent = norm(
        linear(hidden, "q_a_proj"), module.q_a_layernorm_weight.to(torch.float32)
    )
    ref_query = linear(q_latent, "q_b_proj").reshape(DECLARED_SEQ, heads, nope)
    kv_latent = norm(
        linear(hidden, "kv_a_proj_with_mqa"),
        module.kv_a_layernorm_weight.to(torch.float32),
    )
    ref_kv = linear(kv_latent, "kv_b_proj").reshape(DECLARED_SEQ, heads, nope + vdim)
    ref_key_nope = ref_kv[..., :nope]
    ref_value = ref_kv[..., nope:]
    ref_out = linear(attn_out.reshape(DECLARED_SEQ, heads * vdim), "o_proj")

    query, key_nope, value = module.project_qkv(hidden)
    projected = module.project_output(attn_out)

    worst = 0.0
    for _label, got, ref in (
        ("query", query, ref_query),
        ("key_nope", key_nope, ref_key_nope),
        ("value", value, ref_value),
        ("project_output", projected, ref_out),
    ):
        err = float((got.to(torch.float32) - ref).detach().abs().max())
        worst = max(worst, err)
        torch.testing.assert_close(
            got.to(torch.float32), ref, rtol=RTOL, atol=ATOL
        )


def test_route_predicate_r2_five_simulator_dispatches() -> None:
    """The projection call site makes five kernel dispatches and no torch fallback."""

    from vllm_neuron.functional.attention import mla_projections as MP
    from vllm_neuron.utils.neuron_utils import can_run_kernel

    cfg = tiny_config()
    module = build_attention(cfg, seed=50390)
    module.prepare_projection_weights()

    heads = int(cfg.num_attention_heads)
    hidden = torch.zeros(DECLARED_SEQ, int(cfg.hidden_size), dtype=torch.float32)
    attn_out = torch.zeros(
        DECLARED_SEQ, heads, int(cfg.v_head_dim), dtype=torch.float32
    )

    gate = can_run_kernel(hidden)
    assert gate is True, (
        "can_run_kernel is not True, so the NKI route is unavailable and R-2 "
        "cannot be satisfied by correct code"
    )

    # The seam's own gate, read at each of the five widths. `can_run_kernel()` answers
    # "is the simulator route available at all"; this answers "does the kernel serve
    # this geometry", which is the
    # question a refusing substrate member answers with False. Both are asserted
    # because a True on the first with a False on the second is precisely the
    # state in which a call site would be entitled to fall back.
    for name, idim, odim in closed_form_widths(cfg):
        site_gate = MP.can_run_mla_projection(hidden, DECLARED_SEQ, idim, odim)
        assert site_gate is True, f"the seam refuses {name} at {idim}->{odim}"

    MP.reset_mla_projection_dispatch_counters()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        counts = trace_projection(module, hidden, attn_out)

    nki_dispatch, torch_fallback = MP.mla_projection_dispatch_counters()

    assert nki_dispatch == 5, (
        f"R-2 requires 5 simulator dispatches, one per projection site; the "
        f"seam counted {nki_dispatch}"
    )
    assert torch_fallback == 0, (
        f"the fallback counter must read 0; it read {torch_fallback}"
    )
    for fragment, n in counts["chain"].items():
        assert n > 0, (
            f"the trace never entered {fragment}, so the kernel chain this file "
            f"names was not the route taken"
        )


def test_counted_zeros_on_vendor_seams_and_torch_matmul() -> None:
    """check 5 of 5 — two counted zeros: no refused member, no torch matmul. """

    cfg = tiny_config()
    module = build_attention(cfg, seed=60390)
    module.prepare_projection_weights()
    heads = int(cfg.num_attention_heads)
    hidden = torch.zeros(DECLARED_SEQ, int(cfg.hidden_size), dtype=torch.float32)
    attn_out = torch.zeros(
        DECLARED_SEQ, heads, int(cfg.v_head_dim), dtype=torch.float32
    )

    counts = trace_projection(module, hidden, attn_out)
    chain_total = sum(counts["chain"].values())
    assert chain_total > 0, (
        "the trace observed no frame anywhere on the kernel chain, so its zero on "
        "the refused members would be vacuous; the instrument is broken"
    )
    assert not counts["vendor"], (
        f"a refused member was DISPATCHED from this call site: "
        f"{sorted(set(counts['vendor']))[:5]}"
    )

    node, _source = owned_class_node()
    hits = []
    for n in ast.walk(node):
        line = getattr(n, "lineno", None)
        if line is None:
            continue
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult):
            hits.append(f"line {line}: MATMUL_OPERATOR")
        elif isinstance(n, ast.Call):
            dotted = _dotted(n.func)
            leaf = dotted.rsplit(".", 1)[-1] if dotted else ""
            if leaf in TORCH_MATMUL_ATTRS:
                hits.append(f"line {line}: CALL:{dotted}")
    assert not hits, (
        f"the call site carries a torch matmul form, which is the fallback this "
        f"check exists to catch: {hits}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
