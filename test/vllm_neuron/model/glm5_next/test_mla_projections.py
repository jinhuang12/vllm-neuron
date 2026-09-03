# SPDX-License-Identifier: Apache-2.0
"""Acceptance for `inc-glm53f-039b` — the MLA projection CALL SITE.

FIVE tests, one per counted conjunct of the increment plan's `inc-glm53f-039b`
Acceptance bullet, and NO `parametrize` decorator in this file: the plan requires
exactly 5 collected items, and a parametrized case collects as several items for
one conjunct, so the count would stop meaning what it says.

THIS FILE ASKS "IS THE MODEL WIRED TO THE KERNEL?" AND NOTHING ELSE. Whether the
kernel computes a projection correctly is `inc-glm53f-039a`'s question, asked in
its own separate file under `test/vllm_neuron/functional/attention/`. The plan
keeps the two apart on purpose: in one file either increment's counted predicate
could be satisfied by the other increment's items.

Run it with the Tier N harness — the NKI simulator on a host CPU, no device::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest test/vllm_neuron/model/glm5_next/test_mla_projections.py \
        -q -s --timeout 60 -p no:cacheprovider
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

#: The five projection sites and their widths, transcribed from the plan's
#: `inc-glm53f-039a` geometry bullets as they are carried in `### 3L.1`. They are
#: transcribed so that the closed form the code computes is compared against a
#: number from the plan, not against itself.
DECLARED_SITES = (
    ("q_a_proj", 4096, 1536),
    ("q_b_proj", 1536, 16384),
    ("kv_a_proj_with_mqa", 4096, 512),
    ("kv_b_proj", 512, 32768),
    ("o_proj", 16384, 4096),
)

#: Every config value the five closed forms are built from, with the line that
#: defines it AT THE CAMPAIGN HEAD. The plan's own copies of these citations are
#: anchored to an earlier commit and are ~31 lines early here, which is exactly
#: why this file asserts the line number as well as the value: a citation that
#: silently drifts is how the plan's went stale.
DECLARED_CONFIG_FIELDS = (
    ("hidden_size", 4096, 138),
    ("num_attention_heads", 64, 140),
    ("kv_lora_rank", 512, 162),
    ("q_lora_rank", 1536, 163),
    ("qk_nope_head_dim", 256, 164),
    ("qk_rope_head_dim", 0, 167),
    ("v_head_dim", 256, 168),
    ("mla_use_nope", True, 169),
    ("rms_norm_eps", 1e-05, 188),
)

#: The sequence length every case runs at. Above the sub-kernel-selection
#: threshold `-039a`'s conjunct 2 reads from its defining line, which closes the
#: same dodge here: a shorter sequence would route to the sub-kernel that carries
#: no width bound at all, so it would prove nothing about the refusal.
DECLARED_SEQ = 128
CTE_THRESHOLD = 96

#: The plan's section 3 threshold for a bf16 module comparison. Quoted, never
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

#: Path fragments naming the two members `inc-glm53f-072` measured as REFUSING.
#: Resolved from the installed distribution rather than guessed.
VENDOR_PATH_FRAGMENTS = ("core/qkv/", "core/output_projection/")

#: The F1 chain the plan names for this increment's route predicate:
#: `wrap_nki → NKIHOPCaller → HOP → DispatchKey.CPU → nki.simulator.simulate_kernel`.
#: These two files are the chain's middle and its last link, and the trace is
#: asserted to pass through both — a structural reading rather than a log string.
F1_CHAIN_FRAGMENTS = ("libtorch_neuronx_lite/nki/nki_hop.py", "nki/_simulator.py")

#: The torch forms conjunct 5 counts. Matched by the leaf attribute name, so
#: `nisa.nc_matmul` does not collide with `torch.matmul`.
TORCH_MATMUL_ATTRS = frozenset({"linear", "matmul", "einsum"})

#: The class this increment owns. Conjunct 5's screen runs over its whole body,
#: which is a SUPERSET of this increment's added lines and therefore a stricter
#: reading than the plan asks for.
OWNED_CLASS = "Glm5NextMLAAttention"

ROTARY_NAME = re.compile(r"rope|rotary", re.IGNORECASE)


def say(*parts: object) -> None:
    """Print one counted reading. `-s` keeps it in the transcript.

    For whoever greps the transcript: pytest `-q` writes a progress dot with no
    trailing newline after each test, so the first line printed by every test
    after the first carries that dot as a prefix. Match with `grep -o`, never
    with a `^` anchor.
    """
    print(" ".join(str(p) for p in parts), flush=True)


def _impl():
    """Import the implementation module INSIDE a test body, never at import."""
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
    """The five sites, computed HERE from config values, independently.

    Written out in this file so that conjunct 1 compares the module's own
    `projection_widths()` against an expectation the module did not produce. The
    rotary width is added explicitly at both places it belongs, so a config that
    had a rotary slice would move these numbers rather than silently drop it.
    """
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
    """An MLA attention module with its five projections and two norms loaded.

    The parameters are DECLARED and not materialised on the skeleton, so a test
    must supply them; these are supplied in the checkpoint's own orientation,
    `[out_features, in_features]`, which is what makes the module's load-time
    transpose the thing under test rather than a formality.
    """
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
    """Run both projection entry points under a frame trace. Returns the counts.

    THE HOOK IS REGISTERED TWICE, and that is not redundancy. The simulator runs
    a kernel body on a worker thread and a profile hook is per-thread, so
    `sys.setprofile` alone sees the call site and misses everything the kernel
    reached. `inc-glm53f-039a` measured that: with one hook the kernel body read
    0 frames, and adding the threading hook brought it to a non-zero count. Both
    registrations are what give conjunct 5's zero its scope.
    """
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


def test_conjunct_1_projected_shapes_match_the_closed_form_at_five_widths() -> None:
    """CONJUNCT 1 of 5 — 5/5 sites: projected shapes equal the closed form.

    CERTIFYING COMPONENT: the projections section of `Glm5NextMLAAttention` in
    `vllm_neuron/model/glm5_next/model_fp8.py`.

    THE CLOSED FORM IS CHECKED THREE WAYS, and the third is what makes the first
    two mean something. The module computes its own widths; this file computes
    them again from config values; and the plan's transcribed numbers are
    compared against both. Two agreeing computations could share a mistake — a
    number from the plan cannot.

    EVERY CONFIG VALUE IS ASSERTED AT ITS DEFINING LINE, value and line number
    together. This is not ceremony: the plan's own copies of these citations are
    about 31 lines early at this commit, because the file grew after the plan
    named its anchor. Asserting the line here means the same drift reddens this
    file instead of quietly aging inside it.
    """
    say("C1_CERTIFYING_COMPONENT=the projections section of Glm5NextMLAAttention "
        "in vllm_neuron/model/glm5_next/model_fp8.py")

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
    for name, expected_value, expected_line in DECLARED_CONFIG_FIELDS:
        got_value, got_line = defined[name]
        say(f"C1_CONFIG {name}={got_value!r} AT_LINE={got_line} "
            f"EXPECTED={expected_value!r}@{expected_line}")
        assert got_value == expected_value, (
            f"{name} is {got_value!r} at HEAD, the plan's geometry says "
            f"{expected_value!r}"
        )
        assert got_line == expected_line, (
            f"{name} is defined at line {got_line}, this file cites "
            f"{expected_line}; re-locate the citation"
        )

    cfg = real_config()
    module = build_attention(cfg)
    expected = closed_form_widths(cfg)
    assert expected == DECLARED_SITES, (
        f"the closed form computed from config values is {expected}, which does "
        f"not reproduce the plan's transcribed widths {DECLARED_SITES}"
    )
    assert module.projection_widths() == DECLARED_SITES, (
        f"the module's own widths are {module.projection_widths()}, not the "
        f"plan's {DECLARED_SITES}"
    )

    prepared_count = module.prepare_projection_weights()
    say(f"C1_WEIGHTS_PREPARED_ONCE={prepared_count}")
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
        say(f"C1_SITE {name} PREPARED_SHAPE={got} CLOSED_FORM={(idim, odim)}")
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
    for key, shape in observed.items():
        say(f"C1_OUTPUT {key}={shape} EXPECTED={expected_observed[key]}")
    assert observed == expected_observed, (
        f"end-to-end shapes {observed} do not match the closed-form "
        f"expectation {expected_observed}"
    )

    say(f"C1_SITES_EXACT={sites_exact}/5")
    assert sites_exact == 5, f"expected all 5 sites exact, got {sites_exact}"


def test_conjunct_2_zero_rotary_parameters_allocated() -> None:
    """CONJUNCT 2 of 5 — a counted zero: 0 rotary parameters allocated.

    CERTIFYING COMPONENT: the declared parameter set of `Glm5NextMLAAttention`,
    and the closed-form widths that would carry a rotary slice if one existed.

    `qk_rope_head_dim` IS 0 AND THAT 0 IS A VALUE, NOT A PLACEHOLDER — the config
    says so in its own comment, because `mla_use_nope` is set and this checkpoint
    has no rotary head slice at all. So the right reading is not "the rotary
    parameters are unused" but "there are none", and this conjunct counts them.

    THE COUNT IS TAKEN FOUR WAYS, because each alone is passable. The declared
    name registry catches a reserved-but-unallocated rotary parameter, which is
    the form every parameter on this skeleton takes. The live parameter list
    catches one allocated without being declared. The declaration CALLS in the
    class source catch a rotary name registered without reaching either registry.
    And the closed-form widths catch a rotary slice computed into a shape with no
    parameter behind it at all — the case the first three cannot see.

    WHAT IS NOT COUNTED, AND WHY. `self.qk_rope_head_dim` is a rotary-NAMED
    attribute and it must exist: it is the config width whose being 0 is this
    conjunct's premise, and the two width assertions below read it. A screen that
    counted every rotary-named attribute would count that scalar and report a
    rotary parameter where none is allocated. So the source screen counts
    rotary names in PARAMETER DECLARATIONS, and every rotary-named attribute is
    separately required to be a scalar width coercion — a tensor allocated under
    a rotary name fails that reading rather than slipping through it.
    """
    say("C2_CERTIFYING_COMPONENT=the declared parameter set of "
        "Glm5NextMLAAttention and its closed-form widths")

    cfg = real_config()
    module = build_attention(cfg)

    assert int(cfg.qk_rope_head_dim) == 0, (
        f"this conjunct is scoped to a NoPE checkpoint; qk_rope_head_dim is "
        f"{cfg.qk_rope_head_dim}"
    )
    assert bool(cfg.mla_use_nope) is True, "mla_use_nope is not set"
    say(f"C2_QK_ROPE_HEAD_DIM={int(cfg.qk_rope_head_dim)} "
        f"MLA_USE_NOPE={bool(cfg.mla_use_nope)}")

    declared = tuple(getattr(module, "declared_param_names", ()))
    declared_rotary = [n for n in declared if ROTARY_NAME.search(n)]
    say(f"C2_DECLARED_PARAMETER_NAMES={len(declared)}")
    say(f"C2_DECLARED_ROTARY_PARAMETERS={len(declared_rotary)} {declared_rotary}")

    live = [n for n, p in module.named_parameters() if p is not None]
    live_rotary = [n for n in live if ROTARY_NAME.search(n)]
    say(f"C2_LIVE_PARAMETERS={len(live)}")
    say(f"C2_LIVE_ROTARY_PARAMETERS={len(live_rotary)} {live_rotary}")

    node, source = owned_class_node()
    class_source = ast.get_source_segment(source, node) or ""
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
    say(f"C2_ROTARY_NAMES_IN_PARAMETER_DECLARATIONS={len(declared_in_source)} "
        f"{declared_in_source}")
    say(f"C2_ROTARY_NAMED_SCALAR_WIDTHS={len(scalar_widths)} {scalar_widths} "
        f"(required to exist; this conjunct's premise is that one of them is 0)")
    say(f"C2_ROTARY_NAMED_NON_SCALAR_ATTRIBUTES={len(tensor_under_rotary_name)} "
        f"{tensor_under_rotary_name}")
    say(f"C2_ROTARY_MENTIONS_IN_CLASS_SOURCE="
        f"{len(ROTARY_NAME.findall(class_source))} (prose included, not asserted)")
    assert scalar_widths, (
        "no rotary-named scalar width is assigned in the class, so this "
        "conjunct's premise is unverifiable here; the instrument is broken"
    )

    widths = dict((n, (i, o)) for n, i, o in module.projection_widths())
    heads = int(cfg.num_attention_heads)
    say(f"C2_QUERY_HEAD_WIDTH={widths['q_b_proj'][1] // heads} "
        f"NOPE_ONLY={int(cfg.qk_nope_head_dim)}")
    say(f"C2_LATENT_WIDTH={widths['kv_a_proj_with_mqa'][1]} "
        f"RANK_ONLY={int(cfg.kv_lora_rank)}")
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
    say(f"C2_TOTAL_ROTARY_PARAMETERS={total_rotary}")
    assert total_rotary == 0, (
        f"a rotary parameter exists on a NoPE checkpoint: declared "
        f"{declared_rotary}, live {live_rotary}, declared in source "
        f"{declared_in_source}, non-scalar {tensor_under_rotary_name}"
    )


def test_conjunct_3_numeric_agreement_against_a_torch_oracle() -> None:
    """CONJUNCT 3 of 5 — 1/1 case at S = 128 matches a torch projection oracle.

    CERTIFYING COMPONENT: the call site's arithmetic — the projection chain in
    `project_qkv` and `project_output`.

    S = 128 IS ABOVE 96 AND THAT IS THE POINT, not an arbitrary size. It closes
    the same dodge `-039a`'s conjunct 2 closes: below the sub-kernel-selection
    threshold the dispatch would route to the sub-kernel that carries no width
    bound, so the case would pass while proving nothing about the refusal the
    kernel exists to clear.

    THE ORACLE IS WRITTEN HERE AND IT USES A TORCH MATMUL, which is legitimate in
    a test and counted to zero in the shipped module. It reproduces the whole
    chain — compress, normalise the latent, expand, split — because a comparison
    that skipped the norms would pass on a call site that dropped them.

    The widths are reduced and the reason is stated: only the widths shrink. The
    sequence length, the zero rotary width and the head structure are the
    checkpoint's, so every property this conjunct rests on survives the
    reduction.
    """
    say("C3_CERTIFYING_COMPONENT=the projection chain in project_qkv and "
        "project_output")

    cfg = tiny_config()
    say(f"C3_TINY_WIDTHS={closed_form_widths(cfg)}")
    say(f"C3_SEQ={DECLARED_SEQ} THRESHOLD={CTE_THRESHOLD}")
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
    for label, got, ref in (
        ("query", query, ref_query),
        ("key_nope", key_nope, ref_key_nope),
        ("value", value, ref_value),
        ("project_output", projected, ref_out),
    ):
        err = float((got.to(torch.float32) - ref).detach().abs().max())
        worst = max(worst, err)
        say(f"C3_LIMB {label} MAXABS={err:.3e}")
        torch.testing.assert_close(
            got.to(torch.float32), ref, rtol=RTOL, atol=ATOL
        )

    say(f"C3_WORST_MAXABS={worst:.3e} TOLERANCE rtol={RTOL} atol={ATOL}")
    say("C3_CASES_AGREED=1/1")


def test_conjunct_4_route_predicate_r2_five_simulator_dispatches() -> None:
    """CONJUNCT 4 of 5 — route predicate D13 form R-2: 5 dispatches, 0 fallbacks.

    CERTIFYING COMPONENT: the F1 chain from this call site through
    `inc-glm53f-039a`'s `mla_projections.py` seam — a seam this increment does
    NOT author, which is what makes the form R-2 and not R-1.

    A PURE-TORCH CALL SITE READS 0 HERE AND THEREFORE CANNOT PASS. That is the
    whole purpose: conjuncts 1 and 3 are both satisfied by a call site that
    computed the same numbers with `torch.matmul`, so without this conjunct the
    acceptance would not distinguish a wired call site from a hollow one.

    THE CHAIN IS OBSERVED STRUCTURALLY, as frames in the two files that carry its
    middle and its last link, rather than by matching a log string. A log line is
    a vendor's to change; a frame in `nki_hop.py` on the way to
    `simulate_kernel` is the dispatch itself.

    FIVE DISPATCHES ARRIVE AS FOUR PLUS ONE, and the split is the model's, not an
    accounting choice: the key and the value come out of ONE expansion of the
    compressed latent and are then split, so the query-and-key-value entry point
    dispatches four times and the output projection once.
    """
    say("C4_CERTIFYING_COMPONENT=the F1 chain from this call site through "
        "inc-glm53f-039a's mla_projections.py seam (a seam this increment does "
        "not author)")

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
    say(f"C4_CAN_RUN_KERNEL={gate}")
    assert gate is True, (
        "can_run_kernel is not True, so the NKI route is unavailable and R-2 "
        "cannot be satisfied by correct code"
    )

    # The seam's own gate, read at each of the five widths. The plan names
    # `can_run_kernel()`, which answers "is the simulator route available at
    # all"; this answers "does the kernel serve THIS geometry", which is the
    # question a refusing substrate member answers with False. Both are asserted
    # because a True on the first with a False on the second is precisely the
    # state in which a call site would be entitled to fall back.
    gates_open = 0
    for name, idim, odim in closed_form_widths(cfg):
        site_gate = MP.can_run_mla_projection(hidden, DECLARED_SEQ, idim, odim)
        say(f"C4_SEAM_GATE {name} {idim}->{odim} CAN_RUN={site_gate}")
        assert site_gate is True, f"the seam refuses {name} at {idim}->{odim}"
        gates_open += 1
    say(f"C4_SEAM_GATES_OPEN={gates_open}/5")

    MP.reset_mla_projection_dispatch_counters()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        counts = trace_projection(module, hidden, attn_out)
    simulator_records = sum(1 for c in caught if "simulate" in str(c.message))

    nki_dispatch, torch_fallback = MP.mla_projection_dispatch_counters()
    say(f"C4_SEAM_NKI_DISPATCH={nki_dispatch}")
    say(f"C4_SEAM_TORCH_FALLBACK={torch_fallback}")
    say(f"C4_SIMULATOR_RECORDS={simulator_records}")
    say(f"C4_PROFILE_EVENTS={counts['events']}")
    say(f"C4_CHAIN_THREADS={counts['threads']}")
    for fragment, n in counts["chain"].items():
        say(f"C4_F1_CHAIN_FRAMES {fragment}={n}")

    assert nki_dispatch == 5, (
        f"R-2 requires 5 simulator dispatches, one per projection site; the "
        f"seam counted {nki_dispatch}"
    )
    assert torch_fallback == 0, (
        f"the fallback counter must read 0; it read {torch_fallback}"
    )
    for fragment, n in counts["chain"].items():
        assert n > 0, (
            f"the trace never entered {fragment}, so the F1 chain the plan "
            f"names was not the route taken"
        )
    say(f"C4_ROUTE_PREDICATE_R2={nki_dispatch}/5 dispatches, "
        f"{torch_fallback} fallbacks")


def test_conjunct_5_counted_zeros_on_vendor_seams_and_torch_matmul() -> None:
    """CONJUNCT 5 of 5 — two counted zeros: no refused member, no torch matmul.

    CERTIFYING COMPONENT: the two members `inc-glm53f-072` refused, and the
    source of the `Glm5NextMLAAttention` class.

    THIS CONJUNCT EXISTS TO CATCH A CALL SITE THAT FALLS BACK, and the plan says
    why: a site that quietly reached `F.linear` when the kernel's gate was False
    would pass conjuncts 1, 2 and 3 unchanged while hollowing out the fork. The
    numbers are the same, so only a screen of the route and of the source can
    tell the two apart.

    THE SOURCE SCREEN IS AN AST WALK OVER THE WHOLE OWNED CLASS, which is a
    SUPERSET of this increment's added lines and therefore stricter than the plan
    requires. It is a parse and not a text search because the matmul operator
    cannot be told from a decorator by searching, and this file's sibling class
    in the same module uses that operator legitimately — so a file-wide search
    would report a hit that belongs to another increment's section. The literal
    added-lines screen is run separately, by the acceptance driver, over the git
    diff.

    THE DISPATCH SCREEN IS THE SAME FRAME TRACE conjunct 4 uses, so the two
    readings come from one run and cannot disagree about what happened.
    """
    say("C5_CERTIFYING_COMPONENT=the two members inc-glm53f-072 refused, and "
        f"the source of {OWNED_CLASS}")

    cfg = tiny_config()
    module = build_attention(cfg, seed=60390)
    module.prepare_projection_weights()
    heads = int(cfg.num_attention_heads)
    hidden = torch.zeros(DECLARED_SEQ, int(cfg.hidden_size), dtype=torch.float32)
    attn_out = torch.zeros(
        DECLARED_SEQ, heads, int(cfg.v_head_dim), dtype=torch.float32
    )

    counts = trace_projection(module, hidden, attn_out)
    say(f"C5_PROFILE_EVENTS={counts['events']}")
    say(f"C5_VENDOR_DISPATCH_FRAMES={len(counts['vendor'])}")
    for entry in sorted(set(counts["vendor"]))[:10]:
        say(f"C5_VENDOR_FRAME: {entry}")
    chain_total = sum(counts["chain"].values())
    say(f"C5_POSITIVE_CONTROL_F1_CHAIN_FRAMES={chain_total}")
    assert chain_total > 0, (
        "the trace observed no frame anywhere on the F1 chain, so its zero on "
        "the refused members would be vacuous; the instrument is broken"
    )
    assert not counts["vendor"], (
        f"a refused member was DISPATCHED from this call site: "
        f"{sorted(set(counts['vendor']))[:5]}"
    )

    node, _source = owned_class_node()
    say(f"C5_OWNED_CLASS_SPAN={node.lineno}-{node.end_lineno}")
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
    for entry in hits:
        say(f"C5_TORCH_MATMUL_OCCURRENCE {entry}")
    say(f"C5_TORCH_MATMUL_FORMS_IN_OWNED_CLASS={len(hits)}")
    assert not hits, (
        f"the call site carries a torch matmul form, which is the fallback this "
        f"conjunct exists to catch: {hits}"
    )

    say(f"C5_BOTH_ZEROS dispatch={len(counts['vendor'])} source={len(hits)}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
