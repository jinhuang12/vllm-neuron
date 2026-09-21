# SPDX-License-Identifier: Apache-2.0
"""The KDA layer and its two state carriers.

Prefill and decode are compared against a torch reference, the recurrent and
convolution state geometry is read off the layer, and the gate's bound is checked.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# The values this file's cases are built from.
# ---------------------------------------------------------------------------

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "config.json"

#: The tensor-parallel degree this fork serves. At this degree each rank holds one
#: KDA head, which is why the block's single-head case is the per-rank geometry and
#: not a contrivance.
TP_WORLD_SIZE = 64

#: The block's stack depth and case shape.
DECLARED_STACK_LAYERS = 3
DECLARED_DECODE_STEPS = 2

#: KDA geometry from ``linear_attn_config``
#: (``vllm_neuron/model/glm5_next/config.py``).
KDA_NUM_HEADS = 64
KDA_HEAD_SIZE = 128
KDA_CONV_KERNEL_SIZE = 4
DECLARED_GATE_LOWER_BOUND = -5.0

#: Per-rank head count at that degree: the block's ``H = 1``.
DECLARED_PER_RANK_HEADS = 1

#: The chunk width the layer resolves for itself, and the reason it is 8: the
#: intra-chunk seam needs a power of two, and both chunked seams refuse a
#: chunk-local cumulative gate above 60, which a gate bounded by -5 reaches at
#: 12 tokens. 8 is the largest power of two below that.
DECLARED_CHUNK = 8

#: 2 whole chunks and a one-token remainder. Every seam is entered once per
#: layer at this length: the chunked seams take both whole chunks in one
#: dispatch each because their inputs carry the chunk axis, and the single
#: remaining token takes the single-token decode seam.
DECLARED_PREFILL_TOKENS = 2 * DECLARED_CHUNK + 1

#: The per-arm dispatch counts the block declares.
DECLARED_PREFILL_DISPATCHES = 3
DECLARED_DECODE_DISPATCHES = 6
DECLARED_CHUNKED_DISPATCHES_ON_DECODE = 0
DECLARED_FALLBACKS = 0

#: The stack census ``test_kv_spec.py`` pins.
DECLARED_TOTAL_ENTRIES = 45
DECLARED_KDA_ENTRIES = 34
DECLARED_MLA_ENTRIES = 11

#: The block's declared tolerance, and the doc's ~0.005 four-layer scope.
DECLARED_RTOL = 5e-3
DECLARED_ATOL = 1e-5

#: ``test_get_kv_cache_spec_hybrid.py``.
HYBRID_BLOCK_SIZE = 128

#: The four field names, in ``LayerSpec``'s declared order
#: (``vllm_neuron/model/kv_cache.py``). That declaration chose them; they
#: are read from there and never re-spelled.
KDA_STATE_FIELDS = (
    "kda_conv_state_shape",
    "kda_recurrent_state_shape",
    "kda_conv_state_dtype",
    "kda_recurrent_state_dtype",
)

SEED = 20260903


# ---------------------------------------------------------------------------
# Import indirection and the fixture.
# ---------------------------------------------------------------------------
def _impl():
    """Import the implementation module inside a test body, never at import."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _runner_module():
    from vllm_neuron.vllm.worker import neuron_model_runner

    return neuron_model_runner


def _raw() -> dict:
    with open(FIXTURE_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# The seam counters. Five objects, three spellings, imported module-qualified
# because two of them are spelled identically in different modules.
# ---------------------------------------------------------------------------
def _seams():
    from vllm_neuron.functional.kda import chunked_recurrence, decode_state
    from vllm_neuron.functional.kda import depthwise_conv1d, gate_clamp

    return chunked_recurrence, decode_state, depthwise_conv1d, gate_clamp


def _reset_counters() -> None:
    chunked, decode, conv, gate = _seams()
    conv.reset_dispatch_counters()
    gate.reset_gate_clamp_dispatch_counters()
    # These two are separate objects with identical spellings; resetting one
    # does not reset the other.
    chunked.reset_dispatch_counters()
    chunked.reset_inter_dispatch_counters()
    decode.reset_decode_dispatch_counters()


def _read_counters() -> dict[str, tuple[int, int]]:
    chunked, decode, conv, gate = _seams()
    return {
        "conv": conv.dispatch_counters(),
        "gate": gate.gate_clamp_dispatch_counters(),
        "intra": chunked.dispatch_counters(),
        "inter": chunked.inter_dispatch_counters(),
        "decode": decode.decode_dispatch_counters(),
    }


# ---------------------------------------------------------------------------
# The reference. Independent of the layer on every arithmetic choice.
# ---------------------------------------------------------------------------
def _rms(x: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    """``x / sqrt(mean(x**2) + eps) * gain``, the epsilon inside the root."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * gain


def _causal_conv(
    x: torch.Tensor, weight: torch.Tensor, history: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal convolution by an explicit tap sum. """
    tokens, channels = x.shape
    kernel = int(weight.shape[-1])
    padded = torch.cat([history, x], dim=0)
    out = torch.zeros(tokens, channels, dtype=torch.float32)
    for tap in range(kernel):
        out = out + padded[tap : tap + tokens] * weight[:, 0, tap].reshape(1, channels)
    return out, padded[padded.shape[0] - (kernel - 1) :]


def _reference_layer(
    x: torch.Tensor,
    weights: dict[str, torch.Tensor],
    history: torch.Tensor,
    state: list[torch.Tensor] | None,
    *,
    heads: int,
    head_dim: int,
    eps: float,
    conv_carrier_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """One KDA layer, one token at a time, with both carriers passed through. """
    from vllm_neuron.functional.kda.chunked_recurrence import L2_NORM_EPS

    tokens = int(x.shape[0])
    width = heads * head_dim
    x32 = x.to(torch.float32)
    normed = _rms(x32, weights["input_layernorm_weight"], eps)

    q = normed @ weights["q_proj_weight"].t()
    k = normed @ weights["k_proj_weight"].t()
    v = normed @ weights["v_proj_weight"].t()
    raw_beta = normed @ weights["b_proj_weight"].t()
    raw_gate = (normed @ weights["f_a_proj_weight"].t()) @ weights[
        "f_b_proj_weight"
    ].t()
    out_gate = (normed @ weights["g_a_proj_weight"].t()) @ weights[
        "g_b_proj_weight"
    ].t()

    conv_weight = torch.cat(
        (
            weights["q_conv1d_weight"],
            weights["k_conv1d_weight"],
            weights["v_conv1d_weight"],
        ),
        dim=0,
    )
    conv, new_history = _causal_conv(torch.cat((q, k, v), dim=-1), conv_weight, history)
    # The carrier the design declares, applied where the design applies it.
    new_history = new_history.to(conv_carrier_dtype).to(torch.float32)
    conv = torch.nn.functional.silu(conv)
    q, k, v = conv.split(width, dim=-1)

    beta = torch.sigmoid(raw_beta)
    core = torch.empty(tokens, width, dtype=torch.float32)
    new_state: list[torch.Tensor] = []
    a_log = weights["A_log"].reshape(-1)
    dt_bias = weights["dt_bias"].reshape(-1)
    for head in range(heads):
        span = slice(head * head_dim, (head + 1) * head_dim)
        # The gate: lower * sigmoid(exp(A_log) * (g + bias)).
        gate = DECLARED_GATE_LOWER_BOUND * torch.sigmoid(
            torch.exp(a_log[head]) * (raw_gate[:, span] + dt_bias[span].reshape(1, -1))
        )
        q_h, k_h, v_h = q[:, span], k[:, span], v[:, span]
        q_n = q_h / torch.sqrt((q_h * q_h).sum(-1, keepdim=True) + L2_NORM_EPS)
        k_n = k_h / torch.sqrt((k_h * k_h).sum(-1, keepdim=True) + L2_NORM_EPS)
        q_n = q_n * (float(head_dim) ** -0.5)

        carried = (
            torch.zeros(head_dim, head_dim, dtype=torch.float32)
            if state is None
            else state[head].to(torch.float32)
        )
        out = torch.empty(tokens, head_dim, dtype=torch.float32)
        for t in range(tokens):
            carried = carried * torch.exp(gate[t]).unsqueeze(0)
            delta = (v_h[t] - carried @ k_n[t]) * beta[t, head]
            carried = carried + torch.outer(delta, k_n[t])
            out[t] = carried @ q_n[t]
        core[:, span] = out
        new_state.append(carried)

    shaped = core.reshape(tokens, heads, head_dim)
    shaped = _rms(shaped, weights["o_norm_weight"].reshape(1, 1, head_dim), eps)
    # The reference builds this half as a gated RMSNorm whose activation is
    # sigmoid (``kimi_gdn_linear_attn.py``), not silu.
    shaped = shaped * torch.sigmoid(out_gate.reshape(tokens, heads, head_dim))
    attn = shaped.reshape(tokens, width) @ weights["o_proj_weight"].t()
    return x32 + attn, new_history, new_state


# ---------------------------------------------------------------------------
# The one tiny case, built once and read by both checks below.
# ---------------------------------------------------------------------------
def _make_weights(
    hidden: int, heads: int, head_dim: int, kernel: int
) -> dict[str, torch.Tensor]:
    """Random weights at unit-ish scale, so no reading sits in a saturated tail."""
    width = heads * head_dim

    def rnd(*shape: int, scale: float) -> torch.Tensor:
        return (torch.randn(*shape) * scale).to(torch.float32)

    return {
        "q_proj_weight": rnd(width, hidden, scale=hidden**-0.5),
        "k_proj_weight": rnd(width, hidden, scale=hidden**-0.5),
        "v_proj_weight": rnd(width, hidden, scale=hidden**-0.5),
        "b_proj_weight": rnd(heads, hidden, scale=hidden**-0.5),
        "f_a_proj_weight": rnd(head_dim, hidden, scale=hidden**-0.5),
        "f_b_proj_weight": rnd(width, head_dim, scale=head_dim**-0.5),
        "g_a_proj_weight": rnd(head_dim, hidden, scale=hidden**-0.5),
        "g_b_proj_weight": rnd(width, head_dim, scale=head_dim**-0.5),
        "q_conv1d_weight": rnd(width, 1, kernel, scale=0.5),
        "k_conv1d_weight": rnd(width, 1, kernel, scale=0.5),
        "v_conv1d_weight": rnd(width, 1, kernel, scale=0.5),
        "o_norm_weight": 1.0 + rnd(head_dim, scale=0.05),
        "o_proj_weight": rnd(hidden, width, scale=width**-0.5),
        "A_log": rnd(heads, scale=0.3),
        "dt_bias": rnd(width, scale=0.3),
        "input_layernorm_weight": 1.0 + rnd(hidden, scale=0.05),
    }


@pytest.fixture(scope="module")
def case() -> SimpleNamespace:
    """Drive the stack once: one prefill call, then two decode calls. """
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    text_config = Glm5NextTextConfig()
    hidden = int(text_config.hidden_size)
    linear_attn = text_config.linear_attn_config
    # The declared values are checked against their origin, so a config drift
    # reddens this file instead of being absorbed by it.
    assert linear_attn["num_heads"] == KDA_NUM_HEADS
    assert linear_attn["head_dim"] == KDA_HEAD_SIZE
    assert linear_attn["short_conv_kernel_size"] == KDA_CONV_KERNEL_SIZE
    assert linear_attn["gate_lower_bound"] == DECLARED_GATE_LOWER_BOUND

    heads = KDA_NUM_HEADS // TP_WORLD_SIZE
    assert heads == DECLARED_PER_RANK_HEADS
    head_dim = KDA_HEAD_SIZE
    eps = float(text_config.rms_norm_eps)
    total = DECLARED_PREFILL_TOKENS + DECLARED_DECODE_STEPS

    torch.manual_seed(SEED)
    weights = [
        _make_weights(hidden, heads, head_dim, KDA_CONV_KERNEL_SIZE)
        for _ in range(DECLARED_STACK_LAYERS)
    ]
    torch.manual_seed(SEED + 1)
    tokens = torch.randn(total, hidden, dtype=torch.float32)

    layers = []
    for index, layer_weights in enumerate(weights):
        layer = _impl().Glm5NextKDALayer(text_config, index, TP_WORLD_SIZE)
        attention = layer.attention
        for name, tensor in layer_weights.items():
            target = layer if name == "input_layernorm_weight" else attention
            setattr(target, name, nn.Parameter(tensor.clone(), requires_grad=False))
        layers.append(layer)

    # The bank is allocated from the four values get_kv_spec reports -- that is
    # what "the runner bank state" names -- and from nothing else.
    bank = []
    for layer in layers:
        attention = layer.attention
        bank.append(
            (
                torch.zeros(
                    attention.kda_conv_state_shape,
                    dtype=attention.kda_conv_state_dtype,
                ),
                torch.zeros(
                    attention.kda_recurrent_state_shape,
                    dtype=attention.kda_recurrent_state_dtype,
                ),
            )
        )

    def drive(hidden_states: torch.Tensor, is_prefill: bool, start_position: int) -> torch.Tensor:
        out = hidden_states
        for layer, (conv_state, recurrent_state) in zip(layers, bank):
            out = layer(
                out,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
                is_prefill=is_prefill,
                start_position=start_position,
                chunk_size=DECLARED_CHUNK,
            )
        return out

    _reset_counters()
    prefill_out = drive(tokens[:DECLARED_PREFILL_TOKENS], True, 0)
    prefill_counts = _read_counters()
    prefill_state = [rs.clone() for _, rs in bank]

    _reset_counters()
    decode_rows = []
    for step in range(DECLARED_DECODE_STEPS):
        index = DECLARED_PREFILL_TOKENS + step
        decode_rows.append(drive(tokens[index : index + 1], False, index))
    decode_counts = _read_counters()
    decode_out = torch.cat(decode_rows, dim=0)

    # The reference, driven in the same three calls with its own carriers.
    conv_carrier_dtype = layers[0].attention.kda_conv_state_dtype
    kernel_rows = KDA_CONV_KERNEL_SIZE - 1
    histories = [
        torch.zeros(kernel_rows, 3 * heads * head_dim, dtype=torch.float32)
        for _ in range(DECLARED_STACK_LAYERS)
    ]
    states: list[list[torch.Tensor] | None] = [None] * DECLARED_STACK_LAYERS

    def reference(hidden_states: torch.Tensor) -> torch.Tensor:
        out = hidden_states
        for index, layer_weights in enumerate(weights):
            out, histories[index], states[index] = _reference_layer(
                out,
                layer_weights,
                histories[index],
                states[index],
                heads=heads,
                head_dim=head_dim,
                eps=eps,
                conv_carrier_dtype=conv_carrier_dtype,
            )
        return out

    reference_prefill = reference(tokens[:DECLARED_PREFILL_TOKENS])
    reference_decode = torch.cat(
        [
            reference(tokens[DECLARED_PREFILL_TOKENS + step :][:1])
            for step in range(DECLARED_DECODE_STEPS)
        ],
        dim=0,
    )

    return SimpleNamespace(
        layers=layers,
        bank=bank,
        heads=heads,
        head_dim=head_dim,
        prefill_out=prefill_out,
        prefill_counts=prefill_counts,
        prefill_state=prefill_state,
        decode_out=decode_out,
        decode_counts=decode_counts,
        reference_prefill=reference_prefill,
        reference_decode=reference_decode,
    )


# ---------------------------------------------------------------------------
# The prefill arm.# ---------------------------------------------------------------------------
def test_prefill_matches_the_reference_at_three_per_seam(
    case: SimpleNamespace,
) -> None:
    """A three-layer stack over 17 tokens, and 3 dispatches through each seam. """
    assert tuple(case.prefill_out.shape) == (
        DECLARED_PREFILL_TOKENS,
        case.layers[0].attention.o_proj_weight.shape[0],
    )
    torch.testing.assert_close(
        case.prefill_out,
        case.reference_prefill,
        rtol=DECLARED_RTOL,
        atol=DECLARED_ATOL,
    )

    counts = case.prefill_counts
    for seam in ("conv", "gate", "intra", "inter", "decode"):
        dispatches, fallbacks = counts[seam]
        assert dispatches == DECLARED_PREFILL_DISPATCHES, (
            f"{seam} read {dispatches} dispatches, expected "
            f"{DECLARED_PREFILL_DISPATCHES} -- once per layer of a "
            f"{DECLARED_STACK_LAYERS}-layer stack"
        )
        assert fallbacks == DECLARED_FALLBACKS, (
            f"{seam} took the torch fallback {fallbacks} times; a fallback for "
            f"a torch fallback is a route failure, so this counter reads 0 or the "
            f"reading above was not taken through NKI at all"
        )


# ---------------------------------------------------------------------------
# The decode arm.# ---------------------------------------------------------------------------
def test_decode_carries_state_and_never_enters_a_chunked_seam(
    case: SimpleNamespace,
) -> None:
    """Two steps, six dispatches through three seams, zero through two. """
    counts = case.decode_counts
    for seam in ("conv", "gate", "decode"):
        dispatches, fallbacks = counts[seam]
        assert dispatches == DECLARED_DECODE_DISPATCHES, (
            f"{seam} read {dispatches}, expected {DECLARED_DECODE_DISPATCHES} -- "
            f"{DECLARED_STACK_LAYERS} per step over {DECLARED_DECODE_STEPS} steps"
        )
        assert fallbacks == DECLARED_FALLBACKS
    for seam in ("intra", "inter"):
        dispatches, fallbacks = counts[seam]
        assert dispatches == DECLARED_CHUNKED_DISPATCHES_ON_DECODE, (
            f"{seam} read {dispatches} dispatches on a decode arm; a single-token "
            f"decode step must never enter a chunked seam"
        )
        assert fallbacks == DECLARED_FALLBACKS

    # The state persisted, and it is the bank that holds it. Element count and
    # dtype are asserted exactly, against the values get_kv_spec reports.
    for index, (conv_state, recurrent_state) in enumerate(case.bank):
        attention = case.layers[index].attention
        assert tuple(recurrent_state.shape) == tuple(
            attention.kda_recurrent_state_shape
        )
        assert recurrent_state.numel() == case.heads * case.head_dim * case.head_dim
        assert recurrent_state.dtype is attention.kda_recurrent_state_dtype
        assert tuple(conv_state.shape) == tuple(attention.kda_conv_state_shape)
        assert conv_state.dtype is attention.kda_conv_state_dtype
        # It advanced. A bank that was written once and then ignored would leave
        # the prefill's state in place, and a bank never written at all would
        # leave zeros.
        assert not torch.equal(recurrent_state, case.prefill_state[index])
        assert float(recurrent_state.abs().max()) > 0.0

    torch.testing.assert_close(
        case.decode_out,
        case.reference_decode,
        rtol=DECLARED_RTOL,
        atol=DECLARED_ATOL,
    )


_BLOCK_SIZE_GRANULARITY = 64


def _attention_spec_class(layer):
    """The spec class the runner builds for one attention layer. """
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MLAAttentionSpec,
        SlidingWindowSpec,
    )

    if getattr(layer, "latent_kv", False):
        return MLAAttentionSpec
    if layer.sliding_window_size is not None:
        return SlidingWindowSpec
    return FullAttentionSpec


def _runner_block_size(resolved, cache_dtype) -> SimpleNamespace:
    """The block size to drive the runner at, derived from this process's degree. """
    from vllm.v1.kv_cache_interface import MambaSpec

    # A partially-set layer belongs to neither probe set, and that is what keeps
    # The control clears one of the four fields on a deep copy and requires the
    # runner's pairing guard to refuse it; a layer that is neither wholly set nor
    # wholly unset must therefore reach that guard untouched, and must not be fed
    # to a probe here that would raise something else first.
    kda_layers = [
        layer
        for layer in resolved.layers
        if all(getattr(layer, field) is not None for field in KDA_STATE_FIELDS)
    ]
    attention_layers = [
        layer
        for layer in resolved.layers
        if all(getattr(layer, field) is None for field in KDA_STATE_FIELDS)
    ]

    state_page = max(
        (
            MambaSpec(
                block_size=1,
                shapes=(
                    tuple(layer.kda_conv_state_shape),
                    tuple(layer.kda_recurrent_state_shape),
                ),
                dtypes=(
                    layer.kda_conv_state_dtype,
                    layer.kda_recurrent_state_dtype,
                ),
            ).page_size_bytes
            for layer in kda_layers
        ),
        default=0,
    )
    # At ``block_size=1`` the attention page is the per-token slope, because
    # ``AttentionSpec.real_page_size_bytes`` is linear in the block size -- in
    # every one of its three quantisation limbs, so this holds without depending
    # on which limb the vendor takes.
    #
    # ``cache_dtype`` is a parameter and not typed here, because the runner builds
    # its own attention specs from the dtype the fake config below carries. If the
    # two ever disagreed, this derivation would compute a slope the runner does not
    # use; the caller's assertion that the runner's own attention specs report
    # exactly ``attention_page`` is what closes that loop against the world.
    bytes_per_token = max(
        (
            _attention_spec_class(layer)(
                block_size=1,
                num_kv_heads=layer.num_kv_heads,
                head_size=layer.head_size,
                dtype=cache_dtype,
            ).page_size_bytes
            for layer in attention_layers
        ),
        default=0,
    )

    if state_page and bytes_per_token:
        smallest_strictly_larger = state_page // bytes_per_token + 1
        rounded = (
            -(-smallest_strictly_larger // _BLOCK_SIZE_GRANULARITY)
            * _BLOCK_SIZE_GRANULARITY
        )
        block_size = max(rounded, HYBRID_BLOCK_SIZE)
    else:
        block_size = HYBRID_BLOCK_SIZE

    return SimpleNamespace(
        world_size=_impl()._resolve_world_size(),
        block_size=block_size,
        state_page=state_page,
        bytes_per_token=bytes_per_token,
        attention_page=bytes_per_token * block_size,
    )


def _drive_runner_translation(model) -> tuple[dict, SimpleNamespace]:
    """Drive the runner's unbound ``get_kv_cache_spec`` over a real model. """
    # One definition, read by the derivation and by the config the runner reads its
    # own dtype from, so the two cannot drift apart in a later edit. The value is
    # the one this harness already handed the runner before this repair.
    cache_dtype = torch.bfloat16
    geometry = _runner_block_size(model.get_kv_spec(), cache_dtype)
    fake_self = SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=geometry.block_size, cache_dtype="auto"
            ),
            model_config=SimpleNamespace(dtype=cache_dtype),
        ),
        speculative_config=None,
        model=model,
    )
    specs = _runner_module().NeuronModelRunner.get_kv_cache_spec(fake_self)
    return specs, geometry


@pytest.fixture(scope="module")
def real_spec():
    """The real model at this process's resolved degree, and its own ``get_kv_spec``."""
    model = _impl().Glm5NextForConditionalGeneration.from_configs(
        copy.deepcopy(_raw()),
        text_neuron_config=None,
        vision_neuron_config=None,
    )
    return model, model.get_kv_spec()


def test_the_four_state_fields_are_reported_on_the_kda_half(
    real_spec,
) -> None:
    """34 of 34 carry all four, 11 of 11 carry none, and 34 translate to Mamba. """
    from vllm.v1.kv_cache_interface import MambaSpec

    model, spec = real_spec
    assert len(spec.layers) == DECLARED_TOTAL_ENTRIES

    carried = [
        layer
        for layer in spec.layers
        if all(getattr(layer, field) is not None for field in KDA_STATE_FIELDS)
    ]
    bare = [
        layer
        for layer in spec.layers
        if all(getattr(layer, field) is None for field in KDA_STATE_FIELDS)
    ]
    partial = [
        layer
        for layer in spec.layers
        if layer not in carried and layer not in bare
    ]
    assert len(carried) == DECLARED_KDA_ENTRIES
    assert len(bare) == DECLARED_MLA_ENTRIES
    # A partial set is what the runner's pairing guard refuses, so the guard
    # firing 0 times below is a statement about this count being 0.
    assert partial == []

    # The values are the vendor calculators' at this model's world size, and the
    # conv extent order is the resolved layout's.
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateDtypeCalculator,
        MambaStateShapeCalculator,
    )

    world_size = _impl()._resolve_world_size()
    expected_conv, expected_recurrent = MambaStateShapeCalculator.kda_state_shape(
        tp_world_size=world_size,
        num_heads=KDA_NUM_HEADS,
        head_dim=KDA_HEAD_SIZE,
        conv_kernel_size=KDA_CONV_KERNEL_SIZE,
    )
    expected_dtypes = MambaStateDtypeCalculator.kda_state_dtype(torch.bfloat16, "auto")
    for layer in carried:
        assert tuple(layer.kda_conv_state_shape) == tuple(expected_conv)
        assert tuple(layer.kda_recurrent_state_shape) == tuple(expected_recurrent)
        assert layer.kda_conv_state_dtype is expected_dtypes[0]
        assert layer.kda_recurrent_state_dtype is expected_dtypes[1]

    # ...and the runner's translation over that real output.
    specs, geometry = _drive_runner_translation(model)
    mamba_specs = [
        entry for entry in specs.values() if isinstance(entry, MambaSpec)
    ]
    attention_specs = [
        entry for entry in specs.values() if not isinstance(entry, MambaSpec)
    ]
    mamba = len(mamba_specs)
    other = len(attention_specs)
    # The degree and the block size are recorded beside the census, not asserted.
    # The requirement is exactly that of a resolved world size --
    # evidence, never a criterion -- and the reason is that this process's degree is
    # a property of how the test is run, so pinning it would pin the runner rather
    # than the code. What is asserted is the page relation it implies.
    assert len(specs) == DECLARED_TOTAL_ENTRIES
    assert (mamba, other) == (DECLARED_KDA_ENTRIES, DECLARED_MLA_ENTRIES)

    # The pad branch is the one production takes, and this is where that is read.
    # The arm must not drive the runner at a block size whose
    # attention page was smaller than the world-size-1 state page, so
    # the page-size refusal fired and the arm went red. The block size is now
    # derived from this process's own degree, so the ordering is the production one
    # and these three readings say so.
    padded = sorted({entry.page_size_padded for entry in mamba_specs})
    # 1. Every recurrent-state page was raised to the attention page.
    assert padded == [geometry.attention_page]
    # 2. No attention page was padded -- the direction the KV-cache spec's own test
    #    requires too, and the one the allocation arm depends on, because it sizes
    #    its view from the real page.
    assert all(entry.page_size_padded is None for entry in attention_specs)
    # 3. The point of the whole exercise: all 45 entries now report one page. This
    #    also closes the loop on the derivation -- ``bytes_per_token`` was computed
    #    from a probe spec this test built, and this reads it back off the specs the
    #    runner built, so a probe that modelled the runner wrongly fails here by
    #    name instead of quietly choosing a block size on the wrong slope.
    reported = sorted({entry.page_size_bytes for entry in specs.values()})
    assert reported == [geometry.attention_page]


# ---------------------------------------------------------------------------
# The non-vacuity control on the counted zero above.
# ---------------------------------------------------------------------------
def test_the_pairing_guard_is_live_when_one_field_is_cleared(
    real_spec,
) -> None:
    """Clear one field on a copy and the runner refuses the layer. """
    _model, spec = real_spec
    mutated = copy.deepcopy(spec)
    victim = next(
        layer for layer in mutated.layers if layer.kda_conv_state_dtype is not None
    )
    victim.kda_conv_state_dtype = None

    with pytest.raises(ValueError, match="part of it unset"):
        _drive_runner_translation(SimpleNamespace(get_kv_spec=lambda: mutated))


def _gate_tile_bound() -> int:
    """The most tokens the gate seam serves in one call, read from the seam. """
    from vllm_neuron.functional.kda import chunked_recurrence, gate_clamp

    assert gate_clamp.MAX_TILE is chunked_recurrence.MAX_TILE, (
        "the gate seam's tile bound is no longer the same object as the chunked "
        "module's; one of them has been redeclared and this case can no longer "
        "trust it to be the wall"
    )
    return int(gate_clamp.MAX_TILE)


@pytest.fixture(scope="module")
def long_case() -> SimpleNamespace:
    """Drive the stack over a prompt one token longer than the gate serves."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    bound = _gate_tile_bound()
    total = bound + 1

    text_config = Glm5NextTextConfig()
    hidden = int(text_config.hidden_size)
    heads = KDA_NUM_HEADS // TP_WORLD_SIZE
    head_dim = KDA_HEAD_SIZE
    eps = float(text_config.rms_norm_eps)

    torch.manual_seed(SEED)
    weights = [
        _make_weights(hidden, heads, head_dim, KDA_CONV_KERNEL_SIZE)
        for _ in range(DECLARED_STACK_LAYERS)
    ]
    torch.manual_seed(SEED + 1)
    tokens = torch.randn(total, hidden, dtype=torch.float32)

    layers = []
    for index, layer_weights in enumerate(weights):
        layer = _impl().Glm5NextKDALayer(text_config, index, TP_WORLD_SIZE)
        attention = layer.attention
        for name, tensor in layer_weights.items():
            target = layer if name == "input_layernorm_weight" else attention
            setattr(target, name, nn.Parameter(tensor.clone(), requires_grad=False))
        layers.append(layer)

    bank = [
        (
            torch.zeros(
                layer.attention.kda_conv_state_shape,
                dtype=layer.attention.kda_conv_state_dtype,
            ),
            torch.zeros(
                layer.attention.kda_recurrent_state_shape,
                dtype=layer.attention.kda_recurrent_state_dtype,
            ),
        )
        for layer in layers
    ]

    _reset_counters()
    out = tokens
    for layer, (conv_state, recurrent_state) in zip(layers, bank):
        out = layer(
            out,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            is_prefill=True,
            chunk_size=DECLARED_CHUNK,
        )
    counts = _read_counters()

    conv_carrier_dtype = layers[0].attention.kda_conv_state_dtype
    kernel_rows = KDA_CONV_KERNEL_SIZE - 1
    history = [
        torch.zeros(kernel_rows, 3 * heads * head_dim, dtype=torch.float32)
        for _ in range(DECLARED_STACK_LAYERS)
    ]
    state: list[list[torch.Tensor] | None] = [None] * DECLARED_STACK_LAYERS
    want = tokens
    for index, layer_weights in enumerate(weights):
        want, history[index], state[index] = _reference_layer(
            want,
            layer_weights,
            history[index],
            state[index],
            heads=heads,
            head_dim=head_dim,
            eps=eps,
            conv_carrier_dtype=conv_carrier_dtype,
        )

    return SimpleNamespace(
        bound=bound,
        total=total,
        layers=layers,
        heads=heads,
        out=out,
        counts=counts,
        reference=want,
    )


def test_a_prompt_past_the_gate_bound_is_served(
    long_case: SimpleNamespace,
) -> None:
    """A prompt one token past the gate seam's bound returns, and is right. """
    assert long_case.total == long_case.bound + 1
    assert tuple(long_case.out.shape) == (
        long_case.total,
        long_case.layers[0].attention.o_proj_weight.shape[0],
    )
    assert bool(torch.isfinite(long_case.out).all()), (
        "the long prompt returned a non-finite value, so it was served in name "
        "only"
    )
    torch.testing.assert_close(
        long_case.out,
        long_case.reference,
        rtol=DECLARED_RTOL,
        atol=DECLARED_ATOL,
    )

    # The gate is entered once per tile, and the tile count is derived from the
    # bound rather than declared, so this reading follows the seam. The other four
    # seams are untouched by this repair and stay at one entry per layer.
    tiles = -(-long_case.total // long_case.bound)
    per_layer = DECLARED_STACK_LAYERS * DECLARED_PER_RANK_HEADS
    expected = {
        "conv": per_layer,
        "gate": tiles * per_layer,
        "intra": per_layer,
        "inter": per_layer,
        "decode": per_layer,
    }
    for seam, want_dispatches in expected.items():
        dispatches, fallbacks = long_case.counts[seam]
        assert dispatches == want_dispatches, (
            f"{seam} read {dispatches} dispatches, expected {want_dispatches} at "
            f"{long_case.total} tokens"
        )
        assert fallbacks == DECLARED_FALLBACKS, (
            f"{seam} took the torch fallback {fallbacks} times; tiling must reach "
            f"the kernel on every tile, and a torch fallback here is a "
            f"route failure"
        )


def test_the_gate_seam_still_refuses_an_over_long_call(
    long_case: SimpleNamespace,
) -> None:
    """The seam keeps its refusal; only the caller learned to tile. """
    from vllm_neuron.functional.kda.gate_clamp import GateClampError, kda_gate_clamp

    bound = long_case.bound
    head_dim = KDA_HEAD_SIZE
    torch.manual_seed(SEED + 2)
    decay = torch.tensor(0.3, dtype=torch.float32)
    bias = torch.randn(head_dim, dtype=torch.float32) * 0.3

    # Exactly at the bound: served. This is the population check that makes the
    # refusal below a reading about the extent and not about the whole call.
    at_bound = kda_gate_clamp(
        torch.randn(bound, head_dim, dtype=torch.float32),
        decay,
        bias=bias,
        lower=DECLARED_GATE_LOWER_BOUND,
    )
    assert tuple(at_bound.shape) == (bound, head_dim)

    with pytest.raises(GateClampError, match="must both be in"):
        kda_gate_clamp(
            torch.randn(bound + 1, head_dim, dtype=torch.float32),
            decay,
            bias=bias,
            lower=DECLARED_GATE_LOWER_BOUND,
        )

    # And the tiling is exact, not merely inside a tolerance: the gate carries no
    # reduction along the token axis, so a tile boundary cannot move a value.
    whole_input = torch.randn(bound, head_dim, dtype=torch.float32)
    half = bound // 2
    whole = kda_gate_clamp(
        whole_input, decay, bias=bias, lower=DECLARED_GATE_LOWER_BOUND
    )
    tiled = torch.cat(
        [
            kda_gate_clamp(
                whole_input[:half].contiguous(),
                decay,
                bias=bias,
                lower=DECLARED_GATE_LOWER_BOUND,
            ),
            kda_gate_clamp(
                whole_input[half:].contiguous(),
                decay,
                bias=bias,
                lower=DECLARED_GATE_LOWER_BOUND,
            ),
        ],
        dim=0,
    )
    assert torch.equal(whole, tiled), (
        "tiling the gate changed a value, so the token axis carries a reduction "
        "and reassembly in the caller is not sound"
    )
