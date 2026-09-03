# SPDX-License-Identifier: Apache-2.0
"""``inc-glm53f-038a`` acceptance -- WP3: the KDA layer and its state carriers.

THE DECLARED ACCEPTANCE, the block's Tier N harness as ``inc-glm53f-025``:

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
      NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python -m pytest \\
      test/vllm_neuron/model/glm5_next/test_kda_layer.py -q -s --timeout 60 \\
      -p no:cacheprovider

Four arms over ONE tiny case, one test item each, no ``parametrize``:

* A01 -- the prefill arm. A three-layer KDA stack over 17 tokens matches an
  independent torch reference, and each of the five landed seams is entered
  three times, once per layer.
* A02 -- the decode arm. Two single-token steps carry the recurrence through the
  bank; the convolution, the gate and the decode seam read six each while BOTH
  chunked seams read exactly zero; and the two steps' outputs are value-compared
  against the same reference (round 48, Q7).
* A03 -- the four recurrent-state field values, read off the REAL
  ``get_kv_spec`` and driven through the runner's landed translation.
* A04 -- the non-vacuity control (D1.5) on A03's zero: one field cleared on a
  COPY makes the runner's pairing guard raise.

WHY THE REFERENCE CARRIES STATE INSTEAD OF WALKING ONE FLAT SEQUENCE
-------------------------------------------------------------------
The reference below runs the stack in the same three calls the layer does -- one
prefill, then two decode steps -- and carries its own conv history and recurrent
state between them, storing the conv history through ``bfloat16`` exactly where
the bank stores it. That is not a convenience. ``get_kv_spec`` reports the conv
carrier's dtype as ``bfloat16`` (vLLM's own ``kda_state_dtype`` returns the model
dtype for the conv carrier and ``float32`` for the recurrent one), so a decode
step reads a conv history that has been through ``bfloat16``, and a reference
that keeps everything in ``float32`` differs from correct code by that
quantisation alone.

That was measured rather than argued, three readings over one case
(``probe-038a-dtype.out``):

* the bank as declared, against a flat ``float32`` reference: prefill
  ``1.311e-06``, decode ``5.454e-03``;
* the SAME comparison with the conv carrier forced to ``float32``: prefill
  ``1.311e-06``, decode ``1.550e-06`` -- so the ``bfloat16`` conv carrier is the
  whole of the difference and nothing else in the layer contributes to it;
* the bank as declared, against the state-carrying reference used here: prefill
  ``1.669e-06``, decode ``9.537e-07``.

So the tolerance is the block's declared one, unchanged, and the reference is
the one that models the carriers the design declares.

The reference is independent of the layer under test on every step that has an
arithmetic choice in it: it forms its own convolution by an explicit tap sum, its
own gate in ``inc-glm53f-084``'s landed bounded-sigmoid form, and its own
recurrence one token at a time. It never groups tokens into chunks, so agreement
on the prefill arm is a statement that the layer's chunk-plus-remainder split is
associativity-correct.

CONVENTIONS THIS FILE FOLLOWS, ALL FOUR OF THEM ``inc-glm53f-013``'s
-------------------------------------------------------------------
``model_fp8`` is never imported at module level -- ``test_factory.py:280``
is a landed assertion that it stays out of ``sys.modules`` -- so every import of
it goes through :func:`_impl` inside a test body; the model is built through
``from_configs``; the fixture is deep-copied because ``__post_init__`` mutates
``layer_types``; and the fixture is digest-pinned so a silent edit cannot move a
declared value.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Declared values, and the pins that keep them honest.
# ---------------------------------------------------------------------------

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "config.json"

#: Same digest ``test_kv_spec.py:114`` pins.
FIXTURE_SHA256 = "5ed24d23a3e14a038352e1bdc21fd25fc90ff2291d3f6a310acf5d4036665a1d"

#: The registered tensor-parallel degree, as ``test_kv_cache_spec.py:94`` records
#: it. At this degree each rank holds ONE KDA head, which is why the block's
#: single-head case is the registered per-rank geometry and not a contrivance.
REGISTERED_TP_WORLD_SIZE = 64

#: The block's stack depth and case shape.
DECLARED_STACK_LAYERS = 3
DECLARED_DECODE_STEPS = 2

#: KDA geometry from ``linear_attn_config``
#: (``vllm_neuron/model/glm5_next/config.py:152-159``).
DECLARED_KDA_NUM_HEADS = 64
DECLARED_KDA_HEAD_SIZE = 128
DECLARED_KDA_CONV_KERNEL_SIZE = 4
DECLARED_GATE_LOWER_BOUND = -5.0

#: Per-rank head count at the registered degree: the block's ``H = 1``.
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

#: The stack census ``test_kv_spec.py:117-121`` pins.
DECLARED_TOTAL_ENTRIES = 45
DECLARED_KDA_ENTRIES = 34
DECLARED_MLA_ENTRIES = 11

#: The block's declared tolerance, and the doc's ~0.005 four-layer scope.
DECLARED_RTOL = 5e-3
DECLARED_ATOL = 1e-5

#: ``test_get_kv_cache_spec_hybrid.py:77``.
REGISTERED_HYBRID_BLOCK_SIZE = 128

#: The four field names, in ``LayerSpec``'s declared order
#: (``vllm_neuron/model/kv_cache.py:41-44``). ``inc-glm53f-015`` chose them; they
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
    """Import the implementation module INSIDE a test body, never at import."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _runner_module():
    from vllm_neuron.vllm.worker import neuron_model_runner

    return neuron_model_runner


def _raw() -> dict:
    digest = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    assert digest == FIXTURE_SHA256, (
        f"fixture digest moved: {digest} != {FIXTURE_SHA256}; a declared value "
        f"in this file may no longer describe the fixture"
    )
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
    """Depthwise causal convolution by an explicit tap sum.

    ``out[t] = sum_s padded[t + s] * weight[:, 0, s]``. Written out rather than
    taken from the seam's own torch reference, so that a shared error in the tap
    order could not hide. The tap order itself was checked against that
    reference once, at ``9.537e-07`` (``probe-038a-layer.out``).

    Returns the output and the new history: the last ``kernel - 1`` padded rows.
    """
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
    """One KDA layer, one token at a time, with both carriers passed through.

    The recurrence below is the sequential delta rule the landed seams are
    measured against (``chunked_recurrence.py:1308-1353``): the state decays PER
    KEY CHANNEL first, the delta then reads the decayed state, the
    L2-normalisation epsilon sits inside the root, and ``q`` carries
    ``K ** -0.5``. It groups no tokens.
    """
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
        # inc-glm53f-084's landed gate: lower * sigmoid(exp(A_log) * (g + bias)).
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
    # sigmoid (``kimi_gdn_linear_attn.py:219``), not silu.
    shaped = shaped * torch.sigmoid(out_gate.reshape(tokens, heads, head_dim))
    attn = shaped.reshape(tokens, width) @ weights["o_proj_weight"].t()
    return x32 + attn, new_history, new_state


# ---------------------------------------------------------------------------
# The one tiny case, built once and read by A01 and A02.
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
    """Drive the stack once: one prefill call, then two decode calls.

    Module-scoped because the decode arm reads what the prefill arm left in the
    bank -- that carry IS what A02 measures, so the two arms are two readings
    over one world run and not two runs.
    """
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    text_config = Glm5NextTextConfig()
    hidden = int(text_config.hidden_size)
    linear_attn = text_config.linear_attn_config
    # The declared values are checked against their origin, so a config drift
    # reddens this file instead of being absorbed by it.
    assert linear_attn["num_heads"] == DECLARED_KDA_NUM_HEADS
    assert linear_attn["head_dim"] == DECLARED_KDA_HEAD_SIZE
    assert linear_attn["short_conv_kernel_size"] == DECLARED_KDA_CONV_KERNEL_SIZE
    assert linear_attn["gate_lower_bound"] == DECLARED_GATE_LOWER_BOUND

    heads = DECLARED_KDA_NUM_HEADS // REGISTERED_TP_WORLD_SIZE
    assert heads == DECLARED_PER_RANK_HEADS
    head_dim = DECLARED_KDA_HEAD_SIZE
    eps = float(text_config.rms_norm_eps)
    total = DECLARED_PREFILL_TOKENS + DECLARED_DECODE_STEPS

    torch.manual_seed(SEED)
    weights = [
        _make_weights(hidden, heads, head_dim, DECLARED_KDA_CONV_KERNEL_SIZE)
        for _ in range(DECLARED_STACK_LAYERS)
    ]
    torch.manual_seed(SEED + 1)
    tokens = torch.randn(total, hidden, dtype=torch.float32)

    layers = []
    for index, layer_weights in enumerate(weights):
        layer = _impl().Glm5NextKDALayer(text_config, index, REGISTERED_TP_WORLD_SIZE)
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

    def drive(hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        out = hidden_states
        for layer, (conv_state, recurrent_state) in zip(layers, bank):
            out = layer(
                out,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
                is_prefill=is_prefill,
                chunk_size=DECLARED_CHUNK,
            )
        return out

    _reset_counters()
    prefill_out = drive(tokens[:DECLARED_PREFILL_TOKENS], True)
    prefill_counts = _read_counters()
    prefill_state = [rs.clone() for _, rs in bank]

    _reset_counters()
    decode_rows = []
    for step in range(DECLARED_DECODE_STEPS):
        index = DECLARED_PREFILL_TOKENS + step
        decode_rows.append(drive(tokens[index : index + 1], False))
    decode_counts = _read_counters()
    decode_out = torch.cat(decode_rows, dim=0)

    # The reference, driven in the same three calls with its own carriers.
    conv_carrier_dtype = layers[0].attention.kda_conv_state_dtype
    kernel_rows = DECLARED_KDA_CONV_KERNEL_SIZE - 1
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
# A01 -- the prefill arm.
# ---------------------------------------------------------------------------
def test_kda_layer_a01_prefill_matches_the_reference_at_three_per_seam(
    case: SimpleNamespace,
) -> None:
    """A three-layer stack over 17 tokens, and 3 dispatches through each seam.

    The 3s are single-head counts, which at the registered tensor-parallel
    degree is the whole of a rank's work: 64 heads over 64 ranks is one head.
    """
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
    print(
        "A01_PREFILL_MAXABS="
        f"{float((case.prefill_out - case.reference_prefill).abs().max()):.3e}"
    )

    counts = case.prefill_counts
    print(f"A01_PREFILL_COUNTS={counts}")
    for seam in ("conv", "gate", "intra", "inter", "decode"):
        dispatches, fallbacks = counts[seam]
        assert dispatches == DECLARED_PREFILL_DISPATCHES, (
            f"{seam} read {dispatches} dispatches, expected "
            f"{DECLARED_PREFILL_DISPATCHES} -- once per layer of a "
            f"{DECLARED_STACK_LAYERS}-layer stack"
        )
        assert fallbacks == DECLARED_FALLBACKS, (
            f"{seam} took the torch fallback {fallbacks} times; a fallback for "
            f"kernel-class work is a P13 defect, so this counter reads 0 or the "
            f"reading above was not taken through NKI at all"
        )


# ---------------------------------------------------------------------------
# A02 -- the decode arm.
# ---------------------------------------------------------------------------
def test_kda_layer_a02_decode_carries_state_and_never_enters_a_chunked_seam(
    case: SimpleNamespace,
) -> None:
    """Two steps, six dispatches through three seams, zero through two.

    The two zeros are what this arm exists to measure. Neither chunked seam
    accepts an entering state, so a decode step driven through one would restart
    the recurrence from zero on every step (``evidence-038.md`` section 1). What
    carries the recurrence instead is the bank.
    """
    counts = case.decode_counts
    print(f"A02_DECODE_COUNTS={counts}")
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
        # It ADVANCED. A bank that was written once and then ignored would leave
        # the prefill's state in place, and a bank never written at all would
        # leave zeros.
        assert not torch.equal(recurrent_state, case.prefill_state[index])
        assert float(recurrent_state.abs().max()) > 0.0
        print(
            f"A02_BANK{index}_REC_NUMEL={recurrent_state.numel()} "
            f"REC_DTYPE={recurrent_state.dtype} "
            f"CONV_NUMEL={conv_state.numel()} CONV_DTYPE={conv_state.dtype}"
        )

    # Round 48, Q7: the decode OUTPUTS are value-compared, not only counted.
    # A state that was carried wrongly changes these values, so this is the
    # reading that makes the carry a measurement rather than a bookkeeping claim.
    torch.testing.assert_close(
        case.decode_out,
        case.reference_decode,
        rtol=DECLARED_RTOL,
        atol=DECLARED_ATOL,
    )
    print(
        "A02_DECODE_MAXABS="
        f"{float((case.decode_out - case.reference_decode).abs().max()):.3e}"
    )


# ---------------------------------------------------------------------------
# A03 -- the four field values, off the real get_kv_spec.
# ---------------------------------------------------------------------------
def _drive_runner_translation(model) -> dict:
    """Drive the runner's UNBOUND ``get_kv_cache_spec`` over a real model.

    The fake self carries only what that method reads, which is
    ``inc-glm53f-016``'s landed harness at ``test_get_kv_cache_spec_hybrid.py``
    ``:189-201``. The MODEL is the real one, because this arm's whole subject is
    the real ``get_kv_spec``'s output.
    """
    fake_self = SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=REGISTERED_HYBRID_BLOCK_SIZE, cache_dtype="auto"
            ),
            model_config=SimpleNamespace(dtype=torch.bfloat16),
        ),
        speculative_config=None,
        model=model,
    )
    return _runner_module().NeuronModelRunner.get_kv_cache_spec(fake_self)


@pytest.fixture(scope="module")
def real_spec():
    """The REAL model at the registered geometry, and its own ``get_kv_spec``."""
    model = _impl().Glm5NextForConditionalGeneration.from_configs(
        copy.deepcopy(_raw()),
        text_neuron_config=None,
        vision_neuron_config=None,
    )
    return model, model.get_kv_spec()


def test_kda_layer_a03_the_four_state_fields_are_reported_on_the_kda_half(
    real_spec,
) -> None:
    """34 of 34 carry all four, 11 of 11 carry none, and 34 translate to Mamba.

    The instrument is the real unmutated method: no fake model, no
    ``dataclasses.replace``, no hand-written ``LayerSpec``. This is
    ``inc-glm53f-016``'s declared wait discharged -- it certified the runner's
    translation at M1 against field values derived from the vendor calculators,
    and said the real values wait for M3.
    """
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
    print(
        f"A03_FIELD_CENSUS carried={len(carried)} bare={len(bare)} "
        f"partial={len(partial)}"
    )
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
        num_heads=DECLARED_KDA_NUM_HEADS,
        head_dim=DECLARED_KDA_HEAD_SIZE,
        conv_kernel_size=DECLARED_KDA_CONV_KERNEL_SIZE,
    )
    expected_dtypes = MambaStateDtypeCalculator.kda_state_dtype(torch.bfloat16, "auto")
    print(
        f"A03_EXPECTED conv={tuple(expected_conv)} rec={tuple(expected_recurrent)} "
        f"dtypes={expected_dtypes} world_size={world_size}"
    )
    for layer in carried:
        assert tuple(layer.kda_conv_state_shape) == tuple(expected_conv)
        assert tuple(layer.kda_recurrent_state_shape) == tuple(expected_recurrent)
        assert layer.kda_conv_state_dtype is expected_dtypes[0]
        assert layer.kda_recurrent_state_dtype is expected_dtypes[1]

    # ...and the runner's landed translation over that real output.
    specs = _drive_runner_translation(model)
    mamba = sum(1 for entry in specs.values() if isinstance(entry, MambaSpec))
    other = len(specs) - mamba
    print(f"A03_TRANSLATION total={len(specs)} mamba={mamba} non_mamba={other}")
    assert len(specs) == DECLARED_TOTAL_ENTRIES
    assert (mamba, other) == (DECLARED_KDA_ENTRIES, DECLARED_MLA_ENTRIES)


# ---------------------------------------------------------------------------
# A04 -- the non-vacuity control on A03's zero (D1.5).
# ---------------------------------------------------------------------------
def test_kda_layer_a04_the_pairing_guard_is_live_when_one_field_is_cleared(
    real_spec,
) -> None:
    """Clear one field on a COPY and the runner refuses the layer.

    Without this arm, A03's "the guard fired 0 times" would be equally true of a
    guard that can never fire. The copy is deep, so A03's own spec object is not
    touched by this arm whatever order the two run in.

    The spec is handed to the runner through a one-method stand-in rather than
    through the real model, and that is not the thing A03 forbids: A03's subject
    is what the real ``get_kv_spec`` reports, and a correct implementation cannot
    be made to report a partial set, so a mutated copy is the only way to reach
    the guard at all. The subject here is the runner's guard, not the model.
    """
    _model, spec = real_spec
    mutated = copy.deepcopy(spec)
    victim = next(
        layer for layer in mutated.layers if layer.kda_conv_state_dtype is not None
    )
    victim.kda_conv_state_dtype = None
    print(f"A04_CLEARED_ON={victim.name} FIELD=kda_conv_state_dtype")

    with pytest.raises(ValueError, match="part of it unset"):
        _drive_runner_translation(SimpleNamespace(get_kv_spec=lambda: mutated))
