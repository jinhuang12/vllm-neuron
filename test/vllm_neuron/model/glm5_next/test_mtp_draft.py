"""Acceptance for the GLM-5.3-Flash MTP draft head -- ``inc-glm53f-063``, WP9.

THE TWO COUNTED CONJUNCTS, in the plan block's own words: the draft head emits
exactly one draft token per step (``nextn = 1``) over 4/4 steps, and its logits
match a torch reference draft head at ``rtol=1e-2, atol=1e-5``. The route
predicate (D13 form R-3) stands alongside them: ``can_run_kernel()`` True and the
attention module's torch-fallback counter exactly 0 across the four steps.

ONE ITEM PER CONJUNCT, AND A FIRING CONTROL FOR EVERY COUNTED ZERO. No
``parametrize`` -- the sibling KDA layer test states the reason for that shape
(``test_kda_layer.py:11``): six arms, one item each, so a failure names itself in
the item id rather than in a parameter tuple. A zero is not a reading until the
same instrument has been shown reporting non-zero, so the gate zero, the
fallback zero and both value comparisons each carry a control that must fail.

WHAT DRIVES THE HEAD. The checkpoint's MTP layer is layer index 45 --
``num_hidden_layers``, one past the main stack -- and it is
``deepseek_sparse_attention``-typed, so the block is a real
``Glm5NextDSALayer`` built through ``_build_layer`` at the tiny geometry. Two
blocks are built from ONE seed, so their weights are identical, and each side
gets its own cache set: the pool cache, the latent cache and the tail ring are
all mutated in place, so sharing them would compare a run against itself. That
is ``inc-glm53f-051``'s own discipline (``test_dsa_layer.py:1704-1710``), and the
fixture recipe below is re-derived from that file rather than imported from it --
it carries a live repair, and a cross-test import would couple this acceptance to
another increment's open work.

TWO RECORDED COVERAGE LIMITS, both declared rather than discovered later.

1. **The block is half a layer at this base, and the reference mirrors the same
   half.** ``Glm5NextDSALayer.forward`` is pre-norm, the sparse attention half
   and the residual add; ``self.mlp`` raises, because both MLP forwards are still
   ``inc-glm53f-013`` stubs (``model_fp8.py:2127``, ``:2149`` -- the second says
   the dense path lands with ``inc-glm53f-054``'s forward). So THE FEED-FORWARD
   HALF IS ABSENT from the logits path these items read, and neither conjunct
   reads an FFN. ``inc-glm53f-054`` re-derives both conjuncts over the full layer
   when it joins the halves. Ruled as a recorded limit, not a gap.
2. **The reference shares the block's code, though not its state.** This
   increment owns composition -- the two input norms, the fused projection, the
   position-zero mask, the step resolution, the shared-head norm and the logits
   matmul -- and it owns no attention numerics, which are
   ``inc-glm53f-051``'s and are measured by that increment's own test. So the
   reference re-derives exactly what this increment owns and calls the same layer
   class for the rest. That makes the value comparison a test of ORDER AND
   WIRING rather than of attention arithmetic, which is why the reversed-concat
   control below matters: without it the comparison could pass by construction.

Everything the head needs that this base does not yet supply --
``num_nextn_predict_layers`` as a config field, and the MTP keys in the weight
map -- is ``inc-glm53f-064``'s recorded Surface rider. The count is therefore
DERIVED HERE from the digest-pinned checkpoint config rather than typed.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path

import pytest
import torch

# --------------------------------------------------------------------------- #
# THE REGISTERED TOLERANCE PAIR. Cited, never re-authored: it is the plan block's
# own, and this file spells no second pair (P9).

RTOL = 1e-2
ATOL = 1e-5

#: The block's declared step count for conjunct 1: "over 4/4 steps".
STEPS = 4

SENT = "MTPHEAD"

# --------------------------------------------------------------------------- #
# THE DIGEST-PINNED CHECKPOINT CONFIG. Trusted only after its sha256 matches, the
# convention the sibling KDA test states (``test_kda_layer.py:97``, ``:177-184``).

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "hf-config.json"
FIXTURE_SHA256 = "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f"

# --------------------------------------------------------------------------- #
# THE TINY GEOMETRY. Re-derived from ``test_dsa_layer.py:209-219`` -- at the
# checkpoint's own widths one layer plus a vocabulary-wide head is far past what a
# unit test can carry, and the entries below are the widths that file records as
# admissible. ``vocab_size`` is narrowed here too, which that file did not need:
# the logits matmul is this increment's and the checkpoint's real vocabulary would
# be a 150k-row weight.

POOL_SIZE = 4
TOPK_POOLS = 2
PAGE_SIZE = 4
PAGES = 8
PREFILL_TOKENS = 35
INDEX_HEAD_DIM = 128
TINY_VOCAB = 64

TINY_GEOMETRY: dict[str, int] = {
    "hidden_size": 256,
    "num_attention_heads": 4,
    "q_lora_rank": 128,
    "kv_lora_rank": 128,
    "qk_nope_head_dim": 64,
    "qk_rope_head_dim": 0,
    "v_head_dim": 64,
    "index_n_heads": 4,
    "index_head_dim": INDEX_HEAD_DIM,
    "index_kpool": POOL_SIZE,
    "vocab_size": TINY_VOCAB,
}

#: ``head_size``, DERIVED the way the implementation derives it
#: (``model_fp8.py:407-416``) rather than typed.
TINY_HEAD_SIZE = TINY_GEOMETRY["kv_lora_rank"] + TINY_GEOMETRY["qk_rope_head_dim"]

SOFTMAX_SCALE = float(TINY_GEOMETRY["kv_lora_rank"] ** -0.5)

#: The seven DSA counter families, one per module, since a counter is per module
#: (``test_dsa_layer.py:239-257``).
FAMILIES: tuple[str, ...] = (
    "decode_tail_update",
    "index_expand",
    "kpool_hadamard",
    "paged_gather",
    "ragged_pack",
    "score_gemm",
    "topk_select",
)


def say(*parts: object) -> None:
    """Print a reading. The suite runs under ``-s``, so these reach the transcript."""
    print(f"{SENT}|" + "|".join(str(p) for p in parts), flush=True)


# --------------------------------------------------------------------------- #
# IMPORTS INSIDE HELPERS, NEVER AT MODULE LEVEL.


def _impl():
    """The model tree, imported inside a body.

    The sibling tests' idiom and their reason (``test_dsa_layer.py:638-644``): a
    module-level import turns what should be a test failure into a collection
    error, and a landed ``test_factory.py`` item reads whether this module is in
    ``sys.modules``.
    """
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _mtp():
    """The module under test, imported inside a body for the same reason."""
    from vllm_neuron.model.glm5_next import mtp

    return mtp


def gate_live() -> bool:
    from vllm_neuron.utils.neuron_utils import can_run_kernel

    return bool(can_run_kernel())


def _counter_api(family: str):
    """``(reset, read)`` for one family, DISCOVERED on the module, not spelled out.

    ``test_dsa_layer.py:358-368``'s reason, which still holds: one family's pair
    does not follow its module name, and discovery fails loudly if a module ever
    grows a second pair.
    """
    module = importlib.import_module(f"vllm_neuron.functional.dsa.{family}")
    names = [n for n in dir(module) if n.endswith("_dispatch_counters")]
    reset = [n for n in names if n.startswith("reset_")]
    read = [n for n in names if not n.startswith("reset_")]
    assert len(reset) == 1 and len(read) == 1, (family, reset, read)
    return getattr(module, reset[0]), getattr(module, read[0])


def reset_all_counters() -> None:
    for family in FAMILIES:
        _counter_api(family)[0]()


def read_all_counters() -> dict[str, tuple[int, int]]:
    """``{family: (nki_dispatch, torch_fallback)}`` since the last reset."""
    return {f: tuple(int(v) for v in _counter_api(f)[1]()) for f in FAMILIES}


# --------------------------------------------------------------------------- #
# THE FIXTURE.


def _pinned_checkpoint_config() -> dict:
    """The checkpoint's own text config, after its digest is checked."""
    raw = FIXTURE_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == FIXTURE_SHA256, (
        f"{FIXTURE_PATH.name} is not the pinned fixture: sha256 {digest} != "
        f"{FIXTURE_SHA256}. Every value this file derives comes from it, so a "
        f"changed fixture invalidates the derivation rather than the assertion"
    )
    return json.loads(raw.decode())["text_config"]


def _tiny_text_config(**overrides):
    """The checkpoint's config narrowed to :data:`TINY_GEOMETRY`.

    ``dataclasses.replace`` rather than a fresh construction, so every field this
    file does NOT name keeps the checkpoint's value and a config drift reaches
    this file instead of being overwritten by it
    (``test_dsa_layer.py:649-662``). ``index_topk`` is set from the pool count so
    the pool granularity stays derived: ``select_k()`` is
    ``index_topk // index_kpool``.
    """
    from dataclasses import replace

    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    fields = dict(TINY_GEOMETRY)
    fields["index_topk"] = TOPK_POOLS * POOL_SIZE
    fields.update(overrides)
    return replace(Glm5NextTextConfig(), **fields)


def _materialise_indexer(indexer, gen: torch.Generator) -> None:
    """Fill the indexer's declared leaves and prepare its projections.

    Re-derived from ``test_dsa_layer.py:1456-1482``, including its note that the
    compression ape is stored in the checkpoint's bfloat16 rather than pre-cast:
    the implementation casts it per call (``model_fp8.py:3215``), and pre-casting
    here would make a reference agree with a cast the implementation still owes.
    """
    for name, in_features, out_features in indexer.projection_widths():
        weight = torch.randn(
            out_features, in_features, generator=gen, dtype=torch.float32
        ) * (in_features**-0.5)
        setattr(indexer, indexer.PROJECTION_PARAMETERS[name], torch.nn.Parameter(weight))
    head_dim = int(indexer.index_head_dim)
    indexer.k_norm_weight = torch.nn.Parameter(
        1.0 + torch.randn(head_dim, generator=gen, dtype=torch.float32) * 0.05
    )
    indexer.k_norm_bias = torch.nn.Parameter(
        torch.randn(head_dim, generator=gen, dtype=torch.float32) * 0.02
    )
    indexer.index_kpool_compress_ape = torch.nn.Parameter(
        (
            torch.randn(
                int(indexer.index_kpool), head_dim, generator=gen, dtype=torch.float32
            )
            * 0.1
        ).to(torch.bfloat16),
        requires_grad=False,
    )
    assert indexer.prepare_projection_weights() == 4, "four indexer projections must prepare"


def _build_dsa_block(cfg, layer_idx: int, seed: int):
    """One materialised ``Glm5NextDSALayer``, built the way the tree builds one.

    Through ``_build_layer`` on the checkpoint's own layer type, so the block this
    head composes is chosen by the same equality dispatch the stack uses
    (``model_fp8.py:4715-4732``) rather than by naming a class here. Weight scale
    is ``randn * in_features ** -0.5``, which keeps activations order one
    (``test_dsa_layer.py:1487-1490``).
    """
    model_fp8 = _impl()
    from vllm_neuron.model.glm5_next.config import DSA_LAYER_TYPE

    gen = torch.Generator().manual_seed(int(seed))
    layer = model_fp8._build_layer(cfg, int(layer_idx), DSA_LAYER_TYPE, 1)
    assert type(layer) is model_fp8.Glm5NextDSALayer, (
        f"the checkpoint's MTP layer type {DSA_LAYER_TYPE!r} must build a DSA "
        f"layer; got {type(layer).__name__}"
    )
    layer.input_layernorm_weight = torch.nn.Parameter(
        1.0 + torch.randn(int(cfg.hidden_size), generator=gen, dtype=torch.float32) * 0.05
    )
    attention = layer.attention
    for name, in_features, out_features in attention.projection_widths():
        weight = torch.randn(
            out_features, in_features, generator=gen, dtype=torch.float32
        ) * (in_features**-0.5)
        setattr(attention, f"{name}_weight", torch.nn.Parameter(weight))
    for gain_name, width in (
        ("q_a_layernorm_weight", int(cfg.q_lora_rank)),
        ("kv_a_layernorm_weight", int(cfg.kv_lora_rank)),
    ):
        setattr(
            attention,
            gain_name,
            torch.nn.Parameter(
                1.0 + torch.randn(width, generator=gen, dtype=torch.float32) * 0.05
            ),
        )
    assert attention.prepare_projection_weights() == 5, "five MLA projections must prepare"
    assert attention.prepare_absorb_weights() == 2, "and both absorb operands must split"
    _materialise_indexer(attention.indexer, gen)
    return layer


def _caches(cfg):
    """One cache set. Each side gets its OWN, because all three are written in place."""
    return {
        "pool_cache": torch.zeros(
            PAGES * PAGE_SIZE, int(cfg.index_head_dim), dtype=torch.bfloat16
        ),
        "latent_cache": torch.zeros(
            PREFILL_TOKENS + STEPS, 1, TINY_HEAD_SIZE, dtype=torch.float32
        ),
        "tail": torch.zeros(2, int(cfg.index_kpool), int(cfg.index_head_dim), dtype=torch.bfloat16),
    }


def _prefill_slot_mapping(tokens: int, pool: int) -> torch.Tensor:
    """The pool-granular slot per position: the pool's own id where one COMPLETES.

    A ``-1`` is how a position says "my window is not a whole pool"; those rows
    are steered to the trash row rather than dropped
    (``test_dsa_layer.py:1531-1540``, citing ``model_fp8.py:3209``).
    """
    slots = torch.full((int(tokens),), -1, dtype=torch.int32)
    for position in range(int(tokens)):
        if (position + 1) % int(pool) == 0:
            slots[position] = position // int(pool)
    return slots


def _populate_caches(block, cfg, caches) -> None:
    """One prefill through the block, so the decode steps have a context to attend to.

    FIXTURE SETUP, NOT A READING: no item below asserts anything about this call's
    output. It exists because a draft head at decode reads a populated cache, and
    an empty one would make every step attend to zeros.
    """
    hidden = torch.randn(
        PREFILL_TOKENS,
        int(cfg.hidden_size),
        generator=torch.Generator().manual_seed(63_000_001),
        dtype=torch.float32,
    )
    block.forward(
        hidden,
        latent_cache=caches["latent_cache"],
        pool_cache=caches["pool_cache"],
        seq_lens=torch.arange(1, PREFILL_TOKENS + 1, dtype=torch.int32),
        start_position=0,
        softmax_scale=SOFTMAX_SCALE,
        max_seq_len=PREFILL_TOKENS,
        page_size=PAGE_SIZE,
        slot_mapping=_prefill_slot_mapping(PREFILL_TOKENS, int(cfg.index_kpool)),
        tail=None,
    )


def _step_block_kwargs(caches, step: int) -> dict:
    """The block's own decode keyword arguments for one step.

    Built here and threaded through the head unchanged, which is the point of the
    head taking its block as an argument: it knows none of these names.
    """
    position = PREFILL_TOKENS + int(step)
    return {
        "latent_cache": caches["latent_cache"],
        "pool_cache": caches["pool_cache"],
        "seq_lens": torch.tensor([position + 1], dtype=torch.int32),
        "start_position": position,
        "softmax_scale": SOFTMAX_SCALE,
        "max_seq_len": position + 1,
        "page_size": PAGE_SIZE,
        "tail": caches["tail"],
        "position": position,
    }


def _step_inputs(cfg, step: int) -> dict:
    """One step's head-side operands. A fresh draw per step, so no two steps agree."""
    gen = torch.Generator().manual_seed(63_100_000 + int(step))
    return {
        "inputs_embeds": torch.randn(
            1, int(cfg.hidden_size), generator=gen, dtype=torch.float32
        ),
        "previous_hidden_states": torch.randn(
            1, int(cfg.hidden_size), generator=gen, dtype=torch.float32
        ),
        "positions": torch.tensor([PREFILL_TOKENS + int(step)], dtype=torch.int64),
    }


def _head_weights(cfg, seed: int = 63_200_001) -> dict:
    """The four MTP tensors plus the SHARED head weight.

    Layer 45 ships no head tensor of its own, so ``lm_head_weight`` is the root's
    and it travels as an argument rather than as a fifth declared parameter.
    """
    gen = torch.Generator().manual_seed(int(seed))
    hidden = int(cfg.hidden_size)
    return {
        "enorm_weight": 1.0 + torch.randn(hidden, generator=gen, dtype=torch.float32) * 0.05,
        "hnorm_weight": 1.0 + torch.randn(hidden, generator=gen, dtype=torch.float32) * 0.05,
        "eh_proj_weight": torch.randn(
            hidden, 2 * hidden, generator=gen, dtype=torch.float32
        ) * ((2 * hidden) ** -0.5),
        "shared_head_norm_weight": 1.0
        + torch.randn(hidden, generator=gen, dtype=torch.float32) * 0.05,
        "lm_head_weight": torch.randn(
            int(cfg.vocab_size), hidden, generator=gen, dtype=torch.float32
        ) * (hidden**-0.5),
    }


def _build_head(cfg, weights: dict, blocks: list, nextn: int):
    """The draft head with its four parameters materialised."""
    head = _mtp().Glm5NextMultiTokenPredictor(cfg, int(nextn), blocks)
    for layer in head.layers.values():
        for name in ("enorm_weight", "hnorm_weight", "eh_proj_weight", "shared_head_norm_weight"):
            setattr(layer, name, torch.nn.Parameter(weights[name].clone()))
    return head


def _reference_logits(weights: dict, block, step_inputs: dict, block_kwargs: dict) -> torch.Tensor:
    """An independently written torch draft head over the SAME partial block.

    Re-derives every step this increment owns -- the position-zero mask, both
    RMSNorms in fp32, the concatenation in embedding-then-hidden order, the fused
    projection, the shared-head norm and the logits matmul -- from plain torch
    ops. It calls the block for the attention half, which this increment does not
    own; see this module's docstring, limits 1 and 2.
    """

    def rms(x: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
        promoted = x.to(torch.float32)
        inverse = torch.rsqrt(promoted.pow(2).mean(dim=-1, keepdim=True) + eps)
        return ((promoted * inverse) * gain.to(torch.float32)).to(x.dtype)

    eps = float(block.rms_norm_eps)
    embeds = step_inputs["inputs_embeds"]
    mask = (step_inputs["positions"].unsqueeze(-1) == 0).expand_as(embeds)
    embeds = torch.where(mask, torch.zeros_like(embeds), embeds)
    embeds = rms(embeds, weights["enorm_weight"], eps)
    previous = rms(step_inputs["previous_hidden_states"], weights["hnorm_weight"], eps)
    joined = torch.cat([embeds, previous], dim=-1)
    hidden = joined @ weights["eh_proj_weight"].t()
    hidden = block.forward(hidden, **block_kwargs)
    hidden = rms(hidden, weights["shared_head_norm_weight"], eps)
    return hidden @ weights["lm_head_weight"].t()


def _one_token_per_step(drafted: torch.Tensor, positions: int, label: str) -> None:
    """THE CONJUNCT-1 PREDICATE, factored so a control can be measured by it.

    One step resolves one draft layer and a layer produces one hidden state per
    position, so a step's proposal is exactly one token per position -- rank 1,
    and ``positions`` elements. A head that emitted two per step fails on the
    rank, which is what the control below provokes.
    """
    assert drafted.ndim == 1, (
        f"[{label}] a step's proposal must be one token per position, so rank 1; "
        f"got shape {tuple(drafted.shape)}"
    )
    assert drafted.shape[0] == positions, (
        f"[{label}] {positions} position(s) in, {drafted.shape[0]} token(s) out"
    )
    assert drafted.numel() == positions, (
        f"[{label}] {drafted.numel()} token(s) for {positions} position(s)"
    )


# --------------------------------------------------------------------------- #
# ITEM 1 -- the derivation. Nothing below types the count or the layer index.


def test_the_checkpoint_declares_one_draft_layer_at_index_45() -> None:
    """``nextn`` and the MTP layer index are the checkpoint's, read from the pin.

    ``Glm5NextTextConfig`` carries no ``num_nextn_predict_layers`` field at this
    base, which is why the head takes the count as an argument and why this item
    reads it from the digest-pinned config instead. Both facts are printed, so a
    later checkpoint change shows up as a changed reading.
    """
    text_config = _pinned_checkpoint_config()
    nextn = int(text_config["num_nextn_predict_layers"])
    layers = int(text_config["num_hidden_layers"])
    say("derived", "num_nextn_predict_layers", nextn, "num_hidden_layers", layers)
    assert nextn == 1, f"the checkpoint declares {nextn} draft layer(s), not 1"
    assert layers == 45, f"the checkpoint declares {layers} hidden layers, not 45"

    cfg = _tiny_text_config()
    assert int(cfg.num_hidden_layers) == layers, (
        f"the tiny config's num_hidden_layers ({cfg.num_hidden_layers}) disagrees "
        f"with the pinned checkpoint's ({layers}); the head's start index is "
        f"derived from it, so the two must agree"
    )
    head = _build_head(cfg, _head_weights(cfg), [_build_dsa_block(cfg, layers, 63_1)], nextn)
    say("head", "mtp_start_layer_idx", head.mtp_start_layer_idx, "window", head.draft_window_width)
    assert head.mtp_start_layer_idx == layers
    assert head.draft_window_width == nextn
    assert sorted(head.layers.keys()) == [str(layers)]


def test_the_module_under_test_is_the_candidate_tree() -> None:
    """§78.1 -- the import resolved inside the candidate, not an installed copy."""
    module = _mtp()
    origin = Path(module.__file__).resolve()
    say("import_origin", origin)
    assert origin.is_file(), f"{origin} does not exist"
    expected = (Path(__file__).resolve().parents[4] / "vllm_neuron" / "model" / "glm5_next" / "mtp.py")
    assert origin == expected.resolve(), (
        f"the module under test resolved to {origin}, not the candidate's "
        f"{expected}; every reading below would then describe another tree"
    )


def test_importing_the_head_does_not_load_the_model_tree() -> None:
    """The head imports NO model tree, measured in a fresh child process.

    This is a property of the module rather than of a run, and this session's
    ``sys.modules`` cannot answer it: any earlier item here imports
    ``model_fp8`` to build a block, so by the time this item ran the module
    would be loaded whatever ``mtp.py`` does. So it is measured where the
    question is well posed -- a child that imports only the head. The overlay
    already uses that technique for a value its own process has pinned
    (``test/conftest.py:118-123``).

    WHY IT MATTERS: a landed ``test_factory.py`` item reads whether the model
    tree is in ``sys.modules``, so a head that pulled it in at import would
    change that item's reading from another file.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import sys
        from vllm_neuron.model.glm5_next import mtp
        loaded = sorted(n for n in sys.modules if n.endswith("glm5_next.model_fp8"))
        print("MTP_FILE=" + str(mtp.__file__))
        print("MODEL_FP8_LOADED=" + str(bool(loaded)) + " " + str(loaded))
        """
    )
    root = Path(__file__).resolve().parents[4]
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, capture_output=True, text=True, timeout=300
    )
    say("import_isolation", "rc", result.returncode, result.stdout.strip().replace("\n", " ; "))
    assert result.returncode == 0, (
        f"the child could not import the head alone: {result.stderr[-2000:]}"
    )
    assert "MODEL_FP8_LOADED=False" in result.stdout, (
        f"importing the head loaded the model tree: {result.stdout.strip()}. The "
        f"head takes its block as an argument precisely so it does not have to"
    )


# --------------------------------------------------------------------------- #
# ITEM 2 -- the gate, before any count is read.


def test_the_nki_gate_is_live_before_any_count_is_read() -> None:
    """Without this, every family reads 0 and the fallback check is a false pass.

    Asserted rather than assumed, with its inputs printed, so a run in the wrong
    mode says which condition refused (``test_dsa_layer.py:504-509``).
    """
    import os

    from vllm_neuron import envs

    say("gate", "VLLM_NEURON_CPU_MODE", bool(envs.VLLM_NEURON_CPU_MODE))
    say("gate", "NKI_SIMULATOR", os.environ.get("NKI_SIMULATOR"))
    say("gate", "can_run_kernel", gate_live())
    assert gate_live(), (
        "can_run_kernel() is False, so every DSA seam would take its torch path "
        "and the route predicate's zero would be an absence rather than a reading"
    )


# --------------------------------------------------------------------------- #
# CONJUNCT 1 -- exactly one draft token per step, over 4/4 steps.


def test_the_head_emits_exactly_one_draft_token_per_step_over_four_steps() -> None:
    """CONJUNCT 1. Four steps, each proposing one token per position, 4/4."""
    if not gate_live():
        pytest.skip("the NKI gate is not live; the readings would be meaningless")

    text_config = _pinned_checkpoint_config()
    nextn = int(text_config["num_nextn_predict_layers"])
    cfg = _tiny_text_config()
    weights = _head_weights(cfg)
    block = _build_dsa_block(cfg, int(text_config["num_hidden_layers"]), 63_301)
    caches = _caches(cfg)
    _populate_caches(block, cfg, caches)
    head = _build_head(cfg, weights, [block], nextn)

    total = 0
    steps_seen = 0
    for step in range(STEPS):
        inputs = _step_inputs(cfg, step)
        drafted = head.propose_draft_tokens(
            lm_head_weight=weights["lm_head_weight"],
            spec_step_index=step,
            **inputs,
            **_step_block_kwargs(caches, step),
        )
        positions = int(inputs["positions"].numel())
        say("conjunct1", "step", step, "positions", positions, "tokens", drafted.numel())
        _one_token_per_step(drafted, positions, f"step{step}")
        total += int(drafted.numel())
        steps_seen += 1

    say("conjunct1", "steps", steps_seen, "of", STEPS, "total_tokens", total)
    assert steps_seen == STEPS, f"{steps_seen} step(s) ran, not {STEPS}"
    assert total == STEPS, (
        f"{total} draft token(s) over {STEPS} step(s) at one position each; "
        f"exactly {STEPS} is what one token per step means here"
    )


def test_a_head_that_emitted_two_tokens_per_step_fails_conjunct_one() -> None:
    """FIRING CONTROL for conjunct 1: the predicate must reject a wider proposal.

    Widens the logits by one axis, so ``argmax`` yields two tokens per position.
    Nothing about the shape check is changed -- only the value it reads -- so this
    shows the assertion can fail rather than asserting that it would.
    """
    if not gate_live():
        pytest.skip("the NKI gate is not live; the readings would be meaningless")

    text_config = _pinned_checkpoint_config()
    cfg = _tiny_text_config()
    weights = _head_weights(cfg)
    block = _build_dsa_block(cfg, int(text_config["num_hidden_layers"]), 63_302)
    caches = _caches(cfg)
    _populate_caches(block, cfg, caches)
    head = _build_head(cfg, weights, [block], int(text_config["num_nextn_predict_layers"]))

    real = head.compute_draft_logits

    def widened(hidden_states, lm_head_weight):
        logits = real(hidden_states, lm_head_weight)
        return torch.stack([logits, logits.flip(-1)], dim=1)

    head.compute_draft_logits = widened
    drafted = head.propose_draft_tokens(
        lm_head_weight=weights["lm_head_weight"],
        spec_step_index=0,
        **_step_inputs(cfg, 0),
        **_step_block_kwargs(caches, 0),
    )
    say("control_conjunct1", "mutant_shape", tuple(drafted.shape))
    with pytest.raises(AssertionError, match="rank 1"):
        _one_token_per_step(drafted, 1, "mutant")


def test_the_step_index_wraps_by_the_draft_layer_count() -> None:
    """What ``nextn`` MEANS for step resolution, measured on a two-layer window.

    At the checkpoint's count of 1 every step resolves to the one layer, so that
    value alone cannot distinguish a modulo from a head that ignores the step
    entirely. A two-layer window makes the difference visible: four steps must
    resolve 45, 46, 45, 46. No numerics are read here, so the blocks are plain
    stand-ins -- this item measures index arithmetic only.
    """
    cfg = _tiny_text_config()
    start = int(cfg.num_hidden_layers)
    blocks = [torch.nn.Identity(), torch.nn.Identity()]
    head = _build_head(cfg, _head_weights(cfg), blocks, 2)
    resolved = [head.resolve_step_layer_index(step) for step in range(STEPS)]
    say("step_resolution", "window", 2, "resolved", resolved)
    assert resolved == [start, start + 1, start, start + 1], (
        f"a two-layer window must alternate {start} and {start + 1} across "
        f"{STEPS} steps; got {resolved}"
    )
    with pytest.raises(ValueError, match="must not be negative"):
        head.resolve_step_layer_index(-1)


# --------------------------------------------------------------------------- #
# CONJUNCT 2 -- the logits match a torch reference draft head.


def _run_both_sides(seed_impl: int, seed_ref: int, *, reverse_reference_concat: bool = False):
    """Both sides over four steps, each on its own caches. Returns paired logits.

    The two blocks are built from the SAME seed, so their weights are identical
    and the only difference between the sides is the arithmetic under test. Each
    side carries its own caches from the prefill into every step, rather than one
    side cloning the other's: a wrong cache write would otherwise make the sides
    agree with themselves.
    """
    text_config = _pinned_checkpoint_config()
    cfg = _tiny_text_config()
    weights = _head_weights(cfg)
    layer_idx = int(text_config["num_hidden_layers"])

    impl_block = _build_dsa_block(cfg, layer_idx, seed_impl)
    ref_block = _build_dsa_block(cfg, layer_idx, seed_ref)
    impl_caches, ref_caches = _caches(cfg), _caches(cfg)
    _populate_caches(impl_block, cfg, impl_caches)
    _populate_caches(ref_block, cfg, ref_caches)
    head = _build_head(cfg, weights, [impl_block], int(text_config["num_nextn_predict_layers"]))

    reference_weights = dict(weights)
    if reverse_reference_concat:
        # The projection's columns are the checkpoint's in embedding-then-hidden
        # order; swapping the two halves is the mutation this controls for.
        hidden = int(cfg.hidden_size)
        left, right = weights["eh_proj_weight"][:, :hidden], weights["eh_proj_weight"][:, hidden:]
        reference_weights["eh_proj_weight"] = torch.cat([right, left], dim=-1)

    pairs = []
    for step in range(STEPS):
        inputs = _step_inputs(cfg, step)
        got_hidden = head.forward(
            spec_step_index=step, **inputs, **_step_block_kwargs(impl_caches, step)
        )
        got = head.compute_draft_logits(got_hidden, weights["lm_head_weight"])
        expected = _reference_logits(
            reference_weights, ref_block, inputs, _step_block_kwargs(ref_caches, step)
        )
        pairs.append((got, expected))
    return pairs


def test_the_draft_logits_match_a_torch_reference_head_at_every_step() -> None:
    """CONJUNCT 2, at the block's registered ``rtol=1e-2, atol=1e-5``.

    The achieved error is printed BEFORE the assertion, so a failure arrives with
    its own measurement rather than only with a verdict.
    """
    if not gate_live():
        pytest.skip("the NKI gate is not live; the readings would be meaningless")

    pairs = _run_both_sides(63_401, 63_401)
    assert len(pairs) == STEPS, f"{len(pairs)} step(s) compared, not {STEPS}"
    for step, (got, expected) in enumerate(pairs):
        assert got.numel() > 0, f"step {step} produced an empty logits tensor"
        assert got.shape == expected.shape, (
            f"step {step}: {tuple(got.shape)} vs reference {tuple(expected.shape)}"
        )
        error = (got - expected).abs().max().item()
        say("conjunct2", "step", step, "max_abs_err", error, "rtol", RTOL, "atol", ATOL)
        torch.testing.assert_close(got, expected, rtol=RTOL, atol=ATOL)


def test_reversing_the_reference_concat_order_breaks_the_comparison() -> None:
    """FIRING CONTROL for conjunct 2: the comparison must be able to fail.

    The head concatenates the embedding half first and the previous hidden state
    second. Swapping the projection's two column blocks on the reference side is
    an arithmetically valid draft head that is not THIS one, so a comparison that
    still passed would be measuring nothing.
    """
    if not gate_live():
        pytest.skip("the NKI gate is not live; the readings would be meaningless")

    pairs = _run_both_sides(63_402, 63_402, reverse_reference_concat=True)
    errors = [(got - expected).abs().max().item() for got, expected in pairs]
    say("control_conjunct2", "max_abs_err_per_step", errors)
    with pytest.raises(AssertionError):
        for got, expected in pairs:
            torch.testing.assert_close(got, expected, rtol=RTOL, atol=ATOL)


def test_the_embedding_is_zeroed_at_absolute_position_zero() -> None:
    """The position-zero mask, which no counted step above exercises.

    The four counted steps all sit past the prefill, so their positions are
    non-zero and the mask never fires there. This item reads the branch directly:
    at position 0 the head must produce exactly what it produces from a zero
    embedding, because the first position has no previous token to draft from.
    """
    if not gate_live():
        pytest.skip("the NKI gate is not live; the readings would be meaningless")

    text_config = _pinned_checkpoint_config()
    cfg = _tiny_text_config()
    weights = _head_weights(cfg)
    layer_idx = int(text_config["num_hidden_layers"])
    nextn = int(text_config["num_nextn_predict_layers"])
    inputs = _step_inputs(cfg, 0)

    outputs = []
    for embeds in (inputs["inputs_embeds"], torch.zeros_like(inputs["inputs_embeds"])):
        block = _build_dsa_block(cfg, layer_idx, 63_501)
        caches = _caches(cfg)
        _populate_caches(block, cfg, caches)
        head = _build_head(cfg, weights, [block], nextn)
        outputs.append(
            head.forward(
                inputs_embeds=embeds,
                previous_hidden_states=inputs["previous_hidden_states"],
                positions=torch.zeros(1, dtype=torch.int64),
                spec_step_index=0,
                **_step_block_kwargs(caches, 0),
            )
        )
    masked, from_zero = outputs
    error = (masked - from_zero).abs().max().item()
    say("position_zero", "max_abs_err", error)
    torch.testing.assert_close(masked, from_zero, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------- #
# THE ROUTE PREDICATE (D13 form R-3).


def _counters_over_four_steps() -> dict:
    """Four steps through the head, with the counters read after the reset lands."""
    text_config = _pinned_checkpoint_config()
    cfg = _tiny_text_config()
    weights = _head_weights(cfg)
    block = _build_dsa_block(cfg, int(text_config["num_hidden_layers"]), 63_601)
    caches = _caches(cfg)
    _populate_caches(block, cfg, caches)
    head = _build_head(cfg, weights, [block], int(text_config["num_nextn_predict_layers"]))

    # The reset sits after the fixture and before the measured steps, and the
    # landing is proved: a reading taken against a reset that did not land would
    # be partly the prefill's dispatches.
    reset_all_counters()
    after_reset = read_all_counters()
    assert all(value == (0, 0) for value in after_reset.values()), (
        f"the counters did not reset to zero: {after_reset}"
    )

    for step in range(STEPS):
        head.propose_draft_tokens(
            lm_head_weight=weights["lm_head_weight"],
            spec_step_index=step,
            **_step_inputs(cfg, step),
            **_step_block_kwargs(caches, step),
        )
    return read_all_counters()


def test_the_route_predicate_holds_across_the_four_steps() -> None:
    """ROUTE PREDICATE: the gate is live and no seam took its torch path.

    Every DSA seam here is kernel-class, so a torch fallback is a route failure
    and a P13 defect rather than a slow path.
    """
    if not gate_live():
        pytest.skip("the NKI gate is not live; the readings would be meaningless")

    readings = _counters_over_four_steps()
    for family in FAMILIES:
        say("route", "counter", family, readings[family])
    assert gate_live(), "the gate must still be live at the end of the four steps"

    fallbacks = {f: readings[f][1] for f in FAMILIES}
    dispatched = {f: readings[f][0] for f in FAMILIES if readings[f][0] > 0}
    say("route", "torch_fallbacks", fallbacks, "families_dispatched", sorted(dispatched))
    assert dispatched, (
        f"no DSA seam dispatched at all across {STEPS} steps: {readings}. Seven "
        f"zero fallbacks would then mean nothing ran, not that the route held"
    )
    assert all(count == 0 for count in fallbacks.values()), (
        f"a torch fallback ran across the {STEPS} draft steps: {readings}. Every "
        f"DSA seam here is kernel-class (P13), so this is a route failure"
    )


def test_the_fallback_counter_reads_non_zero_when_a_fallback_is_provoked() -> None:
    """FIRING CONTROL for the route predicate's zero (§64).

    Same module, same public entry point, same counter reader -- only the geometry
    differs. Both arms are measured, so the reading that separates them is the
    route and not the instrument: a geometry the kernel serves must move
    ``nki_dispatch``, and one its own gate refuses must move ``torch_fallback``.

    THE REFUSED GEOMETRY IS THE MODULE'S OWN, NOT A MUTATION. ``topk_select``
    rejects ``k == width`` in terms, because the kernel needs sorted output and
    the assert for it lives in the kernel body rather than in the factories
    (``topk_select.py:279-291``, ``:355-362``). So the fallback is reached the way
    production would reach it, and nothing here patches a gate, deletes an
    environment variable or touches a seam.

    WHY NOT CLOSE THE GATE GLOBALLY, which is what round 1 of this file did: the
    MLA projections call ``wrap_nki`` with NO ``can_run_kernel`` check
    (``mla_projections.py:272``), so removing ``NKI_SIMULATOR`` does not route
    that seam to torch -- it makes the NKI HOP's CPU implementation raise
    ``RuntimeError`` before any DSA counter is reached. The smoke run found
    exactly that. Closing one seam's gate is therefore not a way to provoke this
    seam's fallback, and driving the seam directly is.
    """
    if not gate_live():
        pytest.skip("the NKI gate is not live; the readings would be meaningless")

    from vllm_neuron.functional.dsa.topk_select import dsa_topk_select

    cfg = _tiny_text_config()
    # The geometry the counted run actually exercised, DERIVED rather than typed:
    # one score row per prefill token, one column per candidate pool, and the
    # indexer's own selection width.
    rows = PREFILL_TOKENS
    width = PREFILL_TOKENS // int(cfg.index_kpool)
    served_k = int(cfg.index_topk) // int(cfg.index_kpool)
    scores = torch.randn(
        rows, width, generator=torch.Generator().manual_seed(63_701), dtype=torch.float32
    )
    say("control_route", "rows", rows, "width", width, "served_k", served_k)
    assert 0 < served_k < width, (
        f"the served arm needs 0 < k < width to be the kernel's own case; "
        f"k={served_k} width={width}"
    )

    readings = {}
    for arm, k in (("served", served_k), ("refused", width)):
        _counter_api("topk_select")[0]()
        values, _indices = dsa_topk_select(scores, k)
        assert values.numel() > 0, f"[{arm}] the seam returned nothing to read"
        readings[arm] = tuple(int(v) for v in _counter_api("topk_select")[1]())
        say("control_route", arm, "k", k, "counter", readings[arm])

    served_nki, served_fallback = readings["served"]
    refused_nki, refused_fallback = readings["refused"]
    assert served_nki > 0 and served_fallback == 0, (
        f"the served geometry did not take the kernel route: {readings['served']}. "
        f"Then this control says nothing about the counted zero"
    )
    assert refused_fallback > 0, (
        f"the refused geometry produced no torch fallback: {readings['refused']}. "
        f"The route predicate's zero is then not a reading this instrument can "
        f"move, which is what makes it a reading rather than an absence"
    )
    assert refused_nki == 0, (
        f"the refused geometry also dispatched a kernel: {readings['refused']}"
    )


# --------------------------------------------------------------------------- #
# THE THREADING CONTRACT.


def test_the_head_threads_exactly_the_blocks_own_keyword_arguments() -> None:
    """The head passes the block's keyword arguments through UNCHANGED.

    The head declares none of these names, so the contract is that whatever the
    checkpoint's layer type accepts is what a caller may pass. Read off the real
    signature rather than asserted from a list, and controlled by a planted name
    the block does not accept.
    """
    cfg = _tiny_text_config()
    block = _impl().Glm5NextDSALayer(cfg, int(cfg.num_hidden_layers), 1)
    accepted = {
        name
        for name, parameter in inspect.signature(block.forward).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    threaded = set(_step_block_kwargs(_caches(cfg), 0))
    say("threading", "accepted", sorted(accepted), "threaded", sorted(threaded))
    assert threaded, "the threaded set must not be empty"
    assert threaded <= accepted, (
        f"the head threads {sorted(threaded - accepted)}, which the checkpoint's "
        f"layer type does not accept as keyword arguments"
    )

    head_kwargs = {
        name
        for name, parameter in inspect.signature(_mtp().Glm5NextMTPLayer.forward).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    say("threading", "head_own", sorted(head_kwargs))
    assert not (head_kwargs & accepted), (
        f"the head names {sorted(head_kwargs & accepted)}, which belong to the "
        f"block; a shadowed name would be consumed instead of threaded"
    )
    assert "not_a_block_kwarg" not in accepted, "the control name must not be accepted"
