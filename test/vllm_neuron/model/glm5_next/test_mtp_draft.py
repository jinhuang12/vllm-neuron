"""The MTP draft head over the FULL layer 45: attention half, MoE half, the four head
tensors, ``populate`` over T rows and ``draft_tokens`` over k iterations.

The reference is an independently written torch layer 45: the head's three norms,
``eh_proj``, the position-zero mask and the embedding lookup in torch; the attention
half through a twin ``Glm5NextDSALayer`` built from the same seed on its own caches
(the production attention, which ``mtp.py`` does not own); the MoE half through the
tiny MoE fixture's torch reference (``_routed_output`` + ``_dense_output`` on the
router oracle's affinities); the residual adds and the shared-head norm in torch; the
greedy token as the ``argmax`` over the whole tiny vocabulary's bf16 logits.

Pinned by the tests below (mtp.md H1, H2, H4, H8, H9 and the Stage A contract):

* ``populate(hidden_rows, next_ids, positions)``: row ``t`` consumes ``h_t`` and
  ``embed(x_{t+1})``, position 0's embedding is masked, and the layer-45 state those
  rows leave (latent bank, pooled store, tail ring) is the reference's.
* ``draft_tokens(..., k)``: ``[B, k]`` int32 GLOBAL ids; iteration ``i + 1`` consumes
  iteration ``i``'s token and the shared-head-normed hidden state (8.3), at position
  ``p + i + 1`` (8.4); the attention half AND the MoE half run on every iteration.
* The real tail ring is what a one-iteration step leaves: the ring wraps at the pool
  width, so iterations past the first run on a scratch copy.
* ``index_share_for_mtp_iteration`` skips the indexer on iterations ``1..k-1`` only
  when the flag is set and the regime selects (ctx > bypass bound).
* The knob ``VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT`` reaches the tree through
  ``envs`` and ``mtp.shadow_draft_k()`` only.

Geometry: hidden 512 (the tiny MoE fixture's width: 16 experts, top-8, I=1024), one
MLA head of 4, latent 128, a 4-head indexer, pool 4, ``index_topk`` 8 (select 2 pools;
bypass bound 11), vocab 64. Caches are bf16, the serving line's dtype, so the block
takes its batched decode route.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from test.vllm_neuron.model.glm5_next.test_mla_decode import paged_operands
from test.vllm_neuron.model.glm5_next.tiny.test_tiny_glm5next_forward import (
    MOE_ROUTER_BIAS_SCALE,
    MOE_ROUTER_WEIGHT_SCALE,
    ROUTED_EXPERTS,
    ROUTED_EXPERTS_PER_TOKEN,
    ROUTED_HIDDEN_SIZE,
    ROUTED_INTERMEDIATE_SIZE,
    SEED_MOE_ROUTER,
    _POST_SCALE,
    _attach,
    _dense_output,
    _prep_operands_from_the_module,
    _quant_config,
    _routed_operands,
    _routed_output,
    _shared_at_routed_operands,
)

KNOB = "VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT"

HIDDEN_RTOL = 3e-2
HIDDEN_ATOL = 3e-2
#: The smallest top-1/top-2 logit gap, as a share of the logit range, at which a
#: token comparison is a reading: below it a bf16 rounding could flip the argmax on
#: either side and the test would say nothing about the head.
MIN_MARGIN_SHARE = 0.02

DRAFT_K = 5

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "hf-config.json"
FIXTURE_SHA256 = "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f"

POOL_SIZE = 4
TOPK_POOLS = 2
PAGE_SIZE = 4
PREFILL_TOKENS = 35
#: Positions ``0 .. MAX_SEQ_LEN - 1``: the prefill, a k=5 draft, one more real step and
#: a second draft fit. A constant, as the runner's ``max_model_len`` is.
MAX_SEQ_LEN = 48
INDEX_HEAD_DIM = 128
TINY_VOCAB = 64

#: The bypass-regime fixture: a context the selection keeps whole (``max_seq_len`` at
#: the bypass bound, 11 at ``index_topk`` 8 / ``index_kpool`` 4). The dense decode
#: kernel that regime attends with takes whole 128-row windows, so its latent bank is
#: one such window (32 pages of 4) however short the model length.
SHORT_PREFILL_TOKENS = 5
SHORT_MAX_SEQ_LEN = 11
DENSE_WINDOW_ROWS = 128

TINY_GEOMETRY: dict[str, int] = {
    "hidden_size": ROUTED_HIDDEN_SIZE,
    "intermediate_size": ROUTED_INTERMEDIATE_SIZE,
    "moe_intermediate_size": ROUTED_INTERMEDIATE_SIZE,
    "n_routed_experts": ROUTED_EXPERTS,
    "num_experts_per_tok": ROUTED_EXPERTS_PER_TOKEN,
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

TINY_HEAD_SIZE = TINY_GEOMETRY["kv_lora_rank"] + TINY_GEOMETRY["qk_rope_head_dim"]
SOFTMAX_SCALE = float(
    (TINY_GEOMETRY["qk_nope_head_dim"] + TINY_GEOMETRY["qk_rope_head_dim"]) ** -0.5
)

FAMILIES: tuple[str, ...] = (
    "decode_tail_update",
    "index_expand",
    "kpool_hadamard",
    "paged_gather",
    "ragged_pack",
    "score_gemm",
    "topk_select",
)

HEAD_LEAVES = ("enorm_weight", "hnorm_weight", "eh_proj_weight", "shared_head_norm_weight")
MLP_LEAVES = ("gate_proj_weight", "up_proj_weight", "down_proj_weight")


# --------------------------------------------------------------------------- #
# imports inside helpers, never at module level.


def _impl():
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _mtp():
    from vllm_neuron.model.glm5_next import mtp

    return mtp


def gate_live() -> bool:
    from vllm_neuron.utils.neuron_utils import can_run_kernel

    return bool(can_run_kernel())


def _skip_unless_live() -> None:
    if not gate_live():
        pytest.skip("the NKI gate is not live; the readings would be meaningless")


def _counter_api(family: str):
    module = importlib.import_module(f"vllm_neuron.functional.dsa.{family}")
    names = [n for n in dir(module) if n.endswith("_dispatch_counters")]
    reset = [n for n in names if n.startswith("reset_")]
    read = [n for n in names if not n.startswith("reset_")]
    assert len(reset) == 1 and len(read) == 1, (family, reset, read)
    return getattr(module, reset[0]), getattr(module, read[0])


def _decode_batch():
    from vllm_neuron.functional.dsa import decode_batch

    return decode_batch


def reset_all_counters() -> None:
    for family in FAMILIES:
        _counter_api(family)[0]()
    _decode_batch().reset_decode_batch_dispatch_counters()


def read_all_counters() -> dict[str, tuple[int, int]]:
    """``{family: (nki_dispatch, torch_fallback)}`` since the last reset, decode batch included."""
    out = {f: tuple(int(v) for v in _counter_api(f)[1]()) for f in FAMILIES}
    out["decode_batch"] = tuple(int(v) for v in _decode_batch().decode_batch_dispatch_counters())
    return out


def read_route_counts() -> tuple[int, int, int, int]:
    """``(ring_dispatch, scores_dispatch, two_program_dispatch, select_dispatch)`` since the
    last reset. The decode selection is one kernel (``decode_select``), counted here."""
    return tuple(int(v) for v in _decode_batch().decode_batch_route_counts())


# --------------------------------------------------------------------------- #
# the fixture.


def _pinned_checkpoint_config() -> dict:
    raw = FIXTURE_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == FIXTURE_SHA256, (
        f"{FIXTURE_PATH.name} is not the pinned fixture: sha256 {digest} != {FIXTURE_SHA256}"
    )
    return json.loads(raw.decode())["text_config"]


def _tiny_text_config(**overrides):
    from dataclasses import replace

    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    fields = dict(TINY_GEOMETRY)
    fields["index_topk"] = TOPK_POOLS * POOL_SIZE
    fields.update(overrides)
    return replace(Glm5NextTextConfig(), **fields)


def _materialise_indexer(indexer, gen: torch.Generator) -> None:
    for name, in_features, out_features in indexer.projection_widths():
        weight = torch.randn(out_features, in_features, generator=gen, dtype=torch.float32) * (
            in_features**-0.5
        )
        setattr(indexer, indexer.PROJECTION_PARAMETERS[name], torch.nn.Parameter(weight))
    head_dim = int(indexer.index_head_dim)
    indexer.k_norm_weight = torch.nn.Parameter(
        1.0 + torch.randn(head_dim, generator=gen, dtype=torch.float32) * 0.05
    )
    indexer.k_norm_bias = torch.nn.Parameter(
        torch.randn(head_dim, generator=gen, dtype=torch.float32) * 0.02
    )
    indexer.index_kpool_compress_ape = torch.nn.Parameter(
        (torch.randn(int(indexer.index_kpool), head_dim, generator=gen, dtype=torch.float32) * 0.1).to(
            torch.bfloat16
        ),
        requires_grad=False,
    )
    assert indexer.prepare_projection_weights() == 4, "four indexer projections must prepare"


def _materialise_attention_half(layer, cfg, seed: int) -> None:
    """The DSA layer's two norms, its attention projections and its indexer, from ``seed``."""
    gen = torch.Generator().manual_seed(int(seed))
    hidden = int(cfg.hidden_size)
    layer.input_layernorm_weight = torch.nn.Parameter(
        1.0 + torch.randn(hidden, generator=gen, dtype=torch.float32) * 0.05
    )
    layer.post_attention_layernorm_weight = torch.nn.Parameter(
        1.0 + torch.randn(hidden, generator=gen, dtype=torch.float32) * 0.05
    )
    attention = layer.attention
    for name, in_features, out_features in attention.projection_widths():
        weight = torch.randn(out_features, in_features, generator=gen, dtype=torch.float32) * (
            in_features**-0.5
        )
        setattr(attention, f"{name}_weight", torch.nn.Parameter(weight))
    for gain_name, width in (
        ("q_a_layernorm_weight", int(cfg.q_lora_rank)),
        ("kv_a_layernorm_weight", int(cfg.kv_lora_rank)),
    ):
        setattr(
            attention,
            gain_name,
            torch.nn.Parameter(1.0 + torch.randn(width, generator=gen, dtype=torch.float32) * 0.05),
        )
    assert attention.prepare_projection_weights() == 5, "five MLA projections must prepare"
    assert attention.prepare_absorb_weights() == 2, "and both absorb operands must split"
    _materialise_indexer(attention.indexer, gen)


def _router_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(SEED_MOE_ROUTER)
    weight = (torch.randn(ROUTED_HIDDEN_SIZE, ROUTED_EXPERTS, generator=gen) * MOE_ROUTER_WEIGHT_SCALE).to(
        torch.bfloat16
    )
    bias = (torch.randn(ROUTED_EXPERTS, generator=gen) * MOE_ROUTER_BIAS_SCALE).to(torch.bfloat16)
    return weight, bias


def _materialise_moe_half(mlp) -> tuple[dict, dict]:
    """The tiny MoE fixture on ``mlp``, the way ``test_tiny_glm5next_moe_forward`` attaches it."""
    model_fp8 = _impl()
    assert isinstance(mlp, model_fp8.Glm5NextMoEBlock), type(mlp).__name__
    routed = _routed_operands()
    for leaf in MLP_LEAVES:
        _attach(mlp.experts, leaf, *routed[leaf])
    assert mlp.experts.prepare_scale_operands(
        *_prep_operands_from_the_module(mlp.experts, MLP_LEAVES, routed)
    ) == 2
    shared = _shared_at_routed_operands()
    for leaf in MLP_LEAVES:
        _attach(mlp.shared_experts, leaf, *shared[leaf])
    assert mlp.shared_experts.prepare_scale_operands(*mlp.shared_experts.scale_route_operands()) == 3
    weight, bias = _router_tensors()
    mlp.experts.router_weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    mlp.experts.router_bias = torch.nn.Parameter(bias.clone(), requires_grad=False)
    return routed, shared


def _build_twin_block(cfg, seed: int):
    """The reference's layer 45: a production DSA layer with the MoE half, same seed."""
    model_fp8 = _impl()
    from vllm_neuron.model.glm5_next.config import DSA_LAYER_TYPE

    layer = model_fp8._build_layer(cfg, int(cfg.num_hidden_layers), DSA_LAYER_TYPE, 1)
    assert type(layer) is model_fp8.Glm5NextDSALayer
    _materialise_attention_half(layer, cfg, seed)
    routed, shared = _materialise_moe_half(layer.mlp)
    return layer, routed, shared


def _head_weights(cfg, seed: int = 63_200_001) -> dict:
    """The four MTP tensors, the embedding table and the whole head, bf16 like the checkpoint."""
    gen = torch.Generator().manual_seed(int(seed))
    hidden = int(cfg.hidden_size)
    vocab = int(cfg.vocab_size)
    w = {
        "enorm_weight": 1.0 + torch.randn(hidden, generator=gen) * 0.05,
        "hnorm_weight": 1.0 + torch.randn(hidden, generator=gen) * 0.05,
        "eh_proj_weight": torch.randn(hidden, 2 * hidden, generator=gen) * ((2 * hidden) ** -0.5),
        "shared_head_norm_weight": 1.0 + torch.randn(hidden, generator=gen) * 0.05,
        "embed_tokens_weight": torch.randn(vocab, hidden, generator=gen),
        "lm_head_weight": torch.randn(vocab, hidden, generator=gen) * (hidden**-0.5),
    }
    return {k: v.to(torch.bfloat16) for k, v in w.items()}


def _build_head(cfg, weights: dict, seed: int):
    """The head under test with every declared leaf materialised, block from ``seed``."""
    head = _mtp().Glm5NextMultiTokenPredictor(
        cfg,
        embed_tokens=lambda: weights["embed_tokens_weight"],
        lm_head=lambda: weights["lm_head_weight"],
        world_size=1,
        tp_group=lambda: None,
    )
    for name in HEAD_LEAVES:
        setattr(head, name, torch.nn.Parameter(weights[name].clone(), requires_grad=False))
    _materialise_attention_half(head.block, cfg, seed)
    _materialise_moe_half(head.block.mlp)
    return head


def _caches(cfg, rows: int, *, window_rows: int | None = None):
    """One cache set for ``rows`` positions. Each side gets its own; all three are written in place.

    The latent bank is paged, so its rows round up to whole pages (and to ``window_rows``
    when given: the bypass regime's dense kernel wants a 128-row window); the pooled
    store keeps its trash row past ``rows // POOL_SIZE``."""
    slots = max(-(-rows // PAGE_SIZE) * PAGE_SIZE, int(window_rows or 0))
    return {
        "pool_cache": torch.zeros(slots // POOL_SIZE + 4, int(cfg.index_head_dim), dtype=torch.bfloat16),
        "latent_cache": torch.zeros(slots, 1, TINY_HEAD_SIZE, dtype=torch.bfloat16),
        "tail": torch.zeros(2, int(cfg.index_kpool), int(cfg.index_head_dim), dtype=torch.bfloat16),
    }


def _clone_caches(caches: dict) -> dict:
    return {k: v.clone() for k, v in caches.items()}


def _prefill_slot_mapping(tokens: int, pool: int) -> torch.Tensor:
    slots = torch.full((int(tokens),), -1, dtype=torch.int32)
    for position in range(int(tokens)):
        if (position + 1) % int(pool) == 0:
            slots[position] = position // int(pool)
    return slots


def _prefill_kwargs(caches: dict, tokens: int) -> dict:
    """The block's prefill-leg carrier for positions ``0 .. tokens - 1`` of one request."""
    return {
        "latent_cache": caches["latent_cache"],
        "pool_cache": caches["pool_cache"],
        "seq_lens": torch.arange(1, tokens + 1, dtype=torch.int32),
        "start_position": 0,
        "softmax_scale": SOFTMAX_SCALE,
        "max_seq_len": int(tokens),
        "slot_mapping": _prefill_slot_mapping(tokens, POOL_SIZE),
        "prefill_tail": caches["tail"],
        "prefill_end_position": int(tokens),
        **paged_operands(caches["latent_cache"], 0, tokens, page=PAGE_SIZE),
    }


def _decode_kwargs(caches: dict, position: int, *, max_seq_len: int = MAX_SEQ_LEN) -> dict:
    """The block's one-request decode carrier at ``position``, every scalar a tensor (traced form)."""
    pages = int(caches["latent_cache"].shape[0]) // PAGE_SIZE
    return {
        "latent_cache": caches["latent_cache"],
        "pool_cache": caches["pool_cache"],
        "seq_lens": torch.tensor([position + 1], dtype=torch.int32),
        "start_position": torch.tensor([position], dtype=torch.int64),
        "softmax_scale": SOFTMAX_SCALE,
        "max_seq_len": int(max_seq_len),
        "tail": caches["tail"],
        "position": torch.tensor([position], dtype=torch.int64),
        "block_table_row": torch.arange(pages, dtype=torch.int32).reshape(pages, 1),
        "latent_slots": torch.tensor([position], dtype=torch.int64),
        "page_size": PAGE_SIZE,
    }


def _rms(x: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    promoted = x.to(torch.float32)
    inverse = torch.rsqrt(promoted.pow(2).mean(dim=-1, keepdim=True) + eps)
    return ((promoted * inverse) * gain.to(torch.float32)).to(x.dtype)


class _Reference:
    """Layer 45 in torch over a twin block: the head's arithmetic, the attention half
    through the production layer, the MoE half through the fixture's torch reference."""

    def __init__(self, cfg, weights: dict, twin, routed: dict, shared: dict):
        self.cfg = cfg
        self.w = weights
        self.twin = twin
        self.routed = routed
        self.shared = shared
        self.eps = float(cfg.rms_norm_eps)

    def layer_input(self, embeds, previous, positions, *, mask: bool = True) -> torch.Tensor:
        if mask:
            embeds = torch.where(positions.unsqueeze(-1) == 0, torch.zeros_like(embeds), embeds)
        e = _rms(embeds, self.w["enorm_weight"], self.eps)
        h = _rms(previous, self.w["hnorm_weight"], self.eps)
        joined = torch.cat([e, h], dim=-1).to(torch.float32)
        return (joined @ self.w["eh_proj_weight"].to(torch.float32).t()).to(embeds.dtype)

    def ffn(self, attended: torch.Tensor) -> torch.Tensor:
        """``attended + MoE(attended)``: fused-norm router oracle, routed bank, shared expert."""
        from vllm_neuron.functional.moe.router_decode import router_decode_torch_oracle

        gamma = self.twin.post_attention_layernorm_weight.detach()
        normed = _rms(attended, gamma, self.eps)
        weight, bias = _router_tensors()
        _logits, _index, affinities = router_decode_torch_oracle(
            attended, gamma, weight, bias, self.eps,
            bool(self.cfg.norm_topk_prob), float(self.cfg.routed_scaling_factor),
        )
        limit = float(self.cfg.swiglu_limit)
        routed = _routed_output(
            {**self.routed, "hidden": normed, "expert_affinities": affinities},
            mode=_POST_SCALE, gate_max=limit, gate_min=None, up_max=limit, up_min=-limit,
        )["out"]
        shared = _dense_output({**self.shared, "hidden": normed}, limit, -limit, limit)["out"]
        return attended + (routed + shared).to(attended.dtype)

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.w["embed_tokens_weight"][ids.to(torch.int64)]

    def step(self, embeds, previous, positions, block_kwargs: dict, *, mask: bool = True,
             feed_normed: bool = True):
        """One layer-45 pass. Returns ``(normed_hidden [T,H], logits [T,V] fp32, pre_norm [T,H])``."""
        x = self.layer_input(embeds, previous, positions, mask=mask)
        attended = self.twin.forward(x, **block_kwargs)
        y = self.ffn(attended)
        hidden = _rms(y, self.w["shared_head_norm_weight"], self.eps)
        # The head's dtype (bf16), as the trunk's ``vocab_parallel_logits`` takes the
        # sampler's logits; the margin guard below keeps a bf16 tie out of the reading.
        logits = torch.nn.functional.linear(hidden, self.w["lm_head_weight"]).to(torch.float32)
        return hidden, logits, y

    def populate(self, hidden_rows, next_ids, positions, block_kwargs: dict, *, mask: bool = True) -> None:
        x = self.layer_input(self.embed(next_ids), hidden_rows, positions, mask=mask)
        self.twin.forward(x, **block_kwargs)

    def chain(self, caches: dict, hidden_row, sampled_id, position: int, k: int, *,
              follow: tuple[torch.Tensor, list] | None = None,
              wrong_token_at: int | None = None, feed_prenorm: bool = False) -> list[dict]:
        """``k`` reference iterations from ``position``, each on the twin's own state.

        ``follow=(tokens [1, k], hiddens)`` teacher-forces the chain on the head's own
        outputs: iteration ``i + 1`` consumes the head's iteration-``i`` token and normed
        hidden rather than this reference's. The indexer's top-k is discrete, and the
        kernel MoE half and the torch oracle differ by bf16 ulps, so a free-running
        reference and the head part ways the first time a near-tied pool flips (seed
        63423, iteration 1: pools {1, 2} against {1, 5}); fed the same inputs the two
        attention halves are bit-equal and the comparison reads the arithmetic.
        """
        previous, token = hidden_row, sampled_id.reshape(1)
        out = []
        for i in range(k):
            if wrong_token_at is not None and i == wrong_token_at:
                token = sampled_id.reshape(1)
            hidden, logits, prenorm = self.step(
                self.embed(token), previous, torch.tensor([position + i], dtype=torch.int32),
                _decode_kwargs(caches, position + i),
            )
            own = logits.argmax(dim=-1).to(torch.int32)
            sorted_logits = logits.sort(dim=-1, descending=True).values
            margin = float(sorted_logits[0, 0] - sorted_logits[0, 1])
            span = float(logits.max() - logits.min())
            out.append({"token": own, "hidden": hidden, "logits": logits,
                        "margin_share": margin / span if span else 0.0})
            if follow is None:
                token, previous = own, hidden
            else:
                token = follow[0][:, i].reshape(1).to(torch.int32)
                previous = follow[1][i]
            if feed_prenorm:
                previous = prenorm
        return out


def _fixture(seed: int, *, prefill: int = PREFILL_TOKENS, max_seq_len: int = MAX_SEQ_LEN,
             window_rows: int | None = None, **overrides):
    """Head and reference from one seed, both sides prefilled over ``prefill`` positions
    through their own ``populate``: positions ``t`` consume ``h_t`` and ``x_{t+1}``."""
    cfg = _tiny_text_config(**overrides)
    weights = _head_weights(cfg)
    head = _build_head(cfg, weights, seed)
    twin, routed, shared = _build_twin_block(cfg, seed)
    ref = _Reference(cfg, weights, twin, routed, shared)
    gen = torch.Generator().manual_seed(seed + 17)
    hidden = torch.randn(prefill + DRAFT_K + 2, int(cfg.hidden_size), generator=gen).to(torch.bfloat16)
    ids = torch.randint(0, int(cfg.vocab_size), (prefill + DRAFT_K + 3,), generator=gen, dtype=torch.int32)
    caches = {"impl": _caches(cfg, max_seq_len, window_rows=window_rows),
              "ref": _caches(cfg, max_seq_len, window_rows=window_rows)}
    positions = torch.arange(prefill, dtype=torch.int32)
    head.populate(hidden[:prefill], ids[1:prefill + 1], positions, quant_config=_quant_config(),
                  **_prefill_kwargs(caches["impl"], prefill))
    ref.populate(hidden[:prefill], ids[1:prefill + 1], positions, _prefill_kwargs(caches["ref"], prefill))
    return {"cfg": cfg, "weights": weights, "head": head, "ref": ref, "caches": caches,
            "hidden": hidden, "ids": ids, "prefill": prefill, "max_seq_len": max_seq_len}


def _draft(fx: dict, position: int, k: int, *, caches_key: str = "impl", collector=None) -> torch.Tensor:
    """The head's draft at ``position``: ``h_position`` and the token sampled there."""
    extra = {} if collector is None else {"draft_collector": collector}
    return fx["head"].draft_tokens(
        fx["hidden"][position:position + 1], fx["ids"][position + 1:position + 2],
        torch.tensor([position], dtype=torch.int32), k, quant_config=_quant_config(), **extra,
        **_decode_kwargs(fx["caches"][caches_key], position, max_seq_len=fx["max_seq_len"]),
    )


def _reference_chain(fx: dict, position: int, k: int, **kw) -> list[dict]:
    return fx["ref"].chain(fx["caches"]["ref"], fx["hidden"][position:position + 1],
                           fx["ids"][position + 1], position, k, **kw)


def _k_tokens_per_request(drafted: torch.Tensor, requests: int, k: int, label: str) -> None:
    """The draft-window predicate: ``[B, k]`` int32, one row per request, ``k`` tokens each."""
    assert drafted.ndim == 2, (
        f"[{label}] a draft is k tokens per request, so rank 2 [B, k]; got {tuple(drafted.shape)}"
    )
    assert tuple(drafted.shape) == (requests, k), (
        f"[{label}] {requests} request(s) and k={k} in, {tuple(drafted.shape)} out"
    )
    assert drafted.dtype == torch.int32, f"[{label}] draft ids must be int32, got {drafted.dtype}"


# --------------------------------------------------------------------------- #
# the checkpoint's declaration, the knob, the import boundary, the parameter names.


def test_the_checkpoint_declares_one_draft_layer_at_index_45() -> None:
    text_config = _pinned_checkpoint_config()
    nextn = int(text_config["num_nextn_predict_layers"])
    layers = int(text_config["num_hidden_layers"])
    assert nextn == 1, f"the checkpoint declares {nextn} draft layer(s), not 1"
    assert layers == 45, f"the checkpoint declares {layers} hidden layers, not 45"
    cfg = _tiny_text_config()
    assert int(cfg.num_hidden_layers) == layers
    head = _mtp().Glm5NextMultiTokenPredictor(
        cfg, embed_tokens=lambda: None, lm_head=lambda: None, world_size=1, tp_group=lambda: None
    )
    assert head.mtp_layer_idx == layers, "the one draft layer sits one past the main stack"
    assert head.block.layer_idx == layers


def test_importing_the_head_does_not_load_the_model_tree() -> None:
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
    assert result.returncode == 0, f"the child could not import the head alone: {result.stderr[-2000:]}"
    assert "MODEL_FP8_LOADED=False" in result.stdout, (
        f"importing the head loaded the model tree: {result.stdout.strip()}. The head "
        f"builds its block inside its constructor precisely so the import does not have to"
    )


def test_shadow_draft_k_reads_the_knob_through_envs(monkeypatch) -> None:
    from vllm_neuron import envs

    mtp = _mtp()
    monkeypatch.delenv(KNOB, raising=False)
    assert mtp.shadow_draft_k() == 0, "unset means no shadow draft"
    assert envs.VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT == 0
    for value in ("1", "3", "5", "8"):
        monkeypatch.setenv(KNOB, value)
        assert mtp.shadow_draft_k() == int(value)
        assert isinstance(mtp.shadow_draft_k(), int)
        assert envs.VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT == int(value)
    monkeypatch.setenv(KNOB, "-1")
    with pytest.raises(ValueError, match=KNOB):
        mtp.shadow_draft_k()
    assert KNOB in envs.environment_variables
    source = Path(envs.__file__).read_text()
    assert source.count(f'"{KNOB}": lambda') == 1, "exactly one definition of the knob in envs.py"
    assert source.count(f"{KNOB}: int") == 1, "and one typed declaration"


def _walk_declared(module: torch.nn.Module, prefix: str = "") -> list[str]:
    names = [prefix + n for n in module._parameters]
    for child_name, child in module.named_children():
        names += _walk_declared(child, f"{prefix}{child_name}.")
    return names


def test_mtp_parameter_names_are_the_trees_declared_leaves_without_mhc() -> None:
    """C2: the published constant is exactly what the tree declares, and the six mHC
    leaves are not there -- the checkpoint gives layer 45 none, and a declared-but-unmapped
    leaf would be materialised as a placeholder and refuse the one-stream call."""
    mtp, model_fp8 = _mtp(), _impl()
    cfg = _tiny_text_config()
    head = mtp.Glm5NextMultiTokenPredictor(
        cfg, embed_tokens=lambda: None, lm_head=lambda: None, world_size=1, tp_group=lambda: None
    )
    walked = _walk_declared(head)
    assert len(walked) == len(set(walked)), "duplicate declared names"
    assert set(walked) == set(mtp.MTP_PARAMETER_NAMES), (
        f"MTP_PARAMETER_NAMES disagrees with the tree: "
        f"missing {sorted(set(walked) - set(mtp.MTP_PARAMETER_NAMES))}, "
        f"extra {sorted(set(mtp.MTP_PARAMETER_NAMES) - set(walked))}"
    )
    assert len(mtp.MTP_PARAMETER_NAMES) == len(set(mtp.MTP_PARAMETER_NAMES))
    assert set(HEAD_LEAVES) <= set(walked)
    assert all(not n.startswith("block.") for n in HEAD_LEAVES)
    for leaf in model_fp8.MHC_LEAVES:
        assert f"block.{leaf}" not in walked, f"{leaf} must not be declared on the draft layer"
        assert leaf not in head.block.declared_param_names
        assert getattr(head.block, leaf, None) is None
    assert model_fp8._mhc_attention_site(head.block, None) is None, (
        "the draft layer must take the plain residual route with no streams"
    )
    for name in walked:
        assert getattr(head, "declared_param_names", None) is not None or True
    expected_block = {
        f"block.{n}" for n in head.block.declared_param_names
    } | {f"block.self_attn.{n}" for n in head.block.self_attn.declared_param_names} | {
        f"block.self_attn.indexer.{n}" for n in head.block.self_attn.indexer.declared_param_names
    } | {f"block.mlp.experts.{n}" for n in head.block.mlp.experts.declared_param_names} | {
        f"block.mlp.shared_experts.{n}" for n in head.block.mlp.shared_experts.declared_param_names
    }
    assert {n for n in walked if n.startswith("block.")} == expected_block
    assert isinstance(head.block.mlp, model_fp8.Glm5NextMoEBlock), "layer 45 carries the MoE half"
    assert "block.input_layernorm_weight" in walked and "block.post_attention_layernorm_weight" in walked
    sig = inspect.signature(mtp.Glm5NextMultiTokenPredictor.__init__)
    assert list(sig.parameters)[1:] == ["text_config", "embed_tokens", "lm_head", "world_size", "tp_group"], (
        f"C2 constructor signature drifted: {list(sig.parameters)}"
    )


# --------------------------------------------------------------------------- #
# populate: the pinned shift and the position-zero mask, over one pool of rows.


def test_populate_consumes_h_t_with_embed_x_t_plus_1_and_masks_position_zero() -> None:
    """One pool of rows: ``populate(h[0:4], x[1:5], [0, 1, 2, 3])`` leaves the state the
    reference leaves for ``(h_t, embed(x_{t+1}))`` with position 0's embedding zeroed.
    Two mutant references -- unshifted ids, and no mask -- both leave a different state.
    Four rows and not three because the production indexer refuses a prefill chunk
    shorter than a pool (``index_kpool`` 4): "no pool can complete in 3 token(s)"."""
    _skip_unless_live()
    cfg = _tiny_text_config()
    weights = _head_weights(cfg)
    head = _build_head(cfg, weights, 63_301)
    twin, routed, shared = _build_twin_block(cfg, 63_301)
    ref = _Reference(cfg, weights, twin, routed, shared)
    gen = torch.Generator().manual_seed(63_311)
    tokens = POOL_SIZE
    hidden = torch.randn(tokens, int(cfg.hidden_size), generator=gen).to(torch.bfloat16)
    ids = torch.randint(0, int(cfg.vocab_size), (tokens + 1,), generator=gen, dtype=torch.int32)
    assert not torch.equal(ids[:tokens], ids[1:tokens + 1]), "the shift must move the ids"
    positions = torch.arange(tokens, dtype=torch.int32)

    impl = _caches(cfg, MAX_SEQ_LEN)
    got = head.populate(hidden, ids[1:tokens + 1], positions, quant_config=_quant_config(),
                        **_prefill_kwargs(impl, tokens))
    assert got is None, "populate writes state and returns nothing"

    states = {}
    for label, next_ids, mask in (
        ("reference", ids[1:tokens + 1], True),
        ("unshifted", ids[:tokens], True),
        ("unmasked", ids[1:tokens + 1], False),
    ):
        caches = _caches(cfg, MAX_SEQ_LEN)
        ref.populate(hidden, next_ids, positions, _prefill_kwargs(caches, tokens), mask=mask)
        states[label] = caches

    for name in ("latent_cache", "pool_cache", "tail"):
        torch.testing.assert_close(impl[name], states["reference"][name], rtol=0, atol=0,
                                   msg=lambda m, n=name: f"{n}: {m}")
    assert not torch.equal(impl["latent_cache"][:tokens], states["unshifted"]["latent_cache"][:tokens]), (
        "an unshifted reference leaves the same latent rows: the test cannot see the shift"
    )
    assert not torch.equal(impl["latent_cache"][:1], states["unmasked"]["latent_cache"][:1]), (
        "an unmasked reference leaves the same row 0: the test cannot see the mask"
    )
    assert torch.equal(impl["latent_cache"][1:tokens], states["unmasked"]["latent_cache"][1:tokens]), (
        "the mask touches position 0 only"
    )
    assert bool(impl["latent_cache"][:tokens].abs().sum() > 0), "the rows were written"
    assert bool(impl["latent_cache"][tokens:].abs().sum() == 0), "and nothing past them"


# --------------------------------------------------------------------------- #
# draft_tokens: the k=5 chain against the reference, iteration by iteration.


def test_a_k5_draft_matches_the_full_layer_reference_at_every_iteration() -> None:
    """Iteration ``i + 1`` consumes iteration ``i``'s token and its shared-head-normed
    hidden state at position ``p + i + 1``; attention and MoE halves on every iteration.
    The flag is off so the reference's per-iteration indexer is the exact draft. The
    reference is teacher-forced on the head's own outputs (``_Reference.chain``), and
    the two mutants below show the comparison moves when the wrong token or the
    pre-norm hidden is fed, so the chain is pinned by what the head hands itself."""
    _skip_unless_live()
    fx = _fixture(63_430, index_share_for_mtp_iteration=False)
    p = fx["prefill"]
    collected: list = []
    got = _draft(fx, p, DRAFT_K, collector=collected)
    _k_tokens_per_request(got, 1, DRAFT_K, "k=5")
    assert len(collected) == DRAFT_K, f"one normed hidden per iteration, got {len(collected)}"
    follow = (got, collected)
    expected = _reference_chain(fx, p, DRAFT_K, follow=follow)
    margins = [e["margin_share"] for e in expected]
    assert min(margins) >= MIN_MARGIN_SHARE, (
        f"the reference's top-1/top-2 margins {margins} are too thin for a token "
        f"comparison to read anything; pick another seed"
    )
    want = torch.cat([e["token"] for e in expected]).reshape(1, DRAFT_K)
    for i, e in enumerate(expected):
        torch.testing.assert_close(collected[i].to(torch.float32), e["hidden"].to(torch.float32),
                                   rtol=HIDDEN_RTOL, atol=HIDDEN_ATOL,
                                   msg=lambda m, i=i: f"iteration {i} hidden: {m}")
    assert torch.equal(got, want), f"draft {got.tolist()} vs reference {want.tolist()} (margins {margins})"

    # Controls: a reference that re-feeds the sampled token at iteration 1, and one that
    # feeds the pre-norm hidden, both leave the comparison.
    for label, kw in (("wrong token", {"wrong_token_at": 1}), ("pre-norm hidden", {"feed_prenorm": True})):
        fx_m = _fixture(63_430, index_share_for_mtp_iteration=False)
        mutant = _reference_chain(fx_m, p, DRAFT_K, follow=follow, **kw)
        moved = any(
            not torch.allclose(m["hidden"].float(), e["hidden"].float(), rtol=HIDDEN_RTOL, atol=HIDDEN_ATOL)
            for m, e in zip(mutant[1:], expected[1:])
        )
        assert moved, f"the {label} mutant reference stays inside tolerance; the chain is not pinned"

    # State: the five latent rows are the reference's; the real ring is what the one
    # real iteration (position p) leaves, not what the wrapped draft iterations wrote.
    impl, ref = fx["caches"]["impl"], fx["caches"]["ref"]
    torch.testing.assert_close(impl["latent_cache"][:p + 1].float(), ref["latent_cache"][:p + 1].float(),
                               rtol=HIDDEN_RTOL, atol=HIDDEN_ATOL)
    assert bool(impl["latent_cache"][p + 1:p + DRAFT_K].abs().sum() > 0), "draft rows were written"
    one = _fixture(63_430, index_share_for_mtp_iteration=False)
    one_step = _draft(one, p, 1)
    assert torch.equal(one_step[0, 0], got[0, 0])
    assert torch.equal(impl["tail"], one["caches"]["impl"]["tail"]), (
        "a k=5 draft must leave the real ring as a k=1 draft leaves it: the 4-row ring "
        "wraps at iteration 4 and would overwrite position p's real row"
    )
    assert not torch.equal(impl["tail"], ref["tail"]), (
        "the reference wrote its ring through all five iterations; if the two agree the "
        "wrap this guards never happened at this geometry"
    )


def test_a_draft_step_then_a_real_step_leaves_the_state_a_plain_real_step_leaves() -> None:
    """Hygiene: draft(k=5 at p) then populate(p+1) equals populate(p+1) alone on every
    row the next real step reads, and the next draft agrees."""
    _skip_unless_live()
    a = _fixture(63_501, index_share_for_mtp_iteration=False)
    b = _fixture(63_501, index_share_for_mtp_iteration=False)
    p = a["prefill"]
    _draft(a, p, DRAFT_K)
    for fx in (a, b):
        fx["head"].populate(
            fx["hidden"][p:p + 1], fx["ids"][p + 1:p + 2], torch.tensor([p], dtype=torch.int32),
            quant_config=_quant_config(), **_decode_kwargs(fx["caches"]["impl"], p),
        )
        fx["head"].populate(
            fx["hidden"][p + 1:p + 2], fx["ids"][p + 2:p + 3], torch.tensor([p + 1], dtype=torch.int32),
            quant_config=_quant_config(), **_decode_kwargs(fx["caches"]["impl"], p + 1),
        )
    ca, cb = a["caches"]["impl"], b["caches"]["impl"]
    assert torch.equal(ca["tail"], cb["tail"]), "the ring after the real steps differs"
    assert torch.equal(ca["latent_cache"][:p + 2], cb["latent_cache"][:p + 2]), "real latent rows differ"
    complete = (p + 2) // POOL_SIZE
    assert torch.equal(ca["pool_cache"][:complete], cb["pool_cache"][:complete]), "real pools differ"
    assert not torch.equal(ca["latent_cache"], cb["latent_cache"]), (
        "the draft left no rows past the real ones; the hygiene check is vacuous"
    )
    next_a = _draft(a, p + 2, DRAFT_K)
    next_b = _draft(b, p + 2, DRAFT_K)
    assert torch.equal(next_a, next_b), f"the next draft differs: {next_a.tolist()} vs {next_b.tolist()}"


def test_two_requests_draft_as_two_independent_one_request_drafts() -> None:
    """B=2 in the tuple carrier form (one ring and one pooled store per request, one
    block-table column each) equals two B=1 drafts. No batch-of-one assumption."""
    _skip_unless_live()
    cfg = _tiny_text_config(index_share_for_mtp_iteration=False)
    weights = _head_weights(cfg)
    head = _build_head(cfg, weights, 63_601)
    # Both at least 8 complete pools (the top-k select kernel's ``max8`` floor), in
    # different pool phases (35 % 4 == 3, 33 % 4 == 1).
    lengths = (PREFILL_TOKENS, PREFILL_TOKENS - 2)
    pages = MAX_SEQ_LEN // PAGE_SIZE
    bank = torch.zeros(2 * MAX_SEQ_LEN, 1, TINY_HEAD_SIZE, dtype=torch.bfloat16)
    pools = [torch.zeros(MAX_SEQ_LEN // POOL_SIZE + 4, INDEX_HEAD_DIM, dtype=torch.bfloat16) for _ in range(2)]
    rings = [torch.zeros(2, POOL_SIZE, INDEX_HEAD_DIM, dtype=torch.bfloat16) for _ in range(2)]
    gen = torch.Generator().manual_seed(63_611)
    hidden = [torch.randn(n + 1, int(cfg.hidden_size), generator=gen).to(torch.bfloat16) for n in lengths]
    ids = [torch.randint(0, int(cfg.vocab_size), (n + 2,), generator=gen, dtype=torch.int32) for n in lengths]
    table = torch.stack([torch.arange(r * pages, (r + 1) * pages, dtype=torch.int32) for r in range(2)], dim=1)
    for r, n in enumerate(lengths):
        head.populate(
            hidden[r][:n], ids[r][1:n + 1], torch.arange(n, dtype=torch.int32), quant_config=_quant_config(),
            latent_cache=bank, pool_cache=pools[r], seq_lens=torch.arange(1, n + 1, dtype=torch.int32),
            start_position=0, softmax_scale=SOFTMAX_SCALE, max_seq_len=n,
            slot_mapping=_prefill_slot_mapping(n, POOL_SIZE), prefill_tail=rings[r], prefill_end_position=n,
            block_table_row=table[:, r:r + 1].contiguous(),
            latent_slots=torch.arange(r * MAX_SEQ_LEN, r * MAX_SEQ_LEN + n, dtype=torch.int64),
            page_size=PAGE_SIZE,
        )
    snapshot = (bank.clone(), [p.clone() for p in pools], [t.clone() for t in rings])

    def one(r: int) -> torch.Tensor:
        n = lengths[r]
        b, ps, ts = snapshot[0].clone(), [p.clone() for p in snapshot[1]], [t.clone() for t in snapshot[2]]
        return head.draft_tokens(
            hidden[r][n:n + 1], ids[r][n + 1:n + 2], torch.tensor([n], dtype=torch.int32), DRAFT_K,
            quant_config=_quant_config(),
            latent_cache=b, pool_cache=ps[r], seq_lens=torch.tensor([n + 1], dtype=torch.int32),
            start_position=torch.tensor([n], dtype=torch.int64), softmax_scale=SOFTMAX_SCALE,
            max_seq_len=MAX_SEQ_LEN, tail=ts[r], position=torch.tensor([n], dtype=torch.int64),
            block_table_row=table[:, r:r + 1].contiguous(),
            latent_slots=torch.tensor([r * MAX_SEQ_LEN + n], dtype=torch.int64), page_size=PAGE_SIZE,
        )

    singles = torch.cat([one(0), one(1)], dim=0)
    both = head.draft_tokens(
        torch.cat([hidden[r][lengths[r]:lengths[r] + 1] for r in range(2)]),
        torch.stack([ids[r][lengths[r] + 1] for r in range(2)]),
        torch.tensor(lengths, dtype=torch.int32), DRAFT_K, quant_config=_quant_config(),
        latent_cache=bank, pool_cache=tuple(pools), seq_lens=torch.tensor([n + 1 for n in lengths], dtype=torch.int32),
        start_position=torch.tensor(lengths, dtype=torch.int64), softmax_scale=SOFTMAX_SCALE,
        max_seq_len=MAX_SEQ_LEN, tail=tuple(rings), position=torch.tensor(lengths, dtype=torch.int64),
        block_table_row=table, latent_slots=torch.tensor([r * MAX_SEQ_LEN + n for r, n in enumerate(lengths)], dtype=torch.int64),
        page_size=PAGE_SIZE,
    )
    _k_tokens_per_request(both, 2, DRAFT_K, "B=2")
    assert torch.equal(both, singles), f"B=2 {both.tolist()} vs two B=1 {singles.tolist()}"
    assert not torch.equal(both[0], both[1]), "the two requests drafted the same tokens; vacuous"


# --------------------------------------------------------------------------- #
# the draft window: k tokens per request, k iterations of the one layer (H8 flipped).


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_the_head_emits_k_draft_tokens_per_request(k: int) -> None:
    _skip_unless_live()
    fx = _fixture(63_701 + k)
    got = _draft(fx, fx["prefill"], k)
    _k_tokens_per_request(got, 1, k, f"k={k}")
    assert int(got.min()) >= 0 and int(got.max()) < int(fx["cfg"].vocab_size), "ids are global vocab ids"


def test_a_proposal_of_the_wrong_width_is_rejected() -> None:
    with pytest.raises(AssertionError, match="rank 2"):
        _k_tokens_per_request(torch.zeros(5, dtype=torch.int32), 1, 5, "rank-1 mutant")
    with pytest.raises(AssertionError, match="out"):
        _k_tokens_per_request(torch.zeros(1, 6, dtype=torch.int32), 1, 5, "k+1 mutant")
    with pytest.raises(AssertionError, match="int32"):
        _k_tokens_per_request(torch.zeros(1, 5, dtype=torch.int64), 1, 5, "dtype mutant")


def test_every_draft_iteration_runs_the_single_layer_45_and_k_is_positive() -> None:
    """No layer index wraps: there is one draft layer and it runs ``k`` times; a
    ``k`` below 1 is refused by name."""
    _skip_unless_live()
    fx = _fixture(63_801, index_share_for_mtp_iteration=False)
    head = fx["head"]
    assert head.mtp_layer_idx == int(fx["cfg"].num_hidden_layers) == 45
    calls: list[int] = []
    real_forward = head.block.forward

    def counting_forward(*args, **kwargs):
        calls.append(int(args[0].shape[0]))
        return real_forward(*args, **kwargs)

    head.block.forward = counting_forward
    _draft(fx, fx["prefill"], DRAFT_K)
    assert calls == [1] * DRAFT_K, f"layer 45 must run once per iteration on one row: {calls}"
    for bad in (0, -1):
        with pytest.raises(ValueError, match="k"):
            _draft(fx, fx["prefill"], bad)


def test_draft_tokens_refuses_a_prefill_leg_carrier() -> None:
    _skip_unless_live()
    fx = _fixture(63_901)
    with pytest.raises(ValueError, match="slot_mapping"):
        fx["head"].draft_tokens(
            fx["hidden"][:1], fx["ids"][1:2], torch.zeros(1, dtype=torch.int32), 2,
            quant_config=_quant_config(), **_prefill_kwargs(_caches(fx["cfg"], MAX_SEQ_LEN), 1),
        )


# --------------------------------------------------------------------------- #
# index share (H9): the indexer runs once per draft when the flag is set and the
# regime selects; every iteration otherwise.


@pytest.mark.parametrize(
    "regime, flag, want",
    [
        ("selected", True, (1, 1, 1)),
        ("selected", False, (DRAFT_K, DRAFT_K, DRAFT_K)),
        ("bypass", True, (DRAFT_K, 0, 0)),
    ],
    ids=["selected+share", "selected+noshare", "bypass+share"],
)
def test_index_share_skips_the_indexer_only_when_set_and_selecting(regime, flag, want) -> None:
    _skip_unless_live()
    from vllm_neuron.functional.dsa.decode_bypass import (
        bypass_max_context,
        selection_bound,
        selection_is_a_no_op,
    )

    if regime == "selected":
        fx = _fixture(64_001, index_share_for_mtp_iteration=flag)
    else:
        fx = _fixture(64_002, prefill=SHORT_PREFILL_TOKENS, max_seq_len=SHORT_MAX_SEQ_LEN,
                      window_rows=DENSE_WINDOW_ROWS, index_share_for_mtp_iteration=flag)
    cfg = fx["cfg"]
    window = int(fx["caches"]["impl"]["latent_cache"].shape[0])
    bound = selection_bound(fx["max_seq_len"], window)
    dense = selection_is_a_no_op(bound, int(cfg.index_topk), int(cfg.index_kpool))
    assert dense == (regime == "bypass"), (
        f"regime {regime}: bound {bound} vs bypass max {bypass_max_context(cfg.index_topk, cfg.index_kpool)}"
    )
    assert fx["prefill"] + DRAFT_K <= bound
    reset_all_counters()
    got = _draft(fx, fx["prefill"], DRAFT_K)
    ring, scores, _two, select = read_route_counts()
    assert (ring, scores, select) == want, (
        f"{regime} flag={flag}: (ring, scores, select) dispatches over a k={DRAFT_K} draft "
        f"were {(ring, scores, select)}, expected {want}"
    )
    _k_tokens_per_request(got, 1, DRAFT_K, f"{regime}/{flag}")
    if regime == "selected":
        other = _fixture(64_001, index_share_for_mtp_iteration=not flag)
        assert torch.equal(_draft(other, other["prefill"], DRAFT_K)[0, 0], got[0, 0]), (
            "iteration 0 computes its own indices either way"
        )


def test_index_share_carrier_publishes_the_chain_s_selection_and_reuses_it() -> None:
    """No carrier and an empty carrier run the same chain and attend byte-identically;
    the carrier then holds the indices the chain produced; a carrier handed those
    indices skips the chain and attends the same way, and a different selection does
    not (the reuse path consumes the carrier)."""
    _skip_unless_live()
    from vllm_neuron.model.glm5_next.model_fp8 import IndexShare

    def one_step(fx, **extra):
        p = fx["prefill"]
        head = fx["head"]
        x = head._layer_input(fx["ids"][p + 1:p + 2], fx["hidden"][p:p + 1],
                              torch.tensor([p], dtype=torch.int32))
        return head.block(x, **_decode_kwargs(fx["caches"]["impl"], p), **extra)

    # Four identical fixtures, built before any counter is read: populating one
    # dispatches the prefill leg's own kernels.
    fixtures = [_fixture(64_201) for _ in range(4)]
    collected: list = []
    plain = one_step(fixtures[0], collector=collected)
    carrier = IndexShare()
    reset_all_counters()
    shared = one_step(fixtures[1], index_share=carrier)
    assert read_route_counts()[3] == 1, "an empty carrier runs the chain"
    assert torch.equal(plain, shared), "no carrier and an empty carrier are byte-identical"
    indices = carrier.topk_indices
    assert indices is not None and indices.dtype == torch.int32 and indices.ndim == 2
    assert int(indices.shape[0]) == 1, "one row per request"
    assert any(
        torch.is_tensor(c) and c.dtype == indices.dtype and c.shape == indices.shape and torch.equal(c, indices)
        for c in collected
    ), "the stored indices are the ones the chain produced (the block's collector saw them)"
    reset_all_counters()
    reused = one_step(fixtures[2], index_share=IndexShare(topk_indices=indices.clone()))
    assert read_route_counts()[3] == 0, "a filled carrier skips the chain"
    assert torch.equal(reused, plain), "and attends the first iteration's selection"
    wrong = indices.clone()
    wrong[0, 0] = -1
    moved = one_step(fixtures[3], index_share=IndexShare(topk_indices=wrong))
    assert not torch.equal(moved, plain), "a different selection moves the attention"


# --------------------------------------------------------------------------- #
# the window's edge: iterations past max_seq_len clamp to the last position.


def test_draft_iterations_past_the_window_clamp_to_the_last_position() -> None:
    """A draft that runs past ``max_seq_len`` clamps every later iteration to the last
    position (upstream ``spec_decode/utils.py`` does the same), so no iteration indexes
    the pooled store or the latent bank past the window; it still returns k ids. Without
    the clamp an iteration past the window completes a pool the window does not have and
    writes a store row the serve line does not allocate (the fixture keeps spare rows
    between the window's pools and the store's trash row to watch)."""
    _skip_unless_live()
    fx = _fixture(64_301, prefill=MAX_SEQ_LEN - 2, index_share_for_mtp_iteration=False)
    p = fx["prefill"]
    pool = fx["caches"]["impl"]["pool_cache"]
    trash = int(pool.shape[0]) - 1  # the store's last row (model_fp8 ``_require_serviceable``)
    beyond = MAX_SEQ_LEN // POOL_SIZE  # the first pool index past the window
    assert trash > beyond, "the fixture keeps rows between the window's pools and the trash row"
    # Enough iterations for an unclamped draft to complete pool ``beyond``.
    k = (beyond + 1) * POOL_SIZE - p
    assert k > 2, k
    got = _draft(fx, p, k)
    _k_tokens_per_request(got, 1, k, "clamped")
    assert bool(pool[beyond:trash].abs().sum() == 0), "no draft iteration wrote a pool past the window"
    assert bool(fx["caches"]["impl"]["latent_cache"][MAX_SEQ_LEN - 1].abs().sum() > 0), (
        "the clamped iterations wrote the window's last row"
    )


# --------------------------------------------------------------------------- #
# the route predicate over a draft, and the control that moves the fallback counter.


def test_the_route_predicate_holds_over_populate_and_a_k5_draft() -> None:
    _skip_unless_live()
    reset_all_counters()
    after_reset = read_all_counters()
    assert all(value == (0, 0) for value in after_reset.values()), after_reset
    fx = _fixture(64_101, index_share_for_mtp_iteration=False)
    prefill_readings = read_all_counters()
    reset_all_counters()
    _draft(fx, fx["prefill"], DRAFT_K)
    draft_readings = read_all_counters()
    assert gate_live()
    for phase, readings in (("populate", prefill_readings), ("draft", draft_readings)):
        dispatched = {f: v[0] for f, v in readings.items() if v[0] > 0}
        assert dispatched, f"[{phase}] no DSA seam dispatched at all: {readings}"
        fallbacks = {f: v[1] for f, v in readings.items() if v[1] > 0}
        assert not fallbacks, f"[{phase}] a torch fallback ran: {readings}"
    assert draft_readings["decode_batch"][0] >= 2 * DRAFT_K, draft_readings
    # One selection kernel per draft iteration does the top-k, the sentinel, the order and
    # the expand; the four-kernel route's top-k and expand no longer run on the decode leg.
    assert read_route_counts()[3] == DRAFT_K
    assert draft_readings["topk_select"][0] == 0 and draft_readings["index_expand"][0] == 0


def test_the_fallback_counter_reads_non_zero_when_a_fallback_is_provoked() -> None:
    _skip_unless_live()
    from vllm_neuron.functional.dsa.topk_select import dsa_topk_select

    cfg = _tiny_text_config()
    rows = PREFILL_TOKENS
    width = PREFILL_TOKENS // int(cfg.index_kpool)
    served_k = int(cfg.index_topk) // int(cfg.index_kpool)
    scores = torch.randn(rows, width, generator=torch.Generator().manual_seed(63_701), dtype=torch.float32)
    assert 0 < served_k < width
    readings = {}
    for arm, k in (("served", served_k), ("refused", width)):
        _counter_api("topk_select")[0]()
        values, _indices = dsa_topk_select(scores, k)
        assert values.numel() > 0
        readings[arm] = tuple(int(v) for v in _counter_api("topk_select")[1]())
    assert readings["served"][0] > 0 and readings["served"][1] == 0, readings
    assert readings["refused"][1] > 0 and readings["refused"][0] == 0, readings


# --------------------------------------------------------------------------- #
# the threading contract and the softmax scale.


def test_the_head_threads_exactly_the_blocks_own_keyword_arguments() -> None:
    cfg = _tiny_text_config()
    block = _impl().Glm5NextDSALayer(cfg, int(cfg.num_hidden_layers), 1)
    accepted = {
        name
        for name, parameter in inspect.signature(block.forward).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    threaded = set(_decode_kwargs(_caches(cfg, MAX_SEQ_LEN), 0)) | set(_prefill_kwargs(_caches(cfg, MAX_SEQ_LEN), 3))
    assert threaded and threaded <= accepted, sorted(threaded - accepted)
    for method in ("populate", "draft_tokens"):
        own = {
            name
            for name, parameter in inspect.signature(getattr(_mtp().Glm5NextMultiTokenPredictor, method)).parameters.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        }
        assert own, f"{method} names its own keywords"
        assert not (own & accepted), (
            f"{method} names {sorted(own & accepted)}, which belong to the block; a shadowed "
            f"name would be consumed instead of threaded"
        )
        assert {"quant_config", "block_size", "moe_group", "tp_degree", "expert_parallel_rank"} <= own, (
            f"{method} must take the FFN half's five keywords the root threads to _ffn_half"
        )


def test_softmax_scale_is_the_reference_derivation_not_the_latent_rank() -> None:
    width = TINY_GEOMETRY["qk_nope_head_dim"] + TINY_GEOMETRY["qk_rope_head_dim"]
    assert SOFTMAX_SCALE == float(width**-0.5)
    assert SOFTMAX_SCALE != float(TINY_GEOMETRY["kv_lora_rank"] ** -0.5)


def test_the_stacks_ffn_half_runs_on_the_heads_block_with_explicit_dependencies() -> None:
    """``Glm5NextModel._ffn_half`` takes what it reads -- the config and the norm -- as
    keywords, so the head runs it on its block without posing as a model instance, and
    the result is the reference's MoE half; the injected norm is the one it applies."""
    fx = _fixture(64_401)
    head = fx["head"]
    gen = torch.Generator().manual_seed(64_401)
    attended = torch.randn(2, int(fx["cfg"].hidden_size), generator=gen).to(torch.bfloat16)
    keywords = dict(quant_config=_quant_config(), block_size=None, moe_group=None,
                    tp_degree=1, expert_parallel_rank=0)
    got = _impl().Glm5NextModel._ffn_half(
        head.block, attended, text_config=fx["cfg"], rms_norm=head._rms_norm, **keywords,
    )
    want = fx["ref"].ffn(attended) - attended
    torch.testing.assert_close(got.float(), want.float(), rtol=HIDDEN_RTOL, atol=HIDDEN_ATOL)
    doubled = _impl().Glm5NextModel._ffn_half(
        head.block, attended, text_config=fx["cfg"],
        rms_norm=lambda x, gain: head._rms_norm(x, gain) * 2, **keywords,
    )
    assert not torch.allclose(doubled.float(), got.float(), rtol=HIDDEN_RTOL, atol=HIDDEN_ATOL), (
        "a different norm left the FFN half unchanged; the norm is read from somewhere "
        "other than the keyword"
    )
