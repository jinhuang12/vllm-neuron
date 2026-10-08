# SPDX-License-Identifier: Apache-2.0
"""One MoE block at one rank's TP=64 / EP=16 shard shapes, from either model tree.

Shared by ``test_moe_prefill_block.py`` and ``test/hardware/benchmark_moe_prefill_block.py``
so both build and drive the same composition. ``model`` is a ``model_fp8`` module: this
tree's, or the 8aa22fa snapshot's (``test/hardware/baselines/moe_prefill_8aa22fa``). The
same seed gives both trees the same weights and the same operands.

The block (:func:`block_step`) is what the prefill graph runs between the attention
sub-block's output and the next layer's streams:

1. the attention site's ``mhc_post`` (the ``hyper_connection`` combine kernel),
2. the feed-forward site's ``mhc_pre`` (the collapse to one stream, Sinkhorn),
3. ``Glm5NextModel._ffn_half``: the experts' RMSNorm, the router, this rank's routed
   experts, the shared expert,
4. the feed-forward site's ``mhc_post``.

Collectives are the identity: no process group is initialised, so
``_resolve_tp_group`` answers ``None``.

Weights. Checkpoint layer 3 (the first MoE layer, the one the served-graph evidence
names): the mHC leaves, both norm gains, the router weight and its correction bias are
read from the served checkpoint when it is on this host (as stored), and drawn at the
same scale otherwise. The expert banks are random fp8 inside +-224 with random block
grids, at EP rank 0's 18 experts and ``I = 2048 / 4``. The attention module is not
built: the block starts at the attention output.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

CHECKPOINT = Path("/home/ubuntu/glm53f-campaign/lane-serve/models/GLM-5.3-Flash-04c4e9e9")
CONFIG_FIXTURE = (
    Path(__file__).resolve().parents[2] / "model/glm5_next/fixtures/config.json"
)
TP_WORLD = 64
EP_DEGREE = 16
TP_PER_EP = TP_WORLD // EP_DEGREE
MOE_LAYER = 3
FP8_LIMIT = 224.0
BLOCK = 128


def text_config():
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    return Glm5NextTextConfig()


def quant_config(model):
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig

    raw = json.loads(CONFIG_FIXTURE.read_text())
    return model.Glm5NextQuantConfig.from_model_config(Glm5NextConfig.from_configs(raw))


class _Source:
    """A checkpoint tensor as stored, or a random tensor of that shape."""

    def __init__(self, seed: int, use_checkpoint: bool = True):
        self.gen = torch.Generator().manual_seed(int(seed))
        index = CHECKPOINT / "model.safetensors.index.json"
        self.map = (json.loads(index.read_text())["weight_map"]
                    if use_checkpoint and index.exists() else None)

    @property
    def real(self) -> bool:
        return self.map is not None

    def get(self, key: str, shape, dtype, scale=1.0, offset=0.0):
        if self.map is not None and key in self.map:
            from safetensors import safe_open

            with safe_open(str(CHECKPOINT / self.map[key]), framework="pt") as f:
                tensor = f.get_tensor(key).to(dtype)
            if tuple(tensor.shape) != tuple(shape):
                raise ValueError(f"{key}: checkpoint {tuple(tensor.shape)} != {shape}")
            return tensor.contiguous()
        return (torch.randn(*shape, generator=self.gen) * scale + offset).to(dtype)


def _param(module, name: str, tensor: torch.Tensor) -> None:
    setattr(module, name, torch.nn.Parameter(tensor, requires_grad=False))


def _mhc_leaves(layer, src: _Source, prefix: str, hidden: int, streams: int) -> None:
    mix = (2 + streams) * streams
    for site in ("attn", "ffn"):
        _param(layer, f"hc_{site}_fn", src.get(f"{prefix}hc_{site}_fn", (mix, streams * hidden),
                                              torch.bfloat16, scale=(streams * hidden) ** -0.5))
        _param(layer, f"hc_{site}_base", src.get(f"{prefix}hc_{site}_base", (mix,),
                                                torch.float32, scale=0.5))
        _param(layer, f"hc_{site}_scale", src.get(f"{prefix}hc_{site}_scale", (3,),
                                                 torch.float32, scale=0.1, offset=1.0))
    for name in ("input_layernorm", "post_attention_layernorm"):
        _param(layer, f"{name}_weight", src.get(f"{prefix}{name}.weight", (hidden,),
                                               torch.bfloat16, scale=0.05, offset=1.0))


def _fp8(gen: torch.Generator, *shape: int) -> torch.Tensor:
    raw = (torch.randn(*shape, generator=gen) * 48.0).clamp(-FP8_LIMIT, FP8_LIMIT)
    return raw.to(torch.float8_e4m3fn)


def _grid(gen: torch.Generator, fan_in: int, *shape: int) -> torch.Tensor:
    return ((torch.rand(*shape, generator=gen) * 0.5 + 0.75) * fan_in ** -0.5 / 48.0
            ).to(torch.float32)


def _moe_block(model, cfg, src: _Source, prefix: str, device):
    """``Glm5NextMoEBlock`` holding EP rank 0's 18 experts at TP 4 inside the group."""
    block = model.Glm5NextMoEBlock(cfg, world_size=TP_WORLD, ep_degree=EP_DEGREE)
    hidden = int(cfg.hidden_size)
    experts = int(block.experts.num_local_experts)
    inter = int(cfg.moe_intermediate_size) // TP_PER_EP
    routed = int(cfg.n_routed_experts)
    bank = block.experts
    # The bank holds the router transposed, [H, E], as the loader leaves it.
    _param(bank, "router_weight", src.get(f"{prefix}mlp.gate.weight", (routed, hidden),
                                          torch.bfloat16, scale=hidden ** -0.5
                                          ).t().contiguous())
    _param(bank, "router_bias", src.get(f"{prefix}mlp.gate.e_score_correction_bias",
                                        (routed,), torch.float32, scale=0.05))
    gen = src.gen
    hb, ib = hidden // BLOCK, inter // BLOCK
    for leaf, shape, grid in (("gate_proj", (experts, inter, hidden), (experts, ib, hb)),
                              ("up_proj", (experts, inter, hidden), (experts, ib, hb)),
                              ("down_proj", (experts, hidden, inter), (experts, hb, ib))):
        _param(bank, f"{leaf}_weight", _fp8(gen, *shape))
        # Plain attributes, as the loader leaves them (the prep reassigns them), so
        # ``layer.to`` does not move them: they are made on ``device``.
        setattr(bank, f"{leaf}_weight_scale_inv", _grid(gen, shape[2], *grid).to(device))
    shared = block.shared_experts
    sh = max(int(cfg.moe_intermediate_size) * int(cfg.n_shared_experts) // TP_WORLD, BLOCK)
    for leaf, shape in (("gate_proj", (sh, hidden)), ("up_proj", (sh, hidden)),
                        ("down_proj", (hidden, sh))):
        _param(shared, f"{leaf}_weight", _fp8(gen, *shape))
        setattr(shared, f"{leaf}_weight_scale_inv",
                _grid(gen, shape[1], shape[0] // BLOCK, shape[1] // BLOCK).to(device))
    return block


def _stand_in_root(model, layer, cfg):
    """The root's load-time prep walk over one layer, with the root's own method bodies."""
    root_cls = model.Glm5NextForConditionalGeneration

    class _Root(torch.nn.Module):
        _run_load_time_preps = root_cls._run_load_time_preps
        _require_prep_operands_on_device = root_cls._require_prep_operands_on_device
        _sibling_scale_grid_name = staticmethod(root_cls._sibling_scale_grid_name)

        def __init__(self):
            super().__init__()
            self.layer = layer
            self.text_config = cfg

    return _Root()


def moe_layer(model, *, seed: int = 3, device="cpu", use_checkpoint: bool = True):
    """Checkpoint layer 3's two mHC sites, norms and MoE block, prepared as at load."""
    cfg = text_config()
    src = _Source(seed, use_checkpoint)
    prefix = f"model.language_model.layers.{MOE_LAYER}."
    layer = model.Glm5NextDSALayer(cfg, MOE_LAYER, TP_WORLD)
    # The block starts at the attention output; the attention module is never called.
    layer.self_attn = torch.nn.Module()
    _mhc_leaves(layer, src, prefix, int(cfg.hidden_size), int(cfg.hc_mult))
    layer.mlp = _moe_block(model, cfg, src, prefix, device)
    layer.to(device)
    _stand_in_root(model, layer, cfg)._run_load_time_preps(torch.device(device))
    owner = SimpleNamespace(text_config=cfg)
    owner._rms_norm = MethodType(model.Glm5NextModel._rms_norm, owner)
    return SimpleNamespace(layer=layer, cfg=cfg, checkpoint=src.real, ffn_owner=owner)


def block_inputs(cfg, tokens: int, *, seed: int = 11, device="cpu") -> dict:
    """The attention site's outputs for ``tokens`` rows, at a served layer's scale.

    ``attn_out [T, H]`` bf16 (the attention sub-block's output), ``streams
    [T, S, H]`` bf16 (the streams that entered the layer), ``post_mix [T, S, 1]`` and
    ``comb_mix [T, S, S]`` fp32 (the attention site's ``mhc_pre`` mixes: ``2 *
    sigmoid`` and a doubly-stochastic matrix, as Sinkhorn leaves it).
    """
    gen = torch.Generator().manual_seed(int(seed) + int(tokens))
    hidden, hc = int(cfg.hidden_size), int(cfg.hc_mult)
    attn_out = torch.randn(tokens, hidden, generator=gen) * 0.3
    streams = torch.randn(tokens, hc, hidden, generator=gen) * 0.5
    post_mix = 2.0 * torch.sigmoid(torch.randn(tokens, hc, 1, generator=gen))
    comb = torch.rand(tokens, hc, hc, generator=gen) + 0.1
    for _ in range(20):
        comb = comb / comb.sum(dim=-1, keepdim=True)
        comb = comb / comb.sum(dim=-2, keepdim=True)
    return {
        "attn_out": attn_out.to(torch.bfloat16).to(device),
        "streams": streams.to(torch.bfloat16).to(device),
        "post_mix": post_mix.to(torch.float32).to(device),
        "comb_mix": comb.to(torch.float32).to(device),
    }


def block_step(model, case, attn_out, streams, post_mix, comb_mix, expert_rank, quant,
               collector=None) -> torch.Tensor:
    """Attention-site combine, then the feed-forward site around ``_ffn_half``.

    Returns the ``[T, S, H]`` bf16 streams the next layer receives. ``collector``
    (a list) is threaded to ``_ffn_half`` and collects the MoE block's intermediates:
    the experts' normed rows, the router's logits, index, gathered and scattered
    weights, the routed output, then the block output and its cast.
    """
    layer, owner = case.layer, case.ffn_owner
    streams = model._mhc_attention_site(layer, streams).mhc_post(
        attn_out, streams, post_mix, comb_mix)
    site = model._mhc_ffn_site(layer, streams)
    extra = {"collector": collector} if collector is not None else {}
    return site.forward(
        streams,
        lambda single: model.Glm5NextModel._ffn_half(
            owner, layer, single, quant_config=quant, block_size=None, moe_group=None,
            tp_degree=TP_PER_EP, expert_parallel_rank=expert_rank, **extra,
        ),
    )


def block_step_tapped(model, case, attn_out, streams, post_mix, comb_mix, expert_rank,
                      quant):
    """:func:`block_step`, also returning what the router read and what it chose.

    ``(streams_out, layer_input [T, H] bf16, router_logits [T, E] fp32,
    expert_index [T, K])``: ``layer_input`` is the feed-forward site's collapse, the
    rows ``route_tokens`` is handed. For accuracy checks only: the extra graph outputs
    change what the compiler must materialise, so time :func:`block_step` instead.
    """
    layer, owner = case.layer, case.ffn_owner
    streams = model._mhc_attention_site(layer, streams).mhc_post(
        attn_out, streams, post_mix, comb_mix)
    site = model._mhc_ffn_site(layer, streams)
    collector: list = []
    held: list = []

    def sublayer(single):
        held.append(single)
        return model.Glm5NextModel._ffn_half(
            owner, layer, single, quant_config=quant, block_size=None, moe_group=None,
            tp_degree=TP_PER_EP, expert_parallel_rank=expert_rank, collector=collector,
        )

    out = site.forward(streams, sublayer)
    return out, held[0], collector[1], collector[2]


def exact_router(layer_input, gamma, router_weight, eps):
    """fp64 router logits on ``layer_input`` with the fused kernel's two bf16 roundings.

    ``bf16(bf16(x * rstd) * gamma) @ W`` with ``rstd`` and the GEMM in fp64: the router
    8aa22fa's fused kernel defines, with no rounding of its own beyond those two.
    """
    x = layer_input.double()
    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    scaled = (x * rstd).to(torch.bfloat16).double()
    normed = (scaled * gamma.double().reshape(1, -1)).to(torch.bfloat16).double()
    return normed @ router_weight.double()
