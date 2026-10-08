# SPDX-License-Identifier: Apache-2.0
"""One decoder layer at one rank's TP=64 / EP=16 shard shapes, from either model tree.

Shared by the glue tests and ``test/hardware/benchmark_layer_glue.py``, so both build and
drive the same composition. ``model`` is a ``model_fp8`` module: this tree's, or the
0a08ff4 snapshot's (``test/hardware/baselines/glue_0a08ff4``). The same seed gives both
trees the same weights and the same operands.

* :func:`kda_layer` is checkpoint layer 4 (linear attention + MoE), :func:`dsa_layer`
  checkpoint layer 3 (sparse attention + MoE), each with its two mHC sites bound.
* :func:`layer_step` is what ``Glm5NextModel.forward`` does with one layer at decode: the
  layer forward (attention site) and the feed-forward site around ``_ffn_half``, with the
  model's own methods. Collectives are the identity: no process group is initialised, so
  ``_resolve_tp_group`` answers ``None`` at both reductions.

Weights. The mHC leaves, norms, KDA projections and the router are read from the served
checkpoint when it is present (rank 0's rows; bf16 and fp32 as stored) and drawn at the
same scale otherwise. The expert banks are random fp8 inside +-224 (the trn2 range the
checkpoint load squeezes them into) with random block grids; the DSA attention is
``dsa_decode_case.build_attention`` (random, one head per rank).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

CHECKPOINT = Path("/home/ubuntu/glm53f-campaign/lane-serve/models/GLM-5.3-Flash-04c4e9e9")
CONFIG_FIXTURE = (
    Path(__file__).resolve().parents[3] / "vllm_neuron/model/glm5_next/fixtures/config.json"
)
TP_WORLD = 64
EP_DEGREE = 16
TP_PER_EP = TP_WORLD // EP_DEGREE
KDA_LAYER = 4
DSA_LAYER = 3
FP8_LIMIT = 224.0
BLOCK = 128
PAGE = 128
#: The TP=64 serving line's bucket config: ``max_num_seqs = 64`` (so the runner's default
#: decode buckets 1, 2, 4, ..., 64) and one 1024-row prefill chunk. The mHC layers read
#: the step's phase off the row count against them.
SERVED_MAX_NUM_SEQS = 64
SERVED_PREFILL_BUCKETS = (1024,)


def served_decode_buckets() -> list[int]:
    """The decode batch buckets the runner builds for :data:`SERVED_MAX_NUM_SEQS`."""
    from vllm_neuron.utils.bucket_utils import get_default_num_seqs_buckets

    return get_default_num_seqs_buckets(SERVED_MAX_NUM_SEQS)


def text_config():
    """The checkpoint's text config, carrying the serving line's buckets."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
    from vllm_neuron.model.neuron_config import NeuronConfig

    return Glm5NextTextConfig(neuron_config=NeuronConfig(
        num_seqs_buckets=served_decode_buckets(),
        num_batched_tokens_buckets=list(SERVED_PREFILL_BUCKETS)))


def quant_config(model):
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig

    raw = json.loads(CONFIG_FIXTURE.read_text())
    return model.Glm5NextQuantConfig.from_model_config(Glm5NextConfig.from_configs(raw))


class _Source:
    """Rank 0's slice of a checkpoint tensor, or a random tensor of that shape."""

    def __init__(self, seed: int, use_checkpoint: bool = True):
        self.gen = torch.Generator().manual_seed(int(seed))
        index = CHECKPOINT / "model.safetensors.index.json"
        self.map = (json.loads(index.read_text())["weight_map"]
                    if use_checkpoint and index.exists() else None)

    @property
    def real(self) -> bool:
        return self.map is not None

    def get(self, key: str, shape, dtype, rows=None, cols=None, scale=1.0, offset=0.0):
        if self.map is not None and key in self.map:
            from safetensors import safe_open

            with safe_open(str(CHECKPOINT / self.map[key]), framework="pt") as f:
                part = f.get_slice(key)
                if rows is not None and cols is not None:
                    tensor = part[rows[0]:rows[1], cols[0]:cols[1]]
                elif rows is not None:
                    tensor = part[rows[0]:rows[1]]
                elif cols is not None:
                    tensor = part[:, cols[0]:cols[1]]
                else:
                    tensor = part[:]
            tensor = tensor.to(dtype)
            if tuple(tensor.shape) != tuple(shape):
                raise ValueError(f"{key}: checkpoint slice {tuple(tensor.shape)} != {shape}")
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


def moe_block(model, cfg, src: _Source, prefix: str, device="cpu"):
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
    sh = int(cfg.moe_intermediate_size) * int(cfg.n_shared_experts) // TP_WORLD
    sh = max(sh, BLOCK)
    for leaf, shape in (("gate_proj", (sh, hidden)), ("up_proj", (sh, hidden)),
                        ("down_proj", (hidden, sh))):
        _param(shared, f"{leaf}_weight", _fp8(gen, *shape))
        setattr(shared, f"{leaf}_weight_scale_inv",
                _grid(gen, shape[1], shape[0] // BLOCK, shape[1] // BLOCK).to(device))
    return block


def _kda_weights(attn, src: _Source, prefix: str, hidden: int) -> None:
    width = int(attn.num_kv_heads_per_rank) * int(attn.head_dim)
    heads = int(attn.num_kv_heads_per_rank)
    head_dim = int(attn.head_dim)
    kernel = int(attn.short_conv_kernel_size)
    p = f"{prefix}self_attn."
    bf = torch.bfloat16
    rows = (0, width)
    for leaf in ("q_proj", "k_proj", "v_proj"):
        _param(attn, f"{leaf}_weight", src.get(f"{p}{leaf}.weight", (width, hidden), bf,
                                              rows=rows, scale=hidden ** -0.5))
    _param(attn, "b_proj_weight", src.get(f"{p}b_proj.weight", (heads, hidden), bf,
                                          rows=(0, heads), scale=hidden ** -0.5))
    for leaf in ("f_a_proj", "g_a_proj"):
        _param(attn, f"{leaf}_weight", src.get(f"{p}{leaf}.weight", (head_dim, hidden), bf,
                                              scale=hidden ** -0.5))
    for leaf in ("f_b_proj", "g_b_proj"):
        _param(attn, f"{leaf}_weight", src.get(f"{p}{leaf}.weight", (width, head_dim), bf,
                                              rows=rows, scale=head_dim ** -0.5))
    for leaf in ("q_conv1d", "k_conv1d", "v_conv1d"):
        _param(attn, f"{leaf}_weight", src.get(f"{p}{leaf}.weight", (width, 1, kernel), bf,
                                              rows=rows, scale=0.5))
    _param(attn, "o_norm_weight", src.get(f"{p}o_norm.weight", (head_dim,), bf,
                                         scale=0.05, offset=1.0))
    _param(attn, "o_proj_weight", src.get(f"{p}o_proj.weight", (hidden, width), bf,
                                         cols=rows, scale=width ** -0.5))
    _param(attn, "A_log", src.get(f"{p}A_log", (heads,), torch.float32, rows=(0, heads),
                                 scale=0.3))
    _param(attn, "dt_bias", src.get(f"{p}dt_bias", (width,), torch.float32, rows=rows,
                                   scale=0.3))


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


def _prepare(model, layer, cfg, device) -> None:
    layer.to(device)
    _stand_in_root(model, layer, cfg)._run_load_time_preps(torch.device(device))


def kda_layer(model, *, seed: int = 4, device="cpu", use_checkpoint: bool = True):
    """Checkpoint layer 4: KDA attention (one head per rank) + MoE, both sites bound."""
    cfg = text_config()
    src = _Source(seed, use_checkpoint)
    prefix = f"model.language_model.layers.{KDA_LAYER}."
    layer = model.Glm5NextKDALayer(cfg, KDA_LAYER, TP_WORLD)
    hidden = int(cfg.hidden_size)
    _mhc_leaves(layer, src, prefix, hidden, int(cfg.hc_mult))
    _kda_weights(layer.self_attn, src, prefix, hidden)
    layer.mlp = moe_block(model, cfg, src, prefix, device)
    _prepare(model, layer, cfg, device)
    return SimpleNamespace(layer=layer, cfg=cfg, family="kda", checkpoint=src.real,
                           ffn_owner=_ffn_owner(model, cfg))


def dsa_layer(model, *, seed: int = 3, device="cpu", use_checkpoint: bool = True):
    """Checkpoint layer 3: DSA attention (one head per rank, random) + MoE, sites bound."""
    from test.vllm_neuron.functional.dsa import dsa_decode_case

    cfg = text_config()
    src = _Source(seed, use_checkpoint)
    prefix = f"model.language_model.layers.{DSA_LAYER}."
    layer = model.Glm5NextDSALayer(cfg, DSA_LAYER, TP_WORLD)
    hidden = int(cfg.hidden_size)
    _mhc_leaves(layer, src, prefix, hidden, int(cfg.hc_mult))
    layer.mlp = moe_block(model, cfg, src, prefix, device)
    layer.self_attn = dsa_decode_case.build_attention(
        model, dsa_decode_case.decode_config(), seed=seed, device=device)
    _prepare(model, layer, cfg, device)
    return SimpleNamespace(layer=layer, cfg=cfg, family="dsa", checkpoint=src.real,
                           attention_cfg=dsa_decode_case.decode_config(),
                           ffn_owner=_ffn_owner(model, cfg))


def streams_input(cfg, batch: int, *, seed: int = 11, device="cpu") -> torch.Tensor:
    """``[B, hc_mult, H]`` bf16 residual streams at a served layer's magnitude."""
    gen = torch.Generator().manual_seed(int(seed) + int(batch))
    shape = (int(batch), int(cfg.hc_mult), int(cfg.hidden_size))
    return (torch.randn(*shape, generator=gen) * 0.5).to(torch.bfloat16).to(device)


def kda_carriers(case, batch: int, *, seed: int = 21, device="cpu") -> dict:
    """The runner's decode carriers for ``B`` requests: one bank view per request.

    The banks hold ``B + 3`` slots; request ``b`` holds slot ``b + 1`` and is at
    position ``5 + 3 b`` (no request is opening).
    """
    attn = case.layer.self_attn
    gen = torch.Generator().manual_seed(int(seed) + int(batch))
    slots = int(batch) + 3
    conv_bank = (torch.randn(slots, *attn.kda_conv_state_shape, generator=gen) * 0.5
                 ).to(attn.kda_conv_state_dtype).to(device)
    rec_bank = (torch.randn(slots, *attn.kda_recurrent_state_shape, generator=gen) * 0.05
                ).to(attn.kda_recurrent_state_dtype).to(device)
    positions = torch.tensor([5 + 3 * b for b in range(int(batch))], dtype=torch.int32)
    out = {"banks": (conv_bank, rec_bank), "is_prefill": False}
    if int(batch) == 1:
        out.update(conv_state=conv_bank[1], recurrent_state=rec_bank[1],
                   start_position=positions.reshape(()).to(device))
    else:
        out.update(conv_state=tuple(conv_bank[b + 1] for b in range(int(batch))),
                   recurrent_state=tuple(rec_bank[b + 1] for b in range(int(batch))),
                   start_position=positions.to(device))
    return out


def dsa_carriers(case, batch: int, *, context: int = 1024, seed: int = 31,
                 device="cpu") -> dict:
    """The runner's decode carriers for ``B`` requests at ``context`` tokens each.

    The 2048-row decode window (16 pages) of the served bs=1 / bs=64 lines, so the
    layer takes the dense bypass as served. Per-request bank views, as the runner
    passes them at 0a08ff4.
    """
    from test.vllm_neuron.functional.dsa import dsa_batch_case

    cfg = case.attention_cfg
    lengths = [int(context) - b % 4 for b in range(int(batch))]
    window = 2048
    ops = dsa_batch_case.batch_operands(cfg, lengths, max_seq_len=window, seed=seed)
    # ``[pages, B]``: one column per request, the layout the runner hands over. Made on
    # the host: the device refuses ``.contiguous()`` of a transposed integer tensor.
    ops["block_table_row"] = ops["block_table"].t().contiguous()
    ops = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in ops.items()}
    slots = ops["state_slots"].cpu().tolist()
    pools = tuple(ops["pool_bank"][s] for s in slots)
    tails = tuple(ops["tail_bank"][s] for s in slots)
    return {
        "banks": (ops["latent_cache"], ops["pool_bank"], ops["tail_bank"]),
        "latent_cache": ops["latent_cache"],
        "pool_cache": pools,
        "tail": tails,
        "seq_lens": ops["seq_lens"],
        "start_position": ops["position"],
        "position": ops["position"],
        "softmax_scale": ops["softmax_scale"],
        "max_seq_len": window,
        "page_size": PAGE,
        "block_table_row": ops["block_table_row"],
        "latent_slots": ops["latent_slots"],
    }


def _ffn_owner(model, cfg):
    """The stack's ``_ffn_half`` and ``_rms_norm`` bound to a stand-in holding the config."""
    owner = SimpleNamespace(text_config=cfg)
    owner._rms_norm = MethodType(model.Glm5NextModel._rms_norm, owner)
    return owner


def layer_step(model, case, streams: torch.Tensor, carriers: dict,
               expert_rank: torch.Tensor, quant) -> torch.Tensor:
    """One layer of ``Glm5NextModel.forward`` at decode: attention site, then FFN site."""
    layer = case.layer
    keywords = {k: v for k, v in carriers.items() if k != "banks"}
    streams = layer(streams, **keywords, streams=streams)
    owner = case.ffn_owner  # built with the case: a graph cannot construct it
    site = model._mhc_ffn_site(layer, streams)
    return site.forward(
        streams,
        lambda single: model.Glm5NextModel._ffn_half(
            owner, layer, single, quant_config=quant, block_size=None, moe_group=None,
            tp_degree=TP_PER_EP, expert_parallel_rank=expert_rank,
        ),
    )
