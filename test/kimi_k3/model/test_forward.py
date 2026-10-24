# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 decode forward on 8 CPU ranks against a float64 reference, by layer and by MoE layer.

A 4-layer model at hidden 7,168 (a dense first layer of width 4,224, then MoE with latent
512, expert width 3,072 and 64 experts; three KDA layers, then MLA; 12 heads, 3 per rank)
is loaded from one set of random weights on 8 gloo ranks, TP4 by DP2 with EP8. Each replica
takes one decode step of 4 sequences over random KDA states and conv rows and random MLA
cache rows. The reference is the checkpoint's own ``modeling_kimi_linear.py``: its decoder
layer and the modules under it at the same widths on one rank, run in float64 (its float32
casts made float64) on the same weights, the experts' MXFP4 weights dequantized exactly.
Its library calls (short convolution, the recurrent gated delta rule, the gated RMSNorm)
are written out below in float64 from their upstream definitions, and the cached MLA rows
enter it as the keys and values its ``kv_b`` projection makes of them.

A reading is a cosine and an error, the largest difference over the reference's largest
magnitude, per layer on each TP rank's copy of the graded rows. The join items run each
reference layer on the port's own input to it (its stream, bank row and caches); the
end-to-end items run the reference from the token ids. The bounds follow a precision model
of the port's roundings (bf16 steps, fp32 sums, the experts' 8-bit inputs) grown through
each layer by its activations' gains; the model holds while the reference's shares stay in
the share item's limits. The routing items compare each row's 16 experts: they may differ
only where the reference's 16th selection score less its 17th is under a tie gap, a factor
times the RMS of the row's router logits. Such a row leaves the join reading at that layer
and the end-to-end readings from that layer on; more than 2 such rows at a layer in the
join, or 6 end to end, make the run not adjudicable. Set ``KIMI_K3_CHECKPOINT`` to the
checkpoint directory (its config and Python files); run with ``VLLM_NEURON_CPU_MODE=1
NKI_SIMULATOR=1``.
"""

import ast
import importlib.util
import json
import math
import os
import typing
from pathlib import Path
from types import SimpleNamespace

import einops
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

from vllm_neuron.model.kimi_k3 import model as m
from vllm_neuron.model.kimi_k3 import weight_loaders_mxfp4
from vllm_neuron.utils.weight_loader import SliceView, get_weight_loader

CHECKPOINT = Path(
    os.environ.get("KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3")
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "modeling_kimi_linear.py").is_file(),
    reason=f"Kimi K3 checkpoint not found at {CHECKPOINT}",
)

TP, DP, H, K, E, I = 4, 2, 7168, 512, 64, 3072
HEADS, LAYERS, VOCAB = 3 * TP, 4, 10240  # KDA and MLA heads, 3 per rank
T, BLOCK, ROW = 4, 256, 576  # decode tokens per replica; MLA block size and cache row
CONV, STATE = (3, 1152), (3, 128, 128)  # one rank's KDA conv rows and state per page
PAGE = BLOCK * ROW * 2
PREFIX, BLOCKS = [37, 300, 5, 200], [[1], [2, 3], [4], [5]]  # each sequence's MLA cache
MOE_LAYERS, CHOSEN = (1, 2, 3), 16
JOIN = [(0.99987, 2.2e-2), (0.99988, 2.1e-2), (0.99992, 1.8e-2), (0.99993, 1.6e-2)]
END_TO_END = [(0.99987, 2.2e-2), (0.99955, 4.2e-2), (0.99887, 6.7e-2), (0.99788, 9.2e-2)]
TIE_JOIN, TIE_END_TO_END = 6.6e-3, {1: 2.8e-2, 2: 5.2e-2, 3: 7.5e-2}
ATTENTION_SHARE, MLP_SHARE, ROUTED_SHARE = 0.1, {1: 0.75, 2: 0.5, 3: 0.4}, 0.05
MOVED_JOIN, MOVED_END_TO_END = 2, 6  # more moved rows than these: not adjudicable
WIDTHS = dict(num_hidden_layers=LAYERS, hidden_size=H, num_attention_heads=HEADS,
              num_key_value_heads=HEADS, num_experts=E, intermediate_size=1056 * TP,
              moe_intermediate_size=I, routed_expert_hidden_size=K, vocab_size=VOCAB,
              first_k_dense_replace=1)
LINEAR = dict(num_heads=HEADS, kda_layers=[1, 2, 3], full_attn_layers=[4])
FP4 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                   dtype=torch.float64)
REFERENCE = ("SituAndMul", "_get_situ_activation_params", "KimiRMSNorm", "KimiBlockSparseMLP",
             "KimiMLP", "repeat_kv", "eager_attention_forward", "KimiMLAAttention",
             "KimiDeltaAttention", "KimiMoEGate", "KimiSparseMoeBlock", "KimiDecoderLayer",
             "_apply_attn_res")


def _model() -> m.KimiK3ForConditionalGeneration:
    """The 4-layer model on meta at the test widths and this process's groups."""
    from vllm.transformers_utils.config import get_config

    config = get_config(str(CHECKPOINT), trust_remote_code=False).to_dict()
    config["text_config"].update(WIDTHS)
    config["text_config"]["linear_attn_config"].update(LINEAR)
    with torch.device("meta"):
        return m.KimiK3ForConditionalGeneration.from_configs(config)


def _weights(model) -> dict[str, torch.Tensor]:
    """Random whole weights for every parameter but the experts'."""
    gen, out = torch.Generator().manual_seed(7), {}
    for name, p in ((n, p) for n, p in model.named_parameters() if ".experts." not in n):
        leaf, offset = name.rpartition(".")[2], 1.0 if "norm" in name else 0.0
        scale = (0.01 if leaf == "e_score_correction_bias" else 0.1 if offset
                 else 0.5 if leaf in ("A_log", "dt_bias") or "conv" in name else 0.02)
        out[name] = (offset + scale * torch.randn(p.shape, generator=gen)).to(p.dtype)
    return out


def _experts() -> dict[int, list[dict[str, torch.Tensor]]]:
    """Each MoE layer's stored MXFP4 experts: random fp4 codes, E8M0 scales near 2^-6."""
    kinds = (("weight_packed", 2, 0, 256), ("weight_scale", 32, 120, 123))
    gens = {layer: torch.Generator().manual_seed(13 + layer) for layer in MOE_LAYERS}
    return {layer: [{f"{w}.{kind}": torch.randint(lo, hi, (r, c // div), generator=gen,
                                                  dtype=torch.uint8)
                     for w, r, c in (("w1", I, K), ("w3", I, K), ("w2", K, I))
                     for kind, div, lo, hi in kinds} for _ in range(E)]
            for layer, gen in gens.items()}


def _load(model, weights: dict, experts: dict, tp_rank: int) -> None:
    """Load this rank's shard of every weight and the bank of the experts it holds."""
    mappings, _ = weight_loaders_mxfp4.checkpoint_mappings(
        model, model.text_config, model.local_expert_ids())
    for name, p in list(model.named_parameters()):
        path, _, leaf = name.rpartition(".")
        slices = [SliceView(weights[name])] if name in weights else [
            SliceView(experts[int(name.split("layers.")[1].split(".")[0])][int(e)][src])
            for e, src in (key.partition(".experts.")[2].split(".", 1) for key in mappings[name])]
        value = get_weight_loader(p).load(slices, tp_rank)
        setattr(model.get_submodule(path), leaf, nn.Parameter(value, requires_grad=False))
    model.prepare_after_load()


def _caches(replica: int, tp: int) -> dict:
    """Each layer's pages on one rank: random KDA conv rows and states, random MLA rows."""
    caches, conv = {}, CONV[0] * CONV[1] * 2
    for i in range(LAYERS):
        pages = torch.zeros(max(map(max, BLOCKS)) + 1, PAGE, dtype=torch.int8)
        gen = torch.Generator().manual_seed(1000 * replica + 10 * i + (tp if i < 3 else 0))
        if i < 3:
            state = conv + STATE[0] * STATE[1] * STATE[2] * 4
            cache = [pages[:, :conv].view(torch.bfloat16).view(-1, *CONV),
                     pages[:, conv:state].view(torch.float32).view(-1, *STATE), pages]
            cache[0][1:T + 1] = (0.5 * torch.randn(T, *CONV, generator=gen)).bfloat16()
            cache[1][1:T + 1] = 0.05 * torch.randn(T, *STATE, generator=gen)
        else:
            cache = [pages.view(torch.bfloat16).view(-1, 1, BLOCK, ROW)]
            for own, length in zip(BLOCKS, PREFIX):
                rows = (0.5 * torch.randn(len(own) * BLOCK, ROW, generator=gen)).bfloat16()
                cache[0][own, 0] = rows.index_fill(0, torch.arange(length, len(own) * BLOCK), 0) \
                    .view(len(own), BLOCK, ROW)
        caches[f"layers.{i}.self_attn"] = cache
    return caches


def _inputs(replica: int) -> tuple:
    """A replica's token ids, positions and per-layer decode metadata."""
    gen, width = torch.Generator().manual_seed(100 * replica), max(map(len, BLOCKS))
    kda = torch.arange(1, T + 1, dtype=torch.int32).view(T, 1)
    mla = torch.tensor([own + [-1] * (width - len(own)) for own in BLOCKS], dtype=torch.int32)
    slots = torch.tensor([own[n // BLOCK] * BLOCK + n % BLOCK for own, n in zip(BLOCKS, PREFIX)])
    meta = {f"layers.{i}.self_attn": {"block_table_tensor": kda if i < 3 else mla,
                                      "slot_mapping": slots, "max_query_len": 1,
                                      "decode_token_threshold": 1} for i in range(LAYERS)}
    return torch.randint(0, VOCAB, (T,), generator=gen), torch.tensor(PREFIX), meta


def _rank(rank: int, root: Path) -> None:
    """One gloo rank of TP4 by DP2: every decode AttnRes mix and each MoE layer's dispatch sum."""
    from vllm_neuron.parallel.neuron_parallel_state import (
        initialize_neuron_parallel_state, stamp_spmd_rank_tensors)

    replica, tp = divmod(rank, TP)
    dist.init_process_group("gloo", init_method=f"file://{root}/rdzv", rank=rank,
                            world_size=TP * DP)
    initialize_neuron_parallel_state(ep_degree=TP * DP, dp_size=DP, local_rank=rank,
                                     tp_global_ranks=[replica * TP + i for i in range(TP)])
    stamped = stamp_spmd_rank_tensors("cpu")
    model, calls, routed = _model(), [], []
    _load(model, torch.load(root / "weights.pt", mmap=True), torch.load(root / "experts.pt"), tp)
    model.bind_kv_cache(_caches(replica, tp))
    ids, positions, meta = _inputs(replica)
    mix, total = m.attn_res_decode, m._expert_sum

    def mixed(bank, stream, *args, **kwargs):
        out = mix(bank, stream, *args, **kwargs)
        calls.append({"bank": bank[:, :1].clone(), "new_stream": out.new_stream})
        return out

    def summed(x):
        out = total(x)
        routed.append(out.clone())
        return out

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(m, "attn_res_decode", mixed)
        patch.setattr(m, "_expert_sum", summed)
        model.model(ids, meta, False, positions, torch.tensor(tp, dtype=torch.int32))
    torch.save({"calls": calls, "routed": routed}, root / f"rank{rank}.pt")
    del stamped
    dist.destroy_process_group()


def _port_layer(calls: list[dict], i: int) -> tuple[torch.Tensor, ...]:
    """Layer ``i``'s input stream and bank rows, and its output, from one rank's mixes.

    The forward mixes 8 times: at each layer's MLP site (mix 2i), at the attention sites of
    layers 1 to 3 (mix 2i - 1) and at the output site. Mix 2i + 1 folds layer i's MLP
    output into the stream. Layer 0's input is the embedding, which its attention site
    stores as bank row 0.
    """
    out = calls[2 * i + 1]["new_stream"]
    if i == 0:
        return calls[0]["bank"][:, 0], calls[0]["bank"][:, :0], out
    return calls[2 * i - 1]["new_stream"], calls[2 * i - 1]["bank"], out


class _Float64(ast.NodeTransformer):
    """Turn the reference's float32 casts into float64."""

    def visit_Attribute(self, node):
        self.generic_visit(node)
        node.attr = {"float": "double", "float32": "float64"}.get(node.attr, node.attr)
        return node


class _ShortConvolution(nn.Module):
    """Depthwise causal convolution and SiLU, one decode step over a [N, D, W] window."""

    def __init__(self, hidden_size, kernel_size, activation=None, **_):
        super().__init__()
        assert activation == "silu"
        self.weight = nn.Parameter(torch.zeros(hidden_size, 1, kernel_size))

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None):
        window = torch.cat([cache[..., 1:], x.transpose(1, 2)], dim=-1)
        return F.silu((window * self.weight[:, 0]).sum(-1)).unsqueeze(1), window


class _GatedRMSNorm(nn.Module):
    """RMSNorm over the head dimension, times the weight and a sigmoid output gate."""

    def __init__(self, hidden_size, eps=1e-5, activation="swish", **_):
        super().__init__()
        assert activation == "sigmoid"
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, g):
        norm = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight * torch.sigmoid(g)


def _fused_recurrent_kda(q, k, v, g, beta, A_log, dt_bias, initial_state, lower_bound,
                         use_qk_l2norm_in_kernel, use_gate_in_kernel,
                         use_beta_sigmoid_in_kernel, transpose_state_layout, **_):
    """One gated delta-rule step per sequence; the state is [N, H, V, K]."""
    assert use_qk_l2norm_in_kernel and use_gate_in_kernel and use_beta_sigmoid_in_kernel
    assert transpose_state_layout and q.shape[1] == 1
    q, k, v, g, beta = q[:, 0], k[:, 0], v[:, 0], g[:, 0], beta[:, 0].to(v.dtype)
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) * q.shape[-1] ** -0.5
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    gate = A_log.exp()[:, None] * (g + dt_bias.view(g.shape[-2:]))
    state = initial_state * (lower_bound * torch.sigmoid(gate)).exp()[:, :, None, :]
    delta = (v - (state * k[:, :, None, :]).sum(-1)) * torch.sigmoid(beta)[..., None]
    state = state + delta[..., None] * k[:, :, None, :]
    return (state * q[:, :, None, :]).sum(-1).unsqueeze(1), state


def _dequant(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Stored MXFP4 bytes [out, in/2] and E8M0 scales [out, in/32] as float64 [out, in]."""
    nibbles = torch.stack([packed & 15, packed >> 4], dim=-1).reshape(packed.shape[0], -1).long()
    values = FP4[nibbles].reshape(packed.shape[0], -1, 32)
    return (values * torch.exp2(scale.double() - 127)[..., None]).reshape(packed.shape[0], -1)


def _reference_layers(weights: dict, experts: dict) -> list[nn.Module]:
    """The checkpoint's decoder layers at the test widths in float64, on the test's weights."""
    spec = importlib.util.spec_from_file_location(
        "configuration_kimi_k3", CHECKPOINT / "configuration_kimi_k3.py")
    configuration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(configuration)
    text = json.loads((CHECKPOINT / "config.json").read_text())["text_config"]
    text.update(WIDTHS)
    text["linear_attn_config"].update(LINEAR)
    config = configuration.KimiLinearConfig(**text)
    config._attn_implementation = "eager"
    source = (CHECKPOINT / "modeling_kimi_linear.py").read_text()
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in REFERENCE]
    tree = ast.fix_missing_locations(_Float64().visit(ast.Module(nodes, [])))
    scope = {"torch": torch, "nn": nn, "F": F, "math": math, "rearrange": einops.rearrange,
             "Unpack": typing.Unpack, "Callable": typing.Callable, "Cache": object,
             "KimiDynamicCache": object, "KimiLinearConfig": object, "TransformersKwargs": dict,
             "FlashAttentionKwargs": dict, "ShortConvolution": _ShortConvolution,
             "FusedRMSNormGated": _GatedRMSNorm, "fused_recurrent_kda": _fused_recurrent_kda}
    exec(compile(tree, "modeling_kimi_linear.py", "exec"), scope)
    with torch.device("meta"):
        layers = [scope["KimiDecoderLayer"](config, i) for i in range(LAYERS)]
    for i, layer in enumerate(layers):
        for name, p in list(layer.named_parameters()):
            path, _, leaf = name.rpartition(".")
            if ".experts." in name:
                expert, w = name.split(".experts.")[1].split(".")[:2]
                stored = experts[i][int(expert)]
                value = _dequant(stored[f"{w}.weight_packed"], stored[f"{w}.weight_scale"])
            else:
                value = weights[f"model.layers.{i}.{name}"].double().reshape(p.shape)
            setattr(layer.get_submodule(path), leaf, nn.Parameter(value, requires_grad=False))
        layer.eval()
    return layers


class _Past:
    """One sequence's reference cache: KDA conv windows and states, MLA keys and values."""

    def __init__(self):
        self.conv_states, self.recurrent_states, self.keys = {}, {}, {}

    def update(self, key, value, layer):
        """Append the new position's keys and values, as the reference's cache does."""
        keys, values = self.keys[layer]
        self.keys[layer] = torch.cat([keys, key], 2), torch.cat([values, value], 2)
        return self.keys[layer]


def _pasts(layers: list[nn.Module], replica: int) -> list[_Past]:
    """A replica's sequences' reference caches, read from every rank's pages."""
    ranks, pasts = [_caches(replica, tp) for tp in range(TP)], [_Past() for _ in range(T)]
    for i in range(3):
        name = f"layers.{i}.self_attn"
        rows = torch.cat([c[name][0][1:T + 1].view(T, CONV[0], 3, -1) for c in ranks], -1)
        rows = torch.cat([torch.zeros_like(rows[:, :1]), rows], 1).double()
        states = torch.cat([c[name][1][1:T + 1] for c in ranks], 1).double().transpose(-1, -2)
        for n, past in enumerate(pasts):
            past.conv_states[i] = tuple(rows[n:n + 1, :, s].transpose(1, 2) for s in range(3))
            past.recurrent_states[i] = states[n:n + 1]
    attn, cache = layers[3].self_attn, ranks[0]["layers.3.self_attn"][0]
    for past, own, length in zip(pasts, BLOCKS, PREFIX):
        c = cache[own, 0].reshape(-1, ROW)[:length].double()
        kv = attn.kv_b_proj(c[:, :attn.kv_lora_rank]).view(1, length, HEADS, -1).transpose(1, 2)
        k_pass, value = kv.split([attn.qk_nope_head_dim, attn.v_head_dim], -1)
        k_rot = c[:, attn.kv_lora_rank:].view(1, 1, length, -1).expand(1, HEADS, length, -1)
        past.keys[3] = torch.cat([k_pass, k_rot], -1), value
    return pasts


def _layer_run(layer, stream, bank, past) -> dict:
    """One reference layer on one token: its output and bank, and its blocks' outputs."""
    seen = {}
    mlp = layer.block_sparse_moe if hasattr(layer, "block_sparse_moe") else layer.mlp
    hooks = [layer.self_attn.register_forward_hook(lambda _, __, o: seen.update(attention=o)),
             mlp.register_forward_hook(lambda _, __, o: seen.update(mlp=o))]
    if hasattr(layer, "block_sparse_moe"):
        hooks += [mlp.gate.register_forward_pre_hook(lambda _, a: seen.update(gate=a[0])),
                  mlp.routed_expert_up_proj.register_forward_hook(
                      lambda _, __, o: seen.update(routed=o))]
    try:
        out, bank = layer(stream.double().view(1, 1, H), past_key_values=past,
                          block_residual=bank)
    finally:
        for hook in hooks:
            hook.remove()
    return {"out": out.reshape(-1), "bank": bank, **{k: v.reshape(-1) for k, v in seen.items()}}


def _end_to_end(layers, embedding) -> list[list[dict]]:
    """Per layer, the 8 rows' reference taps with the reference run from the token ids."""
    rows = [[] for _ in range(LAYERS)]
    for d in range(DP):
        for token, past in zip(_inputs(d)[0], _pasts(layers, d)):
            stream = embedding[token].double()
            bank = stream.new_zeros(1, 0, H)
            for i, layer in enumerate(layers):
                taps = _layer_run(layer, stream, bank, past)
                stream, bank = taps["out"], taps["bank"]
                rows[i].append(taps)
    return rows


def _join(layers, records: list[dict], tp: int) -> list[list[dict]]:
    """Per layer, the 8 rows' reference taps on TP rank ``tp``'s own input to that layer."""
    rows = [[] for _ in range(LAYERS)]
    for d in range(DP):
        calls = records[d * TP + tp]["calls"]
        for i, layer in enumerate(layers):
            stream, bank, _ = _port_layer(calls, i)
            for n, past in enumerate(_pasts(layers, d)):
                rows[i].append(_layer_run(layer, stream[n], bank[n:n + 1].double(), past))
    return rows


def _routing(chosen: torch.Tensor, taps: list[dict], weights: dict, j: int, factor: float):
    """Each row's routing at MoE layer ``j`` against the reference taps: agree, moved or fail.

    Returns the states, the reference's gaps (its 16th selection score less its 17th) and
    the tie gaps, ``factor`` times the RMS of the row's reference router logits.
    """
    prefix = f"model.layers.{j}.block_sparse_moe.gate."
    logits = torch.stack([tap["gate"] for tap in taps]) @ weights[prefix + "weight"].double().T
    bias = weights[prefix + "e_score_correction_bias"].double()
    top = (torch.sigmoid(logits) + bias).topk(CHOSEN + 1, -1)
    gaps = (top.values[:, CHOSEN - 1] - top.values[:, CHOSEN]).tolist()
    ties = (factor * logits.square().mean(-1).sqrt()).tolist()
    differ = (chosen != top.indices[:, :CHOSEN].sort(-1).values).any(-1).tolist()
    states = [("fail" if gap >= tie else "moved") if moved else "agree"
              for moved, gap, tie in zip(differ, gaps, ties)]
    return states, gaps, ties


def _reading(got: torch.Tensor, ref: torch.Tensor, rows: list[int]) -> tuple[float, float]:
    """Cosine and the largest difference over the reference's largest magnitude, on ``rows``."""
    got, ref = got[rows].double().flatten(), ref[rows].double().flatten()
    return (F.cosine_similarity(got, ref, 0).item(),
            ((got - ref).abs().max() / ref.abs().max()).item())


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> SimpleNamespace:
    """The port's records, the reference end to end and on each rank's own inputs, and routing."""
    root = tmp_path_factory.mktemp("forward")
    weights, experts = _weights(_model()), _experts()
    torch.save(weights, root / "weights.pt")
    torch.save(experts, root / "experts.pt")
    mp.spawn(_rank, args=(root,), nprocs=TP * DP, join=True)
    records = [torch.load(root / f"rank{r}.pt") for r in range(TP * DP)]
    assert [len(r["calls"]) for r in records] == [2 * LAYERS] * (TP * DP)
    assert [len(r["routed"]) for r in records] == [len(MOE_LAYERS)] * (TP * DP)
    layers = _reference_layers(weights, experts)
    with torch.no_grad():
        ends, joins = _end_to_end(layers, weights["model.embed_tokens.weight"]), []
        for tp in range(TP):
            joins.append(_join(layers, records, tp))
    ranks = []
    for tp in range(TP):
        rank = SimpleNamespace(join={}, end={})
        rank.out = [torch.cat([_port_layer(records[d * TP + tp]["calls"], i)[2]
                               for d in range(DP)]) for i in range(LAYERS)]
        for n, j in enumerate(MOE_LAYERS):
            bias = weights[f"model.layers.{j}.block_sparse_moe.gate.e_score_correction_bias"]
            chosen = m.router_select(records[tp]["routed"][n][:, K:], bias, scaling=1.0)[0]
            chosen = chosen.long().sort(-1).values
            rank.join[j] = _routing(chosen, joins[tp][j], weights, j, TIE_JOIN)
            rank.end[j] = _routing(chosen, ends[j], weights, j, TIE_END_TO_END[j])
        ranks.append(rank)
    return SimpleNamespace(ends=ends, joins=joins, ranks=ranks)


def _join_rows(rank, i: int) -> list[int]:
    """The rows graded in the join at layer ``i``: all but its moved rows."""
    states = rank.join[i][0] if i in MOE_LAYERS else ["agree"] * (DP * T)
    return [r for r, state in enumerate(states) if state != "moved"]


def _end_rows(rank, i: int) -> list[int]:
    """The rows still in the end-to-end reading at layer ``i``: none moved at or before it."""
    rows = list(range(DP * T))
    for j in (j for j in MOE_LAYERS if j <= i):
        rows = [r for r in rows if rank.end[j][0][r] != "moved"]
    return rows


def test_reference_shares_inside_the_model(world):
    """At layers 1 to 3 the reference's attention, MLP and routed shares are in the model."""
    def stack(i, key):
        return torch.stack([tap[key] for tap in world.ends[i]])

    def share(part, whole):
        return (part.square().sum() / whole.square().sum()).item()

    outside = {}
    for i in (1, 2, 3):
        attention, mlp = stack(i, "attention"), stack(i, "mlp")
        shares = {"attention": (share(attention, stack(i - 1, "out") + attention),
                                ATTENTION_SHARE),
                  "mlp": (share(mlp, stack(i, "out")), MLP_SHARE[i]),
                  "routed": (share(stack(i, "routed"), mlp), ROUTED_SHARE)}
        print(f"layer {i} shares: " + ", ".join(f"{k} {v:.4f} (limit {b})"
                                                for k, (v, b) in shares.items()))
        outside.update({(i, k): v for k, (v, b) in shares.items() if not v <= b})
    assert not outside, f"not adjudicable, shares outside the model: {outside}"


@pytest.mark.parametrize("layer", range(LAYERS))
def test_join(world, layer):
    """Each rank's layer output on its own input matches the reference layer on that input."""
    cosine, error = JOIN[layer]
    over = {}
    for tp, rank in enumerate(world.ranks):
        ref = torch.stack([tap["out"] for tap in world.joins[tp][layer]])
        rows = _join_rows(rank, layer)
        cos, err = _reading(rank.out[layer], ref, rows)
        print(f"join layer {layer} TP rank {tp}: {len(rows)} rows, cosine {cos:.7f} (at least "
              f"{cosine}), error {err:.3e} (at most {error})")
        if not (cos >= cosine and err <= error):
            over[tp] = f"cosine {cos:.6f}, error {err:.2e}"
    assert not over, f"join layer {layer} outside its bounds, by TP rank: {over}"


@pytest.mark.parametrize("layer", range(LAYERS))
def test_end_to_end(world, layer):
    """Each rank's layer output matches the reference run end to end, on the rows still in."""
    cosine, error = END_TO_END[layer]
    ref = torch.stack([tap["out"] for tap in world.ends[layer]])
    over = {}
    for tp, rank in enumerate(world.ranks):
        rows = _end_rows(rank, layer)
        cos, err = _reading(rank.out[layer], ref, rows)
        print(f"end to end layer {layer} TP rank {tp}: {len(rows)} rows, cosine {cos:.7f} "
              f"(at least {cosine}), error {err:.3e} (at most {error})")
        if not (cos >= cosine and err <= error):
            over[tp] = f"cosine {cos:.6f}, error {err:.2e}"
    assert not over, f"end-to-end layer {layer} outside its bounds, by TP rank: {over}"


@pytest.mark.parametrize("layer", MOE_LAYERS)
def test_routing(world, layer):
    """Each row's 16 experts match the reference's unless its gap is under the tie gap."""
    failed, capped = {}, {}
    for tp, rank in enumerate(world.ranks):
        still = _end_rows(rank, layer - 1)
        for reading, (states, gaps, ties), rows in (("join", rank.join[layer], range(DP * T)),
                                                    ("end to end", rank.end[layer], still)):
            for r in rows:
                print(f"routing layer {layer} TP rank {tp} {reading} row {r}: {states[r]}, "
                      f"gap {gaps[r]:.2e}, tie gap {ties[r]:.2e}")
            failed.update({(tp, reading, r): f"gap {gaps[r]:.2e}, tie gap {ties[r]:.2e}"
                           for r in rows if states[r] == "fail"})
        moved_join = sum(state == "moved" for state in rank.join[layer][0])
        moved_end = DP * T - len(_end_rows(rank, layer))
        print(f"routing layer {layer} TP rank {tp}: moved rows {moved_join} in the join (at most "
              f"{MOVED_JOIN}), {moved_end} out of end to end (at most {MOVED_END_TO_END})")
        if moved_join > MOVED_JOIN or moved_end > MOVED_END_TO_END:
            capped[tp] = f"join {moved_join}, end to end {moved_end}"
    assert not capped, f"not adjudicable, too many moved rows, by TP rank: {capped}"
    assert not failed, f"experts differ at a gap past the tie gap: {failed}"
