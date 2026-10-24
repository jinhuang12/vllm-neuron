# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 expert parallelism on eight CPU ranks, TP4 by DP2, against one-rank calls.

One MoE layer at hidden 1,024, latent 512, expert width 3,072 and 64 experts is loaded on
8 gloo ranks: each holds its tensor-parallel shards of one set of random weights and its
8 experts of a random MXFP4 test checkpoint. Each replica feeds its own tokens, one decode
step and one prefill chunk; the attention output is a fixed row. The one-rank reference
adopts each kernel at its seam and composes the rest in fp64: the AttnRes mix and K slice,
the expert kernels on the dispatched rows, and the tail and projection kernels on its own
shards. The prefill up projection stores bf16, so there it takes the run's combined rows,
which are checked first. The error is the largest difference over the largest reference
magnitude. Set ``KIMI_K3_CHECKPOINT`` to the checkpoint directory (its config only); run
with ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1``.
"""

import copy
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from vllm_neuron.model.kimi_k3 import model as m
from vllm_neuron.model.kimi_k3 import weight_loaders_mxfp4
from vllm_neuron.utils.weight_loader import SliceView, get_weight_loader

CHECKPOINT = Path(
    os.environ.get("KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3")
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "config.json").is_file(),
    reason=f"Kimi K3 checkpoint not found at {CHECKPOINT}",
)

TP, DP, H, K, E, I = 4, 2, 1024, 512, 64, 3072
WORLD, LOCAL = TP * DP, E // (TP * DP)
T, P = 4, 128  # decode tokens and prefill tokens per replica
LAYER = "model.layers.1."  # an MoE layer with KDA attention
COMMON = 37  # the expert every token chooses, by its selection bias
NAN_ROW = 5  # the prefill token of replica 0 whose stream row is all NaN in the second chunk
BOUND = 1e-4
REORDER = (112 + 5) * 2.0**-24  # the norm-scale reorder's fp32 error over the term sum


def _model() -> m.KimiK3ForConditionalGeneration:
    """A 4-layer model on meta at the test widths and this process's groups."""
    from vllm.transformers_utils.config import get_config

    config = copy.deepcopy(get_config(str(CHECKPOINT), trust_remote_code=False).to_dict())
    text = config["text_config"]
    text.update(num_hidden_layers=4, hidden_size=H, num_attention_heads=8, intermediate_size=2112,
                moe_intermediate_size=I, routed_expert_hidden_size=K, num_experts=E,
                vocab_size=10240, first_k_dense_replace=1)
    text["linear_attn_config"].update(num_heads=8, kda_layers=[1, 2, 3], full_attn_layers=[4])
    with torch.device("meta"):
        return m.KimiK3ForConditionalGeneration.from_configs(config)


def _weights(model) -> dict[str, torch.Tensor]:
    """Random whole weights for the MoE layer's parameters, its experts aside."""
    gen, out = torch.Generator().manual_seed(7), {}
    for name, p in model.named_parameters():
        leaf = name.rpartition(".")[2]
        if not name.startswith(LAYER) or ".experts." in name:
            continue
        if name.endswith("e_score_correction_bias"):
            value = 0.01 * torch.randn(p.shape, generator=gen)
            value[COMMON] += 1.0
        elif "norm" in name:
            value = 1 + 0.1 * torch.randn(p.shape, generator=gen)
        else:
            scale = 0.5 if leaf in ("A_log", "dt_bias") or "conv" in name else 0.02
            value = scale * torch.randn(p.shape, generator=gen)
        out[name] = value.to(p.dtype)
    return out


def _experts() -> list[dict[str, torch.Tensor]]:
    """Each expert's stored MXFP4 tensors: random fp4 codes, E8M0 scales near 2^-6."""
    gen = torch.Generator().manual_seed(13)

    def stored(rows, cols):
        packed = torch.randint(0, 256, (rows, cols // 2), generator=gen, dtype=torch.uint8)
        return packed, torch.randint(120, 123, (rows, cols // 32), generator=gen, dtype=torch.uint8)

    out = []
    for _ in range(E):
        parts = {}
        for w, (rows, cols) in (("w1", (I, K)), ("w3", (I, K)), ("w2", (K, I))):
            parts[f"{w}.weight_packed"], parts[f"{w}.weight_scale"] = stored(rows, cols)
        out.append(parts)
    return out


def _load(model, weights: dict, experts: list, tp_rank: int) -> None:
    """Load the MoE layer: this rank's shards, and the bank of the experts it holds."""
    mappings, _ = weight_loaders_mxfp4.checkpoint_mappings(
        model, model.text_config, model.local_expert_ids())
    for name, p in list(model.named_parameters()):
        if not name.startswith(LAYER):
            continue
        if ".experts." in name:
            keys = [key.partition(".experts.")[2].split(".", 1) for key in mappings[name]]
            slices = [SliceView(experts[int(e)][src]) for e, src in keys]
        else:
            slices = [SliceView(weights[name])]
        path, _, leaf = name.rpartition(".")
        value = get_weight_loader(p).load(slices, tp_rank)
        setattr(model.get_submodule(path), leaf, nn.Parameter(value, requires_grad=False))
    model.model.layers[1].prepare_after_load()


def _inputs(replica: int) -> dict[str, torch.Tensor]:
    """A replica's decode and prefill rows: stream, one bank row and the attention output."""
    gen = torch.Generator().manual_seed(100 + replica)

    def rows(n, *shape, dtype=torch.bfloat16):
        return (0.5 * torch.randn(n, *shape, generator=gen)).to(dtype)

    return {"stream": rows(T, H), "bank": rows(T, 1, H), "attn": rows(T, H),
            "prefill_stream": rows(P, H), "prefill_bank": rows(P, 1, H),
            "prefill_attn": rows(P, H, dtype=torch.float32)}


def _rank(rank: int, root: Path) -> None:
    """One gloo rank: a decode step and a prefill chunk, with the MoE seams recorded."""
    from vllm_neuron.parallel.neuron_parallel_state import (
        initialize_neuron_parallel_state, stamp_spmd_rank_tensors)

    dist.init_process_group("gloo", init_method=f"file://{root}/rdzv", rank=rank,
                            world_size=WORLD)
    replica, tp = divmod(rank, TP)
    initialize_neuron_parallel_state(
        ep_degree=WORLD, dp_size=DP, tp_global_ranks=[replica * TP + i for i in range(TP)],
        local_rank=rank)
    stamped = stamp_spmd_rank_tensors("cpu")
    model = _model()
    _load(model, torch.load(root / "weights.pt"), torch.load(root / "experts.pt"), tp)
    layer, rows = model.model.layers[1], _inputs(replica)
    moe = layer.block_sparse_moe
    moe.sub_chunk = DP * P
    taps = {}

    def tap(name, fn):
        def recorded(*args, **kwargs):
            out = fn(*args, **kwargs)
            taps.setdefault(name, []).append((args, out))
            return out
        return recorded

    def mine(x):
        return x if tp == 0 else torch.zeros_like(x)

    layer.self_attn.forward_decode = lambda mix, k_slice, norm, meta: mine(rows["attn"])
    layer.self_attn.forward_prefill = lambda mix, norm, meta: mine(rows["prefill_attn"])
    tp_rank = torch.tensor([tp], dtype=torch.int32)
    with pytest.MonkeyPatch.context() as patch:
        for name in ("_expert_sum", "routed_experts_mx", "routed_expert_tiles_mx"):
            patch.setattr(m, name, tap(name, getattr(m, name)))
        for name in ("_combine", "prefill_output"):
            patch.setattr(m.KimiK3MoE, name, tap(name, getattr(m.KimiK3MoE, name)))
        _, pack, _ = layer.forward_decode(rows["stream"], None, rows["bank"].clone(), None,
                                          tp_rank)
        layer.forward_prefill(rows["prefill_stream"], rows["prefill_bank"].clone(), None, tp_rank)
        poisoned = rows["prefill_stream"].clone()
        if replica == 0:
            poisoned[NAN_ROW] = float("nan")
        layer.forward_prefill(poisoned, rows["prefill_bank"].clone(), None, tp_rank)
    (items, weights, *_), outs = taps["routed_expert_tiles_mx"][0]
    out = {
        "bank": [b.clone() for b in moe.experts.banks()],
        "decode": {"routed": taps["_expert_sum"][0][1], "combined": taps["_combine"][0][1],
                   "pack": pack, "base": taps["routed_experts_mx"][0][0][2]},
        "prefill": {"routed": taps["_expert_sum"][1][1], "combined": taps["_combine"][1][1],
                    "items": items, "weights": weights, "outs": outs,
                    "block": taps["prefill_output"][0][1],
                    "poisoned": taps["prefill_output"][1][1]},
    }
    torch.save(out, root / f"rank{rank}.pt")
    del stamped
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict:
    """The eight ranks' records, and the one-rank layer on the same weights and experts."""
    root = tmp_path_factory.mktemp("ep")
    one = _model()
    weights, experts = _weights(one), _experts()
    torch.save(weights, root / "weights.pt")
    torch.save(experts, root / "experts.pt")
    mp.spawn(_rank, args=(root,), nprocs=WORLD, join=True)
    _load(one, weights, experts, 0)
    return {"layer": one.model.layers[1], "w": weights,
            "ranks": [torch.load(root / f"rank{i}.pt") for i in range(WORLD)]}


def _error(out: torch.Tensor, ref: torch.Tensor) -> float:
    """The largest difference over the largest reference magnitude."""
    return ((out.double() - ref.double()).abs().max() / ref.double().abs().max()).item()


def _w(runs, name: str) -> torch.Tensor:
    return runs["w"][LAYER + "block_sparse_moe." + name + ".weight"]


def _shard(weight: torch.Tensor, dim: int, tp: int) -> torch.Tensor:
    """Tensor-parallel rank ``tp``'s contiguous part of ``weight`` along ``dim``."""
    return weight.chunk(TP, dim)[tp]


def _shared(runs, tp: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank ``tp``'s shared-expert weights: gate and up ``[H, 2I/TP]``, down ``[I/TP, H]``."""
    gate, up = (_shard(_w(runs, f"shared_experts.{n}_proj"), 0, tp) for n in ("gate", "up"))
    down = _shard(_w(runs, "shared_experts.down_proj"), 1, tp).t().contiguous()
    return torch.cat([gate, up]).t().contiguous(), down


def _dispatched(runs, normed: torch.Tensor) -> torch.Tensor:
    """fp64 latent and router logits ``[N, K + E]`` of fp64 normed rows ``[N, H]``."""
    weight = torch.cat([_w(runs, "routed_expert_down_proj"), _w(runs, "gate")])
    return normed.double() @ weight.double().t()


def _seam(runs, phase: str, ref: torch.Tensor) -> None:
    """Every rank's dispatched latent and logits meet the fp64 reference within the bound."""
    for out in runs["ranks"]:
        got = out[phase]["routed"]
        assert got.dtype == torch.float32
        assert _error(got[:, :K], ref[:, :K]) <= BOUND and _error(got[:, K:], ref[:, K:]) <= BOUND


def _block(pack: torch.Tensor) -> torch.Tensor:
    """The fp64 block output of a summed pack: the norm-scaled up part plus the shared part."""
    up, shared, squares = pack.double().split([H, H, 1], -1)
    return up * torch.rsqrt(squares / K + 1e-5) + shared


def _decode_reference(runs) -> dict:
    """The one-rank decode: K slice and mix, dispatched rows, experts, and the summed pack."""
    layer, moe = runs["layer"], runs["layer"].block_sparse_moe
    mixes, normed = [], []
    for replica in range(DP):
        rows = _inputs(replica)
        mix, _, k_slice, _ = layer._mix_decode(rows["stream"], rows["attn"], rows["bank"], "mlp")
        mixes.append(mix)
        normed.append(k_slice.permute(1, 2, 0, 3).reshape(T, H))
    shared = moe.shared_experts
    experts = m.routed_experts_mx(
        runs["ranks"][0]["decode"]["routed"], moe.gate.e_score_correction_bias,
        torch.zeros(1, 1, dtype=torch.int32), *moe.experts.banks(), beta=shared.beta,
        linear_beta=shared.linear_beta, scaling=moe.scaling, replicas=DP, chunks=TP)
    norm, up = _w(runs, "routed_expert_norm").float(), _w(runs, "routed_expert_up_proj")
    packs = []
    for replica in range(DP):
        pack = 0
        for tp in range(TP):
            gate_up, down = _shared(runs, tp)
            gate_up = m._project(mixes[replica], gate_up, layer.post_attention_layernorm.weight,
                                 layer.eps)
            pack = pack + m.moe_fused_tail(
                experts[replica, tp], gate_up,
                _shard(norm, 0, tp).reshape(-1, 1).contiguous(),
                _shard(up, 1, tp).t().contiguous(), down, beta=shared.beta,
                linear_beta=shared.linear_beta).double()
        packs.append(pack)
    return {"dispatched": _dispatched(runs, torch.cat(normed)), "experts": experts,
            "packs": packs}


def _prefill_reference(runs) -> dict:
    """The one-rank prefill: mix, dispatched rows, selection, each expert alone, the block."""
    layer, moe = runs["layer"], runs["layer"].block_sparse_moe
    gain = layer.post_attention_layernorm.weight
    mixes = []
    for replica in range(DP):
        rows = _inputs(replica)
        stream = rows["prefill_stream"] + rows["prefill_attn"].to(torch.bfloat16)
        mixes.append(layer.mlp_mix(stream, rows["prefill_bank"]))
    mix = torch.cat(mixes).double()
    normed = mix * torch.rsqrt(mix.square().mean(-1, keepdim=True) + layer.eps) * gain.double()
    routed = runs["ranks"][0]["prefill"]["routed"]
    picks = [m.router_select(tile, moe.gate.e_score_correction_bias, scaling=moe.scaling)
             for tile in routed[:, K:].split(128)]
    ids, affinity = (torch.cat(part) for part in zip(*picks))
    shared, banks = moe.shared_experts, moe.experts.banks()
    alone, combined = {}, torch.zeros(DP * P, K, dtype=torch.float64)
    for e in range(E):
        token, slot = (ids == e).nonzero(as_tuple=True)
        padded = -(-len(token) // 128) * 128
        rows = torch.zeros(1, padded, K)
        weights = torch.zeros(1, padded)
        rows[0, :len(token)], weights[0, :len(token)] = routed[token, :K], affinity[token, slot]
        out = m.routed_expert_tiles_mx(rows, weights, *[b[e:e + 1] for b in banks],
                                       beta=shared.beta, linear_beta=shared.linear_beta)
        alone[e] = (token, out[0, :len(token)])
        combined[token] += alone[e][1].double()
    norm, up = _w(runs, "routed_expert_norm").float(), _w(runs, "routed_expert_up_proj")
    blocks = []
    for replica in range(DP):
        part, pack = combined[replica * P:(replica + 1) * P].float(), 0
        for tp in range(TP):
            gate_up, down = _shared(runs, tp)
            cols = _shard(part, 1, tp)
            # A 1e-7 difference in the combined rows can move a bf16 output by a whole step.
            adopted = runs["ranks"][replica * TP + tp]["prefill"]["combined"]
            scaled = (adopted * _shard(norm, 0, tp)).to(torch.bfloat16)
            act = m.situ_and_mul(*m._project(mixes[replica], gate_up, gain, layer.eps).chunk(2, -1),
                                 shared.beta, shared.linear_beta)
            pack = pack + torch.cat([
                m._prefill_projection(scaled, _shard(up, 1, tp).t().contiguous()).double(),
                m._prefill_projection(act, down).double(),
                cols.double().square().sum(-1, keepdim=True)], -1)
        blocks.append(_block(pack))
    return {"dispatched": _dispatched(runs, normed), "ids": ids, "affinity": affinity,
            "alone": alone, "combined": combined, "blocks": blocks}


@pytest.fixture(scope="module")
def decode(runs) -> dict:
    return _decode_reference(runs)


@pytest.fixture(scope="module")
def prefill(runs) -> dict:
    return _prefill_reference(runs)


def test_decode_block_output(runs, decode):
    """Dispatched rows, combined rows and the block output after the tail all-reduce, at decode."""
    _seam(runs, "decode", decode["dispatched"])
    for rank, out in enumerate(runs["ranks"]):
        replica, tp = divmod(rank, TP)
        assert _error(out["decode"]["combined"], decode["experts"][replica, tp]) <= BOUND
        assert _error(_block(out["decode"]["pack"]), _block(decode["packs"][replica])) <= BOUND


def test_reordered_norm_scale(runs):
    """The pack's up part scaled by the summed sum of squares, against the fp64 reorder."""
    up, norm = _w(runs, "routed_expert_up_proj").double(), _w(runs, "routed_expert_norm").float()
    ranks = runs["ranks"]
    for replica in range(DP):
        shards = [ranks[replica * TP + tp]["decode"]["combined"] for tp in range(TP)]
        combined = torch.cat(shards, 1)
        terms = (combined * norm).to(torch.bfloat16).double()
        scale = torch.rsqrt(combined.double().square().sum(-1, keepdim=True) / K + 1e-5)
        pack = ranks[replica * TP]["decode"]["pack"].double()
        got = pack[:, :H] * torch.rsqrt(pack[:, -1:] / K + 1e-5)
        error = (got - (terms @ up.t()) * scale).abs() / ((terms.abs() @ up.abs().t()) * scale)
        assert error.max().item() <= REORDER


def test_bank_by_rank(runs, prefill):
    """Rank r holds experts 8r to 8r + 7, passes 8r at decode and weights them at prefill."""
    full = runs["layer"].block_sparse_moe.experts.banks()
    for rank, out in enumerate(runs["ranks"]):
        first = rank * LOCAL
        assert all(torch.equal(b, f[first:first + LOCAL]) for b, f in zip(out["bank"], full))
        assert out["decode"]["base"].tolist() == [[first]]
        chosen = prefill["ids"].unsqueeze(-1) == torch.arange(first, first + LOCAL)
        want = torch.where(chosen, prefill["affinity"].unsqueeze(-1), 0.0).sum(1).t()
        assert torch.equal(out["prefill"]["weights"], want)


def test_prefill_items_and_block_output(runs, prefill):
    """Each item is its expert alone on its choosing rows and 0 elsewhere; then the block."""
    _seam(runs, "prefill", prefill["dispatched"])
    common = prefill["alone"][COMMON][0]
    assert len(common) == DP * P
    for rank, out in enumerate(runs["ranks"]):
        replica, tp = divmod(rank, TP)
        outs = out["prefill"]["outs"]
        assert tuple(outs.shape) == (LOCAL, DP * P, K)
        for n in range(LOCAL):
            token, alone = prefill["alone"][rank * LOCAL + n]
            others = torch.ones(DP * P, dtype=torch.bool)
            others[token] = False
            assert torch.equal(outs[n, token], alone) and bool((outs[n, others] == 0).all())
        rows = prefill["combined"][replica * P:(replica + 1) * P]
        assert _error(out["prefill"]["combined"], _shard(rows, 1, tp)) <= BOUND
        assert _error(out["prefill"]["block"], prefill["blocks"][replica]) <= BOUND


def test_prefill_nan_row_stays_in_its_replica(runs):
    """An all-NaN prefill row in replica 0 leaves replica 1's block output bit-equal."""
    for rank, out in enumerate(runs["ranks"]):
        clean, poisoned = out["prefill"]["block"], out["prefill"]["poisoned"]
        if rank < TP:
            assert not poisoned[NAN_ROW].isfinite().any()
        else:
            assert torch.equal(poisoned, clean)
