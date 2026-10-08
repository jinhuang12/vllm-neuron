# SPDX-License-Identifier: Apache-2.0
"""Time the GLM-5.3-Flash MTP draft head (``mtp.py``) at the per-rank TP=64 decode shape.

Run it through the device lease (``devlease.py slice dense -- ...``); this script does
not select cores. One graph per ``k`` in ``--ks``:
``Glm5NextMultiTokenPredictor.draft_tokens`` for one request, the ``k`` iterations
unrolled in that one graph (the fused form of mtp.md 8.5), each iteration the whole of
layer 45 at this rank's share:

* the head's own arithmetic as its two kernels (``functional/mtp/``): K1' ``mtp_tail_in``
  -- position-0 mask, ``enorm``, ``hnorm``, concat, ``eh_proj`` ([H, 2H] bf16, whole at this
  one rank; the served line row-shards it) -- and K2 ``mtp_tail_out`` -- the shared-head
  norm, this rank's head rows ([2420, H] bf16) and the ``(max, argmax)`` the greedy token
  is read from;
* the attention half -- ``Glm5NextDSALayer`` (one MLA head, latent 512, the indexer at 32
  heads, fp8 projections on a 128x128 scale grid) from ``dsa_decode_case.build_attention``,
  at ctx 1024 in the 2048-row window the serving line decodes in (the indexer bypass);
* the MoE half -- the fused router over 288 experts, the 18 local experts (EP 16) at
  I=512 (2048 over the group's TP 4) through the packed fp8 decode kernel, the shared
  expert at I=128 (2048 / 64 = 32, padded to one 128 block), the post-attention norm and
  the residual add.

Geometry notes. ``vocab_size`` is set to the shard's 2420 rows, so the head is "whole" at
one rank and the token is the plain ``argmax``: the same GEMV as the sharded route without
the 2 x 64-value all-gather, which one rank cannot run. The request's next page is
allocated in the block table (what Stage B's lookahead allocation does): iterations
``1 .. k-1`` write rows 1024..1027, and without that page they would land on the null
block. ``index_share_for_mtp_iteration`` stays the checkpoint's ``True``; in the bypass
regime the carrier stays empty, so every iteration runs the indexer's write stage.

A ``null`` graph (one bf16 add on a [1, H] row) gives the launch floor. The per-iteration
cost is the difference between consecutive ``k`` graphs, reported as ``marginal_us``.

Correctness. ``--mode sim`` (``NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1``) runs the same
graph eagerly through the NKI simulator and writes the ``[1, k]`` ids and the ``k`` normed
hidden rows; ``--mode device`` writes the compiled graph's. ``--compare device.json
sim.json`` reads two outputs and reports, per ``k``, whether the ids are bit-equal, the
hidden rows' relative L2, and the top-1/top-2 logit margin the ids rest on.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import torch  # noqa: E402

import vllm_neuron  # noqa: E402,F401 -- registers the Neuron compilation backend
from vllm_neuron.model.glm5_next import model_fp8, mtp  # noqa: E402
from vllm_neuron.model.glm5_next.config import Glm5NextConfig  # noqa: E402
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import FP8_SCALE_SUFFIX  # noqa: E402
from test.vllm_neuron.functional.dsa import dsa_decode_case as case  # noqa: E402

MAX_SEQ_LEN = 4096
WINDOW_PAGES = 16  # 2048 rows: the serving line's decode context bucket (bypass regime)
CONTEXT = 1024
TP_WORLD = 64
VOCAB = 154880
SHARD_ROWS = VOCAB // TP_WORLD  # 2420
MTP_LAYER_IDX = 45
#: The MoE half's per-rank share on the TP=64 serve line: EP 16 x TP 4 within a group.
EXPERTS = 288
EP_DEGREE = 16
LOCAL_EXPERTS = EXPERTS // EP_DEGREE  # 18
LOCAL_INTERMEDIATE = 2048 // 4  # 512
SHARED_INTERMEDIATE = 128  # 2048 / 64 = 32 rows, padded to one 128-wide scale block
EXPERT_RANK = 5
#: HBM bandwidth one LNC2 core draws, 716 GB/s: the basis of every entitlement in
#: ``reports/mtp-head.md`` 9.2 (a weight-streaming kernel's entitlement is its bytes over it).
HBM_BYTES_PER_US_PER_CORE = 716e9 / 1e6
#: The pinned checkpoint config the quantisation policy (fp8, 128x128 blocks) is read from.
FIXTURE_CONFIG = ROOT / "test" / "vllm_neuron" / "model" / "glm5_next" / "fixtures" / "config.json"
#: The served model's neuronx-cc arguments (``benchmark_moe_decode.MODEL_COMPILER_ARGS``);
#: ``NEURON_CC_FLAGS`` overrides.
MODEL_COMPILER_ARGS = [
    "--auto-cast=none",
    "-O1",
    "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10 "
    "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3",
    "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop",
]
BLOCK_NAMES = ("latent_cache", "pool_cache", "seq_lens", "start_position", "block_table_row",
               "latent_slots", "tail", "position")


def rel_l2(got: torch.Tensor, want: torch.Tensor) -> float:
    got, want = got.detach().cpu().double(), want.detach().cpu().double()
    return float((got - want).norm() / want.norm().clamp_min(1e-30))


def compiler_args():
    return os.environ.get("NEURON_CC_FLAGS") or MODEL_COMPILER_ARGS


def compiled(fn):
    return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False,
                         options={"compiler_args": compiler_args()})


def _sync(out):
    outs = out if isinstance(out, (tuple, list)) else (out,)
    return [o.to("cpu") for o in outs if torch.is_tensor(o)]


def measure(fn, inputs, warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        _sync(fn(*inputs))
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        _sync(fn(*inputs))
        samples.append((time.perf_counter_ns() - started) / 1000.0)
    samples.sort()
    return {"iterations": iterations, "median_us": statistics.median(samples),
            "p90_us": samples[min(len(samples) - 1, int(0.9 * len(samples)))],
            "min_us": samples[0], "max_us": samples[-1]}


# --- building ------------------------------------------------------------------------

def build_config(hidden: int, q_lora: int):
    """The checkpoint config at one MLA head, the head 'whole' at this rank's 2420 rows."""
    return replace(case.decode_config(hidden_size=hidden, q_lora_rank=q_lora), vocab_size=SHARD_ROWS)


def quant_config():
    """The resolved policy the MoE half runs under, from the pinned checkpoint config."""
    raw = json.loads(FIXTURE_CONFIG.read_text())
    return model_fp8.Glm5NextQuantConfig.from_model_config(Glm5NextConfig.from_configs(raw))


def head_tables(cfg, seed: int, device) -> dict:
    """The four MTP tensors, the embedding rows and the head rows, bf16 like the checkpoint."""
    gen = torch.Generator().manual_seed(seed)
    hidden, vocab = int(cfg.hidden_size), int(cfg.vocab_size)
    w = {
        "enorm_weight": 1.0 + torch.randn(hidden, generator=gen) * 0.05,
        "hnorm_weight": 1.0 + torch.randn(hidden, generator=gen) * 0.05,
        "eh_proj_weight": torch.randn(hidden, 2 * hidden, generator=gen) * (2 * hidden) ** -0.5,
        "shared_head_norm_weight": 1.0 + torch.randn(hidden, generator=gen) * 0.05,
        "embed_tokens_weight": torch.randn(vocab, hidden, generator=gen),
        "lm_head_weight": torch.randn(vocab, hidden, generator=gen) * hidden ** -0.5,
    }
    return {k: v.to(torch.bfloat16).to(device) for k, v in w.items()}


def _fp8_bank(experts: int, out_features: int, in_features: int, gen) -> tuple[torch.Tensor, torch.Tensor]:
    """``[E, out, in]`` fp8 bytes inside the trn2 range and an ``[E, out/128, in/128]`` grid."""
    raw = (torch.randn(experts, out_features, in_features, generator=gen) * 48.0).clamp(
        -case.FP8_LIMIT, case.FP8_LIMIT)
    grid = (torch.rand(experts, out_features // case.BLOCK, in_features // case.BLOCK, generator=gen)
            * 0.5 + 0.75) * (in_features ** -0.5) / 48.0
    return raw.to(torch.float8_e4m3fn), grid.to(torch.float32)


def _grid_name(leaf: str) -> str:
    return model_fp8.Glm5NextForConditionalGeneration._sibling_scale_grid_name(leaf)


def _attach(module, leaf: str, weight: torch.Tensor, grid: torch.Tensor, device) -> None:
    setattr(module, leaf, torch.nn.Parameter(weight.to(device), requires_grad=False))
    setattr(module, _grid_name(leaf), torch.nn.Parameter(grid.to(device), requires_grad=False))


def materialise_moe(mlp, cfg, seed: int, device) -> None:
    """This rank's 18-expert bank at I=512, the shared expert at I=128, the router; prepared."""
    gen = torch.Generator().manual_seed(seed)
    hidden = int(cfg.hidden_size)
    experts, shared = mlp.experts, mlp.shared_experts
    assert int(experts.num_local_experts) == LOCAL_EXPERTS, experts.num_local_experts
    gate_w, gate_s = _fp8_bank(LOCAL_EXPERTS, LOCAL_INTERMEDIATE, hidden, gen)
    up_w, up_s = _fp8_bank(LOCAL_EXPERTS, LOCAL_INTERMEDIATE, hidden, gen)
    down_w, down_s = _fp8_bank(LOCAL_EXPERTS, hidden, LOCAL_INTERMEDIATE, gen)
    for leaf, weight, grid in (("gate_proj_weight", gate_w, gate_s), ("up_proj_weight", up_w, up_s),
                               ("down_proj_weight", down_w, down_s)):
        _attach(experts, leaf, weight, grid, device)
    assert experts.prepare_scale_operands(
        experts.gate_proj_weight, experts.up_proj_weight, experts.down_proj_weight,
        getattr(experts, _grid_name("gate_proj_weight")), getattr(experts, _grid_name("up_proj_weight")),
        getattr(experts, _grid_name("down_proj_weight")),
    ) == 2
    # The shared expert's weights are stored ``[H, I]`` (gate, up) and ``[I, H]`` (down).
    s_gate, s_gate_s = case._fp8_weight(hidden, SHARED_INTERMEDIATE, gen)
    s_up, s_up_s = case._fp8_weight(hidden, SHARED_INTERMEDIATE, gen)
    s_down, s_down_s = case._fp8_weight(SHARED_INTERMEDIATE, hidden, gen)
    for leaf, weight, grid in (("gate_proj_weight", s_gate, s_gate_s), ("up_proj_weight", s_up, s_up_s),
                               ("down_proj_weight", s_down, s_down_s)):
        _attach(shared, leaf, weight, grid, device)
    assert shared.prepare_scale_operands(*shared.scale_route_operands()) == 3
    experts.router_weight = torch.nn.Parameter(
        (torch.randn(hidden, EXPERTS, generator=gen) * hidden ** -0.5).to(torch.bfloat16).to(device),
        requires_grad=False)
    experts.router_bias = torch.nn.Parameter(
        ((torch.rand(EXPERTS, generator=gen) - 0.5) * 0.1).to(device), requires_grad=False)


def build_head(cfg, seed: int, device):
    """The head under test: layer 45 with a real-shape attention half and this rank's MoE share."""
    tables = head_tables(cfg, seed, device)
    head = mtp.Glm5NextMultiTokenPredictor(
        cfg,
        embed_tokens=lambda: tables["embed_tokens_weight"],
        lm_head=lambda: tables["lm_head_weight"],
        world_size=1,
        tp_group=lambda: None,
    )
    for name in mtp.HEAD_PARAMETER_NAMES:
        setattr(head, name, torch.nn.Parameter(tables[name].clone(), requires_grad=False))
    block = head.block
    gen = torch.Generator().manual_seed(seed + 1)
    hidden = int(cfg.hidden_size)
    for name in ("input_layernorm_weight", "post_attention_layernorm_weight"):
        setattr(block, name, torch.nn.Parameter(
            (1.0 + torch.randn(hidden, generator=gen) * 0.05).to(device), requires_grad=False))
    block.self_attn = case.build_attention(model_fp8, cfg, seed=seed + 2, device=device)
    # ``_build_layer`` gives the block the undistributed 288-expert bank; the serve line's
    # rank holds 18 of them. The block's forward and the head's are unchanged by the swap.
    block.mlp = model_fp8.Glm5NextMoEBlock(cfg, world_size=1, ep_degree=EP_DEGREE)
    materialise_moe(block.mlp, cfg, seed + 3, device)
    return head


def step_inputs(cfg, seed: int, device) -> tuple:
    gen = torch.Generator().manual_seed(seed)
    hidden = int(cfg.hidden_size)
    return (
        torch.randn(1, hidden, generator=gen).to(torch.bfloat16).to(device),
        torch.randint(0, int(cfg.vocab_size), (1,), generator=gen, dtype=torch.int32).to(device),
        torch.full((1,), CONTEXT - 1, dtype=torch.int32).to(device),
    )


def block_operands(cfg, seed: int, device) -> tuple[tuple, dict]:
    """Layer 45's decode carrier at ctx 1024, with the request's next page allocated."""
    ops = case.decode_operands(cfg, CONTEXT, window_pages=WINDOW_PAGES, max_seq_len=MAX_SEQ_LEN, seed=seed)
    table = ops["block_table_row"]
    used = -(-CONTEXT // case.PAGE)
    if used < int(table.shape[0]):
        bank_pages = int(ops["latent_cache"].shape[0]) // case.PAGE
        free = sorted(set(range(bank_pages)) - set(table[table >= 0].tolist()))
        table[used, 0] = free[0]
    statics = {"softmax_scale": ops["softmax_scale"], "max_seq_len": ops["max_seq_len"],
               "page_size": ops["page_size"]}
    return tuple(ops[name].to(device) for name in BLOCK_NAMES), statics


# --- the graphs ----------------------------------------------------------------------

def make_step(head, qc, k: int, statics: dict):
    """``step(hidden [1,H], sampled [1], positions [1], *carrier) -> (ids [1,k], hiddens [k,1,H])``."""

    def step(hidden, sampled, positions, *flat):
        kwargs = dict(zip(BLOCK_NAMES, flat))
        kwargs.update(statics)
        collected: list = []
        ids = head.draft_tokens(
            hidden, sampled, positions, k,
            quant_config=qc, tp_degree=1, expert_parallel_rank=EXPERT_RANK,
            draft_collector=collected, **kwargs,
        )
        return ids, torch.stack(collected, 0)

    return step


def run_null(args, cfg, device) -> dict:
    torch._dynamo.reset()
    x = torch.randn(1, int(cfg.hidden_size)).to(torch.bfloat16).to(device)
    fn = (lambda t: t + t) if args.mode == "sim" else compiled(lambda t: t + t)
    started = time.perf_counter()
    _sync(fn(x))
    result = {"graph": "null", "first_call_s": time.perf_counter() - started}
    if args.mode == "device":
        result["timing"] = measure(fn, (x,), args.warmup, args.iterations)
    return result


def run_k(k: int, args, cfg, qc, device) -> dict:
    torch._dynamo.reset()
    head = build_head(cfg, args.seed, device)
    inputs = step_inputs(cfg, args.seed + 11, device)
    carrier, statics = block_operands(cfg, args.seed + 99, device)
    step = make_step(head, qc, k, statics)
    fn = step if args.mode == "sim" else compiled(step)
    operands = (*inputs, *carrier)
    started = time.perf_counter()
    ids, hiddens = fn(*operands)
    ids, hiddens = ids.cpu(), hiddens.cpu()
    first_call_s = time.perf_counter() - started
    if not torch.isfinite(hiddens.float()).all():
        raise AssertionError(f"k={k}: the draft's hidden rows are not finite")
    if tuple(ids.shape) != (1, k) or ids.dtype != torch.int32:
        raise AssertionError(f"k={k}: ids {tuple(ids.shape)} {ids.dtype}, expected (1, {k}) int32")
    # The margin the ids rest on, from the same bf16 head rows the route takes.
    head_rows = head._lm_head().cpu()
    logits = torch.nn.functional.linear(hiddens.reshape(k, -1), head_rows).float()
    top2 = logits.topk(2, dim=-1).values
    margins = ((top2[:, 0] - top2[:, 1]) / (logits.max(-1).values - logits.min(-1).values)).tolist()
    result = {
        "graph": "draft_tokens", "k": k, "batch": 1, "ctx": CONTEXT, "window_rows": WINDOW_PAGES * case.PAGE,
        "max_seq_len": MAX_SEQ_LEN, "first_call_s": first_call_s,
        "ids": ids.tolist(), "margin_share": margins,
        "hidden_rows": hiddens.reshape(k, -1).float().tolist(),
        "unit": f"one fused draft of k={k} iterations (layer 45 attention + MoE halves, head, token) for 1 request",
    }
    if args.mode == "device":
        result["timing"] = measure(fn, operands, args.warmup, args.iterations)
    del fn, step, head
    return result


def run_ops(args, cfg, qc, device) -> dict:
    """One draft iteration at B=1 as separately compiled pieces, each fed the tensors the
    previous piece produced (so the MoE routes real activations), timed one by one.

    The tail pieces are the two authored kernels (``functional/mtp/``): K1'
    ``mtp_tail_in`` (embedding gather, mask, ``enorm``/``hnorm``, concat, ``eh_proj``) and
    K2 ``mtp_tail_out`` (residual add, shared-head norm, this rank's 2420 head rows,
    ``(max, argmax)``). K1' is timed twice: as the chain runs it at one rank (the whole
    ``[H, 2H]`` ``eh_proj``, the same bytes Stage A's replicated tail streamed, so the
    chain compares like for like) and standalone at the served per-rank shape (the
    64-row shard, the kernel's own entitlement; its all-gather needs the 64-rank group
    and is the gate trace's). Each weight-streaming piece carries its bytes and its
    entitlement, ``bytes / HBM_BYTES_PER_US_PER_CORE``; the device run adds the measured
    median and the ratio. Each piece also carries its own launch floor (the ``null``
    graph), so the floor is reported beside the raw medians; the fused k=1 graph timed
    on the same inputs is the whole the pieces are compared with. At one rank the token
    route is the whole-head branch of ``draft_token_ids`` (the gather elided).
    """
    from vllm_neuron.functional.draft_token import draft_token_ids
    from vllm_neuron.functional.mtp import tail_in, tail_out

    torch._dynamo.reset()
    head = build_head(cfg, args.seed, device)
    hidden, sampled, positions = step_inputs(cfg, args.seed + 11, device)
    carrier, statics = block_operands(cfg, args.seed + 99, device)
    ffn_keywords = dict(quant_config=qc, block_size=None, moe_group=None, tp_degree=1,
                        expert_parallel_rank=EXPERT_RANK)
    gain = head.shared_head_norm_weight
    lm_head = head._lm_head()
    table = head._embed_tokens()
    eps = float(cfg.rms_norm_eps)
    shard_rows = tail_in.eh_proj_shard_rows(int(cfg.hidden_size), TP_WORLD)
    eh_proj_shard = head.eh_proj_weight[:shard_rows].contiguous()

    def layer_input(token, previous, pos):  # K1' as the chain runs it at one rank: whole eh_proj
        return head._layer_input(token, previous, pos)

    def layer_input_shard(token, previous, pos):  # K1' at the served per-rank shape: 64 rows
        return tail_in.mtp_tail_in(token, table, pos, previous, head.enorm_weight,
                                   head.hnorm_weight, eh_proj_shard, eps=eps)

    def attention(x, *flat):  # layer 45's attention half, the trunk's DSA kernels
        kwargs = dict(zip(BLOCK_NAMES, flat))
        kwargs.update(statics)
        return head.block(x, **kwargs)

    def ffn(attended):  # post-attention norm + MoE (router, routed bank, shared expert) + reduce
        return head._ffn_half(attended, **ffn_keywords)

    def tail(attended, ffn_out):  # K2: residual add + shared-head norm + 2420-row logits + (max, argmax)
        return tail_out.mtp_tail_out(attended, ffn_out, gain, lm_head, eps=eps)

    def select(pair):  # the owner select; the TP gather elided at one rank (whole-head branch)
        return draft_token_ids(pair, shard_rows=int(lm_head.shape[0]), vocab_size=int(cfg.vocab_size),
                               group=None)

    weight_bytes = {
        "3 K1' mtp_tail_in (whole eh_proj, as the one-rank chain runs it)": head.eh_proj_weight.numel() * 2,
        "3s K1' mtp_tail_in (64-row eh_proj shard, served per-rank)": eh_proj_shard.numel() * 2,
        "4 K2 mtp_tail_out (residual+norm+2420-row logits+(max,argmax))": lm_head.numel() * 2,
    }
    pieces = [("3 K1' mtp_tail_in (whole eh_proj, as the one-rank chain runs it)", layer_input),
              ("3s K1' mtp_tail_in (64-row eh_proj shard, served per-rank)", layer_input_shard),
              ("1 attention half", attention),
              ("2 _ffn_half (norm+MoE+reduce)", ffn),
              ("4 K2 mtp_tail_out (residual+norm+2420-row logits+(max,argmax))", tail),
              ("5 draft_token_ids (owner select, gather elided at one rank)", select)]
    wrap = (lambda f: f) if args.mode == "sim" else compiled
    fns = {name: wrap(fn) for name, fn in pieces}
    chain = {}
    chain[pieces[0][0]] = (sampled, hidden, positions)
    chain[pieces[1][0]] = (sampled, hidden, positions)
    results = []
    for name, _ in pieces:
        fn = fns[name]
        inputs = chain[name]
        started = time.perf_counter()
        out = fn(*inputs)
        _sync(out)
        first_call_s = time.perf_counter() - started
        if name.startswith("3 "):
            chain["1 attention half"] = (out, *carrier)
        elif name.startswith("3s"):
            if tuple(out.shape) != (1, shard_rows):
                raise AssertionError(f"the shard piece returned {tuple(out.shape)}, expected (1, {shard_rows})")
        elif name.startswith("1"):
            attended = out
            chain["2 _ffn_half (norm+MoE+reduce)"] = (attended,)
        elif name.startswith("2"):
            chain[pieces[4][0]] = (attended, out)
        elif name.startswith("4"):
            chain[pieces[5][0]] = (out[1],)
        result = {"op": name, "first_call_s": first_call_s}
        if name in weight_bytes:
            result["weight_bytes"] = weight_bytes[name]
            result["entitlement_us"] = weight_bytes[name] / HBM_BYTES_PER_US_PER_CORE
        if args.mode == "device":
            result["timing"] = measure(fn, inputs, args.warmup, args.iterations)
            if name in weight_bytes:
                result["ratio_to_entitlement"] = result["timing"]["median_us"] / result["entitlement_us"]
        results.append(result)
    token = int(out.cpu().reshape(-1)[0])
    # The tail pair (K1' at the served shard + K2) against its summed entitlement: the bar
    # mtp-head.md 9.2 states for the two kernels together (x1.5 of the sum).
    pair = [r for r in results if r["op"].startswith(("3s", "4 "))]
    tail_pair = {"ops": [r["op"] for r in pair],
                 "entitlement_us": sum(r["entitlement_us"] for r in pair)}
    if args.mode == "device":
        tail_pair["median_us"] = sum(r["timing"]["median_us"] for r in pair)
        tail_pair["ratio_to_entitlement"] = tail_pair["median_us"] / tail_pair["entitlement_us"]
    del fns, head
    return {"graph": "ops", "batch": 1, "ctx": CONTEXT, "window_rows": WINDOW_PAGES * case.PAGE,
            "pieces": results, "tail_pair": tail_pair, "token": token,
            "hbm_bytes_per_us_per_core": HBM_BYTES_PER_US_PER_CORE,
            "unit": "one draft iteration at B=1 as separately compiled graphs on the same inputs"}


def compare(device_path: Path, sim_path: Path) -> dict:
    """Per ``k``: ids bit-equal?, hidden rows' relative L2, the margins both sides saw."""
    dev, sim = json.loads(device_path.read_text()), json.loads(sim_path.read_text())
    by_k = lambda report: {c["k"]: c for c in report["cases"] if c.get("graph") == "draft_tokens"}
    d, s = by_k(dev), by_k(sim)
    out = {}
    for k in sorted(set(d) & set(s)):
        hd, hs = torch.tensor(d[k]["hidden_rows"]), torch.tensor(s[k]["hidden_rows"])
        out[str(k)] = {
            "ids_device": d[k]["ids"], "ids_sim": s[k]["ids"],
            "ids_bit_equal": d[k]["ids"] == s[k]["ids"],
            "hidden_rel_l2_per_iteration": [rel_l2(hd[i], hs[i]) for i in range(hd.shape[0])],
            "margin_share_device": d[k]["margin_share"], "margin_share_sim": s[k]["margin_share"],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("sim", "device"), default="device")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--null", action="store_true", help="also time the launch-floor graph")
    parser.add_argument("--ops", action="store_true",
                        help="also time one draft iteration as separately compiled pieces (the two tail kernels among them)")
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--q-lora", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=7_000)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("DEVICE_JSON", "SIM_JSON"))
    args = parser.parse_args()
    if args.compare:
        print(json.dumps(compare(*args.compare), indent=1))
        return
    if args.output is None:
        sys.exit("--output is required unless --compare is given")
    if args.mode == "sim" and os.environ.get("NKI_SIMULATOR") != "1":
        sys.exit("--mode sim needs NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1")
    device = "cpu" if args.mode == "sim" else "neuron:0"
    cfg = build_config(args.hidden, args.q_lora)
    qc = quant_config()
    report = {
        "what": __doc__.split("\n\n")[0],
        "mode": args.mode, "device": device, "tree": str(ROOT),
        "environment": {k: os.environ.get(k) for k in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG", "NEURON_LIBTORCH_CACHE_ROOT",
            "NKI_SIMULATOR", "VLLM_NEURON_CPU_MODE", "NEURON_CC_FLAGS")},
        "compiler_args": compiler_args() if args.mode == "device" else None,
        "cores": "neuron:0 = 1 logical core (LNC2) of the leased slice" if args.mode == "device" else "cpu",
        "args": vars(args) | {"output": str(args.output), "compare": None},
        "config": {"hidden_size": int(cfg.hidden_size), "q_lora_rank": int(cfg.q_lora_rank),
                   "kv_lora_rank": int(cfg.kv_lora_rank), "num_attention_heads": int(cfg.num_attention_heads),
                   "index_n_heads": int(cfg.index_n_heads), "index_head_dim": int(cfg.index_head_dim),
                   "index_kpool": int(cfg.index_kpool), "index_topk": int(cfg.index_topk),
                   "vocab_size_as_built": int(cfg.vocab_size), "vocab_size_model": VOCAB,
                   "rms_norm_eps": float(cfg.rms_norm_eps),
                   "index_share_for_mtp_iteration": bool(cfg.index_share_for_mtp_iteration),
                   "quant_method": str(getattr(qc, "method", None))},
        "moe_geometry": {"experts": EXPERTS, "ep_degree": EP_DEGREE, "local_experts": LOCAL_EXPERTS,
                         "local_intermediate": LOCAL_INTERMEDIATE, "shared_intermediate": SHARED_INTERMEDIATE,
                         "expert_rank": EXPERT_RANK, "top_k": int(cfg.num_experts_per_tok)},
        "cases": [],
    }

    def flush():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1, default=str))

    if args.null:
        print("[mtp-bench] null", file=sys.stderr, flush=True)
        report["cases"].append(run_null(args, cfg, device))
        flush()
    for k in args.ks:
        print(f"[mtp-bench] draft_tokens k={k}", file=sys.stderr, flush=True)
        report["cases"].append(run_k(k, args, cfg, qc, device))
        flush()
    if args.ops:
        print("[mtp-bench] ops", file=sys.stderr, flush=True)
        ops = run_ops(args, cfg, qc, device)
        k1 = next((c for c in report["cases"] if c.get("graph") == "draft_tokens" and c["k"] == 1), None)
        if k1 is not None and ops["token"] != int(k1["ids"][0][0]):
            raise AssertionError(f"the pieces drafted {ops['token']}; the fused k=1 graph {k1['ids'][0][0]}")
        report["cases"].append(ops)
        flush()
    if args.mode == "device":
        timed = {c["k"]: c["timing"]["median_us"] for c in report["cases"] if c.get("graph") == "draft_tokens"}
        ks = sorted(timed)
        report["derived"] = {
            "median_us_by_k": {str(k): timed[k] for k in ks},
            "marginal_us": {f"{a}->{b}": (timed[b] - timed[a]) / (b - a) for a, b in zip(ks, ks[1:])},
        }
        flush()
    summary = [{k: v for k, v in c.items() if k != "hidden_rows"} for c in report["cases"]]
    print(json.dumps({"cases": summary, "derived": report.get("derived")}, indent=1, default=str))


if __name__ == "__main__":
    main()
