# SPDX-License-Identifier: Apache-2.0
"""Time the GLM-5.3-Flash MTP draft head (``mtp.py``) at the per-rank TP=64 decode shape.

Run it through the device lease (``devlease.py slice dense -- ...``); this script does
not select cores. Three graphs are built, each at B in ``--batches`` requests:

* ``null``        -- one bf16 add on a [B, H] row: the graph-launch floor of this harness.
* ``head_only``   -- the head's own arithmetic as ``Glm5NextMTPLayer.forward`` and
  ``propose_draft_tokens`` write it, with the block replaced by the identity: mask at
  position 0, enorm, hnorm, concat, ``eh_proj`` ([H, 2H] bf16, replicated), shared-head
  norm, ``compute_draft_logits`` against this rank's vocab shard ([2420, H] bf16) and the
  ``argmax`` of that shard.
* ``head_block``  -- the same with the real block: ``Glm5NextDSALayer`` (one MLA head,
  latent 512, the indexer at 32 heads, fp8 projections with a 128x128 scale grid) built by
  ``dsa_decode_case.build_attention``, at ctx 1024 in the 2048-row window the serving
  line decodes in (the indexer bypass regime). One forward per request, as the layer takes
  one request; the B forwards sit in one graph.

Each graph is compiled once with one copy of the head and once with ``--chain`` copies
(distinct weights and caches, copy ``i`` feeding its draft hidden state to copy ``i+1``
as ``previous_hidden_states``), and both are timed; the per-step cost is reported two
ways: ``single_launch_us`` (the one-copy graph, what a separate draft NEFF would cost per
step) and ``marginal_us`` (the slope ``(t_chain - t_1) / (chain - 1)``, what the head
costs once inside a graph that already launched).

Correctness: every graph's draft hidden state and shard logits are compared with an
independently written fp32 torch head over the same block run eagerly on CPU
(``rel_l2``), before any timing. With ``--mode sim`` (``NKI_SIMULATOR=1
VLLM_NEURON_CPU_MODE=1``) the graphs run eagerly through the NKI simulator and only the
comparison is reported; narrow ``--hidden``/``--q-lora`` there, as the DSA tests do,
because the simulator's projection cost scales with both and neither changes the head's
arithmetic.

The MoE half of layer 45 (288 routed experts, the shared expert, the post-attention
norm) is NOT part of either graph because ``Glm5NextMTPLayer`` never calls it: the block
it is handed is ``Glm5NextDSALayer``, whose forward is the attention half only. The
report adds that half from the MoE and dense microbenchmarks and says so.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import torch  # noqa: E402

import vllm_neuron  # noqa: E402,F401 -- registers the Neuron compilation backend
from vllm_neuron.model.glm5_next import model_fp8, mtp  # noqa: E402
from vllm_neuron.model.glm5_next.config import DSA_LAYER_TYPE  # noqa: E402
from test.vllm_neuron.functional.dsa import dsa_decode_case as case  # noqa: E402

MAX_SEQ_LEN = 4096
WINDOW_PAGES = 16  # 2048 rows: the serving line's decode context bucket (bypass regime)
CONTEXT = 1024
TP_WORLD = 64
VOCAB = 154880
MTP_LAYER_IDX = 45
#: The fp32 reference against a bf16 graph: the head is three norms, one 8192-wide bf16
#: GEMV and a vocab GEMV; the block adds the DSA layer's own fp8-vs-fp32 gap
#: (``benchmark_dsa_decode.LAYER_REL_L2``).
HEAD_ONLY_REL_L2 = 2e-2
HEAD_BLOCK_REL_L2 = 5e-2


def rel_l2(got: torch.Tensor, want: torch.Tensor) -> float:
    got, want = got.detach().cpu().double(), want.detach().cpu().double()
    return float((got - want).norm() / want.norm().clamp_min(1e-30))


def compiled(fn):
    return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False,
                         options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")})


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


def _sync(out):
    outs = out if isinstance(out, (tuple, list)) else (out,)
    return [o.to("cpu") for o in outs if torch.is_tensor(o)]


# --- building ------------------------------------------------------------------------

def head_weights(cfg, seed: int, device) -> dict:
    """The four MTP tensors at the checkpoint's shapes and dtypes, plus this rank's head shard."""
    gen = torch.Generator().manual_seed(seed)
    hidden = int(cfg.hidden_size)
    shard = int(cfg.vocab_size) // TP_WORLD
    w = {
        "enorm_weight": (1.0 + torch.randn(hidden, generator=gen) * 0.05).to(torch.bfloat16),
        "hnorm_weight": (1.0 + torch.randn(hidden, generator=gen) * 0.05).to(torch.bfloat16),
        "eh_proj_weight": (torch.randn(hidden, 2 * hidden, generator=gen)
                           * (2 * hidden) ** -0.5).to(torch.bfloat16),
        "shared_head_norm_weight": (1.0 + torch.randn(hidden, generator=gen) * 0.05).to(torch.bfloat16),
        "lm_head_shard": (torch.randn(shard, hidden, generator=gen) * hidden ** -0.5).to(torch.bfloat16),
    }
    return {k: v.to(device) for k, v in w.items()}


class IdentityBlock(torch.nn.Module):
    """Stands in for the block in ``head_only``: the head's own cost, nothing else."""

    def forward(self, hidden_states, **_):
        return hidden_states


def build_block(cfg, seed: int, device):
    """``Glm5NextDSALayer`` 45 with a prepared real-shape attention and its input norm."""
    layer = model_fp8._build_layer(cfg, MTP_LAYER_IDX, DSA_LAYER_TYPE, 1)
    gen = torch.Generator().manual_seed(seed)
    layer.input_layernorm_weight = torch.nn.Parameter(
        (1.0 + torch.randn(int(cfg.hidden_size), generator=gen) * 0.05), requires_grad=False)
    layer.self_attn = case.build_attention(model_fp8, cfg, seed=seed + 1, device=device)
    layer.to(device)
    return layer


def build_head(cfg, block, weights: dict, device):
    head = mtp.Glm5NextMultiTokenPredictor(cfg, 1, [block])
    layer = head.layers[str(MTP_LAYER_IDX)]
    for name in ("enorm_weight", "hnorm_weight", "eh_proj_weight", "shared_head_norm_weight"):
        setattr(layer, name, torch.nn.Parameter(weights[name].clone().to(device), requires_grad=False))
    return head.to(device)


def step_inputs(cfg, seed: int, batch: int, device) -> dict:
    gen = torch.Generator().manual_seed(seed)
    hidden = int(cfg.hidden_size)
    return {
        "inputs_embeds": (torch.randn(batch, hidden, generator=gen) * 0.5).to(torch.bfloat16).to(device),
        "previous_hidden_states": (torch.randn(batch, hidden, generator=gen)).to(torch.bfloat16).to(device),
        "positions": torch.full((batch,), CONTEXT - 1, dtype=torch.int64).to(device),
    }


BLOCK_NAMES = ("latent_cache", "pool_cache", "seq_lens", "start_position", "block_table_row",
               "latent_slots", "tail", "position")


def block_operands(cfg, seed: int, device) -> tuple[dict, dict]:
    ops = case.decode_operands(cfg, CONTEXT, window_pages=WINDOW_PAGES, max_seq_len=MAX_SEQ_LEN,
                               seed=seed)
    statics = {"softmax_scale": ops["softmax_scale"], "max_seq_len": ops["max_seq_len"],
               "page_size": ops["page_size"]}
    tensors = {name: ops[name].to(device) for name in BLOCK_NAMES}
    return tensors, statics


# --- the graphs ----------------------------------------------------------------------

def head_graph(heads, shards, statics, batch: int, with_block: bool):
    """``step(embeds, previous, positions, per-copy-per-request block operands...)``.

    Returns ``(draft_hidden_last_copy [B, H], shard_logits_last_copy [B, shard],
    local_argmax [B])``.
    """
    n = len(BLOCK_NAMES)

    def step(embeds, previous, positions, *flat):
        hidden = previous
        at = 0
        logits = None
        for head, shard in zip(heads, shards):
            rows = []
            for b in range(batch):
                kwargs = {}
                if with_block:
                    kwargs = dict(zip(BLOCK_NAMES, flat[at:at + n]))
                    kwargs.update(statics)
                    at += n
                rows.append(head(
                    inputs_embeds=embeds[b:b + 1],
                    previous_hidden_states=hidden[b:b + 1],
                    positions=positions[b:b + 1],
                    **kwargs,
                ))
            hidden = torch.cat(rows, 0)
            logits = head.compute_draft_logits(hidden, shard)
        return hidden, logits, logits.argmax(dim=-1).to(torch.int32)

    return step


@contextlib.contextmanager
def cpu_simulator():
    """Run the CPU reference's NKI kernels through the simulator, also in ``--mode device``.

    ``libtorch_neuronx_lite.nki.nki_hop._cpu_impl`` reads ``NKI_SIMULATOR`` at call time and
    refuses a CPU dispatch without it; the compiled device graph is built and executed outside
    this block, so the flag does not reach it.
    """
    before = os.environ.get("NKI_SIMULATOR")
    os.environ["NKI_SIMULATOR"] = "1"
    try:
        yield
    finally:
        if before is None:
            del os.environ["NKI_SIMULATOR"]
        else:
            os.environ["NKI_SIMULATOR"] = before


def reference(weights_per_copy: list[dict], blocks_cpu: list, inputs: dict, block_ops: list[list[dict]],
              statics: dict, eps: float, with_block: bool):
    """fp32 torch head, independently written, over the same block run eagerly on CPU."""

    def rms(x, gain):
        x = x.to(torch.float32)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gain.to(torch.float32)

    embeds = inputs["inputs_embeds"].cpu()
    hidden = inputs["previous_hidden_states"].cpu()
    positions = inputs["positions"].cpu()
    logits = None
    for copy, (w, block) in enumerate(zip(weights_per_copy, blocks_cpu)):
        e = torch.where((positions.unsqueeze(-1) == 0), torch.zeros_like(embeds), embeds)
        e = rms(e, w["enorm_weight"].cpu()).to(torch.bfloat16)
        h = rms(hidden, w["hnorm_weight"].cpu()).to(torch.bfloat16)
        joined = torch.cat([e, h], -1).to(torch.float32)
        proj = (joined @ w["eh_proj_weight"].cpu().to(torch.float32).t()).to(torch.bfloat16)
        rows = []
        for b in range(proj.shape[0]):
            if with_block:
                ops = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in block_ops[copy][b].items()}
                out = block.forward(proj[b:b + 1], **ops, **statics)
            else:
                out = proj[b:b + 1]
            rows.append(out)
        block_out = torch.cat(rows, 0)
        hidden = rms(block_out, w["shared_head_norm_weight"].cpu()).to(torch.bfloat16)
        logits = (hidden.to(torch.float32) @ w["lm_head_shard"].cpu().to(torch.float32).t())
    return hidden, logits


def run_case(label: str, batch: int, chain: int, args, cfg, device) -> dict:
    with_block = label == "head_block"
    torch._dynamo.reset()
    result = {"graph": label, "batch": batch, "ctx": CONTEXT, "window_rows": WINDOW_PAGES * case.PAGE,
              "max_seq_len": MAX_SEQ_LEN, "hidden": int(cfg.hidden_size), "q_lora_rank": int(cfg.q_lora_rank),
              "vocab_shard_rows": int(cfg.vocab_size) // TP_WORLD, "tp_world": TP_WORLD,
              "unit": "one draft step (one MTP layer + shard logits + local argmax) for all `batch` requests"}
    if label == "null":
        x = (torch.randn(batch, int(cfg.hidden_size))).to(torch.bfloat16).to(device)
        fn = (lambda t: t + t) if args.mode == "sim" else compiled(lambda t: t + t)
        started = time.perf_counter(); _sync(fn(x)); result["first_call_s"] = time.perf_counter() - started
        if args.mode == "device":
            result["timing"] = measure(fn, (x,), args.warmup, args.iterations)
        return result

    timings = {}
    for copies in sorted({1, chain}):
        weights = [head_weights(cfg, 7_000 + 13 * i, device) for i in range(copies)]
        blocks_dev = [build_block(cfg, 9_000 + 31 * i, device) if with_block else IdentityBlock()
                      for i in range(copies)]
        blocks_cpu = [build_block(cfg, 9_000 + 31 * i, "cpu") if with_block else IdentityBlock()
                      for i in range(copies)]
        heads = [build_head(cfg, blocks_dev[i], weights[i], device) for i in range(copies)]
        shards = [w["lm_head_shard"] for w in weights]
        inputs = step_inputs(cfg, 11_000, batch, device)
        flat, block_ops, statics = [], [], {}
        for i in range(copies):
            per_req = []
            for b in range(batch):
                tensors, statics = block_operands(cfg, 99 + 7 * i + 1009 * b, device)
                per_req.append({k: v.cpu() for k, v in tensors.items()})
                if with_block:
                    flat.extend(tensors[name] for name in BLOCK_NAMES)
            block_ops.append(per_req)
        step = head_graph(heads, shards, statics, batch, with_block)
        fn = step if args.mode == "sim" else compiled(step)
        operands = (inputs["inputs_embeds"], inputs["previous_hidden_states"], inputs["positions"], *flat)
        started = time.perf_counter()
        hidden, logits, tokens = fn(*operands)
        hidden, logits, tokens = hidden.cpu(), logits.cpu(), tokens.cpu()
        first_call_s = time.perf_counter() - started
        eps = float(cfg.rms_norm_eps)
        with cpu_simulator():
            ref_hidden, ref_logits = reference(weights, blocks_cpu, inputs, block_ops, statics, eps, with_block)
        check = {"hidden_rel_l2": rel_l2(hidden, ref_hidden), "shard_logits_rel_l2": rel_l2(logits, ref_logits),
                 "argmax_agrees": int((tokens.to(torch.int64) == ref_logits.argmax(-1)).sum()), "rows": batch}
        bound = HEAD_BLOCK_REL_L2 if with_block else HEAD_ONLY_REL_L2
        if not torch.isfinite(hidden.float()).all() or check["hidden_rel_l2"] > bound:
            raise AssertionError(f"{label} B={batch} copies={copies}: vs CPU reference {check} > {bound}")
        entry = {"first_call_s": first_call_s, "check": check}
        if args.mode == "device":
            entry["timing"] = measure(fn, operands, args.warmup, args.iterations)
        timings[copies] = entry
        del fn, step, heads, blocks_dev, blocks_cpu
    result["copies"] = {str(k): v for k, v in timings.items()}
    if args.mode == "device" and chain > 1:
        t1 = timings[1]["timing"]["median_us"]
        tn = timings[chain]["timing"]["median_us"]
        result["single_launch_us"] = t1
        result["marginal_us"] = (tn - t1) / (chain - 1)
        result["chain"] = chain
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("sim", "device"), default="device")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--graphs", nargs="+", default=["null", "head_only", "head_block"])
    parser.add_argument("--chain", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--q-lora", type=int, default=1536)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "sim" and os.environ.get("NKI_SIMULATOR") != "1":
        sys.exit("--mode sim needs NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1")
    device = "cpu" if args.mode == "sim" else "neuron:0"
    cfg = case.decode_config(hidden_size=args.hidden, q_lora_rank=args.q_lora)
    report = {
        "what": __doc__.split("\n\n")[0],
        "mode": args.mode, "device": device, "tree": str(ROOT),
        "environment": {k: os.environ.get(k) for k in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG", "NEURON_LIBTORCH_CACHE_ROOT",
            "NKI_SIMULATOR", "VLLM_NEURON_CPU_MODE", "NEURON_CC_FLAGS")},
        "cores": "neuron:0 = 1 logical core (LNC2: 2 physical cores) of the leased slice" if args.mode == "device" else "cpu",
        "args": vars(args) | {"output": str(args.output)},
        "config": {"hidden_size": int(cfg.hidden_size), "q_lora_rank": int(cfg.q_lora_rank),
                   "kv_lora_rank": int(cfg.kv_lora_rank), "num_attention_heads": int(cfg.num_attention_heads),
                   "index_n_heads": int(cfg.index_n_heads), "index_head_dim": int(cfg.index_head_dim),
                   "index_kpool": int(cfg.index_kpool), "index_topk": int(cfg.index_topk),
                   "vocab_size": int(cfg.vocab_size), "rms_norm_eps": float(cfg.rms_norm_eps)},
        "cases": [],
    }
    for label in args.graphs:
        for batch in args.batches:
            print(f"[mtp-bench] {label} B={batch}", file=sys.stderr, flush=True)
            report["cases"].append(run_case(label, batch, args.chain, args, cfg, device))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report["cases"], indent=1, default=str))


if __name__ == "__main__":
    main()
