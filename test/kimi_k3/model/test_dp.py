# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 attention data parallel on CPU ranks: two replicas of TP4 against one.

A 4-layer model at hidden 7,168 (a dense first layer, then MoE with latent 512, expert width
3,072 and 64 experts; three KDA layers with 12 heads, then MLA) is loaded from one set of
random weights on 8 gloo ranks, TP4 by DP2 with EP8, and on 4 ranks, TP4 by DP1 with EP4,
which run each replica's tokens in turn. Each replica takes a decode step over its own cache
pages and a prefill chunk. The first MoE layer takes the same input in both worlds, so its
combined rows and tail pack differ only by sum order. Past it, a bf16 cast and the expert
kernel's 8-bit input rounding let the prefill outputs drift, so those are not compared. In the
8-rank world one replica decodes while the idle one runs the runner's dummy batch on pages of
random values. The error is the largest difference over the largest magnitude of the
one-replica run. Set ``KIMI_K3_CHECKPOINT`` to the checkpoint directory (its config only);
run with ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1``.
"""

import os
import types
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

TP, DP, H, K, E, I = 4, 2, 7168, 512, 64, 3072
HEADS = 3 * TP  # KDA and MLA heads, 3 per rank
T, P = 4, 256  # decode and prefill tokens per replica; KDA prefill takes multiples of 256
BLOCK, CONTEXT, VOCAB = 256, 8192, 10240
CONV, STATE, ROW = (3, 1152), (3, 128, 128), 576  # per page: KDA conv, KDA state, MLA row
PAGE = BLOCK * ROW * 2
PREFIX, BLOCKS = [37, 300, 5, 200], [[1], [2, 3], [4], [5]]  # each request's cache
DUMMY_BATCHES = (16, 32)  # the KDA decode kernel takes at most 32 tokens
SEAM, LOGITS = 8e-6, 1e-5


def _model() -> m.KimiK3ForConditionalGeneration:
    """A 4-layer model on meta at the test widths and this process's groups."""
    from vllm.transformers_utils.config import get_config

    config = get_config(str(CHECKPOINT), trust_remote_code=False).to_dict()
    text = config["text_config"]
    text.update(num_hidden_layers=4, hidden_size=H, num_attention_heads=HEADS, num_experts=E,
                intermediate_size=1056 * TP, moe_intermediate_size=I,
                routed_expert_hidden_size=K, vocab_size=VOCAB, first_k_dense_replace=1)
    text["linear_attn_config"].update(num_heads=HEADS, kda_layers=[1, 2, 3], full_attn_layers=[4])
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
    gens = {layer: torch.Generator().manual_seed(13 + layer) for layer in (1, 2, 3)}
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


def _views(pages: torch.Tensor, kda: bool) -> list[torch.Tensor]:
    """A layer's cache views of its pages, laid out as the runner's unified page."""
    n, conv = pages.shape[0], CONV[0] * CONV[1] * 2
    if kda:
        state = conv + STATE[0] * STATE[1] * STATE[2] * 4
        return [pages[:, :conv].view(torch.bfloat16).view(n, *CONV),
                pages[:, conv:state].view(torch.float32).view(n, *STATE), pages]
    return [pages.view(torch.bfloat16).view(n, 1, BLOCK, ROW)]


def _caches(replica: int, tp: int, phase: str, lengths=PREFIX, blocks=BLOCKS) -> dict:
    """Each layer's own pages: random states and cached rows at decode, empty at prefill."""
    caches, n = {}, len(lengths)
    for i in range(4):
        cache = _views(torch.zeros(max(map(max, blocks)) + 1, PAGE, dtype=torch.int8), i < 3)
        gen = torch.Generator().manual_seed(1000 * replica + 10 * i + (tp if i < 3 else 0))
        if phase == "decode" and i < 3:
            cache[0][1:n + 1] = (0.5 * torch.randn(n, *CONV, generator=gen)).bfloat16()
            cache[1][1:n + 1] = 0.05 * torch.randn(n, *STATE, generator=gen)
        for own, length in zip(blocks, lengths) if phase == "decode" and i == 3 else ():
            rows = (0.5 * torch.randn(len(own) * BLOCK, ROW, generator=gen)).bfloat16()
            cache[0][own, 0] = rows.index_fill(0, torch.arange(length, len(own) * BLOCK), 0) \
                .view(len(own), BLOCK, ROW)
        caches[f"layers.{i}.self_attn"] = cache
    return caches


def _inputs(replica: int, phase: str, lengths=PREFIX, blocks=BLOCKS) -> tuple:
    """A replica's token ids, positions and per-layer metadata for one step."""
    gen, n = torch.Generator().manual_seed(100 * replica + (phase == "prefill")), len(lengths)
    if phase == "decode":
        width = max(map(len, blocks))
        kda = torch.arange(1, n + 1, dtype=torch.int32).view(n, 1)
        mla = torch.tensor([own + [-1] * (width - len(own)) for own in blocks], dtype=torch.int32)
        slots = [own[length // BLOCK] * BLOCK + length % BLOCK
                 for own, length in zip(blocks, lengths)]
        common, positions = {"slot_mapping": torch.tensor(slots), "max_query_len": 1}, \
            torch.tensor(lengths)
    else:
        kda, mla = (torch.tensor(t, dtype=torch.int32) for t in ([[1]], [[1, -1]]))
        common, n, positions = {"slot_mapping": torch.arange(P) + BLOCK, "max_query_len": P,
                                "cached_seq_len": torch.zeros(1, 1, dtype=torch.int32)}, P, \
            torch.arange(P)
    meta = {f"layers.{i}.self_attn": {"block_table_tensor": kda if i < 3 else mla,
                                      "decode_token_threshold": 1, **common} for i in range(4)}
    return torch.randint(0, VOCAB, (n,), generator=gen), positions, meta


def _dummy_inputs(batch: int, rank: torch.Tensor) -> dict:
    """The decode inputs the runner's dummy batch builds for ``batch`` requests."""
    from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

    spec = types.SimpleNamespace(block_size=BLOCK)
    groups = [types.SimpleNamespace(kv_cache_spec=spec, layer_names=[f"layers.{i}.self_attn"])
              for i in range(4)]
    runner = object.__new__(NeuronModelRunner)
    runner.__dict__.update(
        device=torch.device("cpu"), drafter=None, max_model_len=16 * CONTEXT, _dcp_size=1,
        kv_cache_config=types.SimpleNamespace(kv_cache_groups=groups), rank_tensor=rank,
        neuron_config=types.SimpleNamespace(decode_context_length_buckets=[CONTEXT],
                                            enable_structured_outputs=False),
        speculative_config=None, uses_mrope=False, on_device_sampling=False,
        enable_prompt_embeds=False, encoder_neuron_config=None, use_async_scheduling=False,
        model=None, _use_async_reorder=lambda: False)
    return runner._build_decode_synthetic_inputs(batch, decode_token_threshold=1,
                                                 ctx_bucket=CONTEXT)


def _rank(rank: int, root: Path, world: str) -> None:
    """One gloo rank of ``dp2`` or ``dp1``: each step's seams, and the idle replica's bytes."""
    from vllm_neuron.parallel.neuron_parallel_state import (
        initialize_neuron_parallel_state, stamp_spmd_rank_tensors)

    size, (replica, tp) = (TP * DP if world == "dp2" else TP), divmod(rank, TP)
    dist.init_process_group("gloo", init_method=f"file://{root}/rdzv-{world}", rank=rank,
                            world_size=size)
    if world == "dp2":
        initialize_neuron_parallel_state(ep_degree=size, dp_size=DP, local_rank=rank,
                                         tp_global_ranks=[replica * TP + i for i in range(TP)])
    else:
        initialize_neuron_parallel_state(tp_global_ranks=list(range(TP)), local_rank=rank)
    stamped = stamp_spmd_rank_tensors("cpu")
    model, seen, out = _model(), {}, {}
    _load(model, torch.load(root / "weights.pt", mmap=True), torch.load(root / "experts.pt"), tp)
    rank_t = torch.tensor(tp, dtype=torch.int32)

    def tap(key, fn):
        return lambda *a, **k: seen.setdefault(key, []).append(fn(*a, **k)) or seen[key][-1]

    def packed(self, pack):
        seen.setdefault("pack", []).append(pack.clone())
        return output(self, pack)

    output = m.KimiK3MoE.prefill_output
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(m, "_expert_sum", tap("routed", m._expert_sum))
        patch.setattr(m.KimiK3MoE, "_combine", tap("combined", m.KimiK3MoE._combine))
        patch.setattr(m.KimiK3MoE, "prefill_output", packed)
        for name in ("forward_decode", "forward_prefill"):
            layer = m.KimiK3DecoderLayer
            patch.setattr(layer, name, tap("layer", getattr(layer, name)))
        for d in [replica] if world == "dp2" else range(DP):
            for phase in ("decode", "prefill"):
                model.bind_kv_cache(_caches(d, tp, phase))
                ids, positions, meta = _inputs(d, phase)
                seen.clear()
                hidden = model.model(ids, meta, phase == "prefill", positions, rank_t)
                steps = seen["layer"]
                out[(d, phase)] = {
                    "combined": seen["combined"][0],
                    "pack": steps[1][1] if phase == "decode" else seen["pack"][0],
                    "streams": [s if phase == "prefill" else s[0] for s in steps],
                    "routed": seen["routed"],
                    "logits": model.compute_logits(hidden if phase == "decode" else hidden[-1:])}
        out["dummy"] = {}
        for batch in DUMMY_BATCHES if world == "dp2" else ():
            lengths = [37 * r % 250 + 1 for r in range(batch)]
            blocks = [[r + 1] for r in range(batch)]
            if replica == 0:
                model.bind_kv_cache(_caches(0, tp, "decode", lengths, blocks))
                ids, positions, meta = _inputs(0, "decode", lengths, blocks)
            else:
                gen = torch.Generator().manual_seed(batch + rank)
                pages = [(0.5 * torch.randn(CONTEXT // BLOCK, PAGE // 2, generator=gen)).bfloat16()
                         .view(torch.int8) for _ in range(4)]
                before = [p.clone() for p in pages]
                model.bind_kv_cache({f"layers.{i}.self_attn": _views(pages[i], i < 3)
                                     for i in range(4)})
                kwargs = _dummy_inputs(batch, rank_t)
                ids, positions, meta = (kwargs[key] for key in
                                        ("input_ids", "positions", "attn_metadata"))
            model.model(ids, meta, False, positions, rank_t)
            if replica == 1:
                out["dummy"][batch] = [(p != b).sum(-1) for p, b in zip(pages, before)]
    torch.save(out, root / f"{world}-rank{rank}.pt")
    del stamped
    dist.destroy_process_group()


def _error(a: torch.Tensor, b: torch.Tensor) -> float:
    """The largest difference over the largest magnitude of ``b``."""
    return ((a.double() - b.double()).abs().max() / b.double().abs().max()).item()


@pytest.fixture(scope="module")
def worlds(tmp_path_factory) -> dict:
    """Every rank's records of the two-replica world and the one-replica world."""
    root = tmp_path_factory.mktemp("dp")
    torch.save(_weights(_model()), root / "weights.pt")
    torch.save(_experts(), root / "experts.pt")
    runs = {}
    for world, ranks in (("dp2", TP * DP), ("dp1", TP)):
        mp.spawn(_rank, args=(root, world), nprocs=ranks, join=True)
        runs[world] = [torch.load(root / f"{world}-rank{r}.pt") for r in range(ranks)]
    return runs


def test_idle_replica_dummy_batch_writes_only_the_scratch_page(worlds):
    """The idle replica's dummy batch changes no byte outside page 0, at batches 16 and 32."""
    written = {(batch, rank, layer): int(pages[1:].sum()) for rank in range(TP, TP * DP)
               for batch, counts in worlds["dp2"][rank]["dummy"].items()
               for layer, pages in enumerate(counts) if pages[1:].any()}
    assert sorted(worlds["dp2"][TP]["dummy"]) == list(DUMMY_BATCHES)
    assert not written, f"bytes changed outside page 0, by batch, rank and layer: {written}"


def test_two_replicas_match_one(worlds):
    """The first MoE layer's combined rows and tail pack, and the decode logits, match DP1."""
    readings = {}
    for d in range(DP):
        for tp in range(TP):
            two, one = worlds["dp2"][d * TP + tp], worlds["dp1"][tp]
            for phase in ("decode", "prefill"):
                for seam in ("combined", "pack"):
                    readings[(phase, seam, d, tp)] = (
                        _error(two[(d, phase)][seam], one[(d, phase)][seam]), SEAM)
            readings[("decode", "logits", d, tp)] = (
                _error(two[(d, "decode")]["logits"], one[(d, "decode")]["logits"]), LOGITS)
    over = {key: f"{value:.2e}" for key, (value, bound) in readings.items() if not value <= bound}
    assert not over, f"over the bound, by phase, seam, replica and rank: {over}"
