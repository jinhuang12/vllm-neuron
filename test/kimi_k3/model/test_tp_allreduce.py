# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 tensor-parallel shards and all-reduces on two CPU ranks, against one-rank calls.

A 4-layer model (KDA layers 0 to 2 and MLA layer 3; the dense MLP at layer 0 and MoE at 1
to 3) is built at two ranks' widths, and each gloo rank loads its shards of one set of
random weights through the parameters' own loaders. Each seam is checked alone, where one
rank fits the kernels: the embedding; the MLA latent projection of the rank's K slice; and
the bf16 output partials of the attention and of the dense MLP, against one call over both
ranks' inputs. The error is the largest difference over the largest one-rank magnitude.
The Dynamo graph of one 4-layer decode step counts its all-reduces by call site. The MLA
attention core and the routed experts' partial are fixed rows. Set ``KIMI_K3_CHECKPOINT`` to the
checkpoint directory (default ``~/models/moonshotai/Kimi-K3``); run with
``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1``.
"""

import ast
import copy
import inspect
import os
import re
from collections import Counter
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from vllm_neuron.model.kimi_k3 import model as m
from vllm_neuron.nn import embedding
from vllm_neuron.model.kimi_k3 import weight_loaders_mxfp4
from vllm_neuron.utils.weight_loader import SliceView, get_weight_loader

CHECKPOINT = Path(
    os.environ.get("KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3")
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "config.json").is_file(),
    reason=f"Kimi K3 checkpoint not found at {CHECKPOINT}",
)

TP, T, H = 2, 4, 7168
IDS = [3, 5121, 10239, 4000]  # tokens in both ranks' vocabulary halves
# Two bf16 partial roundings, one of their sum and one of the one-rank output, each at most
# 2^-8 of its value, with no partial over the largest one-rank magnitude.
BF16_BOUND = 1.6e-2
# The function that issues each all-reduce, by the site it serves.
SITES = {
    "KimiK3DecoderLayer.forward_decode": "attention",
    "KimiK3MLA.forward_decode": "MLA latent",
    "VocabDimShardedEmbedding.forward": "embedding",
    "KimiK3MLP.forward_decode": "dense MLP",
    "KimiK3MoE.forward_decode": "MoE",
}


def _model() -> m.KimiK3ForConditionalGeneration:
    """The 4-layer model on meta, at the widths of this process's tensor-parallel group."""
    from vllm.transformers_utils.config import get_config

    config = copy.deepcopy(get_config(str(CHECKPOINT), trust_remote_code=False).to_dict())
    text = config["text_config"]
    text.update(num_hidden_layers=4, num_attention_heads=6, intermediate_size=2112,
                moe_intermediate_size=96, routed_expert_hidden_size=112, num_experts=16,
                vocab_size=10240, first_k_dense_replace=1)
    text["linear_attn_config"].update(num_heads=6, kda_layers=[1, 2, 3], full_attn_layers=[4])
    with torch.device("meta"):
        return m.KimiK3ForConditionalGeneration.from_configs(config)


def _weights(model) -> dict[str, torch.Tensor]:
    """Random whole weights for every parameter; the expert banks are zero."""
    gen, out = torch.Generator().manual_seed(7), {}
    for name, p in model.named_parameters():
        leaf = name.rpartition(".")[2]
        if not p.dtype.is_floating_point or ".experts." in name:
            value = torch.zeros(p.shape)
        elif "norm" in name:
            value = 1 + 0.1 * torch.randn(p.shape, generator=gen)
        else:
            scale = 0.5 if leaf in ("A_log", "dt_bias") or "conv" in name else 0.02
            value = scale * torch.randn(p.shape, generator=gen)
        out[name] = value.to(p.dtype)
    return out


def _load(model, weights: dict, rank: int) -> None:
    """Load this rank's part of each whole weight through the parameter's loader."""
    weight_loaders_mxfp4.checkpoint_mappings(model, model.text_config)
    for name, p in list(model.named_parameters()):
        path, _, leaf = name.rpartition(".")
        if ".experts." in name:
            value = torch.zeros(p.shape, dtype=p.dtype)
        else:
            value = get_weight_loader(p).load([SliceView(weights[name])], rank)
        setattr(model.get_submodule(path), leaf, nn.Parameter(value, requires_grad=False))
    model.prepare_after_load()


def _rows() -> dict[str, torch.Tensor]:
    """The decode inputs: the stream, one bank row, the MoE tail pack and the fixed rows."""
    gen = torch.Generator().manual_seed(11)
    sumsq = 112 * (1 + 0.1 * torch.rand(T, 1, generator=gen))
    return {
        "stream": (0.5 * torch.randn(T, H, generator=gen)).bfloat16(),
        "bank": (0.5 * torch.randn(T, 1, H, generator=gen)).bfloat16(),
        "tail": torch.cat([0.05 * torch.randn(T, 2 * H, generator=gen), sumsq], -1),
        "attended": (0.5 * torch.randn(T, 768, generator=gen)).bfloat16(),
        "combined": torch.randn(1, TP, T, 56, generator=gen),
    }


def _bind(model, rank: int, rows: dict) -> list:
    """Bind KDA page buffers, fix the MLA core to the rank's heads and the experts' partial."""
    layers, latents = model.model.layers, []
    for layer in layers[:3]:
        pages = torch.zeros(2, 294_912, dtype=torch.int8)
        conv = pages[:, :6912].view(torch.bfloat16).view(2, 3, 1152)
        state = pages[:, 6912:203_520].view(torch.float32).view(2, 3, 128, 128)
        layer.self_attn.bind_pages(pages, conv, state)
    heads = m._decode_layout(rows["attended"][:, rank * 384:(rank + 1) * 384])
    layers[3].self_attn._attend = lambda latent, gate, meta: latents.append(latent) or heads
    for layer in layers[1:]:
        layer.block_sparse_moe._select_and_run = lambda routed, base: rows["combined"]
    return latents


def _functions(module) -> dict[int, str]:
    """Each source line of ``module``'s file, mapped to the qualified function holding it."""
    tree, names = ast.parse(inspect.getsource(module)), {}

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef)):
                name = f"{prefix}{child.name}"
                if isinstance(child, ast.FunctionDef):
                    names.update(dict.fromkeys(range(child.lineno, child.end_lineno + 1), name))
                visit(child, f"{name}.")

    visit(tree, "")
    return names


def _call_sites(graphs) -> dict:
    """The site of each all-reduce node in the captured graphs, from its source frames."""
    files = {inspect.getsourcefile(mod): _functions(mod) for mod in (m, embedding)}
    sites, waits, calls = Counter(), 0, []
    for gm in graphs:
        for node in gm.graph.nodes:
            target = str(node.target)
            waits += "wait_tensor" in target
            if node.op != "call_function" or "all_reduce" not in target:
                continue
            frames = re.findall(r'File "([^"]+)", line (\d+)', node.meta.get("stack_trace", ""))
            named = [(files[f].get(int(n)), f"{Path(f).name}:{n}") for f, n in frames if f in files]
            site, line = next(((SITES[q], at) for q, at in reversed(named) if q in SITES),
                              ("other", "?"))
            sites[site] += 1
            stack = [v[0] for v in (node.meta.get("nn_module_stack") or {}).values()]
            calls.append((site, line, stack))
    return {"sites": dict(sites), "waits": waits, "graphs": len(graphs), "calls": calls}


def _seam(call: tuple, total: torch.Tensor) -> dict:
    """A seam's rank input, rank partial and summed output."""
    return {"x": call[0], "partial": call[1], "sum": total}


def _rank(rank: int, root: Path) -> None:
    """One gloo rank: the seams' inputs and outputs, then one counted decode step."""
    from vllm_neuron.parallel.neuron_parallel_state import initialize_neuron_parallel_state

    dist.init_process_group("gloo", init_method=f"file://{root}/rdzv", rank=rank, world_size=TP)
    initialize_neuron_parallel_state(tp_global_ranks=list(range(TP)), local_rank=rank)
    model, rows, r = _model(), _rows(), torch.tensor(rank, dtype=torch.int32)
    _load(model, torch.load(root / "weights.pt"), rank)
    latents, layers, tp_rank = _bind(model, rank, rows), model.model.layers, r.reshape(1)
    meta = {"block_table_tensor": torch.tensor([[1], [0], [0], [0]], dtype=torch.int32)}
    calls, attended = [], []
    project, mix = m._output_projection, m.KimiK3DecoderLayer._mix_decode

    def record(x, weight):
        calls.append((x, project(x, weight)))
        return calls[-1][1]

    def mix_site(layer, stream, tail, bank, site, tp_rank=None):
        if site == "mlp":  # the attention output, summed, reaches the MLP-site mix as its tail
            attended.append(tail)
        return mix(layer, stream, tail, bank, site, tp_rank)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(m, "_output_projection", record)
        patch.setattr(m.KimiK3DecoderLayer, "_mix_decode", mix_site)
        out = {"embed": model.model.embed(torch.tensor(IDS), r)[0]}
        _, dense, _ = layers[0].forward_decode(rows["stream"], None, rows["bank"].clone(), meta,
                                               tp_rank)
        out.update(kda=_seam(calls[0], attended[0]), dense=_seam(calls[1], dense))
        calls.clear()
        layers[3].forward_decode(rows["stream"], rows["tail"], rows["bank"].clone(), None, tp_rank)
        out.update(latent=latents[0], mla=_seam(calls[1], attended[1]))
    graphs = []

    def capture(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    metas = {f"layers.{i}.self_attn": dict(meta) for i in range(4)}
    torch._dynamo.reset()
    step = torch.compile(model.model, backend=capture)
    step(torch.tensor(IDS), metas, False, torch.zeros(T, dtype=torch.long), r)
    out["graph"] = _call_sites(graphs)
    torch.save(out, root / f"rank{rank}.pt")
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict:
    """Both ranks' outputs, and the one-rank model on the same whole weights."""
    root = tmp_path_factory.mktemp("tp")
    one = _model()
    weights = _weights(one)
    torch.save(weights, root / "weights.pt")
    mp.spawn(_rank, args=(root,), nprocs=TP, join=True)
    _load(one, weights, 0)
    return {"one": one, "ranks": [torch.load(root / f"rank{i}.pt") for i in range(TP)]}


def _error(out: torch.Tensor, one: torch.Tensor) -> float:
    """The largest difference over the largest one-rank magnitude."""
    return ((out.double() - one.double()).abs().max() / one.double().abs().max()).item()


def _bf16_seam(ranks: list[dict], one: torch.Tensor) -> None:
    """A bf16 all-reduce meets the one-rank call within the bound, on inputs the bound covers."""
    largest = one.double().abs().max()
    assert all(r["partial"].double().abs().max() <= largest for r in ranks)
    total = ranks[0]["sum"]
    assert total.dtype == torch.bfloat16 and torch.equal(total, ranks[1]["sum"])
    assert _error(total, one) <= BF16_BOUND


def test_embedding_seam(runs):
    """The two ranks' summed vocabulary halves are the one-rank lookup, bit for bit."""
    one = runs["one"].model.embed(torch.tensor(IDS))[0]
    for out in runs["ranks"]:
        assert out["embed"].dtype == one.dtype and torch.equal(out["embed"], one)


def test_mla_latent_seam(runs):
    """The fp32 sum of the K-slice partials of q_a and kv_a is the one-rank call within 1e-5."""
    layer, rows = runs["one"].model.layers[3], _rows()
    _, _, k_slice, _ = layer._mix_decode(rows["stream"], rows["tail"], rows["bank"], "attention")
    one = m._output_projection(k_slice, layer.self_attn.latent_weight)
    for out in runs["ranks"]:
        assert out["latent"].dtype == torch.float32
        assert _error(out["latent"], one) <= 1e-5


def test_attention_seam(runs):
    """The KDA and MLA attention all-reduces of bf16 partials meet one call over both ranks."""
    layers, ranks = runs["one"].model.layers, runs["ranks"]
    core = torch.cat([out["kda"]["x"] for out in ranks], dim=2)
    _bf16_seam([out["kda"] for out in ranks],
               m._output_projection(core, layers[0].self_attn.output_weight))
    attended = m._decode_layout(_rows()["attended"])
    _bf16_seam([out["mla"] for out in ranks],
               m._output_projection(attended, layers[3].self_attn.output_weight))


def test_dense_seam(runs):
    """The dense MLP's all-reduce of bf16 down partials meets one call over both ranks."""
    ranks = runs["ranks"]
    rows = torch.cat([out["dense"]["x"] for out in ranks], dim=2)
    one = m._output_projection(rows, runs["one"].model.layers[0].mlp.down_weight)
    _bf16_seam([out["dense"] for out in ranks], one)


def test_all_reduce_count(runs):
    """The decode graph all-reduces 4 attention outputs, 1 MLA latent, 1 embedding, 1 dense,
    and per MoE layer its dispatch and its tail pack."""
    for out in runs["ranks"]:
        sites = out["graph"]["sites"]
        named = {k: sites.get(k, 0) for k in ("attention", "MLA latent", "embedding")}
        assert named == {"attention": 4, "MLA latent": 1, "embedding": 1}
        assert sites.get("dense MLP", 0) == 1 and sites.get("MoE", 0) == 6
        assert sum(sites.values()) == 13
