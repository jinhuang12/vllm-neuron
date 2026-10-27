# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 MLA under decode context parallelism on two CPU ranks, against the same run at DCP 1.

The 4-layer config (KDA layers 0 to 2, MLA layer 3) loads one set of random weights on two
gloo ranks at TP 2, once at DCP 2 and once at DCP 1. Each world runs the same steps on the
runner's metadata: a prompt of 4 prefill chunks, then 16 decode steps of two sequences, the
prompt's and a new one whose context stays inside its first block. The prefill error is the
largest difference over the largest DCP 1 magnitude, on each rank. The decode steps are read in
bf16 units in the last place: the DCP decode kernel's output, before the output projection,
and the logits. Between the prompt and the decode steps, the runner's jump-forward input step
builds its inputs once, for the last item.

Most kernels on this path run only on a device or the NKI simulator; of the attention kernels
only the DCP decode attention has a torch form for the CPU venue. So torch forms of the others'
math stand in, the same in both worlds: the fused norm and projection, the prefill
projections, the AttnRes mixes, the MoE tail and the paged MLA prefill, and at DCP 1 the paged
MLA decode, as the DCP mode over a one-rank group. The expert banks are zero, so the routed
partial is zero. The KDA layers are not under test: a stateless map, their gated v columns
through ``o_proj``, stands in for their token mixer. Set ``KIMI_K3_CHECKPOINT`` to the
checkpoint directory (default ``~/models/moonshotai/Kimi-K3``); run with
``VLLM_NEURON_CPU_MODE=1``.
"""

import contextlib
import copy
import inspect
import json
import math
import os
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.config import get_config
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups, get_kv_cache_groups
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_neuron.functional.attention.attn_res_decode import AttnResDecodeOutput
from vllm_neuron.model.kimi_k3 import model as m
from vllm_neuron.model.kimi_k3 import weight_loaders_mxfp4
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.utils.weight_loader import SliceView, get_weight_loader
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner
from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

CHECKPOINT = Path(
    os.environ.get("KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3")
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "config.json").is_file(),
    reason=f"Kimi K3 checkpoint not found at {CHECKPOINT}",
)

TP, DCP, BLOCK, MAX_LEN, VOCAB = 2, 2, 256, 4096, 10240
# Each chunk after the first starts at a multiple of DCP blocks, as the runner requires. The
# decode steps and the jump-forward step then cross from block 7, on rank 1, to block 8, on rank 0.
CHUNKS = (512, 512, 512, 511)
PROMPT, STEPS, BLOCKS, BOUND = sum(CHUNKS), 16, 80, 1e-5
# A step whose two largest DCP 1 logits are at most this many units apart is a tie.
TIE_UNITS = 2
JUMP_TOKENS, JUMP_ROWS = 2, 4
NULL_SLOT, PAD_SLOT = 0, -1
Q_LORA, KV_LORA, ROPE = 1536, 512, 64
LATENT, SCALE = KV_LORA + ROPE, (128 + ROPE) ** -0.5
OVERRIDES = dict(num_hidden_layers=4, num_attention_heads=6, intermediate_size=2112,
                 moe_intermediate_size=96, routed_expert_hidden_size=112, num_experts=16,
                 vocab_size=VOCAB, first_k_dense_replace=1)
LINEAR = dict(num_heads=6, kda_layers=[1, 2, 3], full_attn_layers=[4])
ONE_RANK = types.SimpleNamespace(ranks=(0,), all_gather=lambda x, dim: x,
                                 reduce_scatter=lambda x, dim: x)


def _small_model() -> m.KimiK3ForConditionalGeneration:
    """The 4-layer model on the meta device, at this process's tensor-parallel widths."""
    config = copy.deepcopy(get_config(str(CHECKPOINT), trust_remote_code=False).to_dict())
    config["text_config"].update(OVERRIDES)
    config["text_config"]["linear_attn_config"].update(LINEAR)
    with torch.device("meta"):
        return m.KimiK3ForConditionalGeneration.from_configs(config)


def _small_config_dir(root: Path) -> Path:
    """A model directory holding the 4-layer config, for the engine config build."""
    config = json.loads((CHECKPOINT / "config.json").read_text())
    text = config["text_config"]
    text.update({k: v for k, v in OVERRIDES.items() if k in text})
    text["linear_attn_config"].update(LINEAR)
    root.mkdir(exist_ok=True)
    (root / "config.json").write_text(json.dumps(config))
    return root


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


def _engine_config(model_dir: Path, dcp: int):
    """The engine config at TP 2 and ``dcp``, after the platform's block-size step."""
    config = EngineArgs(
        model=str(model_dir), tensor_parallel_size=TP, decode_context_parallel_size=dcp,
        max_model_len=MAX_LEN, enable_prefix_caching=False, gpu_memory_utilization=0.90,
        skip_tokenizer_init=True).create_engine_config()
    current_platform.update_block_size_for_backend(config)
    return config


def _runner(config, model, dcp: int) -> NeuronModelRunner:
    """A model runner at ``dcp`` with a small KV cache bound to ``model``."""
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.vllm_config, runner.model, runner.speculative_config = config, model, None
    runner.drafter, runner.neuron_config, runner.is_dflash_spec = None, NeuronConfig(), False
    runner.max_num_reqs, runner.max_model_len = JUMP_ROWS, config.model_config.max_model_len
    runner.max_num_batched_tokens = config.scheduler_config.max_num_batched_tokens
    runner.device, runner.vocab_size, runner.is_pooling_model = torch.device("cpu"), VOCAB, False
    runner._kv_cache_full_tensors, runner._block_table_groups, runner._swa_groups = {}, {}, {}
    runner._constant_tensors, runner._kv_snapshot_enabled = {}, False
    runner._is_synthetic_model = False
    runner._dcp_size = runner.cp_world_size = dcp
    runner.cp_kv_cache_interleave_size = config.parallel_config.cp_kv_cache_interleave_size
    runner._cp_rank = None
    specs = runner.get_kv_cache_spec()
    layouts = NeuronWorker.__new__(NeuronWorker).get_supported_kv_cache_layouts()
    resolve_kv_cache_layout(config, [layouts], list(specs.values()))
    groups = get_kv_cache_groups(config, specs)
    slots = max(len(g.layer_names) for g in groups)
    runner.initialize_kv_cache(get_kv_cache_config_from_groups(
        config, groups, BLOCKS * slots * config.cache_config.mamba_page_size_padded))
    return runner


def _add(runner, req_id: str, prompt: list, latent: list, state: int) -> None:
    """Add a new request with its latent block ids and its state block."""
    groups = range(len(runner.kv_cache_config.kv_cache_groups))
    runner.input_batch.add_request(CachedRequestState(
        req_id=req_id, prompt_token_ids=prompt, mm_features=[], sampling_params=SamplingParams(),
        generator=None, num_computed_tokens=0, output_token_ids=[],
        block_ids=tuple([state] if runner._is_state_group(g) else latent for g in groups)))


def _metadata(runner, positions: np.ndarray, requests: np.ndarray, rows: int, query: int,
              cached: int) -> dict:
    """The runner's attention metadata for one step of ``positions``."""
    runner._compute_slot_mappings(positions, requests)
    runner._save_block_table_to_device()
    runner._save_slot_mapping_to_device(len(positions))
    return runner._build_attention_metadata(
        padded_num_reqs=rows, total_num_scheduled_tokens=len(positions), max_query_len=query,
        max_num_draft_tokens=0, cached_seq_len=cached, max_decode_ctx_len=cached + 1)


def _norm_projection(hidden, weight, norm_weight, eps):
    """Torch form of the fused RMSNorm and projection: fp32 norm and product, stored bf16."""
    x = hidden.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * norm_weight.float()
    return (x @ weight.float()).to(hidden.dtype)


def _projection(x, weight, heads=False):
    """Torch form of the prefill output projection: an fp32 product in the rows' dtype."""
    return (x.float() @ weight.float()).to(x.dtype)


def _mix(bank, stream, score_weight, eps):
    """Torch form of the AttnRes mix: a softmax over the sources' normed scores, stored bf16."""
    v = torch.cat((bank, stream.unsqueeze(1)), dim=1).float()
    k = v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)
    weights = (k * score_weight.float()).sum(-1).softmax(-1)
    return (weights.unsqueeze(1) @ v).squeeze(1).to(stream.dtype)


def _mix_decode(bank, stream, score_weight, num_sources, *, residual=None, moe_tail=None,
                latent_size=None, bank_slot=None, norm_weight=None, tp_rank=None, tp_size=1,
                dp_rank=None, dp_size=1, eps=1e-5):
    """Torch form of the AttnRes decode mix: the tail fold, the bank store and the K slice."""
    new = stream
    if residual is not None:
        new = stream + residual
    if moe_tail is not None:
        up, shared, squares = moe_tail.split([stream.shape[-1]] * 2 + [1], -1)
        new = stream + (up * torch.rsqrt(squares / latent_size + eps) + shared).to(stream.dtype)
    if bank_slot is not None:
        bank[:, bank_slot] = new
    mix = _mix(bank[:, : num_sources - 1], new, score_weight, eps)
    k_slice = None
    if norm_weight is not None:
        x = mix.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * norm_weight.float()
        if tp_size > 1:
            x = x.view(x.shape[0], tp_size, -1)[:, int(tp_rank)]
        k_slice = m._decode_layout(x)
    return AttnResDecodeOutput(mix, new, k_slice, bank if bank_slot is not None else None)


def _tail(combined, gate_up, latent_norm_weight, latent_up_weight, shared_down_weight, beta,
          linear_beta):
    """Torch form of the fused MoE tail: the fp32 pack ``[up | shared | sum of squares]``."""
    gate, up = gate_up.float().chunk(2, -1)
    up_rows = (combined * latent_norm_weight.float().t()) @ latent_up_weight.float()
    shared = m.situ_and_mul(gate, up, beta, linear_beta) @ shared_down_weight.float()
    return torch.cat([up_rows, shared, combined.square().sum(-1, keepdim=True)], -1)


def _routed_decode(self, routed, base):
    """The routed experts' decode partial ``[R, C, T, K / C]``: zero, as the banks are."""
    replicas, chunks = m._replicas(), m._tp_rank_and_size()[1]
    return routed.new_zeros(replicas, chunks, routed.shape[0] // replicas,
                            self.latent_size // chunks)


def _routed_prefill(self, routed, tp_rank):
    """The routed experts' prefill partial ``[N, K]``: zero, as the banks are."""
    return routed.new_zeros(routed.shape[0], self.latent_size)


def _kda_prefill(self, mix, norm_weight, meta):
    """The stand-in KDA token mixer: the gated v columns through ``o_proj``, fp32 ``[T, H]``."""
    _, _, v, gate, _, _ = self._split_input(
        m._project(mix, self.input_weight, norm_weight, self.eps))
    return (v.float() * torch.sigmoid(gate.float())) @ self.output_weight.float()


def _kda_decode(self, mix, k_slice, norm_weight, meta):
    """The stand-in KDA token mixer at decode, stored bf16."""
    return _kda_prefill(self, mix, norm_weight, meta).to(torch.bfloat16)


def _mla_inputs(qkv_a, q_norm_weight, kv_norm_weight, w_qb_t, w_uk, eps):
    """The absorbed bf16 queries ``[T, H, 576]`` and the bf16 latent rows ``[T, 576]``."""
    x, tokens, heads = qkv_a.float(), qkv_a.shape[0], w_uk.shape[0]

    def norm(v, weight):
        return v * torch.rsqrt(v.square().mean(-1, keepdim=True) + eps) * weight.float()

    q = (norm(x[:, :Q_LORA], q_norm_weight) @ w_qb_t.float()).view(tokens, heads, -1)
    nope = torch.einsum("thn,hnl->thl", q[..., :-ROPE], w_uk.float())
    queries = torch.cat((nope, q[..., -ROPE:]), -1).bfloat16()
    latent = torch.cat((norm(x[:, Q_LORA:Q_LORA + KV_LORA], kv_norm_weight),
                        x[:, Q_LORA + KV_LORA:]), -1)
    return queries, latent.bfloat16()


def _mla_prefill(qkv_a, gate, q_norm_weight, kv_norm_weight, w_qb_t, w_uk, w_uv_t, pages,
                 latent_offset, block_size, block_table, prefix_len, slots, eps):
    """Torch form of the paged MLA prefill: the prefix through the table, then the chunk, causal.

    Each chunk row goes to its slot in ``pages``; a slot outside the cache names block 0, row 0.
    """
    queries, latent = _mla_inputs(qkv_a, q_norm_weight, kv_norm_weight, w_qb_t, w_uk, eps)
    tokens, heads = queries.shape[:2]
    rows = pages.view(torch.bfloat16).view(pages.shape[0], -1, LATENT)
    first, prefix = latent_offset // (2 * LATENT), int(prefix_len)
    ids = block_table.long().clamp(0, pages.shape[0] - 1)
    cached = rows[ids, first : first + block_size].reshape(-1, LATENT)[:prefix]
    keys = torch.cat((cached, latent)).float()
    scores = torch.einsum("thl,kl->thk", queries.float(), keys) * SCALE
    seen = torch.arange(keys.shape[0]) <= prefix + torch.arange(tokens).unsqueeze(1)
    weights = scores.masked_fill(~seen.unsqueeze(1), float("-inf")).softmax(-1)
    out = torch.einsum("thk,kl->thl", weights, keys[:, :KV_LORA])
    out = torch.einsum("thl,hlv->thv", out, w_uv_t.float())
    out = out * torch.sigmoid(gate.float()).view(tokens, heads, -1)
    s = slots.long()
    s = torch.where((s < 0) | (s >= pages.shape[0] * block_size), 0, s)
    rows[s // block_size, first + s % block_size] = latent
    return out.reshape(tokens, -1).bfloat16(), pages


def _mla_decode(*args):
    """The paged MLA decode at one rank: the DCP mode's torch form over a one-rank group."""
    *inputs, table, context_lens, slots, eps = args
    owner = torch.ones(context_lens.shape, dtype=torch.int32)
    return m.mla_paged_decode_dcp(*inputs, table, context_lens, slots, owner, eps, ONE_RANK)


@contextlib.contextmanager
def _stand_ins():
    """The torch forms in place of the kernels the CPU venue cannot run, and the KDA mixer."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(m, "qkv_proj_fp32acc", _norm_projection)
        patch.setattr(m, "_prefill_projection", _projection)
        patch.setattr(m, "attn_res_prefill", _mix)
        patch.setattr(m, "attn_res_decode", _mix_decode)
        patch.setattr(m, "moe_fused_tail", _tail)
        patch.setattr(m, "mla_paged_prefill", _mla_prefill)
        patch.setattr(m, "mla_paged_decode", _mla_decode)
        patch.setattr(m.KimiK3MoE, "_select_and_run", _routed_decode)
        patch.setattr(m.KimiK3MoE, "_prefill_experts", _routed_prefill)
        patch.setattr(m.KimiK3KDA, "forward_prefill", _kda_prefill)
        patch.setattr(m.KimiK3KDA, "forward_decode", _kda_decode)
        yield patch


def _tap(store: list, forward):
    """``forward``, keeping a copy of each output in ``store``."""
    def run(self, *args):
        out = forward(self, *args)
        store.append(out.clone())
        return out
    return run


def _tap_kernel(store: list, kernel):
    """``kernel``, keeping a copy of each output and of its ``owner`` input in ``store``."""
    signature = inspect.signature(kernel)

    def run(*args, **kwargs):
        out, pages = kernel(*args, **kwargs)
        owner = signature.bind(*args, **kwargs).arguments["owner"]
        store.append((out.clone(), owner.clone()))
        return out, pages
    return run


def _serve(config, root: Path, rank: int, dcp: int) -> dict:
    """The prompt's chunks, the jump-forward step, then the decode steps, on this rank."""
    model, ids = _small_model(), torch.load(root / "ids.pt")
    _load(model, torch.load(root / "weights.pt", mmap=True), rank)
    runner = _runner(config, model, dcp)
    per_rank = MAX_LEN // (BLOCK * dcp)
    tables = {"prompt": list(range(3, 3 + per_rank)), "short": [3 + per_rank]}
    _add(runner, "prompt", ids["prompt"].tolist(), tables["prompt"], 1)
    _add(runner, "short", ids["decode"][:1, 1].tolist(), tables["short"], 2)
    seen, rank_t, start = {"prefill": [], "decode": []}, torch.tensor([rank], dtype=torch.int32), 0
    kernel = []
    with _stand_ins() as patch:
        for kind, store in seen.items():
            name = f"forward_{kind}"
            patch.setattr(m.KimiK3MLA, name, _tap(store, getattr(m.KimiK3MLA, name)))
        patch.setattr(m, "mla_paged_decode_dcp", _tap_kernel(kernel, m.mla_paged_decode_dcp))
        for size in CHUNKS:
            positions = np.arange(start, start + size)
            meta = _metadata(runner, positions, np.zeros(size, dtype=np.int64), 1, size, start)
            logits = model(ids["prompt"][start : start + size], torch.from_numpy(positions),
                           attn_metadata=meta, sampling_positions=torch.tensor([size - 1]),
                           rank=rank_t)
            start += size
        jump = _jump_forward(runner, int(ids["decode"][0, 0]), tables)
        steps = []
        for step in range(STEPS):
            positions = np.array([PROMPT + step, step])
            meta = _metadata(runner, positions, np.arange(2), 2, 1, PROMPT + step)
            steps.append(model(ids["decode"][step], torch.from_numpy(positions),
                               attn_metadata=meta, rank=rank_t))
    return {"chunks": seen["prefill"], "next": logits, "steps": torch.stack(steps),
            "attention": torch.stack(seen["decode"]), "jump": jump,
            "kernel": torch.stack([out for out, _ in kernel]),
            "owner": torch.stack([owner for _, owner in kernel])}


def _jump_forward(runner, token: int, tables: dict) -> dict:
    """The runner's jump-forward input step after the prompt, 2 tokens per request: the slots."""
    # The prompt is computed and its first token sampled, as the runner's state update leaves them.
    runner.input_batch.num_computed_tokens_cpu[0] = PROMPT
    runner.input_batch.token_ids_cpu[0, PROMPT] = token
    runner.neuron_config.max_jump_forward_tokens = JUMP_TOKENS
    _, positions, _, meta, *_ = runner._prepare_jf_model_input(
        None, {req: [5, 6] for req in tables}, JUMP_ROWS, PROMPT + 1 + JUMP_TOKENS)
    groups = runner.kv_cache_config.kv_cache_groups
    return {"positions": positions.clone(), "tables": tables,
            "state": [runner._is_state_group(g) for g in range(len(groups))],
            "slots": [meta[g.layer_names[0]]["slot_mapping"].clone() for g in groups]}


def _rank(rank: int, root: Path, model_dir: Path, dcp: int) -> None:
    """One gloo rank at TP 2 and ``dcp``: the prompt, the jump-forward step, the decode steps."""
    from vllm_neuron.parallel.neuron_parallel_state import (
        initialize_neuron_parallel_state, stamp_spmd_rank_tensors)

    dist.init_process_group("gloo", init_method=f"file://{root}/rdzv-{dcp}", rank=rank,
                            world_size=TP)
    initialize_neuron_parallel_state(tp_global_ranks=list(range(TP)), local_rank=rank,
                                     dcp_size=dcp)
    stamped = stamp_spmd_rank_tensors("cpu")
    torch.save(_serve(_engine_config(model_dir, dcp), root, rank, dcp),
               root / f"dcp{dcp}-rank{rank}.pt")
    del stamped
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[int, list[dict]]:
    """Each world's outputs by rank, at DCP 2 and at DCP 1."""
    root = tmp_path_factory.mktemp("dcp-mla")
    torch.save(_weights(_small_model()), root / "weights.pt")
    gen = torch.Generator().manual_seed(5)
    torch.save({"prompt": torch.randint(0, VOCAB, (PROMPT,), generator=gen),
                "decode": torch.randint(0, VOCAB, (STEPS, 2), generator=gen)}, root / "ids.pt")
    model_dir = _small_config_dir(root / "model")
    for dcp in (DCP, 1):
        mp.spawn(_rank, args=(root, model_dir, dcp), nprocs=TP, join=True)
    return {dcp: [torch.load(root / f"dcp{dcp}-rank{r}.pt") for r in range(TP)]
            for dcp in (DCP, 1)}


def _error(got: torch.Tensor, ref: torch.Tensor) -> float:
    """The largest difference over the largest DCP 1 magnitude."""
    got, ref = got.double(), ref.double()
    return float((got - ref).abs().max() / ref.abs().max())


def _unit(x: float) -> float:
    """One bf16 unit in the last place at ``x``: 2^(e - 7), where 2^e <= |x| < 2^(e + 1)."""
    return 2.0 ** (math.frexp(abs(float(x)))[1] - 8)


def _vocab(run: list[dict], step: int, seq: int) -> torch.Tensor:
    """One step's logits over the full vocabulary, both ranks' halves gathered."""
    return torch.cat([out["steps"][step, seq] for out in run]).double()


def _decode_misses(runs, seq: int) -> dict:
    """The seam, logits and argmax reads of sequence ``seq``: what each finds over its bound.

    Units are taken at DCP 1's values: for the seam, the rank's kernel output at that step
    and sequence; for the logits, the full vocabulary at that step.
    """
    misses = {"seam": [], "logits": [], "argmax": []}
    ties = []
    for s in range(STEPS):
        for rank, (got, ref) in enumerate(zip(runs[DCP], runs[1])):
            a, b = got["kernel"][s, :, seq].double(), ref["kernel"][s, :, seq].double()
            move = float((a - b).abs().max()) / _unit(b.abs().max())
            if not move <= 1:
                misses["seam"].append((s, rank, move))
        got, ref = _vocab(runs[DCP], s, seq), _vocab(runs[1], s, seq)
        unit, top = _unit(ref.abs().max()), ref.topk(2).values
        move = float((got - ref).abs().max()) / unit
        if not move <= 1:
            misses["logits"].append((s, move))
        if float(top[0] - top[1]) / unit <= TIE_UNITS:
            ties.append(s)
        elif int(got.argmax()) != int(ref.argmax()):
            misses["argmax"].append(s)
    print(f"sequence {seq}: {len(ties)} tie steps excluded from the argmax read: {ties}")
    return misses


def _owner_misses(run: list[dict]) -> list:
    """Steps where the ranks passing ``owner`` 1 are not the row's block index modulo DCP."""
    misses = []
    for s in range(STEPS):
        for seq, position in enumerate((PROMPT + s, s)):
            passing = [rank for rank, out in enumerate(run) if int(out["owner"][s, seq]) == 1]
            if passing != [position // BLOCK % DCP]:
                misses.append((s, seq, passing))
    return misses


def test_prefill_chunks_match_dcp_1(runs):
    """Each of the prompt's 4 chunks: the MLA layer output; then the next-token logits."""
    readings = []
    for got, ref in zip(runs[DCP], runs[1]):
        assert len(got["chunks"]) == len(ref["chunks"]) == len(CHUNKS)
        readings += [_error(a, b) for a, b in zip(got["chunks"], ref["chunks"])]
        readings.append(_error(got["next"], ref["next"]))
    assert all(r <= BOUND for r in readings), readings


def test_decode_steps_match_dcp_1(runs):
    """The prompt sequence's 16 decode steps after the prompt, and the owner input of both.

    Seam: every element of the DCP decode kernel's bf16 output, the attention output before
    the output projection, is within one bf16 unit in the last place of the largest magnitude
    among DCP 1's elements of that step and sequence. Logits: every step's next-token argmax
    over the full vocabulary, both ranks' halves, equals DCP 1's, and each logit is within one
    bf16 unit in the last place of the largest magnitude among DCP 1's logits of that step.
    Ties: a step where DCP 1's two largest logits are at most two such units apart is excluded
    from the argmax read and disclosed with its count, printed. Owner: for each sequence,
    exactly one rank passes ``owner`` 1 per step, and it is the rank equal to the row's block
    index modulo the DCP size; the row is the step's own latent row.
    """
    assert all(len(out["kernel"]) == len(out["owner"]) == STEPS
               for run in runs.values() for out in run)
    misses = _decode_misses(runs, 0)
    misses["owner"] = _owner_misses(runs[DCP])
    assert not any(misses.values()), misses


def test_short_context_decode_matches_dcp_1_without_nan(runs):
    """A context inside its first block, so rank 1 holds no key: no output is NaN.

    The short sequence's decode steps meet the prompt sequence's seam, logits and argmax reads.
    """
    misses = _decode_misses(runs, 1)
    nan = sum(int(out["steps"][:, 1].isnan().sum() + out["attention"][:, 1].isnan().sum()
                  + out["kernel"][:, :, 1].isnan().sum()) for out in runs[DCP])
    assert nan == 0 and not any(misses.values()), (nan, misses)


def test_jump_forward_slots_follow_each_group_dcp_size(runs):
    """Latent rows get a slot on their owner rank only, state groups DCP 1's, pads the pad slot."""
    differ = checked = 0
    for rank, (got, ref) in enumerate(zip(runs[DCP], runs[1])):
        jump, positions = got["jump"], got["jump"]["positions"].view(JUMP_ROWS, -1)
        for g, slots in enumerate(jump["slots"]):
            slots = slots.view(JUMP_ROWS, -1)
            want = torch.full_like(slots, PAD_SLOT)
            for row, req in enumerate(jump["tables"]):
                if jump["state"][g]:
                    want[row] = ref["jump"]["slots"][g].view(JUMP_ROWS, -1)[row]
                    continue
                p = positions[row]
                block = torch.tensor(jump["tables"][req])[p // (BLOCK * DCP)]
                want[row] = torch.where(p // BLOCK % DCP == rank, block * BLOCK + p % BLOCK,
                                        NULL_SLOT)
            differ += int((slots != want).sum())
            checked += slots.numel()
    assert checked > 0 and differ == 0, differ
