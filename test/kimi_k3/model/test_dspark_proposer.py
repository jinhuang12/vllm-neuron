# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 DSpark speculation through the model runner, on two TP ranks.

A 4-layer target at hidden 7,168 (three KDA layers, then MLA; a dense first layer, then
MoE with 16 experts) runs from seeded random weights beside the DSpark checkpoint's draft,
at TP 2 over gloo. Torch stand-ins replace the kernels that have no CPU path, the draft's
block pass and the verify window's MLA attention. They compute in fp32, each rank runs on
one thread, so a row's value does not depend on the step's other rows, and the LM head
returns fp32 logits. Greedy outputs with speculation then equal those of the run without
it. Set
``KIMI_K3_CHECKPOINT`` and ``KIMI_K3_DSPARK_CHECKPOINT`` to the target and draft
directories (defaults ``~/models/moonshotai/Kimi-K3`` and
``~/models/Inferact/Kimi-K3-DSpark-Block5``). Run with ``VLLM_NEURON_CPU_MODE=1``.
"""

import gc
import os
import socket
import warnings
import zlib
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

MODELS = Path.home() / "models"
TARGET = Path(os.environ.get("KIMI_K3_CHECKPOINT", MODELS / "moonshotai" / "Kimi-K3"))
DRAFT = Path(
    os.environ.get("KIMI_K3_DSPARK_CHECKPOINT", MODELS / "Inferact" / "Kimi-K3-DSpark-Block5")
)

pytestmark = pytest.mark.skipif(
    not ((TARGET / "config.json").is_file() and (DRAFT / "model.safetensors").is_file()),
    reason=f"Kimi K3 config at {TARGET} or DSpark checkpoint at {DRAFT} not found",
)

TP, K, MASK, VOCAB, HIDDEN, ROW = 2, 4, 163839, 163840, 7168, 576
CONTEXT, MAX_LEN, SEQS, BLOCKS = 8192, 16384, 4, 320
PROMPTS, TOKENS = (37, 250, 5, 130), 32
# The run without speculation goes 2k tokens further, so every forced draft has its token.
OFF_TOKENS = TOKENS + 2 * K
MARGIN, MAX_EXCLUDED, MAX_CUT = 1e-4, 4, 1
NEURON = {"decode_context_length_buckets": [CONTEXT], "num_seqs_buckets": [SEQS]}
# The proposal-input sequences: each verify window's first position and accepted count.
STARTS, ACCEPTED = (0, 1, 100, 251, 252, 255, 600, 1010), (0, 4, 3, 4, 0, 1, 2, 4)

HEADS: set[int] = set()  # storages of the target's stacked LM-head halves
CALLS: dict[str, list] = {"block": [], "taps": [], "margins": [], "ids": []}


def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """fp32 RMS norm of the last dim, without a gain."""
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def _qkv(hidden, weight, norm_weight, eps):
    """Stand-in fused norm and projection; the LM head's columns come back in fp32."""
    rows = _rms(hidden, eps).reshape(-1, hidden.shape[-1]) * norm_weight.float().reshape(-1)
    out = torch.cat([rows @ w.float() for w in weight.split(8192, 1)], -1)
    out = out.view(*hidden.shape[:-1], -1)
    return out if weight.untyped_storage().data_ptr() in HEADS else out.bfloat16()


def _prefill_projection(x, weight, heads=False):
    """Stand-in prefill output projection, in the input's dtype."""
    return (x.float() @ weight.float()).to(x.dtype)


def _conv(history, q, k, v, weight):
    """Stand-in causal depthwise conv and SiLU: ``[C, T]`` bf16."""
    x = torch.cat([history, torch.cat([q, k, v], -1)]).float()
    tokens = q.shape[0]
    out = sum(weight[:, s] * x[s : s + tokens] for s in range(weight.shape[1]))
    return F.silu(out).t().contiguous().bfloat16()


def _kda_prefill(q, k, v, g, beta, a_log, dt_bias, state=None, valid=None, *, lower_bound,
                 scale=None):
    """Stand-in KDA recurrence over one sequence, token by token in fp32."""
    heads, tokens, dim = g.shape
    q, k, v = (t.float().view(heads, dim, tokens).transpose(1, 2) for t in (q, k, v))
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * (scale or dim**-0.5)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    gate = a_log.float().exp().view(heads, 1, 1) * (g.float() + dt_bias.float().view(heads, 1, dim))
    decay = torch.exp(lower_bound * torch.sigmoid(gate))
    b = torch.sigmoid(beta.float().view(heads, tokens))
    s = torch.zeros(heads, dim, dim) if state is None else state.float().clone()
    out = []
    for t in range(tokens):
        if valid is not None and not bool(valid[t]):
            out.append(torch.zeros(heads, dim))
            continue
        s = s * decay[:, t, :, None]
        resid = (s * k[:, t, :, None]).sum(1) - v[:, t]
        s = s - b[:, t, None, None] * k[:, t, :, None] * resid[:, None, :]
        out.append((s * q[:, t, :, None]).sum(1))
    return torch.stack(out, 1), s


def _kda_output(core, gate, o_norm_weight, o_proj_weight, eps):
    """Stand-in gated output norm and ``o_proj``: ``[T, H]`` fp32."""
    tokens, width = core.shape
    rows = _rms(core.view(tokens, -1, o_norm_weight.shape[0]), eps) * o_norm_weight.float()
    return (rows.reshape(tokens, width) * torch.sigmoid(gate)) @ o_proj_weight.float().t()


def _kda_decode(x, slots, pages, *weights, lower_bound, norm_eps):
    """Stand-in KDA decode: the verify mode's CPU step with a one-token window."""
    from vllm_neuron.functional.kda.fused_decode import _verify_on_cpu

    return _verify_on_cpu(x, slots, slots[:, None], pages, *weights, float(lower_bound),
                          float(norm_eps))


def _mix(sources, score_weight, eps):
    """AttnRes mix of ``[T, S, H]`` sources by a softmax over their scores, in fp32."""
    scores = (_rms(sources, eps) * score_weight).sum(-1)
    return (scores.softmax(-1).unsqueeze(-1) * sources.float()).sum(1)


def _mix_prefill(bank, stream, score_w, eps):
    """Stand-in AttnRes prefill mix."""
    return _mix(torch.cat([bank, stream[:, None]], 1), score_w, eps).to(stream.dtype)


def _mix_decode(bank, stream, score_weight, num_sources, *, residual=None, moe_tail=None,
                latent_size=None, bank_slot=None, norm_weight=None, tp_rank=None, tp_size=1,
                dp_rank=None, dp_size=1, eps=1e-5):
    """Stand-in AttnRes decode mix: fold the tail in, store the bank row, mix, slice K."""
    from vllm_neuron.functional.attention.attn_res_decode import AttnResDecodeOutput

    new = stream
    if residual is not None:
        new = stream + residual
    if moe_tail is not None:
        up, shared, squares = moe_tail.split([stream.shape[1], stream.shape[1], 1], -1)
        new = stream + (up * torch.rsqrt(squares / latent_size + eps) + shared).bfloat16()
    if bank_slot is not None:
        bank[:, bank_slot] = new
    mix = _mix(torch.cat([bank[:, : num_sources - 1], new[:, None]], 1), score_weight, eps)
    k_slice = None
    if norm_weight is not None:
        cols = _rms(mix, eps) * norm_weight.float()
        width = cols.shape[1] // tp_size
        start = int(tp_rank) * width if tp_rank is not None else 0
        part = cols[:, start : start + width]
        k_slice = part.reshape(part.shape[0], -1, 128).permute(2, 0, 1).unsqueeze(-1).contiguous()
    return AttnResDecodeOutput(mix.bfloat16(), new, k_slice, bank if bank_slot is not None else None)


def _latent(pages, offset, block_size):
    """The latent rows ``[blocks, block_size, 576]`` bf16 view of the page buffer."""
    width = block_size * ROW * 2
    return pages[:, offset : offset + width].view(torch.bfloat16).view(-1, block_size, ROW)


def _store(cache, qkv_a, kv_norm_weight, slots, eps):
    """Write each token's normed latent row and rope part at its slot; -1 writes block 0."""
    kv = qkv_a[:, 1536:]
    rows = torch.cat([_rms(kv[:, :512], eps) * kv_norm_weight.float(), kv[:, 512:]], -1)
    slots = slots.long().clamp(min=0)
    cache[slots // cache.shape[1], slots % cache.shape[1]] = rows.bfloat16()


def _attend(cache, table, length, qkv_a, gate, q_norm_weight, w_qb_t, w_uk, w_uv_t, eps):
    """One token's MLA over the first ``length`` latent rows its table row holds."""
    heads, block_size = w_uk.shape[0], cache.shape[1]
    positions = torch.arange(max(length, 1))
    pages = table[(positions // block_size).clamp(max=table.shape[0] - 1)].long().clamp(min=0)
    rows = cache[pages, positions % block_size].float()
    q = (_rms(qkv_a[None, :1536], eps) * q_norm_weight.float()) @ w_qb_t.float()
    q = q.view(heads, -1)
    q_latent = torch.einsum("hd,hdl->hl", q[:, :128], w_uk.float())
    score = (q_latent @ rows[:, :512].t() + q[:, 128:] @ rows[:, 512:].t()) * q.shape[1] ** -0.5
    out = torch.einsum("hl,hlv->hv", score.softmax(-1) @ rows[:, :512], w_uv_t.float())
    return out * torch.sigmoid(gate.float().view(heads, -1))


def _mla_prefill(qkv_a, gate, q_w, kv_w, w_qb_t, w_uk, w_uv_t, pages, offset, block_size,
                 table, prefix_len, slots, eps):
    """Stand-in MLA prefill chunk: store the rows, then attend causally."""
    cache = _latent(pages, offset, block_size)
    _store(cache, qkv_a, kv_w, slots, eps)
    start = int(prefix_len.reshape(-1)[0])
    out = [_attend(cache, table, start + t + 1, qkv_a[t], gate[t], q_w, w_qb_t, w_uk, w_uv_t, eps)
           for t in range(qkv_a.shape[0])]
    return torch.stack(out).flatten(1).bfloat16(), pages


def _mla_decode(qkv_a, gate, q_w, kv_w, w_qb_t, w_uk, w_uv_t, pages, offset, block_size,
                table, context_lens, slots, eps):
    """Stand-in MLA decode and verify window: store every row, then attend causally.

    Token t belongs to table row ``t // (T / rows)`` and reads rows 0 to its position.
    """
    cache = _latent(pages, offset, block_size)
    _store(cache, qkv_a, kv_w, slots, eps)
    per_row = qkv_a.shape[0] // table.shape[0]
    out = [_attend(cache, table[t // per_row], int(context_lens[t]) + 1, qkv_a[t], gate[t], q_w,
                   w_qb_t, w_uk, w_uv_t, eps) for t in range(qkv_a.shape[0])]
    return torch.stack(out).permute(2, 0, 1).unsqueeze(-1).bfloat16().contiguous(), pages


def _select(logits, score_bias, scaling=1.0):
    """Stand-in router: top 16 by biased sigmoid score, renormalized weights."""
    scores = torch.sigmoid(logits.float())
    ids = (scores + score_bias).topk(16, dim=-1).indices
    weights = scores.gather(1, ids)
    return ids.int(), weights / weights.sum(-1, keepdim=True) * scaling


def _expert(rows, local):
    """Local expert ``local``'s stand-in output on fp32 latent rows."""
    return torch.tanh(rows) * (0.5 + 0.05 * local)


def _tiles(rows, weights, *banks, beta, linear_beta, expose_codes=False):
    """Stand-in prefill expert tiles: each item's weighted expert output."""
    return torch.stack([w[:, None] * _expert(r, n) for n, (r, w) in enumerate(zip(rows, weights))])


def _experts(routed, score_bias, expert_base, *banks, beta, linear_beta, scaling=1.0,
             replicas=2, chunks=32, expose_codes=False):
    """Stand-in decode experts: the local experts' partial in combine order."""
    latent_size = routed.shape[1] - score_bias.shape[0]
    latent, logits = routed.split([latent_size, score_bias.shape[0]], -1)
    ids, weights = _select(logits, score_bias, scaling)
    out = torch.zeros_like(latent)
    for n in range(banks[0].shape[0]):
        out += (weights * (ids == int(expert_base) + n)).sum(-1, keepdim=True) * _expert(latent, n)
    tokens = out.shape[0] // replicas
    return out.view(replicas, tokens, chunks, -1).transpose(1, 2).contiguous()


def _tail(combined, gate_up, latent_norm_weight, latent_up_weight, shared_down_weight, beta,
          linear_beta):
    """Stand-in MoE decode tail: the fp32 pack ``[up | shared | sum of squares]``."""
    from vllm_neuron.model.kimi_k3.model import situ_and_mul

    up = (combined * latent_norm_weight.t()).bfloat16().float() @ latent_up_weight.float()
    shared = situ_and_mul(*gate_up.chunk(2, -1), beta, linear_beta).float() @ shared_down_weight.float()
    return torch.cat([up, shared, combined.square().sum(-1, keepdim=True)], -1)


def _block_pass(self, embedded, positions, attn_metadata):
    """Stand-in draft block pass: each block row plus the mean of its context rows."""
    meta = attn_metadata[self.layers[0].cache_name]
    table, lengths = meta["block_table_tensor"], meta["context_lens"]
    batch = table.shape[0]
    CALLS["block"].append({"positions": positions.view(batch, -1).clone(),
                           "context_lens": lengths.clone()})
    context = []
    for b in range(batch):
        n = int(lengths[b])
        rows = [_latent_rows(layer.self_attn.kv_cache, table[b], n) for layer in self.layers]
        context.append(sum(r.float().mean(0) for r in rows) if n > 0 else torch.zeros(ROW))
    context = torch.stack(context).repeat(1, HIDDEN // ROW + 1)[:, :HIDDEN]
    steps = torch.arange(1, K + 1).view(1, K, 1) / K
    return (embedded.float().view(batch, -1, HIDDEN) + context[:, None] * steps).bfloat16()


def _latent_rows(cache, table_row, n):
    """The first ``n`` latent rows of a ``[blocks, block_size, 576]`` cache by its table row."""
    positions = torch.arange(n)
    pages = table_row[positions // cache.shape[1]].long().clamp(min=0)
    return cache[pages, positions % cache.shape[1]]


def _random_load(self, checkpoint_path, device, cache_dir=None):
    """Seeded random weights: a split weight by name and rank, a whole one by name only."""
    from vllm_neuron.model.kimi_k3.model import _tp_group

    rank = _tp_group().rank_in_group
    for name, p in list(self.named_parameters()):
        path, _, leaf = name.rpartition(".")
        seed = zlib.crc32(f"{name}:{rank if hasattr(p, 'weight_loader') else 'whole'}".encode())
        gen = torch.Generator().manual_seed(seed)
        if p.dtype.is_floating_point:
            norm = "norm" in name
            scale = (0.01 if leaf == "e_score_correction_bias" else 0.1 if norm
                     else 0.5 if leaf in ("A_log", "dt_bias") or "conv" in name else 0.02)
            value = (float(norm) + scale * torch.randn(p.shape, generator=gen)).to(p.dtype)
        else:
            value = torch.zeros(p.shape, dtype=p.dtype)  # the experts' banks, read by no stand-in
        setattr(self.get_submodule(path), leaf, nn.Parameter(value, requires_grad=False))
    self.prepare_after_load()
    HEADS.update(w.untyped_storage().data_ptr() for w in self.head_weights)


def _stand_ins(patch) -> None:
    """Install every stand-in and the random weight load."""
    from vllm_neuron.model.kimi_k3 import dspark, model

    kernels = dict(
        qkv_proj_fp32acc=_qkv, _prefill_projection=_prefill_projection, depthwise_conv1d_silu=_conv,
        kda_prefill=_kda_prefill, kda_output_prefill=_kda_output, kda_fused_decode=_kda_decode,
        attn_res_prefill=_mix_prefill, attn_res_decode=_mix_decode, mla_paged_prefill=_mla_prefill,
        mla_paged_decode=_mla_decode, router_select=_select, routed_expert_tiles_mx=_tiles,
        routed_experts_mx=_experts, moe_fused_tail=_tail,
    )
    for name, fn in kernels.items():
        patch.setattr(model, name, fn)
    patch.setattr(dspark, "qkv_proj_fp32acc", _qkv)
    patch.setattr(dspark.K3DSparkForCausalLM, "block_pass", _block_pass, raising=False)
    patch.setattr(model.KimiK3ForConditionalGeneration, "load_weights", _random_load)
    forward = model.KimiK3ForConditionalGeneration.forward

    def keep_taps(self, *args, **kwargs):
        out = forward(self, *args, **kwargs)
        if isinstance(out, tuple) and len(out) > 2:
            CALLS["taps"].append(out[1])
        return out

    patch.setattr(model.KimiK3ForConditionalGeneration, "forward", keep_taps)


def _cut(config):
    """The target's config cut to 4 layers at the test widths."""
    text = getattr(config, "text_config", None)
    if text is None:  # vLLM's probe of the override on an empty config
        return config
    attn = dict(text.linear_attn_config, num_heads=6, kda_layers=[1, 2, 3], full_attn_layers=[4])
    for key, value in dict(num_hidden_layers=4, num_attention_heads=6, num_experts=16,
                           intermediate_size=1056 * TP, moe_intermediate_size=3072,
                           routed_expert_hidden_size=512, first_k_dense_replace=1,
                           linear_attn_config=attn).items():
        setattr(text, key, value)
    return config


def _engine_config(k: int | None, cut: bool = True, tp: int = TP):
    """The engine config: DSpark with k drafts, or no speculation for k None."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform

    spec = None if k is None else {"method": "dspark", "model": str(DRAFT),
                                   "num_speculative_tokens": k}
    config = EngineArgs(
        model=str(TARGET), tensor_parallel_size=tp, max_model_len=MAX_LEN, max_num_seqs=SEQS,
        enforce_eager=True, skip_tokenizer_init=True, enable_prefix_caching=False,
        language_model_only=True, hf_overrides=_cut if cut else {}, speculative_config=spec,
        async_scheduling=False, additional_config={"neuron_config": dict(NEURON)},
    ).create_engine_config()
    # Text only: the runner builds the K3 class from its text config alone.
    config.additional_config.pop("encoder_neuron_config", None)
    current_platform.update_block_size_for_backend(config)
    if k is not None and cut:
        config.speculative_config.draft_model_config.hf_config.target_layer_ids = [0, 1, 2]
    return config


def _runner(config):
    """A loaded model runner with its KV cache, as the worker and engine core set it up."""
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
    from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups, get_kv_cache_groups

    from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner
    from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

    runner = NeuronModelRunner(config, torch.device("cpu"))
    runner.load_model()
    specs = runner.get_kv_cache_spec()
    layouts = NeuronWorker.__new__(NeuronWorker).get_supported_kv_cache_layouts()
    resolve_kv_cache_layout(config, [layouts], list(specs.values()))
    groups = get_kv_cache_groups(config, specs)
    page = config.cache_config.mamba_page_size_padded
    kv = get_kv_cache_config_from_groups(
        config, groups, BLOCKS * max(len(g.layer_names) for g in groups) * page)
    config.cache_config.num_gpu_blocks = kv.num_blocks
    runner.initialize_kv_cache(kv)
    return runner, kv


def _generate(config, runner, kv, tag: str, prompts, tokens: int, on_step=None):
    """Greedy outputs of every prompt through the scheduler and the runner's step methods."""
    from vllm import SamplingParams
    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
    from vllm.v1.request import Request
    from vllm.v1.structured_output import StructuredOutputManager

    block, hashed = resolve_kv_cache_block_sizes(kv, config)
    scheduler = config.scheduler_config.get_scheduler_cls()(
        vllm_config=config, kv_cache_config=kv, block_size=block, hash_block_size=hashed,
        structured_output_manager=StructuredOutputManager(config))
    requests = [Request(f"{tag}-{i}", ids, SamplingParams(temperature=0.0, max_tokens=tokens,
                                                         ignore_eos=True), None, arrival_time=i)
                for i, ids in enumerate(prompts)]
    for request in requests:
        scheduler.add_request(request)
    while scheduler.has_requests():
        step = scheduler.schedule()
        CALLS["step"] = (scheduler, step)
        out = runner.execute_model(step)
        if out is None:
            out = runner.sample_tokens(None)
        if on_step is not None and step.total_num_scheduled_tokens > 0:
            on_step(scheduler, step, out)
        scheduler.update_from_output(step, out)
        if config.speculative_config is not None and step.total_num_scheduled_tokens > 0:
            drafts = runner.take_draft_token_ids()
            if drafts is not None:
                scheduler.update_draft_token_ids(drafts)
    return [list(r.output_token_ids) for r in requests]


def _margins(patch, margins: dict):
    """Record each sampled row's global top-2 logit margin, by request and output index."""
    import torch.distributed as dist

    from vllm_neuron.model.kimi_k3 import model

    compute = model.KimiK3ForConditionalGeneration.compute_logits

    def keep(self, hidden):
        logits = compute(self, hidden)
        top = logits.float().topk(2, dim=-1).values
        parts = [torch.empty_like(top) for _ in range(TP)]
        dist.all_gather(parts, top, group=model._tp_group().device_group)
        best = torch.cat(parts, -1).topk(2, dim=-1).values
        CALLS["margins"].append(best[:, 0] - best[:, 1])
        return logits

    patch.setattr(model.KimiK3ForConditionalGeneration, "compute_logits", keep)

    def on_step(scheduler, step, out):
        rows = CALLS["margins"].pop()
        for i, rid in enumerate(out.req_ids):
            margins[(rid, len(scheduler.requests[rid].output_token_ids))] = float(rows[i])

    return on_step


def _accepted(rounds: list):
    """Record each verify round's accepted count from the runner's output."""
    def on_step(scheduler, step, out):
        for i, rid in enumerate(out.req_ids):
            if rid in step.scheduled_spec_decode_tokens:
                rounds.append((rid, len(out.sampled_token_ids[i]) - 1))
    return on_step


def _force(runner, off, forced: dict, windows: list):
    """Replace the drafts at the proposer's return: a of them the off run's next tokens.

    At each verify round, also read what the draft pass took, and the draft cache rows at
    the window's slots beside the context write of the window's taps into a scratch cache.
    """
    drafter, draft = runner.drafter, _draft(runner)
    group = _draft_group_index(runner)
    block_size = runner.kv_cache_config.kv_cache_groups[group].kv_cache_spec.block_size
    propose, calls = drafter.propose, {}

    def forced_propose(**kwargs):
        anchors, drafts = propose(**kwargs)
        scheduler, step = CALLS["step"]
        raw = kwargs["raw_sampled_token_ids"]
        valid = ((raw != -1) & (raw < VOCAB)).sum(1)
        rids = runner.input_batch.req_ids[: drafts.shape[0]]
        taps, block = CALLS["taps"][-1], CALLS["block"][-1]
        # Each row's position from the scheduler's counts, the padding rows at 0.
        positions, starts, offset = torch.zeros(taps.shape[0], dtype=torch.long), [], 0
        for rid in rids:
            n = step.num_scheduled_tokens[rid]
            starts.append(scheduler.requests[rid].num_computed_tokens - n)
            positions[offset : offset + n] = starts[-1] + torch.arange(n)
            offset += n
        saved = [layer.self_attn.kv_cache for layer in draft.layers]
        scratch = [torch.zeros(taps.shape[0], ROW, dtype=c.dtype) for c in saved]
        for layer, c in zip(draft.layers, scratch):
            layer.self_attn.kv_cache = c
        draft.precompute_and_store_context_kv(taps, positions, torch.arange(taps.shape[0]))
        for layer, c in zip(draft.layers, saved):
            layer.self_attn.kv_cache = c
        drafts, offset = drafts.clone(), 0
        for b, (rid, q) in enumerate(zip(rids, starts)):
            n, i = step.num_scheduled_tokens[rid], int(rid.rsplit("-", 1)[1])
            if rid in step.scheduled_spec_decode_tokens:
                blocks = scheduler.kv_cache_manager.get_block_ids(rid)[group]
                slots = [blocks[p // block_size] * block_size + p % block_size
                         for p in range(q, q + n)]
                written = [layer.self_attn.kv_cache.reshape(-1, ROW)[slots] for layer in draft.layers]
                differing = sum(int((w.view(torch.int16) != c[offset : offset + n].view(torch.int16))
                                    .sum()) for w, c in zip(written, scratch))
                windows.append(dict(rid=rid, q=q, forced=forced[rid], context=int(block["context_lens"][b]),
                                    positions=block["positions"][b].tolist(), differing=differing))
            offset += n
            calls[rid] = calls.get(rid, -1) + 1
            a = (calls[rid] + i) % (K + 1)
            anchor = len(scheduler.requests[rid].output_token_ids) + int(valid[b]) - 1
            nxt = off[i][anchor + 1 : anchor + 1 + K]
            drafts[b] = torch.tensor(nxt[:a] + [(t + 1) % VOCAB for t in nxt[a:]], dtype=drafts.dtype)
            forced[rid] = a
        return torch.cat([anchors[:, :1], drafts], 1), drafts

    drafter.propose = forced_propose


def _proposal_inputs(runner) -> dict:
    """Run the proposer on 8 verify windows of mixed start; returns the ids and positions."""
    draft, group = _draft(runner), _draft_group_index(runner)
    block_size = runner.kv_cache_config.kv_cache_groups[group].kv_cache_spec.block_size
    batch, width = len(STARTS), 4
    gen = torch.Generator().manual_seed(11)
    raw = torch.randint(0, VOCAB, (batch, K + 1), generator=gen, dtype=torch.int32)
    for b, a in enumerate(ACCEPTED):
        raw[b, a + 1 :] = -1
    table = (1 + torch.arange(batch * width, dtype=torch.int32)).view(batch, width)
    meta = {"block_table_tensor": table, "block_size": block_size, "max_blocks_per_seq": width}
    hook = draft.embed_tokens.register_forward_pre_hook(
        lambda module, args: CALLS["ids"].append(args[0].clone()))
    runner.drafter.propose(
        target_token_ids=torch.zeros(batch * (K + 1), dtype=torch.int32),
        target_positions=(torch.tensor(STARTS)[:, None] + torch.arange(K + 1)).reshape(-1),
        target_hidden_states=(0.5 * torch.randn(batch * (K + 1), 3 * HIDDEN, generator=gen))
        .bfloat16(),
        last_token_indices=(K + 1) * torch.arange(batch) + K,
        attn_metadata={layer.cache_name: meta for layer in draft.layers},
        raw_sampled_token_ids=raw,
    )
    hook.remove()
    return {"raw": raw, "ids": CALLS["ids"][-1].view(batch, K),
            "positions": CALLS["block"][-1]["positions"]}


def _draft(runner):
    """The draft module under the runner's compile wrapper."""
    return getattr(runner.drafter.model, "_orig_mod", runner.drafter.model)


def _draft_group_index(runner) -> int:
    """The index of the KV cache group that holds the draft's first layer."""
    name = _draft(runner).layers[0].cache_name
    return next(i for i, g in enumerate(runner.kv_cache_config.kv_cache_groups)
                if name in g.layer_names)


def _draft_group(runner) -> dict:
    """The KV cache group that holds the draft's layers."""
    names = [layer.cache_name for layer in _draft(runner).layers]
    groups = [g for g in runner.kv_cache_config.kv_cache_groups
              if set(names) & set(g.layer_names)]
    spec = groups[0].kv_cache_spec
    return dict(groups=len(groups), all_in=set(names) <= set(groups[0].layer_names),
                kind=type(spec).__name__, heads=spec.num_kv_heads, width=spec.head_size,
                layers=list(groups[0].layer_names), draft_layers=names)


def _rank(rank: int, port: int, root: str) -> None:
    """One TP rank: the run without speculation, then runs (a) and (b), and the set-up readings."""
    import torch.distributed as dist
    from vllm import ModelRegistry
    from vllm.config import set_current_vllm_config

    from vllm_neuron.model import registry
    from vllm_neuron.parallel.neuron_parallel_state import initialize_neuron_parallel_state
    from vllm_neuron.vllm.spec_decode.dspark import DSparkProposer
    from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

    # One thread: a threaded fp32 matmul splits its sum by the row count, so a row's value
    # would change with the number of rows in the step.
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=TP)
    initialize_neuron_parallel_state(tp_global_ranks=list(range(TP)), local_rank=rank)
    # The configs come first: vLLM checks the registered class when it builds one, and the
    # worker registers the Neuron classes only once the engine config exists.
    configs = {"off": _engine_config(None), "a": _engine_config(K), "b": _engine_config(K),
               "refused": _engine_config(K + 2)}
    for arch, model_cls in registry.get_models():
        ModelRegistry.register_model(arch, model_cls)
    gen = torch.Generator().manual_seed(5)
    prompts = [torch.randint(0, VOCAB, (n,), generator=gen).tolist() for n in PROMPTS]
    out = {}
    with pytest.MonkeyPatch.context() as patch, torch.no_grad():
        _stand_ins(patch)
        for run in ("off", "a", "b"):
            config = configs[run]
            with set_current_vllm_config(config), pytest.MonkeyPatch.context() as inner:
                runner, kv = _runner(config)
                if run == "off":
                    out["margins"] = {}
                    on_step, tokens = _margins(inner, out["margins"]), OFF_TOKENS
                else:
                    out[f"{run}_rounds"], out[f"{run}_windows"] = [], []
                    on_step, tokens = _accepted(out[f"{run}_rounds"]), TOKENS
                if run == "b":
                    _force(runner, out["off"], {}, out["b_windows"])
                out[run] = _generate(config, runner, kv, run, prompts, tokens, on_step)
                if run == "a":
                    out["setup"] = dict(
                        is_dspark=isinstance(runner.drafter, DSparkProposer),
                        group=_draft_group(runner),
                        sizes=(runner.model.config.vocab_size, runner.model.config.hidden_size),
                        buckets=tuple(runner._decode_ctx_bucket_from_max_decode_ctx_len(n, K)
                                      for n in (CONTEXT - 2 * K - 2, CONTEXT - 2 * K - 1)))
                    out["inputs"] = _proposal_inputs(runner)
                del runner
                gc.collect()
        refused = []
        with set_current_vllm_config(configs["refused"]):
            try:
                NeuronModelRunner(configs["refused"], torch.device("cpu"))
            except ValueError as e:
                refused.append(str(e))
        out["setup"]["refused"] = refused
    torch.save(out, Path(root) / f"rank{rank}.pt")
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> list[dict]:
    """Each rank's readings from one two-rank world."""
    root = tmp_path_factory.mktemp("dspark")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    saved, cwd = os.environ.get("VLLM_RPC_BASE_PATH"), os.getcwd()
    # vLLM's broadcaster binds an IPC socket under this path, which must stay short.
    os.environ["VLLM_RPC_BASE_PATH"] = "."
    os.chdir(root)
    try:
        mp.spawn(_rank, args=(port, str(root)), nprocs=TP, join=True)
    finally:
        os.chdir(cwd)
        if saved is None:
            os.environ.pop("VLLM_RPC_BASE_PATH")
        else:
            os.environ["VLLM_RPC_BASE_PATH"] = saved
    return [torch.load(root / f"rank{r}.pt", weights_only=False) for r in range(TP)]


def test_runner_sets_up_the_dspark_drafter(world):
    setup = world[0]["setup"]
    group = setup["group"]
    warnings.warn(f"set-up readings: {setup}")
    errors = [name for name, ok in {
        "drafter is a DSparkProposer": setup["is_dspark"],
        "the 3 draft layers in one group": len(group["draft_layers"]) == 3
        and group["groups"] == 1 and group["all_in"],
        "the group is key-only, one head 576 wide": (group["kind"], group["heads"], group["width"])
        == ("KeyOnlyAttentionSpec", 1, ROW),
        "target sizes 163,840 and 7,168": setup["sizes"] == (VOCAB, HIDDEN),
    }.items() if not ok]
    assert errors == [], errors
    # Headroom 2k + 2 = 10: 8,182 tokens take the 8K bucket, 8,183 the next one.
    assert setup["buckets"] == (CONTEXT, MAX_LEN), setup["buckets"]
    assert len(setup["refused"]) == 1 and "block size" in setup["refused"][0], setup["refused"]


def test_proposal_inputs_are_the_anchor_then_masks_at_the_context_length(world):
    from vllm_neuron.vllm.spec_decode.dspark import DSparkProposer

    differing = []
    for rank, state in enumerate(world):
        raw, ids, positions = (state["inputs"][key] for key in ("raw", "ids", "positions"))
        for b, (q, a) in enumerate(zip(STARTS, ACCEPTED)):
            start = q + a + 1  # the context length after a accepted drafts
            want = ([int(raw[b, a])] + [MASK] * (K - 1), list(range(start, start + K)))
            for name, got, expected in zip(("ids", "positions"), (ids[b], positions[b]), want):
                if got.tolist() != expected:
                    differing.append((rank, b, name, got.tolist(), expected))
    config = _engine_config(K, cut=False, tp=1)
    layers = DSparkProposer(config, torch.device("cpu")).get_target_aux_hidden_state_layers()
    layer_count = config.model_config.hf_text_config.num_hidden_layers
    warnings.warn(f"proposal inputs: {len(differing)} differences; aux layers {layers} "
                  f"from {layer_count} target layers")
    assert differing == [], differing
    assert (layers, layer_count) == ((72, 88, 92), 93)


def _compare(got: list, off: list, excluded: set) -> tuple[list, list]:
    """Differing tokens by (prompt, index), and the prompts cut at an excluded first difference."""
    differing, cut = [], []
    for i, (tokens, reference) in enumerate(zip(got, off)):
        diffs = [g for g in range(TOKENS) if g >= len(tokens) or tokens[g] != reference[g]]
        if diffs and (i, diffs[0]) in excluded:
            cut.append((i, diffs[0]))
        else:
            differing += [(i, g) for g in diffs]
    return differing, cut


def test_speculative_outputs_equal_the_run_without_speculation(world):
    state = world[0]
    excluded = {(int(rid.rsplit("-", 1)[1]), g) for (rid, g), m in state["margins"].items()
                if g < TOKENS and m < MARGIN}
    readings = {"excluded": sorted(excluded)}
    for run in ("a", "b"):
        differing, cut = _compare(state[run], state["off"], excluded)
        readings[run] = dict(differing=differing, cut=cut,
                             ranks_agree=all(s[run] == state[run] for s in world),
                             lengths=[len(t) for t in state[run]])
    readings["a_accepted"] = [n for _, n in state["a_rounds"]]
    # Run (b): each window's accepted count, what the next draft pass read, and its cache rows.
    misses = []
    for rank, s in enumerate(world):
        accepted = {}
        for rid, n in s["b_rounds"]:
            accepted.setdefault(rid, []).append(n)
        seen = {}
        for w in s["b_windows"]:
            n = accepted[w["rid"]][seen.setdefault(w["rid"], 0)]
            seen[w["rid"]] += 1
            start = w["q"] + n + 1
            checks = {"accepted": n == w["forced"], "context rows": w["context"] == start,
                      "block positions": w["positions"] == list(range(start, start + K)),
                      "cache rows": w["differing"] == 0}
            misses += [(rank, w["rid"], w["q"], name) for name, ok in checks.items() if not ok]
    readings["b_windows"] = len(world[0]["b_windows"])
    readings["b_forced"] = sorted({w["forced"] for w in world[0]["b_windows"]})
    readings["b_misses"] = misses
    warnings.warn(f"speculative outputs: {readings}")
    adjudicable = len(excluded) <= MAX_EXCLUDED and all(
        len(readings[run]["cut"]) <= MAX_CUT for run in ("a", "b"))
    assert adjudicable, f"not adjudicable: {len(excluded)} excluded positions, cuts " \
        f"{[readings[run]['cut'] for run in ('a', 'b')]}"
    for run in ("a", "b"):
        assert readings[run]["ranks_agree"] and readings[run]["lengths"] == [TOKENS] * len(PROMPTS)
        assert readings[run]["differing"] == [], (run, readings[run]["differing"])
    assert readings["b_forced"] == list(range(K + 1))
    assert misses == [], misses
