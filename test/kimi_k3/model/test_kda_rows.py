# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 KDA state pages under speculation: one page per verify token, and the next pick.

With 4 draft tokens a request holds 5 pages in each KDA state group. A verify step reads
each sequence's source page and writes the state after window token j to page j, and the
next step starts from page a, a the accepted draft count, picked on the device. The items
run the runner's tables and count paths and the model's KDA layers on the 4-layer config
with 6 KDA heads at TP 2, one gloo process per rank, against a run without speculation
that steps the accepted tokens one at a time. Each rank builds its KDA layers at one
rank's widths, 3 heads, and a layer's input rows come from a random table by token id.
The fused norm and projection and the prefill kernels have no CPU path, so plain torch
stands in for them; the one-token step is the verify step at a window of 1.

Set ``KIMI_K3_CHECKPOINT`` to the checkpoint directory; the default is
``~/models/moonshotai/Kimi-K3``. Run with ``VLLM_NEURON_CPU_MODE=1``.
"""

import copy
import json
import os
import socket
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

CHECKPOINT = Path(
    os.environ.get("KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3")
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "config.json").is_file(),
    reason=f"Kimi K3 checkpoint not found at {CHECKPOINT}",
)

TP, DRAFTS, PROMPT, BLOCKS = 2, 4, 8, 96
WINDOW = 1 + DRAFTS
# The target's picks are ids below 32, so a draft from 32 up is always rejected.
TRUE_IDS, VOCAB = 32, 64
TOLERANCE = 1e-5
# Accepted draft counts per verify step. Sequence i accepts i drafts at each of 3 steps.
FIXED = [{f"s{i}": i for i in range(5)}] * 3
# Mixed counts over 8 steps: b sits out step 4; a ends after step 5 and e takes its pages.
MIXED = [
    {"a": 0, "b": 1, "c": 2, "d": 3},
    {"a": 4, "b": 0, "c": 3, "d": 1},
    {"a": 2, "b": 4, "c": 1, "d": 0},
    {"a": 1, "c": 4, "d": 2},
    {"a": 3, "b": 2, "c": 0, "d": 4},
    {"b": 1, "c": 3, "d": 2, "e": 2},
    {"b": 3, "c": 1, "d": 0, "e": 4},
    {"b": 0, "c": 2, "d": 3, "e": 1},
]


class _DeviceOnly(torch.Tensor):
    """A tensor that counts the ops on it and records each read of its values by the host."""

    ops, reads = 0, []
    HOST = {"item", "tolist", "numpy", "cpu", "__bool__", "__int__", "__float__", "__index__",
            "__array__", "__format__"}

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", str(func))
        if name in cls.HOST:
            cls.reads.append(name)
        cls.ops += 1
        return super().__torch_function__(func, types, args, kwargs or {})


def _plain(t: torch.Tensor) -> torch.Tensor:
    return t.as_subclass(torch.Tensor)


def _norm_project(x, weight, norm_weight, eps):
    """Stand-in for the fused RMSNorm and projection, in fp64: ``[1, T, I]`` bf16."""
    x = x.double()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * norm_weight.double()
    return (x @ weight.double()).bfloat16()


def _forget_projection(x, weight):
    """Stand-in for the prefill output projection, in fp64."""
    return (x.double() @ weight.double()).bfloat16()


def _conv(history, q, k, v, weight):
    """Stand-in for the short conv and SiLU over the conv rows and the chunk: ``[3W, T]`` bf16."""
    rows = torch.cat([history, torch.cat([q, k, v], -1)]).float()
    tokens = rows.shape[0] - weight.shape[1] + 1
    out = sum(rows[s : s + tokens] * weight[:, s] for s in range(weight.shape[1]))
    return F.silu(out).t().bfloat16()


def _recurrence(q, k, v, gates, beta, a_log, dt_bias, *, state, valid, lower_bound):
    """Stand-in for the chunk recurrence: the verify step's gated delta rule over valid tokens."""
    heads, tokens, dim = gates.shape
    q, k, v = (t.float().view(heads, dim, tokens).transpose(1, 2) for t in (q, k, v))
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * dim**-0.5
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    gate = a_log.exp()[:, None, None] * (gates + dt_bias[:, None])
    decay = torch.exp(lower_bound * torch.sigmoid(gate))
    beta, state = torch.sigmoid(beta.float()), state.float().clone()
    core = torch.zeros(heads, tokens, dim)
    for t in range(tokens):
        if valid[t]:
            state = state * decay[:, t, :, None]
            resid = (state * k[:, t, :, None]).sum(-2) - v[:, t]
            state = state - beta[:, t, None, None] * k[:, t, :, None] * resid[:, None, :]
            core[:, t] = (state * q[:, t, :, None]).sum(-2)
    return core, state


def _engine_config():
    """The 4-layer config, 6 KDA heads, at TP 2 with 4 draft tokens."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform

    with open(CHECKPOINT / "config.json") as f:
        attn = json.load(f)["text_config"]["linear_attn_config"]
    kept = {key: [i for i in attn[key] if i <= 4] for key in ("kda_layers", "full_attn_layers")}
    text = {"num_hidden_layers": 4, "linear_attn_config": {**attn, "num_heads": 6, **kept}}
    spec = {"method": "ngram", "num_speculative_tokens": DRAFTS, "prompt_lookup_max": 4,
            "prompt_lookup_min": 2}
    config = EngineArgs(model=str(CHECKPOINT), tensor_parallel_size=TP, max_model_len=4096,
                        enable_prefix_caching=False, hf_overrides={"text_config": text},
                        speculative_config=spec, gpu_memory_utilization=0.9).create_engine_config()
    current_platform.update_block_size_for_backend(config)
    return config


def _model(config, rank: int):
    """The model on the meta device at one rank's widths, its KDA layers given seeded weights."""
    from vllm_neuron.model.kimi_k3 import model as kimi

    hf = copy.deepcopy(config.model_config.hf_config.to_dict())
    attn = hf["text_config"]["linear_attn_config"]
    attn["num_heads"] //= TP
    with torch.device("meta"):
        model = kimi.KimiK3ForConditionalGeneration.from_configs(hf)
    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            if layer.is_kda:
                attn = layer.self_attn.to_empty(device="cpu")
                gen = torch.Generator().manual_seed(100 * i + rank)
                for p in attn.parameters():
                    scale = p.shape[1] ** -0.5 if p.dim() == 2 else 0.5
                    p.copy_(scale * torch.randn(p.shape, generator=gen))
                attn.prepare_after_load()
    return model


def _runner(config, model):
    """A runner holding what cache setup, the step tables and the count paths read."""
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
    from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups, get_kv_cache_groups

    from vllm_neuron.model.neuron_config import NeuronConfig
    from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner
    from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    vars(runner).update(
        vllm_config=config, model=model, speculative_config=config.speculative_config,
        # Context buckets on, so the state groups' exemption from the decode trim is in force.
        neuron_config=NeuronConfig(decode_context_length_buckets=[1024, 4096]),
        max_num_reqs=16, max_model_len=config.model_config.max_model_len,
        max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,
        device=torch.device("cpu"), vocab_size=model.text_config.vocab_size, is_pooling_model=False,
        cp_world_size=1, cp_rank=0, _dcp_size=1, is_dflash_spec=False, _is_synthetic_model=False,
        is_dspark_spec=False,
        _kv_snapshot_enabled=False,
        _kv_cache_full_tensors={}, _block_table_groups={}, _swa_groups={}, _constant_tensors={},
    )
    specs = runner.get_kv_cache_spec()
    layouts = NeuronWorker.__new__(NeuronWorker).get_supported_kv_cache_layouts()
    resolve_kv_cache_layout(config, [layouts], list(specs.values()))
    groups = get_kv_cache_groups(config, specs)
    slots = max(len(g.layer_names) for g in groups)
    size = BLOCKS * slots * config.cache_config.mamba_page_size_padded
    runner.initialize_kv_cache(get_kv_cache_config_from_groups(config, groups, size))
    return runner


class _Run:
    """One rank's speculative run and its reference run over one schedule of accepted counts."""

    def __init__(self, config, rank: int, asynchronous: bool):
        from vllm_neuron.model.kimi_k3 import model as kimi
        from vllm_neuron.vllm.worker.neuron_model_runner import _compute_slot_mapping_cpu

        self.slot_mapping, self.asynchronous = _compute_slot_mapping_cpu, asynchronous
        kimi.kda_fused_verify = self._record
        spec, ref = _model(config, rank), _model(config, rank)
        self.runner = _runner(config, spec)
        groups = self.runner.kv_cache_config.kv_cache_groups
        self.group = {n: gid for gid, g in enumerate(groups) for n in g.layer_names}
        self.state_groups = [gid for gid in range(len(groups)) if self.runner._is_state_group(gid)]
        page = config.cache_config.mamba_page_size_padded
        hidden = config.model_config.get_hidden_size()
        self.layers, self.rows, self.norms, self.calls = [], {}, {}, []
        gen = torch.Generator().manual_seed(0)
        for i, (s, r) in enumerate(zip(spec.model.layers, ref.model.layers)):
            if s.is_kda:
                name, pages = f"layers.{i}.self_attn", torch.zeros(8, page, dtype=torch.int8)
                r.self_attn.bind_pages(pages, kimi._typed_view(pages, s.self_attn._conv_layout),
                                       kimi._typed_view(pages, s.self_attn._state_layout))
                self.layers.append((name, s.self_attn, r.self_attn))
                self.rows[name] = torch.randn(VOCAB, hidden, generator=gen)
                self.norms[name] = 1 + 0.1 * torch.randn(hidden, generator=gen)
        self.free, self.seqs, self.trace, self.stats = list(range(1, BLOCKS)), {}, [], {}
        if asynchronous:
            table = self.runner._state_resume_table
            self.runner._state_resume_table = table.as_subclass(_DeviceOnly)

    def _record(self, x, source, rows, pages, *weights, **kw):
        """The verify call, recording its pages; the kernel reads them on the device."""
        from vllm_neuron.functional.kda.fused_decode import kda_fused_verify

        self.calls.append((_plain(source).clone(), _plain(rows).clone()))
        return kda_fused_verify(x, _plain(source), _plain(rows), pages, *weights, **kw)

    def _table(self, name: str) -> np.ndarray:
        buffer = self.runner.input_batch.block_table[self.group[name]].block_table
        return self.runner._raw_block_table(buffer)

    def add(self, name: str, items: tuple[str, ...], block_ids=None):
        """A new sequence on fresh pages, or on ``block_ids``, with a seeded token stream."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.worker.gpu_input_batch import CachedRequestState

        if block_ids is None:
            sizes = [WINDOW if g in self.state_groups else 1 for g in range(len(self.group))]
            block_ids = tuple([self.free.pop(0) for _ in range(n)] for n in sizes)
        gen = torch.Generator().manual_seed(len(self.seqs))
        ids = torch.randint(0, TRUE_IDS, (PROMPT + 10 * WINDOW,), generator=gen).tolist()
        req = CachedRequestState(req_id=name, prompt_token_ids=ids[:PROMPT], mm_features=[],
                                 sampling_params=SamplingParams(), generator=None,
                                 block_ids=tuple(list(b) for b in block_ids), num_computed_tokens=0,
                                 output_token_ids=[])
        seq = types.SimpleNamespace(name=name, items=items, req=req, ids=ids,
                                    slot=len(self.seqs) + 1, pos=0, count=0, steps=0)
        self.seqs[name] = seq
        return seq

    def _schedule(self, seqs):
        """Make ``seqs`` the persistent batch, as the runner's state update does each step."""
        batch = self.runner.input_batch
        for req_id in [r for r in batch.req_id_to_index if r not in {s.name for s in seqs}]:
            batch.remove_request(req_id)
        new = [s.req for s in seqs if s.name not in batch.req_id_to_index]
        self.runner._zero_new_state_slots(new)
        for req in new:
            batch.add_request(req)
        batch.condense()
        batch.refresh_metadata()
        self.runner._save_block_table_to_device()
        return {s.name: batch.req_id_to_index[s.name] for s in seqs}

    def _sampled(self, rows: list[list[int]], padded: int):
        """The step's sampled ids, a host list, or a padded device tensor with -1 for rejected."""
        if self.asynchronous:
            ids = torch.full((padded, WINDOW), -1, dtype=torch.long)
            for i, row in enumerate(rows):
                ids[i, : len(row)] = torch.tensor(row)
            rows = ids.as_subclass(_DeviceOnly)
        self.runner._update_states_after_model_execute(rows, None)

    def prefill(self, seq):
        """Prefill one sequence alone in both runs; the runner never mixes prefill and decode."""
        self._schedule([seq])
        meta = self.runner._build_attention_metadata(1, PROMPT, PROMPT, DRAFTS, cached_seq_len=0)
        for gid, group in enumerate(self.runner.kv_cache_config.kv_cache_groups):
            slots, table = np.zeros(PROMPT, dtype=np.int64), self._table(group.layer_names[0])
            self.slot_mapping(table, slots, np.arange(PROMPT), np.zeros(PROMPT, dtype=np.int64),
                              group.kv_cache_spec.block_size)
            for name in group.layer_names:
                meta[name]["slot_mapping"] = torch.from_numpy(slots)
        ref_meta = {"block_table_tensor": torch.tensor([[seq.slot]]), "cached_seq_len":
                    torch.tensor([[0]]), "slot_mapping": torch.full((PROMPT,), seq.slot)}
        for name, spec, ref in self.layers:
            rows = self.rows[name][seq.ids[:PROMPT]]
            spec.forward_prefill(rows, self.norms[name], meta[name])
            ref.forward_prefill(rows, self.norms[name], ref_meta)
        self._sampled([[seq.ids[PROMPT]]], 1)
        seq.req.num_computed_tokens = seq.pos = PROMPT
        self.runner.input_batch.num_computed_tokens_cpu[0] = PROMPT

    def step(self, counts: dict[str, int]):
        """A verify step with ``counts`` accepted drafts each, then the reference's steps."""
        seqs = [self.seqs[n] for n in counts]
        row = self._schedule(seqs)
        padded = 1 << len(seqs).bit_length()
        window = torch.full((padded, WINDOW), TRUE_IDS)
        for s in seqs:
            window[row[s.name]] = torch.tensor(
                [s.ids[s.pos + j] if j <= counts[s.name] else TRUE_IDS + j for j in range(WINDOW)])
        ctx = max(s.pos for s in seqs) + WINDOW
        meta = self.runner._build_attention_metadata(padded, padded * WINDOW, WINDOW, DRAFTS,
                                                     cached_seq_len=ctx, max_decode_ctx_len=ctx)
        for name, spec, _ in self.layers:
            mix = self.rows[name][window.flatten()]
            spec.forward_decode(mix, None, self.norms[name], meta[name])
            source, rows = self.calls.pop()
            table = self._table(name)
            bad_rows = int(not np.array_equal(rows.numpy(), table[:padded, :WINDOW]))
            for s in seqs:
                picked, pages = int(source[row[s.name]]), table[row[s.name]]
                for st in self._stats(s):
                    st["picks"] += 1
                    st["wrong_picks"] += int(picked != pages[s.count])
                    st["first_page_0"] &= bool(s.steps > 0 or picked == pages[0])
                    st["rows_wrong"] += bad_rows
        self.trace.append(_plain(meta[self.layers[0][0]]["state_resume_rows"]).tolist())
        order = sorted(seqs, key=lambda s: row[s.name])
        self._sampled([s.ids[s.pos + 1 : s.pos + counts[s.name] + 2] for s in order], padded)
        for j in range(max(counts.values()) + 1):
            batch = [s for s in seqs if counts[s.name] >= j]
            ref_meta = {"block_table_tensor": torch.tensor([[s.slot] for s in batch])}
            for name, _, ref in self.layers:
                ref.forward_decode(self.rows[name][[s.ids[s.pos + j] for s in batch]], None,
                                   self.norms[name], ref_meta)
        for s in seqs:
            s.pos, s.count, s.steps = s.pos + counts[s.name] + 1, counts[s.name], s.steps + 1
            self._compare(s, row[s.name])

    def _stats(self, seq) -> list[dict]:
        new = {"state": 0.0, "conv": 0.0, "picks": 0, "wrong_picks": 0, "rows_wrong": 0,
               "first_page_0": True, "steps": 0}
        return [self.stats.setdefault(item, dict(new)) for item in seq.items]

    def _compare(self, seq, row: int):
        """The relative max error of the sequence's accepted page against the reference's page."""
        stats = self._stats(seq)
        for st in stats:
            st["steps"] += 1
        for name, spec, ref in self.layers:
            page = self._table(name)[row, seq.count]
            for key, part in (("state", "recurrent_state"), ("conv", "conv_state")):
                x, y = getattr(spec, part)[page].float(), getattr(ref, part)[seq.slot].float()
                # A NaN or a zero reference counts as an infinite error.
                err = torch.nan_to_num((x - y).abs().max() / y.abs().max(), nan=float("inf"))
                for st in stats:
                    st[key] = max(st[key], float(err))


def _rank(rank: int, port: int, out: str) -> None:
    """One TP rank: the fixed-count run, then the mixed run with a sequence joining late."""
    import torch.distributed as dist

    from vllm_neuron.model.kimi_k3 import model as kimi
    from vllm_neuron.functional.kda.fused_decode import kda_fused_verify

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=TP)
    kimi.qkv_proj_fp32acc, kimi._prefill_projection = _norm_project, _forget_projection
    kimi.depthwise_conv1d_silu, kimi.kda_prefill = _conv, _recurrence
    kimi.kda_output_prefill = lambda core, gate, norm, proj, eps: torch.zeros(
        core.shape[0], proj.shape[0])
    kimi.kda_fused_decode = lambda x, slots, pages, *w, **kw: kda_fused_verify(
        x, slots, slots[:, None], pages, *w, **kw)
    config = _engine_config()
    result = {}

    fixed = _Run(config, rank, asynchronous=False)
    for i in range(5):
        fixed.prefill(fixed.add(f"s{i}", ("fixed",)))
    for counts in FIXED:
        fixed.step(counts)
    result["fixed"] = fixed.stats["fixed"]

    mixed = _Run(config, rank, asynchronous=True)
    _DeviceOnly.ops, _DeviceOnly.reads = 0, []
    for name in "abcd":
        mixed.prefill(mixed.add(name, ("mixed",)))
    for counts in MIXED[:5]:
        mixed.step(counts)
    a = mixed.seqs["a"]
    key = a.req.block_ids[mixed.state_groups[0]][0]
    stale = int(_plain(mixed.runner._state_resume_table)[key])
    mixed.prefill(mixed.add("e", ("mixed", "join"), block_ids=a.req.block_ids))
    for counts in MIXED[5:]:
        mixed.step(counts)
    reads = {"host_reads": _DeviceOnly.reads, "device_ops": _DeviceOnly.ops}
    result["mixed"] = {**mixed.stats["mixed"], **reads}
    result["join"] = {**mixed.stats["join"], "stale_count": stale}

    traces = [None] * TP
    dist.all_gather_object(traces, [fixed.trace, mixed.trace])
    result["ranks_agree"] = all(t == traces[0] for t in traces)
    with open(os.path.join(out, f"rank{rank}.json"), "w") as f:
        json.dump(result, f)


@pytest.fixture(scope="module")
def ranks(tmp_path_factory) -> list[dict]:
    """Each rank's results, from one two-rank gloo run."""
    out = tmp_path_factory.mktemp("ranks")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_rank, args=(port, str(out)), nprocs=TP, join=True)
    return [json.loads((out / f"rank{r}.json").read_text()) for r in range(TP)]


def _equal_to_reference(item: dict, steps: int, picks: int) -> None:
    assert (item["steps"], item["picks"]) == (steps, picks)
    assert (item["wrong_picks"], item["rows_wrong"]) == (0, 0)
    assert item["state"] <= TOLERANCE and item["conv"] <= TOLERANCE, item


def test_each_count_0_to_4_resumes_from_its_page(ranks):
    for rank in ranks:
        _equal_to_reference(rank["fixed"], steps=3 * 5, picks=3 * 5 * 3)
        assert rank["ranks_agree"]


def test_eight_async_steps_with_mixed_counts_never_read_the_count_on_the_host(ranks):
    for rank in ranks:
        _equal_to_reference(rank["mixed"], steps=31, picks=31 * 3)
        assert rank["mixed"]["host_reads"] == [] and rank["mixed"]["device_ops"] > 0


def test_a_sequence_joining_after_prefill_reads_page_0(ranks):
    for rank in ranks:
        _equal_to_reference(rank["join"], steps=3, picks=3 * 3)
        assert rank["join"]["first_page_0"] and rank["join"]["stale_count"] == MIXED[4]["a"]
