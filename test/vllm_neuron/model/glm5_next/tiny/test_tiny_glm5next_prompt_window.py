# SPDX-License-Identifier: Apache-2.0
"""Any prompt up to max_model_len - 1 tokens is served, on both serve lines, through the runner.

The tiny GLM-5.3-Flash root (three sparse-attention layers, dense and MoE halves, a seeded
random head so greedy tokens discriminate) is driven the way a served request reaches it:
the runner's own metadata builder (``_build_attention_metadata``, which picks the KV
segment per request), its converter (``_glm5next_model_kwargs``, which sizes the window
and the block-table row from that segment) and the root, chunk by chunk of 1024 query
rows at the served KV page of 128 tokens.

1. The standard line (``max_model_len`` 4096, ``kv_segment_size_buckets`` [1024, 2048,
   4096], ``num_batched_tokens_buckets`` [1024]): prompts of 700, 1500, 3000 and 4095
   tokens take the 1024 / 2048 / 4096 / 4096 segment, every chunk's window holds the
   request, and 16 greedy tokens (one for 4095: the prompt and the generated tokens must
   fit in max_model_len) are bit-identical to the single-segment [4096] config. On the
   code before this change every chunk read ``buckets[0]`` = 1024, a 2048-token window,
   and the 3000- and 4095-token prompts stop at "a request longer than its bucket cannot be
   served by this window".
2. The bs=64 line in the tiny model's clothes (``max_num_seqs`` 4, [1024, 8192],
   ``max_model_len`` 8192): requests of 900, 3000 and 7000 tokens admitted one after the
   other while the earlier ones decode, batched decode padded to the bucket of 4. Each
   request's segment is 1024 / 8192 / 8192, and its tokens and its own cache state (latent
   pages, side-cache slot) are those of the same request served alone.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 PYTHONPATH=$PWD python -m pytest \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_prompt_window.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MLAAttentionSpec
from vllm.v1.worker.block_table import MultiGroupBlockTable

from vllm_neuron.vllm.worker import neuron_model_runner
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_decode as dsa
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny

pytestmark = [pytest.mark.forked]

#: The served hybrid_kv_block_size and num_batched_tokens bucket.
PAGE = 128
QUERY = 1024
SEED = 20261007
GENERATED = 16

STANDARD_MAX = 4096
STANDARD_SEGMENTS = [1024, 2048, 4096]
ONE_SEGMENT = [4096]

BS64_MAX = 8192
BS64_SEGMENTS = [1024, 8192]
BS64_SLOTS = 4
BS64_BUCKETS = [1, 2, 4]
BS64_PROMPTS = {"a": 900, "b": 3000, "c": 7000}


def _cdiv(value: int, divisor: int) -> int:
    return -(-int(value) // int(divisor))


def _prompt(tokens: int, seed: int) -> torch.Tensor:
    return torch.randint(
        0, tiny.STACK_VOCAB_SIZE, (tokens,),
        generator=torch.Generator().manual_seed(seed), dtype=torch.int64,
    )


class World:
    """The tiny root, banks for ``slots`` requests of ``max_model_len`` tokens, a runner shell.

    Every step rebuilds the input batch the way the runner's condensed batch looks: the
    step's requests in batch order, each with the pages the scheduler has allocated so
    far (page 0 is vLLM's null block; request ``row`` owns one run of pages after it).
    """

    def __init__(self, segments, *, max_model_len: int, slots: int):
        e2e._require_cpu_mode()
        root = e2e._fixture()["root"]
        head = torch.randn(
            tiny.STACK_VOCAB_SIZE, int(root.text_config.hidden_size),
            generator=torch.Generator().manual_seed(SEED),
        )
        root.lm_head_weight = torch.nn.Parameter(head.to(torch.bfloat16), requires_grad=False)
        self.per_request = _cdiv(max_model_len, PAGE)
        blocks = 1 + slots * self.per_request
        caches = {}
        for spec in root.get_kv_spec().layers:
            assert spec.kda_recurrent_state_shape is None, "this file drives a sparse-only stack"
            shape = (blocks, int(spec.num_kv_heads), PAGE, int(spec.head_size))
            caches[spec.name] = [
                torch.zeros(shape, dtype=spec.dtype) for _ in range(1 if spec.latent_kv else 2)
            ]
        root.bind_kv_cache(caches)
        layer = root.get_kv_spec().layers[0]
        names = [bank["name"] for bank in root.glm5next_layer_banks]
        runner = NeuronModelRunner.__new__(NeuronModelRunner)
        runner.model = root
        runner.max_model_len = int(max_model_len)
        runner.max_num_reqs = int(slots)
        runner.device = torch.device("cpu")
        runner.cp_world_size = 1
        runner._dcp_size = 1
        runner._is_synthetic_model = False
        runner.neuron_config = SimpleNamespace(
            kv_segment_size_buckets=list(segments),
            num_batched_tokens_buckets=[QUERY],
            decode_context_length_buckets=None,
            enable_structured_outputs=False,
        )
        runner.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                KVCacheGroupSpec(
                    names,
                    MLAAttentionSpec(
                        block_size=PAGE,
                        num_kv_heads=int(layer.num_kv_heads),
                        head_size=int(layer.head_size),
                        dtype=layer.dtype,
                        sliding_window=None,
                        attention_chunk_size=None,
                    ),
                )
            ]
        )
        self.root, self.caches, self.runner = root, caches, runner
        self.max_model_len, self.slots = int(max_model_len), int(slots)
        #: (request, cached, segment, window pages) per prefill chunk, in order.
        self.chunks: list[tuple] = []

    def pages(self, row: int, tokens: int) -> list[int]:
        base = 1 + row * self.per_request
        return [base + k for k in range(_cdiv(tokens, PAGE))]

    def _batch(self, requests) -> None:
        """``requests``: (req_id, row, cached, tokens this step, request length) in batch order."""
        table = MultiGroupBlockTable(
            max_num_reqs=self.slots,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=QUERY,
            pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=[PAGE],
            kernel_block_sizes=[PAGE],
        )
        cached = torch.zeros(self.slots, dtype=torch.int32)
        length = np.zeros(self.slots, dtype=np.int32)
        for index, (req_id, row, done, step, total) in enumerate(requests):
            table[0].append_row(self.pages(row, done + step), index)
            cached[index] = done
            length[index] = total
        self.runner.input_batch = SimpleNamespace(
            req_ids=[request[0] for request in requests],
            num_reqs=len(requests),
            block_table=table,
            num_computed_tokens_cpu_tensor=cached,
            num_tokens_no_spec=length,
        )

    def _forward(self, generic: dict) -> tuple[torch.Tensor, dict]:
        kwargs = self.runner._glm5next_model_kwargs(generic)
        return self.runner.model(**kwargs).float(), kwargs

    def prefill(self, req_id: str, row: int, ids: torch.Tensor, *, cached: int, total: int):
        """One prefill chunk of ``req_id``; returns the last row's logits."""
        real = int(ids.shape[0])
        self._batch([(req_id, row, cached, real, total)])
        metadata = self.runner._build_attention_metadata(
            1, QUERY, QUERY, 0, cached, host_only=True
        )
        segments = {int(entry["kv_segment_size"]) for entry in metadata.values()}
        assert len(segments) == 1, segments
        self.runner._glm5next_request_tokens = np.array([real], np.int32)
        logits, kwargs = self._forward(
            {
                "input_ids": torch.cat([ids, ids.new_zeros(QUERY - real)]),
                "positions": torch.arange(QUERY, dtype=torch.long) + cached,
                "attn_metadata": metadata,
                "sampling_positions": torch.tensor([real - 1], dtype=torch.long),
                "sampling_params": None,
                "spec_decode_metadata": None,
                "rank": None,
                "logit_mask": None,
            }
        )
        windows = {
            int(carrier["block_table_row"].shape[0])
            for carrier in kwargs["layer_carriers"] if "block_table_row" in carrier
        }
        assert len(windows) == 1, windows
        self.chunks.append((req_id, cached, segments.pop(), windows.pop()))
        return logits[-1]

    def decode(self, requests, tokens, *, bucket: int) -> torch.Tensor:
        """One decode step for ``requests`` = (req_id, row, cached), padded to ``bucket`` rows."""
        count = len(requests)
        assert count <= bucket
        self._batch([(req_id, row, done, 1, done + 1) for req_id, row, done in requests])
        longest = max(done for _, _, done in requests)
        metadata = self.runner._build_attention_metadata(
            bucket, bucket, 1, 0, longest, max_decode_ctx_len=longest, host_only=True
        )
        self.runner._glm5next_request_tokens = np.array([1] * count, np.int32)
        logits, _ = self._forward(
            {
                "input_ids": torch.tensor(list(tokens) + [0] * (bucket - count), dtype=torch.long),
                "positions": None,
                "attn_metadata": metadata,
                "sampling_positions": torch.arange(count, dtype=torch.long),
                "sampling_params": None,
                "spec_decode_metadata": None,
                "rank": None,
                "logit_mask": None,
            }
        )
        return logits[:count]

    def owned_state(self, req_id: str, row: int, tokens: int) -> dict:
        """The request's own cache state per layer: its latent pages, its slot's side caches."""
        slot = self.runner._glm5next_request_slot_table[req_id]
        pages = self.pages(row, tokens)
        out = {}
        for index, bank in enumerate(self.root.glm5next_layer_banks):
            out[(index, "latent")] = self.caches[bank["name"]][0][pages].clone()
            side = self.runner._glm5next_side_cache_set[index]
            for key in ("pool_cache", "tail"):
                if key in side:
                    out[(index, key)] = side[key][slot].clone()
        return out


@dataclass
class Run:
    tokens: list[int] = field(default_factory=list)
    logits: list[torch.Tensor] = field(default_factory=list)


def _bucket(count: int, buckets) -> int:
    return next(bucket for bucket in buckets if bucket >= count)


def _prefill_all(world: World, req_id: str, row: int, prompt: torch.Tensor) -> Run:
    """Every chunk of ``prompt`` in order; the run holds the first generated token."""
    total = int(prompt.shape[0])
    run = Run()
    for cached in range(0, total, QUERY):
        logits = world.prefill(req_id, row, prompt[cached:cached + QUERY], cached=cached, total=total)
    run.logits.append(logits)
    run.tokens.append(int(logits.argmax()))
    return run


def _decode_steps(world: World, live: dict, steps: int, *, buckets) -> None:
    """``steps`` batched decode steps of every request in ``live``: {req_id: (row, prompt, run)}."""
    order = list(live)
    for _ in range(steps):
        requests = [(req_id, live[req_id][0], live[req_id][1] + len(live[req_id][2].tokens) - 1)
                    for req_id in order]
        fed = [live[req_id][2].tokens[-1] for req_id in order]
        logits = world.decode(requests, fed, bucket=_bucket(len(order), buckets))
        for index, req_id in enumerate(order):
            run = live[req_id][2]
            run.logits.append(logits[index])
            run.tokens.append(int(logits[index].argmax()))


def _generate(world: World, req_id: str, row: int, prompt: torch.Tensor, generated: int) -> Run:
    run = _prefill_all(world, req_id, row, prompt)
    _decode_steps(world, {req_id: (row, int(prompt.shape[0]), run)}, generated - 1, buckets=[1])
    return run


def _window_pages(segment: int, max_model_len: int) -> int:
    return min(max_model_len // PAGE, _cdiv(segment + QUERY, PAGE))


# ---------------------------------------------------------------------------
# The standard line.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt_len, segment", [(700, 1024), (1500, 2048), (3000, 4096), (4095, 4096)]
)
def test_a_prompt_on_the_standard_line_takes_its_segment_and_matches_one_segment(
    prompt_len, segment, caplog
):
    """[1024, 2048, 4096] against [4096]: the same tokens and logits, from a smaller window."""
    generated = min(GENERATED, STANDARD_MAX - prompt_len)
    prompt = _prompt(prompt_len, SEED + prompt_len)
    chunks = _cdiv(prompt_len, QUERY)

    multi = World(STANDARD_SEGMENTS, max_model_len=STANDARD_MAX, slots=1)
    with caplog.at_level(logging.INFO, logger=neuron_model_runner.__name__):
        got = _generate(multi, "req-multi", 0, prompt, generated)
    one = World(ONE_SEGMENT, max_model_len=STANDARD_MAX, slots=1)
    want = _generate(one, "req-one", 0, prompt, generated)

    # The segment the request took, logged once per chunk, and its window.
    assert [chunk[2] for chunk in multi.chunks] == [segment] * chunks, multi.chunks
    assert [chunk[3] for chunk in multi.chunks] == [_window_pages(segment, STANDARD_MAX)] * chunks
    assert [chunk[2] for chunk in one.chunks] == [ONE_SEGMENT[0]] * chunks
    assert [chunk[3] for chunk in one.chunks] == [STANDARD_MAX // PAGE] * chunks
    # (caplog keeps the second world's lines too: the runner logger is at INFO already.)
    logged = [
        record.getMessage() for record in caplog.records
        if "KV segment" in record.getMessage() and "req-multi" in record.getMessage()
    ]
    assert len(logged) == chunks, caplog.text
    assert all(f"KV segment {segment} of {STANDARD_SEGMENTS}" in line for line in logged), logged
    assert [int(line.split(": ")[1].split(" ")[0]) for line in logged] == list(range(0, prompt_len, QUERY))
    # The last chunk's pages fit the window (the converter refuses otherwise).
    assert _cdiv(prompt_len, PAGE) <= _window_pages(segment, STANDARD_MAX)

    assert len(got.tokens) == generated == len(want.tokens)
    assert got.tokens == want.tokens, (got.tokens, want.tokens)
    for step, (left, right) in enumerate(zip(got.logits, want.logits)):
        assert torch.isfinite(left).all(), step
        assert torch.equal(left, right), f"step {step}: logits differ"
    # The comparison discriminates: the head's top-2 gap is wide at every step.
    gaps = [float(torch.topk(row, 2).values.diff().abs()) for row in want.logits]
    assert min(gaps) > 0.0, gaps


# ---------------------------------------------------------------------------
# The bs=64 line in the tiny model's clothes: mixed lengths, concurrently.
# ---------------------------------------------------------------------------


def _reference(req_id: str, prompt: torch.Tensor, generated: int) -> tuple[Run, dict, World]:
    """The request served alone on a fresh world of the same line."""
    world = World(BS64_SEGMENTS, max_model_len=BS64_MAX, slots=BS64_SLOTS)
    run = _prefill_all(world, req_id, 0, prompt)
    _decode_steps(world, {req_id: (0, int(prompt.shape[0]), run)}, generated - 1, buckets=BS64_BUCKETS)
    state = world.owned_state(req_id, 0, int(prompt.shape[0]) + generated - 1)
    return run, state, world


def test_mixed_length_requests_on_the_bs64_line_decode_together_as_each_does_alone():
    """900, 3000 and 7000 tokens admitted one after the other: segments 1024 / 8192 / 8192,
    and each request's tokens and own cache state are those of its single-request run."""
    prompts = {name: _prompt(tokens, SEED + tokens) for name, tokens in BS64_PROMPTS.items()}
    world = World(BS64_SEGMENTS, max_model_len=BS64_MAX, slots=BS64_SLOTS)
    live: dict = {}

    # a: prefill, then two decode steps alone.
    live["a"] = (0, BS64_PROMPTS["a"], _prefill_all(world, "a", 0, prompts["a"]))
    _decode_steps(world, live, 2, buckets=BS64_BUCKETS)
    # b admitted: three prefill chunks while a waits, then two steps of [a, b].
    live["b"] = (1, BS64_PROMPTS["b"], _prefill_all(world, "b", 1, prompts["b"]))
    _decode_steps(world, live, 2, buckets=BS64_BUCKETS)
    # c admitted: seven chunks, then four steps of [a, b, c] padded to the bucket of 4.
    live["c"] = (2, BS64_PROMPTS["c"], _prefill_all(world, "c", 2, prompts["c"]))
    _decode_steps(world, live, 4, buckets=BS64_BUCKETS)

    segments = {name: {chunk[2] for chunk in world.chunks if chunk[0] == name} for name in live}
    assert segments == {"a": {1024}, "b": {8192}, "c": {8192}}, world.chunks
    windows = {name: {chunk[3] for chunk in world.chunks if chunk[0] == name} for name in live}
    assert windows == {"a": {16}, "b": {64}, "c": {64}}, world.chunks
    assert [chunk[1] for chunk in world.chunks if chunk[0] == "c"] == list(range(0, 7000, QUERY))
    slots = {name: world.runner._glm5next_request_slot_table[name] for name in live}
    assert sorted(slots.values()) == [0, 1, 2], slots

    for name, (row, length, run) in live.items():
        want, want_state, _ = _reference(name, prompts[name], len(run.tokens))
        assert run.tokens == want.tokens, (name, run.tokens, want.tokens)
        for step, (left, right) in enumerate(zip(run.logits, want.logits)):
            top = torch.topk(right, 2).values
            gap = float(top[0] - top[1])
            tolerance = dsa.LOGIT_RTOL * float(right.abs().max())
            torch.testing.assert_close(left, right, rtol=0.0, atol=tolerance,
                                       msg=lambda m: f"{name} step {step}: {m}")
            assert gap > 2 * tolerance, (name, step, gap, tolerance)
        got_state = world.owned_state(name, row, length + len(run.tokens) - 1)
        dsa._assert_state_equal(got_state, want_state, f"request {name}", exact_layers=1)
