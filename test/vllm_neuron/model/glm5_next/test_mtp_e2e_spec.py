# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding (method "mtp") on the GLM-5.3-Flash root: the verify leg.

Part 1, the fused verify leg of ``Glm5NextForConditionalGeneration.forward``: a
decode step of ``T = 1 + k`` rows per request with ``spec_decode_metadata`` samples
every row, runs the on-device greedy rejection sampler
(``vllm_neuron/nn/rejection_sampler.py``), populates the draft layer at every verify
row with the token the trunk sampled there, and drafts the next ``k`` tokens from the
last accepted row with a one-row carrier derived from the step's. The stack is
replaced by a fake that returns hidden rows the head maps to chosen tokens, and the
head by a recorder, so these tests pin WHICH rows, ids, positions and carrier
operands reach the two head entry points and WHAT the root returns, for the three
acceptance patterns (all drafts rejected, a partial accept, all accepted) and for a
two-request batch. The kernels that consume ``T`` rows are other workers' and are not
run here.

Part 2, the greedy identity through the real runner: ``NeuronModelRunner`` built from a
``method: mtp`` engine config and driven step by step the way the scheduler drives it
(prefill, then verify steps whose drafts are the runner's own proposals), against the
plain runner's greedy run of the same tiny root. Every runner path is real -- the spec
config patch, the proposer, the translator's ``T``-row carriers and the metadata it
hands the root, the root's rejection sampler and its one-row draft carrier, the
sampler's parse, the per-step state hook (the ring cursor pulled back to the kept
rows), the proposal (placeholders after a prefill, the root's drafts after a decode,
nothing near ``max_model_len``) -- while the trunk's decode rows and the head's drafts
are oracles read off the plain run: the kernels that consume ``T`` rows per request
(DSA indexer ring, MLA decode, KDA checkpoints) are other workers' and this tree has
no ``T``-row path for them yet, so the identity here is of the bookkeeping, and is
restated on the real kernels once they land. The prefill runs the real stack. Each
prompt's drafts follow a policy (every draft accepted, every draft rejected, a fixed
or cycling prefix accepted), so the all-accepted and all-rejected cases are covered by
construction; the plain run's greedy ties do not matter here, the oracle emits its
ids. Eight prompts, 64 tokens, ``k`` in {1, 3}.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

from vllm_neuron.model.glm5_next import mtp as head_module
from vllm_neuron.model.neuron_config import OnDeviceSamplingConfig
from vllm_neuron.nn.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next import test_shadow_draft_e2e as shadow
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_decode as batch
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_first_request as fr
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny

pytestmark = [pytest.mark.forked]

K = 3
T = 1 + K
KNOB = head_module.SHADOW_DRAFT_ENV


# ── part 1: the fused verify leg, stack faked, head recorded ─────────────────


def _metadata_for(drafts: list[list[int]]) -> SpecDecodeMetadata:
    """The runner's spec metadata for ``B`` requests of ``k`` drafts each, rows request-major."""
    counts = [len(row) for row in drafts]
    cu = np.cumsum(counts)
    cu_sampled = np.cumsum([count + 1 for count in counts])
    target = [b * (counts[b] + 1) + j for b in range(len(drafts)) for j in range(counts[b])]
    return SpecDecodeMetadata(
        draft_token_ids=torch.tensor([d for row in drafts for d in row], dtype=torch.int32),
        num_draft_tokens=counts,
        cu_num_draft_tokens=torch.from_numpy(cu.astype(np.int64)),
        cu_num_sampled_tokens=torch.from_numpy(cu_sampled.astype(np.int64)),
        target_logits_indices=torch.tensor(target, dtype=torch.int64),
        bonus_logits_indices=torch.from_numpy(cu_sampled.astype(np.int64) - 1),
        logits_indices=torch.arange(int(cu_sampled[-1]), dtype=torch.int64),
    )


class _HeadRecorder:
    """Records the head's two entry points and returns chosen drafts without running it."""

    def __init__(self, drafts: list[list[int]]):
        self.drafts = drafts
        self.calls: list[tuple[str, dict]] = []

    def populate(self, hidden_rows, next_ids, positions, **kwargs):
        self.calls.append(("populate", {
            "hidden_rows": hidden_rows.clone(), "next_ids": next_ids.clone(),
            "positions": positions.clone(), "kwargs": dict(kwargs),
        }))

    def draft_tokens(self, hidden_rows, sampled_ids, positions, k, **kwargs):
        self.calls.append(("draft", {
            "hidden_rows": hidden_rows.clone(), "sampled_ids": sampled_ids.clone(),
            "positions": positions.clone(), "k": int(k), "kwargs": dict(kwargs),
        }))
        return torch.tensor(self.drafts, dtype=torch.int32)

    def of(self, kind: str) -> list[dict]:
        return [call for name, call in self.calls if name == kind]


def _fake_stack(head: torch.Tensor, targets: list[int]):
    """A stack whose row ``r`` is the head's row for ``targets[r]``, so greedy sampling of
    row ``r`` yields ``targets[r]`` (a random head's rows are near-orthogonal)."""
    rows = head[torch.tensor(targets)].clone()

    def forward(input_ids, **kwargs):
        assert int(input_ids.shape[0]) == len(targets), (input_ids.shape, len(targets))
        return rows

    return forward


def _verify_world(monkeypatch):
    monkeypatch.setenv(KNOB, str(K))
    world = shadow._world()
    world.runner.is_mtp_spec = True
    world.runner.drafter = SimpleNamespace(num_speculative_tokens=K)
    # The opening prefill, converted and run: it sets the ring cursor the verify step continues.
    shadow._step(world, shadow.PROMPT, cached=0, sampling=[len(shadow.PROMPT) - 1])
    return world


def _verify_step(world, *, inputs: list[int], drafts: list[list[int]], targets: list[int], recorder):
    """Convert a verify step of ``inputs`` (request-major, ``T`` per request) and run the root."""
    runner = world.runner
    requests = len(drafts)
    start = len(shadow.PROMPT)
    runner.input_batch.req_ids = list(world.req_ids)
    runner._glm5next_request_tokens = np.array([T] * requests, np.int32)
    metadata = batch._metadata(world, [0] * requests, cached=[start] * requests, tokens=len(inputs))
    for entry in metadata.values():
        entry["max_query_len"] = T
        entry["decode_token_threshold"] = T
    converted = runner._glm5next_model_kwargs({
        "input_ids": torch.tensor(inputs, dtype=torch.int32),
        "positions": None,
        "attn_metadata": metadata,
        "sampling_positions": torch.arange(len(inputs), dtype=torch.long),
        "sampling_params": torch.tensor([shadow.GREEDY_ROW] * len(inputs), dtype=torch.float32),
        "spec_decode_metadata": _metadata_for(drafts),
        "rank": None,
        "logit_mask": None,
    })
    assert converted["draft_k"] == K
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(world.root.model, "forward", _fake_stack(world.head, targets))
        patch.setattr(world.root.mtp, "populate", recorder.populate)
        patch.setattr(world.root.mtp, "draft_tokens", recorder.draft_tokens)
        out = world.root.forward(**converted)
    return out, converted


def _assert_populate_covers_every_row(recorder, *, carriers, targets, start):
    populate = recorder.of("populate")
    assert len(populate) == 1, [name for name, _ in recorder.calls]
    call = populate[0]
    rows = len(targets)
    assert tuple(call["hidden_rows"].shape) == (rows, call["hidden_rows"].shape[1])
    # Row t pairs with the token the trunk sampled there: the draft (== target) on an
    # accepted row, the corrected token on the mismatch row, garbage past it (rewritten
    # before it is read).
    assert call["next_ids"].tolist() == targets
    assert call["next_ids"].dtype == torch.int32
    assert call["positions"].tolist() == [start + t for t in range(T)] * (rows // T)
    assert call["positions"].dtype == torch.int32
    # The step's own T-row carrier, object for object.
    for key in ("tail", "pool_cache", "latent_cache", "block_table_row", "seq_lens", "latent_slots"):
        assert call["kwargs"][key] is carriers[-1][key], key


def _assert_draft_from_row(recorder, *, carriers, row: int, token: int, position: int):
    draft = recorder.of("draft")
    assert len(draft) == 1
    call = draft[0]
    assert call["k"] == K
    assert call["sampled_ids"].tolist() == [token] and call["sampled_ids"].dtype == torch.int32
    assert call["positions"].tolist() == [position] and call["positions"].dtype == torch.int32
    torch.testing.assert_close(
        call["hidden_rows"][0], recorder.of("populate")[0]["hidden_rows"][row], rtol=0, atol=0
    )
    kwargs = call["kwargs"]
    assert "tail" in kwargs and "position" in kwargs and "prefill_tail" not in kwargs
    assert int(kwargs["position"]) == position and int(kwargs["start_position"]) == position
    assert kwargs["seq_lens"].tolist() == [position + 1]
    assert kwargs["latent_slots"].tolist() == [int(carriers[-1]["latent_slots"][row])]
    for key in ("tail", "pool_cache", "latent_cache", "block_table_row"):
        assert kwargs[key] is carriers[-1][key], key


@pytest.mark.parametrize(
    "accept",
    [0, 2, K],
    ids=["all-rejected", "two-of-three", "all-accepted"],
)
def test_the_verify_leg_accepts_then_drafts_from_the_last_accepted_row(monkeypatch, accept):
    world = _verify_world(monkeypatch)
    vocab = shadow.tiny.STACK_VOCAB_SIZE
    start = len(shadow.PROMPT)
    last = shadow.PROMPT[-1]
    drafts = [[11, 23, 37]]
    # Targets: agree with the drafts on the first ``accept`` rows, differ on the next, and
    # a bonus on the last row (used only when every draft is accepted).
    targets = [drafts[0][j] if j < accept else (drafts[0][j] + 1) % vocab for j in range(K)]
    targets.append(97)
    recorder = _HeadRecorder([[5, 6, 7]])
    out, converted = _verify_step(
        world, inputs=[last] + drafts[0], drafts=drafts, targets=targets, recorder=recorder,
    )
    assert isinstance(out, tuple) and len(out) == 2
    accepted, draft_ids = out
    expected = targets[: accept + 1] + [PLACEHOLDER_TOKEN_ID] * (K - accept)
    assert accepted.dtype == torch.int32 and accepted.tolist() == [expected]
    assert draft_ids.tolist() == [[5, 6, 7]]
    carriers = converted["layer_carriers"]
    _assert_populate_covers_every_row(recorder, carriers=carriers, targets=targets, start=start)
    _assert_draft_from_row(
        recorder, carriers=carriers, row=accept, token=targets[accept], position=start + accept
    )


def test_a_t_row_decode_without_spec_metadata_is_refused_by_name(monkeypatch):
    world = _verify_world(monkeypatch)
    runner = world.runner
    start = len(shadow.PROMPT)
    runner._glm5next_request_tokens = np.array([T], np.int32)
    metadata = batch._metadata(world, [0], cached=[start], tokens=T)
    for entry in metadata.values():
        entry["max_query_len"] = T
        entry["decode_token_threshold"] = T
    converted = runner._glm5next_model_kwargs({
        "input_ids": torch.tensor([shadow.PROMPT[-1], 1, 2, 3], dtype=torch.int32),
        "positions": None,
        "attn_metadata": metadata,
        "sampling_positions": torch.arange(T, dtype=torch.long),
        "sampling_params": torch.tensor([shadow.GREEDY_ROW] * T, dtype=torch.float32),
        "spec_decode_metadata": None,
        "rank": None,
        "logit_mask": None,
    })
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(world.root.model, "forward", _fake_stack(world.head, [1, 2, 3, 4]))
        with pytest.raises(ValueError, match="spec_decode_metadata"):
            world.root.forward(**converted)


# ── part 2: the greedy identity through the real runner ──────────────────────

VOCAB = tiny.STACK_VOCAB_SIZE
PAGE = tiny.MLA_PAGE_SIZE
#: The batch-decode regime the indexer's top-k wants (see the translator tests).
MAX_MODEL_LEN = tiny.STACK_TOKENS + 8
#: The prefill bucket every prompt below is padded to; the last bucket is the engine's cap.
PREFILL_BUCKET = 64
#: Tokens generated per prompt; the longest prompt plus these plus a verify step fits.
GENERATED = 64
PROMPT_LENGTHS = (24, 29, 33, 37, 40, 44, 27, 31)
SEED_PROMPTS = 20261008
FIRST_BLOCK = 1
#: The tiny fixture's config names the served model's pad id, past this vocabulary; the
#: placeholder drafts a prefill proposes must be ids the runner's validation admits.
PAD_TOKEN_ID = 0
assert max(PROMPT_LENGTHS) + GENERATED + T + 1 <= MAX_MODEL_LEN
assert max(PROMPT_LENGTHS) <= PREFILL_BUCKET


def _accept_all(_call: int, k: int) -> int:
    return k


def _reject_all(_call: int, _k: int) -> int:
    return 0


def _accept_prefix(length: int):
    return lambda _call, k: min(length, k)


def _cycling(call: int, k: int) -> int:
    return call % (k + 1)


#: One draft policy per prompt: how many leading drafts of each proposal match the plain run.
POLICIES = (
    ("accept-all", _accept_all),
    ("reject-all", _reject_all),
    ("accept-1", _accept_prefix(1)),
    ("accept-2", _accept_prefix(2)),
    ("cycling", _cycling),
    ("accept-all-2", _accept_all),
    ("reject-all-2", _reject_all),
    ("cycling-2", _cycling),
)
assert len(POLICIES) == len(PROMPT_LENGTHS)


def _prompts() -> list[list[int]]:
    return [
        torch.randint(0, VOCAB, (length,), generator=torch.Generator().manual_seed(SEED_PROMPTS + i),
                      dtype=torch.int64).tolist()
        for i, length in enumerate(PROMPT_LENGTHS)
    ]


def _config(k: int | None):
    """The engine config of a greedy on-device-sampling serve of the tiny root; ``k`` None = plain."""
    neuron_config = {
        "num_batched_tokens_buckets": [PREFILL_BUCKET, MAX_MODEL_LEN],
        "num_seqs_buckets": [1],
        "on_device_sampling_config": {},
    }
    return EngineArgs(
        model=str(fr.FIXTURE),
        skip_tokenizer_init=True,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
        max_num_batched_tokens=MAX_MODEL_LEN,
        block_size=PAGE,
        enforce_eager=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        speculative_config=None if k is None else {"method": "mtp", "num_speculative_tokens": k},
        additional_config={"neuron_config": neuron_config},
    ).create_engine_config()


def _root():
    """The tiny root the shadow file serves: seeded zero-mean head, greedy device sampling, the
    draft head materialised when the knob built it."""
    root = e2e._fixture()["root"]
    head = torch.randn(VOCAB, int(root.text_config.hidden_size),
                       generator=torch.Generator().manual_seed(shadow.SEED))
    head = (head - head.mean(dim=1, keepdim=True)).to(torch.bfloat16)
    root.lm_head_weight = torch.nn.Parameter(head, requires_grad=False)
    root.text_config.neuron_config = SimpleNamespace(
        on_device_sampling_config=OnDeviceSamplingConfig(all_greedy=True)
    )
    if root.mtp is not None:
        shadow._materialise_head(root, shadow.SEED_HEAD)
    return root, head


def _runner(config, root) -> NeuronModelRunner:
    """The real runner on the root, the model bound in place of ``load_model``; no warmup: the
    spec warmup would run the verify step on the real stack, which has no ``T``-row path here."""
    runner = NeuronModelRunner(config, device=torch.device("cpu"))
    runner.model = root
    runner.vocab_size = VOCAB
    if runner.drafter is not None:
        runner.drafter.load_model(root)
    runner.vllm_config.model_config.hf_config.pad_token_id = PAD_TOKEN_ID
    runner.initialize_kv_cache(fr._kv_cache_config(runner, num_blocks=e2e._blocks_for(MAX_MODEL_LEN) + 1))
    return runner


def _prefill(req: str, prompt: list[int], groups: int, blocks: list[int], finished: set) -> SchedulerOutput:
    new = NewRequestData(
        req_id=req, prompt_token_ids=list(prompt), mm_features=[],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=GENERATED), pooling_params=None,
        block_ids=tuple(list(blocks) for _ in range(groups)), num_computed_tokens=0, lora_request=None,
    )
    step = SchedulerOutput(
        scheduled_new_reqs=[new], scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req: len(prompt)}, total_num_scheduled_tokens=len(prompt),
        scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={}, num_common_prefix_blocks=[0],
        finished_req_ids=set(finished), free_encoder_mm_hashes=[],
    )
    step.num_scheduled_tokens_padded = {req: PREFILL_BUCKET}
    return step


def _decode(req: str, position: int, generated: int, groups: int, drafts: list[int],
            new_blocks: list[int]) -> SchedulerOutput:
    """The request's next token at ``position`` plus its drafts, as the scheduler schedules them."""
    rows = 1 + len(drafts)
    cached = CachedRequestData(
        req_ids=[req], resumed_req_ids=set(), new_token_ids=[], all_token_ids={},
        new_block_ids=[tuple(list(new_blocks) for _ in range(groups)) if new_blocks else None],
        num_computed_tokens=[position], num_output_tokens=[generated],
    )
    step = SchedulerOutput(
        scheduled_new_reqs=[], scheduled_cached_reqs=cached,
        num_scheduled_tokens={req: rows}, total_num_scheduled_tokens=rows,
        scheduled_spec_decode_tokens={req: list(drafts)} if drafts else {},
        scheduled_encoder_inputs={}, num_common_prefix_blocks=[0], finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    step.num_scheduled_tokens_padded = {req: rows}
    return step


class _Oracle:
    """The trunk's decode rows and the head's drafts, read off the plain run's sequence.

    ``full`` is the prompt followed by the plain run's greedy tokens. On a decode step
    at ``position`` the trunk's row ``t`` predicts ``full[position + t + 1]`` as long as
    the rows before it carry the plain run's ids; from the first row that does not, the
    rows carry an id one off the true one, so a sampler that read them would show. The
    head's drafts from a row at ``last`` are ``full[last + 2 : last + 2 + k]`` with the
    policy's number of leading matches kept and the rest one off.
    """

    def __init__(self, head: torch.Tensor, full: list[int], policy):
        self.head, self.full, self.policy = head, full, policy
        self.position: int | None = None
        self.active = False
        self.calls = 0

    def stack(self, real):
        def forward(input_ids, **kwargs):
            if not self.active:
                return real(input_ids, **kwargs)
            ids = [int(value) for value in input_ids.reshape(-1).tolist()]
            targets, agree = [], True
            for t, token in enumerate(ids):
                agree = agree and token == self.full[self.position + t]
                true = self.full[self.position + t + 1]
                targets.append(true if agree else (true + 1) % VOCAB)
            return self.head[torch.tensor(targets)].clone()
        return forward

    def draft_tokens(self, hidden_rows, sampled_ids, positions, k, **kwargs):
        assert int(positions.numel()) == 1, "one request per step in this regime"
        start = int(positions.reshape(-1)[0]) + 2
        true = self.full[start:start + int(k)]
        match = self.policy(self.calls, int(k))
        self.calls += 1
        return torch.tensor(
            [[true[j] if j < match else (true[j] + 1) % VOCAB for j in range(int(k))]], dtype=torch.int32,
        )


def _proposal(runner) -> list[int]:
    proposed = runner.take_draft_token_ids()
    return [] if proposed is None else [int(value) for value in proposed.draft_token_ids[0]]


def _generate(runner, req: str, prompt: list[int], *, tokens: int, finished: set, oracle=None):
    """Prefill, then decode until ``tokens`` are generated; returns (ids, decode steps)."""
    groups = fr._groups(runner)
    blocks = list(range(FIRST_BLOCK, FIRST_BLOCK + -(-len(prompt) // PAGE)))
    _, out = fr._step(runner, _prefill(req, prompt, groups, blocks, finished))
    generated = [int(value) for value in out.sampled_token_ids[0]]
    assert len(generated) == 1
    position = len(prompt)
    drafts = _proposal(runner)
    steps = 0
    while len(generated) < tokens:
        rows = 1 + len(drafts)
        needed = -(-(position + rows) // PAGE)
        new = list(range(FIRST_BLOCK + len(blocks), FIRST_BLOCK + needed))
        blocks += new
        if oracle is not None:
            oracle.position = position
            oracle.active = True
        try:
            _, out = fr._step(runner, _decode(req, position, len(generated), groups, drafts, new))
        finally:
            if oracle is not None:
                oracle.active = False
        accepted = [int(value) for value in out.sampled_token_ids[0]]
        assert 1 <= len(accepted) <= rows, (accepted, rows)
        generated += accepted
        position += len(accepted)
        steps += 1
        # The hook pulled the request's ring cursor back to the rows it kept.
        slot = runner._glm5next_request_slot_table[req]
        assert runner._glm5next_side_cache_positions[slot] == position
        drafts = _proposal(runner)
        if oracle is None:
            assert drafts == [], "the plain run proposes nothing"
    return generated[:tokens], steps


def test_greedy_speculative_output_is_token_identical_to_the_plain_run(tmp_path, monkeypatch):
    e2e._require_cpu_mode()
    fr._declaring_a_sampler(monkeypatch)
    prompts = _prompts()
    # The plain run, far enough past ``GENERATED`` for every oracle row the verify steps read.
    monkeypatch.delenv(KNOB, raising=False)
    plain_config = _config(None)
    references: list[list[int]] = []
    with fr._parallel_state(tmp_path / "plain", plain_config):
        runner = _runner(plain_config, _root()[0])
        assert runner.drafter is None
        finished: set = set()
        for i, prompt in enumerate(prompts):
            ids, steps = _generate(runner, f"plain-{i}", prompt, tokens=GENERATED + 2 * T, finished=finished)
            assert steps == GENERATED + 2 * T - 1
            references.append(ids)
            finished = {f"plain-{i}"}
    # The deeper ring of a speculative server comes from worker-58's helper, which this
    # tree does not carry; the oracle never reads the ring, so the plain depth serves.
    monkeypatch.setattr(
        NeuronModelRunner, "_glm5next_indexer_ring_rows", staticmethod(lambda pool, k: int(pool))
    )
    report = []
    for k in (1, 3):
        monkeypatch.setenv(KNOB, str(k))
        config = _config(k)
        with fr._parallel_state(tmp_path / f"mtp-{k}", config):
            root, head = _root()
            runner = _runner(config, root)
            assert runner.is_mtp_spec and runner.drafter.num_speculative_tokens == k
            finished = set()
            for i, (prompt, (name, policy)) in enumerate(zip(prompts, POLICIES)):
                oracle = _Oracle(head, prompt + references[i], policy)
                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(root.model, "forward", oracle.stack(root.model.forward))
                    patch.setattr(root.mtp, "draft_tokens", oracle.draft_tokens)
                    ids, steps = _generate(runner, f"mtp-{k}-{i}", prompt, tokens=GENERATED,
                                           finished=finished, oracle=oracle)
                finished = {f"mtp-{k}-{i}"}
                assert ids == references[i][:GENERATED], (k, name, i)
                report.append((k, name, steps))
                if name.startswith("accept-all"):
                    # The first verify step carries a prefill's placeholders; every one after
                    # keeps all k drafts and the bonus.
                    assert steps <= 2 + -(-(GENERATED - 2) // (k + 1)), (k, name, steps)
                if name.startswith("reject-all"):
                    assert steps == GENERATED - 1, (k, name, steps)
    assert len(report) == 2 * len(POLICIES)
