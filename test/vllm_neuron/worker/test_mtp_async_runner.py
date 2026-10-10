# SPDX-License-Identifier: Apache-2.0
"""The runner under the async drafter (method "mtp", ``VLLM_NEURON_GLM5NEXT_MTP_ASYNC``).

Three host corrections of the synchronous drafter move on device, one step late: the state hook no longer pulls the indexer-ring cursor back,
commits the recurrent checkpoints or records the resume row from host ints -- it takes the
step's output on device (``functional/mtp/async_step.mtp_async_take``) into a carry of
device tensors and reads nothing back; the draft proposal hands the scheduler the carry's
``[rows, 1 + k]`` next input ids as a device tensor (the generic async swap feeds them to
the next step) and the drafts-only tensor for the rejection sampler; the translator
corrects the scheduler's optimistic positions on device (``mtp_async_correct``) and lays
every position-derived operand out at the true positions, with the carried resume rows.
Covered here on bare runners (the hook and the proposal) and on the tiny worlds of the
translator tests (the sparse one-request step, the one-ring one-row step, the recurrent
two-request step with a padding row); every expectation is derived from the worlds' own
tables and the host formulas the synchronous translator uses.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_mtp_async_runner.py
"""

from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.functional.kda import fused_decode
from vllm_neuron.functional.mtp import async_step
from vllm_neuron.model.glm5_next import mtp as head_module
from vllm_neuron.vllm.spec_decode.mtp import MtpProposer
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next import test_shadow_draft_e2e as shadow
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_kda as kda
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_first_request as fr
from test.vllm_neuron.worker import test_mtp_proposer as proposer_tests
from test.vllm_neuron.worker import test_mtp_translator as translator

pytestmark = [pytest.mark.forked]

K = 3
T = 1 + K
PAD = 7
MAX_MODEL_LEN = 64
REQ = "req-a"
SLOT = 4
START = 37


@pytest.fixture(scope="module")
def proposer():
    """The real async drafter (knob on, ``--async-scheduling``, one sequence): the runner keys
    its async paths on the proposer's ``async_steps``, so no fake stands in for it."""
    e2e._require_cpu_mode()
    with pytest.MonkeyPatch.context() as patch:
        fr._declaring_a_sampler(patch)
        patch.setenv(proposer_tests.ASYNC_KNOB, "1")
        config = proposer_tests._engine_config(
            proposer_tests._mtp(K), async_scheduling=True, on_device_sampling=True
        )
        built = MtpProposer(config, torch.device("cpu"), True)
    assert built.async_steps is True and built.num_speculative_tokens == K
    return built


# ── a sampler output nobody may read ────────────────────────────────────────


class _Unreadable(torch.Tensor):
    """A device future as the hook sees it: a host read is the hang the async drafter exists
    to avoid, so every read of THIS tensor raises. Tensor operations return plain tensors
    (``__torch_function__`` is disabled), as a NEFF output fed to the next launch does, so
    the watch is on the hook's and the take's direct reads of the sampler's output -- a read
    of a derived tensor is not caught here; the two-rank test's ``waiters`` count is."""

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def of(tensor: torch.Tensor) -> "_Unreadable":
        return tensor.as_subclass(_Unreadable)

    def _refuse(self, *args, **kwargs):
        raise AssertionError("the async drafter read a device future back on the host")

    cpu = tolist = item = numpy = __bool__ = __iter__ = __int__ = __float__ = __index__ = _refuse

    def to(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        if target is not None and str(target).startswith("cpu"):
            self._refuse()
        return torch.Tensor.to(self.as_subclass(torch.Tensor), *args, **kwargs)


def _bind(runner, *names) -> None:
    for name in names:
        setattr(runner, name, functools.partial(getattr(NeuronModelRunner, name), runner))


#: A recurrent bank record as the hook's synchronous branch reads it: with one present that
#: branch commits checkpoints (``commit_kda_checkpoints``), which ``no_commit`` refuses, so
#: the async branch is proven to return before it.
_RECURRENT_BANK = {
    "family": "recurrent", "conv_state": torch.zeros((1, 1, 1, 1)), "recurrent_state": torch.zeros((1, 1, 1, 1)),
}


def _async_runner(proposer, *, record, drafts, req_ids=(REQ,), banks=None):
    """A runner holding what the async hook and proposal read and write."""
    if banks is None:
        banks = [_RECURRENT_BANK]
    runner = SimpleNamespace(
        is_mtp_spec=True,
        drafter=proposer,
        use_async_scheduling=True,
        async_execution_buffer={},
        input_batch=SimpleNamespace(num_reqs=len(req_ids), req_ids=list(req_ids)),
        max_model_len=MAX_MODEL_LEN,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(pad_token_id=PAD))
        ),
        model=SimpleNamespace(glm5next_layer_banks=banks),
        _glm5next_step_record=record,
        _glm5next_side_cache_positions={SLOT: START + T},
        _glm5next_checkpoint_rows={SLOT: 2},
        _glm5next_shadow_last_drafts=drafts,
        _glm5next_async_carry=None,
        _glm5next_async_launch_instance=None,
        _futures_drafts_only=None,
        device=torch.device("cpu"),
    )
    _bind(runner, "_update_states_after_model_execute", "_glm5next_async_drafter",
          "_glm5next_async_settle", "_glm5next_async_launch", "_glm5next_propose_drafts",
          "_glm5next_propose_drafts_async", "_spec_decode_limit", "_placeholder_drafts")
    runner._get_partial_prefill_req_ids = lambda scheduler_output, req_ids: set()
    return runner


def _record(*, width, is_prefill=False, start=START, slot=SLOT):
    return {"slots": [slot], "starts": [start], "counts": [width if width else 9],
            "is_prefill": is_prefill, "width": width}


@pytest.fixture
def no_commit(monkeypatch):
    """The synchronous drafter's checkpoint commit must not run under the async one: a
    device tensor would be read (``fused_decode.commit_kda_checkpoints`` validates values)."""
    def refused(*args, **kwargs):
        raise AssertionError("commit_kda_checkpoints ran under the async drafter")
    monkeypatch.setattr(fused_decode, "commit_kda_checkpoints", refused)


# ── the hook: settle on device ───────────────────────────────────────────────


def test_a_verify_step_settles_on_device_and_reads_nothing_back(no_commit, proposer):
    """The three host corrections moved on device: the cursor stays where the translator put
    it (the device carries the true position), nothing is committed, the resume row is carried
    as a tensor."""
    drafts = torch.arange(2 * K, dtype=torch.int32).reshape(2, K) + 100
    runner = _async_runner(proposer, record=_record(width=T), drafts=drafts)
    sampled = _Unreadable.of(torch.tensor([[5, 6, 7, -1], [9, -1, -1, -1]], dtype=torch.int32))
    runner._update_states_after_model_execute(sampled, None)
    assert runner._glm5next_side_cache_positions == {SLOT: START + T}, "no host pull-back"
    assert runner._glm5next_checkpoint_rows == {SLOT: 2}, "no host resume row"
    carry = runner._glm5next_async_carry
    assert isinstance(carry, async_step.StepCarry)
    assert carry.req_ids == (REQ,) and carry.prev_width == T
    assert carry.checkpoint_rows.tolist() == [2, 0] and carry.checkpoint_rows.dtype == torch.int32
    assert carry.last_accepted.tolist() == [7, 9]
    assert carry.next_input_ids.tolist() == [[7, 100, 101, 102], [9, 103, 104, 105]]
    assert carry.drafts is drafts
    assert runner.async_execution_buffer["futures_last_accepted_token"] is carry.last_accepted


def test_a_prefill_carries_width_one_and_the_placeholder_drafts(no_commit, proposer):
    runner = _async_runner(proposer, record=_record(width=None, is_prefill=True, start=0), drafts=None)
    runner._glm5next_side_cache_positions = {SLOT: 9}
    runner._update_states_after_model_execute(_Unreadable.of(torch.tensor([11], dtype=torch.int32)), None)
    carry = runner._glm5next_async_carry
    assert carry.prev_width == 1 and carry.checkpoint_rows.tolist() == [0]
    assert carry.last_accepted.tolist() == [11]
    assert carry.next_input_ids.tolist() == [[11] + [PAD] * K]
    assert carry.drafts.tolist() == [[PAD] * K] and carry.drafts.dtype == torch.int32
    assert runner._glm5next_side_cache_positions == {SLOT: 9}


def test_a_one_row_decode_carries_width_one_and_the_roots_drafts(no_commit, proposer):
    drafts = torch.full((1, K), 3, dtype=torch.int32)
    runner = _async_runner(proposer, record=_record(width=1), drafts=drafts)
    runner._glm5next_side_cache_positions = {SLOT: START + 1}
    runner._update_states_after_model_execute(_Unreadable.of(torch.tensor([[8]], dtype=torch.int32)), None)
    carry = runner._glm5next_async_carry
    assert carry.prev_width == 1 and carry.checkpoint_rows.tolist() == [0]
    assert carry.next_input_ids.tolist() == [[8, 3, 3, 3]]
    assert runner._glm5next_side_cache_positions == {SLOT: START + 1}


def test_a_synthetic_step_leaves_the_carry_alone(no_commit, proposer):
    runner = _async_runner(proposer, record=None, drafts=None)
    runner._update_states_after_model_execute(_Unreadable.of(torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)), None)
    assert runner._glm5next_async_carry is None


def test_a_sampler_tensor_with_fewer_rows_than_requests_is_refused_by_name(no_commit, proposer):
    runner = _async_runner(proposer, record=_record(width=T), drafts=torch.zeros((2, K), dtype=torch.int32),
                           req_ids=(REQ, "req-b"))
    runner._glm5next_step_record["slots"] = [SLOT, 1]
    runner._glm5next_step_record["starts"] = [START, 12]
    with pytest.raises(ValueError, match=r"2 request\(s\).*1 row"):
        runner._update_states_after_model_execute(torch.tensor([[5, -1, -1, -1]], dtype=torch.int32), None)


def test_a_host_list_under_the_async_drafter_is_refused_by_name(no_commit, proposer):
    runner = _async_runner(proposer, record=_record(width=T), drafts=torch.zeros((1, K), dtype=torch.int32))
    with pytest.raises(ValueError, match="device tensor"):
        runner._update_states_after_model_execute([[5, 6]], None)


def test_a_decode_step_whose_root_returned_no_drafts_is_refused_by_name(no_commit, proposer):
    runner = _async_runner(proposer, record=_record(width=T), drafts=None)
    with pytest.raises(ValueError, match="draft"):
        runner._update_states_after_model_execute(torch.tensor([[5, -1, -1, -1]], dtype=torch.int32), None)


# ── the proposal: the carry's next input ids, as a tensor ───────────────────


def _settled(proposer, record, drafts, sampled):
    runner = _async_runner(proposer, record=record, drafts=drafts)
    runner._update_states_after_model_execute(_Unreadable.of(sampled), None)
    return runner


def test_after_a_verify_step_the_proposal_is_the_carrys_next_input_ids(no_commit, proposer):
    drafts = torch.arange(K, dtype=torch.int32).reshape(1, K) + 100
    runner = _settled(proposer, _record(width=T), drafts, torch.tensor([[5, 6, -1, -1]], dtype=torch.int32))
    proposed = runner._glm5next_propose_drafts(runner.async_execution_buffer.get("sampled"), None)
    assert proposed is runner._glm5next_async_carry.next_input_ids
    assert proposed.tolist() == [[6, 100, 101, 102]]
    assert runner._futures_drafts_only is drafts


def test_after_a_prefill_the_proposal_is_the_sampled_id_and_the_placeholders(no_commit, proposer):
    runner = _settled(proposer, _record(width=None, is_prefill=True, start=0), None, torch.tensor([11], dtype=torch.int32))
    proposed = runner._glm5next_propose_drafts(None, None)
    assert proposed.tolist() == [[11] + [PAD] * K]
    assert runner._futures_drafts_only.tolist() == [[PAD] * K]


def test_a_partial_prefill_chunk_proposes_nothing(no_commit, proposer):
    runner = _settled(proposer, _record(width=None, is_prefill=True, start=0), None, torch.tensor([11], dtype=torch.int32))
    runner._get_partial_prefill_req_ids = lambda scheduler_output, req_ids: {REQ}
    assert runner._glm5next_propose_drafts(None, SimpleNamespace()) == [[]]
    assert runner._futures_drafts_only is None


def test_near_max_model_len_the_proposal_is_still_the_carrys_tensor(no_commit, proposer):
    """The served async scheduler (the plugin's ``NeuronAsyncScheduler``) never consults the
    proposal -- it re-arms ``k`` placeholders every step and switches the request to one-row
    steps near ``max_model_len`` on its own, before vLLM's trim could shorten a step -- so
    proposing nothing near the limit would only lose the device future the next step takes
    its input ids from: the proposal is the carry's tensor on every decode step, however
    close to the limit."""
    drafts = torch.ones((1, K), dtype=torch.int32)
    for start in (MAX_MODEL_LEN - 2 * T, MAX_MODEL_LEN - T, MAX_MODEL_LEN - 2):
        runner = _settled(proposer, _record(width=T, start=start), drafts, torch.tensor([[5, -1, -1, -1]], dtype=torch.int32))
        assert runner._glm5next_propose_drafts(None, None) is runner._glm5next_async_carry.next_input_ids, start
        assert runner._futures_drafts_only is drafts, start


def test_a_synthetic_step_proposes_nothing(no_commit, proposer):
    runner = _async_runner(proposer, record=None, drafts=None)
    assert runner._glm5next_propose_drafts(None, None) == [[]]


def test_a_decode_step_without_a_carry_is_refused_by_name(no_commit, proposer):
    runner = _async_runner(proposer, record=_record(width=T), drafts=torch.ones((1, K), dtype=torch.int32))
    with pytest.raises(ValueError, match="carry"):
        runner._glm5next_propose_drafts(None, None)


# ── the translator: operands at the true positions ──────────────────────────


def _carry(req_ids, *, prev_width: int, checkpoint_rows: list[int]) -> async_step.StepCarry:
    rows = len(checkpoint_rows)
    return async_step.StepCarry(
        req_ids=tuple(req_ids), prev_width=prev_width,
        checkpoint_rows=torch.tensor(checkpoint_rows, dtype=torch.int32),
        last_accepted=torch.zeros(rows, dtype=torch.int32),
        next_input_ids=torch.zeros((rows, T), dtype=torch.int32),
        drafts=torch.zeros((rows, K), dtype=torch.int32),
    )


def _async_server(runner, proposer) -> None:
    """Make the translator world's runner serve the async drafter: the proposer, and the
    host launch state the real ``__init__`` sets (no device: the kernels launch eagerly)."""
    runner.is_mtp_spec = True
    runner.drafter = proposer
    runner.device = torch.device("cpu")
    runner._glm5next_async_launch_instance = None


def test_a_one_request_verify_step_is_laid_out_at_the_true_position(proposer):
    """Three verify steps of one request. The scheduler hands each step the position it
    would stand at had the previous step kept every row; the carry's resume row says how
    many it kept, and every per-row and per-request operand is laid out at the true
    position (the synchronous translator's formulas, ``_slot``), one step late."""
    world = translator._world(1)
    runner = world.runner
    _async_server(runner, proposer)
    start = world.lengths[0]
    slot = None
    # Step 1: the first verify after the prefill; nothing to pull back (prev width 1).
    runner._glm5next_async_carry = _carry(world.req_ids, prev_width=1, checkpoint_rows=[0])
    optimistic = start
    rejected = 0
    for kept in (2, 1, T):
        true = optimistic - rejected
        converted = translator._convert(world, [0], cached=[optimistic], tokens=T, real=[T])
        slot = runner._glm5next_step_record["slots"][0]
        for carrier in translator._sparse(world, converted["layer_carriers"]):
            assert carrier["seq_lens"].tolist() == [true + 1 + t for t in range(T)]
            assert carrier["seq_lens"].dtype == torch.int32
            assert carrier["latent_slots"].tolist() == [translator._slot(world, 0, true + t) for t in range(T)]
            assert carrier["latent_slots"].dtype == torch.int64
            assert carrier["position"].tolist() == [true] and carrier["start_position"].tolist() == [true]
            assert tuple(carrier["block_table_row"].shape) == (translator.WINDOW_BLOCKS, 1)
        # The host record keeps the handed (optimistic) position; the cursor stands past
        # every row of it, as the synchronous translator left it before its hook.
        assert runner._glm5next_step_record["starts"] == [optimistic]
        assert runner._glm5next_side_cache_positions[slot] == optimistic + T
        assert runner._glm5next_async_carry is None, "consumed by the step it corrected"
        # The step's take: ``kept`` rows kept; the scheduler advances by the whole width
        # and learns the rejections one step later.
        rejected = T - kept
        runner._glm5next_async_carry = _carry(world.req_ids, prev_width=T, checkpoint_rows=[kept - 1])
        optimistic = optimistic + T


def test_a_one_row_decode_is_laid_out_at_the_true_position(proposer):
    """The spec-to-non-spec transition step: one row, the previous verify step's rejections
    still to pull back; the one-ring carrier form takes the corrected operands."""
    world = translator._world(1)
    runner = world.runner
    _async_server(runner, proposer)
    start = world.lengths[0]
    runner._glm5next_async_carry = _carry(world.req_ids, prev_width=1, checkpoint_rows=[0])
    translator._convert(world, [0], cached=[start], tokens=T, real=[T])
    runner._glm5next_async_carry = _carry(world.req_ids, prev_width=T, checkpoint_rows=[1])
    true = start + 2
    converted = translator._convert(world, [0], cached=[start + T], tokens=1, real=[1], width=1)
    for carrier in translator._sparse(world, converted["layer_carriers"]):
        assert carrier["seq_lens"].tolist() == [true + 1]
        assert carrier["latent_slots"].tolist() == [translator._slot(world, 0, true)]
        # The 0-d start the synchronous one-ring form hands, so the graph compiled at
        # warmup serves this step; the correction's ``[1]`` start is viewed, not copied.
        assert carrier["start_position"].dim() == 0 and int(carrier["start_position"]) == true
        assert carrier["position"].dim() == 0 and int(carrier["position"]) == true
        assert tuple(carrier["block_table_row"].shape) == (translator.WINDOW_BLOCKS, 1)
        assert torch.is_tensor(carrier["tail"]), "the one-ring form"
    # And the one-row step after it: nothing to pull back.
    runner._glm5next_async_carry = _carry(world.req_ids, prev_width=1, checkpoint_rows=[0])
    converted = translator._convert(world, [0], cached=[start + T + 1], tokens=1, real=[1], width=1)
    for carrier in translator._sparse(world, converted["layer_carriers"]):
        assert int(carrier["position"]) == start + T + 1


def _signatures(carriers: list[dict]) -> list[dict[str, tuple]]:
    """Per carrier, every tensor operand's ``(shape, dtype)``: the part of a compiled graph's
    input signature the translator's operands set (a different shape is a new graph)."""
    return [
        {key: (tuple(value.shape), value.dtype) for key, value in carrier.items() if torch.is_tensor(value)}
        for carrier in carriers
    ]


def _assert_same_signatures(form: str, want: list[dict], got: list[dict]) -> None:
    """Carrier by carrier and operand by operand, so a failure names the operand."""
    assert want and len(got) == len(want), (form, len(got), len(want))
    for index, (sync, under_async) in enumerate(zip(want, got)):
        assert set(under_async) == set(sync), (form, index, sorted(set(under_async) ^ set(sync)))
        for key in sync:
            assert under_async[key] == sync[key], (form, index, key, under_async[key], sync[key])


def test_the_async_sparse_carriers_hand_the_warmed_graphs_the_synchronous_signatures(proposer):
    """Every warmup converts the synchronous form (no correction exists at warmup), so the
    decode graphs compiled at warmup take the synchronous translator's operand signatures. The
    async translator must hand the same ``(shape, dtype)`` per operand, for the verify form and
    for the one-ring one-row form, or the first served step of that form compiles a graph the
    warmup did not."""
    sync_world = translator._world(1)
    translator._mtp_server(sync_world.runner)
    start = sync_world.lengths[0]
    sync_verify = translator._convert(sync_world, [0], cached=[start], tokens=T, real=[T])
    sync_one_row = translator._convert(sync_world, [0], cached=[start + T], tokens=1, real=[1], width=1)
    world = translator._world(1)
    runner = world.runner
    _async_server(runner, proposer)
    runner._glm5next_async_carry = _carry(world.req_ids, prev_width=1, checkpoint_rows=[0])
    async_verify = translator._convert(world, [0], cached=[start], tokens=T, real=[T])
    runner._glm5next_async_carry = _carry(world.req_ids, prev_width=T, checkpoint_rows=[1])
    async_one_row = translator._convert(world, [0], cached=[start + T], tokens=1, real=[1], width=1)
    for form, sync, under_async in (
        ("verify", sync_verify, async_verify), ("one-row", sync_one_row, async_one_row),
    ):
        _assert_same_signatures(
            form,
            _signatures(translator._sparse(sync_world, sync["layer_carriers"])),
            _signatures(translator._sparse(world, under_async["layer_carriers"])),
        )


def test_the_async_recurrent_carriers_hand_the_warmed_graphs_the_synchronous_signatures(proposer):
    """The recurrent form, two requests: the carried ``linear_start`` and ``checkpoint_rows``
    take the host builders' shapes and dtypes at a verify step, at a bucket-padded verify
    step (a padding row) and at a one-row step."""
    starts = list(kda.PROMPTS[:2])
    forms = (
        ("verify", dict(cached=starts, tokens=2 * T, real=[T, T])),
        ("padded verify", dict(cached=starts, tokens=3 * T, real=[T, T])),
        ("one-row", dict(cached=starts, tokens=2, real=[1, 1], width=1)),
    )
    for form, step in forms:
        # Fresh worlds per form: each is the first decode step after the prefill.
        sync_world = kda._world(2)
        translator._mtp_server(sync_world.runner)
        sync = translator._kda_convert(sync_world, [0, 1], **step)
        world = kda._world(2)
        _async_server(world.runner, proposer)
        world.runner._glm5next_async_carry = _carry(world.req_ids, prev_width=1, checkpoint_rows=[0, 0])
        under_async = translator._kda_convert(world, [0, 1], **step)
        _assert_same_signatures(form, _signatures(sync), _signatures(under_async))


def test_the_handed_width_is_the_steps_own_and_only_one_or_the_full_width(proposer):
    """The correction pulls back from the previous step's own width, 1 or ``1 + k``: the two
    widths the served scheduler produces (``NeuronAsyncScheduler`` clears the placeholders,
    stickily, before vLLM's trim could shorten a step). Another width never reaches the
    carry (the translator refuses the ragged step first) and is refused here by name, with
    the derivation of what the generic bookkeeping would have done to it in the docstring."""
    runner = _async_runner(proposer, record=None, drafts=None)
    _bind(runner, "_glm5next_async_handed_width")
    assert runner._glm5next_async_handed_width(1) == 1
    assert runner._glm5next_async_handed_width(T) == T
    for width in (0, 2, K, T + 1):
        with pytest.raises(ValueError, match="1 or 4"):
            runner._glm5next_async_handed_width(width)


def test_the_recurrent_carriers_resume_from_the_carried_row_at_the_true_position(proposer):
    world = kda._world(2)
    runner = world.runner
    _async_server(runner, proposer)
    starts = list(kda.PROMPTS[:2])
    runner._glm5next_async_carry = _carry(world.req_ids, prev_width=1, checkpoint_rows=[0, 0])
    for carrier in translator._kda_convert(world, [0, 1], cached=starts, tokens=2 * T, real=[T, T]):
        assert carrier["state_checkpoints"] == T
        assert carrier["checkpoint_rows"].tolist() == [0, 0]
        assert carrier["start_position"].tolist() == starts
    kept = [3, 1]
    runner._glm5next_async_carry = _carry(world.req_ids, prev_width=T, checkpoint_rows=[n - 1 for n in kept])
    optimistic = [s + T for s in starts]
    true = [s + n for s, n in zip(starts, kept)]
    bucket = 3
    for carrier in translator._kda_convert(world, [0, 1], cached=optimistic, tokens=bucket * T, real=[T, T]):
        assert carrier["checkpoint_rows"].tolist() == [n - 1 for n in kept] + [0]
        assert carrier["checkpoint_rows"].dtype == torch.int32
        assert carrier["start_position"].tolist() == true + [1], "a padding row keeps its state"
        assert carrier["start_position"].dtype == torch.int32


def test_the_translator_uses_the_device_correction_not_the_host_formulas(monkeypatch, proposer):
    """The operands come from ``mtp_async_correct`` (the kernel on device): the host builders
    the synchronous translator calls for positions and slots are not what the carriers hold."""
    world = translator._world(1)
    runner = world.runner
    _async_server(runner, proposer)
    start = world.lengths[0]
    calls = []
    real = async_step.mtp_async_correct

    def recording(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(async_step, "mtp_async_correct", recording)
    runner._glm5next_async_carry = _carry(world.req_ids, prev_width=1, checkpoint_rows=[0])
    translator._convert(world, [0], cached=[start], tokens=T, real=[T])
    assert len(calls) == 1, "one correction per step, shared by every layer"
    (prev, optimistic, column), kwargs = calls[0]
    assert prev.tolist() == [0] and optimistic.tolist() == [start]
    assert tuple(column.shape) == (translator.WINDOW_BLOCKS, 1) and column.dtype == torch.int32
    # The runner hands its launch with every correction; on the host there is none.
    assert kwargs == {
        "prev_width": 1, "width": T, "page_size": translator.PAGE, "padding": 0, "launch": None,
    }


def test_a_decode_step_without_a_carry_is_refused_by_the_translator_by_name(proposer):
    world = translator._world(1)
    _async_server(world.runner, proposer)
    world.runner._glm5next_async_carry = None
    with pytest.raises(ValueError, match="carry"):
        translator._convert(world, [0], cached=[world.lengths[0]], tokens=T, real=[T])


def test_a_carry_naming_other_requests_is_refused_by_name(proposer):
    world = translator._world(1)
    _async_server(world.runner, proposer)
    world.runner._glm5next_async_carry = _carry(("someone-else",), prev_width=1, checkpoint_rows=[0])
    with pytest.raises(ValueError, match="someone-else"):
        translator._convert(world, [0], cached=[world.lengths[0]], tokens=T, real=[T])


def test_the_ring_cursor_check_is_a_bound_under_the_async_drafter(proposer):
    """The host cursor stands past every row of the previous step and the scheduler's
    position lags it by the rejections it has not learned yet: ``0 .. k`` behind is this
    sequence; ahead of the cursor, or more than ``k`` behind, is another point of it."""
    world = translator._world(1)
    runner = world.runner
    _async_server(runner, proposer)
    start = world.lengths[0]
    runner._glm5next_async_carry = _carry(world.req_ids, prev_width=1, checkpoint_rows=[0])
    translator._convert(world, [0], cached=[start], tokens=T, real=[T])
    cursor = start + T
    for handed in (cursor + 1, cursor - K - 1):
        runner._glm5next_async_carry = _carry(world.req_ids, prev_width=T, checkpoint_rows=[K])
        with pytest.raises(ValueError, match="stands at"):
            translator._convert(world, [0], cached=[handed], tokens=T, real=[T])
    runner._glm5next_async_carry = _carry(world.req_ids, prev_width=T, checkpoint_rows=[K])
    converted = translator._convert(world, [0], cached=[cursor - K], tokens=T, real=[T])
    assert translator._sparse(world, converted["layer_carriers"])[0]["position"].tolist() == [cursor - K]


def test_a_prefill_under_the_async_drafter_takes_its_positions_from_the_host(proposer, monkeypatch):
    monkeypatch.setenv(head_module.SHADOW_DRAFT_ENV, str(K))
    world = shadow._world()
    runner = world.runner
    _async_server(runner, proposer)
    runner._glm5next_async_carry = _carry(("stale",), prev_width=T, checkpoint_rows=[2])
    converted = translator._prefill_conversion(world)
    assert runner._glm5next_async_carry is not None, "a prefill consumes no carry"
    for carrier in translator._sparse(world, converted["layer_carriers"]):
        assert int(carrier["start_position"]) == 0
