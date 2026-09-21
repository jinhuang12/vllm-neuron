"""The runner's first request on the tiny root: warmup, one prefill, one decode, integer token ids.

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
        python -m pytest -s -rA test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_first_request.py

The serving path the item tests never drove: a real ``NeuronModelRunner`` built from an engine
config with no sampling or scheduling knob set, its two warmups, then ``execute_model`` on a
scheduler output, the way the worker calls it. Readings: the platform resolves on-device sampling
and async scheduling off from the model class, by name; warmup leaves no async-execution state;
the first request returns integer token ids from the vLLM Sampler over the full vocabulary, each
the strict argmax of the model's logits under a one-id head bias the item adds outside the model, and the
argmax of the reference under the same bias; a request whose generation crosses into a block that is NOT its
neighbour returns those ids through three steps, the last of which attends the second block out of the bank
instead of off the overlay, with two controls beside it -- the bank slot the table names holds that step's
own latent while the block a slice reader would have used does not, and the SEAM'S OUTPUT MOVES past this
file's equality tolerance when the last step is handed a slice reader's row, the two steps before it staying
bit-identical; the spec-to-non-spec transition is read from the recorded
fact, never taken without a speculative config and taken when the fact and the config agree; a
sampled-token future that holds logits is refused by name, where it used to reach ``numpy``. The controls re-create the class as it stood,
declaring a sampler it does not have.
"""

from __future__ import annotations

import contextlib
import logging
import pathlib
import traceback
from types import SimpleNamespace

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.distributed import parallel_state as dist_state
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

import vllm_neuron.vllm.worker.neuron_model_runner as runner_module
from vllm_neuron.functional.attention import mla_sparse as sparse_seam
from vllm_neuron.model.glm5_next import Glm5NextForConditionalGeneration
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as landed
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as item

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The fixture directory the engine config is built from: the checkpoint's config, no weights.
FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
REQUEST = "req-0"
#: The prefill bucket and the decode batch the warmups compile, the capture harness's own.
PREFILL_BUCKET = item.STACK_TOKENS
DECODE_BATCH = 1
#: vLLM keeps block 0 as the null block, so a scheduler hands out blocks from 1.
FIRST_BLOCK = 1
#: The page the serving configuration asks for. At this page the whole prompt sits in ONE block and the
#: generation is what crosses into the second, which is the crossing a served request makes; at this root's own
#: page of four the prompt already spans thirty-two blocks and no crossing is left to read.
SERVED_PAGE = 128
#: A NON-ADJACENT table: block numbers with holes between them, which a recycled pool hands out routinely. The
#: gap is what makes the table a table -- a reader that treated it as a slice would take the holes as content.
CROSSING_TABLE = (FIRST_BLOCK, FIRST_BLOCK + 3)
#: THE STEP THAT READS THE SECOND BLOCK OUT OF THE BANK, and the reason this file drives a second decode at
#: all. The layer hands the kernel this step's own rows beside the window and the kernel overlays them after
#: it copies the pages, so every row a step writes is supplied by the overlay and not by the page it lands
#: in. At position 128 that row IS the second block's only populated row, so the page the table's second
#: entry names is never read there and the same logits come back whichever block the entry holds. One step
#: later the row written is 129 and row 128 comes off the BANK, out of that page. The selection reaches it
#: whatever the scores say: the indexer appends the incomplete final pool's own token indices and derives
#: the tail itself, so a sequence of 130 over a pool of four appends exactly 128 and 129.
BANK_READ_POSITION = 129
#: The block a reader that took the table for a slice would put after the prompt's, which this request never
#: names. The falsifying control below hands the layer a row of these and requires the logits to move.
SLICE_BLOCK = FIRST_BLOCK + 1
#: The sparse seam as it was imported, held so two watched runs inside one item cannot wrap each other.
_REAL_SEAM = sparse_seam.mla_sparse_attention
#: The tolerance the end-to-end file compares the root's logits at.
LOGITS_RTOL, LOGITS_ATOL = 1e-2, 1e-5
#: The head bias the first-request item adds to the root's logits outside the model: +8 at one of the two ids the
#: seeded head scores equally (0 and 160), exact in bf16, so the argmax is that id with a margin far above the
#: comparison tolerance.
HEAD_BIAS_ID, HEAD_BIAS = 160, 8.0
ARCH = "Glm5NextForConditionalGeneration"
PLATFORM_LOGGER = "vllm_neuron.vllm.platform"


@contextlib.contextmanager
def _parallel_state(tmp_path, vllm_config):
    """One rank over gloo with a file rendezvous, under the config vLLM's parallel state and op layer read."""
    with set_current_vllm_config(vllm_config, check_compile=False):
        dist_state.init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{tmp_path / 'rendezvous'}",
            local_rank=0,
            backend="gloo",
        )
        dist_state.ensure_model_parallel_initialized(1, 1)
        try:
            yield
        finally:
            dist_state.destroy_model_parallel()
            dist_state.destroy_distributed_environment()


def _engine_config(*, async_scheduling: bool | None = None, on_device_sampling: bool | None = None,
                   page: int = item.MLA_PAGE_SIZE):
    """The engine config the worker builds for this root; ``None`` leaves a knob unset, as a serve does."""
    # The last prefill bucket must equal max_num_batched_tokens; the prompt picks the first.
    neuron_config: dict = {
        "num_batched_tokens_buckets": [PREFILL_BUCKET, landed.E2E_MAX_SEQ_LEN],
        "num_seqs_buckets": [DECODE_BATCH],
    }
    if on_device_sampling is not None:
        neuron_config["on_device_sampling_config"] = {} if on_device_sampling else None
    return EngineArgs(
        model=str(FIXTURE),
        skip_tokenizer_init=True,
        max_model_len=landed.E2E_MAX_SEQ_LEN,
        max_num_seqs=landed.E2E_MAX_NUM_SEQS,
        max_num_batched_tokens=landed.E2E_MAX_SEQ_LEN,
        block_size=page,
        enforce_eager=True,
        enable_prefix_caching=False,
        async_scheduling=async_scheduling,
        additional_config={"neuron_config": neuron_config},
    ).create_engine_config()


def _declaring_a_sampler(monkeypatch) -> None:
    """The class as it stood: declaring an on-device sampler it does not have, so both knobs stay on."""
    monkeypatch.setattr(Glm5NextForConditionalGeneration, "supports_on_device_sampling", True)


@contextlib.contextmanager
def _platform_rows():
    """Collect the platform's INFO rows while an engine config is built; yields the message list."""
    rows: list[str] = []

    class _Rows(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            rows.append(record.getMessage())

    handler = _Rows(level=logging.INFO)
    platform_logger = logging.getLogger(PLATFORM_LOGGER)
    level = platform_logger.level
    platform_logger.addHandler(handler)
    platform_logger.setLevel(logging.INFO)
    try:
        yield rows
    finally:
        platform_logger.removeHandler(handler)
        platform_logger.setLevel(level)


def _resolution(config) -> dict:
    """What the engine config resolved the two knobs and the scheduler to."""
    return {
        "on_device_sampling_config": config.additional_config["neuron_config"].get(
            "on_device_sampling_config", "absent"
        ),
        "async_scheduling": config.scheduler_config.async_scheduling,
        "scheduler_cls": str(config.scheduler_config.scheduler_cls).rsplit(".", 1)[-1],
    }


def _kv_cache_config(runner: NeuronModelRunner, num_blocks: int | None = None) -> KVCacheConfig:
    """One group per distinct spec, one tensor per layer, the generation's blocks plus the null block."""
    specs = runner.get_kv_cache_spec()
    num_blocks = landed.E2E_BLOCKS + 1 if num_blocks is None else num_blocks
    groups: list[KVCacheGroupSpec] = []
    for name, spec in specs.items():
        for group in groups:
            if group.kv_cache_spec == spec:
                group.layer_names.append(name)
                break
        else:
            groups.append(KVCacheGroupSpec([name], spec))
    tensors = [
        KVCacheTensor(size=num_blocks * spec.page_size_bytes, shared_by=[name])
        for name, spec in specs.items()
    ]
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=tensors, kv_cache_groups=groups)


def _runner(vllm_config, root, num_blocks: int | None = None) -> NeuronModelRunner:
    """The real runner on the tiny root, the model bound in place of ``load_model``."""
    runner = NeuronModelRunner(vllm_config, device=torch.device("cpu"))
    runner.model = root
    runner.vocab_size = item.STACK_VOCAB_SIZE
    runner.initialize_kv_cache(_kv_cache_config(runner, num_blocks))
    return runner


def _head_bias(root) -> torch.Tensor:
    """Add a one-id bias to the root's logits outside the model; returns the bias vector the reference adds too."""
    bias = torch.zeros(item.STACK_VOCAB_SIZE, dtype=torch.float32)
    bias[HEAD_BIAS_ID] = HEAD_BIAS
    forward = root.forward

    def biased(*args, **kwargs):
        logits = forward(*args, **kwargs)
        return logits + bias.to(logits.dtype)

    root.forward = biased
    return bias


def _warm(runner: NeuronModelRunner) -> None:
    """The worker's two warmups at the compile targets this root serves."""
    runner.warmup_prefill(PREFILL_BUCKET, 0)
    runner.warmup_decode(DECODE_BATCH, ctx_bucket=runner.max_model_len)


def _prompt() -> list[int]:
    """The end-to-end file's seeded prompt."""
    return torch.randint(
        0,
        item.STACK_VOCAB_SIZE,
        (item.STACK_TOKENS,),
        generator=torch.Generator().manual_seed(item.SEED_STACK_IDS),
        dtype=torch.int64,
    ).tolist()


def _prefill_step(prompt: list[int], groups: int, table: tuple[int, ...] | None = None) -> SchedulerOutput:
    """A new request's whole prompt, padded to its bucket the way the scheduler pads it."""
    blocks = list(table) if table is not None else list(range(FIRST_BLOCK, FIRST_BLOCK + landed.PROMPT_BLOCKS))
    new = NewRequestData(
        req_id=REQUEST,
        prompt_token_ids=prompt,
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=landed.GENERATED_TOKENS),
        pooling_params=None,
        block_ids=tuple(list(blocks) for _ in range(groups)),
        num_computed_tokens=0,
        lora_request=None,
    )
    step = SchedulerOutput(
        scheduled_new_reqs=[new],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={REQUEST: len(prompt)},
        total_num_scheduled_tokens=len(prompt),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    step.num_scheduled_tokens_padded = {REQUEST: PREFILL_BUCKET}
    return step


def _decode_step(position: int, generated: int, groups: int, page: int = item.MLA_PAGE_SIZE,
                 table: tuple[int, ...] | None = None) -> SchedulerOutput:
    """The request's next token at ``position``, with the block that position opens."""
    opens_block = position % page == 0
    index = position // page
    new_block = table[index] if table is not None else FIRST_BLOCK + index
    cached = CachedRequestData(
        req_ids=[REQUEST],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[tuple([new_block] for _ in range(groups)) if opens_block else None],
        num_computed_tokens=[position],
        num_output_tokens=[generated],
    )
    step = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={REQUEST: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    step.num_scheduled_tokens_padded = {REQUEST: 1}
    return step


def _sparse_bank(runner: NeuronModelRunner):
    """The sparse family's latent bank and its page: the tensor the layer reads its pages out of."""
    banks = getattr(runner.model, "glm5next_layer_banks", None)
    assert banks, "the runner's model carries no layer banks, so no bank row can be read here"
    for bank in banks:
        if bank["family"] == "self_attn":
            return bank["latent_cache"], int(bank["block_size"])
    raise AssertionError("the model carries no sparse self-attention bank")


def _watching_the_seam(monkeypatch, plant: tuple[int, ...] | None = None) -> dict:
    """Record what each sparse call overlays and what it RETURNS, and optionally plant a slice reader's row.

    The seam's own return is recorded because that is the tensor the block table feeds: a wrong page reaches
    the head through this root's three layers and a residual stream, which dilutes it, while the layer's own
    output carries it undiluted. Both are keyed by the bank's ``data_ptr``, so a call is paired with the bank
    it overlays rather than by the order the layers happen to run in.

    ``plant`` replaces the request's own entries in the table row, leaving the bucket's padding as it
    arrived. The switch starts OFF so the steps before the one under test write where the real table says;
    only the step the control names reads the planted row, against a bank the real table filled.

    The ORIGINAL seam is captured once at import, so two runs in one item cannot wrap each other.
    """
    state: dict = {"on": False, "rows": [], "calls": {}}

    def watched(queries, cache, *args, **kwargs):
        row = kwargs.get("block_table_row")
        if state["on"] and plant is not None and row is not None:
            replacement = row.clone()
            for entry in range(min(len(plant), int(replacement.shape[0]))):
                replacement[entry, 0] = int(plant[entry])
            state["rows"].append((row[:, 0].tolist(), replacement[:, 0].tolist()))
            kwargs["block_table_row"] = replacement
        attended = _REAL_SEAM(queries, cache, *args, **kwargs)
        written = kwargs.get("written")
        if written is not None:
            state["calls"].setdefault(cache.data_ptr(), []).append(
                (int(kwargs["write_offset"][0, 0]), written.detach().clone(), attended.detach().clone())
            )
        return attended

    monkeypatch.setattr(sparse_seam, "mla_sparse_attention", watched)
    return state


def _crossing_run(tmp_path, root, prompt: list[int], *, monkeypatch=None, watch: bool = False,
                  plant: tuple[int, ...] | None = None):
    """One crossing request driven three steps: the prompt, the step that opens the second block, and the
    step that reads that block out of the bank. Returns the ids, the logits, the bank and what was tapped.

    ``plant`` turns the slice table on for the THIRD step only, which is the one the bank serves.
    """
    config = _engine_config(page=SERVED_PAGE)
    switch: dict = {}
    with _parallel_state(tmp_path, config):
        runner = _runner(config, root, num_blocks=max(CROSSING_TABLE) + 1)
        _warm(runner)
        if watch or plant is not None:
            switch = _watching_the_seam(monkeypatch, plant)
        groups = _groups(runner)
        steps = (
            _prefill_step(prompt, groups, table=CROSSING_TABLE),
            _decode_step(position=len(prompt), generated=1, groups=groups, page=SERVED_PAGE,
                         table=CROSSING_TABLE),
            _decode_step(position=BANK_READ_POSITION, generated=2, groups=groups, page=SERVED_PAGE,
                         table=CROSSING_TABLE),
        )
        ids: list = []
        logits: list = []
        for step in steps:
            if plant is not None and len(logits) == 2:
                switch["on"] = True
            step_logits, output = _step(runner, step)
            logits.append(step_logits)
            ids.append(output.sampled_token_ids)
        bank, page = _sparse_bank(runner)
        pointer = bank.data_ptr()
        bank = bank.detach().clone()
    return SimpleNamespace(ids=ids, logits=logits, bank=bank, page=page, pointer=pointer,
                           calls=switch.get("calls", {}), planted=switch.get("rows", []))


def _groups(runner: NeuronModelRunner) -> int:
    """How many KV-cache groups the runner allocated, one block list per group in every step."""
    return len(runner.kv_cache_config.kv_cache_groups)


def _step(runner: NeuronModelRunner, step: SchedulerOutput):
    """The worker's two calls for one step: the logits the sampler saw and the output."""
    returned = runner.execute_model(step)
    logits = runner.execute_model_state.logits if runner.execute_model_state is not None else None
    output = runner.sample_tokens(None) if returned is None else returned
    return logits, output


def _recording(runner: NeuronModelRunner, monkeypatch) -> list[str]:
    """Record every reason the runner breaks the async flow for, leaving the break itself in place."""
    reasons: list[str] = []
    original = runner._materialize_pending_async_output

    def recorded(reason: str) -> bool:
        reasons.append(reason)
        return original(reason)

    monkeypatch.setattr(runner, "_materialize_pending_async_output", recorded)
    return reasons


def _frames(raised) -> list[str]:
    """The function names on the traceback, innermost last."""
    return [frame.name for frame in traceback.extract_tb(raised.tb)]


def test_sampling_and_scheduling_resolve_from_the_model_class():
    """No knob set: the platform reads the class, turns both off, and says so by name."""
    landed._require_cpu_mode()
    with _platform_rows() as rows:
        config = _engine_config()
    resolved = _resolution(config)
    named = [row for row in rows if ARCH in row and "no on-device sampler" in row]
    print(f"FIRSTREQ|resolution|declared={Glm5NextForConditionalGeneration.supports_on_device_sampling}"
          f"|{resolved}|rows={named}")
    assert Glm5NextForConditionalGeneration.supports_on_device_sampling is False
    assert resolved == {"on_device_sampling_config": None, "async_scheduling": False,
                        "scheduler_cls": "NeuronScheduler"}, resolved
    kinds = sorted({("On-device sampling is off" in row, "Async scheduling is off" in row) for row in named})
    assert kinds == [(False, True), (True, False)], named


def test_an_explicit_sampler_config_for_the_class_is_refused_by_name():
    """An on-device sampling config for a class without a sampler is refused, naming both."""
    landed._require_cpu_mode()
    with pytest.raises(ValueError) as raised:
        _engine_config(on_device_sampling=True)
    message = " ".join(str(raised.value).split())
    print(f"FIRSTREQ|explicit_sampler|message={message[:300]}")
    assert ARCH in message and "on_device_sampling_config" in message and "no on-device sampler" in message


def test_an_explicit_async_request_is_turned_off_by_name():
    """vLLM resolves the async default before the platform hook, so an explicit request is turned off the same way."""
    landed._require_cpu_mode()
    with _platform_rows() as rows:
        config = _engine_config(async_scheduling=True)
    resolved = _resolution(config)
    named = [row for row in rows if ARCH in row and "Async scheduling is off" in row]
    print(f"FIRSTREQ|explicit_async|{resolved}|rows={named}")
    assert resolved["async_scheduling"] is False and resolved["scheduler_cls"] == "NeuronScheduler", resolved
    assert named, rows


def test_the_runner_refuses_async_without_a_sampler_by_name(tmp_path, monkeypatch):
    """Backstop: the class declaring a sampler, sampling off and async on, the runner refuses naming the class."""
    landed._require_cpu_mode()
    _declaring_a_sampler(monkeypatch)
    config = _engine_config(async_scheduling=True, on_device_sampling=False)
    with _parallel_state(tmp_path, config), pytest.raises(RuntimeError) as raised:
        NeuronModelRunner(config, device=torch.device("cpu"))
    message = str(raised.value)
    print(f"FIRSTREQ|runner_backstop|{_resolution(config)}|message={message[:200]}")
    assert config.scheduler_config.async_scheduling is True
    assert ARCH in message and "synchronous scheduling" in message


def _warmed_state(tmp_path, config, root) -> tuple[bool, dict, bool]:
    """Build the runner, run both warmups, and read the async flag, the buffer, and whether a step is pending.

    A synchronous runner never creates the buffer; an absent buffer reads as empty.
    """
    with _parallel_state(tmp_path, config):
        runner = _runner(config, root)
        _warm(runner)
        buffer = dict(getattr(runner, "async_execution_buffer", {}))
        return runner.use_async_scheduling, buffer, runner.execute_model_state is not None


def test_warmup_leaves_no_async_execution_state(tmp_path):
    """No knob set: both warmups leave the buffer empty and no step pending."""
    landed._require_cpu_mode()
    is_async, buffer, pending = _warmed_state(tmp_path, _engine_config(), landed._fixture()["root"])
    print(f"FIRSTREQ|warmup|resolved|async={is_async}|buffer_keys={sorted(buffer)}|pending={pending}")
    assert is_async is False and buffer == {} and pending is False


def test_warmup_left_no_async_execution_state_as_it_stood(tmp_path, monkeypatch):
    """The class as it stood, async on: both warmups still leave the buffer empty, so there was nothing to clear."""
    landed._require_cpu_mode()
    _declaring_a_sampler(monkeypatch)
    is_async, buffer, pending = _warmed_state(
        tmp_path, _engine_config(async_scheduling=True), landed._fixture()["root"]
    )
    print(f"FIRSTREQ|warmup|as_it_stood|async={is_async}|buffer_keys={sorted(buffer)}|pending={pending}")
    assert is_async is True and buffer == {} and pending is False


def test_the_first_request_returns_integer_token_ids(tmp_path):
    """No knob set: prefill then decode return one int each, the strict argmax of the biased logits and of the biased reference."""
    landed._require_cpu_mode()
    fixture = landed._fixture()
    root = fixture["root"]
    bias = _head_bias(root)
    prompt = _prompt()
    config = _engine_config()
    with _parallel_state(tmp_path, config):
        runner = _runner(config, root)
        _warm(runner)
        prefill_logits, prefill = _step(runner, _prefill_step(prompt, _groups(runner)))
        first = prefill.sampled_token_ids
        decode_logits, decode = _step(
            runner, _decode_step(position=len(prompt), generated=1, groups=_groups(runner))
        )
        second = decode.sampled_token_ids
    print(f"FIRSTREQ|tokens|async={runner.use_async_scheduling}|ods={runner.on_device_sampling}"
          f"|prefill={first}|decode={second}|logits={tuple(prefill_logits.shape)},{prefill_logits.dtype}"
          f"|vocab={runner.vocab_size}")
    assert runner.use_async_scheduling is False and runner.on_device_sampling is False
    assert type(prefill).__name__ == "ModelRunnerOutput" and type(decode).__name__ == "ModelRunnerOutput"
    for label, ids, logits, sequence in (
        ("prefill", first, prefill_logits, prompt),
        ("decode", second, decode_logits, prompt + first[0]),
    ):
        assert ids == [[int(ids[0][0])]] and type(ids[0][0]) is int, (label, ids)
        assert tuple(logits.shape) == (1, runner.vocab_size) == (1, item.STACK_VOCAB_SIZE), (label, logits.shape)
        assert ids[0][0] == int(logits[0].float().argmax()), (label, ids)
        want = landed._reference_logits(fixture, torch.tensor(sequence, dtype=torch.int64))[0].float() + bias
        spread = float((logits[0].float() - want).abs().max())
        top = logits[0].float().topk(2)
        margin = float(top.values[0] - top.values[1])
        tolerance = LOGITS_RTOL * float(want.abs().max()) + LOGITS_ATOL
        ties = int((logits[0].float() == logits[0].float().max()).sum())
        print(f"FIRSTREQ|logits|{label}|tokens={len(sequence)}|max_abs_delta={spread:.6g}"
              f"|reference_argmax={int(want.argmax())}|top2={top.indices.tolist()}"
              f"|margin={margin:.6g}|tolerance={tolerance:.6g}|ties_at_the_max={ties}"
              f"|bias={HEAD_BIAS_ID}:+{HEAD_BIAS:g}")
        assert ids[0][0] == int(want.argmax()), (label, ids, int(want.argmax()))
        assert ties == 1 and margin > tolerance, (label, top.indices.tolist(), margin, tolerance)
        torch.testing.assert_close(logits[0].float(), want, rtol=LOGITS_RTOL, atol=LOGITS_ATOL)


def test_a_first_request_crossing_into_a_block_that_is_not_adjacent_returns_the_same_ids(tmp_path):
    """At the served page the prompt fills one block and the generation crosses into a NON-ADJACENT second.

    The table is what the layer reads a page number out of, so a table with a hole in it is the case a slice
    cannot serve: the prompt's block and the block the first decode opens are not neighbours, and the ids must
    still be the argmax of the reference under the same bias the item above adds.

    THREE STEPS AND NOT TWO, which is what makes the second block's page load-bearing here. The kernel
    overlays the rows a step writes onto the staged window after it copies the pages, so a step's own rows
    never come from a page: two steps would read the second block's only populated row off the overlay and
    pass on whichever block the table named. The third step writes 129 and attends 128 from the bank, out of
    the page the table's second entry names.

    WHAT DISCRIMINATES IS THE TWO CONTROLS BELOW AND NOT THIS ITEM'S EQUALITY, and the numbers say so: a
    slice reader's table moves this step's logits by 0.0078 against the 0.0130 this file compares at, so the
    head holds the wrong page below the tolerance even in a root of three layers. The controls read the bank
    slot and the seam's own output instead, where the difference is undiluted.
    """
    landed._require_cpu_mode()
    fixture = landed._fixture()
    root = fixture["root"]
    bias = _head_bias(root)
    prompt = _prompt()
    holes = tuple(second - first - 1 for first, second in zip(CROSSING_TABLE, CROSSING_TABLE[1:]))
    run = _crossing_run(tmp_path, root, prompt)
    first_ids, second_ids, third_ids = run.ids
    print(f"FIRSTREQ|crossing|page={SERVED_PAGE}|table={list(CROSSING_TABLE)}|holes={list(holes)}"
          f"|prompt_tokens={len(prompt)}|prefill={first_ids}|decode={second_ids}|bank_read={third_ids}"
          f"|bank_read_position={BANK_READ_POSITION}")
    assert holes and all(hole >= 1 for hole in holes), holes
    assert len(prompt) == SERVED_PAGE, (len(prompt), SERVED_PAGE)
    sequences = (prompt, prompt + first_ids[0], prompt + first_ids[0] + second_ids[0])
    for label, ids, logits, sequence in zip(("prefill", "decode", "bank_read"), run.ids, run.logits,
                                            sequences):
        assert ids == [[int(ids[0][0])]] and type(ids[0][0]) is int, (label, ids)
        want = landed._reference_logits(fixture, torch.tensor(sequence, dtype=torch.int64))[0].float() + bias
        spread = float((logits[0].float() - want).abs().max())
        print(f"FIRSTREQ|crossing_logits|{label}|tokens={len(sequence)}|max_abs_delta={spread:.6g}"
              f"|id={ids[0][0]}|reference_argmax={int(want.argmax())}")
        assert ids[0][0] == int(want.argmax()), (label, ids, int(want.argmax()))
        torch.testing.assert_close(logits[0].float(), want, rtol=LOGITS_RTOL, atol=LOGITS_ATOL)


def test_the_crossing_requests_row_lands_in_the_block_the_table_names(tmp_path, monkeypatch):
    """The bank row the third step attends is the second table entry's page, carrying position 128's latent.

    The step that opens the second block overlays its own row, so what proves the write reached the page the
    table named is the BANK: the slot the runner's own formula gives for position 128 -- the table's second
    entry times the page -- holds that step's latent EXACTLY, and the whole page a slice reader would have
    used instead holds nothing at all.

    THE TWO GENERATED LATENTS ARE THE SAME VECTOR HERE, which is why the distance to position 129 is printed
    and not asserted against. This root rotates nothing into the latent (``qk_rope_head_dim`` is 0) and both
    generated ids come back 160, so the first sparse layer's latent for 128 and for 129 is one value and an
    inequality between them would grade the fixture rather than the table. What separates the table from a
    slice is the page: the named slot carries the latent, the slice page carries zeros.
    """
    landed._require_cpu_mode()
    fixture = landed._fixture()
    prompt = _prompt()
    run = _crossing_run(tmp_path, fixture["root"], prompt, monkeypatch=monkeypatch, watch=True)
    overlaid = {offset: rows for offset, rows, _ in run.calls.get(run.pointer, [])}
    assert len(prompt) in overlaid and BANK_READ_POSITION in overlaid, sorted(overlaid)
    named = run.bank[CROSSING_TABLE[1] * run.page, 0, :].float()
    page = run.bank[SLICE_BLOCK * run.page:(SLICE_BLOCK + 1) * run.page, 0, :].float()
    at_128 = overlaid[len(prompt)][0].float()
    at_129 = overlaid[BANK_READ_POSITION][0].float()
    near = float((named - at_128).abs().max())
    far = float((named - at_129).abs().max())
    scale = float(at_128.abs().max())
    print(f"FIRSTREQ|crossing_bank|slot={CROSSING_TABLE[1] * run.page}|page={run.page}"
          f"|offsets_overlaid={sorted(overlaid)}|latent_scale={scale:.6g}"
          f"|delta_to_position_{len(prompt)}={near:.6g}|delta_to_position_{BANK_READ_POSITION}={far:.6g}"
          f"|the_two_generated_latents_are_one_vector={int(torch.equal(at_128, at_129))}"
          f"|slice_block={SLICE_BLOCK}|slice_page_max_abs={float(page.abs().max()):.6g}")
    assert scale > 0, at_128
    assert near == 0.0, (near, scale)
    assert float(page.abs().max()) == 0.0, float(page.abs().max())


def test_the_crossing_logits_move_when_the_second_block_is_read_as_a_slice(tmp_path, monkeypatch):
    """The falsifying control: hand the third step a slice reader's table and the seam's output has to move.

    The item above would pass on a consumer that ignored the table and read the prompt's block plus the one
    after it, unless reading that wrong page changes what comes back. So the same three steps run twice, both
    filling the bank through the real table, and in the second the LAST step is handed a row of consecutive
    blocks: the row the selection force-includes is then taken from a page this request never wrote.

    THE SEAM'S DELTA IS GRADED AGAINST THIS FILE'S OWN EQUALITY TOLERANCE, which is what makes the reading a
    discrimination and not a mere inequality: the wrong page moved the layer's output by 0.665 where that
    tolerance is 0.0246, a factor of twenty-seven, so the difference is far outside the band the head is
    compared in. The head's own delta is graded ABOVE ZERO only, because it measured 0.0078 against 0.0130
    and stays inside that band -- the finding this control exists to record, and the reason the seam and the
    bank are what discriminate here.

    THE FIRST TWO STEPS HAVE TO BE BIT-IDENTICAL between the runs, and that is the third conjunct: the
    planted row is switched on for the last step alone, so a seam return or a logits row that differed
    earlier would mean the two runs diverged for some reason of their own and the last step's delta would
    grade that divergence instead of the table.
    """
    landed._require_cpu_mode()
    fixture = landed._fixture()
    root = fixture["root"]
    prompt = _prompt()
    honest = _crossing_run(tmp_path, root, prompt, monkeypatch=monkeypatch, watch=True)
    slice_table = (CROSSING_TABLE[0], SLICE_BLOCK)
    lying = _crossing_run(tmp_path, root, prompt, monkeypatch=monkeypatch, plant=slice_table)
    attended = {"honest": honest.calls.get(honest.pointer, []), "slice": lying.calls.get(lying.pointer, [])}
    assert len(attended["honest"]) == 3 and len(attended["slice"]) == 3, {k: len(v) for k, v in
                                                                         attended.items()}
    was, now = attended["honest"][2][2].float(), attended["slice"][2][2].float()
    moved = float((was - now).abs().max())
    scale = float(was.abs().max())
    tolerance = LOGITS_RTOL * scale + LOGITS_ATOL
    head_moved = float((honest.logits[2][0].float() - lying.logits[2][0].float()).abs().max())
    head_tolerance = LOGITS_RTOL * float(honest.logits[2][0].float().abs().max()) + LOGITS_ATOL
    print(f"FIRSTREQ|crossing_falsifier|planted={list(slice_table)}|rows_planted={len(lying.planted)}"
          f"|first_row={lying.planted[0] if lying.planted else None}"
          f"|attended_scale={scale:.6g}|attended_delta={moved:.6g}|attended_tolerance={tolerance:.6g}"
          f"|attended_delta_over_tolerance={int(moved > tolerance)}"
          f"|head_delta={head_moved:.6g}|head_tolerance={head_tolerance:.6g}"
          f"|head_delta_over_tolerance={int(head_moved > head_tolerance)}"
          f"|honest_id={honest.ids[2]}|slice_id={lying.ids[2]}")
    assert lying.planted, "no sparse call was handed the planted row, so this control read nothing"
    assert all(before != after for before, after in lying.planted), lying.planted[0]
    for step in (0, 1):
        seam_same = torch.equal(attended["honest"][step][2], attended["slice"][step][2])
        head_same = torch.equal(honest.logits[step], lying.logits[step])
        print(f"FIRSTREQ|crossing_isolation|step={step + 1}|seam_return_is_bit_identical={int(seam_same)}"
              f"|logits_are_bit_identical={int(head_same)}")
        assert seam_same and head_same, (
            f"the runs already differed at step {step + 1}, before the planted row was switched on, so the "
            f"last step's delta below would grade that divergence and not the block table"
        )
    assert moved > tolerance, (
        f"the wrong page moved the layer's output by {moved:.6g}, inside the {tolerance:.6g} this file "
        f"compares tensors equal at, so a slice reader's table is not distinguishable at the seam either"
    )
    assert head_moved > 0.0, head_moved


def test_the_transition_is_read_from_the_recorded_fact(tmp_path, monkeypatch):
    """After a real step the fact is recorded False; a planted True is not read without a speculative config."""
    landed._require_cpu_mode()
    _declaring_a_sampler(monkeypatch)
    root = landed._fixture()["root"]
    prompt = _prompt()
    config = _engine_config(async_scheduling=True)
    with _parallel_state(tmp_path, config):
        runner = _runner(config, root)
        _warm(runner)
        _, prefill = _step(runner, _prefill_step(prompt, _groups(runner)))
        recorded = runner.async_execution_buffer.get("prev_step_was_spec", "absent")
        runner.async_execution_buffer["prev_step_was_spec"] = True
        reasons = _recording(runner, monkeypatch)
        with pytest.raises(TypeError):
            runner.execute_model(_decode_step(position=len(prompt), generated=1, groups=_groups(runner)))
    print(f"FIRSTREQ|transition|output={type(prefill).__name__}|recorded={recorded}"
          f"|speculative_config={runner.speculative_config}|reasons={reasons}")
    assert type(prefill).__name__ == "AsyncNeuronModelRunnerOutput"
    assert recorded is False
    assert runner.speculative_config is None
    assert reasons and all("spec" not in reason for reason in reasons), reasons


def test_a_future_that_holds_logits_is_refused_by_name(tmp_path, monkeypatch):
    """Control 1 with the refusal: the class as it stood, async on, the first decode is refused naming what the future holds."""
    landed._require_cpu_mode()
    _declaring_a_sampler(monkeypatch)
    root = landed._fixture()["root"]
    prompt = _prompt()
    config = _engine_config(async_scheduling=True)
    with _parallel_state(tmp_path, config):
        runner = _runner(config, root)
        assert runner.use_async_scheduling and runner.on_device_sampling
        _warm(runner)
        _step(runner, _prefill_step(prompt, _groups(runner)))
        future = runner.async_execution_buffer["futures_sampled_token_ids"]
        with pytest.raises(TypeError) as raised:
            runner.execute_model(_decode_step(position=len(prompt), generated=1, groups=_groups(runner)))
    message = str(raised.value)
    frames = _frames(raised)
    print(f"FIRSTREQ|refusal|future={tuple(future.shape)},{future.dtype}"
          f"|frames={frames[-4:]}|message={message[:200]}")
    assert future.dtype == torch.bfloat16 and future.ndim == 2
    assert str(future.dtype) in message and str(tuple(future.shape)) in message and REQUEST in message
    assert "logits" in message and "on_device_sampling_config" in message
    assert frames[-4:] == [
        "execute_model", "_materialize_pending_async_output", "get_output", "_refuse_non_integer_future",
    ], frames
    assert "_parse_rejection_sampling_output" not in frames


def test_without_the_refusal_the_same_future_reaches_numpy(tmp_path, monkeypatch):
    """Control 1 as it stood: with the refusal removed, the same step raises the bf16 TypeError from the rejection parser."""
    landed._require_cpu_mode()
    _declaring_a_sampler(monkeypatch)
    monkeypatch.setattr(runner_module, "_refuse_non_integer_future", lambda *args: None)
    root = landed._fixture()["root"]
    prompt = _prompt()
    config = _engine_config(async_scheduling=True)
    with _parallel_state(tmp_path, config):
        runner = _runner(config, root)
        _warm(runner)
        _step(runner, _prefill_step(prompt, _groups(runner)))
        with pytest.raises(TypeError) as raised:
            runner.execute_model(_decode_step(position=len(prompt), generated=1, groups=_groups(runner)))
    message = str(raised.value)
    frames = _frames(raised)
    print(f"FIRSTREQ|control|frames={frames[-4:]}|message={message[:120]}")
    assert "BFloat16" in message, message
    assert frames[-3:] == ["_materialize_pending_async_output", "get_output", "_parse_rejection_sampling_output"], frames


def test_the_transition_is_taken_when_the_fact_and_the_config_agree(tmp_path, monkeypatch):
    """A recorded spec step under a speculative config breaks the async flow at the transition, handing on the last accepted token."""
    landed._require_cpu_mode()
    _declaring_a_sampler(monkeypatch)
    root = landed._fixture()["root"]
    prompt = _prompt()
    config = _engine_config(async_scheduling=True)
    with _parallel_state(tmp_path, config):
        runner = _runner(config, root)
        _warm(runner)
        _step(runner, _prefill_step(prompt, _groups(runner)))
        accepted = torch.tensor([prompt[-1]], dtype=torch.int64)
        runner.speculative_config = SimpleNamespace(num_speculative_tokens=1)
        runner.async_execution_buffer["prev_step_was_spec"] = True
        runner.async_execution_buffer["futures_last_accepted_token"] = accepted
        reasons = _recording(runner, monkeypatch)
        with pytest.raises(TypeError) as raised:
            runner.execute_model(_decode_step(position=len(prompt), generated=1, groups=_groups(runner)))
    frames = _frames(raised)
    print(f"FIRSTREQ|transition_taken|reasons={reasons}"
          f"|bonus_is_the_planted_tensor={runner._transition_bonus_tensor is accepted}"
          f"|composition_changed={runner._batch_composition_changed}|frames={frames[-3:]}")
    assert reasons[:1] == ["spec\u2192non-spec transition"], reasons
    assert runner._transition_bonus_tensor is accepted
    assert runner._batch_composition_changed is True
    assert frames[-3:] == ["_materialize_pending_async_output", "get_output", "_refuse_non_integer_future"], frames
