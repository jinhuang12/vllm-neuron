# SPDX-License-Identifier: Apache-2.0
"""The draft layer's latent rows a verify step writes persist across requests.

With MTP drafting (``k = 3``), the output for identical greedy prompts alternated
between two classes over consecutive ``bs=1`` requests; without a draft head it did
not. The state that survived a request was not a request slot's: it was the draft
layer's (layer 45's) paged ``latent_cache``. A verify step writes that bank ``1 + k``
times in one graph (``populate`` at the ``T = 1 + k`` verify rows, then one row per
draft iteration). Written through a view (``latent_cache[:, 0, :].index_copy_``), the
backend's in-place to out-of-place lowering (``InPlaceToOutOfPlacePass``) moved only
the later uses of that view onto each write's result, so no later view saw a write
and only the last write reached the aliased output the runtime copies back: the
accepted rows' latents did not persist, the draft iterations attended what the
request's pages held before the step, and one draft row was stored. vLLM's
need-sized pool hands consecutive ``bs=1`` requests alternating block sets, each
holding its own leftovers, so each set settled on its own drafts. The trunk's layers
write their banks once per graph, so the last write was the only one and they kept it.

Every test here runs the graph the way the device does: the step is traced with
Dynamo as one graph, the backend's default pass list (``get_default_pass_manager``)
rewrites it, the rewritten graph runs on CPU and each aliased output (``io_map``) is
copied onto its input.

1. One decode step keeps every bank the eager step writes: at ``k = 0`` (no draft
   head), and at ``k = 1`` and ``k = 3`` the draft layer's ``latent_cache`` too.
2. Two identical drafted requests, one after another on the two block sets vLLM
   alternates between, the second set holding another request's leftovers: the same
   ids, the same drafts, the same draft-layer and trunk rows at the positions they
   wrote, and the same per-slot state.
3. A step that writes a latent row reads it back in the same graph: the view the
   attention kernel gathers from (``c_kv``, taken after the write) holds the fresh rows,
   on the trunk's one-request decode (``attend``) and on the draft layer's ``T``-row
   verify write (``_attend_requests``).
4. The request-slot log line.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_glm5next_slot_residue.py
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch._dynamo as dynamo

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next import test_mtp_e2e_spec as verify
from test.vllm_neuron.model.glm5_next import test_shadow_draft_e2e as shadow
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_decode as batch

pytestmark = [pytest.mark.forked]

RUNNER_LOGGER = "vllm_neuron.vllm.worker.neuron_model_runner"
SLOTS = 4

# The served draft count, and the smallest one that writes the draft layer twice.
DRAFT_COUNTS = (1, 3)
SERVED_K = max(DRAFT_COUNTS)
# The drafts every verify step carries.
DRAFTS = (11, 23, 37)
# The request that ran on the second block set before the two identical ones: a
# different, longer prompt, so the rows it leaves are not the identical request's.
PRIOR_PROMPT = shadow.LONG_PROMPT
# The device the default pass list rewrites for. ``"cpu"`` so the rewritten graph runs
# here: the target sets only where ``DeviceRewriterPass`` places tensors, and the
# aliasing and in-place passes, whose output this file reads, are the served ones.
PASS_TARGET = "cpu"


# ── the step through the backend's own passes ───────────────────────────────


def _through_the_backend(fn, kwargs: dict) -> dict:
    """Dynamo-trace ``fn(**kwargs)`` as one graph and run the backend's default passes.

    Returns the rewritten (out-of-place) graph, its ``io_map`` (aliased output index to
    graph input index), the original output count and the graph's inputs. The traced
    call runs the rewritten graph once, which writes nothing back; side effects on
    Python lists (a ``collector``) are replayed with that run's values.
    """
    from libtorch_neuronx_lite.fx_passes import get_default_pass_manager

    kept: dict = {}

    def backend(gm, example_inputs):
        rewritten, meta = get_default_pass_manager().run_passes(
            gm, target_device=PASS_TARGET, compiler_workdir=None
        )
        aliasing = meta["aliasing_output_rewrite"]
        kept.update(
            gm=rewritten, io_map=dict(aliasing["io_map"]),
            outputs=int(aliasing["original_output_count"]), inputs=list(example_inputs),
        )
        return rewritten.forward

    dynamo.reset()
    torch.compile(fn, backend=backend, dynamic=False, fullgraph=True)(**kwargs)
    dynamo.reset()
    return kept


def _serve(lowered: dict) -> tuple:
    """Run the rewritten graph and copy each aliased output onto its input (the runtime's
    write-back); return the graph's own outputs."""
    outputs = lowered["gm"](*lowered["inputs"])
    outputs = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
    for output_index, input_index in lowered["io_map"].items():
        lowered["inputs"][input_index].copy_(outputs[output_index])
    return tuple(outputs[: lowered["outputs"]])


def _served_forward(world, kwargs: dict) -> tuple:
    """The root's forward on ``kwargs`` as the device runs it."""
    return _serve(_through_the_backend(lambda **kw: world.root.forward(**kw), dict(kwargs)))


# ── the tiny root's steps ───────────────────────────────────────────────────


def _world(monkeypatch, k: int):
    """The tiny root after its prompt's prefill; with the draft head at ``k >= 1``."""
    if k:
        return verify._verify_world(monkeypatch, k=k)
    monkeypatch.setenv(verify.KNOB, "0")
    world = shadow._world()
    assert world.root.mtp is None
    shadow._step(world, shadow.PROMPT, cached=0, sampling=[len(shadow.PROMPT) - 1])
    return world


def _prefill_kwargs(world, prompt: list[int]) -> dict:
    """The root's kwargs for request 0's prefill of ``prompt`` (``shadow._step``'s form)."""
    runner = world.runner
    runner.input_batch.req_ids = list(world.req_ids)
    runner._glm5next_request_tokens = None
    return runner._glm5next_model_kwargs({
        "input_ids": torch.tensor(prompt),
        "positions": None,
        "attn_metadata": batch._metadata(world, [0], cached=[0], tokens=len(prompt)),
        "sampling_positions": torch.tensor([len(prompt) - 1], dtype=torch.long),
        "sampling_params": torch.tensor([shadow.GREEDY_ROW], dtype=torch.float32),
        "spec_decode_metadata": None,
        "rank": None,
        "logit_mask": None,
    })


def _decode_kwargs(world, k: int, prompt: list[int] = shadow.PROMPT) -> dict:
    """The root's kwargs for the step after ``prompt``'s prefill: one row at ``k = 0``, a
    verify step of ``1 + k`` rows otherwise. Converted once: the conversion moves the
    runner's ring cursor, so the eager and the traced step share these kwargs."""
    start = len(prompt)
    drafts = [list(DRAFTS[:k])]
    inputs = [prompt[-1]] + drafts[0]
    width = 1 + k
    runner = world.runner
    runner.input_batch.req_ids = list(world.req_ids)
    runner._glm5next_request_tokens = np.array([width], np.int32) if k else None
    metadata = batch._metadata(world, [0], cached=[start], tokens=len(inputs))
    if k:
        for entry in metadata.values():
            entry["max_query_len"] = width
            entry["decode_token_threshold"] = width
    return runner._glm5next_model_kwargs({
        "input_ids": torch.tensor(inputs, dtype=torch.int32),
        "positions": None,
        "attn_metadata": metadata,
        "sampling_positions": torch.arange(len(inputs), dtype=torch.long),
        "sampling_params": torch.tensor([shadow.GREEDY_ROW] * len(inputs), dtype=torch.float32),
        "spec_decode_metadata": verify._metadata_for(drafts) if k else None,
        "rank": None,
        "logit_mask": None,
    })


def _state(world, kwargs: dict) -> dict[str, torch.Tensor]:
    """Every floating-point bank the step can write, by name, one entry per storage: the
    runner-shaped KV caches by cache name, then each other carrier tensor (the runner's
    per-slot side caches) by its path."""
    named: dict[str, torch.Tensor] = {}
    storages: set[int] = set()

    def add(name: str, tensor: torch.Tensor) -> None:
        storage = tensor.untyped_storage().data_ptr()
        if tensor.is_floating_point() and storage not in storages:
            storages.add(storage)
            named[name] = tensor

    for name, tensors in world.caches.items():
        for index, tensor in enumerate(tensors):
            add(f"{name}[{index}]", tensor)

    def walk(path: str, value) -> None:
        if torch.is_tensor(value):
            add(path, value)
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                walk(f"{path}[{index}]", item)
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(f"{path}[{key!r}]", item)

    walk("layer_carriers", kwargs["layer_carriers"])
    return named


def _draft_layer_cache(world) -> str:
    """The name of the draft layer's latent cache in ``_state``."""
    depth = len(world.root.model.layers)
    return f"{world.root.glm5next_layer_banks[depth]['name']}[0]"


def _positions(world, cache: torch.Tensor, rows: list[int]) -> list[int]:
    """Request 0's positions of a ``[blocks, 1, page, width]`` cache's flat rows."""
    page = int(cache.shape[2])
    where = {block * page + offset: index * page + offset
             for index, block in enumerate(world.tables[0]) for offset in range(page)}
    return [where.get(row, -1) for row in rows]


def _rows(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1, int(tensor.shape[-1]))


def _lost_writes(world, name: str, served: torch.Tensor, eager: torch.Tensor,
                 before: torch.Tensor) -> str:
    """Which of the eager step's rows the served step leaves as the eager step does."""
    wrote = (_rows(eager) != _rows(before)).any(dim=1).nonzero().flatten().tolist()
    kept = [row for row in wrote if torch.equal(_rows(served)[row], _rows(eager)[row])]
    if served.dim() == 4:
        wrote, kept = _positions(world, served, wrote), _positions(world, served, kept)
    return f"{name}: the eager step writes position(s) {wrote}; the served step matches it at {kept}"


# ── 1. one decode step keeps what it writes ─────────────────────────────────


@pytest.mark.parametrize("k", (0, *DRAFT_COUNTS), ids=lambda k: f"k{k}")
def test_every_bank_a_decode_step_writes_survives_the_backend_passes(monkeypatch, k):
    world = _world(monkeypatch, k)
    kwargs = _decode_kwargs(world, k)
    state = _state(world, kwargs)
    before = {name: bank.clone() for name, bank in state.items()}
    eager_out = world.root.forward(**dict(kwargs))
    eager = {name: bank.clone() for name, bank in state.items()}
    for name, bank in state.items():
        bank.copy_(before[name])
    served_out = _served_forward(world, kwargs)
    lost = [
        _lost_writes(world, name, bank, eager[name], before[name])
        for name, bank in state.items() if not torch.equal(bank, eager[name])
    ]
    assert not lost, (
        f"k={k}: the served step leaves {len(lost)} bank(s) unlike the eager step "
        f"(the draft layer's latent_cache is {_draft_layer_cache(world) if k else 'absent'}); "
        + "; ".join(lost)
    )
    eager_out = eager_out if isinstance(eager_out, tuple) else (eager_out,)
    assert len(served_out) == len(eager_out)
    for served, reference in zip(served_out, eager_out):
        assert torch.equal(served, reference), (served.tolist(), reference.tolist())


# ── 2. two identical drafted requests on the alternating block sets ─────────


def _two_block_set_world(monkeypatch, k: int):
    """The tiny root with the draft head, nothing prefilled, a one-slot runner and a KV
    pool of two disjoint block sets. vLLM's pool with prefix caching off prepends freed
    blocks in reverse, so consecutive one-request steps alternate between a set and its
    reverse; the second set here is descending for that reason."""
    monkeypatch.setenv(verify.KNOB, str(k))
    world = shadow._world()
    per_request = int(world.window_blocks)
    page = shadow.PAGE
    caches = {}
    for spec in world.root.get_kv_spec().layers:
        shape = (1 + 2 * per_request, int(spec.num_kv_heads), page, int(spec.head_size))
        caches[spec.name] = [
            torch.zeros(shape, dtype=spec.dtype) for _ in range(1 if spec.latent_kv else 2)
        ]
    world.root.bind_kv_cache(caches)
    world.caches = caches
    world.runner.is_mtp_spec = True
    world.runner.drafter = SimpleNamespace(num_speculative_tokens=k)
    sets = (list(range(1, 1 + per_request)), list(range(2 * per_request, per_request, -1)))
    return world, sets


def _served_request(world, request_id: str, prompt: list[int], blocks: list[int], k: int):
    """One drafted request served as the device serves it: the engine finishes the
    previous request, this one takes the slot and ``blocks``, its prefill and one verify
    step run through the backend's passes. Returns the ids, the drafts, every DSA
    cache's rows at the positions the request wrote, and every per-slot side cache."""
    runner = world.runner
    runner._glm5next_note_finished_requests(set(world.req_ids))
    world.req_ids = [request_id]
    world.tables = [list(blocks)]
    runner.requests[request_id] = SimpleNamespace(
        prompt_token_ids=list(prompt), num_prompt_tokens=len(prompt)
    )
    _served_forward(world, _prefill_kwargs(world, prompt))
    kwargs = _decode_kwargs(world, k, prompt)
    accepted, drafts = _served_forward(world, kwargs)
    page = shadow.PAGE
    written = range(len(prompt) + 1 + k)
    rows = {
        f"{name}[0]": torch.stack([
            tensors[0][blocks[position // page], 0, position % page] for position in written
        ])
        for name, tensors in world.caches.items()
    }
    side = {name: bank.clone() for name, bank in _state(world, kwargs).items() if name not in rows}
    return accepted.clone(), drafts.clone(), rows, side


def test_two_identical_drafted_requests_on_the_alternating_block_sets_leave_the_same(monkeypatch):
    world, (first_set, second_set) = _two_block_set_world(monkeypatch, SERVED_K)
    _served_request(world, "req-prior", PRIOR_PROMPT, second_set, SERVED_K)
    first = _served_request(world, "req-first", shadow.PROMPT, first_set, SERVED_K)
    second = _served_request(world, "req-second", shadow.PROMPT, second_set, SERVED_K)
    (ids_a, drafts_a, rows_a, side_a), (ids_b, drafts_b, rows_b, side_b) = first, second
    draft_layer = _draft_layer_cache(world)
    differ = (rows_a[draft_layer] != rows_b[draft_layer]).any(dim=1).nonzero().flatten().tolist()
    assert not differ, (
        f"{draft_layer} (the draft layer's latent_cache): the second identical request, "
        f"on the block set another request used before, leaves position(s) {differ} "
        f"unlike the first"
    )
    for name in rows_a:
        assert torch.equal(rows_a[name], rows_b[name]), name
    assert side_a.keys() == side_b.keys()
    for name in side_a:
        assert torch.equal(side_a[name], side_b[name]), name
    assert torch.equal(drafts_a, drafts_b), (drafts_a.tolist(), drafts_b.tolist())
    assert torch.equal(ids_a, ids_b), (ids_a.tolist(), ids_b.tolist())


# ── 3. a step reads back the latent rows it wrote ───────────────────────────


def _layer_and_carrier(monkeypatch, leg: str):
    """The attention module and carrier of one leg: the trunk's first layer on a
    one-request decode step (``attend``), or the draft layer on the served verify step,
    whose ``T`` rows take the request form (``_attend_requests``)."""
    if leg == "trunk-decode":
        world = _world(monkeypatch, 0)
        kwargs = _decode_kwargs(world, 0)
        return world, world.root.model.layers[0].self_attn, dict(kwargs["layer_carriers"][0])
    world = _world(monkeypatch, SERVED_K)
    kwargs = _decode_kwargs(world, SERVED_K)
    depth = len(world.root.model.layers)
    return world, world.root.mtp.block.self_attn, dict(kwargs["layer_carriers"][depth])


def _read_back(collected: list, latent_slots: torch.Tensor) -> tuple:
    """``(written, slots, c_kv)`` from an attention collector: the latent the write
    carries, its bank rows, and the view the kernel gathers from (the write's two
    entries, then that view, in the dump's positional order)."""
    slots = latent_slots.to(torch.int32)
    index = next(i for i, entry in enumerate(collected)
                 if torch.is_tensor(entry) and entry.dtype == torch.int32
                 and torch.equal(entry, slots))
    return collected[index - 1], slots, collected[index + 1]


@pytest.mark.parametrize("leg", ("trunk-decode", "draft-verify"))
def test_a_step_reads_back_the_latent_rows_it_wrote_in_the_same_graph(monkeypatch, leg):
    world, attention, carrier = _layer_and_carrier(monkeypatch, leg)
    carrier.pop("is_prefill", None)
    latent = carrier["latent_cache"]
    rows = int(carrier["latent_slots"].shape[0])
    generator = torch.Generator().manual_seed(shadow.SEED)
    normed = torch.randn(rows, int(world.root.text_config.hidden_size), generator=generator)
    normed = normed.to(latent.dtype)
    state = _state(world, {"layer_carriers": [carrier]})
    before = {name: bank.clone() for name, bank in state.items()}
    eager: list = []
    attention(normed, collector=eager, **carrier)
    for name, bank in state.items():
        bank.copy_(before[name])
    served: list = []
    _through_the_backend(lambda **kw: attention(normed, collector=served, **kw), carrier)
    written, slots, c_kv = _read_back(served, carrier["latent_slots"])
    fresh = written.to(c_kv.dtype)
    stale = [int(slot) for row, slot in enumerate(slots.tolist())
             if not torch.equal(c_kv[slot], fresh[row])]
    assert not stale, (
        f"{leg}: the view the kernel gathers from (c_kv, taken after the latent write) "
        f"holds the row(s) from before the write at bank slot(s) {stale}"
    )
    _, _, eager_c_kv = _read_back(eager, carrier["latent_slots"])
    assert torch.equal(c_kv, eager_c_kv)


# ── 4. the request-slot log line ────────────────────────────────────────────


def _banks(slots: int = SLOTS) -> list[dict]:
    return [{"name": "linear.0", "family": "linear_attn", "state_slots": slots}]


def _runner(slots: int = SLOTS) -> NeuronModelRunner:
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.max_num_reqs = slots
    runner._glm5next_request_slot_table = {}
    runner._glm5next_side_cache_positions = {}
    runner._glm5next_checkpoint_rows = {}
    return runner


def _slot_lines(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == RUNNER_LOGGER and "request slot" in record.getMessage()
    ]


@pytest.mark.fast
def test_handing_out_and_releasing_a_slot_each_log_one_info_line(caplog):
    """Which slot a request held is the first fact a cross-request residue investigation
    needs, and before this line the server logged no slot ids."""
    runner, banks = _runner(), _banks()
    with caplog.at_level(logging.INFO, logger=RUNNER_LOGGER):
        assert runner._glm5next_request_slots(banks, ["req-a"], synthetic=False) == [0]
        assert runner._glm5next_request_slots(banks, ["req-a", "req-b"], synthetic=False) == [0, 1]
        # A continuing request takes no new slot and logs nothing.
        runner._glm5next_request_slots(banks, ["req-a", "req-b"], synthetic=False)
        runner._glm5next_note_finished_requests({"req-a"})
        assert runner._glm5next_request_slots(banks, ["req-c"], synthetic=False) == [0]
    assert _slot_lines(caplog) == [
        "glm5next request 'req-a' holds request slot 0",
        "glm5next request 'req-b' holds request slot 1",
        "glm5next request 'req-a' released request slot 0",
        "glm5next request 'req-c' holds request slot 0",
    ]
    assert all(
        record.levelno == logging.INFO
        for record in caplog.records
        if record.name == RUNNER_LOGGER and "request slot" in record.getMessage()
    )


@pytest.mark.fast
def test_a_synthetic_step_takes_no_slot_and_logs_nothing(caplog):
    runner, banks = _runner(), _banks()
    with caplog.at_level(logging.INFO, logger=RUNNER_LOGGER):
        assert runner._glm5next_request_slots(banks, [None], synthetic=True) == [0]
        assert runner._glm5next_request_slots(banks, [None, None], synthetic=True) == [0, 1]
    assert _slot_lines(caplog) == []
    assert runner._glm5next_request_slot_table == {}


@pytest.mark.fast
def test_only_tensor_parallel_rank_zero_logs_the_slot_lines(caplog, monkeypatch):
    runner, banks = _runner(), _banks()
    monkeypatch.setattr(NeuronModelRunner, "_glm5next_shadow_rank", lambda self: 3)
    with caplog.at_level(logging.INFO, logger=RUNNER_LOGGER):
        assert runner._glm5next_request_slots(banks, ["req-a"], synthetic=False) == [0]
        runner._glm5next_note_finished_requests({"req-a"})
        runner._glm5next_request_slots(banks, ["req-b"], synthetic=False)
    assert _slot_lines(caplog) == []
