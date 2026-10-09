"""The shadow draft is read-only on trunk state: knob 5 and knob 0 agree bit for bit, every step.

Gate finding F1 (mtp-A, 2026-10-08): 16 of 200 GSM8K generations under knob 5 differed from
knob 0 on the device, while knob 0 reproduces itself across launches. Two readings: (H1) the
knob-5 decode graph is a different compiled graph, so the trunk's rounding differs and greedy
near-ties flip; (H2) a draft write reaches trunk state (a KV or KDA row, the pooled store, the
null block, the ring) and changes later trunk tokens. H1 cannot show on the simulator (no
compiler, eager arithmetic, one rounding), H2 can; so this file runs the knob-5 and the knob-0
arm through the same steps and compares, after EVERY step, every logit and every trunk tensor
a step may write -- the stack layers' banks and their side caches -- bit for bit. The trunk is
the hybrid tiny stack (``test_independent_prefill_state._hybrid_root``: a real KDA layer between
two DSA layers), so both state families are in the compared set: the DSA layers' latent banks
with their pooled store, tail ring and scratch ring, and the KDA layer's convolution and
recurrent banks. The draft layer's own bank and side caches (index ``depth``, the last entry of
the carrier walk) are the one place the draft is allowed to write and are not compared.

What the knob-5 arm does, so the writes the shadow legs can make are all made: three requests
of different lengths, one prefilled in two chunks (the boundary id of the prefill leg); 64
batched decode steps, teacher-forced with one random token per request and step so each step
answers a new input on both arms; a draft of ``K`` positions from every decode step, which at
``MLA_PAGE_SIZE`` 4 and ``K`` 5 crosses into a page the block table does not name on every
step (the slot-0 fallback of ``Glm5NextMultiTokenPredictor._latent_slots_at``), counted from
the carrier the draft layer was handed; the runner's shadow glue under the async form with one
decode step never read back and a shutdown behind it (the scorer is host code and must leave
the tensors alone too). The second test proves the comparison can see a leak: one element of trunk
layer 0's first bank written from inside ``draft_tokens`` is reported at the step it
happens, by the tensor's name.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/test_shadow_draft_trunk_readonly.py
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.model.glm5_next.mtp import Glm5NextMultiTokenPredictor
from vllm_neuron.vllm.worker.neuron_model_runner import NULL_BLOCK_ID, NeuronModelRunner
from test.vllm_neuron.model.glm5_next import test_shadow_draft_e2e as shadow
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_decode as batch
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny
from test.vllm_neuron.model.glm5_next.tiny import test_independent_prefill_state as hybrid

pytestmark = [pytest.mark.forked]

K = 5
KNOB = shadow.KNOB
LOG_KNOB = "VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT_LOG"
PAGE = tiny.MLA_PAGE_SIZE
SEED = 20261008
#: Three requests of different lengths; request 1 is prefilled in two chunks, so the prefill
#: leg's boundary id (the next prompt token, then -1) is exercised on both of its forms.
PROMPTS = (6, 9, 13)
CHUNKED_REQUEST, CHUNK = 1, 5
PREFILL_STEPS = len(PROMPTS) + 1
DECODE_STEPS = 64
#: The decode step whose output is never read back (its record is claimed and dropped), so
#: the shutdown walks a gap: the shadow glue's host-side path, run beside the tensors.
LOST_STEP = 20
#: Room for the last step's draft positions past the last real position.
MAX_MODEL_LEN = max(PROMPTS) + DECODE_STEPS + K + PAGE


# ── the world ────────────────────────────────────────────────────────────────


def _world(monkeypatch, knob: int):
    """The hybrid tiny root at ``knob`` (read at construction), its caches, a runner shell.

    The root is ``_hybrid_root``'s: the tiny DSA stack with a real KDA layer in the middle,
    so the trunk holds both state families. The caches are the runner's own shapes
    (``_runner_shaped_caches``: ``[slots, *state]`` banks for the KDA layer), with the paged
    banks widened to three requests' pages after the null block. The block table the
    translator hands a layer names only the pages a request has used, so a draft position
    past the current page resolves to slot 0. The same seeds build the same head, prompts
    and teacher tokens on both arms.
    """
    if knob:
        monkeypatch.setenv(KNOB, str(knob))
    else:
        monkeypatch.delenv(KNOB, raising=False)
    e2e._require_cpu_mode()
    root = hybrid._hybrid_root()
    assert (root.mtp is not None) == bool(knob)
    head = torch.randn(
        tiny.STACK_VOCAB_SIZE, int(root.text_config.hidden_size),
        generator=torch.Generator().manual_seed(shadow.SEED),
    )
    head = head - head.mean(dim=1, keepdim=True)
    root.lm_head_weight = torch.nn.Parameter(head.to(torch.bfloat16), requires_grad=False)
    if root.mtp is not None:
        shadow._materialise_head(root, shadow.SEED_HEAD)
    per_request = -(-MAX_MODEL_LEN // PAGE)
    blocks = 1 + len(PROMPTS) * per_request
    caches = e2e._runner_shaped_caches(root)
    for spec in root.get_kv_spec().layers:
        if spec.kda_recurrent_state_shape is None:
            caches[spec.name] = [
                torch.zeros((blocks, *bank.shape[1:]), dtype=bank.dtype)
                for bank in caches[spec.name]
            ]
    root.bind_kv_cache(caches)
    generator = torch.Generator().manual_seed(SEED)
    prompts = [
        torch.randint(0, tiny.STACK_VOCAB_SIZE, (length,), generator=generator).tolist()
        for length in PROMPTS
    ]
    fed = torch.randint(
        0, tiny.STACK_VOCAB_SIZE, (DECODE_STEPS, len(PROMPTS)), generator=generator
    ).tolist()
    req_ids = [f"req-{index}" for index in range(len(PROMPTS))]
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=[])
    runner.requests = {
        req_id: SimpleNamespace(prompt_token_ids=list(prompt), num_prompt_tokens=len(prompt))
        for req_id, prompt in zip(req_ids, prompts)
    }
    runner.model = root
    runner.max_model_len = int(MAX_MODEL_LEN)
    runner.max_num_reqs = len(PROMPTS)
    runner.on_device_sampling = False
    tables = [
        [1 + index * per_request + page for page in range(per_request)]
        for index in range(len(PROMPTS))
    ]
    return SimpleNamespace(
        root=root, runner=runner, caches=caches, tables=tables, window_blocks=per_request,
        req_ids=req_ids, prompts=prompts, fed=fed, depth=len(root.model.layers), knob=knob,
    )


def _forward(world, rows, input_ids, *, cached, sampling, carriers=None):
    """One host-logits step for requests ``rows``: ``(logits, drafts or None)``.

    The drafts come off the output through the runner's own hook, which leaves a knob-0
    root's bare tensor alone, so both arms take one call shape.
    """
    runner = world.runner
    runner.input_batch.req_ids = [world.req_ids[row] for row in rows]
    runner._glm5next_request_tokens = None
    converted = runner._glm5next_model_kwargs({
        "input_ids": torch.tensor(input_ids),
        "positions": None,
        "attn_metadata": batch._metadata(world, rows, cached=cached, tokens=len(input_ids)),
        "sampling_positions": torch.tensor(sampling, dtype=torch.long),
        "sampling_params": None,
        "spec_decode_metadata": None,
        "rank": None,
        "logit_mask": None,
    })
    if carriers is not None:
        carriers.append(converted["layer_carriers"])
    logits = runner._glm5next_shadow_take_output(world.root.forward(**converted))
    assert torch.is_tensor(logits) and logits.dim() == 2, type(logits)
    return logits, getattr(runner, "_glm5next_shadow_last_drafts", None)


def _observe(world, logits, drafts, *, is_prefill: bool, read_back: bool = True) -> None:
    """The runner's shadow glue for one step, in its async form: observe, claim, commit.

    ``read_back=False`` is the step whose output thread never ran: the record is claimed and
    dropped, as a crashed thread would leave it, so the shutdown walks a gap.
    """
    runner = world.runner
    if runner._glm5next_shadow_k() <= 0:
        return
    runner._glm5next_shadow_observe(logits, drafts, is_prefill=is_prefill)
    record = runner._glm5next_shadow_claim()
    assert record is not None, "the kwargs hook stashed no step"
    if read_back:
        runner._glm5next_shadow_commit(record, [[int(one)] for one in logits.argmax(-1).tolist()])


def _trunk_state(world) -> dict[str, torch.Tensor]:
    """Every trunk tensor a step may write, cloned, by name.

    The stack layers' banks (the tensors ``bind_kv_cache`` was handed: the latent bank of a
    sparse layer, the convolution and recurrent banks of the linear layer) and their side
    caches (the indexer's pooled store, tail ring and scratch ring). The draft layer's
    entries, at index ``depth``, are its own and are left out.
    """
    state = {}
    banks = world.root.glm5next_layer_banks
    sides = getattr(world.runner, "_glm5next_side_cache_set", None) or []
    for index, bank in enumerate(banks[: world.depth]):
        for position, tensor in enumerate(world.caches[bank["name"]]):
            state[f"layer {index} {bank['name']}[{position}]"] = tensor.clone()
        if index < len(sides):
            for key, tensor in sides[index].items():
                state[f"layer {index} side {key}"] = tensor.clone()
    return state


def _differences(left: dict, right: dict) -> list[str]:
    return [
        name for name in sorted(set(left) | set(right))
        if name not in left or name not in right or not torch.equal(left[name], right[name])
    ]


def _crossings(carriers, cached: list[int]) -> int:
    """How many requests of this step have a draft position in a page the table does not name.

    Read off the draft layer's own carrier (the last mapping): ``block_table_row`` is
    ``[pages, B]`` with ``-1`` past the request's pages; draft iteration ``i`` stands on
    position ``s + i``.
    """
    table = carriers[-1]["block_table_row"]
    count = 0
    for column, position in enumerate(cached):
        pages = [(position + i) // PAGE for i in range(1, K)]
        if any(page < int(table.shape[0]) and int(table[page, column]) < 0 for page in pages):
            count += 1
    return count


def _run(monkeypatch, knob: int, tmp_path, *, steps: int = DECODE_STEPS, on_world=None):
    """Prefill the three requests, then ``steps`` batched decode steps; a trace per step.

    Each trace entry is ``(label, logits, ids, trunk state)``. The knob-5 arm also counts the
    steps whose draft crossed into an unnamed page, keeps every draft tensor, runs the shadow
    glue with ``LOST_STEP`` never read back and shuts it down. ``on_world`` sees the world
    before any step (the leak test aims at a trunk bank through it).
    """
    world = _world(monkeypatch, knob)
    if on_world is not None:
        on_world(world)
    if knob:
        monkeypatch.setenv(LOG_KNOB, str(tmp_path / f"shadow_k{knob}.jsonl"))
        world.runner.use_async_scheduling = True
    trace, drafts_seen, crossings = [], [], 0

    def record(label, logits):
        trace.append((label, logits.detach().clone(), logits.argmax(-1).tolist(), _trunk_state(world)))

    for index, prompt in enumerate(world.prompts):
        chunks = [(0, prompt)]
        if index == CHUNKED_REQUEST:
            chunks = [(0, prompt[:CHUNK]), (CHUNK, prompt[CHUNK:])]
        for start, chunk in chunks:
            logits, drafts = _forward(
                world, [index], chunk, cached=[start], sampling=[len(chunk) - 1]
            )
            _observe(world, logits, drafts, is_prefill=True)
            record(f"prefill request {index} from {start}", logits)
    assert len(trace) == PREFILL_STEPS
    rows = list(range(len(world.prompts)))
    for step in range(steps):
        cached = [len(prompt) + step for prompt in world.prompts]
        carriers = []
        logits, drafts = _forward(
            world, rows, world.fed[step], cached=cached, sampling=rows, carriers=carriers
        )
        if knob:
            crossings += _crossings(carriers[0], cached)
            drafts_seen.append(drafts.clone())
        _observe(world, logits, drafts, is_prefill=False, read_back=step != LOST_STEP)
        record(f"decode {step}", logits)
    if knob:
        world.runner._glm5next_shadow_shutdown()
    return SimpleNamespace(world=world, trace=trace, drafts=drafts_seen, crossings=crossings)


def _assert_traces_agree(off, on) -> None:
    assert [entry[0] for entry in off.trace] == [entry[0] for entry in on.trace]
    for (label, logits_off, ids_off, state_off), (_, logits_on, ids_on, state_on) in zip(
        off.trace, on.trace
    ):
        assert ids_off == ids_on, f"{label}: greedy ids {ids_off} (knob 0) vs {ids_on} (knob {K})"
        assert torch.equal(logits_off, logits_on), f"{label}: the logits differ"
        differing = _differences(state_off, state_on)
        assert not differing, f"{label}: trunk state differs in {differing}"


# ── the guard ────────────────────────────────────────────────────────────────


def test_the_shadow_draft_leaves_every_trunk_tensor_and_every_logit_bit_equal(
    monkeypatch, tmp_path, caplog
):
    off = _run(monkeypatch, 0, tmp_path)
    with caplog.at_level(logging.WARNING):
        on = _run(monkeypatch, K, tmp_path)
    _assert_traces_agree(off, on)

    # Both state families were compared, and both moved: the KDA layer's two banks and the
    # DSA layers' latent banks are in the set and non-zero at the end.
    final = on.trace[-1][3]
    recurrent = [
        spec.name for spec in on.world.root.get_kv_spec().layers[: on.world.depth]
        if spec.kda_recurrent_state_shape is not None
    ]
    assert recurrent, "the hybrid trunk holds a KDA layer"
    for name in recurrent:
        for position in (0, 1):
            key = next(k for k in final if k.endswith(f"{name}[{position}]"))
            assert float(final[key].detach().float().abs().sum()) > 0.0, key
    assert sum("side pool_cache" in k for k in final) == on.world.depth - len(recurrent)

    # The knob-5 arm did what the server does: a [B, K] draft every decode step, written to
    # the draft layer's own bank, crossing into an unnamed page (slot 0) on every step.
    assert len(on.drafts) == DECODE_STEPS
    for draft in on.drafts:
        assert tuple(draft.shape) == (len(PROMPTS), K) and draft.dtype == torch.int32, draft
        assert all(0 <= value < tiny.STACK_VOCAB_SIZE for value in draft.reshape(-1).tolist())
    assert on.crossings == DECODE_STEPS * len(PROMPTS), on.crossings
    draft_bank = on.world.caches[on.world.root.glm5next_layer_banks[on.world.depth]["name"]][0]
    assert float(draft_bank.detach().float().abs().sum()) > 0.0, "the draft layer's bank holds the steps"
    assert float(draft_bank[NULL_BLOCK_ID].detach().float().abs().sum()) > 0.0, (
        "the unnamed-page fallback wrote the draft layer's slot 0"
    )
    # The glue ran, lost one step, and the shutdown named it.
    log = tmp_path / f"shadow_k{K}.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    assert len(records) >= DECODE_STEPS - 1 and any(record["scored"] > 0 for record in records)
    assert f"step {PREFILL_STEPS + LOST_STEP} never read back" in caplog.text, caplog.text


def test_a_draft_write_into_a_trunk_bank_is_reported_at_its_step(monkeypatch, tmp_path):
    """The comparison sees what it is for: one element of trunk layer 0's first bank, written
    from inside ``draft_tokens`` on the knob-5 arm, fails the first decode step by name."""
    steps = 2
    off = _run(monkeypatch, 0, tmp_path, steps=steps)
    original = Glm5NextMultiTokenPredictor.draft_tokens
    target: dict[str, torch.Tensor] = {}

    def aim(world):
        target["bank"] = world.caches[world.root.glm5next_layer_banks[0]["name"]][0]

    def leaking(self, *args, **kwargs):
        drafts = original(self, *args, **kwargs)
        bank = target["bank"]
        bank[1, 0, 0, 0] = bank[1, 0, 0, 0] + 1
        target["leaked"] = True
        return drafts

    monkeypatch.setattr(Glm5NextMultiTokenPredictor, "draft_tokens", leaking)
    on = _run(monkeypatch, K, tmp_path, steps=steps, on_world=aim)
    assert target.get("leaked"), "the leak never ran"
    with pytest.raises(AssertionError, match=r"decode 0: trunk state differs in \['layer 0 "):
        _assert_traces_agree(off, on)
