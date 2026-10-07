# SPDX-License-Identifier: Apache-2.0
"""The shadow draft (MTP stage A) wired into the GLM-5.3-Flash root: alignment and losslessness.

With ``VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT=k`` the root forward hands the layer-45 draft head
the trunk's post-final-norm hidden rows: on the prefill leg it **populates** every row of the
chunk with the id that row's draft consumes (the next prompt token; the last row of the
prompt's final chunk takes the token this very graph samples), and on the decode leg it asks
for **k draft tokens** from the sampled token and returns them beside the sampled ids, which
it never changes.

The classic MTP bug is an off-by-one in which hidden row or which token id reaches the draft,
so the three-token test pins both at every position, on the simulator, with a stand-in head
that records its inputs and returns deterministic drafts (worker-38's head replaces it at
integration; the second half of this file reads the real head):

    prefill  x0 x1 x2 x3          -> populate rows 0..3 with next ids x1, x2, x3, s4 (s4 sampled here)
    decode   s4 at position 4     -> draft from (h_4, s5), s5 sampled here, k drafts out
    decode   s5 at position 5     -> draft from (h_5, s6)
    chunked  [x0..x3] then [x4..x7] -> chunk 1 pairs row 3 with x4 (handed in by the runner, not the
                                    chunk's own, meaningless sample); chunk 2 pairs row 7 with s8

The prompt is four tokens, not three: the tiny stack's DSA indexer refuses a prefill chunk
shorter than its pool size (4), so four is the shortest prefill the trunk accepts; the
alignment pinned is the same (row t pairs with x_{t+1}; the last row with the sampled id).

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/test_shadow_draft_e2e.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.model.glm5_next import model_fp8
from vllm_neuron.model.neuron_config import OnDeviceSamplingConfig
from vllm_neuron.vllm.worker.neuron_model_runner import NULL_BLOCK_ID, NeuronModelRunner

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_decode as batch
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny

pytestmark = [pytest.mark.fast, pytest.mark.forked]

K = 3
SEED = 20261007
PAGE = tiny.MLA_PAGE_SIZE
#: One ``[top_k, top_p, temperature]`` row per request; ``all_greedy`` reads only the shape.
GREEDY_ROW = [-1.0, 1.0, 0.0]
PROMPT = [17, 203, 88, 141]
#: Two chunks of four for the chunked-prefill test.
LONG_PROMPT = [17, 203, 88, 141, 9, 250, 66, 31]
REQ = "req-shadow"
KNOB = "VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT"


# ── the stand-in head ────────────────────────────────────────────────────────


def stub_drafts(sampled_ids: torch.Tensor, positions: torch.Tensor, k: int, vocab: int) -> torch.Tensor:
    """Deterministic fake drafts: a function of the sampled id and the position only."""
    offsets = 7 * (torch.arange(k, dtype=torch.int32) + 1)
    base = sampled_ids.to(torch.int32)[:, None] + positions.to(torch.int32)[:, None]
    return ((base + offsets[None, :]) % vocab).to(torch.int32)


class RecordingStubHead(torch.nn.Module):
    """Contract C3's two methods, recording every input; stands in for worker-38's head."""

    def __init__(self, vocab: int) -> None:
        super().__init__()
        self.vocab = int(vocab)
        self.calls: list[tuple[str, dict]] = []

    def populate(self, hidden_rows, next_ids, positions, **block_kwargs) -> None:
        self.calls.append(("populate", {
            "hidden_rows": hidden_rows.detach().clone(), "next_ids": next_ids.detach().clone(),
            "positions": positions.detach().clone(), "block_kwargs": dict(block_kwargs),
        }))

    def draft_tokens(self, hidden_rows, sampled_ids, positions, k, **block_kwargs):
        self.calls.append(("draft", {
            "hidden_rows": hidden_rows.detach().clone(), "sampled_ids": sampled_ids.detach().clone(),
            "positions": positions.detach().clone(), "k": int(k), "block_kwargs": dict(block_kwargs),
        }))
        return stub_drafts(sampled_ids, positions, k, self.vocab)

    def of(self, kind: str) -> list[dict]:
        return [call for name, call in self.calls if name == kind]


# ── a one-request world without a prefill ───────────────────────────────────


def _world(*, max_model_len: int = tiny.STACK_TOKENS + 8, prompt: list[int] = PROMPT):
    """The tiny root with device sampling on, its caches, a runner shell; nothing prefilled."""
    e2e._require_cpu_mode()
    fixture = e2e._fixture()
    root = fixture["root"]
    head = torch.randn(tiny.STACK_VOCAB_SIZE, int(root.text_config.hidden_size),
                       generator=torch.Generator().manual_seed(SEED))
    root.lm_head_weight = torch.nn.Parameter(head.to(torch.bfloat16), requires_grad=False)
    root.text_config.neuron_config = SimpleNamespace(
        on_device_sampling_config=OnDeviceSamplingConfig(all_greedy=True)
    )
    per_request = -(-max_model_len // PAGE)
    blocks = 1 + per_request
    caches = {}
    for spec in root.get_kv_spec().layers:
        shape = (blocks, int(spec.num_kv_heads), PAGE, int(spec.head_size))
        caches[spec.name] = [torch.zeros(shape, dtype=spec.dtype) for _ in range(1 if spec.latent_kv else 2)]
    root.bind_kv_cache(caches)
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=[REQ])
    runner.requests = {REQ: SimpleNamespace(prompt_token_ids=list(prompt), num_prompt_tokens=len(prompt))}
    runner.model = root
    runner.max_model_len = int(max_model_len)
    runner.max_num_reqs = 1
    runner.on_device_sampling = True
    tables = [[1 + k for k in range(per_request)] + [NULL_BLOCK_ID] * 0]
    return SimpleNamespace(root=root, runner=runner, caches=caches, tables=tables,
                           window_blocks=per_request, req_ids=[REQ], head=head.to(torch.bfloat16))


def _step(world, input_ids, *, cached: int, sampling: list[int], seen: list | None = None,
          layer45: bool = True):
    """One step through the converter and the root with device sampling.

    ``layer45``: append a stand-in carrier for layer 45 (a copy of the last DSA layer's), the
    entry worker-40's carrier walk adds when the knob is on. Returns the root's output.
    """
    runner = world.runner
    runner.input_batch.req_ids = list(world.req_ids)
    runner._glm5next_request_tokens = None
    converted = runner._glm5next_model_kwargs({
        "input_ids": torch.tensor(input_ids),
        "positions": None,
        "attn_metadata": batch._metadata(world, [0], cached=[cached], tokens=len(input_ids)),
        "sampling_positions": torch.tensor(sampling, dtype=torch.long),
        "sampling_params": torch.tensor([GREEDY_ROW] * len(sampling), dtype=torch.float32),
        "spec_decode_metadata": None,
        "rank": None,
        "logit_mask": None,
    })
    assert "device_sampling_params" in converted, sorted(converted)
    if layer45:
        converted["layer_carriers"] = list(converted["layer_carriers"]) + [dict(converted["layer_carriers"][-1])]
    if seen is not None:
        seen.append(converted)
    return world.root.forward(**converted)


def _ids(output) -> list[int]:
    primary = output[0] if isinstance(output, tuple) else output
    return primary.to(torch.int32).reshape(-1).tolist()


@pytest.fixture
def logits_seen(monkeypatch):
    """The logits the device sampler saw, per call (the rows the head projected)."""
    seen: list[torch.Tensor] = []
    sampler = model_fp8.sample_full_vocab

    def recorded(logits, *args, **kwargs):
        seen.append(logits.detach().clone())
        return sampler(logits, *args, **kwargs)

    monkeypatch.setattr(model_fp8, "sample_full_vocab", recorded)
    return seen


def _assert_row_is_what_the_head_projected(hidden_row: torch.Tensor, head: torch.Tensor, logits: torch.Tensor):
    """The recorded hidden row, projected by the head, is the logits row the sampler saw."""
    projected = torch.nn.functional.linear(hidden_row.to(head.dtype)[None, :], head).float()
    torch.testing.assert_close(projected[0], logits.float(), rtol=0, atol=0)


# ── the three-token alignment test (stand-in head) ───────────────────────────


def test_four_tokens_pin_which_row_and_which_id_reach_the_draft(monkeypatch, logits_seen):
    monkeypatch.setenv(KNOB, str(K))
    world = _world()
    stub = RecordingStubHead(tiny.STACK_VOCAB_SIZE)
    world.root.mtp = stub

    # Prefill x0 x1 x2 x3 in one chunk; the graph samples s4.
    out = _step(world, PROMPT, cached=0, sampling=[3])
    assert isinstance(out, tuple) and len(out) == 2, type(out)
    sampled, drafts = out
    assert sampled.dtype == torch.int32 and tuple(sampled.shape) == (1,)
    s4 = int(sampled[0])
    assert tuple(drafts.shape) == (1, K) and drafts.dtype == torch.int32
    assert drafts.tolist() == [[-1] * K], "the prefill leg drafts nothing: sentinel rows"
    assert [name for name, _ in stub.calls] == ["populate"]
    populate = stub.of("populate")[0]
    assert populate["positions"].tolist() == [0, 1, 2, 3]
    assert populate["positions"].dtype == torch.int32
    assert populate["next_ids"].tolist() == [PROMPT[1], PROMPT[2], PROMPT[3], s4]
    assert populate["next_ids"].dtype == torch.int32
    assert tuple(populate["hidden_rows"].shape) == (4, int(world.root.text_config.hidden_size))
    _assert_row_is_what_the_head_projected(populate["hidden_rows"][3], world.head, logits_seen[-1][0])
    assert "prefill_tail" in populate["block_kwargs"] and "tail" not in populate["block_kwargs"]

    # Decode s4 at position 4: the graph samples s5 and drafts from (h_4, s5).
    out = _step(world, [s4], cached=4, sampling=[0])
    sampled, drafts = out
    s5 = int(sampled[0])
    draft = stub.of("draft")
    assert len(draft) == 1 and len(stub.of("populate")) == 1
    assert draft[0]["positions"].tolist() == [4]
    assert draft[0]["sampled_ids"].tolist() == [s5]
    assert draft[0]["sampled_ids"].dtype == torch.int32
    assert draft[0]["k"] == K
    assert tuple(draft[0]["hidden_rows"].shape) == (1, int(world.root.text_config.hidden_size))
    _assert_row_is_what_the_head_projected(draft[0]["hidden_rows"][0], world.head, logits_seen[-1][0])
    assert "tail" in draft[0]["block_kwargs"] and "prefill_tail" not in draft[0]["block_kwargs"]
    assert torch.equal(drafts, stub_drafts(torch.tensor([s5]), torch.tensor([4]), K, tiny.STACK_VOCAB_SIZE))

    # Decode s5 at position 5: drafts from (h_5, s6).
    out = _step(world, [s5], cached=5, sampling=[0])
    sampled, drafts = out
    s6 = int(sampled[0])
    draft = stub.of("draft")
    assert len(draft) == 2
    assert draft[1]["positions"].tolist() == [5]
    assert draft[1]["sampled_ids"].tolist() == [s6]
    _assert_row_is_what_the_head_projected(draft[1]["hidden_rows"][0], world.head, logits_seen[-1][0])
    assert torch.equal(drafts, stub_drafts(torch.tensor([s6]), torch.tensor([5]), K, tiny.STACK_VOCAB_SIZE))


def test_the_draft_receives_the_layer_45_carrier_and_the_stack_its_own(monkeypatch):
    monkeypatch.setenv(KNOB, str(K))
    world = _world()
    stub = RecordingStubHead(tiny.STACK_VOCAB_SIZE)
    world.root.mtp = stub
    seen: list[dict] = []
    _step(world, PROMPT, cached=0, sampling=[3], seen=seen)
    carriers = seen[-1]["layer_carriers"]
    assert len(carriers) == len(world.root.model.layers) + 1
    populate = stub.of("populate")[0]
    # The draft's block kwargs are the appended entry (index = stack depth), object for object.
    assert populate["block_kwargs"]["pool_cache"] is carriers[-1]["pool_cache"]
    assert populate["block_kwargs"]["prefill_tail"] is carriers[-1]["prefill_tail"]
    # Without the extra carrier the root refuses rather than running the draft on a trunk layer.
    with pytest.raises(ValueError, match="carrier"):
        _step(world, [5], cached=4, sampling=[0], layer45=False)


def test_chunked_prefill_pairs_the_chunk_boundary_row_with_the_next_prompt_token(monkeypatch):
    monkeypatch.setenv(KNOB, str(K))
    world = _world(prompt=LONG_PROMPT)
    stub = RecordingStubHead(tiny.STACK_VOCAB_SIZE)
    world.root.mtp = stub
    # Chunk 1: x0..x3, not the prompt's end. Its sample is meaningless and must not reach the
    # draft: row 3 pairs with x4, the first token of the next chunk, handed in by the runner.
    out = _step(world, LONG_PROMPT[:4], cached=0, sampling=[3])
    garbage = _ids(out)[0]
    populate = stub.of("populate")
    assert len(populate) == 1
    assert populate[0]["positions"].tolist() == [0, 1, 2, 3]
    assert populate[0]["next_ids"].tolist() == LONG_PROMPT[1:5]
    if garbage != LONG_PROMPT[4]:
        assert populate[0]["next_ids"][3].item() != garbage
    # Chunk 2: x4..x7 at positions 4..7, the prompt's end: row 7 takes the sampled s8.
    out = _step(world, LONG_PROMPT[4:], cached=4, sampling=[3])
    s8 = _ids(out)[0]
    populate = stub.of("populate")
    assert len(populate) == 2
    assert populate[1]["positions"].tolist() == [4, 5, 6, 7]
    assert populate[1]["next_ids"].tolist() == LONG_PROMPT[5:] + [s8]
    assert stub.of("draft") == []


def test_the_sampled_ids_do_not_depend_on_the_draft(monkeypatch):
    """Contract C6: knob 0 and knob k sample the same ids; knob 0 returns the bare tensor."""

    def run(knob: int) -> tuple[list[int], RecordingStubHead]:
        if knob:
            monkeypatch.setenv(KNOB, str(knob))
        else:
            monkeypatch.delenv(KNOB, raising=False)
        world = _world()                      # a fresh, identically seeded world per arm
        stub = RecordingStubHead(tiny.STACK_VOCAB_SIZE)
        world.root.mtp = stub
        out = _step(world, PROMPT, cached=0, sampling=[3], layer45=bool(knob))
        if not knob:
            assert torch.is_tensor(out), type(out)
        ids = [_ids(out)[0]]
        for step in range(2):
            out = _step(world, [ids[-1]], cached=4 + step, sampling=[0], layer45=bool(knob))
            ids.append(_ids(out)[0])
        return ids, stub

    off, stub_off = run(0)
    assert stub_off.calls == []
    on, stub_on = run(K)
    assert on == off
    assert len(stub_on.of("draft")) == 2 and len(stub_on.of("populate")) == 1
