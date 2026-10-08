# SPDX-License-Identifier: Apache-2.0
"""The shadow draft (MTP stage A) end to end on the simulator: the real head on the tiny root.

With ``VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT=k`` the GLM-5.3-Flash root builds its draft head
(checkpoint layer 45, ``mtp.Glm5NextMultiTokenPredictor``), ``get_kv_spec`` adds the draft
layer's state, the runner's carrier walk hands the root one more carrier than the stack has
layers, and the root forward: on the prefill leg **populates** the draft layer at every row of
the chunk with the id that row's draft consumes (the next prompt token; the last row of the
prompt's final chunk takes the token this very graph samples); on the decode leg asks the head
for **k draft tokens** from the sampled token and returns them beside the sampled ids, which it
never changes.

The classic MTP bug is an off-by-one in which hidden row or which token id reaches the draft,
so the alignment test pins both at every position, on the real head, by recording the head's
inputs on the way in (a thin recorder that delegates every call):

    prefill  x0 x1 x2 x3          -> populate rows 0..3 with next ids x1, x2, x3, s4 (s4 sampled here)
    decode   s4 at position 4     -> draft from (h_4, s5), s5 sampled here, k drafts out
    decode   s5 at position 5     -> draft from (h_5, s6)
    chunked  [x0..x3] then [x4..x7] -> chunk 1 pairs row 3 with x4 (handed in by the runner, not the
                                    chunk's own, meaningless sample); chunk 2 pairs row 7 with s8

The prompt is four tokens, not three: the tiny stack's DSA indexer refuses a prefill chunk
shorter than its pool size (4), so four is the shortest prefill the trunk accepts; the
alignment pinned is the same (row t pairs with x_{t+1}; the last row with the sampled id).

The head's weights are random but fixed (seeded), the tiny stack's own fixtures materialise
its attention and MoE halves; the four head tensors come from the head's own test module.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/test_shadow_draft_e2e.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.model.glm5_next import model_fp8, mtp
from vllm_neuron.model.neuron_config import OnDeviceSamplingConfig
from vllm_neuron.vllm.worker.neuron_model_runner import NULL_BLOCK_ID, NeuronModelRunner

from test.vllm_neuron.model.glm5_next.test_mtp_draft import HEAD_LEAVES, MLP_LEAVES, _head_weights
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_decode as batch
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The recipe's draft count (mtp.md section 8): the gate measures alpha at k = 5.
K = 5
SEED = 20261007
SEED_HEAD = 63_200_001
PAGE = tiny.MLA_PAGE_SIZE
#: One ``[top_k, top_p, temperature]`` row per request; ``all_greedy`` reads only the shape.
GREEDY_ROW = [-1.0, 1.0, 0.0]
PROMPT = [17, 203, 88, 141]
#: Two chunks of four for the chunked-prefill test.
LONG_PROMPT = [17, 203, 88, 141, 9, 250, 66, 31]
REQ = "req-shadow"
KNOB = "VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT"
#: The stack's five feed-forward keywords the decode leg must thread to the head (contract
#: C3: the MoE half of the draft layer runs through the trunk's own ``_ffn_half``).
FFN_KEYWORDS = ("quant_config", "block_size", "moe_group", "tp_degree", "expert_parallel_rank")


# ── the world: tiny root with the real head, its caches, a runner shell ──────


def _materialise_head(root, seed: int) -> None:
    """Random-but-fixed weights on the root-built head: the four head tensors from the
    head's own test fixture; the block's norms, attention and MoE half the tiny stack's way,
    as one more layer of the same stack (gain sites continue the stack's numbering)."""
    head = root.mtp
    cfg = root.text_config
    block = head.block
    weights = _head_weights(cfg, seed)
    for name in HEAD_LEAVES:
        setattr(head, name, torch.nn.Parameter(weights[name].clone(), requires_grad=False))
    depth = len(root.model.layers)
    block.input_layernorm_weight = torch.nn.Parameter(tiny._stack_gain(2 * depth), requires_grad=False)
    block.post_attention_layernorm_weight = torch.nn.Parameter(
        tiny._stack_gain(2 * depth + 1), requires_grad=False
    )
    tiny._materialise_mla_attention(
        block.self_attn, cfg, seed=seed, output_damping=tiny.STACK_ATTENTION_DAMPING
    )
    operands = tiny._stack_bank_operands()
    for leaf in MLP_LEAVES:
        tiny._attach(block.mlp.experts, leaf, *operands[leaf])
    built = block.mlp.experts.prepare_scale_operands(
        *tiny._prep_operands_from_the_module(block.mlp.experts, MLP_LEAVES, operands)
    )
    assert built == 2, built
    generator = torch.Generator().manual_seed(seed)
    block.mlp.experts.router_weight = torch.nn.Parameter(
        (torch.randn(tiny.STACK_HIDDEN_SIZE, tiny.STACK_EXPERTS, generator=generator)
         * tiny.MOE_ROUTER_WEIGHT_SCALE).to(torch.bfloat16),
        requires_grad=False,
    )
    block.mlp.experts.router_bias = torch.nn.Parameter(
        (torch.randn(tiny.STACK_EXPERTS, generator=generator) * tiny.MOE_ROUTER_BIAS_SCALE).to(torch.bfloat16),
        requires_grad=False,
    )


def _world(*, max_model_len: int = tiny.STACK_TOKENS + 8, prompt: list[int] = PROMPT,
           head_seed: int = SEED_HEAD, **config_overrides):
    """The tiny root with device sampling on, its caches, a runner shell; nothing prefilled.

    The knob is read at construction: with it on, the root builds ``mtp`` (materialised here
    from ``head_seed``) and ``get_kv_spec`` lists the draft layer, so the caches below and
    the runner's carriers include it without any help from this file.
    """
    e2e._require_cpu_mode()
    fixture = e2e._fixture(**config_overrides)
    root = fixture["root"]
    head = torch.randn(tiny.STACK_VOCAB_SIZE, int(root.text_config.hidden_size),
                       generator=torch.Generator().manual_seed(SEED))
    # Zero-mean rows: the tiny regime's hidden states ride on a large common offset (the
    # trunk's rows average ~1.2, the head's ~1.0 after its norm), and a head whose rows sum
    # to different values would read that offset, not the direction, making every greedy
    # token the row with the largest sum whatever the weights upstream did.
    head = head - head.mean(dim=1, keepdim=True)
    root.lm_head_weight = torch.nn.Parameter(head.to(torch.bfloat16), requires_grad=False)
    root.text_config.neuron_config = SimpleNamespace(
        on_device_sampling_config=OnDeviceSamplingConfig(all_greedy=True)
    )
    if root.mtp is not None:
        _materialise_head(root, head_seed)
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
          mutate=None):
    """One step through the converter and the root with device sampling.

    ``seen`` collects the converted kwargs; ``mutate`` is applied to them before the forward
    (to hand the root a malformed operand). Returns the root's output.
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
    if seen is not None:
        seen.append(converted)
    if mutate is not None:
        mutate(converted)
    return world.root.forward(**converted)


def _ids(output) -> list[int]:
    primary = output[0] if isinstance(output, tuple) else output
    return primary.to(torch.int32).reshape(-1).tolist()


class _Recorder:
    """Record every input the real head receives; delegate every call to it."""

    def __init__(self, head) -> None:
        # The bound methods, taken before the recorder shadows them on the instance.
        self._populate = head.populate
        self._draft_tokens = head.draft_tokens
        self.calls: list[tuple[str, dict]] = []

    def populate(self, hidden_rows, next_ids, positions, **kwargs):
        self.calls.append(("populate", {
            "hidden_rows": hidden_rows.detach().clone(), "next_ids": next_ids.detach().clone(),
            "positions": positions.detach().clone(), "kwargs": dict(kwargs),
        }))
        return self._populate(hidden_rows, next_ids, positions, **kwargs)

    def draft_tokens(self, hidden_rows, sampled_ids, positions, k, **kwargs):
        self.calls.append(("draft", {
            "hidden_rows": hidden_rows.detach().clone(), "sampled_ids": sampled_ids.detach().clone(),
            "positions": positions.detach().clone(), "k": int(k), "kwargs": dict(kwargs),
        }))
        return self._draft_tokens(hidden_rows, sampled_ids, positions, k, **kwargs)

    def of(self, kind: str) -> list[dict]:
        return [call for name, call in self.calls if name == kind]


def _record(monkeypatch, world) -> _Recorder:
    """Put the recorder in front of the root's real head (bound-method shadowing)."""
    recorder = _Recorder(world.root.mtp)
    monkeypatch.setattr(world.root.mtp, "populate", recorder.populate)
    monkeypatch.setattr(world.root.mtp, "draft_tokens", recorder.draft_tokens)
    return recorder


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


def _draft_layer_latent_rows(world, positions: list[int]) -> torch.Tensor:
    """The draft layer's latent-cache rows at ``positions`` of request 0, ``[n, head_size]``.

    Read off the runner-shaped cache tensor (``[blocks, heads, page, head_size]``) the bank
    is a view of: position ``p`` lives in the request's ``p // PAGE``-th block, row ``p % PAGE``.
    """
    depth = len(world.root.model.layers)
    cache = world.caches[world.root.glm5next_layer_banks[depth]["name"]][0]
    rows = [cache[world.tables[0][p // PAGE], 0, p % PAGE] for p in positions]
    return torch.stack(rows)


# ── construction: the knob builds the head, the spec and the carrier ─────────


def test_the_knob_builds_the_real_head_and_the_runner_hands_it_its_own_carrier(monkeypatch):
    monkeypatch.setenv(KNOB, str(K))
    world = _world()
    assert isinstance(world.root.mtp, mtp.Glm5NextMultiTokenPredictor)
    # The checkpoint's default: the k iterations share the first one's index selection
    # (``index_share_for_mtp_iteration``); every drafting test in this file runs that path.
    assert world.root.text_config.index_share_for_mtp_iteration is True
    depth = len(world.root.model.layers)
    assert len(world.root.get_kv_spec().layers) == depth + 1
    recorder = _record(monkeypatch, world)
    seen: list[dict] = []
    _step(world, PROMPT, cached=0, sampling=[3], seen=seen)
    carriers = seen[-1]["layer_carriers"]
    assert len(carriers) == depth + 1, "the runner's walk adds the draft layer's carrier"
    populate = recorder.of("populate")[0]
    # The head's block kwargs are the last carrier, object for object: layer 45's own state.
    assert populate["kwargs"]["pool_cache"] is carriers[-1]["pool_cache"]
    assert populate["kwargs"]["prefill_tail"] is carriers[-1]["prefill_tail"]
    assert populate["kwargs"]["latent_cache"] is carriers[-1]["latent_cache"]
    assert carriers[-1]["latent_cache"] is not carriers[-2]["latent_cache"]

    # Without the extra carrier the root refuses rather than running the draft on a trunk layer.
    def drop_last(converted):
        converted["layer_carriers"] = list(converted["layer_carriers"])[:-1]

    with pytest.raises(ValueError, match="carrier"):
        _step(world, [5], cached=4, sampling=[0], mutate=drop_last)


def test_k_is_read_through_the_heads_reader_only(monkeypatch):
    """Contract C1: ``mtp.shadow_draft_k`` is the one reader; the root and the runner never
    read the environment themselves."""
    monkeypatch.setenv(KNOB, str(K))
    monkeypatch.setattr(mtp, "shadow_draft_k", lambda: 0)
    world = _world()
    assert world.root.mtp is None
    assert world.runner._glm5next_shadow_k() == 0
    out = _step(world, PROMPT, cached=0, sampling=[3])
    assert torch.is_tensor(out), "k = 0 through the reader: the bare sampled ids"


# ── the alignment test on the real head ──────────────────────────────────────


def test_four_tokens_pin_which_row_and_which_id_reach_the_draft(monkeypatch, logits_seen):
    monkeypatch.setenv(KNOB, str(K))
    world = _world()
    recorder = _record(monkeypatch, world)
    vocab = tiny.STACK_VOCAB_SIZE

    # Prefill x0 x1 x2 x3 in one chunk; the graph samples s4.
    out = _step(world, PROMPT, cached=0, sampling=[3])
    assert isinstance(out, tuple) and len(out) == 2, type(out)
    sampled, drafts = out
    assert sampled.dtype == torch.int32 and tuple(sampled.shape) == (1,)
    s4 = int(sampled[0])
    assert tuple(drafts.shape) == (1, K) and drafts.dtype == torch.int32
    assert drafts.tolist() == [[-1] * K], "the prefill leg drafts nothing: sentinel rows"
    assert [name for name, _ in recorder.calls] == ["populate"]
    populate = recorder.of("populate")[0]
    assert populate["positions"].tolist() == [0, 1, 2, 3]
    assert populate["positions"].dtype == torch.int32
    assert populate["next_ids"].tolist() == [PROMPT[1], PROMPT[2], PROMPT[3], s4]
    assert populate["next_ids"].dtype == torch.int32
    assert tuple(populate["hidden_rows"].shape) == (4, int(world.root.text_config.hidden_size))
    _assert_row_is_what_the_head_projected(populate["hidden_rows"][3], world.head, logits_seen[-1][0])
    assert "prefill_tail" in populate["kwargs"] and "tail" not in populate["kwargs"]
    assert all(name in populate["kwargs"] for name in FFN_KEYWORDS), sorted(populate["kwargs"])

    # Decode s4 at position 4: the graph samples s5 and drafts from (h_4, s5).
    out = _step(world, [s4], cached=4, sampling=[0])
    sampled, drafts = out
    s5 = int(sampled[0])
    draft = recorder.of("draft")
    assert len(draft) == 1 and len(recorder.of("populate")) == 1
    assert draft[0]["positions"].tolist() == [4]
    assert draft[0]["sampled_ids"].tolist() == [s5]
    assert draft[0]["sampled_ids"].dtype == torch.int32
    assert draft[0]["k"] == K
    assert tuple(draft[0]["hidden_rows"].shape) == (1, int(world.root.text_config.hidden_size))
    _assert_row_is_what_the_head_projected(draft[0]["hidden_rows"][0], world.head, logits_seen[-1][0])
    kwargs = draft[0]["kwargs"]
    assert "tail" in kwargs and "position" in kwargs and "prefill_tail" not in kwargs
    # The stack's five feed-forward keywords reach the head (its MoE half runs the trunk's
    # ``_ffn_half``); the quantisation policy is the one object the root resolved.
    assert all(name in kwargs for name in FFN_KEYWORDS), sorted(kwargs)
    assert kwargs["quant_config"] is not None
    assert tuple(drafts.shape) == (1, K) and drafts.dtype == torch.int32
    assert all(0 <= value < vocab for value in drafts.reshape(-1).tolist())

    # Decode s5 at position 5: drafts from (h_5, s6).
    out = _step(world, [s5], cached=5, sampling=[0])
    sampled, drafts = out
    s6 = int(sampled[0])
    draft = recorder.of("draft")
    assert len(draft) == 2
    assert draft[1]["positions"].tolist() == [5]
    assert draft[1]["sampled_ids"].tolist() == [s6]
    _assert_row_is_what_the_head_projected(draft[1]["hidden_rows"][0], world.head, logits_seen[-1][0])


def test_chunked_prefill_pairs_the_chunk_boundary_row_with_the_next_prompt_token(monkeypatch):
    monkeypatch.setenv(KNOB, str(K))
    world = _world(prompt=LONG_PROMPT)
    recorder = _record(monkeypatch, world)
    # Chunk 1: x0..x3, not the prompt's end. Its sample is meaningless and must not reach the
    # draft: row 3 pairs with x4, the first token of the next chunk, handed in by the runner.
    out = _step(world, LONG_PROMPT[:4], cached=0, sampling=[3])
    garbage = _ids(out)[0]
    populate = recorder.of("populate")
    assert len(populate) == 1
    assert populate[0]["positions"].tolist() == [0, 1, 2, 3]
    assert populate[0]["next_ids"].tolist() == LONG_PROMPT[1:5]
    if garbage != LONG_PROMPT[4]:
        assert populate[0]["next_ids"][3].item() != garbage
    # Chunk 2: x4..x7 at positions 4..7, the prompt's end: row 7 takes the sampled s8.
    out = _step(world, LONG_PROMPT[4:], cached=4, sampling=[3])
    s8 = _ids(out)[0]
    populate = recorder.of("populate")
    assert len(populate) == 2
    assert populate[1]["positions"].tolist() == [4, 5, 6, 7]
    assert populate[1]["next_ids"].tolist() == LONG_PROMPT[5:] + [s8]
    assert recorder.of("draft") == []
    # The draft layer's own state holds every prefilled position and nothing past them.
    written = _draft_layer_latent_rows(world, list(range(len(LONG_PROMPT))))
    assert bool((written.float().abs().sum(dim=-1) > 0).all()), "rows 0..7 populated"
    beyond = _draft_layer_latent_rows(world, [len(LONG_PROMPT), len(LONG_PROMPT) + 1])
    assert float(beyond.float().abs().sum()) == 0.0, "nothing past the prompt"


# ── lossless, non-degenerate, the recipe's shapes ────────────────────────────


def _run(monkeypatch, knob: int, *, head_seed: int = SEED_HEAD, steps: int = 3):
    """Two-chunk prefill of ``LONG_PROMPT`` then ``steps`` greedy decode steps.

    Returns the sampled ids in order and the ``[rows, k]`` drafts of every decode step.
    """
    if knob:
        monkeypatch.setenv(KNOB, str(knob))
    else:
        monkeypatch.delenv(KNOB, raising=False)
    world = _world(prompt=LONG_PROMPT, head_seed=head_seed)
    assert (world.root.mtp is not None) == bool(knob)
    _step(world, LONG_PROMPT[:4], cached=0, sampling=[3])
    out = _step(world, LONG_PROMPT[4:], cached=4, sampling=[3])
    if not knob:
        assert torch.is_tensor(out), type(out)
    ids = [_ids(out)[0]]
    drafts = []
    for step in range(steps):
        out = _step(world, [ids[-1]], cached=len(LONG_PROMPT) + step, sampling=[0])
        ids.append(_ids(out)[0])
        if knob:
            assert isinstance(out, tuple) and len(out) == 2
            drafts.append(out[1].clone())
    return ids, drafts, world


def test_the_sampled_ids_do_not_depend_on_the_draft(monkeypatch):
    """Contract C6: knob 0 and knob 5 sample the same ids; knob 0 returns the bare tensor."""
    off, _, _ = _run(monkeypatch, 0)
    on, drafts, _ = _run(monkeypatch, K)
    assert on == off
    assert len(drafts) == 3
    for draft in drafts:
        assert tuple(draft.shape) == (1, K) and draft.dtype == torch.int32
        assert all(0 <= value < tiny.STACK_VOCAB_SIZE for value in draft.reshape(-1).tolist())


def test_the_drafts_follow_the_draft_layers_weights(monkeypatch):
    """The drafts are the head's: the same seed reproduces them, another seed on layer 45
    changes them while the trunk (and so the sampled ids) stays the same."""
    ids_a, drafts_a, _ = _run(monkeypatch, K, head_seed=SEED_HEAD)
    ids_b, drafts_b, _ = _run(monkeypatch, K, head_seed=SEED_HEAD)
    assert ids_a == ids_b
    assert all(torch.equal(a, b) for a, b in zip(drafts_a, drafts_b))
    ids_c, drafts_c, _ = _run(monkeypatch, K, head_seed=SEED_HEAD + 1)
    assert ids_c == ids_a, "layer 45's weights never reach the sampled ids"
    assert any(not torch.equal(a, c) for a, c in zip(drafts_a, drafts_c)), "degenerate drafts"


def test_a_decode_step_advances_the_draft_layers_state_and_touches_nothing_past_its_k_positions(monkeypatch):
    """The k iterations stand on positions ``s .. s + k - 1`` (iteration 0 populates the
    request's own position, the rest write in place and the next real step overwrites them;
    which of them reach the latent bank is the DSA pooling's business, pinned in the head's
    own suite). What the shadow path owes the trunk: nothing at or past ``s + k`` is written,
    and the draft layer's own ring holds the step (it advanced, on its own side cache)."""
    _, _, world = _run(monkeypatch, K, steps=1)
    position = len(LONG_PROMPT)          # the one decode step consumed s8 at position 8
    beyond = _draft_layer_latent_rows(world, [position + K, position + K + 1])
    assert float(beyond.float().abs().sum()) == 0.0
    depth = len(world.root.model.layers)
    ring = world.runner._glm5next_side_cache_set[depth]["tail"][0]
    assert float(ring.float().abs().sum()) > 0.0, "the draft layer's ring holds the step"
    trunk_ring = world.runner._glm5next_side_cache_set[depth - 1]["tail"][0]
    assert ring.data_ptr() != trunk_ring.data_ptr(), "its own side cache, not the trunk's"


# ── refusals by name ─────────────────────────────────────────────────────────


def test_the_root_refuses_a_draft_carrier_that_names_no_leg(monkeypatch):
    """The leg is read off the draft carrier (``prefill_tail`` = prefill, ``position`` =
    decode); a carrier with neither is a protocol disagreement with the runner's carrier walk
    and is refused by name, not by a KeyError."""
    monkeypatch.setenv(KNOB, str(K))
    world = _world()

    def strip_leg(converted):
        carrier = converted["layer_carriers"][-1]
        for key in ("prefill_tail", "tail", "position", "prefill_end_position"):
            carrier.pop(key, None)

    with pytest.raises(ValueError, match="prefill_tail.*position|position.*prefill_tail"):
        _step(world, PROMPT, cached=0, sampling=[3], mutate=strip_leg)


def test_the_root_refuses_a_boundary_tensor_that_does_not_match_the_sampling_rows(monkeypatch):
    """One boundary id per sampling row: a count mismatch would make ``index_copy`` fail
    deep inside the trace, or silently pair rows with the wrong id."""
    monkeypatch.setenv(KNOB, str(K))
    world = _world()

    def two_boundaries(converted):
        converted["shadow_boundary_ids"] = torch.tensor([-1, -1], dtype=torch.int32)

    with pytest.raises(ValueError, match="sampling"):
        _step(world, PROMPT, cached=0, sampling=[3], mutate=two_boundaries)
