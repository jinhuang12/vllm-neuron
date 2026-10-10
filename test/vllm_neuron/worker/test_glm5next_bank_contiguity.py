# SPDX-License-Identifier: Apache-2.0
"""Every tensor the bank form hands the decode graph is contiguous (gate finding, 11:34Z).

The Neuron executor refuses a graph input that is not a contiguous slice of its storage
("Detected non-contiguous slicing for requested Device Tensor", ``libtorchneuron.so``,
``csrc/neuron_op/storage.h``). The bank form hands each recurrent (KDA) layer its two
WHOLE state banks, so the banks themselves, not just their rows, must be contiguous. An
earlier KV allocation built each state as a slot-strided ``torch.as_strided`` view of
one raw buffer (both states side by side inside every slot): one row ``bank[slot]`` was
contiguous, so the per-request view form ran, and the whole bank was not, so the first
bank-form warmup (b2/s2048) failed on device. CPU tensors accept any strides, so no CPU
test saw it; these two read the strides directly.

1. The allocation: on the 45-layer hybrid stack, every recurrent layer's two state banks
   are contiguous, each row is still contiguous (the view form), both live in that
   layer's one raw buffer as disjoint byte regions, and the buffer is still exactly
   ``slots x recurrent_state_slot_bytes`` (the footprint the worker budgets).
2. The hand-off, on the bs=64 serve line (max_model_len 8192, 64 slots, hybrid block 128,
   decode context bucket 2048) through the real runner's ``warmup_decode``: for every
   batch bucket ``B`` in ``[1, 2, 4, 8, 16, 32, 64]`` every tensor leaf of the kwargs the
   runner hands the model is ``is_contiguous()``; the step is the bank form at ``B >= 2``
   (so the assertion has power) and the view form at ``B = 1``.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_glm5next_bank_contiguity.py
"""

from __future__ import annotations

import json
import pathlib

import pytest
import torch

from test.vllm_neuron.worker.test_initialize_kv_cache_hybrid import (
    NUM_BLOCKS_FULL,
    STATE_SLOTS,
    _call,
    _config,
    _drive,
    _fake_layers,
    _raw_fixture,
    _split,
)

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The bs=64 serve line's batch buckets (``num_seqs_buckets``).
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
#: The bs=64 serve line, as ``test/perf/host_kwargs_bs64.py`` builds it.
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 64
DECODE_CTX_BUCKET = 2048
PREFILL_BUCKET = 1024
BLOCK = 128
VOCAB = 154880
KV_BYTES = 3 * 2**30
FIXTURE = pathlib.Path("test/vllm_neuron/model/glm5_next/fixtures")


def _tensor_leaves(value, path=""):
    """``(path, tensor)`` for every tensor leaf of a kwargs tree."""
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _tensor_leaves(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _tensor_leaves(item, f"{path}.{key}" if path else str(key))


def _span(tensor: torch.Tensor) -> tuple[int, int]:
    """The byte range ``[start, end)`` of a contiguous tensor inside its storage."""
    start = tensor.storage_offset() * tensor.element_size()
    return start, start + tensor.numel() * tensor.element_size()


# ── 1. the allocation ─────────────────────────────────────────────────────────


def test_the_recurrent_allocation_gives_contiguous_banks_in_one_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_neuron.vllm.patches.kv_spec_patch import recurrent_state_slot_bytes

    raw = _raw_fixture()
    layers = _fake_layers(raw)
    specs = _call(layers)
    kda_names, _ = _split(specs)
    caches = _drive(_config(specs, NUM_BLOCKS_FULL), layers, monkeypatch)

    strided = []
    for name in kda_names:
        conv, recurrent = caches[name]
        spec = specs[name]
        slot_bytes = recurrent_state_slot_bytes(spec)
        for bank, shape, dtype in zip((conv, recurrent), spec.shapes, spec.dtypes):
            assert tuple(bank.shape) == (STATE_SLOTS, *shape)
            assert bank.dtype == dtype
            if not bank.is_contiguous():
                strided.append((name, tuple(bank.shape), tuple(bank.stride())))
                continue
            # The view form still hands one row per request: a contiguous slice.
            assert all(bank[slot].is_contiguous() for slot in range(STATE_SLOTS))
        if strided:
            continue
        # One raw buffer per recurrent layer, the two states as disjoint byte regions
        # inside it, and the buffer exactly the budgeted slots x slot bytes.
        assert conv.untyped_storage().data_ptr() == recurrent.untyped_storage().data_ptr()
        spans = sorted((_span(conv), _span(recurrent)))
        assert spans[0][1] <= spans[1][0], spans
        assert spans[1][1] <= STATE_SLOTS * slot_bytes
        assert conv.untyped_storage().nbytes() == STATE_SLOTS * slot_bytes
    assert not strided, (
        f"{len(strided)} recurrent state bank(s) are strided views, which the Neuron "
        f"executor refuses as graph inputs; first: {strided[:2]}"
    )


# ── 2. the hand-off on the bs=64 line ─────────────────────────────────────────


def _serve_line_runner(tmp_path, record: dict):
    """The real runner on the bs=64 line with the root's forward recording its kwargs."""
    from vllm.config import set_current_vllm_config
    from vllm.distributed import parallel_state as dist_state
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

    import vllm_neuron.vllm.worker.neuron_model_runner as runner_module
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextForConditionalGeneration as Root

    neuron_config = {
        "num_batched_tokens_buckets": [PREFILL_BUCKET],
        "kv_segment_size_buckets": [MAX_MODEL_LEN],
        "decode_context_length_buckets": [DECODE_CTX_BUCKET],
        "num_seqs_buckets": list(BATCH_BUCKETS),
        "hybrid_kv_block_size": BLOCK,
        "on_device_sampling_config": {"all_greedy": True},
    }
    config = EngineArgs(
        model=str(FIXTURE), skip_tokenizer_init=True, max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS, max_num_batched_tokens=PREFILL_BUCKET, block_size=BLOCK,
        enforce_eager=True, enable_prefix_caching=False, async_scheduling=False,
        additional_config={"neuron_config": neuron_config},
    ).create_engine_config()
    hf = json.load(open(FIXTURE / "config.json"))
    text = hf.get("text_config", hf)
    text["linear_attn_config"]["num_heads"] = 1  # per-rank KDA heads at TP=64

    def recording_forward(input_ids, *, layer_carriers, sampling_positions,
                          device_sampling_params=None, device_logit_mask=None, **rest):
        record["kwargs"] = {
            "input_ids": input_ids, "layer_carriers": layer_carriers,
            "sampling_positions": sampling_positions,
            "device_sampling_params": device_sampling_params,
            "device_logit_mask": device_logit_mask, **rest,
        }
        rows = int(sampling_positions.shape[0])
        if device_sampling_params is not None:
            return torch.zeros(rows, dtype=torch.int32)
        return torch.zeros(rows, VOCAB, dtype=torch.bfloat16)

    context = set_current_vllm_config(config, check_compile=False)
    context.__enter__()
    dist_state.init_distributed_environment(
        world_size=1, rank=0, distributed_init_method=f"file://{tmp_path}/rdv",
        local_rank=0, backend="gloo")
    dist_state.ensure_model_parallel_initialized(1, 1)

    def teardown():
        dist_state.destroy_model_parallel()
        dist_state.destroy_distributed_environment()
        context.__exit__(None, None, None)

    try:
        runner = runner_module.NeuronModelRunner(config, device=torch.device("cpu"))
        root = Root.from_configs(hf, text_neuron_config=runner.neuron_config)
        root.forward = recording_forward
        runner.model = root
        runner.vocab_size = VOCAB
        spec = runner.get_kv_cache_spec()
        kv_config = get_kv_cache_configs(config, [spec], [KV_BYTES])[0]
        runner.initialize_kv_cache(kv_config)
    except BaseException:
        teardown()
        raise
    return runner, teardown


def test_every_decode_graph_input_is_contiguous_for_every_batch_bucket_of_the_bs64_line(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_neuron.vllm.worker import glm5next_state_banks as runner_side

    monkeypatch.delenv(runner_side.STATE_BANKS_ENV, raising=False)
    monkeypatch.delenv("VLLM_NEURON_KDA_FUSED_DECODE", raising=False)
    # The served recipe: on-device greedy sampling, host-only metadata.
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_ON_DEVICE_SAMPLING", "1")
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_HOST_ONLY_METADATA", "1")
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

    record: dict = {}
    runner, teardown = _serve_line_runner(tmp_path, record)
    try:
        runner.warmup_prefill(PREFILL_BUCKET, PREFILL_BUCKET)
        strided: dict[int, list] = {}
        forms: dict[int, str] = {}
        counts: dict[int, int] = {}
        for batch in BATCH_BUCKETS:
            record.clear()
            runner.warmup_decode(batch, ctx_bucket=DECODE_CTX_BUCKET)
            kwargs = record["kwargs"]
            leaves = list(_tensor_leaves(kwargs))
            counts[batch] = len(leaves)
            strided[batch] = [
                (path, tuple(leaf.shape), tuple(leaf.stride()), str(leaf.dtype))
                for path, leaf in leaves
                if not leaf.is_contiguous()
            ]
            first = kwargs["layer_carriers"][0]
            forms[batch] = "bank" if "state_slots" in first else "view"
    finally:
        teardown()

    # Power: the buckets above one request take the bank form (whole banks as inputs).
    assert forms == {batch: ("view" if batch == 1 else "bank") for batch in BATCH_BUCKETS}, forms
    assert all(counts[batch] > 0 for batch in BATCH_BUCKETS), counts
    failing = {batch: found for batch, found in strided.items() if found}
    assert not failing, (
        "non-contiguous decode graph inputs per batch bucket (the Neuron executor refuses "
        "them): "
        + "; ".join(
            f"B={batch}: {len(found)} input(s), first {found[0]}"
            for batch, found in failing.items()
        )
    )


# ── 3. the carrier builder refuses a strided bank by name ─────────────────────


def test_a_bank_form_carrier_refuses_a_strided_bank_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """A strided bank in the bank form is refused where the carrier is built, naming the layer.

    The executor would refuse it after the compile with its own message; the carrier builder
    says which layer and what the cure is before any graph is captured.
    """
    from test.vllm_neuron.worker.test_glm5next_state_banks import _decode, _stack
    from vllm_neuron.vllm.worker import glm5next_state_banks as runner_side

    monkeypatch.delenv(runner_side.STATE_BANKS_ENV, raising=False)
    monkeypatch.delenv("VLLM_NEURON_KDA_FUSED_DECODE", raising=False)
    banks, side, geometries = _stack(linear=2, sparse=1)
    # The earlier layout for the second recurrent layer: both states in one buffer, one slot
    # (conv row + recurrent row) per stride, so the whole bank is not contiguous.
    conv_row, recurrent_row = 2 * 3, 1 * 2 * 2
    raw = torch.zeros(8 * (conv_row + recurrent_row), dtype=torch.float32)
    banks[1]["conv_state"] = torch.as_strided(raw, (8, 2, 3), (conv_row + recurrent_row, 3, 1))
    banks[1]["recurrent_state"] = torch.as_strided(
        raw, (8, 1, 2, 2), (conv_row + recurrent_row, 4, 2, 1), storage_offset=conv_row
    )
    assert not banks[1]["conv_state"].is_contiguous()
    with pytest.raises(ValueError, match="linear.1.*strided") as refused:
        _decode(banks, side, geometries, slots=[3, 0, 5, 1], starts=[5, 6, 7, 8])
    assert "conv_state" in str(refused.value)
    assert runner_side.STATE_BANKS_ENV in str(refused.value)
    # The same banks in the view form (rows are contiguous) are served as before.
    monkeypatch.setenv(runner_side.STATE_BANKS_ENV, "0")
    carriers = _decode(banks, side, geometries, slots=[3, 0, 5, 1], starts=[5, 6, 7, 8])
    assert all(view.is_contiguous() for view in carriers[1]["conv_state"])

    # A sparse side cache that is a strided view is refused the same way.
    monkeypatch.delenv(runner_side.STATE_BANKS_ENV, raising=False)
    banks, side, geometries = _stack(linear=0, sparse=1)
    pool = side[0]["pool_cache"]
    side[0]["pool_cache"] = torch.as_strided(
        torch.zeros(pool.numel() * 2), tuple(pool.shape), (2 * pool.stride(0), *pool.stride()[1:])
    )
    with pytest.raises(ValueError, match="sparse.0.*strided"):
        _decode(banks, side, geometries, slots=[3, 0, 5, 1], starts=[5, 6, 7, 8])
