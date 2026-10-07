# SPDX-License-Identifier: Apache-2.0
"""Layer 45's KV state beside the 11 trunk DSA layers, runner side (MTP stage A, contract C4).

When ``shadow_draft_k() > 0`` GLM-5.3-Flash's MTP layer (absolute index 45, one past the
45-layer stack) is a 12th sparse-attention (DSA) layer for the cache: ``get_kv_spec``
reports a latent page for it, ``bind_kv_cache`` keeps its bank, and the runner, which
walks the banks ``bind_kv_cache`` kept, gives it a pooled-key store and a ring
(``_glm5next_side_caches``) and a carrier (``_glm5next_layer_carriers``) at list index 45,
the same lookup as layer 43's at index 43.

Read here, through the real runner (``get_kv_cache_spec`` -> vLLM's KV config ->
``initialize_kv_cache`` -> ``warmup_prefill`` / ``warmup_decode``) with the root's forward
replaced by a recorder, on the two serve lines (the standard bs=1 line, 4k, one sequence;
the bs=64 @ 8k line):

1. knob 0: the ``LayerSpec`` list, the runner's KV specs, vLLM's KV config (groups,
   tensors, blocks), the banks, the side caches and, per warmup leg, the ordered list of
   every leaf the runner hands the root (path, kind, shape, dtype, stride, offset,
   contiguity, and the bytes of every operand up to 1 MiB) are EQUAL to the record taken
   on 82bee3b (``mtp_state_base_82bee3b.json`` beside this file; regenerate with
   ``python -m test.vllm_neuron.worker.test_mtp_state_runner --record <json>`` from a tree
   at 82bee3b, see ``main``); the worker's KV need and footprint at both lines too;
2. knob 5: exactly one more DSA spec, side caches for 12 layers, layer 45's carrier equal
   in structure (keys, kinds, dtypes, shapes, contiguity) to layer 43's on every leg, on
   its own storage; at B >= 2 the bank form hands layer 45's whole contiguous banks;
3. slotreset's ``empty_slot`` empties layer 45's slot with the other 11 when a slot is
   handed out, and its ring when a request opens at position 0;
4. the per-rank footprint the 12th layer adds, at both lines, from the worker's own
   formula, against a hand formula.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_mtp_state_runner.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.fast, pytest.mark.forked]

BASE_RECORD = pathlib.Path(__file__).with_name("mtp_state_base_82bee3b.json")
FIXTURE = pathlib.Path("test/vllm_neuron/model/glm5_next/fixtures")
VOCAB = 154880
BLOCK = 128
#: Operands up to this size are recorded by content as well as by layout.
DIGEST_MAX_BYTES = 1 << 20
#: The knob value stage A's shadow draft serves (the GPU recipe's k).
DRAFT_K = 5
#: The stack is layers 0..44; the MTP layer is the next index.
STACK = 45
DRAFT_LAYER = 45
#: The trunk's last DSA layer, whose geometry and weights layer 45 shares.
LAST_TRUNK_DSA = 43

#: The two serve lines (``/home/ubuntu/glm53f-wt2/00_brief.md``, "Serving facts").
LINES = {
    # Fast bs=1 recipe: max_model_len 4096, one sequence, segment 1024, batched 1024,
    # decode context bucket 2048, hybrid block 128, prefix caching left at vLLM's default
    # (on) and no --mamba-block-size.
    "bs1": {
        "max_model_len": 4096,
        "max_num_seqs": 1,
        "prefill_bucket": 1024,
        "kv_segment": 1024,
        "decode_ctx": 2048,
        "batch_buckets": (1,),
        "enable_prefix_caching": None,
        "mamba_block_size": None,
    },
    # bs=64 @ 8k: --max-num-seqs 64 --max-model-len 8192 --no-enable-prefix-caching
    # --mamba-block-size 8192, segment 8192, num_seqs_buckets [1..64], ctx bucket 2048.
    "bs64": {
        "max_model_len": 8192,
        "max_num_seqs": 64,
        "prefill_bucket": 1024,
        "kv_segment": 8192,
        "decode_ctx": 2048,
        "batch_buckets": (1, 2, 4, 8, 16, 32, 64),
        "enable_prefix_caching": False,
        "mamba_block_size": 8192,
    },
}


# ── the knob ──────────────────────────────────────────────────────────────────


def set_draft_k(monkeypatch, value: int) -> None:
    """Point contract C1's ``mtp.shadow_draft_k()`` at ``value`` for this test.

    The function is worker-50's (``envs.VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT``); on a tree
    that does not define it yet the attribute is added, so the model's call reads it.
    """
    from vllm_neuron.model.glm5_next import mtp

    monkeypatch.setattr(mtp, "shadow_draft_k", lambda: int(value), raising=False)


# ── recording helpers ─────────────────────────────────────────────────────────


def _digest(tensor: torch.Tensor) -> str:
    flat = tensor.detach().contiguous().reshape(-1)
    if flat.dtype == torch.bool:
        flat = flat.to(torch.uint8)
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()


def leaf_record(path: str, value) -> list:
    """One graph-input leaf: its path and everything a captured graph keys on."""
    if isinstance(value, torch.Tensor):
        small = value.numel() * value.element_size() <= DIGEST_MAX_BYTES
        return [
            path, "tensor", list(value.shape), str(value.dtype), list(value.stride()),
            int(value.storage_offset()), bool(value.is_contiguous()),
            _digest(value) if small else None,
        ]
    if value is None:
        return [path, "None"]
    if isinstance(value, (bool, int, float, str)):
        return [path, type(value).__name__, value]
    return [path, type(value).__name__, repr(value)]


def leaves(value, path: str = "") -> list:
    """Every leaf of a kwargs tree, in the order the tree yields them."""
    out: list = []
    if isinstance(value, dict):
        for key, item in value.items():
            out += leaves(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            out += leaves(item, f"{path}[{index}]")
    else:
        out.append(leaf_record(path, value))
    return out


def carrier_structure(carrier: dict) -> dict:
    """Keys, kinds, dtypes, shapes and contiguity of one layer's carrier, no values."""
    def one(value):
        if isinstance(value, torch.Tensor):
            return ["tensor", list(value.shape), str(value.dtype), bool(value.is_contiguous())]
        if isinstance(value, (list, tuple)):
            return [type(value).__name__, [one(item) for item in value]]
        return [type(value).__name__]

    return {key: one(value) for key, value in carrier.items()}


def spec_record(layer) -> dict:
    return {
        "name": layer.name,
        "num_kv_heads": layer.num_kv_heads,
        "head_size": layer.head_size,
        "dtype": str(layer.dtype),
        "sliding_window_size": layer.sliding_window_size,
        "chunk_size": layer.chunk_size,
        "kda_conv_state_shape": None if layer.kda_conv_state_shape is None
        else list(layer.kda_conv_state_shape),
        "kda_recurrent_state_shape": None if layer.kda_recurrent_state_shape is None
        else list(layer.kda_recurrent_state_shape),
        "kda_conv_state_dtype": str(layer.kda_conv_state_dtype),
        "kda_recurrent_state_dtype": str(layer.kda_recurrent_state_dtype),
        "latent_kv": bool(layer.latent_kv),
    }


def vllm_spec_record(spec) -> dict:
    record = {"class": type(spec).__name__, "block_size": int(spec.block_size),
              "page_size_bytes": int(spec.page_size_bytes)}
    for field in ("num_kv_heads", "head_size", "shapes", "page_size_padded",
                  "num_speculative_blocks"):
        if hasattr(spec, field):
            value = getattr(spec, field)
            record[field] = json.loads(json.dumps(value, default=list))
    for field in ("dtype", "dtypes"):
        if hasattr(spec, field):
            record[field] = str(getattr(spec, field))
    return record


def bank_record(bank: dict) -> dict:
    out = {}
    for key, value in bank.items():
        out[key] = (
            ["tensor", list(value.shape), str(value.dtype), list(value.stride()),
             int(value.storage_offset())]
            if isinstance(value, torch.Tensor) else value
        )
    return out


# ── the real runner on one serve line ─────────────────────────────────────────


def line_runner(line: str, tmp_path, record: dict):
    """The real runner on ``line``, KV cache sized to the worker's own need, forward recorded."""
    from vllm.config import set_current_vllm_config
    from vllm.distributed import parallel_state as dist_state
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

    import vllm_neuron.vllm.worker.neuron_model_runner as runner_module
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextForConditionalGeneration as Root
    from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

    spec = LINES[line]
    neuron_config = {
        "num_batched_tokens_buckets": [spec["prefill_bucket"]],
        "kv_segment_size_buckets": [spec["kv_segment"]],
        "decode_context_length_buckets": [spec["decode_ctx"]],
        "num_seqs_buckets": list(spec["batch_buckets"]),
        "hybrid_kv_block_size": BLOCK,
        "on_device_sampling_config": {"all_greedy": True},
    }
    engine = {
        "model": str(FIXTURE), "skip_tokenizer_init": True,
        "max_model_len": spec["max_model_len"], "max_num_seqs": spec["max_num_seqs"],
        "max_num_batched_tokens": spec["prefill_bucket"], "block_size": BLOCK,
        "enforce_eager": True, "async_scheduling": False,
        "additional_config": {"neuron_config": neuron_config},
    }
    if spec["enable_prefix_caching"] is not None:
        engine["enable_prefix_caching"] = spec["enable_prefix_caching"]
    if spec["mamba_block_size"] is not None:
        engine["mamba_block_size"] = spec["mamba_block_size"]
    config = EngineArgs(**engine).create_engine_config()
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
        worker = SimpleNamespace(vllm_config=config, model_runner=runner)
        need = NeuronWorker._kv_cache_need_bytes(worker)
        kv_spec = runner.get_kv_cache_spec()
        kv_config = get_kv_cache_configs(config, [kv_spec], [need])[0]
        runner.initialize_kv_cache(kv_config)
    except BaseException:
        teardown()
        raise
    return runner, kv_config, need, teardown


def capture(line: str, tmp_path) -> dict:
    """Everything knob 0 must keep byte-identical, on one serve line."""
    spec = LINES[line]
    record: dict = {}
    runner, kv_config, need, teardown = line_runner(line, tmp_path, record)
    try:
        out: dict = {
            "kv_spec": [spec_record(layer) for layer in runner.model.get_kv_spec().layers],
            "kv_cache_spec": {
                name: vllm_spec_record(one) for name, one in runner.get_kv_cache_spec().items()
            },
            "kv_cache_config": {
                "need_bytes": int(need),
                "num_blocks": int(kv_config.num_blocks),
                "groups": [
                    {"class": type(group.kv_cache_spec).__name__,
                     "layer_names": list(group.layer_names)}
                    for group in kv_config.kv_cache_groups
                ],
                "tensors": [
                    {"size": int(tensor.size), "shared_by": list(tensor.shared_by)}
                    for tensor in kv_config.kv_cache_tensors
                ],
            },
            "banks": [bank_record(bank) for bank in runner.model.glm5next_layer_banks],
            "legs": {},
            "carriers": {},
        }
        legs = [("prefill", lambda: runner.warmup_prefill(spec["prefill_bucket"], spec["kv_segment"]))]
        legs += [
            (f"decode_b{batch}",
             (lambda batch=batch: runner.warmup_decode(batch, ctx_bucket=spec["decode_ctx"])))
            for batch in spec["batch_buckets"]
        ]
        for leg, run in legs:
            record.clear()
            run()
            kwargs = record["kwargs"]
            out["legs"][leg] = leaves(kwargs)
            out["carriers"][leg] = [carrier_structure(one) for one in kwargs["layer_carriers"]]
            out.setdefault("live", {})[leg] = record["kwargs"]
        live = runner._glm5next_side_cache_set
        out["side_caches"] = [
            {key: [list(value.shape), str(value.dtype)] for key, value in entry.items()}
            for entry in live
        ]
        out["_runner"] = runner
        return out
    finally:
        teardown()


def comparable(captured: dict) -> dict:
    """The JSON-serialisable part of a capture, the part the base record holds."""
    return json.loads(json.dumps(
        {key: value for key, value in captured.items() if not key.startswith("_") and key != "live"}
    ))


def footprint(line: str, monkeypatch=None, draft_k: int = 0) -> dict:
    """The worker's KV need and footprint at TP=64 for ``line``, by part.

    Through ``test_kv_budget_glm53f``'s harness (the real ``_kv_cache_need_bytes``,
    ``_kv_cache_footprint_bytes``, vLLM grouping, ``kv_cache_allocations`` and the real
    side-cache builder), without editing it.
    """
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

    from test.vllm_neuron.worker import test_kv_budget_glm53f as budget
    from vllm_neuron.vllm.worker.neuron_model_runner import indexer_side_cache_bytes

    spec = LINES[line]
    layers, text_config = budget.glm53f_layer_specs()
    gate_knobs = spec["mamba_block_size"] is not None
    runner = budget.fake_runner(
        layers, max_num_seqs=spec["max_num_seqs"], max_model_len=spec["max_model_len"],
        gate_knobs=gate_knobs, text_config=text_config,
    )
    worker = budget.fake_worker(runner)
    need = worker._kv_cache_need_bytes()
    total = worker._kv_cache_footprint_bytes(need)
    kv_spec = runner.get_kv_cache_spec()
    groups = get_kv_cache_groups(runner.vllm_config, kv_spec)
    side = indexer_side_cache_bytes(
        kv_spec, text_config, max_seq_len=spec["max_model_len"],
        request_slots=spec["max_num_seqs"],
    )
    return {
        "layers": len(layers),
        "dsa_layers": sum(1 for layer in layers if layer.latent_kv),
        "groups": len(groups),
        "layers_per_pool": max(len(group.layer_names) for group in groups),
        "need_bytes": int(need),
        "footprint_bytes": int(total),
        "side_cache_bytes": int(side),
    }


# ── 1. knob 0: byte-identical to 82bee3b ──────────────────────────────────────


def _base() -> dict:
    return json.loads(BASE_RECORD.read_text())


def _first_difference(want, got, path="") -> str | None:
    if type(want) is not type(got):
        return f"{path}: {type(want).__name__} != {type(got).__name__}"
    if isinstance(want, dict):
        for key in sorted(set(want) | set(got)):
            if key not in want or key not in got:
                return f"{path}.{key}: present on one side only"
            found = _first_difference(want[key], got[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(want, list):
        if len(want) != len(got):
            return f"{path}: length {len(want)} != {len(got)}"
        for index, (one, two) in enumerate(zip(want, got)):
            found = _first_difference(one, two, f"{path}[{index}]")
            if found:
                return found
        return None
    return None if want == got else f"{path}: {want!r} != {got!r}"


@pytest.mark.parametrize("line", sorted(LINES))
def test_knob_off_everything_the_runner_builds_is_the_82bee3b_record(
    line, tmp_path, monkeypatch,
) -> None:
    _serve_env(monkeypatch)
    set_draft_k(monkeypatch, 0)
    got = comparable(capture(line, tmp_path))
    want = _base()["lines"][line]
    assert got == want, _first_difference(want, got)


@pytest.mark.parametrize("line", sorted(LINES))
def test_knob_off_the_worker_footprint_is_the_82bee3b_record(line, monkeypatch) -> None:
    set_draft_k(monkeypatch, 0)
    assert footprint(line) == _base()["footprint"][line]


def test_the_record_is_82bee3bs_own_and_covers_both_lines() -> None:
    """The record says which tree wrote it, and it holds both lines' legs."""
    base = _base()
    assert base["base_commit"].startswith("82bee3b")
    for line, spec in LINES.items():
        legs = base["lines"][line]["legs"]
        assert sorted(legs) == sorted(
            ["prefill"] + [f"decode_b{batch}" for batch in spec["batch_buckets"]]
        )
        assert len(base["lines"][line]["kv_spec"]) == STACK
        assert all(legs[leg] for leg in legs)


# ── 2. knob 5: one more DSA layer, carried like layer 43 ─────────────────────


def _serve_env(monkeypatch) -> None:
    from vllm_neuron.vllm.worker import glm5next_state_banks as runner_side

    monkeypatch.delenv(runner_side.STATE_BANKS_ENV, raising=False)
    monkeypatch.delenv("VLLM_NEURON_KDA_FUSED_DECODE", raising=False)
    # The served recipe: on-device greedy sampling, host-only metadata.
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_ON_DEVICE_SAMPLING", "1")
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_HOST_ONLY_METADATA", "1")
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")


def _storage(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


@pytest.mark.parametrize("line", sorted(LINES))
def test_knob_on_adds_one_dsa_layer_whose_state_is_carried_like_layer_43s(
    line, tmp_path, monkeypatch,
) -> None:
    _serve_env(monkeypatch)
    set_draft_k(monkeypatch, DRAFT_K)
    got = capture(line, tmp_path)
    base = _base()["lines"][line]

    # The spec: the stack's 45 entries unchanged, then one DSA entry for layer 45 with
    # layer 43's geometry.
    assert got["kv_spec"][:STACK] == base["kv_spec"]
    assert len(got["kv_spec"]) == STACK + 1
    draft, trunk = got["kv_spec"][DRAFT_LAYER], got["kv_spec"][LAST_TRUNK_DSA]
    assert draft["name"] == f"layers.{DRAFT_LAYER}.self_attn"
    assert {k: v for k, v in draft.items() if k != "name"} == {
        k: v for k, v in trunk.items() if k != "name"
    }
    dsa = [index for index, one in enumerate(got["kv_spec"]) if one["latent_kv"]]
    assert len(dsa) == 12 and dsa[-1] == DRAFT_LAYER
    # The runner's KV spec: the same latent page as layer 43's, in the same group.
    specs = got["kv_cache_spec"]
    assert specs[draft["name"]] == specs[trunk["name"]]
    groups = got["kv_cache_config"]["groups"]
    home = [g for g in groups if trunk["name"] in g["layer_names"]]
    assert len(home) == 1 and draft["name"] in home[0]["layer_names"]

    # The bank: index 45, the sparse family, its own latent buffer.
    banks = got["_runner"].model.glm5next_layer_banks
    assert len(banks) == STACK + 1
    assert banks[DRAFT_LAYER]["name"] == draft["name"]
    assert banks[DRAFT_LAYER]["layer_index"] == DRAFT_LAYER
    assert banks[DRAFT_LAYER]["family"] == "self_attn"
    for key in ("blocks", "block_size", "slots", "head_size"):
        assert banks[DRAFT_LAYER][key] == banks[LAST_TRUNK_DSA][key]
    latent_storages = {_storage(b["latent_bank"]) for b in banks if b["family"] == "self_attn"}
    assert len(latent_storages) == 12

    # The side caches: one set per DSA layer, layer 45's shaped like layer 43's.
    side = got["side_caches"]
    assert len(side) == STACK + 1
    assert sum(1 for entry in side if entry) == 12
    assert side[DRAFT_LAYER] == side[LAST_TRUNK_DSA]
    assert side[:STACK] == base["side_caches"]
    live = got["_runner"]._glm5next_side_cache_set
    for key in ("pool_cache", "tail", "pad_tail"):
        assert live[DRAFT_LAYER][key].is_contiguous()
        assert _storage(live[DRAFT_LAYER][key]) != _storage(live[LAST_TRUNK_DSA][key])

    # The carriers, on every leg: 46 mappings, layer 45's equal to layer 43's in
    # structure, the stack's equal to the knob-0 record's in structure, and layer 45's
    # state on its own storage.
    for leg, carriers in got["carriers"].items():
        assert len(carriers) == STACK + 1, leg
        assert carriers[DRAFT_LAYER] == carriers[LAST_TRUNK_DSA], leg
        assert len(base["carriers"][leg]) == STACK
        kwargs = got["live"][leg]
        mine, trunk_carrier = (kwargs["layer_carriers"][index] for index in (DRAFT_LAYER, LAST_TRUNK_DSA))
        assert mine["latent_cache"] is banks[DRAFT_LAYER]["latent_cache"], leg
        for key in ("pool_cache", "tail", "prefill_tail"):
            if key not in mine:
                continue
            for one, two in zip(
                mine[key] if isinstance(mine[key], tuple) else (mine[key],),
                trunk_carrier[key] if isinstance(trunk_carrier[key], tuple) else (trunk_carrier[key],),
            ):
                assert _storage(one) == _storage(live[DRAFT_LAYER][key if key != "prefill_tail" else "tail"]) \
                    or _storage(one) == _storage(live[DRAFT_LAYER]["pad_tail"]), (leg, key)
                assert _storage(one) != _storage(two), (leg, key)
        # Every leaf the graph would take is contiguous (the executor refuses others).
        strided = [leaf for leaf in leaves(kwargs) if leaf[1] == "tensor" and not leaf[6]]
        assert not strided, (leg, strided[:2])
        if leg.startswith("decode_b") and int(leg[len("decode_b"):]) >= 2:
            # The bank form: layer 45's whole banks and its slot tensor.
            assert mine["pool_cache"] is live[DRAFT_LAYER]["pool_cache"], leg
            assert mine["tail"] is live[DRAFT_LAYER]["tail"], leg
            assert "state_slots" in mine and mine["state_slots"].dtype == torch.int64


# ── 3. slotreset: layer 45's slot is emptied with the other 11 ───────────────


def test_a_handed_out_slot_is_emptied_on_layer_45_too(tmp_path, monkeypatch) -> None:
    _serve_env(monkeypatch)
    set_draft_k(monkeypatch, DRAFT_K)
    record: dict = {}
    runner, _config, _need, teardown = line_runner("bs64", tmp_path, record)
    try:
        banks = runner.model.glm5next_layer_banks
        live = runner._glm5next_live_side_caches(banks)
        assert len(live) == STACK + 1 and live[DRAFT_LAYER]
        gen = torch.Generator().manual_seed(20261007)
        for entry in live:
            for key in entry:
                entry[key].copy_((torch.rand(entry[key].shape, generator=gen) + 0.5).to(entry[key].dtype))
        before = {
            (index, key): entry[key].clone() for index, entry in enumerate(live) for key in entry
        }
        identities = {(index, key): id(entry[key]) for index, entry in enumerate(live) for key in entry}
        # Slots 0..2 held; the new request gets slot 3.
        runner._glm5next_request_slot_table = {"held-0": 0, "held-1": 1, "held-2": 2}
        runner._glm5next_side_cache_positions = {0: 9, 1: 9, 2: 9}
        slots = runner._glm5next_request_slots(
            banks, ["new"], synthetic=False, side_caches=live
        )
        assert slots == [3]
        dsa = [index for index, entry in enumerate(live) if entry]
        assert dsa[-1] == DRAFT_LAYER and len(dsa) == 12
        for index in dsa:
            for key in ("pool_cache", "tail"):
                row = live[index][key][3]
                assert torch.equal(row, torch.zeros_like(row)), (index, key)
                assert not torch.signbit(row.float()).any(), (index, key)
                others = torch.cat([live[index][key][:3], live[index][key][4:]])
                want = torch.cat([before[(index, key)][:3], before[(index, key)][4:]])
                assert torch.equal(others, want), (index, key)
            assert torch.equal(live[index]["pad_tail"], before[(index, "pad_tail")])
        # In place: the same tensors the bank-form graph takes.
        assert identities == {(i, k): id(e[k]) for i, e in enumerate(live) for k in e}

        # A request re-opening at position 0 in the slot it holds gets its ring emptied,
        # layer 45's included.
        for index in dsa:
            live[index]["tail"][1].copy_(torch.full_like(live[index]["tail"][1], 2.0))
        runner._glm5next_position_arm(1, 0, side_caches=live, is_prefill=True)
        for index in dsa:
            assert torch.equal(live[index]["tail"][1], torch.zeros_like(live[index]["tail"][1])), index
    finally:
        teardown()


# ── 4. the footprint the 12th layer adds ─────────────────────────────────────


def _hand_footprint(line: str, *, dsa: int, kda: int = 34) -> dict:
    """The footprint by hand: vLLM's grouping, the worker's need, the runner's buffers.

    vLLM groups the specs by type into groups of ``min(count)`` layers
    (``_get_kv_cache_groups_uniform_page_size``: 34 < 1.5 x 11 and 34 < 1.5 x 12 are both
    false), so the latent group holds ``dsa`` layers and the 34 recurrent layers fill
    ``cdiv(34, dsa)`` groups. The worker prices ``cdiv(L, block)`` blocks per group per
    request (a recurrent group's block is the attention block, or ``L`` with
    ``--mamba-block-size L``), one null block, ``dsa`` layers per pool; the runner
    allocates one latent buffer per pool tensor, one state bank of ``S`` slots per
    recurrent layer, and the side caches of every DSA layer.
    """
    spec = LINES[line]
    seqs, length = spec["max_num_seqs"], spec["max_model_len"]
    page = BLOCK * 512 * 2  # 128 tokens x kv_lora_rank 512 x bf16
    recurrent_block = spec["mamba_block_size"] or BLOCK
    kda_groups = -(-kda // dsa)
    blocks_per_request = -(-length // BLOCK) + kda_groups * -(-length // recurrent_block)
    num_blocks = blocks_per_request * seqs + 1
    need = num_blocks * page * dsa
    slot_bytes = 67840  # recurrent_state_slot_bytes at TP=64 (test_kv_spec_patch pins it)
    side_per_layer = seqs * ((length // 4 + 1) * 128 + 2 * (2 * 4 * 128)) * 2
    return {
        "groups": 1 + kda_groups,
        "need_bytes": need,
        "footprint_bytes": need + kda * seqs * slot_bytes + dsa * side_per_layer,
        "side_cache_bytes": dsa * side_per_layer,
    }


@pytest.mark.parametrize("line", sorted(LINES))
def test_the_worker_formula_counts_layer_45_and_matches_the_hand_formula(
    line, monkeypatch,
) -> None:
    set_draft_k(monkeypatch, 0)
    off = footprint(line)
    set_draft_k(monkeypatch, DRAFT_K)
    on = footprint(line)
    assert (off["dsa_layers"], on["dsa_layers"]) == (11, 12)
    assert (off["layers"], on["layers"]) == (STACK, STACK + 1)
    for measured, dsa in ((off, 11), (on, 12)):
        hand = _hand_footprint(line, dsa=dsa)
        assert measured["groups"] == hand["groups"]
        assert measured["layers_per_pool"] == dsa
        assert measured["need_bytes"] == hand["need_bytes"]
        assert measured["side_cache_bytes"] == hand["side_cache_bytes"]
        assert measured["footprint_bytes"] == hand["footprint_bytes"]
    # The side-cache term counts the 12th layer: exactly one more layer's caches.
    assert on["side_cache_bytes"] * 11 == off["side_cache_bytes"] * 12


# ── recording the base ───────────────────────────────────────────────────────


def main(argv=None) -> int:
    """Write the knob-0 record from the tree on ``sys.path`` (82bee3b for the committed one).

        cd <tree at 82bee3b> && NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 PYTHONPATH=$PWD \\
            python <this file> --record <json> --commit 82bee3b
    """
    import subprocess
    import tempfile

    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=pathlib.Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args(argv)
    import os

    os.environ.pop("VLLM_NEURON_GLM5NEXT_STATE_BANKS", None)
    os.environ.pop("VLLM_NEURON_KDA_FUSED_DECODE", None)
    os.environ["VLLM_NEURON_GLM5NEXT_ON_DEVICE_SAMPLING"] = "1"
    os.environ["VLLM_NEURON_GLM5NEXT_HOST_ONLY_METADATA"] = "1"
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
    import vllm_neuron

    tree = pathlib.Path(vllm_neuron.__file__).resolve().parents[1]
    head = subprocess.run(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    out = {
        "base_commit": args.commit,
        "tree": str(tree),
        "tree_head": head or None,
        "lines": {},
        "footprint": {},
    }
    for line in sorted(LINES):
        with tempfile.TemporaryDirectory() as scratch:
            out["lines"][line] = comparable(capture(line, pathlib.Path(scratch)))
        out["footprint"][line] = footprint(line)
    args.record.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
