# SPDX-License-Identifier: Apache-2.0
"""The MTP draft layer's KV state beside the trunk's DSA layers, runner side.

When the shadow-draft knob is above 0 the root builds the draft head, whose one decoder
block (index ``num_hidden_layers + num_nextn_predict_layers - 1``, past the stack) is a
sparse-attention (DSA) layer: ``get_kv_spec`` reports a latent page for it after the
stack, ``bind_kv_cache`` keeps its bank last, and the runner, which walks the banks
``bind_kv_cache`` kept, gives it a pooled-key store and a ring (``_glm5next_side_caches``)
and a carrier (``_glm5next_layer_carriers``), the last list entry.

Read here, through the real runner (``get_kv_cache_spec`` -> vLLM's KV config ->
``initialize_kv_cache`` -> ``warmup_prefill`` / ``warmup_decode``) with the root's forward
replaced by a recorder, on the two serve lines (the standard bs=1 line, 4k, one sequence;
the bs=64 @ 8k line), every index and count derived from the fixture's config:

1. knob 0: the ``LayerSpec`` list, the runner's KV specs, vLLM's KV config (groups,
   tensors, blocks), the banks, the side caches and, per warmup leg, the ordered list of
   every leaf the runner hands the root (path, kind, shape, dtype, stride, offset,
   contiguity, and the bytes of every operand up to 1 MiB) are EQUAL to the knob-off
   record ``glm5next_state_lists_knob_off.json`` beside this file, taken on the tree its
   ``base_commit`` names (regenerate with ``main``, see its docstring); the worker's KV
   need and footprint at both lines too;
2. knob 5: exactly one more DSA spec, side caches for one more DSA layer, the draft
   layer's carrier equal in structure (keys, kinds, dtypes, shapes, contiguity) to the
   trunk's last DSA layer's on every leg, on its own storage; at B >= 2 the bank form
   hands the draft layer's whole contiguous banks;
3. slotreset's ``empty_slot`` empties the draft layer's slot with the trunk's when a slot
   is handed out, and its ring when a request opens at position 0;
4. the per-rank footprint the draft layer adds, at both lines, from the worker's own
   formula, against a hand formula.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_mtp_state_runner.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.fast, pytest.mark.forked]

KNOB_OFF_RECORD = pathlib.Path(__file__).with_name("glm5next_state_lists_knob_off.json")
FIXTURE = pathlib.Path("test/vllm_neuron/model/glm5_next/fixtures")
#: What the knob-off record holds and how to regenerate it; ``main`` writes it into the
#: record so the file explains itself.
RECORD_ABOUT = (
    "GLM-5.3-Flash KV state lists with the shadow-draft knob off, recorded on the tree "
    "named by base_commit, before the MTP draft layer's state existed: per serve line "
    "(bs1: 4k, one sequence; bs64: 64 x 8k) the get_kv_spec list, the runner's KV specs, "
    "vLLM's KV config, the banks, the side-cache shapes and, per warmup leg, every leaf "
    "the runner hands the root forward (path, kind, shape, dtype, stride, offset, "
    "contiguity, sha256 of operands up to 1 MiB) and each layer's carrier structure; "
    "plus the worker's KV need and footprint at TP=64. test_mtp_state_runner.py checks "
    "the knob-off tree against it. Regenerate from a checkout of base_commit: "
    "NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 PYTHONPATH=$PWD python "
    "test/vllm_neuron/worker/test_mtp_state_runner.py --record <this file> --commit <sha>."
)
#: The fixture checkpoint's vocabulary, the width of the recorder's logits.
VOCAB = json.loads((FIXTURE / "config.json").read_text())["text_config"]["vocab_size"]
#: The hybrid KV block size both serve lines run (``hybrid_kv_block_size``).
BLOCK = 128
#: Operands up to this size are recorded by content as well as by layout.
DIGEST_MAX_BYTES = 1 << 20
#: The knob value stage A's shadow draft serves.
DRAFT_K = 5

#: The two served configurations.
LINES = {
    # Fast bs=1 recipe: max_model_len 4096, one sequence, segments 1024 / 2048 / 4096
    # (the largest segment plus the 1024-row query must cover max_model_len, the bucket
    # rule for a windowed prefill), batched 1024, decode context bucket 2048, hybrid
    # block 128, prefix caching left at vLLM's default (on) and no --mamba-block-size.
    # The prefill leg warms the smallest segment, 1024, as before.
    "bs1": {
        "max_model_len": 4096,
        "max_num_seqs": 1,
        "prefill_bucket": 1024,
        "kv_segment_buckets": (1024, 2048, 4096),
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
        "kv_segment_buckets": (8192,),
        "kv_segment": 8192,
        "decode_ctx": 2048,
        "batch_buckets": (1, 2, 4, 8, 16, 32, 64),
        "enable_prefix_caching": False,
        "mamba_block_size": 8192,
    },
}


# ── the knob ──────────────────────────────────────────────────────────────────


def set_draft_k(monkeypatch, value: int) -> None:
    """Set the shadow-draft knob through its environment variable (read via ``envs``)."""
    from vllm_neuron.model.glm5_next import mtp

    monkeypatch.setenv(mtp.SHADOW_DRAFT_ENV, str(int(value)))


def layout(root) -> dict:
    """Stack length, the draft layer's index and the trunk's DSA layers, from the config."""
    from vllm_neuron.model.glm5_next.config import DSA_LAYER_TYPE

    config = root.text_config
    stack = int(config.num_hidden_layers)
    trunk_dsa = [
        index for index, kind in enumerate(config.layer_types) if kind == DSA_LAYER_TYPE
    ]
    return {
        "stack": stack,
        "draft": stack + int(config.num_nextn_predict_layers) - 1,
        "trunk_dsa": trunk_dsa,
        "last_trunk_dsa": trunk_dsa[-1],
    }


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
        "kv_segment_size_buckets": list(spec["kv_segment_buckets"]),
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


def footprint(line: str) -> dict:
    """The worker's KV need and footprint at TP=64 for ``line``, by part, at the set knob.

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


# ── 1. knob 0: byte-identical to the knob-off record ────────────────────────


def _knob_off() -> dict:
    return json.loads(KNOB_OFF_RECORD.read_text())


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
def test_knob_off_everything_the_runner_builds_is_the_recorded_list(
    line, tmp_path, monkeypatch,
) -> None:
    _serve_env(monkeypatch)
    set_draft_k(monkeypatch, 0)
    got = comparable(capture(line, tmp_path))
    want = _knob_off()["lines"][line]
    assert got == want, _first_difference(want, got)


@pytest.mark.parametrize("line", sorted(LINES))
def test_knob_off_the_worker_footprint_is_the_recorded_one(line, monkeypatch) -> None:
    set_draft_k(monkeypatch, 0)
    assert footprint(line) == _knob_off()["footprint"][line]


def test_the_record_names_its_tree_and_covers_both_lines() -> None:
    """The record says which tree wrote it and what it is, and holds both lines' legs."""
    record = _knob_off()
    assert record["about"] == RECORD_ABOUT
    assert record["tree_head"].startswith(record["base_commit"])
    stack = json.loads((FIXTURE / "config.json").read_text())["text_config"][
        "num_hidden_layers"
    ]
    for line, spec in LINES.items():
        legs = record["lines"][line]["legs"]
        assert sorted(legs) == sorted(
            ["prefill"] + [f"decode_b{batch}" for batch in spec["batch_buckets"]]
        )
        assert len(record["lines"][line]["kv_spec"]) == stack
        assert all(legs[leg] for leg in legs)


# ── 2. knob 5: one more DSA layer, carried like the trunk's last DSA layer ───


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
def test_knob_on_adds_one_dsa_layer_carried_like_the_trunks_last(
    line, tmp_path, monkeypatch,
) -> None:
    _serve_env(monkeypatch)
    set_draft_k(monkeypatch, DRAFT_K)
    got = capture(line, tmp_path)
    off = _knob_off()["lines"][line]
    root = got["_runner"].model
    shape = layout(root)
    stack, trunk = shape["stack"], shape["last_trunk_dsa"]

    # The spec: the stack's entries unchanged, then one DSA entry named for the draft
    # layer's index with the trunk's DSA geometry.
    assert got["kv_spec"][:stack] == off["kv_spec"]
    assert len(got["kv_spec"]) == stack + 1
    draft, sibling = got["kv_spec"][stack], got["kv_spec"][trunk]
    assert draft["name"] == sibling["name"].replace(f"layers.{trunk}.", f"layers.{shape['draft']}.")
    assert {k: v for k, v in draft.items() if k != "name"} == {
        k: v for k, v in sibling.items() if k != "name"
    }
    dsa = [index for index, one in enumerate(got["kv_spec"]) if one["latent_kv"]]
    assert dsa == shape["trunk_dsa"] + [stack]
    # The runner's KV spec: the same latent page as the trunk's, in the same group.
    specs = got["kv_cache_spec"]
    assert specs[draft["name"]] == specs[sibling["name"]]
    groups = got["kv_cache_config"]["groups"]
    home = [g for g in groups if sibling["name"] in g["layer_names"]]
    assert len(home) == 1 and draft["name"] in home[0]["layer_names"]

    # The bank: last, the sparse family, its own latent buffer.
    banks = root.glm5next_layer_banks
    assert len(banks) == stack + 1
    assert banks[stack]["name"] == draft["name"]
    assert banks[stack]["family"] == "self_attn"
    for key in ("blocks", "block_size", "slots", "head_size"):
        assert banks[stack][key] == banks[trunk][key]
    latent_storages = {_storage(b["latent_bank"]) for b in banks if b["family"] == "self_attn"}
    assert len(latent_storages) == len(dsa)

    # The side caches: one set per DSA layer, the draft layer's shaped like the trunk's.
    side = got["side_caches"]
    assert len(side) == stack + 1
    assert [index for index, entry in enumerate(side) if entry] == dsa
    assert side[stack] == side[trunk]
    assert side[:stack] == off["side_caches"]
    live = got["_runner"]._glm5next_side_cache_set
    for key in ("pool_cache", "tail", "pad_tail"):
        assert live[stack][key].is_contiguous()
        assert _storage(live[stack][key]) != _storage(live[trunk][key])

    # The carriers, on every leg: one more mapping, the draft layer's equal to the
    # trunk's last DSA layer's in structure, and its state on its own storage.
    for leg, carriers in got["carriers"].items():
        assert len(carriers) == stack + 1, leg
        assert len(off["carriers"][leg]) == stack
        assert carriers[stack] == carriers[trunk], leg
        kwargs = got["live"][leg]
        mine, theirs = (kwargs["layer_carriers"][index] for index in (stack, trunk))
        assert mine["latent_cache"] is banks[stack]["latent_cache"], leg
        for key in ("pool_cache", "tail", "prefill_tail"):
            if key not in mine:
                continue
            own = (live[stack]["tail" if key == "prefill_tail" else key], live[stack]["pad_tail"])
            for one, two in zip(
                mine[key] if isinstance(mine[key], tuple) else (mine[key],),
                theirs[key] if isinstance(theirs[key], tuple) else (theirs[key],),
            ):
                assert _storage(one) in {_storage(own[0]), _storage(own[1])}, (leg, key)
                assert _storage(one) != _storage(two), (leg, key)
        # Every leaf the graph would take is contiguous (the executor refuses others).
        strided = [leaf for leaf in leaves(kwargs) if leaf[1] == "tensor" and not leaf[6]]
        assert not strided, (leg, strided[:2])
        if leg.startswith("decode_b") and int(leg[len("decode_b"):]) >= 2:
            # The bank form: the draft layer's whole banks and its slot tensor.
            assert mine["pool_cache"] is live[stack]["pool_cache"], leg
            assert mine["tail"] is live[stack]["tail"], leg
            assert mine["state_slots"].dtype == theirs["state_slots"].dtype, leg


# ── 3. slotreset: the draft layer's slot is emptied with the trunk's ─────────


def test_a_handed_out_slot_is_emptied_on_the_draft_layer_too(tmp_path, monkeypatch) -> None:
    _serve_env(monkeypatch)
    set_draft_k(monkeypatch, DRAFT_K)
    record: dict = {}
    runner, _config, _need, teardown = line_runner("bs64", tmp_path, record)
    try:
        banks = runner.model.glm5next_layer_banks
        shape = layout(runner.model)
        live = runner._glm5next_live_side_caches(banks)
        dsa = [index for index, entry in enumerate(live) if entry]
        assert dsa == shape["trunk_dsa"] + [shape["stack"]]
        gen = torch.Generator().manual_seed(20261007)
        for entry in live:
            for key in entry:
                entry[key].copy_((torch.rand(entry[key].shape, generator=gen) + 0.5).to(entry[key].dtype))
        before = {
            (index, key): entry[key].clone() for index, entry in enumerate(live) for key in entry
        }
        identities = {(index, key): id(entry[key]) for index, entry in enumerate(live) for key in entry}
        # Slots 0..2 held; the new request gets the first free one.
        held = {f"held-{slot}": slot for slot in range(3)}
        runner._glm5next_request_slot_table = dict(held)
        runner._glm5next_side_cache_positions = {slot: 9 for slot in held.values()}
        (slot,) = runner._glm5next_request_slots(
            banks, ["new"], synthetic=False, side_caches=live
        )
        assert slot == len(held)
        for index in dsa:
            for key in ("pool_cache", "tail"):
                row = live[index][key][slot]
                assert torch.equal(row, torch.zeros_like(row)), (index, key)
                assert not torch.signbit(row.float()).any(), (index, key)
                others = torch.cat([live[index][key][:slot], live[index][key][slot + 1:]])
                want = torch.cat([before[(index, key)][:slot], before[(index, key)][slot + 1:]])
                assert torch.equal(others, want), (index, key)
            assert torch.equal(live[index]["pad_tail"], before[(index, "pad_tail")])
        # In place: the same tensors the bank-form graph takes.
        assert identities == {(i, k): id(e[k]) for i, e in enumerate(live) for k in e}

        # A request re-opening at position 0 in the slot it holds gets its ring emptied,
        # the draft layer's included.
        for index in dsa:
            live[index]["tail"][1].copy_(torch.full_like(live[index]["tail"][1], 2.0))
        runner._glm5next_position_arm(1, 0, side_caches=live, is_prefill=True)
        for index in dsa:
            assert torch.equal(live[index]["tail"][1], torch.zeros_like(live[index]["tail"][1])), index
    finally:
        teardown()


# ── 4. the footprint the draft layer adds ────────────────────────────────────


def _hand_footprint(line: str, layers, text_config) -> dict:
    """The footprint by hand: vLLM's grouping, the worker's need, the runner's buffers.

    vLLM groups the specs by type (``_get_kv_cache_groups_uniform_page_size``): the group
    size is the smallest type's layer count, or the largest's when that is under 1.5x the
    smallest, and each type fills ``cdiv(count, size)`` groups. The worker prices
    ``cdiv(L, block)`` blocks per group per request (a recurrent group's block is the
    attention block, or ``L`` with ``--mamba-block-size L``), one null block, ``size``
    layers per pool; the runner allocates one latent buffer per pool tensor, one state
    bank of ``max_num_seqs`` slots per recurrent layer (a slot is both states, rounded up
    to ``RECURRENT_SLOT_ALIGN_BYTES``), and the side caches of every DSA layer
    (``pool_cache`` ``[S, L // index_kpool + 1, index_head_dim]``, ``tail`` and
    ``pad_tail`` ``[S, 2, index_kpool, index_head_dim]``, the latent dtype).
    """
    from vllm.utils.math_utils import cdiv

    from vllm_neuron.vllm.patches.kv_spec_patch import RECURRENT_SLOT_ALIGN_BYTES

    spec = LINES[line]
    seqs, length = spec["max_num_seqs"], spec["max_model_len"]
    latent = [layer for layer in layers if layer.latent_kv]
    recurrent = [layer for layer in layers if layer.kda_conv_state_shape is not None]
    assert len(latent) + len(recurrent) == len(layers)
    one = latent[0]
    page = BLOCK * one.num_kv_heads * one.head_size * one.dtype.itemsize
    counts = (len(latent), len(recurrent))
    size = max(counts) if max(counts) < 1.5 * min(counts) else min(counts)
    latent_groups, recurrent_groups = (cdiv(count, size) for count in counts)
    recurrent_block = spec["mamba_block_size"] or BLOCK
    blocks_per_request = (
        latent_groups * cdiv(length, BLOCK) + recurrent_groups * cdiv(length, recurrent_block)
    )
    need = (blocks_per_request * seqs + 1) * page * size
    state = recurrent[0]
    state_bytes = sum(
        math.prod(shape) * dtype.itemsize
        for shape, dtype in (
            (state.kda_conv_state_shape, state.kda_conv_state_dtype),
            (state.kda_recurrent_state_shape, state.kda_recurrent_state_dtype),
        )
    )
    slot_bytes = cdiv(state_bytes, RECURRENT_SLOT_ALIGN_BYTES) * RECURRENT_SLOT_ALIGN_BYTES
    pool, width = int(text_config.index_kpool), int(text_config.index_head_dim)
    side_per_layer = seqs * ((length // pool + 1) * width + 2 * (2 * pool * width)) * one.dtype.itemsize
    return {
        "groups": latent_groups + recurrent_groups,
        "layers_per_pool": size,
        "need_bytes": need,
        "footprint_bytes": need + len(recurrent) * seqs * slot_bytes + len(latent) * side_per_layer,
        "side_cache_bytes": len(latent) * side_per_layer,
    }


@pytest.mark.parametrize("line", sorted(LINES))
def test_the_worker_formula_counts_the_draft_layer_and_matches_the_hand_formula(
    line, monkeypatch,
) -> None:
    from test.vllm_neuron.worker import test_kv_budget_glm53f as budget

    measured = {}
    for knob in (0, DRAFT_K):
        set_draft_k(monkeypatch, knob)
        layers, text_config = budget.glm53f_layer_specs()
        measured[knob] = footprint(line)
        hand = _hand_footprint(line, layers, text_config)
        for key, value in hand.items():
            assert measured[knob][key] == value, (knob, key)
    off, on = measured[0], measured[DRAFT_K]
    # One more layer, and it is a DSA layer whose side caches are counted.
    assert on["layers"] == off["layers"] + 1
    assert on["dsa_layers"] == off["dsa_layers"] + 1
    assert on["side_cache_bytes"] * off["dsa_layers"] == off["side_cache_bytes"] * on["dsa_layers"]


# ── recording the knob-off lists ─────────────────────────────────────────────


def main(argv=None) -> int:
    """Write the knob-off record from the tree on ``sys.path``.

    Run from a checkout of the tree the record is to describe (the committed one was taken
    on the commit its ``base_commit`` names, before the draft layer's state existed):

        cd <checkout> && NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 PYTHONPATH=$PWD \\
            python test/vllm_neuron/worker/test_mtp_state_runner.py \\
            --record <json> --commit <sha>
    """
    import os
    import subprocess
    import tempfile

    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=pathlib.Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args(argv)

    os.environ.pop("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", None)
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
        "about": RECORD_ABOUT,
        "base_commit": args.commit,
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
