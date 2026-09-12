"""The model test helper is itself under test -- it is load-bearing.

Every later module test drives its synthetic weights through
``test/vllm_neuron/model/utils.py``, so a silent defect there would show up as a
numerics failure in some unrelated model test. Three counted predicates, each a
measured value compared against a declared threshold:

1. ``FakeSafeSlice`` round-trips a ``[4, 8]`` fp8 tensor with **max abs diff ==
   0.0** (bit-exact).
2. ``hf_state_to_fake_slices`` maps **exactly 6/6** synthetic HF keys.
3. ``load_weights_from_slices`` loads **6/6** with **0 unmatched**.

Two further tests are guards, not predicates: one proves the "0 unmatched"
counter can report non-zero (a counter that cannot fail is not a measurement),
and one proves this run imports no ``vllm_neuron`` module, which is what keeps
the helper runnable off-host.

**Two mechanical notes.**

*Import by path, on purpose.* ``test/`` has no ``__init__.py``, so pytest
prepends ``<root>/test`` to ``sys.path`` and ``test/vllm_neuron/`` would be
importable as a top-level ``vllm_neuron`` package -- the plugin's own name.
Loading the helper from its file path under a distinct module name avoids that
collision entirely and keeps the import-hygiene check below unambiguous: any
``vllm_neuron`` entry in ``sys.modules`` is then a real plugin import, never
this tree.

*Measured values are written out.* pytest captures stdout for passing tests, so
each predicate's measured value is also written to a machine-readable JSON file
(``$VLLM_NEURON_INC002_RESULTS_JSON``, else a fixed path in the temp dir). The
assertions are the gate; the file makes the numbers auditable after a green run.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

# --------------------------------------------------------------------------- #
# Load the helper under test from its path (see the module docstring).
# --------------------------------------------------------------------------- #

_HELPER_PATH = (
    Path(__file__).resolve().parents[1] / "vllm_neuron" / "model" / "utils.py"
)
_spec = importlib.util.spec_from_file_location("fork_test_model_utils", _HELPER_PATH)
assert _spec is not None and _spec.loader is not None, f"cannot load {_HELPER_PATH}"
model_test_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(model_test_utils)

FakeSafeSlice = model_test_utils.FakeSafeSlice
hf_state_to_fake_slices = model_test_utils.hf_state_to_fake_slices
load_weights_from_slices = model_test_utils.load_weights_from_slices

# --------------------------------------------------------------------------- #
# Measured-value sink.
# --------------------------------------------------------------------------- #

_RESULTS_PATH = Path(
    os.environ.get("VLLM_NEURON_INC002_RESULTS_JSON")
    or Path(tempfile.gettempdir()) / "vllm_neuron_inc002_predicates.json"
)
_RESULTS: dict[str, Any] = {}
_RESULTS_PATH.write_text("{}\n")  # truncate stale values from an earlier run


def _record(**values: Any) -> None:
    _RESULTS.update(values)
    _RESULTS_PATH.write_text(
        json.dumps(_RESULTS, indent=2, sort_keys=True, default=str) + "\n"
    )


# --------------------------------------------------------------------------- #
# Fixtures: synthetic HF weights for one decoder layer, six keys.
# --------------------------------------------------------------------------- #

FP8_DTYPE = torch.float8_e4m3fn
ROUND_TRIP_SHAPE = (4, 8)
HIDDEN = 8
SHARD = 4
NUM_SHARDS = 2
LAYER_IDX = 3

HF_KEY_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.down_proj.weight",
    "input_layernorm.weight",
)
EXPECTED_KEYS = frozenset(
    f"model.layers.{LAYER_IDX}.{suffix}" for suffix in HF_KEY_SUFFIXES
)
# Parameter name -> checkpoint key, the shape the fork's mapping builders emit.
MAPPINGS = {
    "q_proj_weight": f"model.layers.{LAYER_IDX}.self_attn.q_proj.weight",
    "k_proj_weight": f"model.layers.{LAYER_IDX}.self_attn.k_proj.weight",
    "o_proj_weight": f"model.layers.{LAYER_IDX}.self_attn.o_proj.weight",
    "gate_proj_weight": f"model.layers.{LAYER_IDX}.mlp.gate_proj.weight",
    "down_proj_weight": f"model.layers.{LAYER_IDX}.mlp.down_proj.weight",
    "input_layernorm_weight": f"model.layers.{LAYER_IDX}.input_layernorm.weight",
}


def _seeded(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Deterministic, distinct-valued weights; fp8 targets get quantised once."""
    numel = 1
    for dim in shape:
        numel *= dim
    base = (torch.arange(numel, dtype=torch.float32).reshape(*shape) + 1.0) / 8.0
    return base.to(dtype)


def _requalified(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The same weights, already qualified under their checkpoint keys."""
    return {f"model.layers.{LAYER_IDX}.{key}": value for key, value in state.items()}


def _hf_state() -> dict[str, torch.Tensor]:
    return {
        "self_attn.q_proj.weight": _seeded(HIDDEN, HIDDEN),
        "self_attn.k_proj.weight": _seeded(HIDDEN, HIDDEN),
        "self_attn.o_proj.weight": _seeded(HIDDEN, HIDDEN),
        "mlp.gate_proj.weight": _seeded(HIDDEN, HIDDEN, dtype=FP8_DTYPE),
        "mlp.down_proj.weight": _seeded(HIDDEN, HIDDEN),
        "input_layernorm.weight": _seeded(HIDDEN),
    }


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    """Max abs diff in fp32: fp8 subtraction is unimplemented on CPU."""
    return float((left.to(torch.float32) - right.to(torch.float32)).abs().max())


def _bitwise_mismatches(left: torch.Tensor, right: torch.Tensor) -> int:
    lhs = left.contiguous().view(torch.uint8)
    rhs = right.contiguous().view(torch.uint8)
    return int((lhs != rhs).sum())


class _ShardLoader:
    """Duck-typed stand-in for the fork's ``SafetensorsWeightLoader``.

    Reproduces ``sharding_weight_loader``'s transform
    (``vllm_neuron/utils/weight_loader.py:194-230``), so driving it exercises the
    slice interface the real loaders use: ``get_shape()`` plus a tuple of slices.
    """

    def __init__(self, shard_dim: int, transposed: bool = False) -> None:
        self.shard_dim = shard_dim
        self.transposed = transposed

    def load(self, slices: list[Any], rank: int) -> torch.Tensor:
        assert len(slices) == 1, "single-tensor loader"
        slice_obj = slices[0]
        storage_dim = self.shard_dim
        if self.transposed and self.shard_dim in (0, 1):
            storage_dim = 1 - self.shard_dim
        start = (rank % NUM_SHARDS) * SHARD
        key: list[Any] = [slice(None)] * len(slice_obj.get_shape())
        key[storage_dim] = slice(start, start + SHARD)
        result = slice_obj[tuple(key)]
        if self.transposed:
            result = result.T
        return result.contiguous()


def _full_slice_loader(slices: list[Any], rank: int) -> torch.Tensor:
    """A bare ``(slices, rank) -> tensor`` callable: the whole tensor, as-is."""
    return slices[0][:].contiguous()


class _TinyLayer(torch.nn.Module):
    """Six parameters, pre-shaped to what each loader produces at rank R.

    Loader coverage is deliberate: two loader *objects*, one transposed-storage
    object, one bare callable, and two parameters with no loader at all (the
    identity path).
    """

    def __init__(self) -> None:
        super().__init__()
        self.q_proj_weight = torch.nn.Parameter(
            torch.zeros(SHARD, HIDDEN), requires_grad=False
        )
        self.k_proj_weight = torch.nn.Parameter(
            torch.zeros(HIDDEN, SHARD), requires_grad=False
        )
        self.o_proj_weight = torch.nn.Parameter(
            torch.zeros(SHARD, HIDDEN), requires_grad=False
        )
        self.gate_proj_weight = torch.nn.Parameter(
            torch.zeros(HIDDEN, HIDDEN, dtype=FP8_DTYPE), requires_grad=False
        )
        self.down_proj_weight = torch.nn.Parameter(
            torch.zeros(HIDDEN, HIDDEN), requires_grad=False
        )
        self.input_layernorm_weight = torch.nn.Parameter(
            torch.zeros(HIDDEN), requires_grad=False
        )
        self.q_proj_weight.weight_loader = _ShardLoader(shard_dim=0)
        self.k_proj_weight.weight_loader = _ShardLoader(shard_dim=1)
        self.o_proj_weight.weight_loader = _ShardLoader(shard_dim=0, transposed=True)
        self.gate_proj_weight.weight_loader = _full_slice_loader
        # down_proj_weight and input_layernorm_weight carry no loader on purpose.


# --------------------------------------------------------------------------- #
# Predicate 1 -- FakeSafeSlice round-trips fp8 bit-exact.
# --------------------------------------------------------------------------- #


@pytest.mark.fast
def test_fake_safe_slice_round_trips_fp8_bit_exact() -> None:
    source = _seeded(*ROUND_TRIP_SHAPE, dtype=FP8_DTYPE)
    assert source.dtype is FP8_DTYPE and tuple(source.shape) == ROUND_TRIP_SHAPE

    slice_obj = FakeSafeSlice(source)
    assert slice_obj.get_shape() == list(ROUND_TRIP_SHAPE)
    assert slice_obj.shape == ROUND_TRIP_SHAPE
    assert slice_obj.dtype is FP8_DTYPE

    round_tripped = slice_obj[:]
    max_abs_diff = _max_abs_diff(round_tripped, source)
    mismatches = _bitwise_mismatches(round_tripped, source)
    sub_slice_diff = _max_abs_diff(slice_obj[1:3, 2:5], source[1:3, 2:5])
    _record(
        fake_safe_slice_max_abs_diff=max_abs_diff,
        fake_safe_slice_bitwise_mismatches=mismatches,
        fake_safe_slice_sub_slice_max_abs_diff=sub_slice_diff,
        fake_safe_slice_dtype=str(round_tripped.dtype),
        fake_safe_slice_shape=list(round_tripped.shape),
    )

    assert max_abs_diff == 0.0, f"max abs diff {max_abs_diff} != 0.0"
    assert mismatches == 0, f"{mismatches} of {source.numel()} bytes differ"
    assert round_tripped.dtype is FP8_DTYPE
    assert tuple(round_tripped.shape) == ROUND_TRIP_SHAPE
    assert sub_slice_diff == 0.0, f"sub-slice max abs diff {sub_slice_diff} != 0.0"
    # A read returns fresh memory, so a mutating loader cannot corrupt the source.
    assert round_tripped.data_ptr() != source.data_ptr()


# --------------------------------------------------------------------------- #
# Predicate 2 -- hf_state_to_fake_slices maps exactly 6/6 keys.
# --------------------------------------------------------------------------- #


@pytest.mark.fast
def test_hf_state_to_fake_slices_maps_six_of_six() -> None:
    state = _hf_state()
    assert len(state) == 6, "fixture must offer exactly 6 HF keys"

    slice_map = hf_state_to_fake_slices(state, LAYER_IDX)
    mapped = len(slice_map)
    unexpected = sorted(set(slice_map) - EXPECTED_KEYS)
    absent = sorted(EXPECTED_KEYS - set(slice_map))
    bad_type = [
        key for key, value in slice_map.items() if not isinstance(value, FakeSafeSlice)
    ]
    shape_mismatches = [
        suffix
        for suffix, tensor in state.items()
        if slice_map[f"model.layers.{LAYER_IDX}.{suffix}"].get_shape()
        != list(tensor.shape)
    ]
    _record(
        hf_state_keys_in=len(state),
        hf_state_keys_mapped=mapped,
        hf_state_keys_unexpected=len(unexpected),
        hf_state_keys_absent=len(absent),
        hf_state_non_slice_values=len(bad_type),
        hf_state_shape_mismatches=len(shape_mismatches),
    )

    assert mapped == 6, f"mapped {mapped}/6 keys"
    assert not unexpected, f"keys outside the expected set: {unexpected}"
    assert not absent, f"expected keys never produced: {absent}"
    assert not bad_type, f"values that are not FakeSafeSlice: {bad_type}"
    assert not shape_mismatches, f"slices whose shape drifted: {shape_mismatches}"

    # Idempotent for already-qualified keys, and loud for another layer's keys.
    requalified = hf_state_to_fake_slices(_requalified(state), LAYER_IDX)
    assert set(requalified) == EXPECTED_KEYS
    with pytest.raises(ValueError):
        hf_state_to_fake_slices(
            {f"model.layers.{LAYER_IDX + 1}.mlp.down_proj.weight": _seeded(HIDDEN)},
            LAYER_IDX,
        )


# --------------------------------------------------------------------------- #
# Predicate 3 -- load_weights_from_slices loads 6/6 with 0 unmatched.
# --------------------------------------------------------------------------- #


@pytest.mark.fast
@pytest.mark.parametrize("rank", [0, 1])
def test_load_weights_from_slices_loads_six_of_six(rank: int) -> None:
    state = _hf_state()
    slice_map = hf_state_to_fake_slices(state, LAYER_IDX)
    module = _TinyLayer()
    assert len(list(module.named_parameters())) == 6

    result = load_weights_from_slices(
        module, slice_map, MAPPINGS, rank, torch.device("cpu")
    )

    start = rank * SHARD
    expected = {
        "q_proj_weight": state["self_attn.q_proj.weight"][start : start + SHARD, :],
        "k_proj_weight": state["self_attn.k_proj.weight"][:, start : start + SHARD],
        "o_proj_weight": state["self_attn.o_proj.weight"][:, start : start + SHARD].T,
        "gate_proj_weight": state["mlp.gate_proj.weight"],
        "down_proj_weight": state["mlp.down_proj.weight"],
        "input_layernorm_weight": state["input_layernorm.weight"],
    }
    diffs = {
        name: _max_abs_diff(result.state_dict[name], want)
        for name, want in expected.items()
        if name in result.state_dict
    }
    param_diffs = {
        name: _max_abs_diff(getattr(module, name).data, want)
        for name, want in expected.items()
    }
    worst = max([*diffs.values(), *param_diffs.values()])
    _record(
        **{
            f"load_weights_rank{rank}_params": len(list(module.named_parameters())),
            f"load_weights_rank{rank}_loaded": result.num_loaded,
            f"load_weights_rank{rank}_slice_map_keys": len(slice_map),
            f"load_weights_rank{rank}_missing": len(result.missing_keys),
            f"load_weights_rank{rank}_unexpected": len(result.unexpected_keys),
            f"load_weights_rank{rank}_unmatched": len(result.unmatched_keys),
            f"load_weights_rank{rank}_worst_max_abs_diff": worst,
            f"load_weights_rank{rank}_fp8_bitwise_mismatches": _bitwise_mismatches(
                result.state_dict["gate_proj_weight"], state["mlp.gate_proj.weight"]
            ),
        }
    )

    assert result.num_loaded == 6, f"loaded {result.num_loaded}/6"
    assert len(slice_map) == 6
    assert result.missing_keys == [], f"missing: {result.missing_keys}"
    assert result.unexpected_keys == [], f"unexpected: {result.unexpected_keys}"
    assert len(result.unmatched_keys) == 0, f"unmatched: {result.unmatched_keys}"
    assert worst == 0.0, f"a loaded tensor drifted: {diffs} / {param_diffs}"
    # The rank actually reached the loader: rank 1's shard is not rank 0's rows.
    assert torch.equal(
        result.state_dict["q_proj_weight"],
        state["self_attn.q_proj.weight"][start : start + SHARD, :],
    )
    # fp8 survived the loader path bit-exact.
    assert result.state_dict["gate_proj_weight"].dtype is FP8_DTYPE
    assert (
        _bitwise_mismatches(
            result.state_dict["gate_proj_weight"], state["mlp.gate_proj.weight"]
        )
        == 0
    )


# --------------------------------------------------------------------------- #
# Guards.
# --------------------------------------------------------------------------- #


@pytest.mark.fast
def test_unmatched_counter_can_report_non_zero() -> None:
    """"0 unmatched" is only a measurement if the counter can report otherwise."""
    state = _hf_state()
    slice_map = hf_state_to_fake_slices(state, LAYER_IDX)
    slice_map["model.layers.3.mlp.up_proj.weight"] = FakeSafeSlice(_seeded(HIDDEN))
    broken = dict(MAPPINGS)
    broken["down_proj_weight"] = "model.layers.3.mlp.absent.weight"

    result = load_weights_from_slices(
        _TinyLayer(), slice_map, broken, 0, "cpu", strict=False
    )
    _record(
        guard_loaded=result.num_loaded,
        guard_missing=len(result.missing_keys),
        guard_unexpected=len(result.unexpected_keys),
        guard_unmatched=len(result.unmatched_keys),
    )

    assert result.num_loaded == 5
    assert result.missing_keys == ["down_proj_weight"]
    assert sorted(result.unexpected_keys) == [
        "model.layers.3.mlp.down_proj.weight",
        "model.layers.3.mlp.up_proj.weight",
    ]
    assert len(result.unmatched_keys) == 3
    with pytest.raises(RuntimeError, match="Checkpoint key"):
        load_weights_from_slices(_TinyLayer(), slice_map, broken, 0, "cpu")


@pytest.mark.fast
def test_acceptance_imports_no_vllm_neuron_module() -> None:
    """Arm A: the helper is free of plugin imports, so it runs off-host."""
    plugin_modules = sorted(
        name
        for name in sys.modules
        if name == "vllm_neuron" or name.startswith("vllm_neuron.")
    )
    helper_source = _HELPER_PATH.read_text()
    _record(
        vllm_neuron_modules_imported=len(plugin_modules),
        vllm_neuron_modules=plugin_modules,
        helper_path=str(_HELPER_PATH),
    )
    assert plugin_modules == [], f"plugin modules imported: {plugin_modules}"
    for line in helper_source.splitlines():
        stripped = line.strip()
        assert not stripped.startswith(("import vllm_neuron", "from vllm_neuron")), (
            f"helper imports the plugin: {stripped!r}"
        )
