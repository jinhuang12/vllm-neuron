"""The synthetic-weight helpers in ``test/vllm_neuron/model/utils.py``.

Every module test drives its synthetic weights through this helper, so a silent
defect there shows up as a numerics failure in some unrelated model test.
Covered here: ``FakeSafeSlice`` round-trips an fp8 tensor bit-exact,
``hf_state_to_fake_slices`` maps HuggingFace keys onto checkpoint keys, and
``load_weights_from_slices`` loads a tiny layer through the fork's real loader
interface -- including the missing-key, unexpected-key and strict-mode paths.

The helper is loaded from its file path under a distinct module name: ``test/``
has no ``__init__.py``, so pytest prepends ``<root>/test`` to ``sys.path`` and
``test/vllm_neuron/`` would otherwise be importable under the plugin's own name.
Loading it by path keeps the import-hygiene check below unambiguous -- any
``vllm_neuron`` entry in ``sys.modules`` is then a real plugin import, never this
tree.
"""

from __future__ import annotations

import importlib.util
import sys
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


def _plugin_modules() -> set[str]:
    return {
        name
        for name in sys.modules
        if name == "vllm_neuron" or name.startswith("vllm_neuron.")
    }


# Measured as a delta around the import, not as an absolute snapshot: another test
# in the same process may legitimately have imported the plugin already, and this
# reading is about what the helper itself pulls in.
_plugin_before_helper = _plugin_modules()
_spec.loader.exec_module(model_test_utils)
PLUGIN_MODULES_THE_HELPER_IMPORTED = sorted(_plugin_modules() - _plugin_before_helper)

FakeSafeSlice = model_test_utils.FakeSafeSlice
hf_state_to_fake_slices = model_test_utils.hf_state_to_fake_slices
load_weights_from_slices = model_test_utils.load_weights_from_slices

# --------------------------------------------------------------------------- #
# Synthetic HF weights for one decoder layer.
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


@pytest.mark.fast
def test_fake_safe_slice_round_trips_fp8_bit_exact() -> None:
    """Reading a slice back gives the same fp8 bytes, whole or sub-sliced."""
    source = _seeded(*ROUND_TRIP_SHAPE, dtype=FP8_DTYPE)

    slice_obj = FakeSafeSlice(source)
    assert slice_obj.get_shape() == list(ROUND_TRIP_SHAPE)
    assert slice_obj.shape == ROUND_TRIP_SHAPE
    assert slice_obj.dtype is FP8_DTYPE

    round_tripped = slice_obj[:]
    max_abs_diff = _max_abs_diff(round_tripped, source)
    mismatches = _bitwise_mismatches(round_tripped, source)
    sub_slice_diff = _max_abs_diff(slice_obj[1:3, 2:5], source[1:3, 2:5])

    assert max_abs_diff == 0.0, f"max abs diff {max_abs_diff} != 0.0"
    assert mismatches == 0, f"{mismatches} of {source.numel()} bytes differ"
    assert round_tripped.dtype is FP8_DTYPE
    assert tuple(round_tripped.shape) == ROUND_TRIP_SHAPE
    assert sub_slice_diff == 0.0, f"sub-slice max abs diff {sub_slice_diff} != 0.0"
    # A read returns fresh memory, so a mutating loader cannot corrupt the source.
    assert round_tripped.data_ptr() != source.data_ptr()


@pytest.mark.fast
def test_hf_state_to_fake_slices_maps_every_key() -> None:
    """Every HF key becomes a qualified checkpoint key holding a ``FakeSafeSlice``."""
    state = _hf_state()

    slice_map = hf_state_to_fake_slices(state, LAYER_IDX)
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


@pytest.mark.fast
@pytest.mark.parametrize("rank", [0, 1])
def test_load_weights_from_slices_loads_every_mapped_parameter(rank: int) -> None:
    """Each parameter gets its rank's shard, bit-exact and with nothing left over."""
    state = _hf_state()
    slice_map = hf_state_to_fake_slices(state, LAYER_IDX)
    module = _TinyLayer()

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
        name: _max_abs_diff(result.state_dict[name], tensor)
        for name, tensor in expected.items()
        if name in result.state_dict
    }
    param_diffs = {
        name: _max_abs_diff(getattr(module, name).data, tensor)
        for name, tensor in expected.items()
    }
    worst = max([*diffs.values(), *param_diffs.values()])

    assert result.num_loaded == len(MAPPINGS), f"loaded {result.num_loaded}"
    assert result.missing_keys == [], f"missing: {result.missing_keys}"
    assert result.unexpected_keys == [], f"unexpected: {result.unexpected_keys}"
    assert result.unmatched_keys == [], f"unmatched: {result.unmatched_keys}"
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


@pytest.mark.fast
def test_load_weights_reports_missing_and_unexpected_keys() -> None:
    """A mapping that points at an absent key is reported, and strict mode raises."""
    state = _hf_state()
    slice_map = hf_state_to_fake_slices(state, LAYER_IDX)
    slice_map["model.layers.3.mlp.up_proj.weight"] = FakeSafeSlice(_seeded(HIDDEN))
    broken = dict(MAPPINGS)
    broken["down_proj_weight"] = "model.layers.3.mlp.absent.weight"

    result = load_weights_from_slices(
        _TinyLayer(), slice_map, broken, 0, "cpu", strict=False
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
def test_helper_imports_no_plugin_module() -> None:
    """The helper pulls in no ``vllm_neuron`` module, so it runs off-host."""
    assert PLUGIN_MODULES_THE_HELPER_IMPORTED == [], (
        f"importing the helper pulled in {PLUGIN_MODULES_THE_HELPER_IMPORTED}"
    )
    for line in _HELPER_PATH.read_text().splitlines():
        stripped = line.strip()
        assert not stripped.startswith(("import vllm_neuron", "from vllm_neuron")), (
            f"helper imports the plugin: {stripped!r}"
        )
