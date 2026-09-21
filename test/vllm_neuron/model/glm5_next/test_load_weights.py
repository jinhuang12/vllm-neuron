# SPDX-License-Identifier: Apache-2.0
"""``Glm5NextForConditionalGeneration.load_weights`` on a miniature checkpoint.

The tests write a real safetensors file for a four-layer miniature of the
model and load it through the pipelined reader on the CPU, so the attached
loaders, the out-of-band scale read and the load-time preps all run. Two
miniature configurations differ in one field: all layers dense (the load
completes) or one dense layer then routed banks. The fixtures and writers
here are shared by the sibling ``test_*`` modules in this directory."""

from __future__ import annotations

import ast
import contextlib
import inspect
import json
import logging
import math
import textwrap
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import vllm_neuron
from vllm_neuron.model.glm5_next.config import (
    DSA_LAYER_TYPE,
    Glm5NextConfig,
    Glm5NextTextConfig,
)
from vllm_neuron.model.glm5_next.model_fp8 import (
    Glm5NextForConditionalGeneration,
    Glm5NextSharedExpertRouteError,
    Glm5NextSharedExperts,
    Glm5NextWeightLoadError,
    _is_fp8_dtype,
)
from vllm_neuron.model.glm5_next.quantization import DEFAULT_WEIGHT_BLOCK_SIZE
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
    DSA_SCALED_PROJECTIONS,
    FP8_SCALE_SUFFIX,
    Glm5NextExpertBankNotLoadableError,
    Glm5NextWeightMapError,
    MAPPED_KEY_PLAIN,
    MAPPED_KEY_QUANTISED_WEIGHT,
    MAPPED_KEY_SCALE_GRID,
    MAPPED_KEY_STACKED_BANK,
    block_grid_shape,
    blockwise_scale_loader,
    build_weight_mappings,
    classify_mapped_keys,
    scale_keys,
)
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

from test.vllm_neuron.model.glm5_next.fixtures import weight_index


FIXTURES_DIR = Path(__file__).parent / "fixtures"
REAL_CONFIG_PATH = FIXTURES_DIR / "hf-config.json"

#: Layer 45 is the multi-token-prediction head and ``model.visual`` the vision
#: tower; the language model loads neither.
MTP_LAYER_PREFIX = "model.language_model.layers.45."
VISION_PREFIX = "model.visual."

#: Key counts of the published checkpoint index, whole and by family.
REAL_INDEX_TOTAL_KEYS = 76_108
REAL_INDEX_MTP_KEYS = 1_760
REAL_INDEX_VISION_KEYS = 347
REAL_INDEX_IN_SCOPE_KEYS = 74_001

MINI_LAYERS = 4
MINI_ROUTED_EXPERTS = 4
MINI_SHARED_EXPERTS = 1
MINI_FIRST_K_DENSE = 1

#: With every layer dense no routed bank is built, so the load completes.
MINI_ALL_DENSE_FIRST_K = MINI_LAYERS

#: One whole 128x128 quantisation block, so its scale grid is a single value.
MINI_WEIGHT_SHAPE = (128, 128)
MINI_SCALE_SHAPE = (1, 1)
MINI_PLAIN_SHAPE = (4,)

#: Leaves the checkpoint stores in float32 rather than the config dtype.
FLOAT32_CHECKPOINT_LEAVES = ("A_log", "dt_bias")

#: ``(parameter leaf, checkpoint leaf)``; the map renames the router bias.
FLOAT32_MIX_FAMILIES = (
    ("hc_attn_base", "hc_attn_base"),
    ("hc_attn_scale", "hc_attn_scale"),
    ("hc_ffn_base", "hc_ffn_base"),
    ("hc_ffn_scale", "hc_ffn_scale"),
    ("router_bias", "e_score_correction_bias"),
)

FLOAT32_MIX_CHECKPOINT_LEAVES = tuple(
    checkpoint_leaf for _, checkpoint_leaf in FLOAT32_MIX_FAMILIES
)

MINI_CHECKPOINT_FILE = "model.safetensors"

#: Shrunk so every MLA projection's closed form fits one 128x128 block.
MINI_MLA_WIDTHS = dict(
    hidden_size=128,
    num_attention_heads=4,
    qk_nope_head_dim=16,
    qk_rope_head_dim=0,
    v_head_dim=16,
    q_lora_rank=32,
    kv_lora_rank=32,
)


def _mini_config(first_k_dense: int) -> Glm5NextConfig:
    """The miniature quantised config; ``first_k_dense`` is the one field the two variants differ in."""
    return Glm5NextConfig(
        text_config=Glm5NextTextConfig(
            num_hidden_layers=MINI_LAYERS,
            n_routed_experts=MINI_ROUTED_EXPERTS,
            n_shared_experts=MINI_SHARED_EXPERTS,
            first_k_dense_replace=first_k_dense,
            tie_word_embeddings=False,
            **MINI_MLA_WIDTHS,
        )
    )


def _dense_config() -> Glm5NextConfig:
    """Every layer dense, so every map entry has a loader and the load completes."""
    return _mini_config(MINI_ALL_DENSE_FIRST_K)


def _routed_config() -> Glm5NextConfig:
    """One dense layer, then routed expert banks."""
    return _mini_config(MINI_FIRST_K_DENSE)


def _dense_model() -> Glm5NextForConditionalGeneration:
    return Glm5NextForConditionalGeneration(_dense_config())


def _routed_model() -> Glm5NextForConditionalGeneration:
    return Glm5NextForConditionalGeneration(_routed_config())


def _mappings_for(config: Glm5NextConfig) -> dict[str, str | list[str]]:
    """The weight map built from the same two config members ``load_weights`` reads."""
    return build_weight_mappings(
        config.text_config,
        quantised=config.is_block_quantized,
        modules_to_not_convert=tuple(config.modules_to_not_convert or ()),
    )


def _is_scale_key(key: str) -> bool:
    return key.endswith(f".{FP8_SCALE_SUFFIX}")


def _keys_of(mappings: dict[str, str | list[str]], name: str) -> list[str]:
    """One map entry's checkpoint keys as a list; a lone key is stored as a bare string."""
    keys = mappings[name]
    return [keys] if isinstance(keys, str) else list(keys)


def _mla_key_overrides(
    model: Glm5NextForConditionalGeneration,
    mappings: dict[str, str | list[str]],
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Real ``(shape, dtype)`` per MLA checkpoint key.

    ``prepare_projection_weights`` checks each projection against the module's
    own ``projection_widths()``, so these cannot be written at a placeholder
    shape. The scaled projections are fp8 with an fp32 grid; ``kv_b_proj`` is
    bf16 with no grid, as in the published checkpoint.
    """
    overrides: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
    for path, module in model.named_modules():
        if not hasattr(type(module), "projection_widths"):
            continue
        for name, idim, odim in module.projection_widths():
            quantised = name in DSA_SCALED_PROJECTIONS
            attribute = getattr(type(module), "PROJECTION_PARAMETERS", {}).get(
                name, f"{name}_weight"
            )
            weight_param = f"{path}.{attribute}"
            if weight_param not in mappings:
                continue
            for key in _keys_of(mappings, weight_param):
                overrides[key] = (
                    (odim, idim),
                    torch.float8_e4m3fn if quantised else torch.bfloat16,
                )
            if not quantised:
                continue
            scale_param = f"{path}.{name}_{FP8_SCALE_SUFFIX}"
            grid = block_grid_shape((odim, idim), DEFAULT_WEIGHT_BLOCK_SIZE)
            for key in _keys_of(mappings, scale_param):
                overrides[key] = (grid, torch.float32)
    return overrides


def _checkpoint_plain_dtype(name: str) -> torch.dtype:
    """The dtype the checkpoint stores one unquantised tensor in, by leaf name."""
    leaf = name.rsplit(".", 1)[-1]
    return (
        torch.float32
        if leaf in FLOAT32_CHECKPOINT_LEAVES + FLOAT32_MIX_CHECKPOINT_LEAVES
        else torch.bfloat16
    )


def _write_miniature_checkpoint(
    directory: Path,
    mappings: dict[str, str | list[str]],
    model: Glm5NextForConditionalGeneration,
    extra_overrides: dict[str, torch.Tensor] | None = None,
) -> int:
    """Write a real safetensors file holding one tensor per mapped key; return the count.

    Tensors are typed the way the checkpoint types them: fp8 bytes for a
    quantised weight, fp32 for a scale grid, the checkpoint dtype otherwise.
    Every tensor is constant unless ``extra_overrides`` supplies a whole tensor,
    which a test that asks which rows a rank received must do.
    """
    overrides = _mla_key_overrides(model, mappings)
    tensors: dict[str, torch.Tensor] = {}
    for keys in mappings.values():
        key_list = [keys] if isinstance(keys, str) else list(keys)
        quantised_pair = classify_mapped_keys(keys) in (
            MAPPED_KEY_QUANTISED_WEIGHT,
            MAPPED_KEY_STACKED_BANK,
        )
        for key in key_list:
            if key in tensors:
                continue
            if extra_overrides and key in extra_overrides:
                tensors[key] = extra_overrides[key]
            elif key in overrides:
                shape, dtype = overrides[key]
                tensors[key] = (
                    torch.ones(shape, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                    if dtype is torch.float8_e4m3fn
                    else torch.full(shape, 0.5, dtype=dtype)
                )
            elif _is_scale_key(key):
                tensors[key] = torch.full(
                    MINI_SCALE_SHAPE, 0.5, dtype=torch.float32
                )
            elif quantised_pair:
                tensors[key] = torch.ones(
                    MINI_WEIGHT_SHAPE, dtype=torch.bfloat16
                ).to(torch.float8_e4m3fn)
            else:
                tensors[key] = torch.ones(
                    MINI_PLAIN_SHAPE, dtype=_checkpoint_plain_dtype(key)
                )
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(directory / MINI_CHECKPOINT_FILE))
    return len(tensors)


def _implied_numels(
    directory: Path, mappings: dict[str, str | list[str]]
) -> dict[str, int]:
    """Element count each mapped parameter's own checkpoint slices imply.

    Read back from the file. A weight-plus-scale entry counts its weight keys
    only, because the loader reads the scale out of band; a lone scale entry
    counts itself.
    """
    implied: dict[str, int] = {}
    with safe_open(str(directory / MINI_CHECKPOINT_FILE), framework="pt") as opened:
        present = set(opened.keys())
        for name in mappings:
            key_list = _keys_of(mappings, name)
            if not set(key_list) <= present:
                continue
            weights = [k for k in key_list if not _is_scale_key(k)]
            population = weights or key_list
            implied[name] = sum(
                math.prod(opened.get_slice(k).get_shape()) for k in population
            )
    return implied


def _not_lazy_count(model: torch.nn.Module) -> int:
    """How many declared parameters hold a real tensor rather than a lazy placeholder."""
    return sum(
        1
        for _, param in model.named_parameters()
        if not torch.nn.parameter.is_lazy(param)
    )


def _token_checkpoint_directory(tmp_path: Path) -> Path:
    """A directory with one tiny safetensors file, so ``load_weights`` passes its file-count check."""
    directory = tmp_path / "token-checkpoint"
    directory.mkdir()
    save_file(
        {"token": torch.zeros(1, dtype=torch.float32)},
        str(directory / "model-00001-of-00001.safetensors"),
    )
    return directory


@pytest.fixture
def single_rank_process_group(tmp_path):
    """A one-rank gloo process group; ``load_sharded_pipelined`` reads the default store."""
    if torch.distributed.is_initialized():
        yield
        return
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{tmp_path / 'pg-rendezvous'}",
        world_size=1,
        rank=0,
    )
    try:
        yield
    finally:
        torch.distributed.destroy_process_group()


@pytest.fixture
def keep_the_loaded_tensors(monkeypatch):
    """Hold the post-prep release off, so the loaded weights and grids stay readable."""
    from vllm_neuron.model.glm5_next import model_fp8

    monkeypatch.setattr(
        model_fp8, "_release_replaced_parameters", lambda module, *names: 0
    )


def test_every_declared_parameter_is_materialised_and_loaded(
    keep_the_loaded_tensors, tmp_path, single_rank_process_group
) -> None:
    """An all-dense load fills every declared parameter with its own slice count; a bank with no geometry refuses cleanly."""
    dense_dir = tmp_path / "dense"
    model = _dense_model()
    n = len(model.declared_parameter_names())
    assert n > 0, "the dense miniature tree declares no parameter; nothing to load"

    mappings = _mappings_for(_dense_config())
    written = _write_miniature_checkpoint(dense_dir, mappings, model)
    assert written > 0, "the dense miniature checkpoint holds no tensor"

    before = _not_lazy_count(model)
    model.load_weights(str(dense_dir), torch.device("cpu"), None)
    after = _not_lazy_count(model)

    assert before == 0, (
        f"{before} of {n} parameters already held a real tensor before the load"
    )
    assert after == n, (
        f"{n - after} of {n} declared parameters still hold an unfilled "
        f"placeholder after the load"
    )
    assert len(model.declared_parameter_names()) == n, (
        "the declared-name set changed across the load"
    )

    implied = _implied_numels(dense_dir, mappings)
    loaded = {name: param.numel() for name, param in model.named_parameters()}
    checked = sorted(set(implied) & set(loaded))
    mismatches = {
        name: (loaded[name], implied[name])
        for name in checked
        if loaded[name] != implied[name]
    }

    assert len(checked) == n, (
        f"only {len(checked)} of the {n} declared parameters are both mapped and "
        f"written; unmapped or unwritten: {sorted(set(loaded) - set(implied))[:5]}"
    )
    assert mismatches == {}, (
        f"{len(mismatches)} parameters received fewer or more elements than "
        f"their own checkpoint slices imply, e.g. "
        f"{[(k, *mismatches[k]) for k in sorted(mismatches)[:3]]} as "
        f"(parameter, loaded, implied)"
    )

    routed_dir = tmp_path / "routed"
    routed = _routed_model()
    routed_declared = len(routed.declared_parameter_names())
    routed_mappings = _mappings_for(_routed_config())
    assert _write_miniature_checkpoint(routed_dir, routed_mappings, routed) > 0

    banks = {
        name: _keys_of(routed_mappings, name)
        for name in routed_mappings
        if len([k for k in _keys_of(routed_mappings, name) if _is_scale_key(k)]) > 1
    }
    assert banks, (
        "the routed configuration produced no multi-scale-key entry, so the bank "
        "refusal cannot fire"
    )

    # Assigning None on the instance shadows the method, so the loader's
    # ``callable(getattr(...))`` check reads the geometry as absent.
    withheld = 0
    for _, module in routed.named_modules():
        if callable(getattr(module, "local_expert_indices", None)):
            module.local_expert_indices = None
            withheld += 1
    assert withheld > 0, (
        "no module in this tree declares local_expert_indices, so withholding it "
        "changed nothing"
    )

    with pytest.raises(Glm5NextExpertBankNotLoadableError) as raised:
        routed.load_weights(str(routed_dir), torch.device("cpu"), None)
    message = str(raised.value)
    named = [name for name in banks if name in message]

    assert named, f"the refusal names no bank parameter: {message}"
    key_count = len(banks[named[0]])
    assert str(key_count) in message, (
        f"the refusal does not report the key count {key_count} of the "
        f"parameter it named: {message}"
    )
    assert "declares no expert geometry" in message.lower(), (
        f"the refusal does not name the missing declaration, so a reader cannot "
        f"tell this refusal from any other bank refusal: {message}"
    )
    assert _not_lazy_count(routed) == 0, (
        "the refusal left parameters holding real tensors"
    )
    assert len(list(routed.named_parameters())) == 0, (
        f"the refusal left {len(list(routed.named_parameters()))} materialised "
        f"placeholders behind out of {routed_declared} declared, so the tree is "
        f"half-built and a later load cannot tell it from a fresh one"
    )


class _MapCaptured(Exception):
    """Raised by the observer once it holds the map ``load_weights`` hands to the reader."""


def test_the_map_load_weights_hands_over_covers_the_in_scope_index(
    tmp_path,
    monkeypatch,
) -> None:
    """The map handed to the reader claims every in-scope checkpoint key and nothing outside the index."""
    weight_map = weight_index.weight_map()
    total = len(weight_map)
    mtp = {k for k in weight_map if k.startswith(MTP_LAYER_PREFIX)}
    vision = {k for k in weight_map if k.startswith(VISION_PREFIX)}
    in_scope = set(weight_map) - mtp - vision

    assert total == REAL_INDEX_TOTAL_KEYS
    assert len(mtp) == REAL_INDEX_MTP_KEYS
    assert len(vision) == REAL_INDEX_VISION_KEYS
    assert len(in_scope) == REAL_INDEX_IN_SCOPE_KEYS
    assert len(in_scope) + len(mtp) + len(vision) == total, (
        "the three parts do not sum to the whole, so a family is being counted "
        "twice or not at all"
    )

    real_config = Glm5NextConfig.from_configs(
        json.loads(REAL_CONFIG_PATH.read_text())
    )
    model = Glm5NextForConditionalGeneration(real_config)

    observed: list[dict[str, str | list[str]]] = []

    def observer(_checkpoint, rank, world_size, _model, handed_over, _device):
        observed.append(handed_over)
        raise _MapCaptured(
            f"captured the handed-over map: {len(handed_over)} entries, "
            f"rank {rank} of world size {world_size}"
        )

    monkeypatch.setattr(
        SafetensorsCheckpoint, "load_sharded_pipelined", observer, raising=True
    )

    with pytest.raises(_MapCaptured):
        model.load_weights(
            str(_token_checkpoint_directory(tmp_path)),
            torch.device("cpu"),
            None,
        )

    materialised_at_handover = len(list(model.named_parameters()))

    assert len(observed) == 1, (
        f"the observer recorded {len(observed)} maps, not the one load_weights built"
    )
    assert materialised_at_handover > 0, (
        "the hand-over was reached with an unmaterialised tree; the reader must "
        "run after materialisation"
    )

    mappings = observed[0]

    bank_suffixes = ("gate_proj_weight", "up_proj_weight", "down_proj_weight")
    banks = {
        name: keys
        for name, keys in mappings.items()
        if ".mlp.experts." in name and name.endswith(bank_suffixes)
    }
    experts = real_config.text_config.n_routed_experts
    bank_key_counts = sorted({len(keys) for keys in banks.values()})
    bank_scale_counts = sorted({len(scale_keys(keys)) for keys in banks.values()})

    assert banks, "the captured map holds no routed expert bank entry"
    assert experts == 288, (
        f"the published config declares {experts} routed experts, not the 288 "
        f"the counts below assume"
    )
    assert bank_key_counts == [2 * experts], (
        f"a bank entry does not carry one weight and one scale key per expert: "
        f"{bank_key_counts} against {2 * experts}"
    )
    assert bank_scale_counts == [experts], (
        f"a bank entry does not carry one scale key per expert: "
        f"{bank_scale_counts} against {experts}"
    )

    claimed: set[str] = set()
    for keys in mappings.values():
        claimed.update([keys] if isinstance(keys, str) else keys)

    unclaimed = in_scope - claimed
    absent_from_index = claimed - set(weight_map)

    assert unclaimed == set(), (
        f"{len(unclaimed)} in-scope checkpoint keys are claimed by no mapping, "
        f"e.g. {sorted(unclaimed)[:5]}"
    )
    assert absent_from_index == set(), (
        f"{len(absent_from_index)} mapped keys are absent from the index, e.g. "
        f"{sorted(absent_from_index)[:5]}"
    )
    assert not (claimed & mtp), "the map claims a multi-token-prediction key"


def test_the_reader_iterates_no_parameter_until_they_are_materialised() -> None:
    """``named_parameters()`` is empty before materialisation and complete after it."""
    model = _dense_model()
    n = len(model.declared_parameter_names())
    assert n > 0

    before = len(list(model.named_parameters()))
    assert before == 0, (
        f"the reader would iterate {before} parameters on an unmaterialised "
        f"tree; materialisation must precede the load"
    )

    mappings = _mappings_for(_dense_config())
    materialised = model._materialise_declared_parameters(
        mappings, torch.device("cpu")
    )
    after = len(list(model.named_parameters()))

    assert materialised == n, (
        f"materialisation visited {materialised} parameters but the tree "
        f"declares {n}; the two walks have drifted apart"
    )
    assert after == n, (
        f"the reader would iterate {after} parameters after materialisation, "
        f"not the {n} declared"
    )


def test_an_absent_checkpoint_refuses_by_name_and_leaves_the_tree_alone(
    tmp_path,
) -> None:
    """A missing checkpoint path is refused by name before anything is materialised."""
    model = _dense_model()
    n = len(model.declared_parameter_names())
    missing = tmp_path / "there-is-no-checkpoint-here"

    with pytest.raises(Glm5NextWeightLoadError) as raised:
        model.load_weights(str(missing), torch.device("cpu"), None)

    assert str(missing) in str(raised.value), (
        f"the refusal does not name the path it was given: {raised.value}"
    )
    assert _not_lazy_count(model) == 0, (
        "the refusal left parameters holding real tensors, so the tree is "
        "half-built"
    )
    assert len(list(model.named_parameters())) == 0, (
        f"the refusal left {len(list(model.named_parameters()))} materialised "
        f"placeholders behind out of {n} declared"
    )


def _out_of_band_entries(mappings: dict[str, str | list[str]]) -> dict[str, str]:
    """Map entries whose scale the weight loader drops, as ``{param: scale key}``."""
    declared = {
        _keys_of(mappings, name)[0]
        for name in mappings
        if len(_keys_of(mappings, name)) == 1 and _is_scale_key(_keys_of(mappings, name)[0])
    }
    found: dict[str, str] = {}
    for name in mappings:
        keys = _keys_of(mappings, name)
        scales = [k for k in keys if _is_scale_key(k)]
        if len(keys) >= 2 and len(scales) == 1 and scales[0] not in declared:
            found[name] = scales[0]
    return found


def _scale_attribute_of(param_name: str) -> tuple[str, str]:
    """Module path and attribute name a dropped scale grid is stored under."""
    module_path, _, leaf = param_name.rpartition(".")
    base = leaf[: -len("_weight")] if leaf.endswith("_weight") else leaf
    return module_path, f"{base}_{FP8_SCALE_SUFFIX}"


def _derived_dsa_scale_names(config: Glm5NextConfig) -> set[str]:
    """The scale-grid parameter names the sparse-attention layers add, derived from the config."""
    indices = [
        index
        for index, kind in enumerate(config.text_config.layer_types)
        if kind == DSA_LAYER_TYPE
    ]
    return {
        f"model.layers.{index}.self_attn.{leaf}_{FP8_SCALE_SUFFIX}"
        for index in indices
        for leaf in DSA_SCALED_PROJECTIONS
    }


@contextlib.contextmanager
def _captured_cast_lines():
    """Collect the reader's dtype-mismatch log lines while a load runs."""
    collected: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            text = record.getMessage()
            if "Mismatch between parameter" in text and "casting to" in text:
                collected.append(text)

    handler = _Collect(level=logging.DEBUG)
    root = logging.getLogger()
    previous = root.level
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)
    try:
        yield collected
    finally:
        root.removeHandler(handler)
        root.setLevel(previous)


def _real_call_sites(function_name: str) -> int:
    """Count the ``ast.Call`` sites of ``function_name`` under ``vllm_neuron/``."""
    package = Path(vllm_neuron.__file__).parent
    real = 0
    for path in sorted(package.rglob("*.py")):
        text = path.read_text()
        if function_name not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            named = (
                callee.attr
                if isinstance(callee, ast.Attribute)
                else callee.id
                if isinstance(callee, ast.Name)
                else None
            )
            if named == function_name:
                real += 1
    return real


def _statement_positions(method, *, calls: tuple[str, ...], anchor: str):
    """Line of the earliest call to each named callee and of the latest ``anchor`` call in ``method``."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    found: dict[str, list[int]] = {name: [] for name in calls}
    anchor_found: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        named = callee.attr if isinstance(callee, ast.Attribute) else None
        if named in found:
            found[named].append(node.lineno)
        if named == anchor:
            anchor_found.append(node.lineno)

    positions = {name: (min(lines) if lines else -1) for name, lines in found.items()}
    anchor_at = max(anchor_found) if anchor_found else -1
    return positions, anchor_at


_LOAD_TIME_PREPS = ("prepare_projection_weights", "prepare_scale_operands")
_PREP_CALLER = "_run_load_time_preps"


def _prep_call_homes(source: str) -> dict[str, list[tuple[str, int]]]:
    """Every load-time prep call site in ``source`` with the innermost function it lives in."""
    homes: dict[str, list[tuple[str, int]]] = {name: [] for name in _LOAD_TIME_PREPS}
    tree = ast.parse(source)
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    def span(function) -> tuple[int, int]:
        return function.lineno, function.end_lineno or function.lineno

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func
        named = callee.attr if isinstance(callee, ast.Attribute) else None
        if named not in homes:
            continue
        containing = [
            function
            for function in functions
            if span(function)[0] <= node.lineno <= span(function)[1]
        ]
        innermost = min(containing, key=lambda f: span(f)[1] - span(f)[0])
        homes[named].append((innermost.name, node.lineno))
    return homes


def _every_prep_call_lives_in_the_caller(source: str) -> tuple[bool, dict]:
    """True when each prep is called exactly once and only from ``_PREP_CALLER``."""
    homes = _prep_call_homes(source)
    verdict = all(
        len(sites) == 1 and sites[0][0] == _PREP_CALLER for sites in homes.values()
    )
    return verdict, homes


def test_the_scale_grids_stay_fp32(
    keep_the_loaded_tensors, tmp_path, single_rank_process_group
) -> None:
    """Every scale grid, dropped or mapped, arrives fp32 with no cast; an absent scale key refuses by name."""
    model = _dense_model()
    mappings = _mappings_for(_dense_config())
    directory = tmp_path / "dense"
    _write_miniature_checkpoint(directory, mappings, model)

    dropped = _out_of_band_entries(mappings)
    assert dropped, "no map entry drops a scale, so there is no out-of-band grid"

    with _captured_cast_lines() as cast_lines:
        model.load_weights(str(directory), torch.device("cpu"), None)

    fp32 = 0
    for param_name in dropped:
        module_path, attribute = _scale_attribute_of(param_name)
        grid = getattr(model.get_submodule(module_path), attribute, None)
        assert grid is not None, (
            f"{param_name}'s scale was dropped by the loader and never read out "
            f"of band, so {module_path}.{attribute} does not exist"
        )
        assert grid.dtype is torch.float32, (
            f"{module_path}.{attribute} arrived {grid.dtype}, not fp32"
        )
        assert grid.device.type == "cpu", (
            f"{module_path}.{attribute} is on {grid.device}, not the target"
        )
        fp32 += 1
    assert fp32 == len(dropped)

    loaded = dict(model.named_parameters())
    through_map = [name for name in mappings if name.endswith(FP8_SCALE_SUFFIX)]
    assert through_map, "the map carries no scale-grid parameter at all"
    not_fp32 = [n for n in through_map if loaded[n].dtype is not torch.float32]
    assert not_fp32 == [], f"these scale grids are not fp32: {not_fp32[:4]}"

    assert len(cast_lines) == 0, (
        f"the load cast {len(cast_lines)} tensors to a placeholder dtype, e.g. "
        f"{cast_lines[0] if cast_lines else ''}"
    )
    control = _dense_model()
    control._placeholder_dtype = (
        lambda keys, *, param_name, mappings: control.text_config.torch_dtype
    )
    with _captured_cast_lines() as control_lines:
        control.load_weights(str(directory), torch.device("cpu"), None)
    assert len(control_lines) > 0, (
        "a load whose every placeholder took the config dtype cast nothing, so "
        "the zero above shows nothing"
    )

    quantised = set(build_weight_mappings(model.text_config, quantised=True))
    plain = set(build_weight_mappings(model.text_config, quantised=False))
    derived = _derived_dsa_scale_names(_dense_config())
    assert plain <= quantised, f"names only in plain: {sorted(plain - quantised)}"
    assert quantised - plain == derived, (
        f"symmetric difference: {sorted((quantised - plain) ^ derived)[:4]}"
    )

    victim_param, victim_key = sorted(dropped.items())[0]
    absent_key = f"absent.{victim_key}"
    tampered = dict(mappings)
    tampered[victim_param] = [
        key for key in _keys_of(mappings, victim_param) if key != victim_key
    ] + [absent_key]
    opened = SafetensorsCheckpoint(str(directory))
    with pytest.raises(Glm5NextWeightLoadError) as refusal:
        _dense_model()._load_out_of_band_scales(
            opened, tampered, torch.device("cpu")
        )
    message = str(refusal.value)
    assert absent_key in message, message
    assert victim_param in message, message
    assert "1.0" in message, (
        f"the refusal does not say what it refuses to do, which is the whole "
        f"reason it is not a default: {message}"
    )
    read = _dense_model()._load_out_of_band_scales(
        SafetensorsCheckpoint(str(directory)), mappings, torch.device("cpu")
    )
    assert read == len(dropped)


def test_the_load_time_preps_have_exactly_one_production_call_site_each() -> None:
    """Each load-time prep is called from exactly one place in the package."""
    for prep in _LOAD_TIME_PREPS:
        real = _real_call_sites(prep)
        assert real == 1, f"{prep} has {real} production call sites, not one"


def test_the_blockwise_scale_loader_arity_contract_is_honoured() -> None:
    """Every lone-grid entry in the real map carries one key, and the loader refuses two slices."""
    real_config = Glm5NextConfig.from_configs(json.loads(REAL_CONFIG_PATH.read_text()))
    mappings = _mappings_for(real_config)
    grids = [
        name
        for name in mappings
        if classify_mapped_keys(mappings[name]) == MAPPED_KEY_SCALE_GRID
    ]
    assert grids, "the real map has no scale-grid entry"

    wrong_arity = [name for name in grids if len(_keys_of(mappings, name)) != 1]
    assert wrong_arity == [], (
        f"{len(wrong_arity)} scale-grid entries carry a key count the loader "
        f"refuses, e.g. {wrong_arity[:4]}"
    )

    # The population's control: the same measurement over the entries that are
    # NOT scale grids finds multi-key ones, so "all one key" above is a property
    # of the grids and not of the map as a whole.
    others = [
        name
        for name in mappings
        if classify_mapped_keys(mappings[name]) != MAPPED_KEY_SCALE_GRID
        and len(_keys_of(mappings, name)) != 1
    ]
    assert others, (
        "no entry anywhere in the map carries more than one key, so the count "
        "above cannot distinguish a grid from anything else"
    )

    grid = torch.full(MINI_SCALE_SHAPE, 0.5, dtype=torch.float32)
    loader = blockwise_scale_loader(param_name=sorted(grids)[0])
    with pytest.raises(Glm5NextWeightMapError) as refusal:
        loader.load([_WholeTensorSlice(grid), _WholeTensorSlice(grid)], 0)
    message = str(refusal.value)
    assert "expects 1 slice" in message and "got 2" in message, message

    accepted = loader.load([_WholeTensorSlice(grid)], 0)
    assert accepted.dtype is torch.float32


class _WholeTensorSlice:
    """A stand-in for ``PySafeSlice`` that supports only ``slice[:]``."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def __getitem__(self, item):
        return self._tensor[item]


def test_the_load_time_preps_run_after_the_device_by_name(
    keep_the_loaded_tensors, tmp_path, single_rank_process_group
) -> None:
    """The preps run after ``load_state_dict`` and after the scale read, and both refusals name their cause."""
    positions, anchor = _statement_positions(
        Glm5NextForConditionalGeneration.load_weights,
        calls=("_run_load_time_preps", "_load_out_of_band_scales"),
        anchor="load_state_dict",
    )
    assert anchor > 0, "load_weights no longer calls load_state_dict at all"
    for name, position in positions.items():
        assert position > 0, f"load_weights no longer calls {name}"
        assert position > anchor, (
            f"{name} is called at statement {position}, before the anchor at "
            f"{anchor}; it would run on placeholders"
        )

    assert (
        positions["_load_out_of_band_scales"] < positions["_run_load_time_preps"]
    )

    module_source = Path(
        inspect.getsourcefile(Glm5NextForConditionalGeneration)
    ).read_text()
    lives_in_the_caller, homes = _every_prep_call_lives_in_the_caller(module_source)
    for prep in _LOAD_TIME_PREPS:
        sites = homes[prep]
        assert len(sites) == 1, f"{prep} has {len(sites)} call sites, not one: {sites}"
        enclosing, line = sites[0]
        assert enclosing == _PREP_CALLER, (
            f"{prep} is called from {enclosing!r} at line {line}, not from "
            f"{_PREP_CALLER!r}"
        )
    assert lives_in_the_caller

    shared = Glm5NextSharedExperts(_dense_config().text_config)
    with pytest.raises(Glm5NextSharedExpertRouteError) as case_a:
        shared.prepare_scale_operands(
            gate_proj_weight=torch.ones(MINI_WEIGHT_SHAPE, dtype=torch.bfloat16),
            up_proj_weight=torch.ones(MINI_WEIGHT_SHAPE, dtype=torch.bfloat16),
            down_proj_weight=torch.ones(MINI_WEIGHT_SHAPE, dtype=torch.bfloat16),
            gate_proj_scale=None,
            up_proj_scale=torch.ones(MINI_SCALE_SHAPE, dtype=torch.float32),
            down_proj_scale=torch.ones(MINI_SCALE_SHAPE, dtype=torch.float32),
        )
    assert "gate_proj_scale" in str(case_a.value), str(case_a.value)
    assert "load the checkpoint before preparing" in str(case_a.value)
    assert not hasattr(shared, Glm5NextSharedExperts.PREPARED_SCALE_OPERANDS_ATTR), (
        "the refusal left the operands attribute behind, so a later read would "
        "find a half-built dict"
    )

    model = _dense_model()
    directory = tmp_path / "dense"
    _write_miniature_checkpoint(directory, _mappings_for(_dense_config()), model)
    model.load_weights(str(directory), torch.device("cpu"), None)
    attn_path = next(
        path
        for path, module in model.named_modules()
        if hasattr(type(module), "prepare_projection_weights")
    )
    attn = model.get_submodule(attn_path)
    names = [name for name, _, _ in attn.projection_widths()]
    with pytest.raises(Glm5NextWeightLoadError) as case_b:
        model._require_prep_operands_on_device(
            attn_path, attn, names, torch.device("meta")
        )
    assert attn_path in str(case_b.value), str(case_b.value)
    assert "cpu" in str(case_b.value), str(case_b.value)
    assert "strand" in str(case_b.value), str(case_b.value)

    checked = model._require_prep_operands_on_device(
        attn_path, attn, names, torch.device("cpu")
    )
    assert checked > 0

    stranded = _dense_model()
    stranded.load_weights(str(directory), torch.device("cpu"), None)
    stranded_attn = stranded.get_submodule(attn_path)
    prepared_attr = type(stranded_attn).PREPARED_WEIGHTS_ATTR
    before = {
        name: operand.device.type
        for name, operand in getattr(stranded_attn, prepared_attr).items()
    }
    assert before, "the prep stored no operand"
    stranded_attn.to("meta")
    after = {
        name: operand.device.type
        for name, operand in getattr(stranded_attn, prepared_attr).items()
    }
    assert set(before.values()) == {"cpu"}
    assert after == before, (
        "moving the module moved the prepared operands with it; they stay where "
        "the prep built them, which is why the device check exists"
    )


def test_the_scaled_mla_weights_reach_the_dequant_as_fp8(
    keep_the_loaded_tensors, tmp_path, single_rank_process_group
) -> None:
    """A lone fp8 weight key whose grid is a sibling entry keeps an fp8 placeholder, so the dequant branch is reachable."""
    real_config = Glm5NextConfig.from_configs(json.loads(REAL_CONFIG_PATH.read_text()))
    real_model = Glm5NextForConditionalGeneration(real_config)
    real_mappings = _mappings_for(real_config)
    scale_names = _derived_dsa_scale_names(real_config)
    weight_names = sorted(
        f"{name[: -len('_' + FP8_SCALE_SUFFIX)]}_weight" for name in scale_names
    )
    assert weight_names, "the real config declares no scaled MLA weight"
    missing = [name for name in weight_names if name not in real_mappings]
    assert missing == [], f"derived but unmapped: {missing[:4]}"
    typed = [
        name
        for name in weight_names
        if real_model._placeholder_dtype(
            real_mappings[name], param_name=name, mappings=real_mappings
        )
        is torch.float8_e4m3fn
    ]
    assert len(typed) == len(weight_names), (
        f"{len(weight_names) - len(typed)} scaled MLA weights would take a "
        f"placeholder dtype that is not fp8, e.g. "
        f"{sorted(set(weight_names) - set(typed))[:4]}"
    )
    paired = [
        name
        for name in weight_names
        if len(_keys_of(real_mappings, name)) == 2
        and classify_mapped_keys(real_mappings[name]) == MAPPED_KEY_QUANTISED_WEIGHT
    ]
    assert len(paired) == len(weight_names)

    ordinary = sorted(
        name
        for name in real_mappings
        if name.endswith("_weight")
        and len(_keys_of(real_mappings, name)) == 1
        and classify_mapped_keys(real_mappings[name]) == MAPPED_KEY_PLAIN
        and name not in set(weight_names)
    )
    assert ordinary, "there is no unscaled lone weight key to control against"
    control_dtype = real_model._placeholder_dtype(
        real_mappings[ordinary[0]],
        param_name=ordinary[0],
        mappings=real_mappings,
    )
    assert control_dtype is real_config.text_config.torch_dtype

    model = _dense_model()
    mappings = _mappings_for(_dense_config())
    directory = tmp_path / "dense"
    _write_miniature_checkpoint(directory, mappings, model)
    model.load_weights(str(directory), torch.device("cpu"), None)

    attn_path = next(
        path
        for path, module in model.named_modules()
        if hasattr(type(module), "prepare_projection_weights")
    )
    attn = model.get_submodule(attn_path)
    reached = 0
    for leaf in DSA_SCALED_PROJECTIONS:
        weight = getattr(attn, f"{leaf}_weight")
        assert _is_fp8_dtype(weight.dtype), (
            f"{attn_path}.{leaf}_weight arrived {weight.dtype}; the checkpoint "
            f"holds fp8 bytes, so the dequant branch is unreachable and the "
            f"bytes would be used as if they were numbers"
        )
        reached += 1
    assert reached == len(DSA_SCALED_PROJECTIONS)

    mlp_path = attn_path.rsplit(".", 1)[0] + ".mlp"
    mlp_weight = getattr(model.get_submodule(mlp_path), "gate_proj_weight")
    assert _is_fp8_dtype(mlp_weight.dtype)
