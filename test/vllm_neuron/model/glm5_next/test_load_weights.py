# SPDX-License-Identifier: Apache-2.0
"""``inc-glm53f-091a`` -- the end-to-end weight LOADING entry point.

FOUR counted items, one per conjunct, no ``parametrize`` anywhere in this file
(D1.2). The remaining four conjuncts -- the fp32 scale grids, the orphan
call-site count, the loader arity contract and the prep ordering -- are
``inc-glm53f-091b``'s and are DELIBERATELY ABSENT from this file by name, so
this half's four cannot be satisfied by that half's work or the other way
round.

WHY THREE OF THE FOUR DRIVE THE REAL ``load_weights``. No landed test calls
``load_sharded_pipelined`` and the test tree has no safetensors writer, so a
test built only from fake slices would leave ``load_weights`` itself never
executed -- which is exactly the gap this increment exists to close. Conjuncts
1, 3 and 7 therefore write a real miniature safetensors checkpoint into
``tmp_path`` and load it, so ``SafetensorsCheckpoint``, the pipelined reader,
the attached loaders and ``load_state_dict(assign=True)`` all really run.
Conjunct 2 stays at the dictionary level against the real published weight
index, which holds no tensors at all.

WHY TWO MINIATURE CONFIGURATIONS AND NOT ONE. A routed expert bank arrives as
ONE map entry holding every expert's key, and no loader in this package stacks
experts yet, so ``load_weights`` REFUSES such an entry by name and
``inc-glm53f-095`` is where it stops refusing. A configuration whose layers are
all dense therefore loads, and a configuration with a routed bank refuses --
both are readings of conjunct 1 rather than one working case and one skipped
one. :func:`_dense_config` and :func:`_routed_config` are those two, and the
difference between them is one field.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import textwrap
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm_neuron.model.glm5_next.config import Glm5NextConfig, Glm5NextTextConfig
from vllm_neuron.model.glm5_next.model_fp8 import (
    Glm5NextForConditionalGeneration,
    Glm5NextWeightLoadError,
)
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
    FP8_SCALE_SUFFIX,
    MAPPED_KEY_QUANTISED_WEIGHT,
    Glm5NextExpertBankNotLoadableError,
    build_weight_mappings,
    classify_mapped_keys,
)

# --------------------------------------------------------------------------- #
# The real published fixtures, and the two exclusion prefixes.
#
# Both constants are ``inc-glm53f-078``'s, named here exactly as that increment
# named them so conjunct 2's population is the same partition its landed items
# assert rather than a second one that happens to agree today.
# --------------------------------------------------------------------------- #

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REAL_INDEX_PATH = FIXTURES_DIR / "model.safetensors.index.json"
REAL_CONFIG_PATH = FIXTURES_DIR / "hf-config.json"

MTP_LAYER_PREFIX = "model.language_model.layers.45."
VISION_PREFIX = "model.visual."

#: ``inc-glm53f-078``'s counts off the published index, and the subtraction that
#: produces the in-scope population conjunct 2 is measured over.
REAL_INDEX_TOTAL_KEYS = 76_108
REAL_INDEX_MTP_KEYS = 1_760
REAL_INDEX_VISION_KEYS = 347
REAL_INDEX_IN_SCOPE_KEYS = 74_001

#: The miniature stack. Four layers is enough for every family the map knows --
#: one dense MLP layer then MoE layers -- and small enough that a real
#: checkpoint of it is written and read inside one test.
MINI_LAYERS = 4
MINI_ROUTED_EXPERTS = 4
MINI_SHARED_EXPERTS = 1
MINI_FIRST_K_DENSE = 1

#: All four layers dense. ``model_fp8.py:1997`` and ``weight_loaders_fp8.py:412``
#: both branch on ``layer_idx < first_k_dense_replace``, the tree and the map on
#: the same field, so this one number is the whole difference between a
#: configuration that loads and one that refuses. ``n_routed_experts`` is left at
#: its miniature value rather than zeroed: the config validator requires at least
#: one (``config.py:277``), and no layer builds a bank here anyway.
MINI_ALL_DENSE_FIRST_K = MINI_LAYERS

#: The miniature tensor shapes. A weight is one full 128x128 quantisation block
#: so its scale grid is a single value; the exact numbers do not matter to any
#: item here, only that the dtypes are the ones the loaders expect.
MINI_WEIGHT_SHAPE = (128, 128)
MINI_SCALE_SHAPE = (1, 1)
MINI_PLAIN_SHAPE = (4,)

#: The one file each miniature checkpoint is written to. Named once so the writer
#: and the shape reader below cannot disagree about where it is.
MINI_CHECKPOINT_FILE = "model.safetensors"


def _mini_config(first_k_dense: int) -> Glm5NextConfig:
    """The miniature model config, quantised, with no BF16 skip list.

    One builder with one varying field, so the two configurations below cannot
    drift apart in anything except the thing they differ in.
    """
    return Glm5NextConfig(
        text_config=Glm5NextTextConfig(
            num_hidden_layers=MINI_LAYERS,
            n_routed_experts=MINI_ROUTED_EXPERTS,
            n_shared_experts=MINI_SHARED_EXPERTS,
            first_k_dense_replace=first_k_dense,
            tie_word_embeddings=False,
        )
    )


def _dense_config() -> Glm5NextConfig:
    """Every layer dense, so every map entry has a loader and the load runs."""
    return _mini_config(MINI_ALL_DENSE_FIRST_K)


def _routed_config() -> Glm5NextConfig:
    """One dense layer then routed banks, so the load must refuse by name."""
    return _mini_config(MINI_FIRST_K_DENSE)


def _dense_model() -> Glm5NextForConditionalGeneration:
    return Glm5NextForConditionalGeneration(_dense_config())


def _routed_model() -> Glm5NextForConditionalGeneration:
    return Glm5NextForConditionalGeneration(_routed_config())


def _mappings_for(config: Glm5NextConfig) -> dict[str, str | list[str]]:
    """A REFERENCE map, built from the same settings ``load_weights`` reads.

    The settings are the same two config members ``load_weights`` reads
    (``config.py:400`` and ``:408``), so this is the right map to compare a load
    against.

    WHAT IT IS NOT EVIDENCE OF, corrected by B65-M1. This helper says nothing
    about the map ``load_weights`` actually builds, because it calls
    ``build_weight_mappings`` itself. An earlier docstring here claimed that "a
    test that agreed with a wrong ``load_weights`` is not possible", and that was
    FALSE: a ``load_weights`` that built its map off the wrong config member, or
    dropped it, would still agree with this helper. The item whose subject IS the
    handed-over map observes it inside the running ``load_weights`` instead --
    see ``test_the_map_load_weights_hands_over_covers_the_in_scope_index``.
    """
    return build_weight_mappings(
        config.text_config,
        quantised=config.is_block_quantized,
        modules_to_not_convert=tuple(config.modules_to_not_convert or ()),
    )


def _is_scale_key(key: str) -> bool:
    return key.endswith(f".{FP8_SCALE_SUFFIX}")


def _keys_of(mappings: dict[str, str | list[str]], name: str) -> list[str]:
    """One map entry's checkpoint keys as a list.

    The map stores a lone key as a bare string and a fused family as a list, so
    every count over an entry has to normalise first. One helper, so no reading
    in this file counts a string's characters by accident.
    """
    keys = mappings[name]
    return [keys] if isinstance(keys, str) else list(keys)


def _write_miniature_checkpoint(
    directory: Path, mappings: dict[str, str | list[str]]
) -> int:
    """Write a REAL safetensors file holding one tensor per mapped key.

    Returns how many tensors were written. Each tensor is typed the way the
    checkpoint types it -- fp8 bytes for a quantised weight, fp32 for a scale
    grid, the config dtype otherwise -- because the loaders act on the dtype:
    the weight path squeezes fp8 bytes into the trn2 range and the scale path
    compensates an fp32 grid. Shapes are miniature and arbitrary; nothing here
    asserts a shape.
    """
    tensors: dict[str, torch.Tensor] = {}
    for keys in mappings.values():
        key_list = [keys] if isinstance(keys, str) else list(keys)
        quantised_pair = classify_mapped_keys(keys) == MAPPED_KEY_QUANTISED_WEIGHT
        for key in key_list:
            if key in tensors:
                continue
            if _is_scale_key(key):
                tensors[key] = torch.full(
                    MINI_SCALE_SHAPE, 0.5, dtype=torch.float32
                )
            elif quantised_pair:
                tensors[key] = torch.ones(
                    MINI_WEIGHT_SHAPE, dtype=torch.bfloat16
                ).to(torch.float8_e4m3fn)
            else:
                tensors[key] = torch.ones(MINI_PLAIN_SHAPE, dtype=torch.bfloat16)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(directory / MINI_CHECKPOINT_FILE))
    return len(tensors)


def _implied_numels(
    directory: Path, mappings: dict[str, str | list[str]]
) -> dict[str, int]:
    """How many elements each mapped parameter's OWN checkpoint slices imply.

    DERIVED FROM THE FILE, never from the shape constants above. The shapes are
    read back through ``safe_open(...).get_slice(key).get_shape()`` -- the very
    call the code under test reads its slices with
    (``utils/checkpoints.py:680-681``) -- so this reading cannot agree with a
    wrong load by sharing a table with it.

    WHICH KEYS COUNT, and why it is not simply "all of them". The element count
    that reaches the parameter is decided by the loader the entry was given, and
    there are two shapes of answer:

    * an entry holding a weight and its scale companion is loaded by
      :func:`wrap_with_blockwise_fp8_downscale`, whose transform keeps the WEIGHT
      slice; the scale is read out of band. So the scale key contributes nothing
      and the implied count is the weight keys' total.
    * an entry that is nothing BUT a scale key is loaded by
      :func:`blockwise_scale_loader`, which compensates the grid and returns it
      at its own shape. So that key is the population.

    Neither transform reshapes and neither pads, and the reader's only other act
    on the tensor is a dtype cast (``utils/checkpoints.py:571-576``), which
    cannot change an element count. So for a load that delivers everything the
    two numbers are equal, and a loader that dropped a slice makes them differ --
    which is exactly the defect this reading exists to catch.
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
    """How many declared parameters hold a real tensor rather than a placeholder.

    The predicate is torch's OWN: ``torch.nn.parameter.is_lazy`` is the function
    ``nn.Module._load_from_state_dict`` consults when it decides whether to
    check a shape, so this reading and torch's behaviour cannot disagree. It is
    public, unlike the dtype tables next to it.

    ``numel() > 0`` is deliberately NOT the predicate: an unfilled placeholder
    raises ``ValueError`` on ``numel()``, so the before-the-load control would
    throw instead of counting zero.
    """
    return sum(
        1
        for _, param in model.named_parameters()
        if not torch.nn.parameter.is_lazy(param)
    )


def _token_checkpoint_directory(tmp_path: Path) -> Path:
    """A directory that gets past ``load_weights``'s opener and holds no weights.

    ``load_weights`` refuses a checkpoint whose ``get_num_files()`` reads zero
    (``model_fp8.py:3309-3314``), so even a dict-level reading needs ONE
    ``.safetensors`` file to exist. This one holds a single one-element tensor and
    is never opened: the run refuses at materialisation, which is before
    ``load_sharded_pipelined``, so not one tensor byte of it is read. Writing the
    real checkpoint instead is not an option at any size -- its published index
    lists 76,108 tensors.
    """
    directory = tmp_path / "token-checkpoint"
    directory.mkdir()
    save_file(
        {"token": torch.zeros(1, dtype=torch.float32)},
        str(directory / "model-00001-of-00001.safetensors"),
    )
    return directory


def _mappings_flow_in_load_weights() -> tuple[int, int]:
    """How often ``load_weights`` binds ``mappings``, and hands that name onward.

    This reads ``load_weights``'s own source, because the fact needed is about the
    code rather than about one run. The hand-over to ``load_sharded_pipelined``
    (``model_fp8.py:3348``) cannot be reached while a routed expert bank refuses,
    so the item observes the map one line earlier and needs to know the two lines
    cannot disagree: ONE binding, and a hand-over passing THAT SAME NAME, is what
    makes the observed object the object the hand-over would pass.

    A count, not a pattern match. A diff that rebinds ``mappings`` -- a filter, a
    copy, a re-read -- moves the first number, and the item reddens by position
    rather than by anyone's judgement.
    """
    tree = ast.parse(
        textwrap.dedent(
            inspect.getsource(Glm5NextForConditionalGeneration.load_weights)
        )
    )
    bindings = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and node.id == "mappings"
    )
    handovers = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load_sharded_pipelined"
        and any(
            isinstance(arg, ast.Name) and arg.id == "mappings"
            for arg in node.args
        )
    )
    return bindings, handovers


@pytest.fixture
def single_rank_process_group(tmp_path):
    """A one-rank CPU process group, because the reader requires one.

    NOT test convenience. ``load_sharded_pipelined`` takes the default
    distributed store on its first line
    (``utils/checkpoints.py:330-332``,
    ``torch.distributed.distributed_c10d._get_default_store()``) to tell the
    ranks which checkpoint files have reached the page cache, so with no process
    group initialised it raises ``ValueError: Default process group has not been
    initialized`` before reading a byte. Measured, not assumed: that is exactly
    how this item failed on its first run.

    One rank, ``gloo``, rendezvous through a file rather than a port -- the
    fork's own convention for CPU-mode rendezvous, and a port would collide with
    a parallel run. ``_resolve_world_size()`` reads 1 and ``_resolve_rank()``
    reads 0 inside it, the same pair they read undistributed, so the load under
    test is the undistributed one either way.
    """
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


# --------------------------------------------------------------------------- #
# (1) Every declared parameter is materialised AND loaded.
# --------------------------------------------------------------------------- #


def test_every_declared_parameter_is_materialised_and_loaded(
    tmp_path, single_rank_process_group
) -> None:
    """(1) The load DELIVERS every element it claimed, or it REFUSES by name.

    Certifies ``load_weights``. THREE readings, because the first two of them
    were each shown insufficient by measurement rather than by argument.

    (i) On an all-dense configuration, N/N declared parameters hold a real
    tensor after the load and 0/N before. The control is the SAME expression on
    the SAME tree in the same process, so the reading moves rather than comparing
    two objects: a ``load_weights`` that materialised the tree and returned early
    reads 0/N.

    (ii) On a configuration with a routed expert bank the load REFUSES, naming
    the parameter, its key count and ``inc-glm53f-095``, and leaves NOTHING
    behind -- 0 not-lazy and 0 materialised placeholders. The refusal IS the
    reading here. A skipped parameter would not be.

    (iii) On the dense configuration, every mapped parameter's loaded element
    count equals the count its OWN checkpoint slices imply. This exists because
    readings of the first shape are DELIVERY checks that a wrong-shaped tensor
    passes: nothing between the loader and the parameter validates a shape --
    ``utils/checkpoints.py`` has no such check and the lazy placeholder makes
    torch skip its own -- so "not lazy" was satisfied by a tensor holding one
    expert of 288. That was measured on this increment before it landed.

    DISCLOSED SCOPE OF (iii), so no reader over-reads it. On the dense
    population every entry has exactly ONE weight key, and the run prints that
    count. So (iii) cannot fire on the dropped-slice class HERE; reading (ii) is
    what holds that class at this increment, and ``inc-glm53f-095``'s first
    conjunct is (iii) again on a bank, which is where it becomes that detector.
    What (iii) certifies here is the whole transform chain: a compensation, a
    downscale or a future shard that changed an element count reddens it.
    """
    # ── (i) an all-dense configuration loads ─────────────────────────────────
    dense_dir = tmp_path / "dense"
    model = _dense_model()
    n = len(model.declared_parameter_names())
    assert n > 0, "the dense miniature tree declares no parameter; nothing to load"

    mappings = _mappings_for(_dense_config())
    written = _write_miniature_checkpoint(dense_dir, mappings)
    assert written > 0, "the dense miniature checkpoint holds no tensor"

    before = _not_lazy_count(model)
    model.load_weights(str(dense_dir), torch.device("cpu"), None)
    after = _not_lazy_count(model)

    print(f"CONJUNCT1_DENSE_DECLARED={n}")
    print(f"CONJUNCT1_DENSE_MAP_ENTRIES={len(mappings)}")
    print(f"CONJUNCT1_DENSE_CHECKPOINT_TENSORS={written}")
    print(f"CONJUNCT1_DENSE_NOT_LAZY_BEFORE={before}")
    print(f"CONJUNCT1_DENSE_NOT_LAZY_AFTER={after}")

    assert before == 0, (
        f"the control did not read 0/{n}: {before} parameters already held a "
        f"real tensor before the load, so this item cannot certify the load"
    )
    assert after == n, (
        f"{n - after} of {n} declared parameters still hold an unfilled "
        f"placeholder after the load"
    )
    assert len(model.declared_parameter_names()) == n, (
        "the declared-name set changed across the load"
    )

    # ── (iii) every mapped parameter received all of its elements ────────────
    implied = _implied_numels(dense_dir, mappings)
    loaded = {name: param.numel() for name, param in model.named_parameters()}
    checked = sorted(set(implied) & set(loaded))
    multi_weight = [
        name
        for name in checked
        if len([k for k in _keys_of(mappings, name) if not _is_scale_key(k)]) > 1
    ]
    mismatches = {
        name: (loaded[name], implied[name])
        for name in checked
        if loaded[name] != implied[name]
    }

    print(f"CONJUNCT1_DELIVERY_ENTRIES_CHECKED={len(checked)}")
    print(f"CONJUNCT1_DELIVERY_ENTRIES_WITH_MORE_THAN_ONE_WEIGHT_KEY={len(multi_weight)}")
    print(f"CONJUNCT1_DELIVERY_MISMATCHES={len(mismatches)}")

    assert len(checked) == n, (
        f"the delivery reading covers {len(checked)} of the {n} declared "
        f"parameters, so it is measured over a subset rather than the whole "
        f"population; unmapped or unwritten: "
        f"{sorted(set(loaded) - set(implied))[:5]}"
    )
    assert mismatches == {}, (
        f"{len(mismatches)} parameters received fewer or more elements than "
        f"their own checkpoint slices imply, e.g. "
        f"{[(k, *mismatches[k]) for k in sorted(mismatches)[:3]]} as "
        f"(parameter, loaded, implied)"
    )

    # ── (ii) a routed expert bank refuses by name and leaves nothing ─────────
    routed_dir = tmp_path / "routed"
    routed = _routed_model()
    routed_declared = len(routed.declared_parameter_names())
    routed_mappings = _mappings_for(_routed_config())
    assert _write_miniature_checkpoint(routed_dir, routed_mappings) > 0

    banks = {
        name: _keys_of(routed_mappings, name)
        for name in routed_mappings
        if len([k for k in _keys_of(routed_mappings, name) if _is_scale_key(k)]) > 1
    }
    assert banks, (
        "the routed configuration produced no multi-scale-key entry, so this "
        "reading would certify nothing; the refusal it exercises could not fire"
    )

    with pytest.raises(Glm5NextExpertBankNotLoadableError) as raised:
        routed.load_weights(str(routed_dir), torch.device("cpu"), None)
    message = str(raised.value)
    named = [name for name in banks if name in message]

    print(f"CONJUNCT1_BANK_DECLARED={routed_declared}")
    print(f"CONJUNCT1_BANK_ENTRIES={len(banks)}")
    print(f"CONJUNCT1_BANK_NAMED_IN_MESSAGE={len(named)}")
    print(f"CONJUNCT1_BANK_NOT_LAZY_AFTER={_not_lazy_count(routed)}")
    print(f"CONJUNCT1_BANK_MATERIALISED_AFTER={len(list(routed.named_parameters()))}")

    assert named, f"the refusal names no bank parameter: {message}"
    key_count = len(banks[named[0]])
    print(f"CONJUNCT1_BANK_KEY_COUNT={key_count}")
    assert str(key_count) in message, (
        f"the refusal does not report the key count {key_count} of the "
        f"parameter it named: {message}"
    )
    assert "inc-glm53f-095" in message, (
        f"the refusal does not name the increment that answers it: {message}"
    )
    assert _not_lazy_count(routed) == 0, (
        "the refusal left parameters holding real tensors"
    )
    assert len(list(routed.named_parameters())) == 0, (
        f"the refusal left {len(list(routed.named_parameters()))} materialised "
        f"placeholders behind out of {routed_declared} declared, so the tree is "
        f"half-built and a later load cannot tell it from a fresh one"
    )


# --------------------------------------------------------------------------- #
# (2) The map handed to the reader covers the in-scope checkpoint, both ways.
# --------------------------------------------------------------------------- #


def test_the_map_load_weights_hands_over_covers_the_in_scope_index(
    tmp_path,
) -> None:
    """(2) Zero in-scope index keys unclaimed, and zero mapped keys not in the index.

    Certifies the INTEGRATION -- that ``load_weights`` did not drop, rename or
    double-claim a family on the way to the reader. ``build_weight_mappings`` in
    isolation is already certified by ``inc-glm53f-078``'s
    ``test_skeleton_real_index_coverage_is_one_hundred_percent``; this is a
    different claim about a different subject.

    THE SUBJECT IS THE OBJECT, NOT THE VALUE (B65-M1, plan revision 157). An
    earlier form of this item rebuilt the map with ``_mappings_for`` and counted
    over that. The counts were right and the object was wrong: a ``load_weights``
    that built its map off the wrong config member, or dropped it, would have
    left this item green. So the map is now taken OUT OF THE RUNNING
    ``load_weights``.

    HOW, AND WHY NOT AT THE HAND-OVER ITSELF. The hand-over
    (``model_fp8.py:3348``) cannot be reached while a routed expert bank refuses,
    and this checkpoint's banks each carry 288 experts, so no run observes that
    line at ``inc-glm53f-091a``. The observer therefore sits on
    ``_materialise_declared_parameters`` one line earlier (``:3339``) and records
    its argument before calling the real method, so production code still builds
    the map and still decides to refuse. ``_mappings_flow_in_load_weights`` then
    reads the source and reports that ``mappings`` is bound once and that the
    hand-over passes that same name, which is what makes the observed object the
    object the hand-over would pass.

    THE REFUSAL IS THE MEANS HERE, NOT A SECOND SUBJECT -- conjunct 1 owns it.
    It is asserted so that a SILENT COMPLETION reddens this item: when
    ``inc-glm53f-095`` makes stacked banks loadable, this item fails loudly and
    gets re-read, instead of quietly measuring a path that no longer refuses.

    No process group is initialised, on purpose. The fixture that provides one
    exists for the reader's default-store call inside ``load_sharded_pipelined``,
    which this run never reaches.

    The population is ``inc-glm53f-078``'s in-scope partition of the published
    index, re-derived here from the fixture rather than restated: a count over
    the raw 76,108 keys would read 2,107 unclaimed and would be the WRONG
    population, because layer 45 is the multi-token-prediction layer and the
    vision tower is a separate encoder.
    """
    weight_map = json.loads(REAL_INDEX_PATH.read_text())["weight_map"]
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
    print(f"CONJUNCT2_INDEX_TOTAL_KEYS={total}")
    print(f"CONJUNCT2_INDEX_MTP_KEYS={len(mtp)}")
    print(f"CONJUNCT2_INDEX_VISION_KEYS={len(vision)}")
    print(f"CONJUNCT2_IN_SCOPE_POPULATION={len(in_scope)}")

    real_config = Glm5NextConfig.from_configs(
        json.loads(REAL_CONFIG_PATH.read_text())
    )
    model = Glm5NextForConditionalGeneration(real_config)
    print(f"CONJUNCT2_DECLARED_NAMES={len(model.declared_parameter_names())}")

    observed: list[dict[str, str | list[str]]] = []
    real_materialise = model._materialise_declared_parameters

    def observer(handed_over, device):
        observed.append(handed_over)
        return real_materialise(handed_over, device)

    model._materialise_declared_parameters = observer

    with pytest.raises(Glm5NextExpertBankNotLoadableError) as refusal:
        model.load_weights(
            str(_token_checkpoint_directory(tmp_path)),
            torch.device("cpu"),
            None,
        )

    message = str(refusal.value)
    print(f"CONJUNCT2_REFUSAL_CLASS={type(refusal.value).__name__}")
    print(f"CONJUNCT2_CAPTURED_MAPS={len(observed)}")
    print(f"CONJUNCT2_NAMED_PARAMETERS_AFTER={len(list(model.named_parameters()))}")

    assert len(observed) == 1, (
        f"the observer recorded {len(observed)} maps where conjunct 2 needs "
        f"exactly the one load_weights built"
    )
    assert "576 checkpoint keys" in message and "288 scale keys" in message, (
        f"the refusal does not name the key counts it refused on: {message!r}"
    )
    assert "inc-glm53f-095" in message, (
        f"the refusal does not name where stacked banks land: {message!r}"
    )
    assert list(model.named_parameters()) == [], (
        "the refusal left parameters registered, so it half built the tree"
    )

    bindings, handovers = _mappings_flow_in_load_weights()
    print(f"CONJUNCT2_MAPPINGS_BINDINGS_IN_SOURCE={bindings}")
    print(f"CONJUNCT2_HANDOVER_PASSES_THAT_NAME={handovers}")
    assert bindings == 1, (
        f"`mappings` is bound {bindings} times inside load_weights, so the "
        f"object observed at the materialiser is not provably the one the "
        f"hand-over passes"
    )
    assert handovers == 1, (
        "load_sharded_pipelined is not passed the `mappings` name, so this "
        "item's subject is no longer the map that is handed over"
    )

    mappings = observed[0]
    print(f"CONJUNCT2_CAPTURED_MAP_ENTRIES={len(mappings)}")

    claimed: set[str] = set()
    for keys in mappings.values():
        claimed.update([keys] if isinstance(keys, str) else keys)

    unclaimed = in_scope - claimed
    absent_from_index = claimed - set(weight_map)
    print(f"CONJUNCT2_UNCLAIMED={len(unclaimed)}")
    print(f"CONJUNCT2_MAPPED_KEYS_ABSENT_FROM_INDEX={len(absent_from_index)}")
    print(f"CONJUNCT2_MTP_KEYS_CLAIMED={len(claimed & mtp)}")

    assert unclaimed == set(), (
        f"{len(unclaimed)} in-scope checkpoint keys are claimed by no mapping, "
        f"e.g. {sorted(unclaimed)[:5]}"
    )
    assert absent_from_index == set(), (
        f"{len(absent_from_index)} mapped keys are absent from the index, e.g. "
        f"{sorted(absent_from_index)[:5]}"
    )
    assert not (claimed & mtp), "the map claims a multi-token-prediction key"


# --------------------------------------------------------------------------- #
# (3) The reader iterates nothing until the parameters are materialised.
# --------------------------------------------------------------------------- #


def test_the_reader_iterates_no_parameter_until_they_are_materialised() -> None:
    """(3) A counted zero with the control that makes it mean something (D1.5).

    Certifies the forced ordering inside ``load_weights``. The reader decides
    what to load by iterating ``list(model.named_parameters())``
    (``utils/checkpoints.py:348``, consumed at ``:402``) and torch omits a
    ``register_parameter(name, None)`` declaration from that list, so on the
    declared-but-unmaterialised tree the reader sees EXACTLY ZERO parameters and
    a load that skipped materialisation would read nothing and report success.

    The expression read here is the reader's own, so the two cannot disagree.
    """
    model = _dense_model()
    n = len(model.declared_parameter_names())
    assert n > 0

    before = len(list(model.named_parameters()))
    assert before == 0, (
        f"the reader would iterate {before} parameters on an unmaterialised "
        f"tree; this counted zero is the reason materialisation must precede "
        f"the load, and a nonzero reading contradicts the design"
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


# --------------------------------------------------------------------------- #
# (7) An absent checkpoint refuses by name and materialises nothing.
# --------------------------------------------------------------------------- #


def test_an_absent_checkpoint_refuses_by_name_and_leaves_the_tree_alone(
    tmp_path,
) -> None:
    """(7) The refusal names the path, and the tree is left at 0/N.

    Certifies the refusal inside ``load_weights``. Both halves matter: a
    refusal that named nothing would send a reader to the wrong place, and a
    refusal that fired after materialisation would leave a half-built tree that
    a later load could not distinguish from a fresh one.
    """
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
