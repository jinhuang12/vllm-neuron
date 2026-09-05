# SPDX-License-Identifier: Apache-2.0
"""``inc-glm53f-091`` -- the end-to-end weight LOADING entry point.

THIRTEEN counted items and no ``parametrize`` decorator anywhere in this file
(D1.2). The first four are ``inc-glm53f-091a``'s, one per conjunct. The next five
are ``inc-glm53f-091b``'s: the fp32 scale grids, the orphan call-site count, the
loader arity contract, the prep ordering, and the placeholder dtype of a scaled
MLA weight. The last four are ``inc-glm53f-095``'s stacked expert bank, one per
conjunct, and the count moved from NINE to THIRTEEN there. Each group is kept in
the order it landed and every item names the conjunct it reads, so no group's
items can be satisfied by another group's work.

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
    _WEIGHT_LEAF_SUFFIX,
    _is_fp8_dtype,
    _scale_prep_leaves,
)
from vllm_neuron.model.glm5_next.quantization import DEFAULT_WEIGHT_BLOCK_SIZE
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
    DSA_SCALED_PROJECTIONS,
    FP8_SCALE_SUFFIX,
    MAPPED_KEY_PLAIN,
    MAPPED_KEY_QUANTISED_WEIGHT,
    MAPPED_KEY_SCALE_GRID,
    MAPPED_KEY_STACKED_BANK,
    Glm5NextExpertBankNotLoadableError,
    Glm5NextWeightMapError,
    bank_layout,
    block_grid_shape,
    blockwise_scale_loader,
    build_weight_mappings,
    classify_mapped_keys,
    compensate_block_scales,
    downscale_fp8_weight_bytes,
    loader_for_mapped_keys,
    scale_keys,
    stacked_expert_bank_loader,
    stacked_expert_scale_loader,
)
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

# ``_is_fp8_dtype`` is imported deliberately, private name and all: it is the
# EXACT predicate ``_dequantised_projection_weight`` branches on
# (``model_fp8.py:2827``), so an item asking whether the dequant branch is
# reachable has to ask the same question the branch asks rather than a
# look-alike dtype comparison of its own.

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

#: All four layers dense. ``model_fp8.py:2013`` and ``weight_loaders_fp8.py:412``
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

#: The five MLA widths, shrunk by ``inc-glm53f-091b`` so the sparse-attention
#: layer's projections can be written at their REAL closed-form shapes.
#:
#: WHY THEY MOVED. ``-091b`` calls ``prepare_projection_weights`` at the end of
#: the load, and that method checks each weight against
#: ``projection_widths()``'s closed form. At the config's own widths the closed
#: forms run to ``(16384, 1536)``, so a checkpoint holding them is hundreds of
#: megabytes and no test can write one. At the widths below every closed form
#: is at most ``128 x 128`` -- one whole ``DEFAULT_WEIGHT_BLOCK_SIZE`` dequant
#: block, so each scale grid stays the ``(1, 1)`` the writer already writes.
#: Shrinking a width in a miniature config is this suite's own idiom
#: (``test_shared_expert_scale_prep.py:137``, ``test_router.py:1073``).
#:
#: The rotary slice is left at 0 because the checkpoint's own is 0; the query
#: head width is then the nope width alone, exactly as
#: ``projection_widths()``'s docstring records.
#:
#: MEASURED TO MOVE NO LANDED COUNT (``probe-091b-preps.out``, arms A and B):
#: declared names 110, map entries 110, out-of-band entries 12, not-lazy after
#: the load 110 and the sparse-attention layer index ``[3]`` all read the same
#: at the config's widths and at these.
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
            **MINI_MLA_WIDTHS,
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


def _mla_key_overrides(
    model: Glm5NextForConditionalGeneration,
    mappings: dict[str, str | list[str]],
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """The MLA family's real shapes and dtypes, per checkpoint key.

    ``inc-glm53f-091b``. Every other key in the miniature checkpoint is written
    at an arbitrary shape, because nothing reads one. The sparse-attention
    projections are the exception: ``prepare_projection_weights`` checks each of
    them against ``projection_widths()``, so a ``(4,)`` placeholder shape makes
    the prep refuse and the load cannot complete.

    NOTHING HERE IS SPELLED TWICE. The shape comes from the module's own
    ``projection_widths()`` -- the same closed form the code under test checks
    against -- the scale grid's shape from ``block_grid_shape``, and the
    CHECKPOINT KEY from the map. Writing the key by hand would be a second
    naming convention that could drift from the one the loader reads.

    The four scaled projections are written as fp8 bytes with an fp32 grid, and
    ``kv_b_proj`` as bf16 with no grid, because that is what the published
    checkpoint holds: ``DSA_SCALED_PROJECTIONS`` is the list of leaves that
    carry a ``weight_scale_inv`` companion, and the docstring beside it records
    that ``kv_b_proj`` carries none.
    """
    overrides: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
    for path, module in model.named_modules():
        if not hasattr(type(module), "projection_widths"):
            continue
        for name, idim, odim in module.projection_widths():
            quantised = name in DSA_SCALED_PROJECTIONS
            weight_param = f"{path}.{name}_weight"
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


def _write_miniature_checkpoint(
    directory: Path,
    mappings: dict[str, str | list[str]],
    model: Glm5NextForConditionalGeneration,
) -> int:
    """Write a REAL safetensors file holding one tensor per mapped key.

    Returns how many tensors were written. Each tensor is typed the way the
    checkpoint types it -- fp8 bytes for a quantised weight, fp32 for a scale
    grid, the config dtype otherwise -- because the loaders act on the dtype:
    the weight path squeezes fp8 bytes into the trn2 range and the scale path
    compensates an fp32 grid.

    Shapes are miniature and arbitrary EXCEPT for the MLA family, which
    :func:`_mla_key_overrides` writes at its closed form because
    ``prepare_projection_weights`` checks it. Nothing here asserts a shape.
    """
    overrides = _mla_key_overrides(model, mappings)
    tensors: dict[str, torch.Tensor] = {}
    for keys in mappings.values():
        key_list = [keys] if isinstance(keys, str) else list(keys)
        # A BANK IS QUANTISED TOO, and saying so here is not a new behaviour --
        # it is how this writer already behaved before ``inc-glm53f-095`` gave a
        # bank its own classifier kind. Until then a bank answered
        # ``MAPPED_KEY_QUANTISED_WEIGHT`` and its weight keys were written as
        # fp8 at ``MINI_WEIGHT_SHAPE``; with the fourth kind and this line
        # unchanged they would have fallen to the ``else`` below and been written
        # as bf16 at ``MINI_PLAIN_SHAPE`` -- a four-element "expert weight" that
        # no reading in this file would have named. This is the THIRD consumer of
        # ``classify_mapped_keys`` that ``classify_mapped_keys``'s own docstring
        # warned about, and the fix is to ask the question it means to ask.
        quantised_pair = classify_mapped_keys(keys) in (
            MAPPED_KEY_QUANTISED_WEIGHT,
            MAPPED_KEY_STACKED_BANK,
        )
        for key in key_list:
            if key in tensors:
                continue
            if key in overrides:
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
    * an EXPERT BANK is loaded by :func:`stacked_expert_bank_loader`
      (``inc-glm53f-095``), which stacks one rank's expert weights and leaves
      their scales to their own loader. So the population is again the weight
      keys, and at expert-parallel degree 1 -- every configuration in this file
      -- one rank owns every expert, so the sum below is the whole bank. The
      line needs no bank branch to say that, which is why it has none.

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
    (``model_fp8.py:3379-3384``), so even a dict-level reading needs ONE
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
    (``model_fp8.py:3418``) cannot be reached while a routed expert bank refuses,
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

    (ii) On a configuration with a routed expert bank whose owning module
    declares NO EXPERT GEOMETRY the load REFUSES, naming the parameter, its key
    count and the missing declaration, and leaves NOTHING behind -- 0 not-lazy
    and 0 materialised placeholders. The refusal IS the reading here. A skipped
    parameter would not be.

    RE-ANCHORED BY ``inc-glm53f-095`` (design entry ``design-20260905-r``). This
    reading used to exercise the refusal on a WELL-FORMED bank, which was correct
    while no loader in the package could stack one. ``-095`` gives that case a
    loader, so the old form asserted a refusal the package no longer owes --
    measured before a line of it was written, in
    ``increments/probe-095-refusal-collision-host-r2.out``: the nine items here
    read ``9 passed`` unpatched and ``2 failed, 7 passed`` with the bank refusal
    monkeypatched away, this item being one of the two. What the reading
    CERTIFIES is unchanged -- that a refusal on the load path leaves the tree
    byte-for-byte as it arrived -- so it moved to the bank shape that still
    refuses rather than being deleted. It doubles as ``-095`` conjunct (1)'s
    control, where the same bank WITH its geometry declared loads E/E experts.

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
    written = _write_miniature_checkpoint(dense_dir, mappings, model)
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

    # ── (ii) a bank with no declared geometry refuses and leaves nothing ─────
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
        "the routed configuration produced no multi-scale-key entry, so this "
        "reading would certify nothing; the refusal it exercises could not fire"
    )

    # WITHHOLD THE GEOMETRY DECLARATION, on the instance, before the load.
    # ``local_expert_indices`` is a method on ``Glm5NextRoutedExperts``, so it
    # cannot be deleted from an instance; assigning ``None`` shadows it in the
    # instance dict, which is exactly what the loader's own check reads
    # (``getattr(owner, "local_expert_indices", None)`` then ``callable``). One
    # attribute is withheld and nothing else about the tree changes, so the
    # refusal below can only be about the missing declaration.
    withheld = 0
    for _, module in routed.named_modules():
        if callable(getattr(module, "local_expert_indices", None)):
            module.local_expert_indices = None
            withheld += 1
    print(f"CONJUNCT1_MODULES_WITH_GEOMETRY_WITHHELD={withheld}")
    assert withheld > 0, (
        "no module in this tree declares local_expert_indices, so withholding it "
        "changed nothing and the refusal below would be about something else"
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
    assert "DECLARES NO EXPERT GEOMETRY" in message, (
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


# --------------------------------------------------------------------------- #
# (2) The map handed to the reader covers the in-scope checkpoint, both ways.
# --------------------------------------------------------------------------- #


class _MapCaptured(Exception):
    """Conjunct 2's own stop, raised by its observer once it holds the map.

    RE-ANCHORED by plan revision 182 ruling (b). This class exists so that the
    item below ends its run on something this file owns, instead of on whatever
    production code happens to refuse next.
    """


def test_the_map_load_weights_hands_over_covers_the_in_scope_index(
    tmp_path,
    monkeypatch,
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

    HOW: THE CAPTURE IS AT THE HAND-OVER ITSELF. RE-ANCHORED by plan revision 182
    ruling (b) and by the B65r2 rider (revision 161), which said this observer
    moves to the reader call when ``inc-glm53f-095`` lands. It could not before: a
    routed expert bank refused inside step 3, so no run reached the reader and the
    observer sat one step earlier, on ``_materialise_declared_parameters``, with a
    7-line window after it in which an in-place mutation of ``mappings`` would
    have moved neither ``ast`` count. The bank loads now, the observer is
    ``SafetensorsCheckpoint.load_sharded_pipelined``, and the captured object IS
    the object handed over. ``_mappings_flow_in_load_weights`` stays as a
    cross-check on the source, no longer as the identity argument.

    THE STOP IS THIS ITEM'S OWN, NOT PRODUCTION'S. The observer raises
    ``_MapCaptured`` once it holds the map and never calls the real reader, which
    keeps this run off the process group that reader's default store needs. The
    earlier form ended on a bank refusal that belongs to conjunct 1 of
    ``test_every_declared_parameter_is_materialised_and_loaded``, and
    ``inc-glm53f-095`` was about to remove it.

    TWO READINGS RE-EXPRESSED, NONE LOST. The 576 and 288 key counts came out of
    the refusal message; they are readings on the captured bank entries now, one
    weight and one scale key per published routed expert. The zero-parameters
    reading was about a clean refusal -- conjunct 1's subject, asserted at reading
    (ii) above -- so its counterpart is asserted here: the parameter list is
    NON-EMPTY at the hand-over, which is what shows the capture happened after
    step 3 rather than instead of it.

    No process group is initialised, on purpose. The fixture that provides one
    exists for the reader's default-store call inside ``load_sharded_pipelined``,
    which ``_MapCaptured`` guarantees this run never reaches.

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

    # The reader is a method on the checkpoint object ``load_weights`` builds
    # locally (``model_fp8.py:3736``), so the patch is on the class and the
    # fixture restores it. The call site passes all five arguments positionally,
    # so a signature change there reaches this observer as a loud TypeError.
    def observer(_checkpoint, rank, world_size, _model, handed_over, _device):
        observed.append(handed_over)
        raise _MapCaptured(
            f"conjunct 2 holds the handed-over map: {len(handed_over)} entries, "
            f"rank {rank} of world size {world_size}"
        )

    monkeypatch.setattr(
        SafetensorsCheckpoint, "load_sharded_pipelined", observer, raising=True
    )

    with pytest.raises(_MapCaptured) as capture:
        model.load_weights(
            str(_token_checkpoint_directory(tmp_path)),
            torch.device("cpu"),
            None,
        )

    materialised_at_handover = len(list(model.named_parameters()))
    print(f"CONJUNCT2_STOP_CLASS={type(capture.value).__name__}")
    print(f"CONJUNCT2_CAPTURED_MAPS={len(observed)}")
    print(f"CONJUNCT2_MATERIALISED_AT_HANDOVER={materialised_at_handover}")

    assert len(observed) == 1, (
        f"the observer recorded {len(observed)} maps where conjunct 2 needs "
        f"exactly the one load_weights built"
    )
    assert materialised_at_handover > 0, (
        "the hand-over was reached with an unmaterialised tree, so the capture "
        "happened instead of step 3 rather than after it"
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

    # The two key counts, read off the captured map (plan revision 182 ruling
    # (b)). A routed bank entry is named by ``_add_moe_mlp``
    # (``weight_loaders_fp8.py:679``); the shared-expert entries are a different
    # shape and ``.mlp.shared_experts.`` does not contain ``.mlp.experts.``.
    bank_suffixes = ("gate_proj_weight", "up_proj_weight", "down_proj_weight")
    banks = {
        name: keys
        for name, keys in mappings.items()
        if ".mlp.experts." in name and name.endswith(bank_suffixes)
    }
    experts = real_config.text_config.n_routed_experts
    bank_key_counts = sorted({len(keys) for keys in banks.values()})
    bank_scale_counts = sorted({len(scale_keys(keys)) for keys in banks.values()})
    bank_kinds = sorted({classify_mapped_keys(keys) for keys in banks.values()})
    print(f"CONJUNCT2_PUBLISHED_ROUTED_EXPERTS={experts}")
    print(f"CONJUNCT2_BANK_ENTRIES={len(banks)}")
    print(f"CONJUNCT2_BANK_ENTRY_KEY_COUNTS={bank_key_counts}")
    print(f"CONJUNCT2_BANK_ENTRY_SCALE_COUNTS={bank_scale_counts}")
    print(f"CONJUNCT2_BANK_ENTRY_KINDS={bank_kinds}")

    assert banks, (
        "the captured map holds no routed expert bank entry, so the two key "
        "counts this conjunct reads have no subject"
    )
    assert experts == 288, (
        f"the published config declares {experts} routed experts, so the 576 and "
        f"288 this conjunct reads are no longer the counts of record"
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


# --------------------------------------------------------------------------- #
# ``inc-glm53f-091b`` -- conjuncts 4, 5, 6 and 8, plus the placeholder-dtype
# item the rev 165 ruling added.
#
# FIVE more counted items, one per conjunct and one for the ruling, which took
# this file to NINE with no ``parametrize`` decorator anywhere in it (D1.2).
# ``inc-glm53f-095``'s four take it to THIRTEEN, below. The helpers below belong to
# this half and are kept together so a reader can see which half owns what.
# --------------------------------------------------------------------------- #


def _out_of_band_entries(mappings: dict[str, str | list[str]]) -> dict[str, str]:
    """Every map entry whose scale the loader DROPS, as ``{param: scale key}``.

    An entry holding a weight and exactly one scale is served by
    ``wrap_with_blockwise_fp8_downscale``, whose base transform keeps
    ``slices[0]`` (``weight_loaders_fp8.py:1339``), so its scale reaches nothing
    through the reader and has to be read out of band. Derived from the map here
    rather than restated from a table, so this population and the production
    reader's cannot disagree about which entries they mean.
    """
    found: dict[str, str] = {}
    for name in mappings:
        keys = _keys_of(mappings, name)
        scales = [k for k in keys if _is_scale_key(k)]
        if len(keys) >= 2 and len(scales) == 1:
            found[name] = scales[0]
    return found


def _scale_attribute_of(param_name: str) -> tuple[str, str]:
    """The module path and attribute name a dropped scale is stored under.

    One helper, so the test and the production reader cannot spell that
    attribute two different ways.
    """
    module_path, _, leaf = param_name.rpartition(".")
    base = leaf[: -len("_weight")] if leaf.endswith("_weight") else leaf
    return module_path, f"{base}_{FP8_SCALE_SUFFIX}"


def _derived_dsa_scale_names(config: Glm5NextConfig) -> set[str]:
    """``inc-glm53f-085``'s difference set, DERIVED and never typed as 44.

    B65-N3's ask. The size comes from the sparse-attention layer set the config
    declares times the landed projection tuple, so a config with a different
    schedule yields a different number and this reading still means the same
    thing.
    """
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
    """The reader's dtype-mismatch log lines, collected while a load runs.

    NOT ``pytest.warns``. The reader reports a dtype it did not expect through
    ``logger.warning`` and then CASTS the tensor
    (``utils/checkpoints.py:570-576``), so the signal is a log record rather than
    a Python warning, and the cast is what makes the count matter: the line is
    not advisory, it is the narrowing being announced.

    Matching is on the message's own two fixed phrases rather than on a logger
    name, so a reader that moves module does not silently empty this list.
    """
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


def _real_call_sites(function_name: str) -> tuple[int, list[str]]:
    """Real call sites of ``function_name`` under ``vllm_neuron/``, and the rest.

    Returns ``(real call count, every excluded mention as a printable line)``.

    THE COUNTING RULE IS THE PARSER, not a pattern. Every ``.py`` file under the
    package is parsed and a hit counts only when it is an :class:`ast.Call` whose
    callee carries that name. That excludes all four declared classes by
    construction -- a string literal, a docstring, a comment, and another
    module's same-named function reached only through its own import -- instead
    of by four hand-written filters that could each be wrong on their own.

    Every mention the rule EXCLUDED is returned, so the rule is read rather than
    trusted: the item prints those lines and a reader can judge them.
    """
    package = Path(vllm_neuron.__file__).parent
    real = 0
    excluded: list[str] = []
    for path in sorted(package.rglob("*.py")):
        text = path.read_text()
        if function_name not in text:
            continue
        call_lines: set[int] = set()
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
                call_lines.add(node.lineno)
                real += 1
        for number, line in enumerate(text.split("\n"), start=1):
            if function_name in line and number not in call_lines:
                excluded.append(
                    f"{path.relative_to(package.parent)}:{number}: {line.strip()}"
                )
    return real, excluded


def _statement_positions(method, *, calls: tuple[str, ...], anchor: str):
    """Where a method calls each named callee, and where its anchor line sits.

    Returns ``(positions, anchor position)`` as statement line numbers relative
    to the method, read by :mod:`ast`. A position of ``-1`` means not present.

    A POSITION READ, not a text search. A diff that moves a call above the
    anchor moves a number here, so conjunct 8's ordering fails by arithmetic
    rather than by anyone's judgement.

    EARLIEST FOR A CALL, LATEST FOR THE ANCHOR, with every position found
    printed. RE-ANCHORED by rider B69r2-N3 (plan revision 180) at
    ``inc-glm53f-095``, the first increment to touch this file since. An earlier
    form kept ONE position per callee and overwrote it, so a SECOND, EARLY call to
    the same callee moved neither of conjunct 8's ordering reads: the misplaced
    call was invisible to both, and only the miniature's dynamic pre-flight could
    catch it. ``min`` for a call and ``max`` for the anchor is what the ordering
    claim actually says -- the EARLIEST prep call comes after the LATEST anchor
    call. ``ast.walk`` is breadth-first, so "the last one seen" was never reliably
    the last one in the source either. No caller changes: at the shipped source
    every callee here has exactly one call site, so ``min`` and ``max`` read the
    same number the overwriting form read.
    """
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

    every_position = {name: sorted(lines) for name, lines in found.items()}
    print(f"STATEMENT_POSITIONS_EVERY_CALL={every_position}")
    print(f"STATEMENT_POSITIONS_EVERY_ANCHOR={sorted(anchor_found)}")
    positions = {name: (min(lines) if lines else -1) for name, lines in found.items()}
    anchor_at = max(anchor_found) if anchor_found else -1
    return positions, anchor_at


#: The two load-time preps conjunct 8 pins, and the one method allowed to call
#: them. Named once so the item, its control and the mutation builder cannot
#: drift apart.
_LOAD_TIME_PREPS = ("prepare_projection_weights", "prepare_scale_operands")
_PREP_CALLER = "_run_load_time_preps"

#: What the mutation builder puts where it deletes a call. Named rather than
#: inlined because a bare keyword inside an f-string trips the linter's tokeniser.
_NO_OP_STATEMENT = "pass"


def _prep_call_homes(source: str) -> dict[str, list[tuple[str, int]]]:
    """Every load-time prep call site, with the function it actually lives in.

    Takes SOURCE TEXT rather than a module or a live object, and that signature is
    the point: the same predicate can then be run against a deliberately mutated
    COPY of the text, which is how conjunct 8's negative control fires without a
    byte being written to the tree.

    Returns ``{prep name: [(enclosing function, file line), ...]}``. A prep with
    no call site maps to an empty list rather than vanishing from the mapping, so
    a missing call reads as a missing call and not as a missing key.

    THE ENCLOSING FUNCTION IS READ FROM THE TREE -- the innermost
    :class:`ast.FunctionDef` whose line span contains the call -- and never from
    indentation or from a backwards text search for a ``def``. Call sites are
    :class:`ast.Call` nodes, so the four mentions inside string literals and the
    two ``def`` statements of the preps themselves are excluded by construction.
    """
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
    """Conjunct 8's part two: each prep is called ONCE, and only from the caller.

    Returns ``(verdict, homes)``.

    THE ITEM AND ITS NEGATIVE CONTROL BOTH CALL THIS ONE FUNCTION. A control that
    re-implements the predicate proves something about the re-implementation and
    nothing about the predicate the item uses, which is the defect B69-M1 found
    in the first form of this item.
    """
    homes = _prep_call_homes(source)
    verdict = all(
        len(sites) == 1 and sites[0][0] == _PREP_CALLER for sites in homes.values()
    )
    return verdict, homes


def _source_with_the_scale_prep_moved_out(source: str) -> str:
    """B69's mutation, applied to a COPY of the module text. The tree is untouched.

    It deletes the ``prepare_scale_operands`` call from the caller and plants an
    equivalent call inside ``load_weights`` immediately ABOVE the anchor -- the
    exact shape the reviewer used to make the unrepaired item pass on wrong code.

    The result is only ever PARSED. It is never written to disk and never
    executed, so the planted call's names need not resolve to anything.
    """
    def indent_of(line: str) -> str:
        return " " * (len(line) - len(line.lstrip()))

    lines = source.split("\n")
    ((_, call_line),) = _prep_call_homes(source)["prepare_scale_operands"]
    anchor_line = next(
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load_state_dict"
    )
    planted = "module.prepare_scale_operands(**operands)"
    # The removal first, because it keeps the line count fixed; the insertion
    # second, because it shifts every line after it.
    lines[call_line - 1] = indent_of(lines[call_line - 1]) + _NO_OP_STATEMENT
    lines.insert(anchor_line - 1, f"{indent_of(lines[anchor_line - 1])}{planted}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# (4) The scale grids stay fp32, and the ones the loader drops are read.
# --------------------------------------------------------------------------- #


def test_the_scale_grids_stay_fp32(tmp_path, single_rank_process_group) -> None:
    """(4) Five readings in one item, per the block's rev 137 bullet.

    CERTIFYING COMPONENT (D1.4): ``load_weights``'s out-of-band scale read --
    which scales it reads, from where, at what dtype, and what it does when one
    is missing. Not ``blockwise_scale_loader``, which conjunct 6 certifies, and
    not the placeholder rule, which the ruling's own item certifies.

    (i) every dropped scale arrives fp32 on the target device; (ii) every scale
    grid that DOES travel through the map arrives fp32 too; (iii) the load emits
    no dtype-override line, with a control that makes that zero mean something;
    (iv) ``inc-glm53f-085``'s difference set is what this file DERIVES it to be
    rather than the literal 44 (B65-N3); (v) an ABSENT scale key refuses by
    name instead of reading a default of 1.0.

    Reading (v) calls the read directly with one map entry pointing at a key the
    checkpoint does not hold. It stays in this item because it certifies the same
    component: what the out-of-band read does with the key it was asked for.
    """
    model = _dense_model()
    mappings = _mappings_for(_dense_config())
    directory = tmp_path / "dense"
    _write_miniature_checkpoint(directory, mappings, model)

    dropped = _out_of_band_entries(mappings)
    print(f"CONJUNCT4_OUT_OF_BAND_POPULATION={len(dropped)}")
    assert dropped, "no map entry drops a scale, so this item measures nothing"

    with _captured_cast_lines() as cast_lines:
        model.load_weights(str(directory), torch.device("cpu"), None)

    # (i) Every dropped scale reached its own module, fp32, on the target device.
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
    print(f"CONJUNCT4_DROPPED_SCALES_READ_AS_FP32={fp32}")
    assert fp32 == len(dropped)

    # (ii) Every scale grid that travels THROUGH the map is fp32 as well.
    loaded = dict(model.named_parameters())
    through_map = [name for name in mappings if name.endswith(FP8_SCALE_SUFFIX)]
    print(f"CONJUNCT4_SCALE_GRID_PARAMETERS={len(through_map)}")
    assert through_map, "the map carries no scale-grid parameter at all"
    not_fp32 = [n for n in through_map if loaded[n].dtype is not torch.float32]
    print(f"CONJUNCT4_SCALE_GRID_PARAMETERS_NOT_FP32={len(not_fp32)}")
    assert not_fp32 == [], f"these scale grids are not fp32: {not_fp32[:4]}"

    # (iii) A counted zero with a control that MOVES (D1.5).
    print(f"CONJUNCT4_DTYPE_CAST_LINES={len(cast_lines)}")
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
    print(f"CONJUNCT4_CONTROL_DTYPE_CAST_LINES={len(control_lines)}")
    assert len(control_lines) > 0, (
        "a load whose every placeholder took the config dtype cast nothing, so "
        "the zero above is vacuous"
    )

    # (iv) B65-N3: -085's difference set, derived here rather than assumed.
    quantised = set(build_weight_mappings(model.text_config, quantised=True))
    plain = set(build_weight_mappings(model.text_config, quantised=False))
    derived = _derived_dsa_scale_names(_dense_config())
    print(f"CONJUNCT4_DERIVED_DIFFERENCE_SIZE={len(derived)}")
    print(f"CONJUNCT4_QUANTISED_MINUS_PLAIN={len(quantised - plain)}")
    assert plain <= quantised, f"names only in plain: {sorted(plain - quantised)}"
    assert quantised - plain == derived, (
        f"symmetric difference: {sorted((quantised - plain) ^ derived)[:4]}"
    )
    assert len(plain - quantised) == 0
    assert len(plain - (quantised - {sorted(plain)[0]})) == 1, (
        "the zero above is vacuous"
    )

    # (v) A scale key the checkpoint does not hold refuses BY NAME rather than
    # reading a default of 1.0.
    #
    # A DIRECT CALL on the read, and that is a deliberate choice. Withholding the
    # key from the FILE instead would measure the pipelined reader, which refuses
    # a missing key of its own long before this method runs; the subject here is
    # what the out-of-band read does with the key it was asked for. Conjunct 8
    # takes its two refusals the same way and for the same reason.
    victim_param, victim_key = sorted(dropped.items())[0]
    absent_key = f"absent.{victim_key}"
    tampered = dict(mappings)
    tampered[victim_param] = [
        key for key in _keys_of(mappings, victim_param) if key != victim_key
    ] + [absent_key]
    print(f"CONJUNCT4_ABSENT_SCALE_KEY={absent_key}")
    opened = SafetensorsCheckpoint(str(directory))
    with pytest.raises(Glm5NextWeightLoadError) as refusal:
        _dense_model()._load_out_of_band_scales(
            opened, tampered, torch.device("cpu")
        )
    message = str(refusal.value)
    print(f"CONJUNCT4_REFUSAL_NAMES_THE_KEY={absent_key in message}")
    print(f"CONJUNCT4_REFUSAL_NAMES_THE_PARAMETER={victim_param in message}")
    assert absent_key in message, message
    assert victim_param in message, message
    assert "1.0" in message, (
        f"the refusal does not say what it refuses to do, which is the whole "
        f"reason it is not a default: {message}"
    )
    # The refusal's own control, and it MOVES: the SAME direct call with the map
    # untampered reads every dropped scale instead of refusing, so the refusal
    # above is about the absent key and not about this call shape.
    read = _dense_model()._load_out_of_band_scales(
        SafetensorsCheckpoint(str(directory)), mappings, torch.device("cpu")
    )
    print(f"CONJUNCT4_UNTAMPERED_DIRECT_CALL_READ={read}")
    assert read == len(dropped)


# --------------------------------------------------------------------------- #
# (5) The orphaned preps now have exactly one production caller each.
# --------------------------------------------------------------------------- #


def test_the_load_time_preps_have_exactly_one_production_call_site_each() -> None:
    """(5) A wiring count, and the rule that produces it is printed.

    CERTIFYING COMPONENT (D1.4): the wiring inside ``load_weights``, not the
    arithmetic either prep performs. Both preps had ZERO production call sites
    before this increment: every mention under ``vllm_neuron/`` was a test, a
    docstring, a comment or an error-message literal.

    THE COUNT IS 1 AND NOT "AT LEAST 1" on purpose. Two call sites would mean
    the operands are built twice, and the second build would silently replace the
    first -- both preps store by plain ``setattr``, so nothing would report it.
    """
    for prep in ("prepare_projection_weights", "prepare_scale_operands"):
        real, excluded = _real_call_sites(prep)
        print(f"CONJUNCT5_{prep}_REAL_CALL_SITES={real}")
        print(f"CONJUNCT5_{prep}_EXCLUDED_MENTIONS={len(excluded)}")
        for line in excluded:
            print(f"  EXCLUDED {line}")
        assert real == 1, (
            f"{prep} has {real} production call sites where the block declares "
            f"exactly one; the excluded mentions above are the rule being read"
        )

    # The control: the same rule, on a name that IS called more than once in the
    # package, reads more than one. Without it a rule that counted nothing
    # everywhere would satisfy the assertions above.
    control_name = "build_weight_mappings"
    control_real, _ = _real_call_sites(control_name)
    print(f"CONJUNCT5_CONTROL_{control_name}_REAL_CALL_SITES={control_real}")
    assert control_real > 1, (
        f"the counting rule found {control_real} call sites of {control_name}, "
        f"so it cannot be shown able to count more than one"
    )

    # And the second control: a name nothing calls reads exactly 0, so a rule
    # that counted every mention would fail here.
    absent = _real_call_sites("prepare_projection_weights_that_does_not_exist")
    print(f"CONJUNCT5_CONTROL_ABSENT_NAME_REAL_CALL_SITES={absent[0]}")
    assert absent[0] == 0


# --------------------------------------------------------------------------- #
# (6) ``blockwise_scale_loader``'s arity contract.
# --------------------------------------------------------------------------- #


def test_the_blockwise_scale_loader_arity_contract_is_honoured() -> None:
    """(6) Every entry it serves carries one key, and more than one refuses.

    CERTIFYING COMPONENT (D1.4): the loader choice for a lone scale grid -- that
    the entries routed to ``blockwise_scale_loader`` are exactly the ones it can
    serve. Its refusal is landed code; what this item certifies is that nothing
    in the map reaches it with the wrong arity.

    Counted on the REAL configuration, not the miniature, because that is where
    every scale-grid family exists at once.
    """
    real_config = Glm5NextConfig.from_configs(json.loads(REAL_CONFIG_PATH.read_text()))
    mappings = _mappings_for(real_config)
    grids = [
        name
        for name in mappings
        if classify_mapped_keys(mappings[name]) == MAPPED_KEY_SCALE_GRID
    ]
    print(f"CONJUNCT6_SCALE_GRID_ENTRIES={len(grids)}")
    assert grids, "the real map has no scale-grid entry, so this item is vacuous"

    wrong_arity = [name for name in grids if len(_keys_of(mappings, name)) != 1]
    print(f"CONJUNCT6_SCALE_GRID_ENTRIES_NOT_CARRYING_ONE_KEY={len(wrong_arity)}")
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
    print(f"CONJUNCT6_NON_GRID_ENTRIES_WITH_MORE_THAN_ONE_KEY={len(others)}")
    assert others, (
        "no entry anywhere in the map carries more than one key, so the count "
        "above cannot distinguish a grid from anything else"
    )

    # The refusal, exercised once BY NAME. Two slices of the shape a real grid
    # has, so the refusal is reached on arity and not on a shape check.
    grid = torch.full(MINI_SCALE_SHAPE, 0.5, dtype=torch.float32)
    loader = blockwise_scale_loader(param_name=sorted(grids)[0])
    with pytest.raises(Glm5NextWeightMapError) as refusal:
        loader.load([_WholeTensorSlice(grid), _WholeTensorSlice(grid)], 0)
    message = str(refusal.value)
    print(f"CONJUNCT6_REFUSAL={message}")
    assert "expects 1 slice" in message and "got 2" in message, message

    # And the same loader accepts exactly one, so the refusal above is about the
    # arity rather than about this test's stand-in slice.
    accepted = loader.load([_WholeTensorSlice(grid)], 0)
    print(f"CONJUNCT6_ONE_SLICE_ACCEPTED_DTYPE={accepted.dtype}")
    assert accepted.dtype is torch.float32


class _WholeTensorSlice:
    """The one thing a loader transform does to a slice: ``slice[:]``.

    A stand-in for ``PySafeSlice`` so the arity refusal can be reached without a
    checkpoint file. It supports exactly the operation the transforms use, and
    nothing else, so it cannot quietly satisfy a transform that did something
    different.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    def __getitem__(self, item):
        return self._tensor[item]


# --------------------------------------------------------------------------- #
# (8) The preps run after the device, by name.
# --------------------------------------------------------------------------- #


def test_the_load_time_preps_run_after_the_device_by_name(
    tmp_path, single_rank_process_group
) -> None:
    """(8) An ordering read by position, plus both refusal cases by name.

    CERTIFYING COMPONENT (D1.4): the ordering inside ``load_weights`` -- a
    different component from conjunct 6's arity and conjunct 5's wiring count.

    THE ANCHOR IS ``load_state_dict(..., assign=True)`` and not
    ``load_sharded_pipelined``, because the reader only returns a state dict:
    that line is what turns a shape-free placeholder into a real on-device
    tensor (qwen3 precedent ``qwen3/model.py:1022-1033``).

    THE ORDERING IS READ IN TWO PARTS, because the two prep calls sit one level
    down in ``_run_load_time_preps``. Part one: inside ``load_weights`` the call
    to that method comes AFTER the anchor. Part two: EACH PREP CALL'S OWN
    ENCLOSING FUNCTION IS THAT CALLER, read by :mod:`ast` over the module source,
    and both prep call lines are recorded. Five positions, not three.

    PART TWO IS THE B69-M1 REPAIR AND THIS PARAGRAPH IS WHY IT WAS NEEDED. The
    first form of this item argued that conjunct 5's count of exactly one call
    site each already ruled out any earlier path. That inference does not follow:
    one call site says nothing about WHERE that site is. The reviewer moved the
    scale prep out of the caller to directly above the anchor and every one of
    the nine items still passed -- part one keeps reading the CALLER's position,
    which had not moved, and the misplaced call is never executed on the
    miniature (no shared expert) or on the real configuration (the load stops at
    the bank refusal first). A proxy plus an inference is not a position read.

    SO THE ITEM NOW CARRIES THAT MUTATION AS ITS NEGATIVE CONTROL (D1.5): the
    same predicate is run against a mutated COPY of the module text and must
    FAIL on it. The copy is parsed, never written and never executed.
    """
    positions, anchor = _statement_positions(
        Glm5NextForConditionalGeneration.load_weights,
        calls=("_run_load_time_preps", "_load_out_of_band_scales"),
        anchor="load_state_dict",
    )
    print(f"CONJUNCT8_ANCHOR_POSITION={anchor}")
    print(f"CONJUNCT8_PREP_CALL_POSITION={positions['_run_load_time_preps']}")
    print(f"CONJUNCT8_SCALE_READ_POSITION={positions['_load_out_of_band_scales']}")
    assert anchor > 0, "load_weights no longer calls load_state_dict at all"
    for name, position in positions.items():
        assert position > 0, f"load_weights no longer calls {name}"
        assert position > anchor, (
            f"{name} is called at statement {position}, before the anchor at "
            f"{anchor}; it would run on placeholders"
        )

    # The scale read comes before the preps, because a prep reads the scale the
    # read installs. A position, not a sentence.
    assert (
        positions["_load_out_of_band_scales"] < positions["_run_load_time_preps"]
    )

    # Part two: where each prep call ITSELF lives. Read over the module's own
    # source text, so the reading is about the shipped file and not about an
    # object this test built.
    module_source = Path(
        inspect.getsourcefile(Glm5NextForConditionalGeneration)
    ).read_text()
    lives_in_the_caller, homes = _every_prep_call_lives_in_the_caller(module_source)
    for prep in _LOAD_TIME_PREPS:
        sites = homes[prep]
        print(f"CONJUNCT8_{prep.upper()}_CALL_SITES={len(sites)}")
        assert len(sites) == 1, f"{prep} has {len(sites)} call sites, not one: {sites}"
        enclosing, line = sites[0]
        print(f"CONJUNCT8_{prep.upper()}_CALL_LINE={line}")
        print(f"CONJUNCT8_{prep.upper()}_ENCLOSING_FUNCTION={enclosing}")
        assert enclosing == _PREP_CALLER, (
            f"{prep} is called from {enclosing!r} at line {line}, not from "
            f"{_PREP_CALLER!r} -- the ordering part one reads no longer governs it"
        )
    assert lives_in_the_caller

    # And the control, which MOVES (D1.5): B69's mutation on a COPY of that text
    # must FAIL the same predicate. The copy is parsed first on its own, so a
    # syntax error in the mutation cannot be mistaken for a moved call.
    mutated = _source_with_the_scale_prep_moved_out(module_source)
    ast.parse(mutated)
    assert mutated != module_source
    still_in_the_caller, mutated_homes = _every_prep_call_lives_in_the_caller(mutated)
    mutated_enclosing = mutated_homes["prepare_scale_operands"][0][0]
    print("CONJUNCT8_CONTROL_MUTATED_COPY_PARSES=True")
    print(f"CONJUNCT8_CONTROL_MUTATED_ENCLOSING_FUNCTION={mutated_enclosing}")
    print(f"CONJUNCT8_CONTROL_PREDICATE_ON_THE_MUTATED_COPY={still_in_the_caller}")
    assert mutated_enclosing == "load_weights", mutated_enclosing
    assert not still_in_the_caller, (
        "the predicate passed on a copy with the scale prep moved out of its "
        "caller, so it is still the hollow read B69-M1 found"
    )

    # The mutation stayed in memory. Read the file again and compare.
    on_disk = Path(inspect.getsourcefile(Glm5NextForConditionalGeneration)).read_text()
    print(f"CONJUNCT8_CONTROL_SOURCE_FILE_UNCHANGED={on_disk == module_source}")
    assert on_disk == module_source, "the mutation reached the file on disk"

    # Case A: the prep's own refusal when a scale was never materialised. It is
    # -090's landed message at ``model_fp8.py:1563-1568``, and NOT ``:1589``,
    # which is a different method's never-ran refusal.
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
    print(f"CONJUNCT8_CASE_A={str(case_a.value)}")
    assert "gate_proj_scale" in str(case_a.value), str(case_a.value)
    assert "load the checkpoint before preparing" in str(case_a.value)
    assert not hasattr(shared, Glm5NextSharedExperts.PREPARED_SCALE_OPERANDS_ATTR), (
        "the refusal left the operands attribute behind, so a later read would "
        "find a half-built dict"
    )

    # Case B: the pre-flight this increment authors. No refusal for it exists in
    # the landed code -- the .device count inside prepare_scale_operands is 0 --
    # so the check lives on the caller's side and is exercised here by name.
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
    print(f"CONJUNCT8_CASE_B={str(case_b.value)}")
    assert attn_path in str(case_b.value), str(case_b.value)
    assert "cpu" in str(case_b.value), str(case_b.value)
    assert "strand" in str(case_b.value), str(case_b.value)

    # Case B's control, and it MOVES (D1.5): the same operands against the
    # device they are actually on pass, and the count of what was checked is
    # nonzero, so the refusal above is about the device and not about the check
    # refusing everything.
    checked = model._require_prep_operands_on_device(
        attn_path, attn, names, torch.device("cpu")
    )
    print(f"CONJUNCT8_CASE_B_CONTROL_OPERANDS_CHECKED={checked}")
    assert checked > 0

    # And the reading the design names as case B's ground: prepare, move the
    # module, and find the prepared operands still on the device they were built
    # on. This is what makes the pre-flight necessary rather than decorative --
    # after the prep there is nothing left to detect.
    #
    # READ ON THE PROJECTION PREP, deliberately. Both preps store their product
    # the same way -- a plain dict attribute, which ``nn.Module._apply`` never
    # visits -- and this one needs no 256-block scale grid to run, so the reading
    # is about the STORAGE and not about a kernel geometry. A separate model, so
    # moving a module to ``meta`` cannot disturb the readings above.
    stranded = _dense_model()
    stranded.load_weights(str(directory), torch.device("cpu"), None)
    stranded_attn = stranded.get_submodule(attn_path)
    prepared_attr = type(stranded_attn).PREPARED_WEIGHTS_ATTR
    before = {
        name: operand.device.type
        for name, operand in getattr(stranded_attn, prepared_attr).items()
    }
    assert before, "the prep stored nothing, so this reading has no subject"
    stranded_attn.to("meta")
    print(
        "CONJUNCT8_PARAMETER_DEVICE_AFTER_MOVE="
        f"{next(iter(stranded_attn.parameters())).device.type}"
    )
    after = {
        name: operand.device.type
        for name, operand in getattr(stranded_attn, prepared_attr).items()
    }
    print(f"CONJUNCT8_OPERAND_DEVICES_BEFORE_MOVE={sorted(set(before.values()))}")
    print(f"CONJUNCT8_OPERAND_DEVICES_AFTER_MOVE={sorted(set(after.values()))}")
    assert set(before.values()) == {"cpu"}
    assert after == before, (
        "moving the module moved the prepared operands, so a prep-before-device "
        "ordering would be self-correcting and the pre-flight unnecessary"
    )


# --------------------------------------------------------------------------- #
# The rev 165 ruling's own item: the scaled MLA weights stay fp8.
# --------------------------------------------------------------------------- #


def test_the_scaled_mla_weights_reach_the_dequant_as_fp8(
    tmp_path, single_rank_process_group
) -> None:
    """The placeholder rule types a lone fp8 weight key fp8, not the config dtype.

    CERTIFYING COMPONENT (D1.4): ``_placeholder_dtype``'s sibling clause.

    WHY THIS ITEM EXISTS. ``inc-glm53f-085`` gave each of the four scaled MLA
    projections its own scale-grid entry, which left each of those weights alone
    in its entry -- and a lone weight key classifies ``plain``. Under the rule as
    first designed those weights took the config dtype, the reader narrowed the
    checkpoint's fp8 bytes to bf16 and then
    ``_dequantised_projection_weight`` saw a real dtype and returned the weight
    UNCHANGED: the dequant did nothing, silently, on every scaled projection of
    every sparse-attention layer. Measured before the clause existed, with the
    dense-MLP two-key entry as the control that still reached the dequant:
    ``probe-091b-dsa-dtype.out``, ``DEQUANT_BRANCHES_REACHED=0`` of 4.

    TWO POPULATIONS, both derived. The rule itself is read over the REAL
    configuration's scaled weights -- the whole 44 -- and the end-to-end
    consequence is read on the miniature, which is the only one a test can load.
    """
    # (i) The rule, over the real configuration's whole population.
    real_config = Glm5NextConfig.from_configs(json.loads(REAL_CONFIG_PATH.read_text()))
    real_model = Glm5NextForConditionalGeneration(real_config)
    real_mappings = _mappings_for(real_config)
    scale_names = _derived_dsa_scale_names(real_config)
    weight_names = sorted(
        f"{name[: -len('_' + FP8_SCALE_SUFFIX)]}_weight" for name in scale_names
    )
    print(f"RULING_REAL_SCALED_WEIGHTS={len(weight_names)}")
    assert weight_names, "the real config declares no scaled MLA weight"
    # The derivation is checked against the map before it is used, so a name this
    # file derived but the map does not carry fails HERE, by name, instead of as a
    # KeyError three lines down. The published skip list is why this matters: it
    # withholds ``kv_b_proj`` on every sparse-attention layer, and
    # ``DSA_SCALED_PROJECTIONS`` is the tuple that already excludes it.
    missing = [name for name in weight_names if name not in real_mappings]
    print(f"RULING_DERIVED_NAMES_THE_MAP_DOES_NOT_CARRY={len(missing)}")
    assert missing == [], f"derived but unmapped: {missing[:4]}"
    typed = [
        name
        for name in weight_names
        if real_model._placeholder_dtype(
            real_mappings[name], param_name=name, mappings=real_mappings
        )
        is torch.float8_e4m3fn
    ]
    print(f"RULING_REAL_SCALED_WEIGHTS_TYPED_FP8={len(typed)}")
    assert len(typed) == len(weight_names), (
        f"{len(weight_names) - len(typed)} scaled MLA weights would take a "
        f"placeholder dtype that is not fp8, e.g. "
        f"{sorted(set(weight_names) - set(typed))[:4]}"
    )
    # Each of them is a LONE key classified plain, which is the case the clause
    # exists for. Without this the assertion above would also pass if -085 were
    # reverted and the entries went back to being two-key lists.
    lone_plain = [
        name
        for name in weight_names
        if len(_keys_of(real_mappings, name)) == 1
        and classify_mapped_keys(real_mappings[name]) == MAPPED_KEY_PLAIN
    ]
    print(f"RULING_REAL_SCALED_WEIGHTS_THAT_ARE_LONE_PLAIN_KEYS={len(lone_plain)}")
    assert len(lone_plain) == len(weight_names)

    # The rule's control: an ordinary weight with no sibling scale entry still
    # takes the config dtype, so the clause did not simply type everything fp8.
    ordinary = sorted(
        name
        for name in real_mappings
        if name.endswith("_weight")
        and len(_keys_of(real_mappings, name)) == 1
        and classify_mapped_keys(real_mappings[name]) == MAPPED_KEY_PLAIN
        and name not in set(weight_names)
    )
    print(f"RULING_CONTROL_ORDINARY_LONE_WEIGHTS={len(ordinary)}")
    assert ordinary, "there is no unscaled lone weight key to control against"
    control_dtype = real_model._placeholder_dtype(
        real_mappings[ordinary[0]],
        param_name=ordinary[0],
        mappings=real_mappings,
    )
    print(f"RULING_CONTROL_PARAM={ordinary[0]}")
    print(f"RULING_CONTROL_DTYPE={control_dtype}")
    assert control_dtype is real_config.text_config.torch_dtype

    # (ii) The consequence, end to end on the miniature.
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
        print(f"  RULING_{leaf}_LOADED_DTYPE={weight.dtype}")
        assert _is_fp8_dtype(weight.dtype), (
            f"{attn_path}.{leaf}_weight arrived {weight.dtype}; the checkpoint "
            f"holds fp8 bytes, so the dequant branch is unreachable and the "
            f"bytes would be used as if they were numbers"
        )
        reached += 1
    print(f"RULING_DEQUANT_BRANCHES_REACHED={reached}")
    print(f"RULING_SCALED_PROJECTIONS={len(DSA_SCALED_PROJECTIONS)}")
    assert reached == len(DSA_SCALED_PROJECTIONS)

    # The end-to-end control: the dense-MLP weight, whose scale travels in the
    # SAME entry, was fp8 before this clause existed and still is. If it had
    # regressed, the clause would have moved the wrong case.
    mlp_path = attn_path.rsplit(".", 1)[0] + ".mlp"
    mlp_weight = getattr(model.get_submodule(mlp_path), "gate_proj_weight")
    print(f"RULING_CONTROL_TWO_KEY_ENTRY_DTYPE={mlp_weight.dtype}")
    assert _is_fp8_dtype(mlp_weight.dtype)


# --------------------------------------------------------------------------- #
# inc-glm53f-095 -- the expert-stacked load. Four items, selected by ``-k
# stacked``, one per conjunct.
#
# WHY THESE ITEMS BRING THEIR OWN CONFIGURATION. The landed ``_routed_config``
# exists to REFUSE ("so the load must refuse by name", ``:186``), and a routed
# load that completes runs one thing that configuration was never asked to
# survive: ``Glm5NextSharedExperts.prepare_scale_operands``, which reaches
# ``scale_grid_shape`` and demands extents divisible by ``BLOCK_QUANT_SIZE`` =
# 256. The miniature is 128, so the routed load dies in the SHARED-expert prep
# -- nothing to do with the bank. That is a pre-existing constraint of the
# miniature the bank refusal has been masking, and it is measured rather than
# argued: ``increments/probe-095-collateral-host.out`` reads
# ``BlockwiseFp8MmError: weight extent [128,128] is not a whole number of
# 256x256 blocks`` from ``blockwise_fp8_mm.py:283-287``, reached through
# ``model_fp8.py:1576``.
#
# The bank does not need that prep. ``Glm5NextRoutedExperts`` defines NEITHER
# load-time prep -- only ``Glm5NextSharedExperts`` (scale) and
# ``Glm5NextMLAAttention`` (projection) do -- so ``_run_load_time_preps``'s
# ``hasattr(type(module), ...)`` gate never visits a bank at all. So these items
# set ``n_shared_experts=0``, which ``model_fp8.py:1857`` reads as "build no
# shared-expert module", and the routed load completes on the bank's own path.
# The landed constants are REUSED rather than copied, so a change to the
# miniature moves these items with the other nine.
# --------------------------------------------------------------------------- #

#: The expert-parallel degree conjunct 3 splits the bank across, and the count
#: it expects on each rank. Two ranks over ``MINI_ROUTED_EXPERTS`` experts.
STACKED_EP_DEGREE = 2
STACKED_EXPERTS_PER_RANK = MINI_ROUTED_EXPERTS // STACKED_EP_DEGREE


def _stacked_config() -> Glm5NextConfig:
    """A routed configuration whose load COMPLETES, so the bank can be read.

    ``_routed_config``'s fields with one change, ``n_shared_experts=0``, for the
    reason the section header records. Everything else -- layer count, expert
    count, the MLA widths -- is the landed constant, so these items and the other
    nine move together.
    """
    return Glm5NextConfig(
        text_config=Glm5NextTextConfig(
            num_hidden_layers=MINI_LAYERS,
            n_routed_experts=MINI_ROUTED_EXPERTS,
            n_shared_experts=0,
            first_k_dense_replace=MINI_FIRST_K_DENSE,
            tie_word_embeddings=False,
            **MINI_MLA_WIDTHS,
        )
    )


def _stacked_model() -> Glm5NextForConditionalGeneration:
    """The routed model at the fork's own expert-parallel degree, which is 1.

    ``Glm5NextForConditionalGeneration.__init__`` takes a config and nothing else
    (``model_fp8.py:3454``), so a degree is not something a test can pass in
    here. Conjunct 3 therefore declares its two-rank geometry on a stand-in owner
    (:func:`_stacked_bank_geometry`) and calls the loader directly, which is also
    the honest shape of that reading: the loader's contract is with whatever
    module declares the partition, not with this constructor.
    """
    return Glm5NextForConditionalGeneration(_stacked_config())


def _bank_entries(mappings: dict[str, str | list[str]]) -> dict[str, list[str]]:
    """Every map entry that is an expert bank, by the classifier's own answer.

    Asks ``classify_mapped_keys`` rather than counting scale keys again, so this
    population is the same one the loader chooser routes and cannot drift from
    it.
    """
    return {
        name: _keys_of(mappings, name)
        for name in mappings
        if classify_mapped_keys(mappings[name]) == MAPPED_KEY_STACKED_BANK
    }


def _checkpoint_tensor(directory: Path, key: str) -> torch.Tensor:
    """One checkpoint tensor, read through the same call the loaders read with."""
    with safe_open(str(directory / MINI_CHECKPOINT_FILE), framework="pt") as opened:
        return opened.get_slice(key)[:]


def _slice_pairs(directory: Path, keys: list[str]) -> list[torch.Tensor]:
    """A bank entry's keys as tensors, in checkpoint order, for a transform.

    A transform takes anything with ``__getitem__`` and ``get_shape``; plain
    tensors satisfy both, and reading them here rather than inside the item keeps
    the item's own lines about the reading it makes.
    """
    return [_checkpoint_tensor(directory, key) for key in keys]


def _distinguish_bank_experts(directory: Path, keys: list[str]) -> dict[str, float]:
    """Give every expert in one bank entry its OWN bytes, in the written file.

    ``_write_miniature_checkpoint`` writes every quantised weight as ``ones`` and
    every scale grid as ``0.5``. That is right for the readings that landed with
    it and useless for conjuncts (2) and (3): experts holding identical bytes
    cannot be told apart, so a rotation control cannot move and no stacked row can
    be attributed to an expert. MEASURED, not argued -- the first run of those two
    items failed on exactly that, the rotation reading ``0.0`` and stacked row 0
    matching all four references (``accept-095-pre-host.out``).

    So the file is re-written once here: expert ``e`` gets the value ``e + 1``
    through its whole weight and half that through its scale grid. Every other
    tensor is carried across unchanged, READ BACK FROM THE FILE rather than
    rebuilt, so this helper reaches no item that does not call it and no landed
    reading moves. Values 1 to E are exact in ``float8_e4m3fn`` and far below the
    240 squeeze ceiling, so the distinction survives both the dtype and the
    downscale.

    Returns ``{key: the value written}``, so an item can name what it expects
    rather than recompute the convention.
    """
    path = directory / MINI_CHECKPOINT_FILE
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as opened:
        for key in opened.keys():
            tensors[key] = opened.get_tensor(key)

    written: dict[str, float] = {}
    layout = bank_layout(keys, param_name="the bank entry under test")
    for expert in range(layout.experts):
        weight_key = keys[layout.weight_at[expert]]
        scale_key = keys[layout.scale_at[expert]]
        value = float(expert + 1)
        tensors[weight_key] = torch.full(
            tuple(tensors[weight_key].shape), value, dtype=torch.bfloat16
        ).to(torch.float8_e4m3fn)
        tensors[scale_key] = torch.full(
            tuple(tensors[scale_key].shape), value / 2.0, dtype=torch.float32
        )
        written[weight_key] = value
        written[scale_key] = value / 2.0

    save_file(tensors, str(path))
    return written


def _stacked_bank_geometry(experts: int, per_rank: int):
    """A stand-in owner declaring a fixed expert partition, for a direct call.

    Conjunct 3 needs a geometry it chose rather than the one the model resolved,
    so this is one object instead of a real model it does not need. It is the only
    caller's owner: the refusal reading builds its own bare object, because an
    owner that declares NOTHING is what that reading is about.
    """

    class Owner:
        num_routed_experts = experts

        def local_expert_indices(self, rank: int) -> tuple[int, ...]:
            return tuple(range(rank * per_rank, (rank + 1) * per_rank))

    return Owner()


def test_the_stacked_bank_delivers_every_expert_or_refuses_by_name(
    tmp_path, single_rank_process_group
) -> None:
    """(1) EVERY EXPERT ARRIVES, or the bank refuses and leaves nothing.

    Certifies :func:`stacked_expert_bank_loader` as ``load_weights`` reaches it
    through ``get_weight_loader``. TWO readings, and the second is the FIRST's
    CONTROL rather than a separate subject.

    (i) On a miniature checkpoint with a routed bank of E experts, each loaded
    bank parameter's element count equals the SUM of the element counts its OWN
    checkpoint weight slices imply -- derived from the file by
    :func:`_implied_numels`, never from a shape constant -- and its leading axis
    is E, so E/E experts are present. This is the reading ``-091a`` failed
    silently: it loaded 16,384 elements where its slices implied 65,536 and
    reported success, because nothing between the loader and the parameter
    validates a shape.

    (ii) THE SAME BANK, in the SAME run, REFUSES when its owning module declares
    no expert geometry -- ``Glm5NextExpertBankNotLoadableError`` naming the
    parameter, its key count and the missing declaration. That is what makes (i)
    a measurement instead of an observation: one thing changes, the geometry
    declaration, and the answer moves from "E/E loaded" to "refused by name". A
    reading that cannot move is not a reading, which is why ``-091``'s own
    refusal reading was re-anchored onto this one when this increment made the
    well-formed bank loadable (design entry ``design-20260905-r``).
    """
    directory = tmp_path / "stacked"
    model = _stacked_model()
    mappings = _mappings_for(_stacked_config())
    written = _write_miniature_checkpoint(directory, mappings, model)
    banks = _bank_entries(mappings)

    print(f"CONJUNCT1_CHECKPOINT_TENSORS={written}")
    print(f"CONJUNCT1_MAP_ENTRIES={len(mappings)}")
    print(f"CONJUNCT1_BANK_ENTRIES={len(banks)}")
    print(f"CONJUNCT1_EXPERTS_DECLARED={MINI_ROUTED_EXPERTS}")
    assert banks, (
        "this configuration produced no expert-bank entry, so conjunct (1) would "
        "certify nothing; the loader it exercises could not be reached"
    )

    before = _not_lazy_count(model)
    model.load_weights(str(directory), torch.device("cpu"), None)
    implied = _implied_numels(directory, mappings)
    loaded = dict(model.named_parameters())

    mismatches = {}
    axes = {}
    for name in sorted(banks):
        parameter = loaded[name]
        axes[name] = int(parameter.shape[0])
        if parameter.numel() != implied[name]:
            mismatches[name] = (parameter.numel(), implied[name])

    print(f"CONJUNCT1_NOT_LAZY_BEFORE={before}")
    print(f"CONJUNCT1_NOT_LAZY_AFTER={_not_lazy_count(model)}")
    print(f"CONJUNCT1_BANK_NUMEL_MISMATCHES={len(mismatches)}")
    print(f"CONJUNCT1_BANK_LEADING_AXES={sorted(set(axes.values()))}")
    print(
        "CONJUNCT1_ONE_BANK="
        f"{sorted(banks)[0]} numel={loaded[sorted(banks)[0]].numel()} "
        f"implied={implied[sorted(banks)[0]]}"
    )

    assert before == 0, (
        f"{before} parameters already held a real tensor before the load, so "
        f"this item cannot certify the load"
    )
    assert mismatches == {}, (
        f"{len(mismatches)} bank parameters received a different number of "
        f"elements than their own checkpoint slices imply, e.g. "
        f"{[(k, *mismatches[k]) for k in sorted(mismatches)[:3]]} as "
        f"(parameter, loaded, implied) -- an expert was dropped or invented"
    )
    assert set(axes.values()) == {MINI_ROUTED_EXPERTS}, (
        f"a bank's leading axis is not the expert count: read "
        f"{sorted(set(axes.values()))}, expected [{MINI_ROUTED_EXPERTS}]. At "
        f"expert-parallel degree 1 one rank owns every expert, so every bank "
        f"stacks E of them"
    )

    # ── (ii) the same bank, with the geometry declaration withheld ───────────
    name = sorted(banks)[0]
    with pytest.raises(Glm5NextExpertBankNotLoadableError) as refusal:
        stacked_expert_bank_loader(banks[name], param_name=name, owner=object())
    message = str(refusal.value)
    print(f"CONJUNCT1_UNDECLARED_REFUSAL={message[:160]!r}")
    assert name in message, (
        f"the refusal does not name the parameter it refused: {message}"
    )
    assert str(len(banks[name])) in message, (
        f"the refusal does not name the entry's key count: {message}"
    )
    assert "DECLARES NO EXPERT GEOMETRY" in message, (
        f"the refusal does not name the missing declaration: {message}"
    )


def test_the_stacked_bank_holds_each_expert_bit_identically(
    tmp_path, single_rank_process_group
) -> None:
    """(2) EACH EXPERT IS THE RIGHT ONE, BIT-IDENTICALLY, weight and scale.

    Certifies the INDEXING of :func:`stacked_expert_bank_loader` and
    :func:`stacked_expert_scale_loader`. A stack is a permutation of its inputs,
    so any difference at all is an indexing defect and the threshold is
    ``max abs diff == 0.0`` rather than a tolerance.

    WHAT EACH EXPERT IS COMPARED AGAINST, and why it is not the raw checkpoint
    tensor. The bank loader is composed under
    :func:`wrap_with_blockwise_fp8_downscale`, so on a platform that squeezes
    into the 240 range the stored bytes are the SQUEEZED ones -- and comparing
    against unsqueezed bytes would fail for a reason that has nothing to do with
    indexing. So expert ``e``'s reference is the checkpoint tensor through the
    SAME elementwise squeeze, :func:`downscale_fp8_weight_bytes`, and the scale
    row's reference is the grid through the SAME compensation,
    :func:`compensate_block_scales`. On a platform needing no squeeze both are
    the identity and the comparison is against the raw bytes, which is why the
    item states no platform of its own.

    THE CONTROL IS THE ROTATION. The same readings are taken against a reference
    list rotated by one expert, which must DIFFER -- otherwise a comparison that
    accidentally compared a tensor with itself, or a checkpoint whose experts all
    hold the same bytes, would satisfy the equality above and certify nothing.
    The landed writer gives every expert the SAME bytes, so
    :func:`_distinguish_bank_experts` re-writes the file first; the first run of
    this item failed on the rotation for that reason and not for another.
    """
    directory = tmp_path / "stacked"
    model = _stacked_model()
    mappings = _mappings_for(_stacked_config())
    _write_miniature_checkpoint(directory, mappings, model)
    banks = _bank_entries(mappings)
    name = sorted(banks)[0]
    keys = banks[name]
    layout = bank_layout(keys, param_name=name)
    values = _distinguish_bank_experts(directory, keys)
    print(f"CONJUNCT2_DISTINGUISHED_KEYS={len(values)}")

    weight_keys = [keys[i] for i in layout.weight_at]
    scale_key_list = [keys[i] for i in layout.scale_at]
    print(f"CONJUNCT2_ENTRY={name}")
    print(f"CONJUNCT2_KEYS={len(keys)} EXPERTS={layout.experts}")
    print(f"CONJUNCT2_FIRST_WEIGHT_KEY={weight_keys[0]}")
    print(f"CONJUNCT2_FIRST_SCALE_KEY={scale_key_list[0]}")

    owner = model.get_submodule(name.rsplit(".", 1)[0])
    slices = _slice_pairs(directory, keys)
    stacked = stacked_expert_bank_loader(
        keys, param_name=name, owner=owner
    ).transform(slices, 0)
    grids = stacked_expert_scale_loader(
        keys, param_name=name, owner=owner
    ).transform(slices, 0)

    weight_refs = [
        downscale_fp8_weight_bytes(_checkpoint_tensor(directory, key))
        for key in weight_keys
    ]
    scale_refs = [
        compensate_block_scales(_checkpoint_tensor(directory, key)).scale_inv
        for key in scale_key_list
    ]

    def worst(stack, refs) -> float:
        return max(
            float((stack[e].to(torch.float32) - refs[e].to(torch.float32)).abs().max())
            for e in range(len(refs))
        )

    aligned_weight = worst(stacked, weight_refs)
    aligned_scale = worst(grids, scale_refs)
    rotated_weight = worst(stacked, weight_refs[1:] + weight_refs[:1])
    rotated_scale = worst(grids, scale_refs[1:] + scale_refs[:1])

    print(f"CONJUNCT2_STACKED_SHAPE={tuple(stacked.shape)}")
    print(f"CONJUNCT2_SCALE_SHAPE={tuple(grids.shape)}")
    print(f"CONJUNCT2_WORST_ABS_DIFF_WEIGHT={aligned_weight}")
    print(f"CONJUNCT2_WORST_ABS_DIFF_SCALE={aligned_scale}")
    print(f"CONJUNCT2_CONTROL_ROTATED_WEIGHT={rotated_weight}")
    print(f"CONJUNCT2_CONTROL_ROTATED_SCALE={rotated_scale}")

    assert stacked.shape[0] == layout.experts, (
        f"the stack holds {stacked.shape[0]} experts, not {layout.experts}"
    )
    assert grids.shape[0] == layout.experts, (
        f"the scale stack holds {grids.shape[0]} rows, not {layout.experts}"
    )
    assert aligned_weight == 0.0, (
        f"expert weights are not bit-identical to their own checkpoint tensors "
        f"through the same squeeze: worst abs diff {aligned_weight}. A stack is "
        f"a permutation of its inputs, so this is an indexing defect"
    )
    assert aligned_scale == 0.0, (
        f"expert scale rows are not bit-identical to their own grids through the "
        f"same compensation: worst abs diff {aligned_scale}"
    )
    assert rotated_weight > 0.0 and rotated_scale > 0.0, (
        f"the rotation control did not move (weight {rotated_weight}, scale "
        f"{rotated_scale}), so the equalities above would hold for a stack in "
        f"any order and certify nothing about indexing"
    )


def test_the_stacked_bank_gives_each_rank_its_declared_experts(
    tmp_path, single_rank_process_group
) -> None:
    """(3) THE RANK'S SUBSET IS THE DECLARED SUBSET, counted both ways.

    Certifies that :func:`stacked_expert_bank_loader` selects THIS RANK's experts
    at the declared expert-parallel degree -- and it is the first test in this
    repository of
    :func:`~vllm_neuron.utils.weight_loader.expert_parallel_interleaved_loader`,
    which had ZERO callers before this increment, so its documented layout is
    exercised here rather than trusted.

    At degree 2 over E experts, rank 0 and rank 1 each hold exactly E/2, the two
    subsets are DISJOINT, and their union is all E. Both directions are counted
    because either alone is satisfiable by a defect: equal counts alone allow
    both ranks to hold expert 0, and a union of E alone allows one rank to hold
    everything.

    WHICH EXPERTS A RANK ACTUALLY GOT IS IDENTIFIED BY CONTENT, not by trusting
    the loader's own arithmetic: each stacked row is matched against the
    per-expert reference tensors, so the answer comes from the bytes. The DEGREE-1
    load is the control -- one rank owning every expert -- so a selection that
    silently ignored the degree reads all E on both ranks and reddens.
    """
    directory = tmp_path / "stacked"
    model = _stacked_model()
    mappings = _mappings_for(_stacked_config())
    _write_miniature_checkpoint(directory, mappings, model)
    banks = _bank_entries(mappings)
    name = sorted(banks)[0]
    keys = banks[name]
    layout = bank_layout(keys, param_name=name)
    values = _distinguish_bank_experts(directory, keys)
    print(f"CONJUNCT3_DISTINGUISHED_KEYS={len(values)}")
    slices = _slice_pairs(directory, keys)
    references = [
        downscale_fp8_weight_bytes(_checkpoint_tensor(directory, keys[i]))
        for i in layout.weight_at
    ]

    def experts_in(stack) -> list[int]:
        """Which global expert each stacked row IS, decided by its bytes."""
        found = []
        for row in range(stack.shape[0]):
            matches = [
                e
                for e in range(layout.experts)
                if float(
                    (stack[row].to(torch.float32) - references[e].to(torch.float32))
                    .abs()
                    .max()
                )
                == 0.0
            ]
            assert len(matches) == 1, (
                f"stacked row {row} matches {len(matches)} of the "
                f"{layout.experts} expert references, so this reading cannot say "
                f"which expert it holds; the miniature's experts must be "
                f"distinguishable for conjunct (3) to mean anything"
            )
            found.append(matches[0])
        return found

    split = _stacked_bank_geometry(layout.experts, STACKED_EXPERTS_PER_RANK)
    loader = stacked_expert_bank_loader(keys, param_name=name, owner=split)
    rank0 = experts_in(loader.transform(slices, 0))
    rank1 = experts_in(loader.transform(slices, 1))

    whole = _stacked_bank_geometry(layout.experts, layout.experts)
    degree1 = experts_in(
        stacked_expert_bank_loader(keys, param_name=name, owner=whole).transform(
            slices, 0
        )
    )

    print(f"CONJUNCT3_EP_DEGREE={STACKED_EP_DEGREE}")
    print(f"CONJUNCT3_EXPERTS_TOTAL={layout.experts}")
    print(f"CONJUNCT3_RANK0_EXPERTS={rank0}")
    print(f"CONJUNCT3_RANK1_EXPERTS={rank1}")
    print(f"CONJUNCT3_OVERLAP={sorted(set(rank0) & set(rank1))}")
    print(f"CONJUNCT3_UNION_SIZE={len(set(rank0) | set(rank1))}")
    print(f"CONJUNCT3_CONTROL_DEGREE1_EXPERTS={degree1}")

    assert len(rank0) == STACKED_EXPERTS_PER_RANK, (
        f"rank 0 holds {len(rank0)} experts at degree {STACKED_EP_DEGREE}, not "
        f"{STACKED_EXPERTS_PER_RANK}"
    )
    assert len(rank1) == STACKED_EXPERTS_PER_RANK, (
        f"rank 1 holds {len(rank1)} experts at degree {STACKED_EP_DEGREE}, not "
        f"{STACKED_EXPERTS_PER_RANK}"
    )
    assert set(rank0) & set(rank1) == set(), (
        f"the two ranks share experts {sorted(set(rank0) & set(rank1))}, so the "
        f"same weight was loaded twice and the partition is not a partition"
    )
    assert set(rank0) | set(rank1) == set(range(layout.experts)), (
        f"the two ranks together hold {sorted(set(rank0) | set(rank1))}, not all "
        f"{layout.experts} experts, so an expert reached no rank at all"
    )
    assert degree1 == list(range(layout.experts)), (
        f"the degree-1 control holds {degree1} rather than every expert in "
        f"order, so the selection is not reading the declared geometry"
    )


def test_the_stacked_bank_refusal_is_gone_for_this_case_only(tmp_path) -> None:
    """(4) THE REFUSAL IS GONE FOR THIS CASE AND ONLY THIS CASE.

    Certifies :func:`loader_for_mapped_keys`'s routing and the refusals that
    remain. FOUR readings, each naming what it certifies:

    (i) a WELL-FORMED bank now gets a loader instead of a refusal -- the one
    case this increment retires -- and it classifies
    :data:`MAPPED_KEY_STACKED_BANK`, with ``_placeholder_dtype`` answering fp8
    for that kind. The dtype reading is here because the classifier has TWO
    consumers and a fourth kind the second consumer did not know would type
    every bank placeholder bf16 while its loader delivered fp8.

    (ii) an ODD key count still REFUSES by name -- it cannot be pairs at all.

    (iii) a NON-ALTERNATING entry (a scale where a weight belongs) still REFUSES
    by name, naming the offending position.

    (iv) a MULTI-WEIGHT entry with NO scale key still REFUSES by name, naming
    the parameter and both counts. This is ``B65-N1``: the refusal that used to
    live here was keyed on the SCALE count, so this shape passed to the default
    loader, whose ``len(slices) == 1`` assertion failed loudly but named no
    parameter and fired after registration.
    """
    model = _stacked_model()
    mappings = _mappings_for(_stacked_config())
    banks = _bank_entries(mappings)
    name = sorted(banks)[0]
    keys = banks[name]
    owner = model.get_submodule(name.rsplit(".", 1)[0])

    # ── (i) the case this increment retires ─────────────────────────────────
    kind = classify_mapped_keys(keys)
    loader = loader_for_mapped_keys(keys, param_name=name, owner=owner)
    dtype = model._placeholder_dtype(keys, param_name=name, mappings=mappings)
    print(f"CONJUNCT4_WELL_FORMED_KIND={kind}")
    print(f"CONJUNCT4_WELL_FORMED_GETS_A_LOADER={loader is not None}")
    print(f"CONJUNCT4_PLACEHOLDER_DTYPE={dtype}")
    assert kind == MAPPED_KEY_STACKED_BANK, (
        f"a well-formed bank classifies {kind!r}, not {MAPPED_KEY_STACKED_BANK!r}"
    )
    assert loader is not None and loader.transform is not None, (
        "a well-formed bank did not get a transforming loader, so the refusal "
        "this increment retires is still in force"
    )
    assert _is_fp8_dtype(dtype), (
        f"a bank placeholder is typed {dtype}, not fp8. The classifier's second "
        f"consumer does not know the fourth kind, so every bank load would warn "
        f"on a dtype mismatch against a loader that delivers fp8"
    )

    # ── (ii)-(iv) the shapes that still refuse ──────────────────────────────
    weight, scale = keys[0], keys[1]
    malformed = {
        "an odd key count": (keys[:-1], "ODD count"),
        "a scale where a weight belongs": ([scale] + keys[1:], "do not alternate"),
        "several weights and no scale": ([weight, weight + ".dup"], "0 scale keys"),
    }
    refusals = {}
    for defect, (entry, expected) in malformed.items():
        with pytest.raises(Glm5NextExpertBankNotLoadableError) as refusal:
            loader_for_mapped_keys(entry, param_name=name, owner=owner)
        message = str(refusal.value)
        refusals[defect] = message
        print(f"CONJUNCT4_REFUSED[{defect}]={message[:120]!r}")
        assert name in message, (
            f"the refusal for {defect} does not name the parameter: {message}"
        )
        assert expected in message, (
            f"the refusal for {defect} does not name the defect "
            f"({expected!r} absent): {message}"
        )
        assert str(len(entry)) in message, (
            f"the refusal for {defect} does not name the key count: {message}"
        )
    print(f"CONJUNCT4_REFUSALS_CHECKED={len(refusals)}")
    assert len(refusals) == 3, "one of the malformed shapes did not refuse"


# --------------------------------------------------------------------------- #
# inc-glm53f-095b -- the bank's scale grids reach the module, and the prep
# loop's leaf derivation reads PRESENCE. Three items, selected by ``-k
# bankscale``, one per conjunct.
#
# WHY THE BANK'S OWN SCALE PREP IS NOT HERE. It was designed here and moved to
# ``inc-glm53f-054`` at design entry ``design-20260905-x``, on a reading this
# seat took first: the miniature every load in this file reads is 128 x 128 with
# (1, 1) grids, ``BLOCK_QUANT_SIZE`` is 256, and both ``retile_block_scales``
# and ``scale_grid_shape`` refuse an extent that is not a multiple of it. The
# prep loop's gate is a TYPE test, so a prep on the bank would fire inside
# ``-095``'s own conjunct-1 load -- the one completing load in this file that
# carries a bank -- and break it. Measured in
# ``increments/probe-095b-geometry-r3.out``; ``-054``'s configuration already
# requires ``hidden_size % 256 == 0``, so the prep belongs there.
#
# So these three items read the two halves that DO belong here: the grids
# arriving, and the derivation that will hand them over when ``-054`` adds the
# prep. Every configuration and constant is the landed one, reused rather than
# copied.
# --------------------------------------------------------------------------- #

#: The three projection leaves that have a scale grid, stated INDEPENDENTLY of
#: the code under test because it is the expected answer: a value derived from
#: the same declaration the helper reads could not disagree with it. The items
#: below assert against this AND against the declaration tuple, so a change to
#: ``Glm5NextRoutedExperts`` reddens them rather than passing beside them.
BANKSCALE_LEAVES = ("gate_proj_weight", "up_proj_weight", "down_proj_weight")


def _scale_grid_attribute(leaf: str) -> str:
    """The attribute a weight leaf's scale grid arrives under -- ASKED, not retyped.

    LEG D ROUND 1 FAILED HERE, and this helper is the repair. This section first
    wrote the name as ``f"{leaf}_{FP8_SCALE_SUFFIX}"`` in four places, which
    doubles the weight suffix: the rule STRIPS ``_weight`` before it appends, so
    ``gate_proj_weight`` names ``gate_proj_weight_scale_inv`` and not
    ``gate_proj_weight_weight_scale_inv``. Items (1) and (2) both went red, item
    (1) reading zero grids present and item (2)'s shared-expert control attaching
    its grids where nothing would look for them.

    The rule has ONE definition,
    :meth:`Glm5NextForConditionalGeneration._sibling_scale_grid_name`, and this
    asks that definition. The source helper's own docstring says a second copy of
    the naming rule is the drift this file's one-classifier convention exists to
    prevent; round 1 is what that sentence was warning about.

    ASKING DOES NOT MAKE THE ITEMS CIRCULAR. The name is not what they measure.
    They measure that a grid ARRIVES, with the right leading axis, holding expert
    ``e``'s own bytes in row ``[e]``, at the one name both source sites use -- the
    ``setattr`` in the reader and the ``hasattr`` in ``_scale_prep_leaves``. If
    those two ever disagreed, the arrival reading would go red no matter which of
    them this helper agreed with. Item (1) also keeps the wrong name of round 1 as
    a printed control, so the fact that the rule strips is itself on the record.
    """
    return Glm5NextForConditionalGeneration._sibling_scale_grid_name(leaf)


def _alter_one_scale_slice(directory: Path, key: str) -> float:
    """Add 1.0 to every element of ONE scale key, in the written file.

    The D1.5 control for conjunct (1). Every other tensor is carried across
    unchanged, read back from the file rather than rebuilt, so the only thing
    that differs between the two loads is this one expert's grid -- which is what
    makes "exactly one row moved" a measurement of the row-to-expert mapping
    rather than of the write.
    """
    path = directory / MINI_CHECKPOINT_FILE
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as opened:
        for name in opened.keys():
            tensors[name] = opened.get_tensor(name)
    tensors[key] = tensors[key].to(torch.float32) + 1.0
    save_file(tensors, str(path))
    return float(tensors[key].flatten()[0])


def _bank_owner_paths(banks: dict[str, list[str]]) -> list[str]:
    """The module paths that own the bank entries, deduplicated and sorted."""
    return sorted({name.rsplit(".", 1)[0] for name in banks})


def test_bankscale_grids_arrive_on_the_bank_as_plain_attributes(
    tmp_path, single_rank_process_group
) -> None:
    """(1) THE BANK'S SCALE GRIDS ARRIVE, as plain attributes, row per expert.

    Certifies the bank branch of
    :meth:`Glm5NextForConditionalGeneration._load_out_of_band_scales` as
    ``load_weights`` reaches it. FOUR readings and each has its own control.

    (i) After a completing load of ``-095``'s miniature bank, every bank module
    carries ``gate_proj_weight_scale_inv``, ``up_proj_weight_scale_inv`` and
    ``down_proj_weight_scale_inv``, and NONE of those names is in
    ``named_parameters()`` -- asserted by name, because that is the whole design
    ground for the plain-attribute convention: a registered parameter would add a
    name the weight map does not carry.

    (ii) Each grid's leading axis is the module's OWN ``num_local_experts``, read
    off the module rather than from a constant.

    (iii) Row ``[e]`` is bit-identical to expert ``e``'s checkpoint grid through
    the SAME compensation the loader applies, :func:`compensate_block_scales`.
    ``torch.equal``, not a tolerance: a stack is a permutation of its inputs, so
    any difference at all is an indexing defect. The landed writer gives every
    expert identical bytes, so :func:`_distinguish_bank_experts` runs first --
    otherwise every row would match every reference and the reading would certify
    nothing.

    (iv) THE READER'S RETURN RISES BY EXACTLY THREE PER BANK, read from the
    return value of a direct call. Its control MOVES: the same call with the bank
    entries removed from the map reads only the lone grids, and the difference is
    the bank-entry count.

    THE D1.5 CONTROL FOR (iii) is a second checkpoint identical to the first
    except that ONE expert's scale slice has 1.0 added. Exactly one row of the
    loaded grid differs, and it is that expert's row. A reader that stacked the
    same grid E times, or ignored the file, would move zero rows or all of them.
    """
    device = torch.device("cpu")
    directory = tmp_path / "bankscale"
    model = _stacked_model()
    mappings = _mappings_for(_stacked_config())
    _write_miniature_checkpoint(directory, mappings, model)
    banks = _bank_entries(mappings)
    assert banks, (
        "this configuration produced no expert-bank entry, so conjunct (1) would "
        "certify nothing: the branch it exercises could not be reached"
    )
    owner_paths = _bank_owner_paths(banks)
    subject = sorted(banks)[0]
    subject_keys = banks[subject]
    layout = bank_layout(subject_keys, param_name=subject)
    _distinguish_bank_experts(directory, subject_keys)

    print(f"BANKSCALE1_BANK_ENTRIES={len(banks)}")
    print(f"BANKSCALE1_BANK_MODULES={len(owner_paths)}")
    print(f"BANKSCALE1_SUBJECT={subject} EXPERTS={layout.experts}")

    model.load_weights(str(directory), device, None)

    # ── (i) the three attributes, and none of them a parameter ───────────────
    # The names are ASKED for. Round 1 retyped them and doubled the weight
    # suffix, so the wrong form is printed beside the right one and asserted to
    # differ: that keeps "the rule strips" a reading rather than a memory.
    resolved = [_scale_grid_attribute(leaf) for leaf in BANKSCALE_LEAVES]
    retyped = [f"{leaf}_{FP8_SCALE_SUFFIX}" for leaf in BANKSCALE_LEAVES]
    print(f"BANKSCALE1_ATTRIBUTES={'|'.join(resolved)}")
    print(f"BANKSCALE1_CONTROL_RETYPED_ATTRIBUTES={'|'.join(retyped)}")
    assert all(a != b for a, b in zip(resolved, retyped)), (
        f"the resolved attribute names {resolved} equal the retyped ones "
        f"{retyped}, so the naming rule no longer strips the weight suffix and "
        f"round 1's defect would now pass unnoticed"
    )

    parameter_names = set(dict(model.named_parameters()))
    present: dict[str, tuple[int, ...]] = {}
    missing: list[str] = []
    as_parameters: list[str] = []
    axes: dict[str, int] = {}
    for path in owner_paths:
        module = model.get_submodule(path)
        for leaf in BANKSCALE_LEAVES:
            attribute = _scale_grid_attribute(leaf)
            dotted = f"{path}.{attribute}"
            grid = getattr(module, attribute, None)
            if grid is None:
                missing.append(dotted)
                continue
            present[dotted] = tuple(grid.shape)
            axes[dotted] = int(grid.shape[0]) - int(module.num_local_experts)
            if dotted in parameter_names:
                as_parameters.append(dotted)

    print(f"BANKSCALE1_GRIDS_PRESENT={len(present)}")
    print(f"BANKSCALE1_GRIDS_MISSING={missing}")
    print(f"BANKSCALE1_GRIDS_IN_NAMED_PARAMETERS={as_parameters}")
    print(f"BANKSCALE1_ONE_GRID={sorted(present)[0]} shape={present[sorted(present)[0]]}")
    print(f"BANKSCALE1_LEADING_AXIS_MINUS_LOCAL_EXPERTS={sorted(set(axes.values()))}")

    assert missing == [], (
        f"{len(missing)} bank scale grids never arrived, e.g. {missing[:3]}; the "
        f"reader's bank branch did not run or stored them under other names"
    )
    assert len(present) == 3 * len(owner_paths), (
        f"expected three grids on each of {len(owner_paths)} bank modules, found "
        f"{len(present)}"
    )
    assert as_parameters == [], (
        f"{len(as_parameters)} scale grids were registered as parameters, e.g. "
        f"{as_parameters[:3]}. Registering one adds a name to named_parameters() "
        f"that the weight map does not carry, which is the map widening the "
        f"plain-attribute convention exists to avoid"
    )
    # ── (ii) the leading axis is the module's own local expert count ──────────
    assert set(axes.values()) == {0}, (
        f"a grid's leading axis is not its module's num_local_experts: read "
        f"differences {sorted(set(axes.values()))}, expected [0]"
    )

    # ── (iii) row [e] is expert e's own grid through the same compensation ────
    owner = model.get_submodule(subject.rsplit(".", 1)[0])
    attribute = _scale_grid_attribute(subject.rsplit(".", 1)[1])
    grid = getattr(owner, attribute)
    references = [
        compensate_block_scales(
            _checkpoint_tensor(directory, subject_keys[position])
        ).scale_inv
        for position in layout.scale_at
    ]
    unequal = [
        expert
        for expert in range(layout.experts)
        if not torch.equal(grid[expert], references[expert])
    ]
    print(f"BANKSCALE1_ROWS_COMPARED={layout.experts}")
    print(f"BANKSCALE1_ROWS_NOT_BIT_IDENTICAL={unequal}")
    assert unequal == [], (
        f"rows {unequal} are not bit-identical to their own expert's grid "
        f"through compensate_block_scales. A stack is a permutation of its "
        f"inputs, so this is an indexing defect and not a tolerance question"
    )

    # ── (iv) the return rises by exactly three per bank, with its control ────
    without_banks = {
        name: keys for name, keys in mappings.items() if name not in banks
    }
    full_read = _stacked_model()._load_out_of_band_scales(
        SafetensorsCheckpoint(str(directory)), mappings, device
    )
    lone_read = _stacked_model()._load_out_of_band_scales(
        SafetensorsCheckpoint(str(directory)), without_banks, device
    )
    print(f"BANKSCALE1_READ_WITH_BANKS={full_read}")
    print(f"BANKSCALE1_READ_CONTROL_WITHOUT_BANK_ENTRIES={lone_read}")
    print(f"BANKSCALE1_RISE={full_read - lone_read}")
    assert full_read - lone_read == len(banks), (
        f"the reader's return rose by {full_read - lone_read} with the bank "
        f"entries in the map, not by the {len(banks)} bank entries it read"
    )
    assert len(banks) == 3 * len(owner_paths), (
        f"{len(banks)} bank entries over {len(owner_paths)} bank modules is not "
        f"three per bank, so the rise above is not the per-bank rise"
    )

    # ── the D1.5 control: alter ONE scale slice, move EXACTLY one row ─────────
    control_directory = tmp_path / "bankscale-control"
    control_model = _stacked_model()
    _write_miniature_checkpoint(control_directory, mappings, control_model)
    _distinguish_bank_experts(control_directory, subject_keys)
    altered_expert = layout.experts - 1
    altered_key = subject_keys[layout.scale_at[altered_expert]]
    _alter_one_scale_slice(control_directory, altered_key)
    control_model.load_weights(str(control_directory), device, None)
    control_grid = getattr(
        control_model.get_submodule(subject.rsplit(".", 1)[0]), attribute
    )
    moved = [
        expert
        for expert in range(layout.experts)
        if not torch.equal(grid[expert], control_grid[expert])
    ]
    print(f"BANKSCALE1_CONTROL_ALTERED_KEY={altered_key}")
    print(f"BANKSCALE1_CONTROL_ROWS_MOVED={moved}")
    assert moved == [altered_expert], (
        f"altering expert {altered_expert}'s scale slice moved rows {moved}. "
        f"Zero rows means the reader is not reading this file; every row means "
        f"it is not reading per expert"
    )


def test_bankscale_leaf_derivation_reads_presence_not_declaration(
    tmp_path, single_rank_process_group
) -> None:
    """(2) THE DERIVATION READS PRESENCE, and the bank is where that shows.

    Certifies :func:`_scale_prep_leaves`, the helper ``inc-glm53f-095b`` factored
    out of ``_run_load_time_preps``. TWO readings and TWO controls, all four in
    this one run.

    (i) On the loaded bank the helper returns exactly the three projection leaves
    and NOT ``router_weight`` -- which the bank does declare, which does end in
    the weight suffix, and which has no scale grid anywhere in this tree.

    (ii) On a shared-expert module with its three grids attached it returns that
    module's three leaves, so the presence test did not narrow the landed caller.

    THE FIRST CONTROL MOVES ON THE SAME MODULE: the UNFACTORED derivation -- the
    declaration tuple filtered by the weight suffix and nothing else, which is
    what the loop read before this increment -- returns four leaves for the same
    bank, including ``router_weight``. So (i) is a measurement of the presence
    conjunct and not of the tuple's contents.

    THE SECOND CONTROL MOVES ON THE SAME CLASS: a shared-expert module with NO
    grids attached returns zero leaves, so the helper is reading the attributes
    and not the class.

    BOTH ALTERNATIVE DERIVATIONS ARE ALREADY ON RECORD, measured before this item
    was authored: the unfactored four-leaf / eight-operand reading in
    ``increments/probe-095b-readfirst-r3.out``, and entry v's
    declaration-membership variant -- zero leaves for the shared expert and the
    six-argument ``TypeError`` -- in ``increments/probe-095b-conjunct.out``.
    """
    device = torch.device("cpu")
    directory = tmp_path / "bankscale-leaves"
    model = _stacked_model()
    mappings = _mappings_for(_stacked_config())
    _write_miniature_checkpoint(directory, mappings, model)
    banks = _bank_entries(mappings)
    assert banks, "no bank entry, so this item has no subject"
    model.load_weights(str(directory), device, None)

    bank = model.get_submodule(_bank_owner_paths(banks)[0])
    declared = tuple(getattr(bank, "declared_param_names", ()))
    leaves = _scale_prep_leaves(bank)

    # The first control: what the loop derived BEFORE this increment factored it.
    unfactored = [
        leaf for leaf in declared if leaf.endswith(_WEIGHT_LEAF_SUFFIX)
    ]

    print(f"BANKSCALE2_BANK_DECLARES={'|'.join(declared)}")
    print(f"BANKSCALE2_LEAVES={'|'.join(leaves)}")
    print(f"BANKSCALE2_CONTROL_UNFACTORED_LEAVES={'|'.join(unfactored)}")
    print(f"BANKSCALE2_ROUTER_IN_LEAVES={'router_weight' in leaves}")
    print(f"BANKSCALE2_ROUTER_IN_UNFACTORED={'router_weight' in unfactored}")
    print(
        "BANKSCALE2_ROUTER_SCALE_ON_THE_MODULE="
        f"{hasattr(bank, _scale_grid_attribute('router_weight'))}"
    )

    assert leaves == list(BANKSCALE_LEAVES), (
        f"the helper returned {leaves} for a loaded bank, not the three "
        f"projection leaves {list(BANKSCALE_LEAVES)} whose grids arrived"
    )
    assert "router_weight" in declared, (
        "the bank no longer declares router_weight, so this item's control has "
        "lost its subject and the reading below certifies nothing"
    )
    assert "router_weight" in unfactored and len(unfactored) == len(leaves) + 1, (
        f"the unfactored derivation returned {unfactored}, which does not differ "
        f"from the factored one by router_weight; the control did not move, so "
        f"the reading above would hold for a helper that filtered nothing"
    )

    # ── (ii) the landed caller's own module, with and without its grids ───────
    shared = Glm5NextSharedExperts(_dense_config().text_config)
    bare = _scale_prep_leaves(shared)
    for leaf in BANKSCALE_LEAVES:
        setattr(shared, _scale_grid_attribute(leaf), torch.zeros(1, 1))
    attached = _scale_prep_leaves(shared)

    print(f"BANKSCALE2_SHARED_DECLARES={'|'.join(shared.declared_param_names)}")
    print(f"BANKSCALE2_SHARED_CONTROL_NO_GRIDS={bare}")
    print(f"BANKSCALE2_SHARED_WITH_GRIDS={'|'.join(attached)}")

    assert attached == list(BANKSCALE_LEAVES), (
        f"the helper returned {attached} for a shared expert with its three grids "
        f"attached, so the presence test narrowed the landed caller"
    )
    assert bare == [], (
        f"the helper returned {bare} for a shared expert with NO grids attached, "
        f"so it is reading the class rather than the attributes and the reading "
        f"above would hold whether the grids arrived or not"
    )


def test_bankscale_prep_loop_visits_exactly_what_it_did_before(
    tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """(3) THE LOOP'S BEHAVIOUR IS UNCHANGED for every landed module.

    Certifies ``_run_load_time_preps``'s visit, read from its RETURN VALUE. The
    bank now carries three scale grids, and the question this item answers is
    whether that changed who the loop calls. It did not, and it could not: the
    loop's gate is a TYPE test and ``Glm5NextRoutedExperts`` defines no
    ``prepare_scale_operands`` -- ``inc-glm53f-054``'s work, moved there at design
    entry ``design-20260905-x``.

    (i) On a completing load of the bank configuration, the returned pair equals
    the module counts the loop's own two gates select, derived from the tree
    rather than written down. The scale count is zero because this configuration
    builds no shared-expert module, and no bank is visited.

    THE CONTROL MOVES, which is what makes that zero a reading. A stub
    ``prepare_scale_operands`` is planted on the BANK'S TYPE -- the thing the gate
    actually tests -- and the scale count rises by the number of bank modules. The
    stub is reached through the full landed path: the device pre-flight passes,
    ``_scale_prep_leaves`` hands over three leaves, and the operands the loop
    collects include the three grids conjunct (1) read. So the zero above means
    "no bank declares a prep", not "the loop cannot reach one".
    """
    device = torch.device("cpu")
    directory = tmp_path / "bankscale-loop"
    model = _stacked_model()
    mappings = _mappings_for(_stacked_config())
    _write_miniature_checkpoint(directory, mappings, model)
    banks = _bank_entries(mappings)
    owner_paths = _bank_owner_paths(banks)
    assert owner_paths, "no bank module, so this item has no subject"
    model.load_weights(str(directory), device, None)

    expected_projection = sum(
        1
        for _, module in model.named_modules()
        if hasattr(type(module), "prepare_projection_weights")
    )
    expected_scale = sum(
        1
        for _, module in model.named_modules()
        if hasattr(type(module), "prepare_scale_operands")
    )
    projection_calls, scale_calls = model._run_load_time_preps(device)

    print(f"BANKSCALE3_MODULES_WITH_A_PROJECTION_PREP={expected_projection}")
    print(f"BANKSCALE3_MODULES_WITH_A_SCALE_PREP={expected_scale}")
    print(f"BANKSCALE3_RETURNED_PAIR=({projection_calls}, {scale_calls})")
    print(f"BANKSCALE3_BANK_MODULES={len(owner_paths)}")

    assert (projection_calls, scale_calls) == (expected_projection, expected_scale), (
        f"the loop returned ({projection_calls}, {scale_calls}) where its own two "
        f"gates select ({expected_projection}, {expected_scale}) modules, so it "
        f"visited something other than what it tests for"
    )
    assert scale_calls == 0, (
        f"the loop ran {scale_calls} scale preps on a configuration with no "
        f"shared-expert module, so it is now visiting a bank -- which is "
        f"inc-glm53f-054's work and breaks the 128-extent load this file reads"
    )

    # ── the same reading on the DENSE configuration ──────────────────────────
    # Six of the seven completing loads in this file build ``_dense_model()``, and
    # the STOP condition of this increment's block is that none of their readings
    # moves. The bank branch cannot reach this tree -- an all-dense config has no
    # sparse layer and so no bank entry -- and the pair is read here to say so
    # from the return value rather than from that argument.
    dense_directory = tmp_path / "bankscale-loop-dense"
    dense = _dense_model()
    dense_mappings = _mappings_for(_dense_config())
    _write_miniature_checkpoint(dense_directory, dense_mappings, dense)
    dense.load_weights(str(dense_directory), device, None)
    dense_banks = _bank_entries(dense_mappings)
    dense_expected = (
        sum(
            1
            for _, module in dense.named_modules()
            if hasattr(type(module), "prepare_projection_weights")
        ),
        sum(
            1
            for _, module in dense.named_modules()
            if hasattr(type(module), "prepare_scale_operands")
        ),
    )
    dense_pair = dense._run_load_time_preps(device)
    print(f"BANKSCALE3_DENSE_BANK_ENTRIES={len(dense_banks)}")
    print(f"BANKSCALE3_DENSE_RETURNED_PAIR={dense_pair}")
    print(f"BANKSCALE3_DENSE_GATES_SELECT={dense_expected}")
    assert dense_banks == {}, (
        f"the all-dense configuration produced {len(dense_banks)} bank entries, "
        f"so it is no longer the bank-free tree the landed items read"
    )
    assert dense_pair == dense_expected, (
        f"the loop returned {dense_pair} on the dense tree where its own gates "
        f"select {dense_expected}"
    )

    # ── the control, and it MOVES: plant a prep on the BANK'S TYPE ────────────
    handed: dict[str, list[str]] = {}

    def stub(self, **operands) -> None:
        handed[type(self).__name__] = sorted(operands)

    bank_type = type(model.get_submodule(owner_paths[0]))
    monkeypatch.setattr(bank_type, "prepare_scale_operands", stub, raising=False)
    planted_projection, planted_scale = model._run_load_time_preps(device)

    print(f"BANKSCALE3_CONTROL_PLANTED_ON={bank_type.__name__}")
    print(f"BANKSCALE3_CONTROL_RETURNED_PAIR=({planted_projection}, {planted_scale})")
    print(f"BANKSCALE3_CONTROL_OPERANDS_HANDED_OVER={handed}")

    assert planted_scale == len(owner_paths), (
        f"with a prep planted on {bank_type.__name__} the loop ran "
        f"{planted_scale} scale preps over {len(owner_paths)} bank modules, so "
        f"the zero above is not a reading of who declares a prep"
    )
    assert planted_projection == projection_calls, (
        f"planting a SCALE prep moved the projection count from "
        f"{projection_calls} to {planted_projection}"
    )
    expected_operands = sorted(
        [leaf for leaf in BANKSCALE_LEAVES]
        + [f"{leaf[: -len(_WEIGHT_LEAF_SUFFIX)]}_scale" for leaf in BANKSCALE_LEAVES]
    )
    assert handed.get(bank_type.__name__) == expected_operands, (
        f"the planted prep was handed {handed.get(bank_type.__name__)}, not the "
        f"six operands {expected_operands} the three arrived grids imply"
    )
