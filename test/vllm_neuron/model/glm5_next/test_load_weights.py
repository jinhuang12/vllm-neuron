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
from typing import NamedTuple

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import vllm_neuron
import vllm_neuron.model.glm5_next.factory  # noqa: F401 -- bound as _FACTORY below
import vllm_neuron.parallel.neuron_parallel_state  # noqa: F401 -- bound as _NPS below
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
    dequantise_blockwise,
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

    THE PARAMETER NAME NOW COMES FROM THE MODULE TOO, and that is this function's
    ``inc-glm53f-051`` repair rather than a refinement. Appending ``_weight`` to a
    site name WAS the second naming convention the paragraph above warns about,
    and it drifted the moment a projection site arrived whose parameter is a bare
    checkpoint tensor: ``Glm5NextDSAIndexer``'s ``index_kpool_compress_gate`` has
    no ``.weight`` leaf, so ``<path>.index_kpool_compress_gate_weight`` is absent
    from the map, this loop skipped the site, and the writer below gave the gate
    ``MINI_PLAIN_SHAPE``. The prep then refused the ``(4,)`` and twelve items in
    this file went red. A module that publishes ``PROJECTION_PARAMETERS`` is asked
    for the name; one that does not keeps the suffix rule unchanged, which is why
    the MLA family's own four overrides do not move.

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


def _write_miniature_checkpoint(
    directory: Path,
    mappings: dict[str, str | list[str]],
    model: Glm5NextForConditionalGeneration,
    extra_overrides: dict[str, torch.Tensor] | None = None,
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

    ``extra_overrides`` SUPPLIES WHOLE TENSORS, not shapes, and it is
    ``inc-glm53f-094``'s one change here. Every tensor this writer builds itself is
    CONSTANT (``torch.ones``, ``torch.full``), which is enough for a reading that
    counts tensors and wrong for a reading that asks WHICH ROWS a rank got: against
    a constant, any slice passes for any other. A caller that measures indexing
    therefore passes its own values. Landed callers pass nothing and are unchanged.
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
# So these items set ``n_shared_experts=0``, which ``model_fp8.py`` reads as "build
# no shared-expert module", and the routed load completes on the bank's own path.
#
# ``inc-glm53f-054a`` CHANGED THE SECOND HALF OF THIS PARAGRAPH, which used to read
# "The bank does not need that prep. ``Glm5NextRoutedExperts`` defines NEITHER
# load-time prep ... so ``_run_load_time_preps``'s ``hasattr(type(module), ...)``
# gate never visits a bank at all." Item (i) gave the bank its own
# ``prepare_scale_operands``, so that gate now DOES visit every bank, and the bank's
# prep retiles -- which refuses an extent that is not a whole 256 block. The
# sentence above about the shared expert is unchanged and still the reason
# ``n_shared_experts`` is 0 here; what moved is the BANK's own extents, which
# :func:`_blocked_bank_overrides` writes at 512 by 256 for exactly these items.
# ``MINI_WEIGHT_SHAPE`` stays ``(128, 128)`` and stays every other item's shape.
# The landed constants are otherwise REUSED rather than copied, so a change to the
# miniature moves these items with the other nine.
# --------------------------------------------------------------------------- #

#: The expert-parallel degree conjunct 3 splits the bank across, and the count
#: it expects on each rank. Two ranks over ``MINI_ROUTED_EXPERTS`` experts.
STACKED_EP_DEGREE = 2
STACKED_EXPERTS_PER_RANK = MINI_ROUTED_EXPERTS // STACKED_EP_DEGREE


#: The bank's widths for the items whose load now RUNS the bank's scale prep --
#: ``inc-glm53f-054a``'s migration of the five items that reached it.
#:
#: A NEW NAME, NOT A REBINDING, on the precedent :data:`DEFERRED_NARROW` states
#: for the same situation. :data:`MINI_WEIGHT_SHAPE` stays ``(128, 128)`` and stays
#: every other item's shape; the increment plan's hand-off bullet says in words
#: that it is not widened, and it is not. What changed is that the bank now
#: declares ``prepare_scale_operands``, so a load that carries a bank reaches a
#: retile that refuses any extent which is not a whole ``256`` block
#: (``blockwise_fp8_retile.py:232-245``) -- and ``(128, 128)`` is not one. Only the
#: items that reach the prep are given extents that are.
#:
#: WHY 512 BY 256. 256 is the smallest width the consumer admits at all (DECISIONS
#: §83 ruling 1, the ground :data:`DEFERRED_NARROW` records). The out extent is
#: DOUBLE that on purpose: at 256 by 256 every grid is a single block, and a
#: single-block grid cannot tell a coarsening apart from no coarsening at all, so
#: the axis the retile actually folds would be untested.
BLOCKED_BANK_OUT = 512
BLOCKED_BANK_IN = 256

#: The bank's three leaves with the dim each is written along, in the frame the
#: bank registers them: gate and up as ``[I, H]`` and down as ``[H, I]``. The same
#: dims :data:`DEFERRED_FAMILIES` states for the same three, so the two fixtures
#: cannot disagree about the bank's orientation.
BLOCKED_BANK_FAMILIES: dict[tuple[str, str], tuple[int, int]] = {
    ("Glm5NextRoutedExperts", "gate_proj_weight"): (0, BLOCKED_BANK_OUT),
    ("Glm5NextRoutedExperts", "up_proj_weight"): (0, BLOCKED_BANK_OUT),
    ("Glm5NextRoutedExperts", "down_proj_weight"): (1, BLOCKED_BANK_OUT),
}


def _blocked_bank_overrides(
    model: Glm5NextForConditionalGeneration,
    mappings: dict[str, str | list[str]],
) -> dict[str, torch.Tensor]:
    """The bank's checkpoint tensors at 256-blocked extents, VALUES unchanged.

    Only the extents move. Every tensor is built the way
    :func:`_write_miniature_checkpoint` builds it -- ``torch.ones`` squeezed into
    fp8 for a weight, ``torch.full(0.5)`` for a grid -- so no item that reads a
    VALUE can move, and the items that read an extent read it from the checkpoint's
    own slices through :func:`_implied_numels` rather than from a constant.

    Nothing here spells a checkpoint key: the keys come from the map, the shapes
    from :data:`BLOCKED_BANK_FAMILIES`, and a grid's shape from
    ``block_grid_shape`` -- the same closed form the loader divides by. That is
    :func:`_shard_key_overrides`'s discipline, followed here for the same reason.

    ALL SCALES EQUAL IS THE POINT, not laziness. Every ratio inside a ``256``
    block is then exactly 1, so the retile's rescale is bit-exact and its two
    losslessness counters read zero. A fixture with unequal scales would make
    those counters report the fixture rather than the layout.
    """
    overrides: dict[str, torch.Tensor] = {}
    for path, module in model.named_modules():
        cls = type(module).__name__
        for (family, leaf), (shard_dim, full) in BLOCKED_BANK_FAMILIES.items():
            if cls != family:
                continue
            param = f"{path}.{leaf}"
            if param not in mappings:
                continue
            keys = _keys_of(mappings, param)
            scales = scale_keys(keys)
            weights = [key for key in keys if key not in scales]
            shape = (
                (full, BLOCKED_BANK_IN)
                if shard_dim == 0
                else (BLOCKED_BANK_IN, full)
            )
            grid_shape = block_grid_shape(shape, DEFAULT_WEIGHT_BLOCK_SIZE)
            for key in weights:
                overrides[key] = torch.ones(
                    shape, dtype=torch.bfloat16
                ).to(torch.float8_e4m3fn)
            for key in scales:
                overrides[key] = torch.full(grid_shape, 0.5, dtype=torch.float32)
    return overrides


def _stacked_checkpoint(
    directory: Path,
    mappings: dict[str, str | list[str]],
    model: Glm5NextForConditionalGeneration,
) -> int:
    """:func:`_write_miniature_checkpoint` with the bank at 256-blocked extents.

    ONE call site for the migration, so the five items that reach the bank's scale
    prep cannot drift apart in the widths they load. Everything except the bank is
    written exactly as the landed writer writes it.
    """
    return _write_miniature_checkpoint(
        directory,
        mappings,
        model,
        extra_overrides=_blocked_bank_overrides(model, mappings),
    )


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
    written = _stacked_checkpoint(directory, mappings, model)
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
    _stacked_checkpoint(directory, mappings, model)
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
    _stacked_checkpoint(control_directory, mappings, control_model)
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
    _stacked_checkpoint(directory, mappings, model)
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

    Certifies ``_run_load_time_preps``'s visit, read from its RETURN VALUE.

    ``inc-glm53f-054a`` MOVED THIS ITEM'S SCALE COUNT, and the increment plan said
    it would: "the prep's arrival makes ``_run_load_time_preps`` visit the bank
    (its scale-call count rises by the bank count) -- read it here". Before item
    (i) the bank declared no ``prepare_scale_operands`` and the count was zero.
    It declares one now, the loop's gate is a type test, so every bank module is
    visited and the count is the number of bank modules. The fixture moved with
    it: this item's bank is written at 256-blocked extents
    (:func:`_blocked_bank_overrides`), because the prep it now reaches retiles and
    the retile refuses anything narrower. ``MINI_WEIGHT_SHAPE`` did not move.

    (i) On a completing load of the bank configuration, the returned pair equals
    the module counts the loop's own two gates select, derived from the tree
    rather than written down, and the scale half of that pair equals the bank
    count. Both sides come from the tree, so the agreement is a reading of the
    loop and not of a number kept here.

    THE FALSIFIER IS THE DENSE ARM, and this is where it changed. The planted stub
    below used to be what made the zero a reading; a zero that is now a bank count
    needs a case with no bank instead, which the all-dense tree is -- it declares
    no bank entry and its scale count is still zero. So the two arms bracket the
    reading: bank present, count rises; bank absent, count zero.

    THE STUB ARM STILL EARNS ITS PLACE, for a different reading. It captures the
    SIX OPERANDS the loop hands the bank's prep -- three weights and the three
    grids conjunct (1) read -- which the real prep consumes and never reports. The
    stub is reached through the full landed path, so the device pre-flight passing
    and ``_scale_prep_leaves`` yielding three leaves are part of what it shows.
    """
    device = torch.device("cpu")
    directory = tmp_path / "bankscale-loop"
    model = _stacked_model()
    mappings = _mappings_for(_stacked_config())
    _stacked_checkpoint(directory, mappings, model)
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
    # ``inc-glm53f-054a``. THIS COUNT MOVED, from 0 to the number of bank modules,
    # and the increment plan's hand-off bullet says so in advance: "the prep's
    # arrival makes ``_run_load_time_preps`` visit the bank (its scale-call count
    # rises by the bank count) -- read it here". The bank declares
    # ``prepare_scale_operands`` as of item (i), and the loop's gate is a type test,
    # so the visit is not optional. The number is read off the tree's own bank
    # modules rather than typed, so it follows the fixture instead of pinning it.
    assert scale_calls == len(owner_paths), (
        f"the loop ran {scale_calls} scale preps over {len(owner_paths)} bank "
        f"modules on a configuration with no shared-expert module. Every scale "
        f"prep on this tree is a bank's, so the two must agree: fewer means a "
        f"bank was skipped, more means something else declared a prep"
    )

    # ── the same reading on the DENSE configuration ──────────────────────────
    # Most of the completing loads in this file build ``_dense_model()`` -- six of
    # the seven that existed before ``inc-glm53f-054a`` item (iii) added one -- and
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

    # ── the stub arm: read the SIX OPERANDS the loop hands the bank's prep ────
    # ``inc-glm53f-054a`` re-purposed this arm rather than deleting it. It was the
    # control that made a zero a reading; the zero is a bank count now and the
    # all-dense arm above is what brackets it. What the stub still shows, and
    # nothing else does, is WHICH operands the loop collects and hands over -- the
    # real prep consumes them and reports nothing about them.
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
        f"{planted_scale} scale preps over {len(owner_paths)} bank modules; the "
        f"stub stands in for the real prep on the same type, so the count it "
        f"produces must be the same one the real prep produced above"
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


# --------------------------------------------------------------------------- #
# inc-glm53f-094 -- the tensor-parallel WEIGHT SHARD geometry.
#
# FOUR counted items, one per conjunct, and no ``parametrize`` decorator (D1.2).
# The file header at ``:4`` still says THIRTEEN: it stopped being updated at
# ``inc-glm53f-095b``, which added three, and this section adds four, so the file
# holds TWENTY. Correcting that header is a rider for the owner of the next
# section here, not this increment's surface.
#
# EVERY READING BELOW COMES FROM A REAL LOAD OF A REAL MINIATURE CHECKPOINT at a
# synthetic world size, because that is what conjunct (1) asks for: "ranks 0 and
# 1 each load the real miniature checkpoint". Calling the loader transforms
# directly would be cheaper and would leave the attachment site -- the component
# this increment adds (D1.4) -- never executed.
#
# THE GEOMETRY IS WRITTEN OUT AGAIN HERE, ON PURPOSE. ``SHARD_FAMILIES`` restates
# the ratified shard table (``increments/shard-table-094.md``, sha256
# ``3b5bf411cd63e76ca377c9513ec7ff0a5e93fd28f3d11033af32b7de1954ef11``) as this
# file's own independent statement of which family shards on which dim and how
# wide it is. It is NOT read from ``_SHARD_GEOMETRY``, so the code under test and
# this table CAN disagree -- and if they do, these items go red. A test that
# derived its expectation from the thing it measures would pass whatever either
# one said.
# --------------------------------------------------------------------------- #

_MODEL_FP8 = vllm_neuron.model.glm5_next.model_fp8
_WL_FP8 = vllm_neuron.model.glm5_next.weight_loaders_fp8

# ``inc-glm53f-101``. Two more modules reached by NAME rather than by a re-import,
# because the code under test imports them FUNCTION-LOCALLY -- the expert-parallel
# getters inside ``weight_loaders_fp8`` and ``_resolve_ep_degree`` inside
# ``Glm5NextRoutedExperts.__init__``. A function-local import resolves the attribute
# on the module object at call time, so patching the module here is what the loader
# actually reads.
_NPS = vllm_neuron.parallel.neuron_parallel_state
_FACTORY = vllm_neuron.model.glm5_next.factory

#: The shard fixture's linear-attention widths: the config's own four keys
#: (``config.py:165-172``) with two values shrunk. EIGHT heads of THIRTY-TWO make
#: a full head width of 256 and a per-rank width of 128 at world size 2 --
#: exactly one ``DEFAULT_WEIGHT_BLOCK_SIZE`` row block, so the shard boundary
#: falls ON a block boundary and conjunct (3) has an aligned case to measure. The
#: config's own 64 x 128 would put 8192 rows in the checkpoint.
SHARD_LINEAR_ATTN = {
    "num_heads": 8,
    "head_dim": 32,
    "short_conv_kernel_size": 4,
    "gate_lower_bound": -5.0,
}

#: The dense MLP's intermediate width, 512 for the same reason it used to be 256:
#: 256 per rank is one whole CONSUMER block. Shrunk for size, not for correctness.
#:
#: IT MOVED AT ``inc-glm53f-101`` (plan revision 232, DECISIONS section 80(a)), and
#: the reason is a boundary that got wider rather than a convenience. The number
#: that has to divide is no longer the checkpoint's 128-row tile but the consumer's
#: 256-row block: ``blockwise_fp8_mm.scale_grid_shape`` refuses any weight extent
#: that is not a whole number of them. At 256 the per-rank shard was 128, half a
#: consumer block, so the padded loader rounded the width to 512 and both conjunct
#: (1) and conjunct (2) read a shape no rank was meant to hold. At 512 the pad is a
#: no-op at world size 2 and a REAL pad at world size 4, which item (3) reads.
SHARD_INTERMEDIATE = 512

SHARD_WORLD = 2

#: The extent no family shards. Deliberately NOT 128 and not equal to any sharded
#: extent, so a loader that sliced the wrong dimension could not return a shape
#: that passes for the right one.
SHARD_NARROW = 8

_KDA_FULL = SHARD_LINEAR_ATTN["num_heads"] * SHARD_LINEAR_ATTN["head_dim"]
_KDA_HEADS = SHARD_LINEAR_ATTN["num_heads"]

#: The suffix that makes a declared leaf a WEIGHT leaf, so a sibling scale grid
#: can be named from it. The production rule is ``model_fp8.py``'s
#: ``_WEIGHT_LEAF_SUFFIX`` with ``_sibling_scale_grid_name`` on top of it; this is
#: that rule stated once here rather than spelled inline at the one place below
#: that needs it.
_LEAF_WEIGHT_SUFFIX = "_weight"

#: The MLA head-width three, in the miniature's own numbers. Derived from
#: ``MINI_MLA_WIDTHS`` rather than typed, because that dict is what the fixture
#: builds the module from -- a width changed there moves these with it instead of
#: leaving a second set of numbers to drift.
_MLA_HEADS = MINI_MLA_WIDTHS["num_attention_heads"]
_MLA_Q_B_FULL = _MLA_HEADS * (
    MINI_MLA_WIDTHS["qk_nope_head_dim"] + MINI_MLA_WIDTHS["qk_rope_head_dim"]
)
_MLA_KV_B_FULL = _MLA_HEADS * (
    MINI_MLA_WIDTHS["qk_nope_head_dim"] + MINI_MLA_WIDTHS["v_head_dim"]
)
_MLA_O_PROJ_FULL = _MLA_HEADS * MINI_MLA_WIDTHS["v_head_dim"]

#: ``(declaring class, declared leaf) -> (shard dim, full extent on that dim)``.
#: The EIGHTEEN families the ratified table calls sharded once ``inc-glm53f-100``
#: has landed: twelve on ``Glm5NextKDAAttention``, three on ``Glm5NextDenseMLP``
#: and the MLA head-width three. The six still deferred -- the shared expert's
#: three and the routed bank's three, both to ``inc-glm53f-101`` -- are absent
#: here exactly as they are absent from the code's table, which is what conjunct
#: (4) counts.
#:
#: THE MLA THREE ARE ``inc-glm53f-100``'s ADDITION, and this is the deferral
#: above being kept rather than a new claim. The text this replaces read: "The
#: FIFTEEN families the ratified table calls sharded at this increment: twelve on
#: ``Glm5NextKDAAttention`` and three on ``Glm5NextDenseMLP``. The nine deferred
#: families -- the MLA head-width three to ``inc-glm53f-100``, the shared
#: expert's three and the routed bank's three to ``inc-glm53f-101`` -- are absent
#: here exactly as they are absent from the code's table, which is what conjunct
#: (4) counts."
SHARD_FAMILIES: dict[tuple[str, str], tuple[int, int]] = {
    ("Glm5NextKDAAttention", "q_proj_weight"): (0, _KDA_FULL),
    ("Glm5NextKDAAttention", "k_proj_weight"): (0, _KDA_FULL),
    ("Glm5NextKDAAttention", "v_proj_weight"): (0, _KDA_FULL),
    ("Glm5NextKDAAttention", "f_b_proj_weight"): (0, _KDA_FULL),
    ("Glm5NextKDAAttention", "g_b_proj_weight"): (0, _KDA_FULL),
    ("Glm5NextKDAAttention", "q_conv1d_weight"): (0, _KDA_FULL),
    ("Glm5NextKDAAttention", "k_conv1d_weight"): (0, _KDA_FULL),
    ("Glm5NextKDAAttention", "v_conv1d_weight"): (0, _KDA_FULL),
    ("Glm5NextKDAAttention", "o_proj_weight"): (1, _KDA_FULL),
    ("Glm5NextKDAAttention", "b_proj_weight"): (0, _KDA_HEADS),
    ("Glm5NextKDAAttention", "A_log"): (0, _KDA_HEADS),
    ("Glm5NextKDAAttention", "dt_bias"): (0, _KDA_HEADS),
    ("Glm5NextDenseMLP", "gate_proj_weight"): (0, SHARD_INTERMEDIATE),
    ("Glm5NextDenseMLP", "up_proj_weight"): (0, SHARD_INTERMEDIATE),
    ("Glm5NextDenseMLP", "down_proj_weight"): (1, SHARD_INTERMEDIATE),
    ("Glm5NextMLAAttention", "q_b_proj_weight"): (0, _MLA_Q_B_FULL),
    ("Glm5NextMLAAttention", "kv_b_proj_weight"): (0, _MLA_KV_B_FULL),
    ("Glm5NextMLAAttention", "o_proj_weight"): (1, _MLA_O_PROJ_FULL),
}

#: The extent on the dimension a family is NOT sharded on, for the families where
#: that extent is not the arbitrary :data:`SHARD_NARROW`.
#:
#: The MLA three are the only entries, and they are here because their shape is
#: CHECKED rather than arbitrary: ``prepare_projection_weights`` compares each MLA
#: projection against the module's own closed form, and every item on this fixture
#: reaches it through ``load_weights``. Written at ``SHARD_NARROW`` the three would
#: refuse the load instead of being sharded by it. ``_mla_key_overrides`` writes
#: the same closed form for the same reason; this table is what lets the shard
#: writer agree with it while still writing position-identifying values, which a
#: constant tensor cannot do (see :func:`_shard_pattern`).
SHARD_OTHER_EXTENT: dict[tuple[str, str], int] = {
    ("Glm5NextMLAAttention", "q_b_proj_weight"): MINI_MLA_WIDTHS["q_lora_rank"],
    ("Glm5NextMLAAttention", "kv_b_proj_weight"): MINI_MLA_WIDTHS["kv_lora_rank"],
    ("Glm5NextMLAAttention", "o_proj_weight"): MINI_MLA_WIDTHS["hidden_size"],
}

#: The two KDA leaves that are one number per head rather than a matrix.
SHARD_ONE_DIMENSIONAL = ("A_log", "dt_bias")

#: The dense MLP's three leaves, in the order conjunct (3) reports them.
SHARD_DENSE_LEAVES = ("gate_proj_weight", "up_proj_weight", "down_proj_weight")

#: RIDER N3. The three grid attribute names as LITERALS, so this file states them
#: once itself instead of only ever asking the code what it calls them. Conjunct
#: (3) asserts these against the model's ``_sibling_scale_grid_name`` and against
#: this file's ``_scale_grid_attribute`` -- which is ONE derivation checked twice,
#: not two, because that helper asks the model. THE LITERAL IS THE INDEPENDENT
#: STATEMENT, and it is the whole of what N3 adds: if the naming rule ever changes,
#: the two derivations move together and only these three strings object.
SHARD_GRID_ATTRIBUTES = (
    "gate_proj_weight_scale_inv",
    "up_proj_weight_scale_inv",
    "down_proj_weight_scale_inv",
)


def _shard_config(first_k_dense: int) -> Glm5NextConfig:
    """The shard fixture: :func:`_stacked_config`'s fields with two widths shrunk.

    ``n_shared_experts=0`` is carried over from :func:`_stacked_config` and is not
    a convenience. No completing load on THIS fixture has a shared-expert module,
    because the landed shared-expert scale prep cannot run on a 128-block
    miniature (``inc-glm53f-095b``, which hands that fixture to ``-054``). So the
    shared expert's three deferred families cannot be observed by a real load
    here at all, and conjunct (4) prints that rather than quietly counting six
    deferred families where the table names nine.

    THE FILE-WIDE CLAIM THIS SENTENCE USED TO MAKE IS NO LONGER TRUE, and it is
    narrowed rather than deleted so the reason survives. It read "No completing
    load in this file has a shared-expert module". ``inc-glm53f-054a`` item (iii)
    added one -- ``test_blocked_the_shared_expert_prep_completes_a_load_and_the_
    retile_ran``, on the 256-blocked checkpoint, where the prep can run because
    the extents are whole blocks and the load-path retile publishes the grid it
    wants. This fixture still has none, for the reason above.

    ``first_k_dense`` IS THE ONE VARYING FIELD, and the two values it takes are
    both readings rather than one real case and one convenience -- the same reason
    :func:`_dense_config` and :func:`_routed_config` differ by one field.

    * ``MINI_ALL_DENSE_FIRST_K`` builds NO routed expert bank, and it is what the
      items needing a RANK-1 load use. The landed bank refuses any rank at or
      above its EXPERT-parallel degree (``factory.py:132-136``), and this
      campaign's production expert-parallel degree is 1 (``factory.py:204-211``),
      so at tensor-parallel world size 2 the bank is replicated and every rank's
      expert-parallel rank is 0 -- but the loader passes the GLOBAL rank, so rank
      1 is refused. That refusal is `inc-glm53f-101`'s to answer, not this
      increment's, and conjunct (4) measures it rather than describing it.
    * ``MINI_FIRST_K_DENSE`` keeps the bank, and conjunct (4) needs it: the bank's
      three families are part of the replicated set that conjunct counts. It loads
      at rank 0 only, which the bank serves.
    """
    return Glm5NextConfig(
        text_config=Glm5NextTextConfig(
            num_hidden_layers=MINI_LAYERS,
            n_routed_experts=MINI_ROUTED_EXPERTS,
            n_shared_experts=0,
            first_k_dense_replace=first_k_dense,
            tie_word_embeddings=False,
            linear_attn_config=SHARD_LINEAR_ATTN,
            intermediate_size=SHARD_INTERMEDIATE,
            **MINI_MLA_WIDTHS,
        )
    )


def _shard_full_shape(
    family: str, leaf: str, shard_dim: int, full: int
) -> tuple[int, ...]:
    """The FULL (unsharded) shape this family's checkpoint tensor is written at.

    The other extent is the arbitrary :data:`SHARD_NARROW` unless the family
    declares a real one in :data:`SHARD_OTHER_EXTENT`. ``family`` is a parameter
    rather than derived from ``leaf`` because a leaf name alone does not identify
    a family: ``o_proj_weight`` is declared by both ``Glm5NextKDAAttention`` and
    ``Glm5NextMLAAttention``, and only the second has a checked closed form.
    """
    if leaf in SHARD_ONE_DIMENSIONAL:
        return (full,)
    other = SHARD_OTHER_EXTENT.get((family, leaf), SHARD_NARROW)
    return (full, other) if shard_dim == 0 else (other, full)


def _shard_pattern(
    shape: tuple[int, ...], dim: int, dtype: torch.dtype
) -> torch.Tensor:
    """A tensor whose value identifies its POSITION ALONG ``dim`` and nothing else.

    A shard selects along one dimension, so a value that is distinct per index
    along that dimension is what makes "did this rank get the right rows"
    answerable at all. The landed writer writes a CONSTANT
    (``torch.ones``/``torch.full``), and against a constant any slice passes for
    any other -- which is why these items supply their own tensors.

    FP8 IS BUILT FROM ITS OWN BYTES rather than cast from integers. ``e4m3`` has
    three mantissa bits, so 17 and 19 do not exist in it and a cast would collapse
    distinct rows onto one value. Reinterpreting the byte ``(i % 119) + 8`` gives a
    distinct finite value per index with no rounding at all.

    THE BYTE RANGE IS 8 TO 126, AND BOTH ENDS ARE CHOSEN. Bytes 1 to 7 are
    subnormal, where the 240/448 squeeze rounds several distinct bytes onto one
    value and weakens what conjunct (2) can detect; bytes 127 and 255 are ``NaN``
    in ``e4m3fn``, which no equality comparison survives. That leaves 119 values,
    so index ``i`` and ``i + 119`` share one -- the single aliasing this pattern
    has, stated rather than left to be discovered. Rank 0's first row (byte 8) and
    rank 1's first row (byte 17) differ, so a half-tensor indexing error cannot
    hide inside it.
    """
    extent = shape[dim]
    index = torch.arange(extent, dtype=torch.int64)
    if dtype is torch.float8_e4m3fn:
        line = ((index % 119) + 8).to(torch.uint8).view(torch.float8_e4m3fn)
    else:
        line = (index + 1).to(dtype)
    view = [1] * len(shape)
    view[dim] = extent
    return line.reshape(view).expand(shape).contiguous()


#: The classes whose LOAD PATH coarsens a checkpoint scale grid onto the consumer's
#: 256 granularity. Read off the two call sites that do it -- the republish the two
#: dense-shaped classes go through (``model_fp8.py:6915``) and the routed bank's own
#: prep (``:2281-2293``) -- rather than from a guess about which families are
#: quantised. A family outside this tuple keeps :func:`_shard_pattern`'s ramp,
#: because nothing rescales its weights there and the ramp's per-128-tile
#: distinctness is the stronger position reading.
_RETILED_AT_LOAD_CLASSES = (
    "Glm5NextDenseMLP",
    "Glm5NextSharedExperts",
    "Glm5NextRoutedExperts",
)

#: The exponents the pow2 grid cycles through, one per 256 block along the shard
#: dim. Eight of them, which is the widest block count any family in the deferred
#: fixture has -- the shared expert's 2048 shard extent is 8 blocks of 256 -- so no
#: tensor this fixture writes aliases at all. Small magnitudes on purpose: the
#: values multiply fp8 bytes in the dequantisations these items compare.
_POW2_GRID_EXPONENTS = tuple(range(-3, 5))


def _tiles_per_consumer_block() -> int:
    """How many checkpoint ``128`` tiles one consumer ``256`` block covers.

    Derived from the consumer's own block size and the checkpoint's, never typed:
    a fixture that hardcoded 2 would keep writing 2 the day either side moved.
    """
    block = _WL_FP8.consumer_block_quant_size()
    tiles, remainder = divmod(block, DEFAULT_WEIGHT_BLOCK_SIZE[0])
    assert remainder == 0 and len(set(DEFAULT_WEIGHT_BLOCK_SIZE)) == 1, (
        f"the consumer's {block} block is not a whole number of the checkpoint's "
        f"{DEFAULT_WEIGHT_BLOCK_SIZE} tiles, so no grid this fixture writes can be "
        f"one the coarsening reproduces and the items below would be measuring the "
        f"fixture instead of the loader"
    )
    return tiles


def _pow2_block_grid_pattern(shape: tuple[int, ...], dim: int) -> torch.Tensor:
    """A ``128``-tile scale grid the ``256`` coarsening reproduces BIT-EXACTLY.

    ``inc-glm53f-054a`` repair batch R6 item R-T2 wrote this, on a measurement
    rather than a preference. The coarsening keeps ONE scale per 256 block and
    rescales the block's other three 128 tiles into it
    (``blockwise_fp8_retile.py:443-480``), so on :func:`_shard_pattern`'s fp32 ramp
    -- 1, 2, 3, ... along the shard dim -- the first block's ratio is exactly 2 and
    every later one is a fraction like 4/3. Both consequences were measured on the
    host: the fractions raise ``inexact_rescales``, and an fp8 byte at the maximum
    doubled leaves what fp8-e4m3 can hold, which the landed cast turned into a
    silent NaN and R6 item R-P1 now REFUSES by name
    (``blockwise_fp8_retile.py:461-476``).

    So a grid whose four 128-tile scales AGREE inside every 256 block is not a
    fixture retuned to pass. It is the only grid family on which a BIT-EXACT
    reassembly can be asked at all: :func:`_as_the_loader_left_it` undoes the
    republish's FRAME and cannot undo a requantisation (see its own docstring), so
    any ratio other than 1 moves the weight bytes the comparison is about. With
    every ratio 1 the coarsening moves the layout and no byte, which is exactly
    what those items claim.

    Every value is an exact power of two, so the dequantisations these items
    compare stay exact in fp32 and the retained scale satisfies the complete
    losslessness condition ``inc-glm53f-024`` part 5 states. Each 256 BLOCK along
    ``dim`` gets its own value, so "did this rank get the right blocks" is still
    answerable -- and a rank boundary is always a whole block here, because the
    consumer refuses any other shard.

    WHAT IT GIVES UP, STATED RATHER THAN LEFT TO BE FOUND. A scale-position mixup
    INSIDE one 256 block is invisible on this grid, where the ramp would catch it.
    No single fixture can hold both readings. The ramp reading is kept by the item
    that keeps the ramp and asserts the refusal fires by name,
    :func:`test_blocked_a_ramp_scale_grid_refuses_instead_of_emitting_nan`.
    """
    per_block = _tiles_per_consumer_block()
    extent = shape[dim]
    block = torch.arange(extent, dtype=torch.int64) // per_block
    exponents = torch.tensor(_POW2_GRID_EXPONENTS, dtype=torch.int64)
    line = torch.ldexp(
        torch.ones(extent, dtype=torch.float32), exponents[block % exponents.numel()]
    )
    view = [1] * len(shape)
    view[dim] = extent
    return line.reshape(view).expand(shape).contiguous()


def _shard_key_overrides(
    model: Glm5NextForConditionalGeneration,
    mappings: dict[str, str | list[str]],
) -> dict[str, torch.Tensor]:
    """The FULL tensors the shard fixture's checkpoint holds, per checkpoint key.

    Nothing here spells a checkpoint key: the key comes from the map, the shape
    from :data:`SHARD_FAMILIES`, and a grid's shape from ``block_grid_shape`` --
    the same closed form the loader divides by. The dtype question, "is this
    family quantised", is answered by asking the map whether the entry carries a
    scale key, not from a list of family names kept here.

    A SPLIT MAP ENTRY IS ASKED THAT QUESTION A SECOND WAY, and that is
    ``inc-glm53f-100``'s change here. The DSA half's scaled projections carry the
    grid as a mapped parameter of their OWN -- ``weight_loaders_fp8.py:529-540``
    adds ``<leaf>_weight`` and ``<leaf>_weight_scale_inv`` as two entries -- so for
    them a one-key entry does NOT mean unquantised. Asking only about the entry's
    own keys would have written the MLA three as bf16 and silently dropped the fp8
    reading ``_mla_key_overrides`` already gave them, which is a fixture that no
    longer resembles the checkpoint. A lone weight key therefore also asks whether
    the map carries its sibling grid, by the same name rule the reader uses
    (``model_fp8.py``'s ``_sibling_scale_grid_name``). ``kv_b_proj`` has no
    sibling in this checkpoint and stays bf16, which is the reading
    ``inc-glm53f-078`` recorded.
    """
    overrides: dict[str, torch.Tensor] = {}
    for path, module in model.named_modules():
        cls = type(module).__name__
        for (family, leaf), (shard_dim, full) in SHARD_FAMILIES.items():
            if cls != family:
                continue
            param = f"{path}.{leaf}"
            if param not in mappings:
                continue
            keys = _keys_of(mappings, param)
            scales = scale_keys(keys)
            weight_key = next(key for key in keys if key not in scales)
            shape = _shard_full_shape(family, leaf, shard_dim, full)
            if not scales and leaf.endswith(_LEAF_WEIGHT_SUFFIX):
                grid_param = (
                    f"{param[: -len(_LEAF_WEIGHT_SUFFIX)]}_{FP8_SCALE_SUFFIX}"
                )
                if grid_param in mappings:
                    scales = _keys_of(mappings, grid_param)
            if scales:
                overrides[weight_key] = _shard_pattern(
                    shape, shard_dim, torch.float8_e4m3fn
                )
                overrides[scales[0]] = _shard_pattern(
                    block_grid_shape(shape, DEFAULT_WEIGHT_BLOCK_SIZE),
                    shard_dim,
                    torch.float32,
                )
            else:
                overrides[weight_key] = _shard_pattern(
                    shape, shard_dim, torch.bfloat16
                )
    return overrides


def _seed_page_cache_signal() -> None:
    """Supply the absent rank's page-cache signal, through the store the ranks use.

    NOT test convenience, and not a patch of anything under test.
    ``_load_to_page_cache`` splits the checkpoint's FILES across ranks round robin
    -- ``idx % world_size == rank`` (``utils/checkpoints.py:521``) -- and the
    reading loop then waits until every file's key appears in the shared store. A
    ONE-FILE miniature at world size 2 therefore gives rank 1 no file to cache and
    no key to add, and its load would wait for a key only another PROCESS could
    set. There is no other process inside a pytest item.

    The transform path -- what these items measure -- is untouched: the reader
    still opens the file and reads every tensor itself. Adding the key is
    idempotent, so rank 0 and the repeat loads are unaffected.
    """
    store = torch.distributed.distributed_c10d._get_default_store()
    store.add(MINI_CHECKPOINT_FILE, 1)


def _load_at_world(
    directory: Path,
    world_size: int,
    rank: int,
    monkeypatch,
    first_k_dense: int,
) -> Glm5NextForConditionalGeneration:
    """Build a model AT a synthetic world size and rank, then load the checkpoint.

    The world size is patched at the RESOLVER and the model built afterwards, and
    that ordering is the point: ``Glm5NextKDAAttention.__init__`` divides its head
    count by the world size it is given (``model_fp8.py:2215``), so a model built
    at world size 1 and re-labelled 2 would carry a full-width head count and a
    sharding loader at once -- the exact defect these items exist to detect,
    constructed by the test itself.
    """
    monkeypatch.setattr(_MODEL_FP8, "_resolve_world_size", lambda: world_size)
    monkeypatch.setattr(_MODEL_FP8, "_resolve_rank", lambda: rank)
    model = Glm5NextForConditionalGeneration(_shard_config(first_k_dense))
    assert model.world_size == world_size, (
        f"the model resolved world size {model.world_size}, not the patched "
        f"{world_size}, so this is not the load this item means to measure"
    )
    _seed_page_cache_signal()
    model.load_weights(str(directory), torch.device("cpu"), None)
    return model


def _shard_checkpoint(
    tmp_path: Path, first_k_dense: int
) -> tuple[Path, dict[str, torch.Tensor], dict]:
    """Write ONE checkpoint holding the full tensors; return it and its contents."""
    config = _shard_config(first_k_dense)
    mappings = _mappings_for(config)
    reference = Glm5NextForConditionalGeneration(config)
    overrides = _shard_key_overrides(reference, mappings)
    # ``inc-glm53f-054a``'s migration. This fixture's own narrow width is
    # :data:`SHARD_NARROW`, which is 8 and stays 8 -- it is ``-094``'s constant and
    # every family here except the bank is measured against it. The bank is the one
    # family whose load now runs a scale prep, and that prep retiles, and the retile
    # refuses an extent that is not a whole 256 block. So the bank alone takes
    # :data:`BLOCKED_BANK_IN`, from its own table, and the eighteen sharded families
    # are untouched. On the all-dense layout this adds no key at all, because no bank
    # module exists to be found.
    overrides.update(_blocked_bank_overrides(reference, mappings))
    assert len(overrides) >= len(SHARD_FAMILIES), (
        f"only {len(overrides)} checkpoint keys were given shard tensors, fewer "
        f"than the {len(SHARD_FAMILIES)} families this file declares sharded, so "
        f"some family is missing from the map and would be measured against a "
        f"constant"
    )
    directory = tmp_path / f"shard-{first_k_dense}"
    _write_miniature_checkpoint(
        directory, mappings, reference, extra_overrides=overrides
    )
    return directory, overrides, mappings


def _sharded_leaves(
    model: Glm5NextForConditionalGeneration,
) -> list[tuple[str, torch.nn.Module, str, int, int]]:
    """Every ``(path, module, leaf, shard_dim, full)`` THIS FILE calls sharded."""
    found: list[tuple[str, torch.nn.Module, str, int, int]] = []
    for path, module in model.named_modules():
        cls = type(module).__name__
        declared = getattr(module, "declared_param_names", ())
        for (family, leaf), (shard_dim, full) in SHARD_FAMILIES.items():
            if cls == family and leaf in declared:
                found.append((path, module, leaf, shard_dim, full))
    return found


def _loaded(model: Glm5NextForConditionalGeneration, dotted: str) -> torch.Tensor:
    loaded = dict(model.named_parameters())
    assert dotted in loaded, f"{dotted} is not in named_parameters() after the load"
    return loaded[dotted].data


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    """Exact difference, taken in fp32 so an fp8 pair can be compared at all."""
    if left.numel() == 0:
        return 0.0
    return (left.to(torch.float32) - right.to(torch.float32)).abs().max().item()


# --------------------------------------------------------------------------- #
# (1) Every sharded family lands at its declared per-rank shape.
# --------------------------------------------------------------------------- #


#: The two classes whose loaded tensors are REPUBLISHED before any forward sees them.
#: ``inc-glm53f-054a`` repair round 1: their load path now transposes each weight and
#: its scale grid once, into the frame ``blockwise_fp8_mm`` multiplies in, because the
#: checkpoint's own layout is the transpose of it and the kernel scale operand is built
#: at load from the stored extents (so a per-forward transpose would agree on shape and
#: be wrong on numbers). The routed expert bank is NOT here: it has its own prep and
#: this republish never runs on it.
_REPUBLISHED_CLASSES = ("Glm5NextDenseMLP", "Glm5NextSharedExperts")


def _as_the_loader_left_it(
    module: torch.nn.Module, name: str, tensor: torch.Tensor
) -> torch.Tensor:
    """One loaded tensor put back in the frame the LOADER delivered it in.

    ``inc-glm53f-054a`` repair round 1. Every reading below that is about the
    LOADER's own work -- which dim a family shards on, what a per-rank slice is,
    whether two ranks reassemble the whole tensor -- asks its question of the
    checkpoint's layout, and the load path no longer leaves the two republished
    classes in that layout. So those readings pass through here, and their expected
    values do NOT move: the declared shard dims, the per-rank widths and the
    bit-exact reassembly are all still asserted against the same numbers this file
    always asserted them against.

    THE UNDO IS A TRANSPOSE AND NOTHING ELSE. The republish's other step, coarsening
    a ``128``-tile grid onto the ``256`` public one, is NOT undone here and cannot
    be: it requantises. A reading that needs the raw grid VALUES of a republished
    class has to say so itself; this helper only restores the FRAME.

    It keys on the class rather than on a shape, because a square weight's frame is
    invisible in its shape and a reading that guessed from the shape would silently
    stop undoing anything the day a miniature stopped being square.
    """
    if type(module).__name__ not in _REPUBLISHED_CLASSES:
        return tensor
    if not name.startswith(("gate_proj", "up_proj", "down_proj")):
        return tensor
    if tensor.dim() != 2:
        return tensor
    return tensor.t()


def _in_the_loader_frame(
    model: Glm5NextForConditionalGeneration, dotted: str
) -> torch.Tensor:
    """``_as_the_loader_left_it`` for a reading that holds only the dotted name."""
    path, leaf = dotted.rsplit(".", 1)
    return _as_the_loader_left_it(
        model.get_submodule(path), leaf, _loaded(model, dotted)
    )


def test_shard_every_sharded_family_lands_at_its_declared_per_rank_shape(
    tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """Conjunct (1), counted N/N, with the world-size-1 load as the moving control.

    THE EXPECTED SHAPE IS NEVER ASKED OF THE CODE. It is the full extent this file
    declares in :data:`SHARD_FAMILIES` divided by the world size, with the other
    extent unchanged; ``_shard_geometry_for`` is not called here.

    BOTH RANKS LOAD, because the conjunct says so: "ranks 0 and 1 each load the
    real miniature checkpoint". Rank 0 alone would leave the second half of every
    tensor unread, and the second half is where an off-by-one start index lives.

    CERTIFYING COMPONENT (D1.4): the loader attachment, ``load_weights`` reaching
    the geometry through ``get_weight_loader`` -- not the geometry table alone,
    which a source reading could have checked.
    """
    directory, _, _ = _shard_checkpoint(tmp_path, MINI_ALL_DENSE_FIRST_K)

    whole = _load_at_world(directory, 1, 0, monkeypatch, MINI_ALL_DENSE_FIRST_K)
    control = {
        f"{path}.{leaf}": tuple(
            _as_the_loader_left_it(
                module, leaf, _loaded(whole, f"{path}.{leaf}")
            ).shape
        )
        for path, module, leaf, _, _ in _sharded_leaves(whole)
    }
    print(f"CONJUNCT1_FAMILIES_AT_WORLD_1={len(control)}")

    per_rank = {
        rank: _load_at_world(
            directory, SHARD_WORLD, rank, monkeypatch, MINI_ALL_DENSE_FIRST_K
        )
        for rank in range(SHARD_WORLD)
    }
    leaves = _sharded_leaves(per_rank[0])
    assert leaves, "no sharded family is present in the tree, so this item is empty"

    checked = 0
    families_seen: set[tuple[str, str]] = set()
    for rank, sharded in sorted(per_rank.items()):
        for path, module, leaf, shard_dim, full in _sharded_leaves(sharded):
            dotted = f"{path}.{leaf}"
            full_shape = _shard_full_shape(
                type(module).__name__, leaf, shard_dim, full
            )
            expected = list(full_shape)
            expected[shard_dim] = full // SHARD_WORLD
            # READ IN THE LOADER'S FRAME: the declared per-rank shape below is a
            # statement about the checkpoint's layout, and the two republished
            # classes no longer store that layout.
            got = tuple(
                _as_the_loader_left_it(module, leaf, _loaded(sharded, dotted)).shape
            )
            assert got == tuple(expected), (
                f"{dotted} loaded {got} at rank {rank} of world size {SHARD_WORLD}; "
                f"this file declares it sharded on dim {shard_dim} of a full "
                f"{full_shape}, so its per-rank shape is {tuple(expected)}"
            )
            assert control[dotted] == full_shape, (
                f"{dotted} loaded {control[dotted]} at world size 1 but the "
                f"checkpoint holds {full_shape}; the control is not reading the "
                f"whole tensor, so the comparison above means nothing"
            )
            families_seen.add((type(module).__name__, leaf))
            checked += 1

    # BOTH SIDES IN THE LOADER'S FRAME. ``control`` was read in that frame, so a
    # raw read here would count a republished family as "moved" because its two
    # axes swapped, which is not what this reading is for: it is for showing that
    # the shard changed an extent (D1.5).
    moved = {
        rank: sum(
            1
            for dotted, shape in control.items()
            if shape != tuple(_in_the_loader_frame(sharded, dotted).shape)
        )
        for rank, sharded in sorted(per_rank.items())
    }
    print(f"CONJUNCT1_PER_RANK_SHAPES_AS_DECLARED={checked}/{SHARD_WORLD * len(leaves)}")
    print(f"CONJUNCT1_DISTINCT_FAMILIES={len(families_seen)}/{len(SHARD_FAMILIES)}")
    print(f"CONJUNCT1_CONTROL_SHAPES_THAT_MOVED_PER_RANK={moved}/{len(control)}")
    assert checked == SHARD_WORLD * len(leaves)
    assert len(families_seen) == len(SHARD_FAMILIES), (
        f"the load exercised {len(families_seen)} of the {len(SHARD_FAMILIES)} "
        f"families this file declares sharded; missing "
        f"{sorted(set(SHARD_FAMILIES) - families_seen)}"
    )
    for rank, count in moved.items():
        assert count == len(control), (
            f"only {count} of {len(control)} shapes differ between world size 1 and "
            f"rank {rank} of world size {SHARD_WORLD}; a reading that does not move "
            f"cannot show that the shard happened (D1.5)"
        )


# --------------------------------------------------------------------------- #
# (2) The shards re-assemble the tensor bit-identically.
# --------------------------------------------------------------------------- #


def test_shard_the_two_ranks_reassemble_every_family_bit_identically(
    tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """Conjunct (2). Rank 0 then rank 1 along the shard dim, max abs diff == 0.0.

    THE COMPARISON IS AGAINST THE WORLD-SIZE-1 LOAD, not against the bytes in the
    file, and that is deliberate. A quantised weight is squeezed into the trn2
    range on the way in (``wrap_with_blockwise_fp8_downscale``), so the file's
    bytes are not what any rank ends up holding. Both sides of this comparison go
    through the same squeeze, which leaves the INDEXING as the only thing left for
    it to measure -- and the indexing is what a shard is.
    """
    directory, _, _ = _shard_checkpoint(tmp_path, MINI_ALL_DENSE_FIRST_K)
    dense = MINI_ALL_DENSE_FIRST_K

    whole = _load_at_world(directory, 1, 0, monkeypatch, dense)
    rank0 = _load_at_world(directory, SHARD_WORLD, 0, monkeypatch, dense)
    rank1 = _load_at_world(directory, SHARD_WORLD, 1, monkeypatch, dense)

    leaves = _sharded_leaves(whole)
    assert leaves, "no sharded family is present in the tree, so this item is empty"

    exact = 0
    rejoined = 0
    worst = 0.0
    ranks_differ = 0
    for path, module, leaf, shard_dim, full in leaves:
        dotted = f"{path}.{leaf}"
        # BOTH SIDES IN THE LOADER'S FRAME. ``narrow`` below takes the slice on the
        # dim this file declares the family sharded on, which is a dim of the
        # CHECKPOINT's layout; on a republished class the stored tensor's dims are
        # swapped, and narrowing 256 rows out of a dim of 8 would raise rather than
        # fail an assertion.
        reference = _as_the_loader_left_it(module, leaf, _loaded(whole, dotted))
        per_rank = full // SHARD_WORLD
        for rank, model in ((0, rank0), (1, rank1)):
            mine = _as_the_loader_left_it(module, leaf, _loaded(model, dotted))
            expected = reference.narrow(shard_dim, rank * per_rank, per_rank)
            assert tuple(mine.shape) == tuple(expected.shape), (
                f"{dotted} at rank {rank} is {tuple(mine.shape)} and the matching "
                f"slice of the whole tensor is {tuple(expected.shape)}"
            )
            difference = _max_abs_diff(mine, expected)
            worst = max(worst, difference)
            assert difference == 0.0, (
                f"{dotted} at rank {rank} differs from indices "
                f"[{rank * per_rank}:{(rank + 1) * per_rank}] of dim {shard_dim} of "
                f"the world-size-1 tensor by {difference}; a shard is a slice, so "
                f"any difference at all is an indexing defect"
            )
            exact += 1

        # THE CONJUNCT'S OWN OPERATION, not just its consequence: "concatenating
        # rank 0's and rank 1's tensors along the shard dim equals the unsharded
        # tensor". The two slice comparisons above imply it, and this reads it the
        # way the criterion is written -- and it also fails if the two halves are
        # each right but no longer join to the declared full extent.
        # Both pieces back in the loader's frame first: ``shard_dim`` is a dim of
        # the CHECKPOINT's layout, which the republished classes no longer store.
        reassembled = torch.cat(
            [
                _in_the_loader_frame(rank0, dotted),
                _in_the_loader_frame(rank1, dotted),
            ],
            dim=shard_dim,
        )
        assert tuple(reassembled.shape) == tuple(reference.shape), (
            f"{dotted}: the two ranks' tensors concatenate to "
            f"{tuple(reassembled.shape)}, and the world-size-1 tensor is "
            f"{tuple(reference.shape)}"
        )
        joined = _max_abs_diff(reassembled, reference)
        worst = max(worst, joined)
        assert joined == 0.0, (
            f"{dotted}: rank 0 and rank 1 concatenated along dim {shard_dim} differ "
            f"from the unsharded tensor by {joined}"
        )
        rejoined += 1

        if (
            _max_abs_diff(
                _in_the_loader_frame(rank0, dotted).narrow(shard_dim, 0, 1),
                _in_the_loader_frame(rank1, dotted).narrow(shard_dim, 0, 1),
            )
            != 0.0
        ):
            ranks_differ += 1

    print(f"CONJUNCT2_SLICES_BIT_IDENTICAL={exact}/{2 * len(leaves)}")
    print(f"CONJUNCT2_REASSEMBLED_EQUALS_WHOLE={rejoined}/{len(leaves)}")
    print(f"CONJUNCT2_MAX_ABS_DIFF={worst}")
    print(f"CONJUNCT2_FAMILIES_WHOSE_RANKS_DIFFER={ranks_differ}/{len(leaves)}")
    assert exact == 2 * len(leaves)
    assert rejoined == len(leaves)
    assert worst == 0.0
    assert ranks_differ == len(leaves), (
        f"only {ranks_differ} of {len(leaves)} families hold DIFFERENT data on the "
        f"two ranks; for the rest, rank 1 could be reading rank 0's rows and every "
        f"assertion above would still pass (D1.5)"
    )


# --------------------------------------------------------------------------- #
# (3) The scale grid follows its weight, and a misaligned shard refuses by name.
# --------------------------------------------------------------------------- #


def test_shard_the_scale_grid_follows_its_weight_and_refuses_misalignment(
    tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """Conjunct (3), and rider N3's three literal attribute names.

    THE DENSE MLPS' THREE GRIDS EACH ARE THE WHOLE POPULATION HERE -- twelve on
    this fixture, whose four layers are all dense. ``Glm5NextKDAAttention``'s
    twelve families are UNQUANTISED in this checkpoint, so they have no grid to
    follow, and every other quantised sharded family is deferred to ``-100`` or
    ``-101``. The count is read off the tree rather than written down, so the
    number moves with the fixture instead of going stale beside it.

    THE GRID IS NOT A PARAMETER. It travels in its weight's own map entry and is
    read by ``_load_out_of_band_scales``, which stores it as a plain attribute --
    so this item reads attributes, not ``named_parameters()``.
    """
    directory, overrides, mappings = _shard_checkpoint(
        tmp_path, MINI_ALL_DENSE_FIRST_K
    )
    layout = MINI_ALL_DENSE_FIRST_K

    whole = _load_at_world(directory, 1, 0, monkeypatch, layout)
    rank0 = _load_at_world(directory, SHARD_WORLD, 0, monkeypatch, layout)
    rank1 = _load_at_world(directory, SHARD_WORLD, 1, monkeypatch, layout)

    dense = [
        path
        for path, module in whole.named_modules()
        if type(module).__name__ == "Glm5NextDenseMLP"
    ]
    assert dense, "the shard fixture built no dense MLP, so conjunct (3) is empty"

    # RIDER N3: the three names as literals. ``_scale_grid_attribute`` asks the
    # model, so it is the same derivation read a second way; the LITERAL is what
    # this file states on its own.
    for leaf, literal in zip(SHARD_DENSE_LEAVES, SHARD_GRID_ATTRIBUTES, strict=True):
        from_model = whole._sibling_scale_grid_name(leaf)
        from_file = _scale_grid_attribute(leaf)
        assert from_model == literal, (
            f"the model derives {from_model!r} as {leaf}'s grid attribute; this "
            f"file names it {literal!r}. One of the two is wrong, and a test that "
            f"only asked the code could not tell which"
        )
        assert from_file == literal, (
            f"this file's own helper derives {from_file!r}, not {literal!r}"
        )
    print(f"CONJUNCT3_N3_LITERAL_GRID_NAMES={len(SHARD_GRID_ATTRIBUTES)}")

    followed = 0
    grids = 0
    for path in dense:
        for leaf, attribute in zip(
            SHARD_DENSE_LEAVES, SHARD_GRID_ATTRIBUTES, strict=True
        ):
            shard_dim, full = SHARD_FAMILIES[("Glm5NextDenseMLP", leaf)]
            grid_key = scale_keys(_keys_of(mappings, f"{path}.{leaf}"))[0]
            written = overrides[grid_key]
            blocks = (full // SHARD_WORLD) // DEFAULT_WEIGHT_BLOCK_SIZE[shard_dim]
            assert blocks >= 1, (
                f"{path}.{leaf}'s per-rank extent is narrower than one block, so "
                f"there is no aligned grid shard for this item to measure"
            )
            # THE GRID IS READ IN THE LOADER'S FRAME, for this item's own reason: it
            # compares the grid against the one the CHECKPOINT holds and against
            # slices taken on the dim the checkpoint shards. The dense MLP's load
            # path now republishes both weight and grid transposed. Nothing is
            # coarsened here -- this fixture's per-rank hidden extent is
            # SHARD_NARROW, which is not a whole consumer block, so the republish's
            # first step records a skip and only the frame moves.
            whole_module = whole.get_submodule(path)
            reference = _as_the_loader_left_it(
                whole_module, attribute, getattr(whole_module, attribute)
            )
            assert tuple(reference.shape) == tuple(written.shape), (
                f"{path}.{attribute} arrived {tuple(reference.shape)} at world size "
                f"1 but the checkpoint holds {tuple(written.shape)}"
            )
            grids += 1
            for rank, model in ((0, rank0), (1, rank1)):
                rank_module = model.get_submodule(path)
                mine = getattr(rank_module, attribute, None)
                assert mine is not None, (
                    f"{path}.{attribute} does not exist after the rank-{rank} load, "
                    f"so a sharded weight's grid never arrived at all"
                )
                mine = _as_the_loader_left_it(rank_module, attribute, mine)
                expected = reference.narrow(shard_dim, rank * blocks, blocks)
                assert tuple(mine.shape) == tuple(expected.shape), (
                    f"{path}.{attribute} is {tuple(mine.shape)} at rank {rank}; its "
                    f"weight is sharded on dim {shard_dim}, so its grid must be "
                    f"{tuple(expected.shape)} -- a full grid here would describe "
                    f"blocks this rank does not hold, and the dequant would scale "
                    f"the wrong ones"
                )
                difference = _max_abs_diff(mine, expected)
                assert difference == 0.0, (
                    f"{path}.{attribute} at rank {rank} differs from the matching "
                    f"blocks of the whole grid by {difference}"
                )
                assert mine.dtype is torch.float32, (
                    f"{path}.{attribute} arrived {mine.dtype}, not fp32"
                )
                followed += 1

    moved = sum(
        1
        for path in dense
        for attribute in SHARD_GRID_ATTRIBUTES
        if _max_abs_diff(
            getattr(rank0.get_submodule(path), attribute),
            getattr(rank1.get_submodule(path), attribute),
        )
        != 0.0
    )
    print(f"CONJUNCT3_GRIDS_MEASURED={grids}")
    print(f"CONJUNCT3_GRID_SHARDS_FOLLOWING_THEIR_WEIGHT={followed}/{2 * grids}")
    print(f"CONJUNCT3_GRIDS_DIFFERING_BETWEEN_RANKS={moved}/{grids}")
    assert grids == len(dense) * len(SHARD_GRID_ATTRIBUTES)
    assert followed == 2 * grids
    assert moved == grids, (
        f"only {moved} of {grids} grids differ between the two ranks, so a grid "
        f"that ignored the rank entirely would pass the readings above (D1.5)"
    )

    # A BLOCK-MISALIGNED SHARD REFUSES BY NAME. The aligned case beside it is what
    # makes the refusal a boundary rather than a blanket. The exception class is
    # the one ``_refuse`` raises for every refusal in that section, bank or not.
    # ``inc-glm53f-101`` (DECISIONS section 80(b)): the aligned case is now one whole
    # CONSUMER block, which is two checkpoint tiles. One tile alone is what the new
    # gate refuses, so the old aligned value became the third control below.
    aligned = _WL_FP8.shard_geometry_for_grid(
        _WL_FP8.ShardGeometry(
            shard_dim=0,
            shard_size=2 * DEFAULT_WEIGHT_BLOCK_SIZE[0],
            num_shards=SHARD_WORLD,
        ),
        param_name="probe.gate_proj_weight_scale_inv",
    )
    print(f"CONJUNCT3_ALIGNED_GRID_SHARD_SIZE={aligned.shard_size}")
    assert aligned.shard_size == 2, (
        f"an aligned shard of exactly one consumer block gave "
        f"{aligned.shard_size} grid rows, not 2"
    )
    misaligned = DEFAULT_WEIGHT_BLOCK_SIZE[0] + 1
    with pytest.raises(Glm5NextExpertBankNotLoadableError) as refusal:
        _WL_FP8.shard_geometry_for_grid(
            _WL_FP8.ShardGeometry(
                shard_dim=0, shard_size=misaligned, num_shards=SHARD_WORLD
            ),
            param_name="probe.gate_proj_weight_scale_inv",
        )
    message = str(refusal.value)
    print(f"CONJUNCT3_MISALIGNED_REFUSAL={message}")
    assert "probe.gate_proj_weight_scale_inv" in message, (
        f"the refusal does not name the parameter: {message}"
    )
    assert str(DEFAULT_WEIGHT_BLOCK_SIZE[0]) in message, (
        f"the refusal does not name the block boundary it enforced: {message}"
    )

    # THE THIRD CONTROL, and the one the consumer's gate exists for
    # (``inc-glm53f-101``, DECISIONS section 80(b)). 384 rows is THREE whole
    # checkpoint tiles, so the tile rule above lets it pass; it is one and a half
    # consumer blocks, so the kernel cannot index it. Without this reading the new
    # gate could be deleted and every assertion above would still pass.
    consumer_block = _WL_FP8.consumer_block_quant_size()
    tile_clearing_block_missing = 3 * DEFAULT_WEIGHT_BLOCK_SIZE[0]
    print(f"CONJUNCT3_CONSUMER_BLOCK={consumer_block}")
    print(f"CONJUNCT3_TILE_CLEARING_SHARD={tile_clearing_block_missing}")
    assert tile_clearing_block_missing % DEFAULT_WEIGHT_BLOCK_SIZE[0] == 0, (
        f"{tile_clearing_block_missing} is not a whole number of "
        f"{DEFAULT_WEIGHT_BLOCK_SIZE[0]}-row tiles, so it would be refused by the "
        f"tile rule and this control would certify nothing about the consumer's"
    )
    assert tile_clearing_block_missing % consumer_block != 0, (
        f"{tile_clearing_block_missing} IS a whole number of {consumer_block}-row "
        f"consumer blocks, so it is not the case this control means to construct"
    )
    with pytest.raises(Glm5NextExpertBankNotLoadableError) as consumer_refusal:
        _WL_FP8.shard_geometry_for_grid(
            _WL_FP8.ShardGeometry(
                shard_dim=0,
                shard_size=tile_clearing_block_missing,
                num_shards=SHARD_WORLD,
            ),
            param_name="probe.gate_proj_weight_scale_inv",
        )
    consumer_message = str(consumer_refusal.value)
    print(f"CONJUNCT3_CONSUMER_REFUSAL={consumer_message}")
    assert "probe.gate_proj_weight_scale_inv" in consumer_message, (
        f"the consumer refusal does not name the parameter: {consumer_message}"
    )
    assert str(consumer_block) in consumer_message, (
        f"the consumer refusal does not name the {consumer_block}-row block it "
        f"enforced: {consumer_message}"
    )


# --------------------------------------------------------------------------- #
# (4) The unsharded families are untouched.
# --------------------------------------------------------------------------- #


def test_shard_the_unsharded_families_are_untouched_both_directions(
    tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """Conjunct (4), both directions: the set that MOVED between world sizes is
    exactly the set this file declares sharded, and the set that stayed identical
    is exactly its complement.

    THE DEFERRED FAMILIES ARE COUNTED HERE, in the ratified table's own words:
    "replicated-in-effect, attachment deferred". Whichever of them this
    configuration builds has to load identically at both world sizes, which is
    what "attaches nothing to them" means once it is measured instead of asserted.
    """
    directory, _, _ = _shard_checkpoint(tmp_path, MINI_FIRST_K_DENSE)
    # Named for what the configuration CONTAINS rather than for the field, because
    # ``routed`` is taken further down by the bank's leaf names.
    with_bank = MINI_FIRST_K_DENSE

    whole = _load_at_world(directory, 1, 0, monkeypatch, with_bank)
    rank0 = _load_at_world(directory, SHARD_WORLD, 0, monkeypatch, with_bank)

    left = dict(whole.named_parameters())
    right = dict(rank0.named_parameters())
    assert set(left) == set(right), (
        f"the two loads registered different parameter names, so no family can be "
        f"compared. Only at world 1: {sorted(set(left) - set(right))[:3]}; only at "
        f"world {SHARD_WORLD}: {sorted(set(right) - set(left))[:3]}"
    )
    print(f"CONJUNCT4_PARAMETERS_COMPARED={len(left)}")

    # ``inc-glm53f-101``, the FIFTH moved number (DECISIONS §83 ruling 2). The
    # routed bank's three families join the declared-sharded set here. They were
    # counted in the REPLICATED set at ``-094`` -- "replicated-in-effect,
    # attachment deferred to inc-glm53f-101" in the ratified table's own words --
    # and this reading ends that count, because `-101` attached them: at world 2
    # with expert-parallel degree 1 the bank's experts are all local and its
    # intermediate width divides across the whole world, so every one of its
    # leaves differs between the two world sizes. Measured before the edit:
    # ``CONJUNCT4_MOVED=48`` against ``CONJUNCT4_DECLARED_SHARDED=39``, the nine
    # names being the bank's three leaves in the three MoE layers
    # (``accept-101-r9-host.out``). ``declared`` now reads 48.
    declared = {
        f"{path}.{leaf}" for path, _, leaf, _, _ in _sharded_leaves(whole)
    } | {
        f"{path}.{leaf}"
        for path, module, leaf, _, _ in _deferred_leaves(whole)
        if type(module).__name__ in DEFERRED_EP_GROUP_CLASSES
    }
    identical = {
        name
        for name in left
        if tuple(left[name].shape) == tuple(right[name].shape)
        and _max_abs_diff(left[name].data, right[name].data) == 0.0
    }
    moved = set(left) - identical

    print(f"CONJUNCT4_IDENTICAL_AT_BOTH_WORLD_SIZES={len(identical)}")
    print(f"CONJUNCT4_MOVED={len(moved)}")
    print(f"CONJUNCT4_DECLARED_SHARDED={len(declared)}")
    assert moved == declared, (
        f"the set that MOVED between world sizes is not the set this file declares "
        f"sharded. Moved but not declared: {sorted(moved - declared)[:5]}. Declared "
        f"but did not move: {sorted(declared - moved)[:5]}"
    )
    assert identical == set(left) - declared, (
        "the set that stayed identical is not exactly the complement of the sharded "
        "set, so some parameter is neither replicated nor sharded"
    )

    # The deferred families, counted by the module that declares them.
    by_owner: dict[str, set[str]] = {}
    for name in sorted(identical):
        owner = type(whole.get_submodule(name.rpartition(".")[0])).__name__
        by_owner.setdefault(owner, set()).add(name.rpartition(".")[2])
    # The same tally over the set that MOVED, needed because the routed clause is
    # now a counted zero: an empty replicated tally reads the same whether the bank
    # left the replicated set or was never built, and D1.5 wants the reading to
    # move. This is the side that must be non-empty.
    by_owner_moved: dict[str, set[str]] = {}
    for name in sorted(moved):
        owner = type(whole.get_submodule(name.rpartition(".")[0])).__name__
        by_owner_moved.setdefault(owner, set()).add(name.rpartition(".")[2])
    mla = sorted(by_owner.get("Glm5NextMLAAttention", ()))
    routed = sorted(by_owner.get("Glm5NextRoutedExperts", ()))
    shared = sorted(by_owner.get("Glm5NextSharedExperts", ()))
    print(f"CONJUNCT4_MLA_LEAVES_REPLICATED_IN_EFFECT={mla}")
    print(f"CONJUNCT4_ROUTED_LEAVES_REPLICATED_IN_EFFECT={routed}")
    print(f"CONJUNCT4_SHARED_EXPERT_LEAVES_PRESENT={shared}")
    # ``inc-glm53f-100`` REVISED THIS CLAUSE, and the old text is kept beside the
    # new because the reading changed rather than the code drifting. It read: "no
    # MLA parameter stayed identical, so the three families deferred to
    # inc-glm53f-100 cannot be counted replicated-in-effect here". The three
    # head-width families are no longer deferred -- they are declared above and
    # they MOVE, counted by ``moved == declared``. What stays identical on this
    # module is the LATENT side: ``q_a_proj``, ``kv_a_proj_with_mqa``, the two
    # layernorms and the scale grids. Those are replicated by the architecture and
    # not by deferral -- MLA compresses one latent per TOKEN, not per head, so
    # there is no head axis to split them on -- and the clause now holds them.
    assert mla, (
        "no MLA parameter stayed identical. The head-width three are sharded from "
        "inc-glm53f-100 on, but the LATENT projections and layernorms have no head "
        "axis to shard and must still be replicated, so an empty list here means "
        "something sharded that this architecture cannot shard"
    )
    # ── THE ROUTED CLAUSE, INVERTED BY ``inc-glm53f-101`` ─────────────────────
    # DECISIONS §83 ruling 2. The clause this replaces read, in these words:
    #
    #     assert routed, (
    #         "no routed-expert parameter stayed identical, so the families
    #         deferred to inc-glm53f-101 cannot be counted replicated-in-effect
    #         here")
    #
    # It asserted the bank stayed REPLICATED, and the message named this
    # increment as the one that would end that reading. `-101` attached the
    # bank's three families, so the assertion now says the opposite of what it
    # said, and the old text is quoted rather than deleted so the reversal is
    # legible -- the form the (v) replacement below already uses.
    # THE THREE PROJECTIONS, NOT EVERY LEAF THE CLASS DECLARES. The two routers
    # are frozen REPLICATED by the ratified table (§54 Part 3, "both routers
    # replicated") and they are declared on this same module, so a clause over the
    # whole class would demand the routers move too. Measured: a first draft of
    # this inversion failed on `['router_bias', 'router_weight']`
    # (`selftest-101-r10-host.out`), which is the table's answer and not a defect.
    bank_projections = {
        leaf
        for (family, leaf) in DEFERRED_FAMILIES
        if family in DEFERRED_EP_GROUP_CLASSES
    }
    routed_that_moved = sorted(by_owner_moved.get("Glm5NextRoutedExperts", ()))
    projections_still_replicated = sorted(set(routed) & bank_projections)
    routers_still_replicated = sorted(set(routed) - bank_projections)
    print(f"CONJUNCT4_ROUTED_LEAVES_THAT_MOVED={routed_that_moved}")
    print(
        "CONJUNCT4_ROUTED_PROJECTIONS_STILL_REPLICATED="
        f"{projections_still_replicated}"
    )
    print(f"CONJUNCT4_ROUTERS_STILL_REPLICATED={routers_still_replicated}")
    assert projections_still_replicated == [], (
        f"a routed-expert PROJECTION stayed identical across the two world sizes: "
        f"{projections_still_replicated}. inc-glm53f-101 attached the bank's three "
        f"families, so at world 2 with expert-parallel degree 1 every one of them "
        f"must differ -- the experts are all local and the intermediate width "
        f"divides across the whole world. A leaf that did not move is a family the "
        f"attachment missed"
    )
    assert sorted(set(routed_that_moved) & bank_projections) == sorted(
        bank_projections
    ), (
        f"the three bank projections did not all move: moved "
        f"{sorted(set(routed_that_moved) & bank_projections)} of "
        f"{sorted(bank_projections)}. An empty replicated reading above would then "
        f"mean this configuration built no bank rather than that -101 attached it"
    )
    assert routers_still_replicated, (
        "no router stayed identical. The ratified table freezes both routers "
        "REPLICATED (§54 Part 3), so the inversion above has widened past the "
        "three families inc-glm53f-101 attached"
    )
    assert shared == [], (
        "this fixture built a shared-expert module. The load is only known to "
        "complete without one (inc-glm53f-095b), so if one is present the "
        "disclosure in _shard_config is stale and the shared expert's three "
        "deferred families should be counted here rather than named as absent"
    )

    # ── RIDER N4: the firing control for a counted ZERO, in this item because ──
    # this item's instrument is the same one. ``inc-glm53f-095b``'s bankscale
    # scan prints ``BANKSCALE1_GRIDS_IN_NAMED_PARAMETERS`` and asserts the list
    # is EMPTY (``:2608``), and a membership test against
    # ``named_parameters()`` that never fires reads empty for two different
    # reasons: nothing was registered, or the test was asking the wrong
    # question. D1.5 wants the reading to MOVE, so the same predicate is put to
    # a module where a grid IS registered and read as 1.
    #
    # A THROWAWAY MODULE, NOT THIS MODEL. Registering a grid on a loaded module
    # would change what the assertions above just measured.
    probe = torch.nn.Module()
    grid_name = SHARD_GRID_ATTRIBUTES[0]
    probe.register_parameter(
        grid_name, torch.nn.Parameter(torch.zeros(1, 1), requires_grad=False)
    )
    setattr(probe, SHARD_GRID_ATTRIBUTES[1], torch.zeros(1, 1))
    probe_parameters = set(dict(probe.named_parameters()))
    registered = [n for n in SHARD_GRID_ATTRIBUTES if n in probe_parameters]
    attached = [
        n
        for n in SHARD_GRID_ATTRIBUTES
        if getattr(probe, n, None) is not None and n not in probe_parameters
    ]
    print(f"CONJUNCT4_N4_CONTROL_GRIDS_IN_NAMED_PARAMETERS={registered}")
    print(f"CONJUNCT4_N4_CONTROL_GRIDS_AS_PLAIN_ATTRIBUTES={attached}")
    assert registered == [grid_name], (
        f"the membership predicate read {registered} on a module where "
        f"{grid_name!r} IS a registered parameter; it cannot detect a registered "
        f"grid, so the empty list the bankscale scan reads is not evidence of "
        f"anything"
    )
    assert attached == [SHARD_GRID_ATTRIBUTES[1]], (
        f"the predicate read {attached} as plain attributes; a grid set with "
        f"setattr must NOT appear in named_parameters(), which is the convention "
        f"the bankscale zero is asserting"
    )

    # ── THE ROUTED BANK NOW LOADS AT RANK 1, AND THE MAP IS WHAT MOVED ────────
    # ``inc-glm53f-101``, replacing the reading ``-094`` left here. That reading
    # asserted the OPPOSITE -- a rank-1 load of the routed fixture refused with
    # "outside the partition" -- and named this increment as the owner of its
    # revision in its own words: "If it now completes, the bank's rank derivation
    # was fixed and this reading ... should be revisited together with
    # inc-glm53f-101". This is that revision. The behaviour changed, so the reading
    # changed with it; the old text is quoted here so the repair is legible rather
    # than silently absent.
    #
    # WHAT THE DEFECT WAS. ``Glm5NextRoutedExperts`` keeps two degrees:
    # ``tp_degree``, the tensor-parallel world size, and ``ep_degree``, the
    # expert-parallel degree the bank's partition is built from
    # (``model_fp8.py:1029-1037``). This campaign's production expert-parallel
    # degree is 1 (``factory.py:204-211``), so the partition holds ONE rank while
    # the load supplies the GLOBAL rank -- and every global rank above 0 was
    # refused (``factory.py:132-137``).
    #
    # WHAT THE REPAIR IS. ``_expert_parallel_rank_map`` maps the global rank to the
    # rank the partition was built over BEFORE the owner is asked which experts are
    # local. At degree 1 that map is the constant 0, which is what the package
    # itself declares the degree to mean in two places
    # (``parallel/neuron_parallel_state.py:1202-1206`` and
    # ``factory.py:260-264``): the bank is local on every rank.
    bank_rank1 = _load_at_world(directory, SHARD_WORLD, 1, monkeypatch, with_bank)
    banks = [
        (path, module)
        for path, module in bank_rank1.named_modules()
        if type(module).__name__ == "Glm5NextRoutedExperts"
    ]
    print(f"CONJUNCT4_BANK_MODULES_AT_RANK1={[p for p, _ in banks]}")
    assert banks, (
        "this configuration built no Glm5NextRoutedExperts module, so a rank-1 "
        "bank load is not what was just measured and the reading has no subject"
    )
    loaded_at_rank1 = dict(bank_rank1.named_parameters())
    leading: list[tuple[str, int, int]] = []
    for path, module in banks:
        for leaf in getattr(module, "declared_param_names", ()):
            dotted = f"{path}.{leaf}"
            if dotted not in loaded_at_rank1 or loaded_at_rank1[dotted].dim() < 2:
                continue
            leading.append(
                (
                    dotted,
                    int(loaded_at_rank1[dotted].shape[0]),
                    int(module.num_local_experts),
                )
            )
    print(f"CONJUNCT4_BANK_EP_DEGREE_AT_RANK1={banks[0][1].ep_degree}")
    print(f"CONJUNCT4_BANK_TP_DEGREE_AT_RANK1={banks[0][1].tp_degree}")
    print(f"CONJUNCT4_BANK_LOCAL_EXPERTS_AT_RANK1={banks[0][1].num_local_experts}")
    print(f"CONJUNCT4_BANK_LEADING_AXES_AT_RANK1={leading}")
    assert len(leading) >= 1, (
        "the rank-1 load completed but no bank parameter with a leading expert "
        "axis was found, so 'the bank loaded' has not been measured"
    )
    assert banks[0][1].ep_degree == 1, (
        f"this route's expert-parallel degree read {banks[0][1].ep_degree}, not 1, "
        f"so the constant-0 map is not the map under test here"
    )
    assert banks[0][1].num_local_experts == MINI_ROUTED_EXPERTS, (
        f"at expert-parallel degree 1 every expert is local, so rank 1 should own "
        f"all {MINI_ROUTED_EXPERTS}; the module declares "
        f"{banks[0][1].num_local_experts}"
    )
    for dotted, axis, expected in leading:
        assert axis == expected, (
            f"{dotted} loaded a leading axis of {axis} at rank 1 while its module "
            f"declares {expected} local experts, so the rank got a different "
            f"number of experts than the partition assigns it"
        )

    # THE CONTROL THAT MOVES. The partition's own bound check is UNCHANGED -- what
    # the repair changed is which rank reaches it. Asking the partition directly
    # for rank 1 at degree 1 must still refuse with the same message the old
    # reading asserted, so this reading distinguishes "the map was fixed" from
    # "the check was deleted", which is the way this repair could have been faked.
    with pytest.raises(ValueError) as still_bounded:
        banks[0][1].expert_partition.local_expert_indices(1)
    bounded_text = str(still_bounded.value)
    print(f"CONJUNCT4_PARTITION_STILL_BOUNDS_RANK1={bounded_text}")
    assert "outside the partition" in bounded_text, (
        f"the partition no longer refuses an out-of-range rank: {bounded_text}. "
        f"The repair was supposed to map the rank, not remove the bound check, so "
        f"a bank at a real expert-parallel degree would now be placed by guess"
    )
    assert "rank 1" in bounded_text and "1 ranks" in bounded_text, (
        f"the refusal does not name both the rank it was given and the size of "
        f"the partition it was checked against: {bounded_text}"
    )


# --------------------------------------------------------------------------- #
# inc-glm53f-101 -- THE SIX DEFERRED FAMILIES, sharded from the checkpoint's own
# width. Plan revision 232, design entry ``design-20260905-ap`` as amended at
# DECISIONS section 80. Five items, one per conjunct, no ``parametrize``.
#
# WHY THESE SIX NEEDED A BLOCK OF THEIR OWN. ``inc-glm53f-094`` shards a family by
# resolving its per-rank extent when the loader is attached. Neither the shared
# expert nor the routed bank can be served that way: ``Glm5NextSharedExperts``
# holds ``num_shared_experts`` and ``swiglu_limit`` and no width, and
# ``Glm5NextRoutedExperts`` holds counts and degrees. Their width is read off the
# checkpoint tensor instead, which is also the one source that cannot disagree
# with the weights.
#
# AND WHY THE DENSE THREE MOVED WITH THEM. Their old width function floored --
# 12288 // 64 is 192 -- and 192 is neither a whole checkpoint tile nor a whole
# consumer block, so the real model could not load at the registered degree at
# all. They now take the same padded route.
# --------------------------------------------------------------------------- #

#: The shared expert's intermediate width. A multiple of ``4 x 256``, so at world
#: size 4 the pad is a NO-OP and this family's reassembly is exact end to end --
#: the padded case is the dense three's, deliberately, so one item reads a pad and
#: another reads its absence.
SHARED_INTERMEDIATE = 2048

#: One routed expert's intermediate width. A multiple of ``tp_per_ep x 256``, for
#: the same reason and with the same intent.
BANK_INTERMEDIATE = 512

#: The world these items load at, and the expert-parallel degree inside it. FOUR
#: rather than two because two cannot tell the bank's divisor apart from the
#: world: at world 4 with ``ep_degree`` 2 the bank divides by 2 and the shared
#: expert by 4, so a loader that used the wrong one reads a different shape.
SHARD_EP_WORLD = 4
SHARD_EP_DEGREE = 2
SHARD_TP_PER_EP = SHARD_EP_WORLD // SHARD_EP_DEGREE

#: ``(declaring class, declared leaf) -> (shard dim, full extent)`` for the SIX.
#: Stated here rather than read from the code's table, for the reason
#: :data:`SHARD_FAMILIES` states: the code under test and the expectation must not
#: share one source.
DEFERRED_FAMILIES: dict[tuple[str, str], tuple[int, int]] = {
    ("Glm5NextSharedExperts", "gate_proj_weight"): (0, SHARED_INTERMEDIATE),
    ("Glm5NextSharedExperts", "up_proj_weight"): (0, SHARED_INTERMEDIATE),
    ("Glm5NextSharedExperts", "down_proj_weight"): (1, SHARED_INTERMEDIATE),
    ("Glm5NextRoutedExperts", "gate_proj_weight"): (0, BANK_INTERMEDIATE),
    ("Glm5NextRoutedExperts", "up_proj_weight"): (0, BANK_INTERMEDIATE),
    ("Glm5NextRoutedExperts", "down_proj_weight"): (1, BANK_INTERMEDIATE),
}

#: The three classes, split by which rank count divides them. The bank is the one
#: family whose divisor is not the world size.
DEFERRED_WHOLE_WORLD_CLASSES = ("Glm5NextDenseMLP", "Glm5NextSharedExperts")
DEFERRED_EP_GROUP_CLASSES = ("Glm5NextRoutedExperts",)


#: The extent no family in THIS fixture shards, and the reason it is 256 rather
#: than :data:`SHARD_NARROW`'s 8. These items load a model that HAS a shared
#: expert, so the load path runs the landed
#: ``Glm5NextSharedExperts.prepare_scale_operands`` (``model_fp8.py:1857``), which
#: goes through the consumer's own ``scale_grid_shape`` -- and that function
#: refuses an extent that is not a whole number of 256 x 256 blocks on EITHER
#: dimension (``functional/blockwise_fp8_mm.py:284``). At narrow width 8 every one
#: of these items died there, measured at ``accept-101-r8-host.out``
#: (``weight extent [512,8] is not a whole number of 256x256 blocks``). 256 is the
#: smallest width the consumer admits, ruled at DECISIONS §83 ruling 1.
#:
#: A NEW NAME, NOT A REBINDING. :data:`SHARD_NARROW` stays 8 and stays
#: ``inc-glm53f-094``'s constant; that fixture has no shared expert and needs no
#: consumer-valid width. Two fixtures, two widths, each stated where it is used.
DEFERRED_NARROW = 256


def _deferred_full_shape(
    family: str, leaf: str, shard_dim: int, full: int
) -> tuple[int, ...]:
    """:func:`_shard_full_shape`'s form at this fixture's own narrow width.

    ``family`` is a parameter for the reason :func:`_shard_full_shape` gives: a
    leaf name alone does not identify a family, and ``o_proj_weight`` is declared
    by both ``Glm5NextKDAAttention`` and ``Glm5NextMLAAttention``. The three MLA
    head-width families declare a real other extent in
    :data:`SHARD_OTHER_EXTENT` and take it; every other family falls back to
    :data:`DEFERRED_NARROW`, which is what the six deferred families do.
    """
    if leaf in SHARD_ONE_DIMENSIONAL:
        return (full,)
    other = SHARD_OTHER_EXTENT.get((family, leaf), DEFERRED_NARROW)
    return (full, other) if shard_dim == 0 else (other, full)


def _padded_shard_extent(full: int, num_shards: int, block: int) -> int:
    """One rank's extent after the width is rounded up to ``num_shards x block``.

    The expectation's OWN arithmetic, written here from the rule in plain words
    rather than imported from the loader, so the two can disagree.
    """
    step = num_shards * block
    return (math.ceil(full / step) * step) // num_shards


def _deferred_config(shared_experts: int = MINI_SHARED_EXPERTS) -> Glm5NextConfig:
    """:func:`_shard_config`'s fixture with the shared expert switched ON.

    ``n_shared_experts`` is 1 here where :func:`_shard_config` sets 0, and that is
    the whole difference. ``-094`` set it to 0 because the shared expert's three
    families were REPLICATED-IN-EFFECT at that increment and its own conjunct (4)
    said so; this block attaches them, so they have to be observed by a real load.

    ``shared_experts`` IS THE FIRING CONTROL'S ONE VARYING FIELD. At 0 the load
    completes and records no refusal, which is what makes the recorded gap below a
    reading of the shared expert's prep rather than of the fixture at large
    (DECISIONS §84 ruling (ii)).
    """
    return Glm5NextConfig(
        text_config=Glm5NextTextConfig(
            num_hidden_layers=MINI_LAYERS,
            n_routed_experts=MINI_ROUTED_EXPERTS,
            n_shared_experts=shared_experts,
            first_k_dense_replace=MINI_FIRST_K_DENSE,
            tie_word_embeddings=False,
            linear_attn_config=SHARD_LINEAR_ATTN,
            intermediate_size=SHARD_INTERMEDIATE,
            **MINI_MLA_WIDTHS,
        )
    )


def _deferred_key_overrides(
    model: Glm5NextForConditionalGeneration,
    mappings: dict[str, str | list[str]],
    *,
    ramp_grids: bool = False,
) -> dict[str, torch.Tensor]:
    """FULL tensors for the six deferred families, plus ``-094``'s fifteen.

    A SCALE GRID A RETILED FAMILY WILL LOAD IS WRITTEN POW2 PER 256 BLOCK, and that
    is R6 item R-T2's change here. Every family in :data:`_RETILED_AT_LOAD_CLASSES`
    has whole-256 extents in this fixture, so its load coarsens the grid; on the
    ramp that coarsening rescales weight bytes, which contradicts the bit-exact
    readings below and now REFUSES where a byte leaves the fp8 range (R6 item
    R-P1). :func:`_pow2_block_grid_pattern` carries the reasoning and what it gives
    up. Every other family keeps the ramp: nothing rescales its weights.

    ``ramp_grids=True`` writes the ramp for EVERY family, which is the fixture the
    refusal item needs and the only caller that asks for it.

    A bank entry is E weight keys and E scale keys interleaved, so its arm writes
    one tensor per expert at that expert's own full width -- the loader's job is to
    take a column of each, and a bank written at one shared shape could not tell a
    column error from an expert error.

    Nothing here spells a checkpoint key: the keys come from the map and the shapes
    from :data:`DEFERRED_FAMILIES`, the same discipline
    :func:`_shard_key_overrides` follows.

    EVERY FAMILY THIS FIXTURE WRITES TAKES :data:`DEFERRED_NARROW` UNLESS
    :data:`SHARD_OTHER_EXTENT` DECLARES ITS OWN. ``inc-glm53f-107`` added the three
    MLA rows there -- ``q_b_proj_weight``, ``kv_b_proj_weight`` and
    ``o_proj_weight``, at their own declared extents rather than the 256 fallback --
    so the older sentence, that every family takes ONE narrow width, stopped being
    true at that commit and is corrected here. The reason the OTHER families still
    share one width is unchanged: ``inc-glm53f-094``'s writer is not reused for the
    fifteen, because a fixture holding two narrow widths for families whose refusal
    must stay legible would give the consumer's grid check a different answer per
    family and the reason for a refusal would stop being readable. The
    fifteen keep their SHARD extents and their dims exactly as
    :data:`SHARD_FAMILIES` states them -- only the extent no family shards changes.
    """
    overrides: dict[str, torch.Tensor] = {}
    every_family = {**SHARD_FAMILIES, **DEFERRED_FAMILIES}
    for path, module in model.named_modules():
        cls = type(module).__name__
        for (family, leaf), (shard_dim, full) in every_family.items():
            if cls != family:
                continue
            param = f"{path}.{leaf}"
            if param not in mappings:
                continue
            keys = _keys_of(mappings, param)
            scales = scale_keys(keys)
            weights = [key for key in keys if key not in scales]
            shape = _deferred_full_shape(family, leaf, shard_dim, full)
            if not scales:
                # The map answers "is this family quantised", not a name list here.
                for key in weights:
                    overrides[key] = _shard_pattern(shape, shard_dim, torch.bfloat16)
                continue
            grid_shape = block_grid_shape(shape, DEFAULT_WEIGHT_BLOCK_SIZE)
            for key in weights:
                overrides[key] = _shard_pattern(
                    shape, shard_dim, torch.float8_e4m3fn
                )
            coarsened = not ramp_grids and family in _RETILED_AT_LOAD_CLASSES
            for key in scales:
                overrides[key] = (
                    _pow2_block_grid_pattern(grid_shape, shard_dim)
                    if coarsened
                    else _shard_pattern(grid_shape, shard_dim, torch.float32)
                )
    return overrides


def _deferred_checkpoint(
    tmp_path: Path, *, ramp_grids: bool = False, name: str = "deferred"
) -> tuple[Path, dict, dict]:
    """One checkpoint holding every full tensor these five items read.

    ``ramp_grids`` and ``name`` exist for R6 item R-T2's refusal item alone: it
    needs the SAME checkpoint with the ramp scale grid restored, written beside this
    one rather than over it, so the two loads in that item differ in exactly the one
    field it varies.
    """
    config = _deferred_config()
    mappings = _mappings_for(config)
    reference = Glm5NextForConditionalGeneration(config)
    overrides = _deferred_key_overrides(reference, mappings, ramp_grids=ramp_grids)
    directory = tmp_path / name
    _write_miniature_checkpoint(
        directory, mappings, reference, extra_overrides=overrides
    )
    return directory, overrides, mappings


class _FixtureGroup:
    """The two fields this file's code reads off a ``GroupCoordinator``.

    Not a stand-in for the real class: the two attributes are the two the loader
    asks for -- ``rank_in_group`` (``parallel/neuron_parallel_state.py:1206``) and
    ``world_size`` (``:1199``) -- and their VALUES come from the package's own
    ``_build_ep_group_ranks`` in every item below, never from a number typed here.
    """

    def __init__(self, rank_in_group: int, world_size: int) -> None:
        self.rank_in_group = rank_in_group
        self.world_size = world_size


def _mesh_answers(world_size: int, ep_degree: int, rank: int) -> tuple[int, int]:
    """``(ep_rank, column)`` for one global rank, from the PACKAGE's own mesh.

    ``_build_ep_group_ranks`` returns ``(ep_tp_groups, ep_groups)`` -- rows then
    columns (``:218-232``). A rank's EP index is which ROW it is in, because the EP
    group is the column group and ``get_neuron_ep_rank`` reads its position there
    (``:1202-1206``); its shard column is its POSITION IN THAT ROW. Both come from
    the same call, so this helper cannot describe a mesh the package does not build.
    """
    rows, _columns = _NPS._build_ep_group_ranks(world_size, ep_degree)
    for row_index, row in enumerate(rows):
        if rank in row:
            return row_index, row.index(rank)
    raise AssertionError(
        f"global rank {rank} is in none of the {len(rows)} EP-TP rows the package "
        f"built for world {world_size} at expert-parallel degree {ep_degree}"
    )


class _DeferredLoad(NamedTuple):
    """One load's outcome: the model with its shards attached, and the prep's count.

    ``prepared`` is how many scale operands the shared expert's prep built -- 3 on
    every completing load that carries one -- and ``None`` for the firing control
    that carries no shared expert. Both are readings; neither is a failure of this
    file.

    ``inc-glm53f-054a`` REPLACED THE FIELD THIS TUPLE CARRIED, and the replacement
    was named in advance. It was ``refusal``: the ``BlockwiseFp8MmError`` text from
    a prep that could not read a checkpoint-tile grid, because nothing retiled it.
    :func:`_load_at_ep`'s own comment said that when ``-054`` landed the retile
    this reading would flip to "the prep built 3" and the capture would become a
    completing load. Item (iv) landed it, so it did.
    """

    model: Glm5NextForConditionalGeneration
    prepared: int | None


def _load_at_ep(
    directory: Path,
    world_size: int,
    rank: int,
    ep_degree: int,
    monkeypatch,
    shared_experts: int = MINI_SHARED_EXPERTS,
) -> _DeferredLoad:
    """Load at a synthetic world size, rank AND expert-parallel degree.

    THE EP ANSWERS COME FROM THE PACKAGE'S MESH, not from arithmetic here. The two
    getters the loader reads are patched to report what ``_build_ep_group_ranks``
    says for this rank, so a test that agreed with a wrong loader would have to
    disagree with the shipped mesh builder to do it.

    THE MODEL COMES BACK EVEN WHEN THE PREP REFUSES, and that is the point: the
    shards are attached before the prep runs, so every shape and byte these items
    read is on the module the real ``load_weights`` populated.
    """
    ep_rank, column = _mesh_answers(world_size, ep_degree, rank)
    monkeypatch.setattr(_MODEL_FP8, "_resolve_world_size", lambda: world_size)
    monkeypatch.setattr(_MODEL_FP8, "_resolve_rank", lambda: rank)
    monkeypatch.setattr(_FACTORY, "_resolve_ep_degree", lambda given: ep_degree)
    monkeypatch.setattr(_NPS, "get_neuron_ep_rank", lambda: ep_rank)
    monkeypatch.setattr(
        _NPS,
        "get_neuron_ep_tp_group",
        lambda: _FixtureGroup(column, world_size // ep_degree),
    )
    model = Glm5NextForConditionalGeneration(_deferred_config(shared_experts))
    assert model.world_size == world_size, (
        f"the model resolved world size {model.world_size}, not the patched "
        f"{world_size}"
    )
    _seed_page_cache_signal()
    if shared_experts == 0:
        # THE FIRING CONTROL. Same fixture, same narrow width, no shared expert --
        # so the load runs to the end and there is no prep to count. Without this
        # side the reading below would be the same whether the prep built three
        # operands or the fixture simply loaded with nothing to prepare.
        model.load_weights(str(directory), torch.device("cpu"), None)
        return _DeferredLoad(model, None)

    # THE GAP IS CLOSED, AND THE SAME ARITHMETIC READS IT EITHER WAY. Until
    # ``inc-glm53f-054a`` item (iv), nothing on the shared expert's load path
    # retiled its scale grid from the checkpoint's ``(128, 128)`` tiles onto the
    # 256-granularity PUBLIC grid the landed ``prepare_scale_operands`` demands, so
    # this load attached every shard and then refused inside the prep. DECISIONS §84
    # placed that retile in this block; the comment that stood here named the flip
    # in advance -- "the prep built 3", and the capture becomes a completing load.
    #
    # THE TWO GRIDS BELOW ARE UNCHANGED and are still this file's own arithmetic.
    # They used to be the two the refusal had to name: the grid the loader attached
    # and the grid the prep wanted. They are now the grid the retile STARTED from
    # and the grid the module ARRIVED at, so the same two numbers read the fix that
    # read the defect, and a retile that published the wrong granularity fails here
    # rather than passing quietly.
    model.load_weights(str(directory), torch.device("cpu"), None)

    block = _WL_FP8.consumer_block_quant_size()
    rows = _padded_shard_extent(SHARED_INTERMEDIATE, world_size, block)
    cols = DEFERRED_NARROW
    tile_grid = (
        rows // DEFAULT_WEIGHT_BLOCK_SIZE[0],
        cols // DEFAULT_WEIGHT_BLOCK_SIZE[1],
    )
    public_grid = (rows // block, cols // block)

    shared = [
        (path, module)
        for path, module in model.named_modules()
        if type(module).__name__ == "Glm5NextSharedExperts"
    ]
    assert shared, (
        f"this load was asked for {shared_experts} shared experts and built no "
        f"Glm5NextSharedExperts module, so there is no prep here to read"
    )

    # The prep's loop takes gate_proj first, so gate_proj is the projection the
    # refusal used to name and the one read here, for continuity.
    built: set[int] = set()
    for path, module in shared:
        prepared = getattr(module, Glm5NextSharedExperts.PREPARED_SCALE_OPERANDS_ATTR)
        built.add(len(prepared))
        health = getattr(module, Glm5NextSharedExperts.SHARED_RETILE_HEALTH_ATTR)
        record = health["gate_proj_weight"]
        assert record["retiled"] is True, (
            f"{path}.gate_proj_weight was not retiled: {record.get('reason')}. At "
            f"[{rows},{cols}] both extents are whole {block} blocks, so a skip "
            f"here means the retile could not read the extents it was given"
        )
        assert tuple(record["checkpoint_grid"]) == tile_grid, (
            f"{path} retiled from grid {tuple(record['checkpoint_grid'])}, not the "
            f"checkpoint-tile grid {tile_grid} this world size produces"
        )
        assert tuple(record["public_grid"]) == public_grid, (
            f"{path} published grid {tuple(record['public_grid'])}, not the public "
            f"grid {public_grid} the prep demands at [K={rows}, N={cols}]"
        )
        # WHAT THE MODULE ARRIVES AT IS THE PUBLIC GRID TRANSPOSED, and this is the
        # one reading in this helper that ``inc-glm53f-054a`` repair round 1 moved.
        # The retile still publishes ``public_grid`` -- that is asserted three lines
        # up, off its own health record -- and the republish that FOLLOWS the retile
        # then turns the weight and its grid together into the frame
        # ``blockwise_fp8_mm`` multiplies in. So the attribute the prep reads holds
        # the reversed pair. Asserting the reversed pair keeps the reading
        # falsifiable: a retile that published the wrong granularity still fails
        # here, and so does a republish that moved one of the pair without the other.
        compute_grid = tuple(reversed(public_grid))
        grid = getattr(module, f"gate_proj_{FP8_SCALE_SUFFIX}")
        assert tuple(grid.shape) == compute_grid, (
            f"{path}.gate_proj_{FP8_SCALE_SUFFIX} is {tuple(grid.shape)} on the "
            f"module after the load, not the {compute_grid} the republish leaves "
            f"(the published {public_grid} transposed); the retile has to replace "
            f"the attribute the prep reads, not a copy of it"
        )
        weight = _loaded(model, f"{path}.gate_proj_weight")
        assert tuple(weight.shape) == (cols, rows), (
            f"{path}.gate_proj_weight is {tuple(weight.shape)} after the load; the "
            f"loader delivers [{rows},{cols}] and the republish has to leave the "
            f"[K={cols}, N={rows}] frame the seam contracts on, or the prepared "
            f"scale operand above was built from the other frame"
        )

    assert built == {3}, (
        f"the shared experts built {sorted(built)} scale operands, not 3 each. "
        f"Three projections, one operand apiece, and the load completed -- so a "
        f"shortfall means a projection was skipped rather than refused"
    )

    # THE DENSE MLP IS ENROLLED IN THE SAME REPUBLISH, and a REAL load is the only
    # place that can say so. ``inc-glm53f-054a`` repair round 1 gave that class the
    # method, and ``_run_load_time_preps`` finds it by
    # ``hasattr(type(module), "retile_checkpoint_scale_grids")`` -- so the health
    # record below exists only if the enrolment fired on this load. A conjunct that
    # called the method directly would prove the method and not the enrolment, which
    # is the blindness the review found in the first place.
    #
    # THE FRAME FLIP IS READ FROM THE RECORD AND NOT FROM A SHAPE, deliberately: at
    # this fixture's world size the dense per-rank weight is SQUARE, so its transpose
    # is invisible in its shape and a shape assertion here would pass either way.
    dense_frames = 0
    for path, module in model.named_modules():
        if type(module).__name__ != "Glm5NextDenseMLP":
            continue
        health = getattr(module, module.DENSE_RETILE_HEALTH_ATTR, None)
        assert health is not None, (
            f"{path} carries no republish health record after a real load, so the "
            f"load-time prep loop did not reach Glm5NextDenseMLP at all -- the "
            f"dense route is back to refusing the loader's frame at layer 0"
        )
        for leaf, record in health.items():
            assert record.get("transposed") is True, (
                f"{path}.{leaf} was not republished into the compute frame: "
                f"{record}"
            )
            assert tuple(record["compute_frame"]) == tuple(
                reversed(tuple(record["loader_frame"]))
            ), (
                f"{path}.{leaf} went from {record['loader_frame']} to "
                f"{record['compute_frame']}, which is not that pair transposed"
            )
            dense_frames += 1
    print(f"DEFERRED_DENSE_REPUBLISHED_PROJECTIONS={dense_frames}")
    return _DeferredLoad(model, 3)


def _deferred_leaves(
    model: Glm5NextForConditionalGeneration,
) -> list[tuple[str, torch.nn.Module, str, int, int]]:
    """Every ``(path, module, leaf, shard_dim, full)`` among the SIX."""
    found: list[tuple[str, torch.nn.Module, str, int, int]] = []
    for path, module in model.named_modules():
        cls = type(module).__name__
        declared = getattr(module, "declared_param_names", ())
        for (family, leaf), (shard_dim, full) in DEFERRED_FAMILIES.items():
            if cls == family and leaf in declared:
                found.append((path, module, leaf, shard_dim, full))
    return found


# --------------------------------------------------------------------------- #
# (1) SHAPES -- every deferred family lands at its declared per-rank extent.
# --------------------------------------------------------------------------- #


def test_sharedshard_every_deferred_family_lands_at_its_declared_per_rank_shape(
    tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """Conjunct (1), counted N/N, with the world-size-1 load as the moving control.

    THREE DIVISORS IN ONE READING, which is why the world is 4 and not 2. The dense
    three divide by the world size and pad, the shared three divide by the world
    size and do not, and the bank divides by ``tp_per_ep`` -- half the world -- and
    holds half the experts. At world size 2 with expert-parallel degree 1 all three
    collapse to the same number and a loader using the wrong divisor would pass.

    The expected extent is this file's own arithmetic
    (:func:`_padded_shard_extent`), written from the rule in words, never asked of
    the loader.
    """
    directory, overrides, mappings = _deferred_checkpoint(tmp_path)
    block = _WL_FP8.consumer_block_quant_size()
    print(f"CONJUNCT1D_CONSUMER_BLOCK={block}")
    print(f"CONJUNCT1D_WORLD={SHARD_EP_WORLD} EP_DEGREE={SHARD_EP_DEGREE}")
    print(f"CONJUNCT1D_TP_PER_EP={SHARD_TP_PER_EP}")

    loads = {
        rank: _load_at_ep(
            directory, SHARD_EP_WORLD, rank, SHARD_EP_DEGREE, monkeypatch
        )
        for rank in range(SHARD_EP_WORLD)
    }
    # THE PREP'S OWN COUNT is read on every rank before anything else: each load
    # attached its shards, the load-path retile published the public grid, and the
    # -054a-owned prep built three operands. :func:`_load_at_ep` checks that grid
    # against this file's arithmetic; here the count is only counted, so an item
    # cannot read shards from a load that took some other path to completing.
    models = {rank: load.model for rank, load in loads.items()}
    prepped = [rank for rank, load in loads.items() if load.prepared == 3]
    print(f"CONJUNCT1D_RANKS_WHOSE_PREP_BUILT_THREE={sorted(prepped)}")
    print(f"CONJUNCT1D_PREPARED_OPERANDS_PER_RANK={loads[0].prepared}")
    assert sorted(prepped) == sorted(models), (
        f"only ranks {sorted(prepped)} built the shared expert's three scale "
        f"operands, of {sorted(models)}. A rank that built none either never "
        f"reached the prep or the load-path retile did not publish its grid"
    )
    whole_load = _load_at_ep(directory, 1, 0, 1, monkeypatch)
    whole = whole_load.model
    assert whole_load.prepared == 3, (
        f"the world-size-1 load built {whole_load.prepared} scale operands, not "
        f"3, so the prep's success is a property of the sharded widths rather "
        f"than of the retile that publishes the grid at any width"
    )

    # THE FIRING CONTROL (DECISIONS §84 ruling (ii)). The same fixture at the same
    # narrow width with NO shared expert loads to the end and has no prep to run, so
    # the readings above are of the shared expert's prep and not of a fixture that
    # would load either way.
    control = _load_at_ep(
        directory, SHARD_EP_WORLD, 0, SHARD_EP_DEGREE, monkeypatch, shared_experts=0
    )
    print(f"CONJUNCT1D_CONTROL_PREPARED={control.prepared}")
    assert control.prepared is None, (
        f"the no-shared-expert control reported {control.prepared} prepared "
        f"operands. Then the count above is not the shared expert's prep and the "
        f"reading is misnamed"
    )
    control_shards = _sharded_leaves(control.model) + [
        row
        for row in _deferred_leaves(control.model)
        if type(row[1]).__name__ in DEFERRED_EP_GROUP_CLASSES
    ]
    print(f"CONJUNCT1D_CONTROL_FAMILIES_LOADED={len(control_shards)}")
    assert control_shards, (
        "the control load attached no sharded family, so it is not a load of the "
        "same fixture and cannot control anything"
    )

    expected_local_experts = MINI_ROUTED_EXPERTS // SHARD_EP_DEGREE
    checked = 0
    readings: list[str] = []
    for rank, model in models.items():
        for path, module, leaf, shard_dim, full in _deferred_leaves(model):
            cls = type(module).__name__
            dotted = f"{path}.{leaf}"
            # IN THE LOADER'S FRAME: ``expected`` below is this file's own rule about
            # the CHECKPOINT's extents divided by a world size, and the two
            # republished classes no longer store that layout.
            got = tuple(
                _as_the_loader_left_it(module, leaf, _loaded(model, dotted)).shape
            )
            if cls in DEFERRED_EP_GROUP_CLASSES:
                # A bank carries a LEADING expert axis, so its declared dim moves
                # one place right and the leading extent is this EP rank's experts.
                #
                # THE BANK PADS NOTHING AS OF ``inc-glm53f-106``. Its three routed
                # rows declare ``require_consumer_block`` instead of
                # ``pad_to_consumer_block``, so the per-rank extent is a plain
                # division and an inadmissible one is REFUSED rather than rounded
                # up. The expectation is changed here to say that, because a
                # padding rule left in place for a family that no longer pads is a
                # second writer of the same number -- which is the defect class
                # review B90-101 opened.
                #
                # THE ``family`` ARGUMENT BELOW IS ``inc-glm53f-107``'s AND IS
                # CARRIED THROUGH THIS REBASE DELIBERATELY. ``-107`` gave
                # :func:`_deferred_full_shape` a leading ``family`` parameter
                # because a leaf name alone does not identify a family, and this
                # increment's own edit sits on the lines either side of that call.
                # The two changes are orthogonal -- ``-107`` fixes WHICH other
                # extent the fixture reads, this increment fixes WHETHER the bank
                # pads -- so both survive, and dropping the argument would
                # ``TypeError`` rather than fail an assertion.
                per_rank = full // SHARD_TP_PER_EP
                padded = _padded_shard_extent(full, SHARD_TP_PER_EP, block)
                print(
                    f"CONJUNCT1D_BANK_RULE unpadded={per_rank} padded_would_be="
                    f"{padded} full={full} tp_per_ep={SHARD_TP_PER_EP} block={block}"
                )
                assert per_rank == padded, (
                    f"this fixture's bank divides {full} over {SHARD_TP_PER_EP} "
                    f"ranks to {per_rank}, where the old padding rule says {padded}. "
                    f"They disagree, so switching this expectation to the unpadded "
                    f"rule MOVES a frozen number and is not a seat's change to make "
                    f"(design entry 85) -- stop and report instead of editing it"
                )
                base = list(_deferred_full_shape(cls, leaf, shard_dim, full))
                base[shard_dim] = per_rank
                expected = (expected_local_experts, *base)
                divisor = (
                    f"tp_per_ep {SHARD_TP_PER_EP} with NO padding -- an "
                    f"inadmissible extent is refused, not rounded up"
                )
            else:
                per_rank = _padded_shard_extent(full, SHARD_EP_WORLD, block)
                expected = list(_deferred_full_shape(cls, leaf, shard_dim, full))
                expected[shard_dim] = per_rank
                expected = tuple(expected)
                divisor = (
                    f"world {SHARD_EP_WORLD} after rounding up to a multiple of "
                    f"{block} per rank"
                )
            assert got == expected, (
                f"{dotted} loaded {got} at rank {rank}; this file's rule says "
                f"{expected} -- full extent {full} on dim {shard_dim}, divided by "
                f"{divisor}"
            )
            checked += 1
            if rank == 0:
                readings.append(f"{cls}.{leaf}={got}")
    print(f"CONJUNCT1D_RANK0_SHAPES={sorted(readings)}")
    print(
        f"CONJUNCT1D_PER_RANK_SHAPES_AS_DECLARED={checked}/"
        f"{SHARD_EP_WORLD * len(_deferred_leaves(models[0]))}"
    )
    # ATTACHMENT THEN PREP (DECISIONS §84 ruling (ii)). Every shape above was read
    # off a module whose load COMPLETED, and each rank in ``models`` is a rank in
    # ``prepped`` by the assertion at the top of this item. So the attachment
    # finished and the prep ran on it. Item (2) reads the same shards' BYTES.
    print(
        f"CONJUNCT1D_ATTACHED_ON_PREPARED_RANKS={checked} on ranks "
        f"{sorted(prepped)}"
    )
    assert checked == SHARD_EP_WORLD * len(_deferred_leaves(models[0]))
    # Section 79.1: a subset reading over a measured set also asserts non-empty.
    assert checked > 0, (
        "no deferred family was measured at all, so every assertion above is "
        "vacuously true"
    )

    # THE CONTROL THAT MOVES. At world size 1 the table's reader returns no
    # geometry, so every one of the six holds its whole checkpoint tensor.
    whole_checked = 0
    for path, module, leaf, shard_dim, full in _deferred_leaves(whole):
        dotted = f"{path}.{leaf}"
        # In the loader's frame, for the reason given at the rank loop above.
        got = tuple(
            _as_the_loader_left_it(module, leaf, _loaded(whole, dotted)).shape
        )
        base = list(
            _deferred_full_shape(type(module).__name__, leaf, shard_dim, full)
        )
        expected = (
            (MINI_ROUTED_EXPERTS, *base)
            if type(module).__name__ in DEFERRED_EP_GROUP_CLASSES
            else tuple(base)
        )
        assert got == expected, (
            f"{dotted} loaded {got} at world size 1, not the whole {expected}"
        )
        whole_checked += 1
    print(f"CONJUNCT1D_WHOLE_AT_WORLD_ONE={whole_checked}")
    assert whole_checked > 0, "the world-size-1 control measured nothing"
    assert whole_checked == len(_deferred_leaves(whole))
    assert len(overrides) > 0 and len(mappings) > 0


def test_deferredwidth_the_mla_three_take_their_other_extent_from_the_table() -> None:
    """The three MLA head-width families resolve their real other extent.

    WHY A DIRECT READ RATHER THAN A SHAPE ASSERTION. This fixture's writer and
    its expectations share one function, :func:`_deferred_full_shape`, so a wrong
    width agrees with itself and the per-rank shape assertions in this section
    cannot see it. Reading the function itself is the only place the
    disagreement shows.

    THE KDA CONTROL IS THE FIRING HALF. ``o_proj_weight`` is declared by both
    ``Glm5NextKDAAttention`` and ``Glm5NextMLAAttention``, and only the second
    declares an other extent, so a lookup keyed on the leaf alone would hand the
    KDA row the MLA row's ``hidden_size``. The resolved extent is asserted on its
    own and not only through the whole tuple, because ``_KDA_FULL`` and
    :data:`DEFERRED_NARROW` are both 256 in this fixture and a tuple comparison
    alone could not say which of the two it had read.
    """
    expected_other = {
        ("Glm5NextMLAAttention", "q_b_proj_weight"): MINI_MLA_WIDTHS["q_lora_rank"],
        ("Glm5NextMLAAttention", "kv_b_proj_weight"): MINI_MLA_WIDTHS["kv_lora_rank"],
        ("Glm5NextMLAAttention", "o_proj_weight"): MINI_MLA_WIDTHS["hidden_size"],
        ("Glm5NextKDAAttention", "o_proj_weight"): DEFERRED_NARROW,
    }
    readings: list[str] = []
    others: dict[tuple[str, str], int] = {}
    for (family, leaf), want_other in expected_other.items():
        shard_dim, full = SHARD_FAMILIES[(family, leaf)]
        shape = _deferred_full_shape(family, leaf, shard_dim, full)
        assert len(shape) == 2, (
            f"{family}.{leaf} is not a two-dimensional leaf here, so the other "
            f"extent has no place to be read: {shape}"
        )
        other = shape[1 - shard_dim]
        assert shape[shard_dim] == full, (
            f"{family}.{leaf} put {shape[shard_dim]} on its shard dim {shard_dim} "
            f"rather than the full extent {full}"
        )
        assert other == want_other, (
            f"{family}.{leaf} resolved its other extent to {other}; this file's "
            f"table says {want_other}. Were the lookup keyed on the leaf alone, "
            f"both o_proj_weight rows would take "
            f"{MINI_MLA_WIDTHS['hidden_size']} and this reading would go red"
        )
        others[(family, leaf)] = other
        readings.append(f"{family}.{leaf}={shape}")
    print(f"CONJUNCT1E_DEFERRED_OTHER_EXTENTS={sorted(readings)}")
    assert len(readings) == len(expected_other) > 0

    # AND NOT ALL THROUGH THE FALLBACK. Without this, a function that ignored the
    # table entirely could still satisfy every assertion above on a fixture whose
    # declared widths happened to equal DEFERRED_NARROW.
    fell_back = sorted(
        key
        for key, value in others.items()
        if key[0] == "Glm5NextMLAAttention" and value == DEFERRED_NARROW
    )
    assert not fell_back, (
        f"an MLA row still reads the {DEFERRED_NARROW} fallback instead of its "
        f"declared extent: {fell_back}"
    )


# --------------------------------------------------------------------------- #
# (2) REASSEMBLY -- the shards put back together are the checkpoint's tensor.
# --------------------------------------------------------------------------- #


def test_sharedshard_the_group_reassembles_every_deferred_family_bit_identically(
    tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """Conjunct (2), bit for bit, plus the two refusals the design names.

    REASSEMBLED OVER THE RIGHT AXIS FOR EACH FAMILY. The whole-world three
    concatenate along their shard dim over all four ranks; the bank concatenates
    along its shard dim over the ranks of one EP-TP ROW and along the expert axis
    over the EP rows. Reading a bank along the world instead would double-count
    every expert, which is the error this shape of reading exists to catch.

    A PADDED FAMILY IS COMPARED ON ITS REAL EXTENT ONLY, and the pad itself is item
    (3)'s reading rather than a tolerance here. Comparison is ``max abs diff``
    against exactly 0.0, so nothing is being called close.
    """
    directory, overrides, mappings = _deferred_checkpoint(tmp_path)
    loads = {
        rank: _load_at_ep(
            directory, SHARD_EP_WORLD, rank, SHARD_EP_DEGREE, monkeypatch
        )
        for rank in range(SHARD_EP_WORLD)
    }
    # THE PREP'S OWN COUNT is read on every rank before anything else: each load
    # attached its shards, the load-path retile published the public grid, and the
    # -054a-owned prep built three operands. :func:`_load_at_ep` checks that grid
    # against this file's arithmetic; here the count is only counted, so an item
    # cannot read shards from a load that took some other path to completing.
    models = {rank: load.model for rank, load in loads.items()}
    prepped = [rank for rank, load in loads.items() if load.prepared == 3]
    print(f"CONJUNCT2D_RANKS_WHOSE_PREP_BUILT_THREE={sorted(prepped)}")
    assert sorted(prepped) == sorted(models), (
        f"only ranks {sorted(prepped)} built the shared expert's three scale "
        f"operands, of {sorted(models)}. A rank that built none either never "
        f"reached the prep or the load-path retile did not publish its grid"
    )

    reassembled = 0
    for path, module, leaf, shard_dim, full in _deferred_leaves(models[0]):
        cls = type(module).__name__
        dotted = f"{path}.{leaf}"
        keys = _keys_of(mappings, dotted)
        scales = scale_keys(keys)
        weight_keys = [key for key in keys if key not in scales]
        if cls in DEFERRED_EP_GROUP_CLASSES:
            rows, _cols = _NPS._build_ep_group_ranks(
                SHARD_EP_WORLD, SHARD_EP_DEGREE
            )
            per_expert = []
            for row in rows:
                columns = [_loaded(models[rank], dotted) for rank in row]
                per_expert.append(torch.cat(columns, dim=shard_dim + 1))
            joined = torch.cat(per_expert, dim=0)
            expected = torch.stack(
                [overrides[key] for key in weight_keys]
            )
            trim = [slice(None)] * joined.dim()
            trim[shard_dim + 1] = slice(0, full)
            got = joined[tuple(trim)]
        else:
            # THE PIECES ARE PUT BACK IN THE LOADER'S FRAME BEFORE THEY ARE JOINED.
            # ``shard_dim`` is a dim of the CHECKPOINT's layout, and the two
            # republished classes no longer store that layout, so a cat on the
            # stored dim would join along the wrong axis.
            #
            # THE FRAME IS ALL THIS UNDOES. The republish's other step requantises a
            # 128-tile grid onto the 256 public one for any weight whose extents are
            # whole blocks, and this item's comparison is bit-exact, so it also
            # depends on that coarsening being lossless on THIS fixture's grids. That
            # is a property of the fixture, not of the loader, and the first host run
            # of this file is what settles it -- ``inc-glm53f-054a`` hands that
            # reading forward rather than weakening the equality to hide it.
            got = torch.cat(
                [
                    _as_the_loader_left_it(
                        module, leaf, _loaded(models[rank], dotted)
                    )
                    for rank in range(SHARD_EP_WORLD)
                ],
                dim=shard_dim,
            )
            expected = overrides[weight_keys[0]]
            trim = [slice(None)] * got.dim()
            trim[shard_dim] = slice(0, full)
            got = got[tuple(trim)]
        # The checkpoint's fp8 bytes are squeezed on the way in, so the comparison
        # is against the same squeeze the loader applies rather than raw bytes.
        reference = downscale_fp8_weight_bytes(expected)
        assert got.shape == reference.shape, (
            f"{dotted} reassembled to {tuple(got.shape)}, not the checkpoint's "
            f"{tuple(reference.shape)}"
        )
        diff = _max_abs_diff(got, reference)
        assert diff == 0.0, (
            f"{dotted} reassembled with max abs diff {diff}, not 0.0 -- the shards "
            f"do not put the checkpoint tensor back together"
        )
        reassembled += 1
    print(f"CONJUNCT2D_FAMILIES_REASSEMBLED={reassembled}")
    assert reassembled > 0, "no family was reassembled, so this item read nothing"
    assert reassembled == len(_deferred_leaves(models[0]))

    # REFUSAL ONE -- a ragged expert-parallel degree, refused by the fork's own gate.
    from vllm_neuron.model.glm5_next.factory import (
        RaggedExpertPartitionError,
        require_uniform_expert_partition,
    )

    ragged_degree = MINI_ROUTED_EXPERTS + 1
    assert MINI_ROUTED_EXPERTS % ragged_degree != 0, (
        f"{MINI_ROUTED_EXPERTS} experts divide evenly by {ragged_degree}, so this "
        f"is not the ragged case"
    )
    with pytest.raises(RaggedExpertPartitionError) as ragged:
        require_uniform_expert_partition(MINI_ROUTED_EXPERTS, ragged_degree)
    print(f"CONJUNCT2D_RAGGED_REFUSAL={str(ragged.value)[:160]}")

    # REFUSAL TWO -- an intermediate shard the CONSUMER cannot take.
    block = _WL_FP8.consumer_block_quant_size()
    not_a_whole_block = block + DEFAULT_WEIGHT_BLOCK_SIZE[0]
    assert not_a_whole_block % block != 0
    with pytest.raises(Glm5NextExpertBankNotLoadableError) as unusable:
        _WL_FP8.shard_geometry_for_grid(
            _WL_FP8.ShardGeometry(
                shard_dim=0,
                shard_size=not_a_whole_block,
                num_shards=SHARD_TP_PER_EP,
            ),
            param_name="probe.experts.gate_proj_weight_scale_inv",
        )
    message = str(unusable.value)
    print(f"CONJUNCT2D_CONSUMER_REFUSAL={message[:200]}")
    assert "probe.experts.gate_proj_weight_scale_inv" in message
    assert str(block) in message


# --------------------------------------------------------------------------- #
# (3) PADDING EXACTNESS -- zeros, ones, and a dequantisation that does not move.
# --------------------------------------------------------------------------- #


def test_sharedshard_the_pad_is_zeros_and_ones_and_dequantises_exactly(
    tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """Conjunct (3). The padded ranks hold fp8 zero and grid 1.0, and the pad
    changes no number the model would compute.

    WHY THE DENSE THREE ARE THE SUBJECT. At world 4 the dense intermediate 512 pads
    to 1024, so ranks 2 and 3 hold no real row at all -- the strongest form of the
    padded case. The shared three at 2048 need no pad at this world, and that
    absence is read here too, so "the pad happened" and "the pad did not happen"
    are both measurements rather than one assumption.

    THE EXACTNESS CLAIM IS AN EQUALITY, NOT A TOLERANCE. A padded weight row is fp8
    zero and its grid entry is 1.0, so the row dequantises to exactly 0.0; a zero
    row contributes exactly nothing to the down projection's contraction. So the
    padded emulation must equal the unpadded reference at max abs diff 0.0 and no
    numeric pair is authored.

    ``inc-glm53f-054a`` REPAIR ROUND 1 CHANGED HOW THE MODULE SIDE IS READ, and not
    what is claimed (DECISIONS §706-§707). The claim above is frozen. What moved is
    one assertion that had described the pre-republish design -- that the module's
    grid IS the checkpoint's own rows -- which the republish makes false by design,
    because it coarsens that grid onto the consumer's 256 and turns it together with
    its weight. The module side is now dequantised at the grid the module actually
    carries, in the loader's frame, and the reference stays the checkpoint's own
    unpadded tensors at the raw grid and the checkpoint's 128 granularity: the
    model's semantics, re-implementing no part of the republish. THREE readings were
    added rather than removed: the republish's own losslessness counter is asserted at
    zero for these projections, so a red run names the cause; the convention is read
    at the numbers per projection, with the compensated form as the control that
    moves; and the pad is read once on its own, against the same module tensors with
    the padded ranks left out, so a republish finding and a pad defect cannot be
    mistaken for each other.
    """
    directory, overrides, mappings = _deferred_checkpoint(tmp_path)
    block = _WL_FP8.consumer_block_quant_size()
    loads = {
        rank: _load_at_ep(
            directory, SHARD_EP_WORLD, rank, SHARD_EP_DEGREE, monkeypatch
        )
        for rank in range(SHARD_EP_WORLD)
    }
    # THE PREP'S OWN COUNT is read on every rank before anything else: each load
    # attached its shards, the load-path retile published the public grid, and the
    # -054a-owned prep built three operands. :func:`_load_at_ep` checks that grid
    # against this file's arithmetic; here the count is only counted, so an item
    # cannot read shards from a load that took some other path to completing.
    models = {rank: load.model for rank, load in loads.items()}
    prepped = [rank for rank, load in loads.items() if load.prepared == 3]
    print(f"CONJUNCT3D_RANKS_WHOSE_PREP_BUILT_THREE={sorted(prepped)}")
    assert sorted(prepped) == sorted(models), (
        f"only ranks {sorted(prepped)} built the shared expert's three scale "
        f"operands, of {sorted(models)}. A rank that built none either never "
        f"reached the prep or the load-path retile did not publish its grid"
    )

    dense_paths = sorted(
        {
            path
            for path, module in models[0].named_modules()
            if type(module).__name__ == "Glm5NextDenseMLP"
        }
    )
    print(f"CONJUNCT3D_DENSE_MODULES={dense_paths}")
    assert dense_paths, "the fixture built no dense MLP, so this item reads nothing"

    per_rank = _padded_shard_extent(SHARD_INTERMEDIATE, SHARD_EP_WORLD, block)
    real_ranks = SHARD_INTERMEDIATE // per_rank
    print(f"CONJUNCT3D_PER_RANK={per_rank} REAL_RANKS={real_ranks}")
    assert real_ranks < SHARD_EP_WORLD, (
        f"every one of the {SHARD_EP_WORLD} ranks holds a real row, so this fixture "
        f"constructs no padded rank and the readings below would be vacuous"
    )

    zero_ranks = 0
    for path in dense_paths:
        for leaf in SHARD_DENSE_LEAVES:
            # The shard dim is unpacked and unused here: this item reads the PAD,
            # which item (1) already located on the declared dim. Named with a
            # leading underscore rather than deleted -- the deletion this line
            # replaced sat inside the rank loop below, so the second padded rank
            # re-deleted an already-deleted name.
            _shard_dim, _full = SHARD_FAMILIES[("Glm5NextDenseMLP", leaf)]
            attribute = SHARD_GRID_ATTRIBUTES[SHARD_DENSE_LEAVES.index(leaf)]
            for rank in range(real_ranks, SHARD_EP_WORLD):
                weight = _loaded(models[rank], f"{path}.{leaf}")
                grid = getattr(models[rank].get_submodule(path), attribute)
                as_float = weight.to(torch.float32)
                assert bool((as_float == 0.0).all()), (
                    f"{path}.{leaf} at rank {rank} sits wholly past the real "
                    f"{SHARD_INTERMEDIATE} rows, so every element must be fp8 zero; "
                    f"max abs is {as_float.abs().max().item()}"
                )
                assert bool((grid.to(torch.float32) == 1.0).all()), (
                    f"{path}.{leaf}'s grid at rank {rank} must be all 1.0 past the "
                    f"real rows; it holds values from "
                    f"{grid.min().item()} to {grid.max().item()}"
                )
                zero_ranks += 1
    print(f"CONJUNCT3D_PADDED_RANK_READINGS={zero_ranks}")
    print(f"CONJUNCT3D_PADDED_RANKS_EXPECTED={SHARD_EP_WORLD - real_ranks} per leaf")
    assert zero_ranks > 0, "no padded rank was read"

    # THE SHARED THREE NEED NO PAD AT THIS WORLD, and that is read rather than said.
    shared_per_rank = _padded_shard_extent(SHARED_INTERMEDIATE, SHARD_EP_WORLD, block)
    print(f"CONJUNCT3D_SHARED_PER_RANK={shared_per_rank}")
    assert shared_per_rank * SHARD_EP_WORLD == SHARED_INTERMEDIATE, (
        f"the shared expert's {SHARED_INTERMEDIATE} padded to "
        f"{shared_per_rank * SHARD_EP_WORLD}; this item's premise is that it does not"
    )

    # THE EMULATION. Dequantise each rank's dense shards, put them back in rank
    # order, and run gate * up through down. Then do the same from the checkpoint's
    # own unpadded tensors. The two must agree exactly.
    path = dense_paths[0]
    # READ IN THE LOADER'S FRAME. Since the republish the module stores gate as
    # [H, I_per_rank], so the stored second extent is the per-rank intermediate and
    # not the hidden size this vector has to be as long as.
    hidden = int(_in_the_loader_frame(models[0], f"{path}.gate_proj_weight").shape[1])
    torch.manual_seed(0)
    x = torch.randn(hidden, dtype=torch.float32)

    # WHAT THE MODULE CARRIES SINCE THE REPUBLISH, printed rather than asserted
    # (DECISIONS §706-§707). This block used to assert that the module's grid IS the
    # checkpoint's own rows. That described the pre-republish design and is now false
    # BY DESIGN: the republish coarsens the checkpoint's 128-tile grid onto the
    # consumer's 256 and turns it together with its weight. So the layout is REPORTED
    # here, at both frames and with the block size it implies, and the thing the old
    # equality existed to protect -- that the module's NUMBERS are the checkpoint's
    # numbers at the checkpoint's own convention -- is read at the values further
    # down, where the compensated form is the control that moves.
    _gate_keys = _keys_of(mappings, f"{path}.gate_proj_weight")
    _gate_scale = scale_keys(_gate_keys)[0]
    _raw_grid = overrides[_gate_scale]
    _compensated = compensate_block_scales(_raw_grid).scale_inv
    _module = models[0].get_submodule(path)
    _on_module = getattr(_module, SHARD_GRID_ATTRIBUTES[0])
    _published = _as_the_loader_left_it(_module, SHARD_GRID_ATTRIBUTES[0], _on_module)
    _weight_lf = _in_the_loader_frame(models[0], f"{path}.gate_proj_weight")
    _stored = _loaded(models[0], f"{path}.gate_proj_weight")
    print(f"CONJUNCT3D_MODULE_WEIGHT_AS_STORED={tuple(_stored.shape)}")
    print(f"CONJUNCT3D_MODULE_WEIGHT_IN_THE_LOADER_FRAME={tuple(_weight_lf.shape)}")
    print(f"CONJUNCT3D_MODULE_GRID_AS_STORED={tuple(_on_module.shape)}")
    print(f"CONJUNCT3D_MODULE_GRID_IN_THE_LOADER_FRAME={tuple(_published.shape)}")
    print(f"CONJUNCT3D_CHECKPOINT_GRID={tuple(_raw_grid.shape)}")
    print(f"CONJUNCT3D_CHECKPOINT_GRID_BLOCK={tuple(DEFAULT_WEIGHT_BLOCK_SIZE)}")
    _implied_block = (
        _weight_lf.shape[0] // _published.shape[0],
        _weight_lf.shape[1] // _published.shape[1],
    )
    print(f"CONJUNCT3D_MODULE_GRID_IMPLIED_BLOCK={_implied_block}")
    assert not torch.equal(_raw_grid, _compensated), (
        "compensation is a no-op on this fixture's grid, so the convention control "
        "further down cannot tell the two conventions apart and the reference is "
        "unguarded"
    )

    # THE REPUBLISH'S OWN LOSSLESSNESS COUNTER, read off its health record and
    # asserted at zero for these projections (DECISIONS §706 item 4). The coarsening
    # keeps one scale per 256 block and rescales the other three 128 tiles into it, so
    # it is exact only where each ratio is a power of two; the counter is the
    # republish's own report of how often it was not. Asserting it here means a red
    # run names the CAUSE, not only the moved number, and it is a finding on the
    # landed retile rather than a tolerance to widen.
    inexact: dict[tuple[int, str], int] = {}
    for rank in range(SHARD_EP_WORLD):
        module = models[rank].get_submodule(path)
        health = getattr(module, module.DENSE_RETILE_HEALTH_ATTR, None)
        assert health is not None, (
            f"{path} at rank {rank} carries no republish health record after a real "
            f"load, so the load-time prep loop never reached Glm5NextDenseMLP"
        )
        for leaf, record in health.items():
            if not record.get("retiled"):
                continue
            inexact[(rank, leaf)] = int(record["inexact_rescales"])
    print(f"CONJUNCT3D_RETILED_PROJECTIONS={len(inexact)}")
    print(f"CONJUNCT3D_RETILE_INEXACT_RESCALES={sorted(inexact.values())}")
    assert inexact, (
        f"no dense projection was coarsened at world size {SHARD_EP_WORLD}, so this "
        f"reading is vacuous and the exactness below is not testing the coarsening "
        f"at all"
    )
    _worst_inexact = max(inexact.values())
    assert _worst_inexact == 0, (
        f"the republish rescaled {_worst_inexact} 128 tiles inexactly on this "
        f"fixture, so the coarsening changed weight NUMBERS and not just the layout: "
        f"{sorted(key for key, count in inexact.items() if count)}. That is a finding "
        f"against the landed retile, to be handed back with this count -- never a "
        f"tolerance and never a fixture retuned to pass (DECISIONS §706 item 4)"
    )

    def _dequantised(rank: int, leaf: str) -> torch.Tensor:
        """One rank's shard, dequantised AT THE GRID THE MODULE ACTUALLY CARRIES.

        The block size is derived from the weight and its grid rather than named as a
        constant, because the republish leaves a whole-block weight at the consumer's
        256 granularity and leaves any other extent at the checkpoint's 128. A
        constant would be right for one of those and silently wrong for the other;
        derived, a grid that does not divide its weight is refused by
        ``dequantise_blockwise`` itself.

        Both tensors are put back in the loader's frame first, so what this returns
        is the checkpoint's own layout and the concatenations below still join on the
        checkpoint's declared shard dim.
        """
        module = models[rank].get_submodule(path)
        attribute = SHARD_GRID_ATTRIBUTES[SHARD_DENSE_LEAVES.index(leaf)]
        weight = _as_the_loader_left_it(
            module, leaf, _loaded(models[rank], f"{path}.{leaf}")
        )
        grid = _as_the_loader_left_it(module, attribute, getattr(module, attribute))
        block = (
            weight.shape[0] // grid.shape[0],
            weight.shape[1] // grid.shape[1],
        )
        return dequantise_blockwise(weight, grid, block).to(torch.float32)

    gate_padded = torch.cat(
        [_dequantised(rank, "gate_proj_weight") for rank in range(SHARD_EP_WORLD)],
        dim=0,
    )
    up_padded = torch.cat(
        [_dequantised(rank, "up_proj_weight") for rank in range(SHARD_EP_WORLD)],
        dim=0,
    )
    down_padded = torch.cat(
        [_dequantised(rank, "down_proj_weight") for rank in range(SHARD_EP_WORLD)],
        dim=1,
    )
    h_padded = (gate_padded @ x) * (up_padded @ x)
    y_padded = down_padded @ h_padded

    def _reference(leaf: str, *, uncompensated: bool = False) -> torch.Tensor:
        """The same three tensors whole, with BOTH halves of the trn2 pair applied.

        CORRECTED BY ``inc-glm53f-054c``, and the correction is one word: the grid is
        now compensated on the reference side, because the load path compensates it.
        The trn2 encoding is a matched pair -- the weight bytes are squeezed into the
        240 range and the per-block grid is multiplied by the inverse factor -- and
        ``_publish_compute_frame_operands`` applies the second half at
        ``model_fp8.py:7118``. Before ``-054c`` NOTHING applied it for these two
        classes, so a reference carrying only the squeeze agreed with the product, and
        this reading passed while the effective matrix stood at ``240/448`` of the
        checkpoint's magnitude.

        ``uncompensated=True`` IS NOW THE CONTROL ARM: it is the pre-``-054c`` value,
        the one that half-applied pair produced. It must NOT match, and if it does then
        the compensation is not reaching the grid and this reading cannot tell the fixed
        load path from the broken one (DECISIONS §706-§707).

        WHY THE REFERENCE STILL CARRIES THE SQUEEZE, rather than being the raw
        checkpoint numbers. ``downscale_fp8_weight_bytes`` multiplies by 240/448 and
        casts BACK to fp8, so it re-quantises: the squeezed bytes are not
        ``w * 240/448`` exactly. A raw-checkpoint reference is therefore unreachable at
        the EXACT equality this reading asserts, and loosening that equality to a
        tolerance would give up the bit-exactness that makes the pad readings worth
        having. The un-squeezed comparison belongs where a tolerance is honest and is
        made there instead, over the real loader and the real prep, in
        ``test_scale_compensation_054c.py``.

        The history is kept because it is the same defect twice, in opposite
        directions. The FIRST version of this reference compensated, the instrument
        caught a ratio of exactly ``(448/240) ** 3`` -- one factor per leaf --
        (``probe-101-r11c-pad-repair.out``) and the reference was changed to match the
        product. The product was the thing that was wrong.
        """
        keys = _keys_of(mappings, f"{path}.{leaf}")
        scales = scale_keys(keys)
        weight_key = next(key for key in keys if key not in scales)
        grid = overrides[scales[0]]
        if not uncompensated:
            grid = compensate_block_scales(grid).scale_inv
        return dequantise_blockwise(
            downscale_fp8_weight_bytes(overrides[weight_key]),
            grid,
            DEFAULT_WEIGHT_BLOCK_SIZE,
        ).to(torch.float32)

    h_whole = (_reference("gate_proj_weight") @ x) * (_reference("up_proj_weight") @ x)
    y_whole = _reference("down_proj_weight") @ h_whole

    # WHICH CONVENTION THE MODULE CARRIES, read at the NUMBERS, per projection
    # (DECISIONS §706-§707). Each padded stack's real rows -- real columns, for the
    # down projection -- are compared against the checkpoint's own tensor with BOTH
    # halves of the trn2 pair applied, which must agree exactly, and against the
    # pre-``-054c`` half-applied form, which must NOT: if both agreed, neither reading
    # could tell the conventions apart. The two arms SWAPPED at ``-054c`` because the
    # load path changed, not because the reading did -- the grid is compensated now.
    # This is per projection so that a red run names which one moved.
    stacks = {
        "gate_proj_weight": gate_padded,
        "up_proj_weight": up_padded,
        "down_proj_weight": down_padded,
    }
    conventions: dict[str, tuple[float, float]] = {}
    for leaf, stack in stacks.items():
        whole = _reference(leaf)
        other = _reference(leaf, uncompensated=True)
        real = (
            stack[:, : whole.shape[1]]
            if leaf == "down_proj_weight"
            else stack[: whole.shape[0]]
        )
        raw_diff = _max_abs_diff(real, whole)
        uncompensated_diff = _max_abs_diff(real, other)
        conventions[leaf] = (raw_diff, uncompensated_diff)
        print(f"CONJUNCT3D_{leaf.upper()}_VS_CHECKPOINT_PAIRED={raw_diff}")
        print(f"CONJUNCT3D_{leaf.upper()}_VS_UNCOMPENSATED={uncompensated_diff}")
    for leaf, (raw_diff, uncompensated_diff) in conventions.items():
        assert uncompensated_diff != 0.0, (
            f"{leaf} matches the UNCOMPENSATED form as well as the paired one, so the "
            f"448/240 compensation is not reaching this grid and this item cannot tell "
            f"the fixed load path from the pre-inc-glm53f-054c one"
        )
        assert raw_diff == 0.0, (
            f"{leaf}'s real rows differ from the checkpoint's own tensor by "
            f"{raw_diff} at the checkpoint's own convention. The load path is meant "
            f"to change this tensor's LAYOUT and not its numbers, so a non-zero "
            f"reading here is a finding against the republish -- most likely its 256 "
            f"coarsening requantising a block whose four 128 tiles do not share a "
            f"power-of-two scale ratio. It is never a tolerance to widen and never a "
            f"fixture to retune (DECISIONS §706 item 4): hand it back with this "
            f"number"
        )

    # THE PAD, ON ITS OWN (DECISIONS §707 item 5). The same module tensors with and
    # without the padded ranks. Both sides come from the module, so whatever the
    # republish did to the numbers cancels and what is left is the pad's own
    # contribution -- which tells a republish finding and a pad defect apart in one
    # run, instead of leaving one to be blamed for the other.
    def _stacked(leaf: str, ranks: range, dim: int) -> torch.Tensor:
        return torch.cat([_dequantised(rank, leaf) for rank in ranks], dim=dim)

    real_only = range(real_ranks)
    h_real = (
        _stacked("gate_proj_weight", real_only, 0) @ x
    ) * (_stacked("up_proj_weight", real_only, 0) @ x)
    y_real = _stacked("down_proj_weight", real_only, 1) @ h_real
    pad_only_diff = _max_abs_diff(y_padded, y_real)
    print(f"CONJUNCT3D_PADDED_VS_REAL_RANKS_ONLY_MAX_ABS_DIFF={pad_only_diff}")
    print(f"CONJUNCT3D_REAL_RANKS_INTERMEDIATE={tuple(h_real.shape)}")
    assert pad_only_diff == 0.0, (
        f"the padded ranks change the output by {pad_only_diff} against the same "
        f"module tensors with those ranks left out, so the pad itself contributes. "
        f"This side of the item is independent of the checkpoint's numbers, so a red "
        f"here is a pad defect and not a republish finding"
    )

    print(f"CONJUNCT3D_PADDED_INTERMEDIATE={tuple(h_padded.shape)}")
    print(f"CONJUNCT3D_WHOLE_INTERMEDIATE={tuple(h_whole.shape)}")
    print(f"CONJUNCT3D_OUTPUT_MAX_ABS_DIFF={_max_abs_diff(y_padded, y_whole)}")
    assert h_padded.shape[0] > h_whole.shape[0], (
        f"the padded intermediate is {h_padded.shape[0]} wide and the unpadded "
        f"{h_whole.shape[0]}; if they matched, no pad was exercised"
    )
    tail = h_padded[h_whole.shape[0] :]
    print(f"CONJUNCT3D_PAD_TAIL_MAX_ABS={tail.abs().max().item()}")
    assert tail.abs().max().item() == 0.0, (
        "the padded intermediate's tail is not exactly zero, so the padded rows are "
        "contributing to the down projection"
    )
    assert _max_abs_diff(y_padded, y_whole) == 0.0, (
        "the padded shards compute a different output from the unpadded reference; "
        "the pad is not exact"
    )


# --------------------------------------------------------------------------- #
# (4) The six LEFT the replicated set, both directions.
# --------------------------------------------------------------------------- #


def test_sharedshard_the_six_families_left_the_replicated_set_both_directions(
    tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """Conjunct (4) re-read at this candidate, both directions and both non-empty.

    ``inc-glm53f-094``'s own conjunct (4) counted these six IN the replicated set
    and said in those words that ``inc-glm53f-101`` would consume them. This is that
    consumption, measured the same way: the set of families that changed between two
    ranks is compared against the set this file declares sharded, in both
    directions, so neither a family that stayed replicated nor one that shards
    without being declared can hide.

    BOTH SETS ARE ASSERTED NON-EMPTY (section 79.1). A subset relation between two
    empty sets holds, and would hold if the load had read nothing at all.
    """
    directory, _overrides, _mappings = _deferred_checkpoint(tmp_path)
    load0 = _load_at_ep(directory, SHARD_EP_WORLD, 0, SHARD_EP_DEGREE, monkeypatch)
    load1 = _load_at_ep(directory, SHARD_EP_WORLD, 1, SHARD_EP_DEGREE, monkeypatch)
    rank0, rank1 = load0.model, load1.model
    recorded = [load0.prepared, load1.prepared]
    print(f"CONJUNCT4D_BOTH_RANKS_PREPARED_OPERANDS={recorded}")
    assert load0.prepared == 3 and load1.prepared == 3, (
        f"the two loads built {recorded} scale operands, not three each, so they "
        f"are not the same kind of load and their difference is not a shard reading"
    )

    declared_sharded = {
        f"{path}.{leaf}"
        for path, _module, leaf, _dim, _full in _deferred_leaves(rank0)
    } | {
        f"{path}.{leaf}"
        for path, _module, leaf, _dim, _full in _sharded_leaves(rank0)
    }
    print(f"CONJUNCT4D_DECLARED_SHARDED={len(declared_sharded)}")
    assert declared_sharded, "this file declares no sharded family, so nothing is read"

    left = dict(rank0.named_parameters())
    right = dict(rank1.named_parameters())
    assert set(left) == set(right), (
        f"the two ranks registered different parameter names, so a family could "
        f"drop out of the comparison unseen. Only at rank 0: "
        f"{sorted(set(left) - set(right))[:3]}; only at rank 1: "
        f"{sorted(set(right) - set(left))[:3]}"
    )
    common = sorted(left)
    print(f"CONJUNCT4D_PARAMETERS_COMPARED={len(common)}")
    assert common, "the two ranks share no parameter name, so no comparison happened"

    moved = {
        name
        for name in common
        if tuple(left[name].shape) != tuple(right[name].shape)
        or _max_abs_diff(left[name].data, right[name].data) != 0.0
    }
    still = set(common) - moved
    print(f"CONJUNCT4D_MOVED={len(moved)} STILL={len(still)}")
    assert moved, "no parameter differs between the two ranks, so nothing is sharded"
    assert still, "every parameter differs, so the replicated set is empty"

    # DIRECTION ONE: every family this file declares sharded actually moved.
    declared_but_still = sorted(name for name in declared_sharded if name in still)
    print(f"CONJUNCT4D_DECLARED_BUT_STILL={declared_but_still[:5]}")
    assert not declared_but_still, (
        f"{len(declared_but_still)} declared-sharded families read identically on "
        f"both ranks, first {declared_but_still[:3]} -- they are still replicated"
    )

    # DIRECTION TWO: nothing moved that this file did not declare sharded.
    moved_but_undeclared = sorted(moved - declared_sharded)
    print(f"CONJUNCT4D_MOVED_BUT_UNDECLARED={moved_but_undeclared[:5]}")
    assert not moved_but_undeclared, (
        f"{len(moved_but_undeclared)} families differ between ranks without being "
        f"declared sharded, first {moved_but_undeclared[:3]}"
    )

    # AND THE SIX SPECIFICALLY, named so a count cannot stand in for them. The
    # coverage check is over the TABLE'S OWN KEYS, not over a length: four layers
    # make more than six dotted names, so a length of six could be reached with
    # one class missing entirely.
    found_keys = {
        (type(module).__name__, leaf)
        for _path, module, leaf, _dim, _full in _deferred_leaves(rank0)
    }
    missing_keys = sorted(set(DEFERRED_FAMILIES) - found_keys)
    six = sorted(
        f"{path}.{leaf}"
        for path, _module, leaf, _dim, _full in _deferred_leaves(rank0)
    )
    print(f"CONJUNCT4D_THE_SIX_DOTTED_NAMES={len(six)}")
    print(f"CONJUNCT4D_TABLE_KEYS_FOUND={len(found_keys)} MISSING={missing_keys}")
    assert not missing_keys, (
        f"the tree holds no parameter for {missing_keys}, so a class the table "
        f"names was never built and this item did not read it"
    )
    assert all(name in moved for name in six), (
        f"one of the six did not move between ranks: "
        f"{[name for name in six if name not in moved][:3]}"
    )


# --------------------------------------------------------------------------- #
# (5) GROUP READ -- the mesh answers, and the disagreement that is refused.
# --------------------------------------------------------------------------- #


def test_sharedshard_the_column_comes_from_the_group_and_refuses_a_disagreement(
    monkeypatch,
) -> None:
    """Conjunct (5). The loader reads the mesh; a division would read something else.

    THE MESH IS THE PACKAGE'S OWN, built by ``_build_ep_group_ranks`` at the real
    registered world size of 64, not a fixture invented here. On this platform that
    call substitutes ``_TRN2_MESH`` whenever the world is 64 and the row is 8
    (``uses_noncontiguous_mesh``), and the whole point of remedy part 3 is that on
    such a mesh neither ``rank // tp_per_ep`` nor ``rank % tp_per_ep`` is the right
    answer.

    THREE READINGS, and the third is what makes the first two mean something:

    1. THE ROW. Global rank 12 sits in row 0, while ``12 // 8`` is 1. The rank map
       returns the group's answer.
    2. THE COLUMN, DISAGREEING. Global rank 4 sits at column 0 of its row, while
       ``4 % 8`` is 4. The column reader REFUSES BY NAME, because the loader would
       otherwise write a column the kernel does not read.
    3. THE REGISTERED DEGREE. At expert-parallel degree 16 the same two answers
       AGREE for every rank, and the column reader returns without refusing. Without
       this the refusal above could be a blanket rather than a boundary.
    """
    world = 64
    disagreeing_degree = 8
    registered_degree = 16
    tp_per_ep = world // disagreeing_degree

    rows, _cols = _NPS._build_ep_group_ranks(world, disagreeing_degree)
    print(f"CONJUNCT5D_ROWS={len(rows)} ROW_SIZE={len(rows[0])}")
    print(f"CONJUNCT5D_ROW_0={rows[0]}")
    assert _NPS.uses_noncontiguous_mesh(world, tp_per_ep), (
        f"the package does not call world {world} with row {tp_per_ep} "
        f"non-contiguous, so this item's premise is gone"
    )

    # 1. THE ROW.
    row_rank, row_column = _mesh_answers(world, disagreeing_degree, 12)
    print(f"CONJUNCT5D_RANK12_GROUP_ROW={row_rank} MODULO_ROW={12 // tp_per_ep}")
    assert row_rank != 12 // tp_per_ep, (
        f"the group puts rank 12 in row {row_rank} and the division also says "
        f"{12 // tp_per_ep}; they agree, so this reading distinguishes nothing"
    )
    monkeypatch.setattr(_NPS, "get_neuron_ep_rank", lambda: row_rank)
    owner = type(
        "Owner", (), {"ep_degree": disagreeing_degree, "tp_degree": world}
    )()
    to_partition_rank = _WL_FP8._expert_parallel_rank_map(
        owner, "probe.experts.gate_proj_weight"
    )
    got_row = to_partition_rank(12)
    print(f"CONJUNCT5D_RANK_MAP_RETURNED={got_row}")
    assert got_row == row_rank, (
        f"the rank map returned {got_row} where the group says {row_rank}"
    )
    assert got_row != 12 // tp_per_ep, (
        f"the rank map returned the divided answer {12 // tp_per_ep}"
    )
    del row_column

    # 2. THE COLUMN, DISAGREEING.
    _row_of_4, column_of_4 = _mesh_answers(world, disagreeing_degree, 4)
    print(
        f"CONJUNCT5D_RANK4_GROUP_COLUMN={column_of_4} "
        f"MODULO_COLUMN={4 % tp_per_ep}"
    )
    assert column_of_4 != 4 % tp_per_ep, (
        f"the group puts rank 4 at column {column_of_4} and the modulo also says "
        f"{4 % tp_per_ep}; they agree, so there is no disagreement to refuse"
    )
    monkeypatch.setattr(
        _NPS,
        "get_neuron_ep_tp_group",
        lambda: _FixtureGroup(column_of_4, tp_per_ep),
    )
    with pytest.raises(Glm5NextExpertBankNotLoadableError) as refusal:
        _WL_FP8._expert_parallel_shard_column(
            4, tp_per_ep, disagreeing_degree, "probe.experts.gate_proj_weight"
        )
    message = str(refusal.value)
    print(f"CONJUNCT5D_DISAGREEMENT_REFUSAL={message[:220]}")
    assert message.startswith("probe.experts.gate_proj_weight "), (
        f"the refusal does not name the parameter first: {message}"
    )
    # The two numbers are asserted IN THEIR PHRASES, not as bare digits: "0" and
    # "4" occur in a file:line cite in the same message, so a substring test on the
    # digit alone would pass on a refusal that named neither column.
    assert f"at column {column_of_4}" in message, (
        f"the refusal does not say which column the group reported: {message}"
    )
    assert f"= {4 % tp_per_ep} " in message or message.rstrip().endswith(
        f"= {4 % tp_per_ep}"
    ), f"the refusal does not say which column the consumer derives: {message}"

    # 3. THE REGISTERED DEGREE, where the two agree and nothing is refused.
    registered_tp_per_ep = world // registered_degree
    agreements = 0
    for rank in range(world):
        row_index, column = _mesh_answers(world, registered_degree, rank)
        if column == rank % registered_tp_per_ep:
            agreements += 1
        del row_index
    print(
        f"CONJUNCT5D_REGISTERED_DEGREE={registered_degree} "
        f"COLUMN_AGREEMENTS={agreements}/{world}"
    )
    assert agreements == world, (
        f"only {agreements} of {world} ranks agree at the registered degree "
        f"{registered_degree}; the registered value would refuse at load"
    )
    _row, registered_column = _mesh_answers(world, registered_degree, 12)
    monkeypatch.setattr(
        _NPS,
        "get_neuron_ep_tp_group",
        lambda: _FixtureGroup(registered_column, registered_tp_per_ep),
    )
    accepted = _WL_FP8._expert_parallel_shard_column(
        12, registered_tp_per_ep, registered_degree, "probe.experts.gate_proj_weight"
    )
    print(f"CONJUNCT5D_REGISTERED_COLUMN_ACCEPTED={accepted}")
    assert accepted == registered_column, (
        f"the column reader returned {accepted} where the group says "
        f"{registered_column} at the registered degree"
    )

    # 4. DEGREE 1, WHERE THERE IS NO GROUP AT ALL. This reading exists because its
    # absence broke a landed item: the first version of the reader asked the group
    # before it asked the degree, so every degree-1 bank load was refused with a
    # message claiming the module declared a degree above 1 when it declared 1
    # (`accept-101-r8-host.out`). The getter is patched to RAISE here, so if the
    # reader touches the group at all this reading fails instead of passing quietly.
    def _the_group_must_not_be_asked():
        raise AssertionError(
            "Neuron EP-TP group is not initialized. "
            "Call initialize_neuron_parallel_state() with ep_degree > 1."
        )

    monkeypatch.setattr(_NPS, "get_neuron_ep_tp_group", _the_group_must_not_be_asked)
    at_degree_one = [
        _WL_FP8._expert_parallel_shard_column(
            rank, world, 1, "probe.experts.gate_proj_weight"
        )
        for rank in (0, 1, 12, 63)
    ]
    print(f"CONJUNCT5D_DEGREE_ONE_COLUMNS={at_degree_one}")
    assert at_degree_one == [0, 1, 12, 63], (
        f"at expert-parallel degree 1 the group is the whole world, so each rank's "
        f"column is its own rank; the reader returned {at_degree_one}"
    )


# =========================================================================== #
# inc-glm53f-105 -- WP7: the sharded projections' FP8 scale grids shard too.
#
# WHY THIS NEEDS ITS OWN WIDTHS. Everything above runs on MINI_MLA_WIDTHS, whose
# q_b_proj is 64 rows -- less than one 128-row quantisation block. A grid of one
# tile describes a rank's half exactly as well as it describes the whole tensor,
# so on that fixture the defect this item measures CANNOT occur and its control
# cannot fire. These widths give the grid a block to lose along each shard
# dimension, which is what makes both readings real, AND leave every per-rank
# shard a whole number of the CONSUMER's blocks, without which the load is
# refused for a reason that has nothing to do with this item.
# =========================================================================== #

#: MLA widths whose scale grids have rows to divide. ``qk_nope_head_dim`` and
#: ``v_head_dim`` are the real checkpoint's per-head width, because TWO block
#: sizes rule here and the smaller one is not the binding one:
#:
#: * ``DEFAULT_WEIGHT_BLOCK_SIZE`` is the 128-row CHECKPOINT tile. It fixes how
#:   many entries a grid holds, so it is what gives a grid anything to divide.
#: * ``consumer_block_quant_size()`` reads ``BLOCK_QUANT_SIZE``, the 256-row
#:   block the block-FP8 kernel indexes its scales by. ``inc-glm53f-101``
#:   already refuses a shard that is not a whole number of THOSE, and that
#:   refusal is upstream of everything this item measures.
#:
#: An earlier version of these widths used 64 and reasoned only about the tile:
#: at world size 2 each rank took 128 rows, which clears the 128-row tile and is
#: refused by the consumer rule, so the item failed on its own fixture instead of
#: on the code it measures. 64 is also a per-head width the production checkpoint
#: cannot have. The item that measures the real ones,
#: ``test_gridshard_b_every_sharded_mla_head_width_is_a_whole_quant_block``,
#: lands in the SAME changeset and reads 256 and up, so a fixture carrying 64
#: asks a question the model never answers.
#:
#: At 256 each rank of two takes two whole consumer blocks of q_b_proj's rows
#: and two of o_proj's columns, and each per-rank grid still holds fewer tiles
#: than the unsharded one, which is the reading this item exists for. Every
#: expected value below is still computed from the loaded shape by
#: ``block_grid_shape``; nothing here is typed into an assertion.
GRID_SHARD_MLA_WIDTHS = dict(
    hidden_size=128,
    num_attention_heads=4,
    qk_nope_head_dim=256,
    qk_rope_head_dim=0,
    v_head_dim=256,
    q_lora_rank=32,
    kv_lora_rank=32,
)

#: The two DSA scaled projections ``inc-glm53f-100`` shards. ``q_a_proj`` and
#: ``kv_a_proj_with_mqa`` are latent-side and replicated; ``kv_b_proj`` is bf16 in
#: this checkpoint and carries no grid at all.
GRID_SHARD_LEAVES = ("q_b_proj", "o_proj")


def _grid_shard_config() -> Glm5NextConfig:
    """:func:`_shard_config` with the wider MLA widths and nothing else changed."""
    return Glm5NextConfig(
        text_config=Glm5NextTextConfig(
            num_hidden_layers=MINI_LAYERS,
            n_routed_experts=MINI_ROUTED_EXPERTS,
            n_shared_experts=0,
            first_k_dense_replace=MINI_ALL_DENSE_FIRST_K,
            tie_word_embeddings=False,
            linear_attn_config=SHARD_LINEAR_ATTN,
            intermediate_size=SHARD_INTERMEDIATE,
            **GRID_SHARD_MLA_WIDTHS,
        )
    )


def _grid_shard_checkpoint(tmp_path: Path) -> Path:
    """A checkpoint written at the CLOSED FORM, grids included.

    ``_mla_key_overrides`` supplies both -- the weight at ``(odim, idim)`` from the
    module's own ``projection_widths()`` and the grid at ``block_grid_shape`` of
    that -- so nothing here states a shape. The reference model is built BEFORE any
    world size is patched, which is what makes those the whole tensors' shapes.

    EVERY OTHER DECLARED-SHARDED FAMILY IS STORED SHARDABLY TOO, and that is this
    fixture's repair. Passing ``grids`` as the only override left every family
    outside the MLA three to the writer's plain branch at
    :data:`MINI_PLAIN_SHAPE`, a 1-D placeholder; the pin's ``sharding_weight_loader``
    then sharded ``model.layers.0.self_attn.o_proj_weight`` on dim 1 and raised
    ``IndexError: list assignment index out of range`` at world size 2, before any
    reading of this item could run.
    """
    config = _grid_shard_config()
    mappings = _mappings_for(config)
    reference = Glm5NextForConditionalGeneration(config)

    # THE GRIDS CARRY POSITION-IDENTIFYING VALUES, and without this the offset
    # reading below would be vacuous: this writer's own default grid is
    # ``torch.full(..., 0.5)``, and against a constant any block passes for any
    # other, so rank 1 could be handed rank 0's tile scales and no assertion in
    # this item would notice. Same reason ``_shard_key_overrides`` exists for the
    # weights. The shape comes from the module's closed form through
    # ``block_grid_shape`` and the dimension from the production table, so nothing
    # here states a number of its own.
    grids: dict[str, torch.Tensor] = {}
    for path, module in reference.named_modules():
        if type(module).__name__ != "Glm5NextMLAAttention":
            continue
        widths = {
            name: (idim, odim) for name, idim, odim in module.projection_widths()
        }
        for leaf in GRID_SHARD_LEAVES:
            grid_param = f"{path}.{leaf}_{FP8_SCALE_SUFFIX}"
            if grid_param not in mappings:
                continue
            idim, odim = widths[leaf]
            shape = block_grid_shape((odim, idim), DEFAULT_WEIGHT_BLOCK_SIZE)
            dim = _mla_grid_shard_dim(leaf)
            for key in _keys_of(mappings, grid_param):
                grids[key] = _shard_pattern(shape, dim, torch.float32)
    assert len(grids) == len(GRID_SHARD_LEAVES), (
        f"wrote position-identifying values for {len(grids)} grids, not the "
        f"{len(GRID_SHARD_LEAVES)} this item measures, so one of them is absent "
        f"from the map and would be read against a constant"
    )

    # THE SHARD TABLE'S OWN TENSORS, MINUS WHATEVER THE MODULE ALREADY ANSWERS FOR.
    # A wholesale merge of ``_shard_key_overrides`` would be wrong, not merely
    # redundant: it builds the MLA fulls out of :data:`MINI_MLA_WIDTHS`, which are
    # NOT this fixture's widths, so it would rewrite the three MLA projections at
    # 16-wide heads and destroy the grid-to-weight pairing this item exists to
    # measure -- turning a loud IndexError into a quiet wrong number. So its keys
    # are subtracted wherever ``_mla_key_overrides`` already speaks.
    #
    # DEFERRING TO THAT HELPER IS THE POINT, AND IT IS SELF-MAINTAINING. It covers
    # every module that publishes ``projection_widths()``, not literally the MLA
    # classes, so a future family that starts publishing widths is deferred to
    # automatically and the module stays the single source of those shapes. A
    # hand-listed set of class names here would drift from it.
    #
    # ONLY KEYS ARE SUBTRACTED, NEVER VALUES. The two helpers share a key space but
    # not a value type -- ``(shape, dtype)`` there, a tensor here -- and the writer
    # assigns an ``extra_overrides`` value straight into the checkpoint, so a tuple
    # leaking into this merge would be written as if it were a tensor.
    module_shapes = _mla_key_overrides(reference, mappings)
    shard_over = {
        key: tensor
        for key, tensor in _shard_key_overrides(reference, mappings).items()
        if key not in module_shapes
    }
    # ``grids`` last: the position-identifying grids must win over the shard
    # table's, which is also why the MLA grid keys are inside the subtraction above.
    extra_overrides = {**shard_over, **grids}

    # PRESENCE AND RANK, over every key a declared-sharded family maps here. This
    # states the CLASS of the defect rather than the one instance of it: a key no
    # override governs falls to the writer's own default, and a store whose rank
    # does not exceed its declared shard dim cannot be sharded at all. The check is
    # fixture-local on purpose -- the shared writer has landed callers that
    # legitimately store plain placeholders, and a blanket guard there would fire on
    # them.
    #
    # ``governed`` mirrors the writer's precedence (``extra_overrides`` first, then
    # ``_mla_key_overrides``), so it answers what the file will actually hold rather
    # than what this function happens to have built.
    governed: dict[str, tuple[int, ...]] = {
        key: tuple(tensor.shape) for key, tensor in extra_overrides.items()
    }
    for key, (module_shape, _dtype) in module_shapes.items():
        governed.setdefault(key, tuple(module_shape))
    ungoverned: list[str] = []
    unshardable: list[str] = []
    walked = 0
    mla_walked = 0
    rows: list[tuple[str, tuple[int, ...], int, int, str]] = []
    for path, module in reference.named_modules():
        cls = type(module).__name__
        for (family, leaf), (shard_dim, _full) in SHARD_FAMILIES.items():
            if cls != family:
                continue
            param = f"{path}.{leaf}"
            if param not in mappings:
                continue
            for key in _keys_of(mappings, param):
                walked += 1
                if key in module_shapes:
                    mla_walked += 1
                if key not in governed:
                    ungoverned.append(f"{key} ({family}.{leaf}, dim {shard_dim})")
                    continue
                shape = governed[key]
                if len(shape) <= shard_dim:
                    unshardable.append(
                        f"{key} ({family}.{leaf}) is stored {shape}, rank "
                        f"{len(shape)}, but is declared sharded on dim {shard_dim}"
                    )
                    continue
                extent = shape[shard_dim]
                rows.append(
                    (
                        key,
                        shape,
                        shard_dim,
                        extent // SHARD_WORLD,
                        "yes" if extent % SHARD_WORLD == 0 else "NO",
                    )
                )
    assert not ungoverned, (
        f"{len(ungoverned)} checkpoint key(s) belong to a family SHARD_FAMILIES "
        "declares sharded but no override governs them, so the writer would store "
        "each at MINI_PLAIN_SHAPE and the load would shard a placeholder: "
        f"{ungoverned}"
    )
    assert not unshardable, (
        f"{len(unshardable)} store(s) are sharded on a dimension they do not have, "
        f"which is the IndexError this fixture was repaired for: {unshardable}"
    )
    # THE POPULATION, MEASURED RATHER THAN ASSUMED. Both assertions above are presence
    # checks, and a presence check over an EMPTY key set passes in silence. This one
    # cannot be exercised on a machine without torch, so the run itself carries the
    # evidence that the walk reached something: the walked count must be non-zero, and
    # every key the walk reached must have ended as a row rather than vanishing between
    # the two refusals.
    assert walked > 0, (
        "the SHARD_FAMILIES walk reached no checkpoint key at all, so both assertions "
        "above passed over an empty set and proved nothing about this fixture"
    )
    assert len(rows) == walked, (
        f"the walk reached {walked} checkpoint key(s) but {len(rows)} produced a row, "
        "so a key was neither measured nor refused"
    )
    for key, shape, shard_dim, per_rank, exact in sorted(rows):
        print(
            f"STORESHAPE_ROW={key} shape={tuple(shape)} dim={shard_dim} "
            f"per_rank={per_rank} divides={exact}"
        )
    print(f"STORESHAPE_ROWS={len(rows)} GOVERNED_KEYS={len(governed)}")
    print(f"STORESHAPE_COUNT={walked} {walked - mla_walked} {mla_walked}")
    print(
        "STORESHAPE_COUNT_FIELDS=governed changed unchanged_mla  (unchanged_mla are "
        "the keys _mla_key_overrides already answers for, which this fixture "
        "deliberately leaves at the module's own closed form)"
    )

    directory = tmp_path / "grid-shard"
    written = _write_miniature_checkpoint(
        directory, mappings, reference, extra_overrides=extra_overrides
    )
    assert written, "the grid fixture wrote no tensors"
    return directory


def _load_grid_at_world(
    directory: Path, world_size: int, rank: int, monkeypatch
) -> Glm5NextForConditionalGeneration:
    """:func:`_load_at_world` on the grid config -- patch first, build second."""
    monkeypatch.setattr(_MODEL_FP8, "_resolve_world_size", lambda: world_size)
    monkeypatch.setattr(_MODEL_FP8, "_resolve_rank", lambda: rank)
    model = Glm5NextForConditionalGeneration(_grid_shard_config())
    assert model.world_size == world_size, (
        f"the model resolved world size {model.world_size}, not the patched "
        f"{world_size}"
    )
    _seed_page_cache_signal()
    model.load_weights(str(directory), torch.device("cpu"), None)
    return model


def _mla_grid_shard_dim(leaf: str) -> int:
    """The dimension the production table shards this projection's weight on.

    Read out of ``_SHARD_GEOMETRY`` rather than restated, so this item cannot
    disagree with the table it is measuring.
    """
    declared = _MODEL_FP8._SHARD_GEOMETRY["Glm5NextMLAAttention"][
        f"{leaf}{_LEAF_WEIGHT_SUFFIX}"
    ]
    return declared.shard_dim


def _mla_grids(
    model: Glm5NextForConditionalGeneration,
) -> dict[str, tuple[tuple[int, ...], torch.Tensor]]:
    """``dotted leaf -> (weight shape, grid tensor)`` per sharded scaled projection.

    The grid of a DECLARED parameter is a parameter, so ``getattr`` finds it under
    the name ``_sibling_scale_grid_name`` builds -- which is what this reads. (The
    dense-MLP grids are NOT declared parameters and land as plain attributes
    through the out-of-band reader; that is a different path and not this one.)
    """
    found: dict[str, tuple[tuple[int, ...], torch.Tensor]] = {}
    for path, module in model.named_modules():
        if type(module).__name__ != "Glm5NextMLAAttention":
            continue
        for leaf in GRID_SHARD_LEAVES:
            weight = getattr(module, f"{leaf}{_LEAF_WEIGHT_SUFFIX}", None)
            grid = getattr(module, f"{leaf}_{FP8_SCALE_SUFFIX}", None)
            if weight is None or grid is None:
                continue
            found[f"{path}.{leaf}"] = (tuple(weight.shape), grid.data)
    return found


def test_gridshard_a_sharded_projections_scale_grid_shards_with_its_weight(
    tmp_path: Path, monkeypatch, single_rank_process_group
) -> None:
    """inc-glm53f-105. A sharded FP8 weight's grid describes THIS RANK's blocks.

    ``inc-glm53f-100`` shards ``q_b_proj`` and ``o_proj``, both of which carry a
    ``weight_scale_inv`` grid. Left whole, such a grid describes the unsharded
    tensor while the weight is this rank's half, so ``dequantise_blockwise``
    refuses the pair and the load fails at real widths. That is the defect this
    increment repairs. Five readings, each printed before it is asserted:

    1. the unsharded grid spans MORE blocks along the shard dimension than one
       rank's does, so every reading below is distinguishable at all;
    2. at world size 2 the load COMPLETES -- the point, because
       ``prepare_projection_weights`` and ``dequantise_blockwise`` run inside
       ``load_weights`` -- and each grid holds exactly the block count its own
       loaded weight implies, computed with ``block_grid_shape``, never typed;
    3. each rank's grid is the CORRECT slice of the unsharded grid, by value and
       at the right offset, which is what says the shard is indexed rather than
       merely the right size;
    4. THE CONTROL: with the grid's geometry resolved back to ``None`` -- the
       behaviour before this increment, weights left sharded, nothing else changed
       -- the same load raises ``Glm5NextWeightMapError``.
    5. the projection seam's dispatch counters read ``(0, 0)`` across every load
       this item performs, the control's included, which is the ``0`` dispatches
       this increment's registered acceptance states.

    ON READING (5), HONESTLY: NO FIRING CONTROL IS STAGED FOR ITS ZERO INSIDE THIS
    ITEM. The zero is not structural -- the seam has a real NKI route that does
    increment ``nki_dispatch`` -- so the reading claims something falsifiable: the
    weight-loading path never enters that seam. What this item does not do is
    demonstrate the instrument firing, because the only way to fire it here would
    be to dispatch the tiled NKI matmul inside a load test, a dependency this item
    does not otherwise carry. The same accessor is read NONZERO, at world size 2,
    by ``test_mla_decode.py``'s ``inc-glm53f-100`` route-predicate item, which
    lands in the SAME changeset; a reviewer wanting the firing half of this
    reading should read it there. Note also WHY the zero holds, which is checkable
    without running anything: ``model_fp8`` imports the seam lazily, inside the
    projection methods themselves, so a load that never projects never imports it
    -- this item's own import is what brings the module in.

    ON READING (2), HONESTLY: its assertion is DOMINATED and cannot fail on its
    own. ``_require_grid`` raises inside ``load_weights`` for a mismatched pair, so
    a wrong grid shape surfaces as reading (2)'s load raising, never as its
    assert. It is kept because it states the criterion in the criterion's own
    words; the measuring is done by (1), (3) and (4). Reading (3) is the one that
    would catch a grid sharded at the wrong OFFSET -- a shape check cannot, and
    without it this item would accept rank 1 being handed rank 0's block.
    """
    from vllm_neuron.functional.attention import mla_projections

    # Reset BEFORE the fixture, so reading (5) below covers every load this item
    # performs rather than a suffix of them.
    mla_projections.reset_mla_projection_dispatch_counters()

    directory = _grid_shard_checkpoint(tmp_path)

    whole = _load_grid_at_world(directory, 1, 0, monkeypatch)
    whole_grids = _mla_grids(whole)
    assert whole_grids, (
        "no MLA module reported both a weight and a scale grid, so this item has "
        "no subject; the fixture or the declared names have moved"
    )
    print(f"GRIDSHARD_1_SUBJECTS={sorted(whole_grids)}")
    print(
        "GRIDSHARD_1_WHOLE="
        f"{sorted((k, w, tuple(g.shape)) for k, (w, g) in whole_grids.items())}"
    )

    per_rank = {
        rank: _mla_grids(_load_grid_at_world(directory, SHARD_WORLD, rank, monkeypatch))
        for rank in range(SHARD_WORLD)
    }
    print(
        "GRIDSHARD_2_PER_RANK="
        f"{sorted((r, k, w, tuple(g.shape)) for r, f in per_rank.items() for k, (w, g) in f.items())}"
    )
    for rank, found in sorted(per_rank.items()):
        assert set(found) == set(whole_grids), (
            f"rank {rank} reported different subjects from the world-1 load: only "
            f"whole {sorted(set(whole_grids) - set(found))}, only rank "
            f"{sorted(set(found) - set(whole_grids))}"
        )

    narrowed = 0
    sliced = 0
    for dotted, (whole_shape, whole_grid) in sorted(whole_grids.items()):
        leaf = dotted.rpartition(".")[2]
        dim = _mla_grid_shard_dim(leaf)
        for rank, found in sorted(per_rank.items()):
            weight_shape, grid = found[dotted]
            expected = block_grid_shape(weight_shape, DEFAULT_WEIGHT_BLOCK_SIZE)
            assert tuple(grid.shape) == expected, (
                f"{dotted} loaded a {weight_shape} weight at rank {rank} of world "
                f"size {SHARD_WORLD} with a {tuple(grid.shape)} grid; a blockwise "
                f"grid holds one value per tile, so this weight's grid is "
                f"{expected}. A grid describing the unsharded tensor scales the "
                f"wrong blocks"
            )
            extent = expected[dim]
            mine = whole_grid.narrow(dim, rank * extent, extent)
            difference = _max_abs_diff(grid, mine)
            assert difference == 0.0, (
                f"{dotted} at rank {rank} is not blocks "
                f"[{rank * extent}:{(rank + 1) * extent}] of dim {dim} of the "
                f"unsharded grid: they differ by {difference}. The shape is right, "
                f"so this is an OFFSET defect -- the rank is being handed another "
                f"rank's tile scales"
            )
            sliced += 1
        if tuple(whole_grid.shape) != tuple(per_rank[0][dotted][1].shape):
            narrowed += 1
    print(f"GRIDSHARD_2_GRIDS_CHECKED={len(whole_grids) * SHARD_WORLD}")
    print(f"GRIDSHARD_3_SLICE_READINGS={sliced}")
    print(f"GRIDSHARD_1_GRIDS_THAT_NARROWED={narrowed}")
    assert narrowed == len(whole_grids), (
        f"only {narrowed} of {len(whole_grids)} grids differ from their unsharded "
        f"shape, so for the rest this item would pass whether or not the grid was "
        f"sharded. These widths exist to prevent exactly that"
    )

    # ── THE CONTROL. Resolve a GRID leaf's geometry back to None -- what
    # ``_shard_geometry_for`` did before inc-glm53f-105 -- and leave the weights'
    # geometry alone, so the ONLY difference is the thing this increment added.
    landed = _MODEL_FP8._shard_geometry_for

    def without_grid_geometry(module, leaf, world_size):
        if leaf.endswith(f"_{FP8_SCALE_SUFFIX}"):
            return None
        return landed(module, leaf, world_size)

    monkeypatch.setattr(_MODEL_FP8, "_shard_geometry_for", without_grid_geometry)
    with pytest.raises(Glm5NextWeightMapError) as refusal:
        _load_grid_at_world(directory, SHARD_WORLD, 0, monkeypatch)
    message = str(refusal.value)
    print(f"GRIDSHARD_4_CONTROL_REFUSAL={message!r}")
    assert "scale grid shape" in message, (
        f"the control raised Glm5NextWeightMapError for some other reason, so it "
        f"does not show that the whole grid is what this increment fixes: "
        f"{message}"
    )

    dispatches = mla_projections.mla_projection_dispatch_counters()
    print(f"GRIDSHARD_5_DISPATCHES={dispatches}")
    assert dispatches == (0, 0), (
        f"the projection seam counted {dispatches} (nki, torch_fallback) across "
        f"this item's loads, and this increment's registered acceptance states 0 "
        f"dispatches: weight loading dequantises and shards, it does not project"
    )


# =========================================================================== #
# inc-glm53f-105b -- the condition that lets inc-glm53f-105 decline to pad.
#
# WHY THIS ITEM EXISTS. inc-glm53f-101 added ``_DeclaredShard.pad_to_consumer_block``,
# which rounds a family's full width UP to a multiple of ``world_size`` times the
# consumer's block extent so that every rank's shard is a whole block. The MLA
# entries inc-glm53f-100 declares do NOT set it, and that is a decision rather than
# an omission: the axis those three projections shard carries WHOLE HEADS, so padding
# it would add columns belonging to no head, which is the thing head partitioning
# exists to prevent. The dense MLP's intermediate width has no such structure, which
# is why the same flag is right there and wrong here.
#
# Declining to pad is only safe because of a config fact: each MLA head width is
# ITSELF a whole number of blocks, so any whole-head split inherits that and no
# rank can be handed a part-tile. That fact is not a law. A checkpoint with a
# DeepSeek-style split -- 128 nope plus 64 rope, 192 to a head -- breaks it, and then
# ``shard_geometry_for_grid`` REFUSES the load rather than corrupting it. The refusal
# is the right behaviour and a stopped load is still a stopped load.
#
# So the reason is paired with a gate on the condition it rests on. If a future
# config moves a head width off the block extent, this item fails and names the leaf,
# instead of the failure surfacing as a refused load nobody predicted.
# =========================================================================== #


def test_gridshard_b_every_sharded_mla_head_width_is_a_whole_quant_block() -> None:
    """inc-glm53f-105b. Why -105 does not pad, and the gate that expires the reason.

    Four readings: two subjects, and a control for each.

    1. every MLA head width that inc-glm53f-100 shards is a whole number of the
       CHECKPOINT TILE, ``DEFAULT_WEIGHT_BLOCK_SIZE``, measured on the dimension
       each family is actually sharded on -- dim 0 for the two column-parallel
       projections, dim 1 for the row-parallel one, because a tile is not square in
       principle even though this checkpoint's is;
    2. and a whole number of the CONSUMER's block, imported from
       ``consumer_block_quant_size()``. These are TWO boundaries and not one
       restated: ``shard_geometry_for_grid`` refuses on the tile and, separately,
       on the consumer's block, so a width can clear the first and still stop the
       load at the second;
    3. reading 1's control, a DeepSeek-style split that is not a whole tile;
    4. reading 2's control, a width that CLEARS the tile and still leaves half a
       consumer block -- the case that made this item read green through a load
       that refused, before the repair.

    WHAT THIS DOCSTRING USED TO SAY, AND WHY IT WAS WRONG. Reading 1 called
    ``DEFAULT_WEIGHT_BLOCK_SIZE`` "the consumer's quantisation blocks". It is the
    checkpoint's tile; the consumer's block is a different number from a different
    module, and conflating them is what left the second boundary ungated. The FILED
    evidence record ``../increments/evidence-105b.md:16`` repeats the same wording
    and keeps its bytes; the erratum belongs to this repair's own record.

    NOTHING IS TYPED. The head widths come from ``Glm5NextTextConfig``'s own
    defaults and the block extent from ``DEFAULT_WEIGHT_BLOCK_SIZE``, so a config
    change moves this item's subject with it rather than leaving a second set of
    numbers behind to drift. That is also why the item asserts a PROPERTY and not
    the numbers 256, 512 and 256: the numbers are today's answer, the property is
    the claim.
    """
    config = Glm5NextTextConfig()

    #: leaf -> (dim it shards on, its per-head width along that dim). The widths
    #: are summed the way ``projection_widths`` sums them: the rotary slice is 0 on
    #: this checkpoint and that 0 is a value, so a config that had one would
    #: otherwise be short by exactly that slice.
    head_widths = {
        "q_b_proj_weight": (0, config.qk_nope_head_dim + config.qk_rope_head_dim),
        "kv_b_proj_weight": (0, config.qk_nope_head_dim + config.v_head_dim),
        "o_proj_weight": (1, config.v_head_dim),
    }

    def part_tile(shard_dim: int, head_width: int) -> int:
        """The remainder a single head leaves in the CHECKPOINT tile. 0 is whole."""
        return head_width % DEFAULT_WEIGHT_BLOCK_SIZE[shard_dim]

    #: The CONSUMER's block extent, IMPORTED from the consumer and never typed here.
    #: A literal would be a second place for the consumer's granularity to live,
    #: which is the reason ``consumer_block_quant_size`` exists and says so in its
    #: own docstring. It is one number rather than a pair because
    #: ``blockwise_fp8_mm.BLOCK_QUANT_SIZE`` is one number and the consumer applies
    #: it on either dimension.
    consumer_block = _WL_FP8.consumer_block_quant_size()

    def part_block(head_width: int) -> int:
        """The remainder a single head leaves in the CONSUMER's block. 0 is whole."""
        return head_width % consumer_block

    print(f"GRIDSHARD_B_BLOCK_SIZE={DEFAULT_WEIGHT_BLOCK_SIZE}")
    print(
        "GRIDSHARD_B_HEAD_WIDTHS="
        f"{ {leaf: width for leaf, (_, width) in sorted(head_widths.items())} }"
    )
    print(f"GRIDSHARD_B_HEADS={config.num_attention_heads}")

    offenders = {
        leaf: width
        for leaf, (dim, width) in head_widths.items()
        if part_tile(dim, width)
    }
    print(f"GRIDSHARD_B_HEAD_WIDTHS_LEAVING_A_PART_TILE={offenders}")

    # READING 2, THE SECOND BOUNDARY. Same widths, the consumer's block instead of
    # the checkpoint's tile.
    print(f"GRIDSHARD_B_CONSUMER_BLOCK={consumer_block}")
    block_offenders = {
        leaf: width
        for leaf, (_dim, width) in head_widths.items()
        if part_block(width)
    }
    print(f"GRIDSHARD_B_HEAD_WIDTHS_LEAVING_A_PART_BLOCK={block_offenders}")

    # READING 2, THE CONTROL, in this same output. A DeepSeek-style split of 128
    # nope plus 64 rope gives 192 to a head, which is one and a half blocks. It is
    # run through the SAME predicate, so a predicate that had stopped discriminating
    # would show up here rather than being reported as reading 1's clean result.
    control_widths = {"q_b_proj_weight": (0, 128 + 64)}
    control_offenders = {
        leaf: width
        for leaf, (dim, width) in control_widths.items()
        if part_tile(dim, width)
    }
    print(f"GRIDSHARD_B_CONTROL_PART_TILE_WIDTHS={control_offenders}")
    assert control_offenders, (
        "the part-tile predicate did not flag a 192-wide head against a "
        f"{DEFAULT_WEIGHT_BLOCK_SIZE} block, so it cannot detect a violation and "
        "reading 1's empty result below says nothing"
    )

    # READING 4, READING 2'S CONTROL, AND THE REASON THIS ITEM WAS REPAIRED. A q_b
    # split of 256 nope plus 128 rope gives 384 to a head. That CLEARS the checkpoint
    # tile -- 384 is three whole 128-row tiles -- and still leaves half a consumer
    # block. Before the repair this item measured the tile alone, so a checkpoint like
    # that read GREEN here while ``shard_geometry_for_grid`` refused the load at the
    # consumer boundary: the refused-load-nobody-predicted this gate exists to catch.
    # ``shard_geometry_for_grid``'s own docstring names this width in those words.
    escaping_width = 256 + 128
    escaping_tile = part_tile(0, escaping_width)
    escaping_block = part_block(escaping_width)
    print(
        f"GRIDSHARD_B_CONTROL_ESCAPING_WIDTH={escaping_width} "
        f"TILE_REMAINDER={escaping_tile} CONSUMER_REMAINDER={escaping_block}"
    )
    assert escaping_tile == 0 and escaping_block, (
        f"the {escaping_width}-wide control no longer clears the tile while failing "
        f"the consumer block: tile remainder {escaping_tile}, consumer remainder "
        f"{escaping_block}, against tile {DEFAULT_WEIGHT_BLOCK_SIZE} and consumer "
        f"block {consumer_block}. This control is the whole reason the item reads "
        f"both boundaries, so if it stops discriminating then reading 2's empty "
        f"result below says nothing"
    )

    assert offenders == {}, (
        f"an MLA head width is not a whole number of quantisation blocks: "
        f"{offenders}, against a block size of {DEFAULT_WEIGHT_BLOCK_SIZE}. "
        f"inc-glm53f-100 shards these three on whole heads, so a head that leaves "
        f"a part tile means some rank's shard ends inside a block whose single "
        f"scale cannot be divided between two ranks, and "
        f"``shard_geometry_for_grid`` refuses the load. inc-glm53f-105 declines "
        f"``pad_to_consumer_block`` because padding a head-bearing axis invents "
        f"columns belonging to no head; that decision rested on this condition, so "
        f"this failure means the decision needs revisiting rather than the widths "
        f"being wrong"
    )

    assert block_offenders == {}, (
        f"an MLA head width is not a whole number of the CONSUMER's blocks: "
        f"{block_offenders}, against a consumer block of {consumer_block} imported "
        f"from ``consumer_block_quant_size()``. Such a width can clear the "
        f"{DEFAULT_WEIGHT_BLOCK_SIZE} checkpoint tile and still be refused, because "
        f"``shard_geometry_for_grid`` makes TWO refusals and this is the second one. "
        f"inc-glm53f-105 declines ``pad_to_consumer_block`` on a head-bearing axis, "
        f"and that decision rested on BOTH boundaries holding, so this failure means "
        f"the decision needs revisiting rather than the widths being wrong"
    )


# ---------------------------------------------------------------------------
# inc-glm53f-105b, second premise: the one reachable block size
# ---------------------------------------------------------------------------
# WHY THIS ITEM EXISTS. ``inc-glm53f-101`` gave its ``sharded_scale_grid_loader`` a
# third parameter, ``block_size``, and forwards it into ``shard_geometry_for_grid``.
# ``inc-glm53f-105``'s compensating sibling takes no such parameter, so it always
# uses the default. That asymmetry is safe for ONE reason and it is a reason that
# can stop being true: the build supports exactly one block size, and the default
# IS that value. ``block_size`` is not a label -- inside
# ``shard_geometry_for_grid`` it picks ``extent``, the divisor the sharded grid's
# width is computed from -- so if a second shape ever became reachable, the sibling
# would honour a caller's choice and this one would silently keep 128x128.
#
# This item is that premise written as a gate rather than as a comment. It fails the
# day someone adds a second supported block size, and the failure names the function
# that must then grow the parameter. It is the same shape as the part-tile gate
# above: a decision recorded with the condition it rests on, so the decision expires
# loudly instead of quietly becoming wrong.
def test_gridshard_b_one_reachable_block_size_is_why_105_omits_the_parameter() -> None:
    from vllm_neuron.model.glm5_next import quantization as _quant

    supported = _quant.SUPPORTED_WEIGHT_BLOCK_SIZES
    default = _quant.DEFAULT_WEIGHT_BLOCK_SIZE
    print(f"GRIDSHARD_B_SUPPORTED_BLOCK_SIZES={sorted(supported)}")
    print(f"GRIDSHARD_B_DEFAULT_BLOCK_SIZE={default}")
    print(f"GRIDSHARD_B_SUPPORTED_COUNT={len(supported)}")

    # Which of the two loaders can be handed a block size, read off the live
    # signatures rather than restated, so a signature change moves this reading.
    def takes_block_size(fn: object) -> bool:
        return "block_size" in inspect.signature(fn).parameters  # type: ignore[arg-type]

    sibling_takes = takes_block_size(_WL_FP8.sharded_scale_grid_loader)
    mine_takes = takes_block_size(_WL_FP8.compensating_sharded_scale_grid_loader)
    geom_default = inspect.signature(
        _WL_FP8.shard_geometry_for_grid
    ).parameters["block_size"].default
    print(f"GRIDSHARD_B_SIBLING_TAKES_BLOCK_SIZE={sibling_takes}")
    print(f"GRIDSHARD_B_MINE_TAKES_BLOCK_SIZE={mine_takes}")
    print(f"GRIDSHARD_B_GEOMETRY_DEFAULT_BLOCK_SIZE={geom_default}")

    # CONTROL, FIRST. The predicate below is "more than one reachable shape", and a
    # reading of 1 from it means nothing until it has been shown saying 2 about a set
    # that really holds two. Without this, a predicate that always answered "one"
    # would pass this item forever.
    control_set = frozenset({default, (64, 64)})
    print(f"GRIDSHARD_B_CONTROL_SET={sorted(control_set)}"
          f" CONTROL_COUNT={len(control_set)}")
    assert len(control_set) > 1, (
        "the two-shape control set did not read as more than one shape, so the "
        "count below cannot detect a second supported block size and its reading "
        "of 1 says nothing"
    )

    assert len(supported) == 1 and supported == frozenset({default}), (
        f"this build now supports {sorted(supported)} rather than exactly "
        f"[{default}]. inc-glm53f-105's compensating_sharded_scale_grid_loader "
        f"takes no block_size parameter and so always uses the default; that was "
        f"safe only while the default was the one reachable value. Give it the "
        f"parameter and forward it into shard_geometry_for_grid, the way "
        f"inc-glm53f-101's sharded_scale_grid_loader already does, before a "
        f"checkpoint with another shape can reach this path"
    )
    assert geom_default == default, (
        f"shard_geometry_for_grid defaults block_size to {geom_default}, not "
        f"{default}. inc-glm53f-105 omits the argument, so the default is what its "
        f"path actually uses; a default that is not the supported shape means the "
        f"omission now picks the wrong divisor"
    )
    assert sibling_takes and not mine_takes, (
        f"the two loaders' signatures moved: sibling takes block_size="
        f"{sibling_takes}, this increment's takes block_size={mine_takes}. This "
        f"item exists to explain that exact asymmetry, so a change to either "
        f"signature means the explanation needs rewriting rather than re-asserting"
    )


# --------------------------------------------------------------------------- #
# inc-glm53f-106 -- the routed bank REFUSES an inadmissible shard instead of
# padding one, and the grid conversion keeps the expert-parallel degree.
#
# WHY THESE FOUR ITEMS EXIST. Review B90-101 found that `-101`'s acceptance passed
# 25/25 with two defects present, and it passed because the registered fixture made
# both defects invisible: BANK_INTERMEDIATE=512 at tp_per_ep=2 makes the pad a no-op,
# so the ruled behaviour (refuse) and the landed behaviour (pad) produced the SAME
# number at every tested width. Each item below therefore names the reason it would
# RED on the pre-repair code, and item 1 carries the falsifier for the rule itself.
#
# THE SELECTOR IS `bankpad` AND THAT IS DELIBERATE. It contains neither `stacked`
# (4 frozen items), `sharedshard` (5) nor `shard` as those selectors match, so no
# frozen expected count moves -- design entry 85 puts widening one out of a seat's
# reach.
# --------------------------------------------------------------------------- #

BANKPAD_PARAM = "layers.0.mlp.experts.gate_proj_weight"
BANKPAD_SIBLING_PARAM = "layers.0.mlp.gate_proj_weight"


class _BankpadSlice:
    """The two things ``tensor_width_sharding_loader`` asks of a checkpoint slice.

    ``get_shape`` and ``__getitem__``, and nothing else -- read off that function's
    own body (``utils/weight_loader.py:443``, ``:456``) rather than guessed, so this
    stub cannot drift into supporting a call the real loader never makes.
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        self._shape = tuple(shape)
        self._tensor = torch.zeros(self._shape, dtype=torch.float32)

    def get_shape(self) -> list[int]:
        return list(self._shape)

    def __getitem__(self, key):
        return self._tensor[key]


def _bankpad_bank_geometry(ep_degree: int, world_size: int):
    """The bank's geometry from the TABLE, never hand-built here.

    ``_shard_geometry_for`` is the single reader of ``_SHARD_GEOMETRY``, and the
    declaration is what this increment changed -- so an item that constructed a
    geometry directly would assert its own opinion of the table instead of the
    table. The owner is synthesised with the DECLARING CLASS NAME because that name
    is the table's key (``model_fp8.py``'s ``_SHARD_GEOMETRY`` is keyed by name, not
    by class object, and says so).
    """
    from vllm_neuron.model.glm5_next import model_fp8 as _MF8

    owner = type("Glm5NextRoutedExperts", (), {"ep_degree": ep_degree})()
    return _MF8._shard_geometry_for(owner, "gate_proj_weight", world_size)


def test_bankpad_the_bank_refuses_an_inadmissible_extent_and_a_sibling_family_still_loads() -> None:
    """Item 1. The bank refuses what it used to pad, and the rule stays narrow.

    TWO READINGS IN ONE OUTPUT, and the second is not decoration. The lead's ruling
    (DECISIONS 296-298) refused the predicate "deferred AND no pad implies whole
    consumer blocks" precisely because it would refuse a future deferred family that
    is not consumed by the block-FP8 kernel. Reading B is the falsifier for that
    risk: a sibling deferred family that declares NO requirement loads an unaligned
    extent exactly as it did before this increment. Without reading B, reading A is
    consistent with a rule that refuses everything.

    WHY THIS REDS ON THE PRE-REPAIR CODE. Before this increment the three routed rows
    passed ``pad_to_consumer_block=True``, so the geometry carried a pad of 256 and
    ``load_time_shard_size`` rounded 512 up to 1024: every rank got 256 rows, of
    which 128 were real, and the load SUCCEEDED. Reading A expects a refusal and
    would fail on that success. After the repair the rows declare
    ``require_consumer_block`` instead and the load refuses by name.
    """
    world = 4
    block = _WL_FP8.consumer_block_quant_size()
    geometry = _bankpad_bank_geometry(ep_degree=1, world_size=world)
    print(
        f"BANKPAD1_GEOMETRY num_shards={geometry.num_shards} "
        f"pad={geometry.pad_to_multiple_of} require={geometry.require_multiple_of} "
        f"degree={geometry.expert_parallel_degree}"
    )
    assert geometry.pad_to_multiple_of is None, (
        f"the bank still declares a pad of {geometry.pad_to_multiple_of}; this "
        f"increment's first change is that it declares a requirement instead"
    )
    assert geometry.require_multiple_of == block, (
        f"the bank declares require_multiple_of={geometry.require_multiple_of}, not "
        f"the consumer's own {block}; the number must be imported, never typed"
    )
    assert geometry.num_shards == world, (
        f"at expert-parallel degree 1 the bank must divide by the WORLD ({world}), "
        f"and this geometry says {geometry.num_shards} -- if this ever reads 1 the "
        f"case below is not the case this item means to measure"
    )

    # READING A: the bank refuses, and the per-rank extent it refuses is the one the
    # unpadded division actually produces.
    full = 512
    experts = 2
    expected_extent = full // world
    assert expected_extent % block != 0, (
        f"{full} over {world} ranks is {expected_extent}, which IS a whole {block} "
        f"block, so this case cannot distinguish a refusal from an acceptance"
    )
    stack_whole = lambda _slices, _rank: torch.zeros((experts, full, 8))
    transform = _WL_FP8._column_of_each_expert(geometry, BANKPAD_PARAM, stack_whole)
    with pytest.raises(Glm5NextExpertBankNotLoadableError) as refusal:
        transform([], 0)
    message = str(refusal.value)
    print(f"BANKPAD1_REFUSAL={message[:260]}")
    assert message.startswith(f"{BANKPAD_PARAM} "), (
        f"the refusal does not name the parameter first: {message}"
    )
    assert f"loads {expected_extent} rows per rank" in message, (
        f"the refusal does not say the per-rank extent it refused: {message}"
    )
    assert f"{block}-row block" in message, (
        f"the refusal does not name the consumer's block: {message}"
    )
    # THE SUB-BLOCK ARM OF THE MESSAGE, and it is read here on purpose. 128 rows
    # under a 256-row block puts the largest admissible extent BELOW one whole block
    # at zero, so a message that named "the nearest admissible extents" would offer
    # this rank no rows at all as its advice. That defect was found by walking the
    # message against this very case, so this case is where it is measured.
    assert below_and_above(expected_extent, block)[0] == 0, (
        f"{expected_extent} is at least one whole {block} block, so this case reads "
        f"the other arm of the message and the sub-block arm goes unmeasured"
    )
    assert f"narrower than one whole block" in message, (
        f"the refusal took the nearest-neighbours arm at a sub-block extent: {message}"
    )
    assert f"smallest admissible per-rank extent is {block}" in message, (
        f"the refusal does not name a usable smallest extent: {message}"
    )
    assert f"full width of {block * world}" in message, (
        f"the refusal does not name an admissible full width: {message}"
    )

    # READING B, THE FALSIFIER: a sibling deferred family declaring NO requirement
    # loads the same unaligned extent without complaint.
    sibling = _WL_FP8.DeferredShardGeometry(shard_dim=0, num_shards=world)
    assert sibling.require_multiple_of is None, (
        "the default of require_multiple_of moved; a family that declares nothing "
        "must stay unbound or this reading proves the opposite of what it says"
    )
    loaded = _WL_FP8._sharding_loader(sibling, BANKPAD_SIBLING_PARAM).transform(
        [_BankpadSlice((full, 8))], 0
    )
    got = tuple(loaded.shape)
    print(f"BANKPAD1_SIBLING_LOADED={got} UNALIGNED_BY={got[0] % block}")
    assert got == (expected_extent, 8), (
        f"the sibling family loaded {got}, not {(expected_extent, 8)}; this reading "
        f"only falsifies the over-broad rule if it loads the SAME unaligned extent "
        f"the bank was just refused for"
    )
    assert got[0] % block != 0, (
        f"the sibling's extent {got[0]} is a whole {block} block, so it was never a "
        f"candidate for the refusal and falsifies nothing"
    )


def test_bankpad_a_width_the_pad_would_move_still_refuses_at_the_ruled_degree() -> None:
    """Item 2. A width the pad would move, at the degree that actually reaches the rule.

    WHAT THIS ITEM ADDS THAT ITEM 1 DOES NOT, and it is an arm and not a width. The
    refusal names the admissible neighbours out of two DIFFERENT arms, and item 1 can
    only ever reach one of them. Item 1's per-rank extent is narrower than a whole
    block, so :func:`refuse_inadmissible_shard_extent` takes its "narrower than one
    whole block" arm and names a SINGLE width (``weight_loaders_fp8.py:1736-1741``).
    This item's 384 rows floor to 256, so the OTHER arm runs and names BOTH
    neighbours, 256 and 512 (``:1730-1734``). Without this item that arm never
    executes, and the assertion below is on the arm rather than on the sentence.

    WHY THE DEGREE IS 1 AND NOT 2, MEASURED RATHER THAN PREFERRED. The first version
    of this item ran at degree 2 and proved nothing about the extent rule. At any
    degree above 1 ``_expert_parallel_shard_column`` asks for the EP-TP group, no
    group is initialised under test, and it refuses for THAT reason first
    (``:2503-2519``) -- so the extent check never ran and the item would have passed
    on the wrong refusal. At degree 1 the same function short-circuits at ``:2500``,
    ``if ep_degree <= 1: return rank % tp_per_ep``, before the group lookup, so the
    extent rule is what this item can reach. The degree-2 preemption is not a
    hypothesis: the counted run under grant 019 printed that other refusal.

    THE BANK IS STILL SHARDED AT DEGREE 1, which is why the fixture is not vacuous.
    ``_shard_geometry_for`` hands back a deferred geometry whose ``num_shards`` is the
    WORLD at degree 1, and the assertion below reads that back from the table instead
    of assuming it.

    WHY THIS REDS ON THE PRE-REPAIR CODE. Before this increment the three routed rows
    declared ``pad_to_consumer_block``, so 1536 rounded up to 2048 and every rank held
    512 rows -- two whole blocks, of which 384 were real -- and the load SUCCEEDED.
    This item expects a refusal, so it fails on that success.
    """
    world = 4
    ep_degree = 1
    block = _WL_FP8.consumer_block_quant_size()
    geometry = _bankpad_bank_geometry(ep_degree=ep_degree, world_size=world)
    tp_per_ep = world // ep_degree
    assert geometry.num_shards == tp_per_ep, (
        f"the bank's rank count is {geometry.num_shards} where tp_per_ep is "
        f"{tp_per_ep}; the divisor moved and this case is not the ruled one"
    )

    full = 1536
    unpadded = full // tp_per_ep
    padded = _padded_shard_extent(full, tp_per_ep, block)
    below, above = below_and_above(unpadded, block)
    print(
        f"BANKPAD2_FULL={full} TP_PER_EP={tp_per_ep} UNPADDED={unpadded} "
        f"PADDED={padded} BLOCK={block}"
    )
    print(f"BANKPAD2_NEIGHBOURS={below} and {above}")
    assert padded != unpadded, (
        f"padding {full} over {tp_per_ep} ranks gives {padded}, the same as the "
        f"unpadded {unpadded}: the pad is a NO-OP at this width, which is exactly "
        f"the masking this item was minted to remove"
    )
    assert unpadded % block != 0, (
        f"{unpadded} is a whole {block} block, so the unpadded shard is admissible "
        f"and there is nothing here to refuse"
    )
    assert below >= block, (
        f"flooring {unpadded} to a multiple of {block} gives {below}, under one whole "
        f"block, so this width takes the narrower-than-a-block arm item 1 already "
        f"owns and the two-neighbour arm this item exists for stays unexecuted"
    )

    stack_whole = lambda _slices, _rank: torch.zeros((2, full, 8))
    transform = _WL_FP8._column_of_each_expert(geometry, BANKPAD_PARAM, stack_whole)
    with pytest.raises(Glm5NextExpertBankNotLoadableError) as refusal:
        transform([], 0)
    message = str(refusal.value)
    print(f"BANKPAD2_REFUSAL={message[:260]}")
    assert f"admissible are {below} and {above}" in message, (
        f"the refusal does not name BOTH admissible neighbours {below} and {above}, "
        f"so it did not take the two-neighbour arm this item exists to execute: "
        f"{message}"
    )


def below_and_above(extent: int, required: int) -> tuple[int, int]:
    """The two admissible neighbours of ``extent``, computed here independently.

    Deliberately NOT imported from the module under test: an expectation that calls
    the same helper as the code cannot detect the helper being wrong. Two lines of
    arithmetic written twice is the point.
    """
    below = (extent // required) * required
    return below, below + required


def test_bankpad_a_bank_grid_column_comes_from_the_group_not_the_modulo(
    monkeypatch,
) -> None:
    """Item 3. The bank's GRID reaches its column through the group, not a division.

    WHAT B90 FOUND AND WHY ITEM (5) DID NOT CATCH IT. Conjunct (5) calls
    ``_expert_parallel_shard_column`` DIRECTLY, so it certifies the reader and never
    the grid's route to it. The grid's route runs
    ``stacked_expert_scale_loader`` -> ``shard_geometry_for_grid`` ->
    ``_column_of_each_expert`` -> the reader, and the conversion in the middle dropped
    ``expert_parallel_degree``. At degree 1 the reader short-circuits to
    ``rank % tp_per_ep`` with no group read and no refusal, so every bank GRID at
    every real degree above 1 took the modulo column the ruling forbids while the
    WEIGHT half kept the group column.

    WHY THIS REDS ON THE PRE-REPAIR CODE. The converted geometry carried degree 1,
    so the disagreement below could not be seen and the transform returned quietly.
    """
    world = 64
    disagreeing_degree = 8
    tp_per_ep = world // disagreeing_degree
    weight_geometry = _bankpad_bank_geometry(
        ep_degree=disagreeing_degree, world_size=world
    )
    converted = _WL_FP8.shard_geometry_for_grid(
        weight_geometry, BANKPAD_PARAM, DEFAULT_WEIGHT_BLOCK_SIZE
    )
    print(
        f"BANKPAD3_WEIGHT_DEGREE={weight_geometry.expert_parallel_degree} "
        f"CONVERTED_DEGREE={converted.expert_parallel_degree}"
    )
    assert converted.expert_parallel_degree == disagreeing_degree, (
        f"the grid conversion carries degree {converted.expert_parallel_degree} "
        f"where the weight declared {disagreeing_degree}; the column reader will "
        f"short-circuit and read no group"
    )

    assert _NPS.uses_noncontiguous_mesh(world, tp_per_ep), (
        f"the package does not call world {world} with row {tp_per_ep} "
        f"non-contiguous, so this item's premise is gone"
    )
    _row, column_of_4 = _mesh_answers(world, disagreeing_degree, 4)
    print(f"BANKPAD3_GROUP_COLUMN={column_of_4} MODULO_COLUMN={4 % tp_per_ep}")
    assert column_of_4 != 4 % tp_per_ep, (
        f"the group puts rank 4 at column {column_of_4} and the modulo also says "
        f"{4 % tp_per_ep}; they agree, so there is no disagreement to refuse"
    )
    monkeypatch.setattr(
        _NPS, "get_neuron_ep_tp_group", lambda: _FixtureGroup(column_of_4, tp_per_ep)
    )
    stack_whole = lambda _slices, _rank: torch.zeros((2, 2048 // 128, 8))
    transform = _WL_FP8._column_of_each_expert(converted, BANKPAD_PARAM, stack_whole)
    with pytest.raises(Glm5NextExpertBankNotLoadableError) as refusal:
        transform([], 4)
    message = str(refusal.value)
    print(f"BANKPAD3_REFUSAL={message[:240]}")
    assert f"at column {column_of_4}" in message, (
        f"the refusal does not report the group's column, so the grid path did not "
        f"reach the group: {message}"
    )

    # THE BOUNDARY: at the registered degree the two answers agree and the grid path
    # returns without refusing. Without this the refusal above could be a blanket.
    registered_degree = 16
    registered_tp_per_ep = world // registered_degree
    registered_geometry = _WL_FP8.shard_geometry_for_grid(
        _bankpad_bank_geometry(ep_degree=registered_degree, world_size=world),
        BANKPAD_PARAM,
        DEFAULT_WEIGHT_BLOCK_SIZE,
    )
    _r, registered_column = _mesh_answers(world, registered_degree, 12)
    monkeypatch.setattr(
        _NPS,
        "get_neuron_ep_tp_group",
        lambda: _FixtureGroup(registered_column, registered_tp_per_ep),
    )
    rows = 2048 // 128
    accepted = _WL_FP8._column_of_each_expert(
        registered_geometry,
        BANKPAD_PARAM,
        lambda _slices, _rank: torch.zeros((2, rows, 8)),
    )([], 12)
    print(f"BANKPAD3_REGISTERED_ACCEPTED_SHAPE={tuple(accepted.shape)}")
    assert tuple(accepted.shape) == (2, rows // registered_tp_per_ep, 8), (
        f"the registered degree loaded {tuple(accepted.shape)}; at "
        f"{registered_tp_per_ep} ranks the grid's {rows} rows must divide to "
        f"{rows // registered_tp_per_ep}"
    )


def test_bankpad_the_grid_conversion_carries_the_degree_through_both_returns() -> None:
    """Item 4. BOTH returns of the conversion, because item 3 reaches only one.

    ``shard_geometry_for_grid``'s deferred branch has two exits -- the pad-is-None
    exit and the converted-pad exit -- and the repair had to touch both. Item 3
    drives the first (the bank declares no pad after this increment), so this item
    exists to read the second, which no behavioural path of the repaired bank takes
    any more. Exactly two readings, both the declared degree.

    WHY THIS REDS ON THE PRE-REPAIR CODE. Both exits omitted the field, so both
    returned the dataclass default of 1.
    """
    degree = 8
    block = DEFAULT_WEIGHT_BLOCK_SIZE[0]
    readings = []
    for label, pad in (("no_pad", None), ("converted_pad", block * 2)):
        geometry = _WL_FP8.DeferredShardGeometry(
            shard_dim=0,
            num_shards=4,
            pad_to_multiple_of=pad,
            expert_parallel_degree=degree,
        )
        converted = _WL_FP8.shard_geometry_for_grid(
            geometry, BANKPAD_PARAM, DEFAULT_WEIGHT_BLOCK_SIZE
        )
        readings.append((label, converted.expert_parallel_degree))
        print(
            f"BANKPAD4_{label.upper()}_DEGREE={converted.expert_parallel_degree} "
            f"PAD_IN={pad} PAD_OUT={converted.pad_to_multiple_of}"
        )
    assert len(readings) == 2, (
        f"this item read {len(readings)} of the conversion's two deferred exits"
    )
    assert all(value == degree for _label, value in readings), (
        f"the conversion did not carry the declared degree {degree} through both "
        f"returns: {readings}"
    )


# --------------------------------------------------------------------------- #
# ``inc-glm53f-054a`` hand-off item (iii): the first COMPLETING load in this file
# that carries a shared-expert module.
#
# WHY IT IS NEEDED. The landed shared-expert scale prep had never run through
# ``load_weights`` in any item here. Of the seven completing loads this file had,
# one carried a routed bank and none carried a shared expert -- every fixture that
# completes either sets ``n_shared_experts=0`` (``_stacked_config``,
# ``_shard_config``, ``_grid_shard_config``; ``_shard_config``'s docstring records
# that it is not a convenience) or is all-dense and so builds no MoE block at all
# (``_dense_config``). The two fixtures that do build the module both refused.
#
# NO NEW CHECKPOINT WRITER IS MINTED, and that is a measurement rather than a
# shortcut. Item (iii) asks for a 256-blocked checkpoint carrying a shared expert,
# and ``_deferred_checkpoint`` already writes one: its six MoE families come from
# :data:`DEFERRED_FAMILIES` at :data:`SHARED_INTERMEDIATE` and
# :data:`BANK_INTERMEDIATE` by :data:`DEFERRED_NARROW` -- 2048 and 512 by 256, all
# whole multiples of the consumer's block. A second writer differing only in the
# shared expert's width would be a copy of 200 lines with nothing new to say. What
# was missing was a LOAD that completes through it, so that is what this section
# adds: the same checkpoint at world size 1, where nothing shards and the only
# thing standing between the load and the prep was the retile.
#
# THE 128-BLOCK GRID IS THE POINT, not an accident of the fixture. The checkpoint
# holds one scale per 128-tile, exactly as the published one does, and the prep
# consumes the 256 public grid. So a completing load here is evidence that the
# load-path retile ran; without it this same load is ``-101``'s recorded refusal.
# --------------------------------------------------------------------------- #

#: World size 1 and expert-parallel degree 1, so every family loads whole and no
#: padding or column arithmetic stands between the checkpoint and the prep. The
#: sharded readings are ``-094``'s and ``-101``'s and are not repeated here.
BLOCKED_WORLD = 1
BLOCKED_EP_DEGREE = 1


def _load_blocked(
    directory: Path,
    monkeypatch,
    shared_experts: int = MINI_SHARED_EXPERTS,
) -> Glm5NextForConditionalGeneration:
    """Load the 256-blocked checkpoint at world 1, where nothing shards.

    ``shared_experts`` IS THE FIRING CONTROL'S ONE VARYING FIELD, the same field
    :func:`_deferred_config` varies for the same reason: at 0 no module of the
    class exists, so a reading taken at 1 is a reading of the shared expert and
    not of the fixture at large.
    """
    monkeypatch.setattr(_MODEL_FP8, "_resolve_world_size", lambda: BLOCKED_WORLD)
    monkeypatch.setattr(_MODEL_FP8, "_resolve_rank", lambda: 0)
    monkeypatch.setattr(
        _FACTORY, "_resolve_ep_degree", lambda given: BLOCKED_EP_DEGREE
    )
    monkeypatch.setattr(_NPS, "get_neuron_ep_rank", lambda: 0)
    monkeypatch.setattr(
        _NPS,
        "get_neuron_ep_tp_group",
        lambda: _FixtureGroup(0, BLOCKED_WORLD // BLOCKED_EP_DEGREE),
    )
    model = Glm5NextForConditionalGeneration(_deferred_config(shared_experts))
    assert model.world_size == BLOCKED_WORLD, (
        f"the model resolved world size {model.world_size}, not the patched "
        f"{BLOCKED_WORLD}, so this is not the load this item means to measure"
    )
    _seed_page_cache_signal()
    model.load_weights(str(directory), torch.device("cpu"), None)
    return model


def _modules_named(
    model: Glm5NextForConditionalGeneration, class_name: str
) -> list[tuple[str, torch.nn.Module]]:
    """Every ``(path, module)`` whose type has this name, from the tree itself.

    The class name is a STRING because that is what the prep loop's own gate
    compares (``model_fp8.py``'s ``hasattr`` on the type, read per module), and
    because importing the bank's class here would add an import for a count the
    tree already answers.
    """
    return [
        (path, module)
        for path, module in model.named_modules()
        if type(module).__name__ == class_name
    ]


def test_blocked_the_shared_expert_prep_completes_a_load_and_the_retile_ran(
    tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """Item (iii). The shared expert's prep runs inside a COMPLETING load.

    Five conjuncts, and the firing control that makes them readings.

    (1) The load completes. Until the load-path retile landed, this same
        checkpoint refused inside the prep, and ``-101`` attempt 2 recorded the
        refusal by name -- so completion is the evidence the retile ran, not a
        restatement of it.
    (2) Every shared-expert module carries three prepared scale operands, read
        through the class's own attribute name rather than a string here.
    (3) The retile's health record says all three projections were retiled, and
        the public grid it published is the one the weight's own extents imply,
        computed here from the consumer's block size rather than read back from
        the record.
    (4) Both losslessness counters read zero on this fixture. They are counters,
        not assertions: the grid writes a distinct value per 256 BLOCK along the
        shard dim, so a layout that dropped or invented a scale would move them.
        R6 item R-T2 changed that granularity from ``-095b``'s per-128-tile ramp
        and corrected this sentence with it -- on the ramp the coarsening rescales
        weight bytes, which is what these counters were reporting and what R-P1 now
        refuses; :func:`_pow2_block_grid_pattern` records the trade and
        :func:`test_blocked_a_ramp_scale_grid_refuses_instead_of_emitting_nan`
        keeps the ramp reading.
    (5) Every routed bank carries its prepared kernel operands. This is the plan's
        own sentence for this item -- the prep's arrival makes the loop visit the
        bank -- read on the module the loop visited.

    THE CONTROL: the same fixture with no shared expert. The load still
    completes and no module of the class exists, so conjuncts (2) to (4) are
    readings of the shared expert rather than of a load that would pass anyway.
    """
    directory, _overrides, _mappings = _deferred_checkpoint(tmp_path)
    block = _WL_FP8.consumer_block_quant_size()

    model = _load_blocked(directory, monkeypatch)

    shared = _modules_named(model, "Glm5NextSharedExperts")
    assert shared, (
        "this configuration built no Glm5NextSharedExperts module, so there is "
        "nothing here to read and the conjuncts below would pass vacuously"
    )

    for path, module in shared:
        # (2) the prep built three operands.
        prepared = getattr(module, Glm5NextSharedExperts.PREPARED_SCALE_OPERANDS_ATTR)
        assert len(prepared) == 3, (
            f"{path} carries {len(prepared)} prepared scale operands, not 3; the "
            f"prep builds one per projection and the load completed, so a "
            f"shortfall means a projection was skipped rather than refused"
        )

        # (3) the retile ran on all three, and published the implied grid.
        health = getattr(module, Glm5NextSharedExperts.SHARED_RETILE_HEALTH_ATTR)
        leaves = _scale_prep_leaves(module)
        assert len(leaves) == 3, (
            f"{path} offers {leaves} to the prep loop, not three leaves; this "
            f"item's arithmetic below is per projection"
        )
        for leaf in leaves:
            record = health[leaf]
            assert record["retiled"] is True, (
                f"{path}.{leaf} was NOT retiled: {record.get('reason')}. On this "
                f"fixture every MoE extent is a whole {block} block, so a skip "
                f"here means the retile could not read the extents it was given"
            )
            weight = getattr(module, leaf)
            rows, cols = int(weight.shape[0]), int(weight.shape[1])
            implied = (rows // block, cols // block)
            grid_name = f"{leaf[: -len(_WEIGHT_LEAF_SUFFIX)]}_{FP8_SCALE_SUFFIX}"
            grid = getattr(module, grid_name)
            assert tuple(grid.shape) == implied, (
                f"{path}.{grid_name} is {tuple(grid.shape)} after the load; a "
                f"[{rows},{cols}] weight implies the public grid {implied} at "
                f"the consumer's {block}-block granularity. A grid that survived "
                f"at checkpoint granularity is the refusal -101 recorded"
            )
            assert grid.dtype is torch.float32, (
                f"{path}.{grid_name} is {grid.dtype}; the scale grids stay fp32 "
                f"through the retile, which is what -091's own item pins"
            )

            # (4) the two counters, which can move.
            assert record["emitted_unsupplied"] == 0, (
                f"{path}.{leaf} emitted {record['emitted_unsupplied']} slots the "
                f"input did not supply, so the retiled layout does not decode "
                f"back to the grid it was given"
            )
            assert record["input_scales_dropped"] == 0, (
                f"{path}.{leaf} dropped {record['input_scales_dropped']} input "
                f"scales, so the coarser grid cannot reproduce them bit-exactly"
            )

    # (5) the loop visited every bank, which is what item (i)'s arrival changed.
    banks = _modules_named(model, "Glm5NextRoutedExperts")
    assert banks, (
        "this configuration built no routed bank, so conjunct (5) has nothing "
        "to read; first_k_dense_replace must leave at least one MoE layer"
    )
    bank_class = type(banks[0][1])
    for path, module in banks:
        prepared = getattr(module, bank_class.PREPARED_KERNEL_OPERANDS_ATTR, None)
        assert prepared, (
            f"{path} carries no prepared kernel operands after a completing "
            f"load. The prep loop's gate is a type test and this class now "
            f"defines prepare_scale_operands, so the loop must have visited it"
        )
        assert len(prepared) == 4, (
            f"{path} carries {len(prepared)} prepared kernel operands, not the "
            f"four block_quant_expert_mm takes: {sorted(prepared)}"
        )

    # THE CONTROL. Same checkpoint, same widths, no shared expert.
    control = _load_blocked(directory, monkeypatch, shared_experts=0)
    assert not _modules_named(control, "Glm5NextSharedExperts"), (
        "the control built a shared-expert module at n_shared_experts=0, so the "
        "field this control varies is not the field the tree reads and the "
        "readings above are not attributable to the shared expert"
    )
    assert _modules_named(control, "Glm5NextRoutedExperts"), (
        "the control built no routed bank either, so it varies more than the one "
        "field it declares and cannot isolate anything"
    )


def test_blocked_a_ramp_scale_grid_refuses_instead_of_emitting_nan(
    tmp_path, monkeypatch, single_rank_process_group
) -> None:
    """R6 item R-T2's pair: the grid the coarsening cannot reproduce is REFUSED.

    This is the reading the ramp grid used to carry and the pow2 grid above gives
    up, kept here where it belongs. The SAME checkpoint is written twice, differing
    in one field -- the scale grid's family -- and the two loads are compared:

    * the pow2 grid loads and completes, which is the arming half. Without it a
      refusal below could be any load failure wearing the right words.
    * the ramp grid, 1, 2, 3, ... per 128 tile, makes the first 256 block's ratio
      exactly 2, and an fp8 byte at the maximum doubled leaves what fp8-e4m3 holds.
      The landed cast is not saturating, so until R6 item R-P1 that produced a
      SILENT NaN which every shape reading passed straight over; now the retile
      refuses and names the expert, the 256 block, the 128 tile, the ratio, the
      retained scale, the bound and both health counters
      (``blockwise_fp8_retile.py:461-476``).

    A clamp is deliberately not the alternative: it would ship numbers the
    checkpoint does not contain (lead ruling ``LEAD-LOG.md`` §901).

    The refusal is read off the exception CHAIN rather than the outermost type,
    because the load path is entitled to wrap it; what this item claims is that a
    ``BlockwiseFp8RetileError`` is in that chain and that its sentence names the
    coordinates, not that nothing re-raises it.
    """
    from vllm_neuron.functional.moe.blockwise_fp8_retile import (
        BlockwiseFp8RetileError,
    )

    pow2_directory, pow2_overrides, _mappings = _deferred_checkpoint(tmp_path)
    ramp_directory, ramp_overrides, _ramp_mappings = _deferred_checkpoint(
        tmp_path, ramp_grids=True, name="deferred-ramp"
    )

    # THE ONE FIELD THIS ITEM VARIES, measured rather than declared: the two
    # checkpoints differ on the grids of the retiled families and nowhere else.
    # The comparison goes through fp32 because the weight tensors are fp8, where a
    # dtype-native equality is not something this file assumes it has.
    def _same(left: torch.Tensor, right: torch.Tensor) -> bool:
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            return False
        return bool(torch.equal(left.to(torch.float32), right.to(torch.float32)))

    assert sorted(ramp_overrides) == sorted(pow2_overrides), (
        "the two fixtures do not even hold the same keys, so they differ in more "
        "than the grid family this item varies"
    )
    differing = sorted(
        key
        for key, tensor in ramp_overrides.items()
        if not _same(tensor, pow2_overrides[key])
    )
    print(f"RAMPREFUSAL_KEYS_THAT_DIFFER={len(differing)}")
    assert differing, (
        "the ramp fixture and the pow2 fixture hold identical tensors, so this "
        "item varies nothing and the refusal below would not be attributable to "
        "the grid"
    )
    assert all(key.endswith(FP8_SCALE_SUFFIX) for key in differing), (
        f"the two fixtures differ on tensors that are not scale grids: "
        f"{[key for key in differing if not key.endswith(FP8_SCALE_SUFFIX)][:6]}. "
        f"This item varies the grid family and must vary nothing else"
    )

    # THE ARMING HALF. Same widths, same weights, pow2 grid: the load completes.
    armed = _load_blocked(pow2_directory, monkeypatch)
    assert _modules_named(armed, "Glm5NextSharedExperts"), (
        "the pow2 load built no shared-expert module, so the refusal below cannot "
        "be attributed to the grid the retile read"
    )

    with pytest.raises(Exception) as raised:  # noqa: B017 -- the chain is the claim
        _load_blocked(ramp_directory, monkeypatch)

    chain: list[BaseException] = []
    error: BaseException | None = raised.value
    while error is not None and error not in chain:
        chain.append(error)
        error = error.__cause__ or error.__context__
    types = [type(item).__name__ for item in chain]
    refusals = [item for item in chain if isinstance(item, BlockwiseFp8RetileError)]
    print(f"RAMPREFUSAL_EXCEPTION_CHAIN={types}")
    for item in refusals:
        print(f"RAMPREFUSAL_MESSAGE={str(item)[:400]}")
    assert refusals, (
        f"the ramp grid raised {types}, with no BlockwiseFp8RetileError anywhere in "
        f"the chain. Either the retile no longer refuses an unrepresentable rescale "
        f"-- in which case it is emitting NaN again -- or the load failed for some "
        f"other reason and this item is measuring that instead"
    )
    message = str(refusals[0])
    for phrase in (
        "REFUSES",
        "256-block",
        "128-tile",
        "ratio",
        "retained scale",
        "inexact_rescales",
        "input_scales_dropped",
    ):
        assert phrase in message, (
            f"the refusal does not say {phrase!r}: {message[:300]}. The whole point "
            f"of refusing rather than emitting NaN is that the message locates the "
            f"block and reports the counters"
        )
