"""The dense load path publishes the checkpoint's own 128-granularity scale grid.

The fixture uses extents that are a whole number of 256-wide blocks (512 and 768),
which is what the real checkpoint has, so the publisher takes the same arm it takes on a
real load. A grid coarsened to ``(K//256, N//256)`` would be refused by
``to_kernel_scale_layout``, which indexes at 128, so the publish and the following prep
must agree on granularity.
The methods run in the order the load path runs them: attach the weights and the
checkpoint's grids, ``retile_checkpoint_scale_grids()``, then ``prepare_scale_operands``
built the way ``_run_load_time_preps`` builds it. Nothing here needs a device or a
checkpoint file, and no forward is run.

The last two tests are about where the transpose copy is taken. A Neuron tensor refuses
``.contiguous()`` on a strided view, so every relayout on the load path has to copy on
the host first. One test holds the published frame's values and strides unchanged; the
other reads the load path's source and wants no unguarded materialisation. Neither can
prove the device-side refusal is gone, because this file runs on the host.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm_neuron.functional.blockwise_fp8_mm import (
    SCALE_BLOCK_SIZE,
    kernel_scale_shape,
    scale_grid_shape,
)
from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
from vllm_neuron.model.glm5_next.model_fp8 import (
    Glm5NextDenseMLP,
    Glm5NextSharedExperts,
    Glm5NextSharedExpertRouteError,
)
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
    FP8_SCALE_SUFFIX,
    compensate_block_scales,
    dense_consumer_block_quant_size,
)

_FP8 = torch.float8_e4m3fn

#: The checkpoint's own tiling, which is what the grids the loader attaches are on.
CHECKPOINT_TILE = 128

#: Both extents hold a whole number of 256-wide blocks -- 512 is two of them and four
#: 128-wide tiles, 768 is three and six -- so the publisher takes the arm the real
#: checkpoint takes. They are also unequal, because the publish is followed by a
#: transpose and a square fixture would read the same under a swapped axis pair.
HIDDEN = 512
INTERMEDIATE = 768

_LEAVES = ("gate_proj_weight", "up_proj_weight", "down_proj_weight")


def _grid_name(leaf: str) -> str:
    """The sibling grid attribute name, by the product's own rule."""
    return f"{leaf[: -len('_weight')]}_{FP8_SCALE_SUFFIX}"


def _leaf_shape(leaf: str) -> tuple[int, int]:
    """``[H, I]`` for gate and up, ``[I, H]`` for down -- the loader's own layout."""
    if leaf == "down_proj_weight":
        return (INTERMEDIATE, HIDDEN)
    return (HIDDEN, INTERMEDIATE)


def _module(*, coarse_grids: bool = False) -> nn.Module:
    """A shared-expert module with the three weights and their grids attached.

    ``coarse_grids=True`` attaches ``(rows//256, cols//256)`` grids instead of the
    checkpoint's own: a grid built for a different consumer must be refused rather than
    reshaped.
    """
    torch.manual_seed(0)
    module = Glm5NextSharedExperts(
        Glm5NextTextConfig(
            hidden_size=HIDDEN,
            moe_intermediate_size=INTERMEDIATE,
            n_shared_experts=1,
        )
    )
    for leaf in _LEAVES:
        rows, cols = _leaf_shape(leaf)
        weight = (torch.randint(1, 8, (rows, cols)).to(torch.float32)).to(_FP8)
        module.register_parameter(leaf, nn.Parameter(weight, requires_grad=False))
        if coarse_grids:
            grid_shape = (rows // (2 * CHECKPOINT_TILE), cols // (2 * CHECKPOINT_TILE))
        else:
            grid_shape = (rows // CHECKPOINT_TILE, cols // CHECKPOINT_TILE)
        # Powers of two, so nothing here depends on rounding.
        grid = torch.ldexp(
            torch.ones(grid_shape, dtype=torch.float32),
            torch.arange(grid_shape[0] * grid_shape[1]).reshape(grid_shape) % 3 - 1,
        )
        setattr(module, _grid_name(leaf), grid)
    return module


def test_a_whole_256_checkpoint_publishes_at_128_and_the_prep_accepts_it() -> None:
    """The publish and the following prep agree on granularity at real extents.

    The prep call is built the way ``_run_load_time_preps`` builds it, by keyword from the
    module's own attributes after the publish, so a wrong grid reaches it exactly as it
    would on a real load.
    """
    if dense_consumer_block_quant_size() != SCALE_BLOCK_SIZE:
        raise AssertionError(
            f"the load path's dense consumer granularity is "
            f"{dense_consumer_block_quant_size()} while the kernel indexes "
            f"{SCALE_BLOCK_SIZE}; these two must be one number or this whole file "
            f"is testing a coincidence. The routed bank's block is a different "
            f"number and is not this file's subject"
        )
    module = _module()
    published = module.retile_checkpoint_scale_grids()
    health = getattr(module, Glm5NextSharedExperts.SHARED_RETILE_HEALTH_ATTR)

    for leaf in _LEAVES:
        rows, cols = _leaf_shape(leaf)
        record = health[leaf]
        assert record["published"] is True, (
            f"{leaf} was not published: {record.get('reason')}. At [{rows},{cols}] both extents "
            f"are whole {SCALE_BLOCK_SIZE} blocks, so a skip here means the step could not read "
            f"the extents it was given"
        )
        assert record["retiled"] is False, (
            f"{leaf} reports a retile. The dense path coarsens nothing: a "
            f"True here means the 256 coarsening came back"
        )
        assert tuple(record["public_grid"]) == tuple(record["checkpoint_grid"]), (
            f"{leaf} published {tuple(record['public_grid'])} from the checkpoint grid "
            f"{tuple(record['checkpoint_grid'])}; this step publishes the checkpoint's own grid "
            f"and must not reshape it"
        )
        assert tuple(record["public_grid"]) == scale_grid_shape(rows, cols), (
            f"{leaf} published {tuple(record['public_grid'])} where the kernel's own "
            f"scale_grid_shape({rows}, {cols}) declares {scale_grid_shape(rows, cols)}"
        )
        # The publish is followed by an unconditional transpose, so the grid the module
        # carries is the published grid transposed. That is the frame the prep reads.
        carried = getattr(module, _grid_name(leaf))
        assert tuple(carried.shape) == tuple(reversed(scale_grid_shape(rows, cols))), (
            f"{leaf}'s grid is {tuple(carried.shape)} on the module; the compute frame is the "
            f"published grid transposed"
        )
        assert carried.dtype is torch.float32, (
            f"{leaf}'s grid is {carried.dtype}; the grids stay fp32 through this step"
        )

    assert published == len(_LEAVES), (
        f"the publish reported {published} projections, not {len(_LEAVES)}"
    )

    # The call the load path makes, built the way it builds it.
    operands: dict[str, torch.Tensor] = {}
    for leaf in _LEAVES:
        operands[leaf] = getattr(module, leaf)
        operands[f"{leaf[: -len('_weight')]}_scale"] = getattr(module, _grid_name(leaf))
    built = module.prepare_scale_operands(**operands)
    prepared = getattr(module, Glm5NextSharedExperts.PREPARED_SCALE_OPERANDS_ATTR)
    assert built == len(_LEAVES), f"the prep built {built} operands, not {len(_LEAVES)}"
    for leaf in _LEAVES:
        rows, cols = _leaf_shape(leaf)
        name = leaf[: -len("_weight")]
        # The prep reads the stored weight's extents, and the publish transposed them, so
        # the kernel operand is declared against (cols, rows). The extents are unequal, so
        # this reading fails if the two frames are ever swapped.
        expected = kernel_scale_shape(cols, rows)
        assert name in prepared, f"the prep built no operand named {name!r}: {sorted(prepared)}"
        assert tuple(prepared[name].shape) == expected, (
            f"{name}'s prepared operand is {tuple(prepared[name].shape)}; "
            f"kernel_scale_shape({cols}, {rows}) declares {expected}"
        )


def test_a_grid_at_any_other_granularity_is_refused_by_name() -> None:
    """A grid at another granularity is refused by name rather than reshaped.

    A ``(rows//256, cols//256)`` grid was built for a different consumer, and every
    downstream shape check would accept it, so the publisher has to refuse it here.
    """
    module = _module(coarse_grids=True)
    with pytest.raises(Glm5NextSharedExpertRouteError) as refusal:
        module.retile_checkpoint_scale_grids()
    message = str(refusal.value)
    for phrase in ("publishes", str(CHECKPOINT_TILE), "different consumer"):
        assert phrase in message, (
            f"the refusal does not say {phrase!r}: {message[:300]}. A refusal that does not name "
            f"the granularity it wanted sends a reader to the wrong place"
        )


def test_both_dense_classes_route_to_one_publisher() -> None:
    """The dense MLP and the shared expert share one definition of this step.

    Both classes' methods call the same module-level function, read off their code
    objects, so testing the publisher through one class covers the other.
    """
    names = {
        cls.__name__: cls.retile_checkpoint_scale_grids.__code__.co_names
        for cls in (Glm5NextSharedExperts, Glm5NextDenseMLP)
    }
    for cls_name, co_names in names.items():
        assert "_publish_compute_frame_operands" in co_names, (
            f"{cls_name}.retile_checkpoint_scale_grids does not call "
            f"_publish_compute_frame_operands; it has its own body, which is the drift the shared "
            f"definition exists to prevent"
        )


def test_the_published_frame_keeps_its_values_and_its_strides() -> None:
    """The published weight and grid are the direct transpose, byte and stride alike.

    The publisher takes its transpose copy on a host copy, because a Neuron tensor refuses
    ``.contiguous()`` on a strided view. Where the copy happens must change nothing else,
    so every projection is compared against the direct expression over the same inputs,
    recomputed here rather than stored.
    """
    module = _module()
    raw = {
        leaf: (
            getattr(module, leaf).detach().clone(),
            getattr(module, _grid_name(leaf)).detach().clone(),
        )
        for leaf in _LEAVES
    }

    module.retile_checkpoint_scale_grids()

    for leaf in _LEAVES:
        raw_weight, raw_grid = raw[leaf]
        # The publish does not touch the weight, so its reference is the raw tensor
        # transposed. The grid is compensated first, so its reference runs the same
        # compensator the publisher runs.
        expected_weight = raw_weight.t().contiguous()
        expected_grid = compensate_block_scales(raw_grid).scale_inv.t().contiguous()
        got_weight = getattr(module, leaf).data
        got_grid = getattr(module, _grid_name(leaf))
        # fp8 has no ordering, so the bytes are compared as int8. Both sides are asserted
        # contiguous, which is what makes the view legal.
        assert got_weight.is_contiguous(), (
            f"{leaf}'s published weight is not contiguous, strides {got_weight.stride()}; the "
            f"kernel reads it as a dense buffer"
        )
        assert expected_weight.is_contiguous()
        assert torch.equal(got_weight.view(torch.int8), expected_weight.view(torch.int8)), (
            f"{leaf}'s published weight differs from the transpose of the tensor the fixture "
            f"bound; taking the copy on the host must move no byte"
        )
        assert got_weight.stride() == expected_weight.stride(), (
            f"{leaf}'s published weight has strides {got_weight.stride()} where the direct "
            f"transpose gives {expected_weight.stride()}"
        )
        assert got_weight.dtype is expected_weight.dtype, (
            f"{leaf}'s published weight is {got_weight.dtype}, not {expected_weight.dtype}"
        )
        assert got_grid.is_contiguous(), (
            f"{_grid_name(leaf)} is not contiguous, strides {got_grid.stride()}"
        )
        assert torch.equal(got_grid, expected_grid), (
            f"{_grid_name(leaf)}'s published grid differs from the compensated grid transposed; "
            f"taking the copy on the host must change no number"
        )
        assert got_grid.stride() == expected_grid.stride(), (
            f"{_grid_name(leaf)} has strides {got_grid.stride()} where the direct transpose "
            f"gives {expected_grid.stride()}"
        )


def test_the_load_path_materialises_no_strided_view_on_a_device_tensor() -> None:
    """No prep on the load path materialises a strided view on a device tensor.

    A Neuron tensor refuses ``.contiguous()`` on a transposed, permuted, sliced or
    broadcast view, so every such copy has to be taken on a host copy. This reads the
    source rather than running it.

    The scan set is derived at both ends. ``_run_load_time_preps`` enrols a prep by
    name, so those literals are the entry points; they are read out of that function and
    followed along call edges, including the imports a prep makes inside its own body,
    so the kernel-operand builders and the retile producer are scanned where they are
    defined and a prep added later is covered without editing this test.

    The scan is order-aware, because the host hop is a statement rather than part of a
    receiver chain: a prep copies its operands to the host once and moves the finished
    operands back once, and the relayouts in between are legal because they run after
    that hop. So each function is walked in source order, tracking which names are
    host-bound, and the flag is carried across call edges -- a callee handed only host
    operands computes on the host wherever it is defined.

    It carries a control at the end: the same predicate, run with no host boundary, must
    still find the serve-time transpose in ``forward``. Without it an empty reading could
    just as well mean the scan read the wrong file.
    """
    import ast
    import collections
    import inspect
    import pathlib

    from vllm_neuron.model.glm5_next import model_fp8

    ENTRY = "_run_load_time_preps"
    ROOT_MODULE = "vllm_neuron.model.glm5_next.model_fp8"
    PACKAGE = "vllm_neuron"
    STRIDED = {
        "t", "transpose", "permute", "T", "expand", "unfold", "broadcast_to",
        "as_strided", "__getitem__",
    }
    MATERIALISERS = {"contiguous", "reshape", "view", "flatten", "ravel"}
    # These copy a repeated or broadcast view on their own, with no strided step in the
    # chain to give them away: torch builds ``repeat_interleave`` as an expand and then a
    # contiguous copy, which is the refused copy one call deeper.
    MATERIALISES_ALONE = {"repeat_interleave", "repeat", "tile"}
    TO_THE_HOST = {"_on_the_host", "cpu"}
    TO_THE_DEVICE = {"_on_the_device"}
    # A tensor read off a module is device-resident, so a name bound from ``getattr``
    # starts on the device and must be hopped like a parameter. Without this the walk
    # would enter a producer host-side when one of its arguments had been hopped while
    # another came straight off the module.
    FROM_THE_MODULE = {"getattr"}

    package_root = pathlib.Path(inspect.getsourcefile(model_fp8)).parents[3]

    def source_of(dotted: str):
        """``(tree, {name: [defs]}, {function: {bound_name: module}})`` for one module."""
        path = package_root.joinpath(*dotted.split("."))
        path = path.with_suffix(".py") if not path.is_dir() else path / "__init__.py"
        if not path.is_file():
            return None
        tree = ast.parse(path.read_text())
        defs = collections.defaultdict(list)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs[node.name].append(node)
        # Import bindings, whether written at module level or inside a function body.
        # The load path imports its producers inside the prep that calls them, so a
        # module-level-only reading would stop at this file's own edge.
        imports: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    imports[alias.asname or alias.name] = node.module
        return tree, defs, imports

    modules = {ROOT_MODULE: source_of(ROOT_MODULE)}
    assert modules[ROOT_MODULE], f"{ROOT_MODULE} did not resolve under {package_root}"
    assert modules[ROOT_MODULE][1][ENTRY], f"{ENTRY} is not defined in {ROOT_MODULE}"

    def module_of(dotted: str):
        if dotted not in modules:
            modules[dotted] = source_of(dotted)
        return modules[dotted]

    def receiver_chain(call: ast.Call) -> list[str]:
        """Every attribute name on the receiver chain, nearest first."""
        # A subscript is followed rather than ending the walk, because a slice is a
        # strided view like any other and `x[a:b].contiguous()` is the same copy.
        names: list[str] = []
        cursor: ast.expr = call
        while True:
            if isinstance(cursor, ast.Call):
                cursor = cursor.func
            elif isinstance(cursor, ast.Attribute):
                names.append(cursor.attr)
                cursor = cursor.value
            elif isinstance(cursor, ast.Subscript):
                names.append("__getitem__")
                cursor = cursor.value
            else:
                return names

    def receiver_root(call: ast.Call) -> str | None:
        """The variable the materialised value ultimately comes from, if it is a plain name."""
        cursor: ast.expr = call
        while True:
            if isinstance(cursor, ast.Call):
                cursor = cursor.func
            elif isinstance(cursor, (ast.Attribute, ast.Subscript)):
                cursor = cursor.value
            elif isinstance(cursor, ast.Name):
                return cursor.id
            else:
                return None

    def names_in(node: ast.AST) -> set[str]:
        return {inner.id for inner in ast.walk(node) if isinstance(inner, ast.Name)}

    def calls_named(node: ast.AST, wanted: set[str]) -> bool:
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call):
                if isinstance(inner.func, ast.Name) and inner.func.id in wanted:
                    return True
                if isinstance(inner.func, ast.Attribute) and inner.func.attr in wanted:
                    return True
        return False

    def targets_of(assign: ast.Assign) -> set[str]:
        return {
            inner.id
            for target in assign.targets
            for inner in ast.walk(target)
            if isinstance(inner, ast.Name)
        }

    # One pass per function, in source order, because the host hop is a statement: the
    # materialisations between the hop and the move back carry no ``.cpu()`` of their own
    # and are legal because of where they sit.
    unguarded: list[tuple[str, int, str]] = []
    raw_by_module: dict[str, int] = collections.defaultdict(int)
    visited: set[tuple[str, str, bool]] = set()
    frontier = [(ROOT_MODULE, ENTRY, False)]
    seeds_read: list[str] = []

    while frontier:
        dotted, name, host_side = frontier.pop()
        if (dotted, name, host_side) in visited:
            continue
        visited.add((dotted, name, host_side))
        resolved = module_of(dotted)
        if not resolved:
            continue
        _tree, defs, imports = resolved
        for node in defs.get(name, []):
            host_names: set[str] = set()
            # A prep's parameters are the device-resident operands, so they start here
            # and leave only by being hopped.
            device_names: set[str] = {
                argument.arg
                for argument in getattr(node.args, "args", [])
                if argument.arg != "self"
            }
            ordered = sorted(
                (
                    inner
                    for inner in ast.walk(node)
                    if isinstance(inner, (ast.Assign, ast.Call))
                ),
                key=lambda inner: (inner.lineno, inner.col_offset),
            )
            for inner in ordered:
                if isinstance(inner, ast.Assign):
                    written = targets_of(inner)
                    if calls_named(inner.value, TO_THE_HOST):
                        host_names |= written
                        device_names -= written
                    elif calls_named(inner.value, TO_THE_DEVICE) or calls_named(
                        inner.value, FROM_THE_MODULE
                    ):
                        host_names -= written
                        device_names |= written
                    elif names_in(inner.value) & device_names:
                        host_names -= written
                        device_names |= written
                    elif names_in(inner.value) & host_names:
                        host_names |= written
                        device_names -= written
                    continue
                chain = receiver_chain(inner)
                if isinstance(inner.func, ast.Attribute) and (
                    inner.func.attr in MATERIALISES_ALONE
                    or (inner.func.attr in MATERIALISERS and STRIDED & set(chain))
                ):
                    raw_by_module[dotted] += 1
                    root = receiver_root(inner)
                    guarded = (
                        host_side
                        or "cpu" in chain
                        or (root is not None and root in host_names)
                    )
                    if not guarded:
                        unguarded.append(
                            (f"{dotted}.{name}", inner.lineno, ".".join(reversed(chain)))
                        )
                # The call edge carries the host flag: a callee handed only host
                # operands computes on the host wherever it is defined, which is how the
                # imported producers are covered.
                called = (
                    inner.func.id
                    if isinstance(inner.func, ast.Name)
                    else inner.func.attr if isinstance(inner.func, ast.Attribute) else None
                )
                if called is None:
                    continue
                arguments = list(inner.args) + [kw.value for kw in inner.keywords]
                handed = {
                    argument_name
                    for argument in arguments
                    for argument_name in names_in(argument)
                }
                # A callee is entered on the host only when every operand it is handed is
                # there. One hopped argument beside one straight off the module is the
                # shape of a dequantisation that copies on the device.
                child_host = host_side or (
                    bool(handed & host_names) and not (handed & device_names)
                )
                if called in defs:
                    frontier.append((dotted, called, child_host))
                elif called in imports and imports[called].startswith(PACKAGE):
                    frontier.append((imports[called], called, child_host))
        # The entry point dispatches its preps by name, as string literals.
        if name == ENTRY:
            for inner in ast.walk(defs[ENTRY][0]):
                if isinstance(inner, ast.Constant) and inner.value in defs:
                    seeds_read.append(inner.value)
                    frontier.append((dotted, inner.value, False))

    scanned_modules = sorted({dotted for dotted, _name, _host in visited})
    entered_on_the_host = sorted(
        f"{dotted}.{name}" for dotted, name, host in visited if host and dotted != ROOT_MODULE
    )

    assert seeds_read, (
        f"no dispatch name was read out of {ENTRY}, so the scan set is that one function "
        f"and this test would pass without looking at a single prep"
    )
    imported = [dotted for dotted in scanned_modules if dotted != ROOT_MODULE]
    assert imported, (
        "the walk never left this module, so the producers the preps import were not "
        "scanned and the reading below would cover one module only"
    )
    assert sum(count for dotted, count in raw_by_module.items() if dotted != ROOT_MODULE), (
        f"the scan found no materialisation of a strided view in any imported module "
        f"({imported}), so it is not reading the producers it reached: the two "
        f"kernel-scale-operand builders and the scale-layout bridge each broadcast with "
        f"expand(...).contiguous() and must be seen"
    )
    assert entered_on_the_host, (
        "no imported producer is entered on the host, so the preps are handing device "
        "operands to the retile producer and the kernel-operand builders, each of which "
        "computes on the operand it is given"
    )
    assert unguarded == [], (
        f"the load path materialises a strided view on a device-resident operand, at "
        f"{sorted(unguarded)}. On a Neuron tensor that raises rather than copying, so the "
        f"prep must take its host copies before this point"
    )

    # The control for the empty reading above: the same predicate, run over this module with
    # no host boundary at all, must still find the serve-time transpose in ``forward``. That
    # copy is a different question and out of scope here, but a predicate that finds nothing
    # anywhere is broken rather than clean, and then the empty reading would say nothing.
    everywhere = []
    for inner in ast.walk(modules[ROOT_MODULE][0]):
        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
            chain = receiver_chain(inner)
            if inner.func.attr in MATERIALISERS and STRIDED & set(chain) and "cpu" not in chain:
                everywhere.append((inner.lineno, ".".join(reversed(chain))))
    assert everywhere, (
        "the predicate finds no strided materialisation anywhere in this module, not even the "
        "serve-time one in `forward`, so it is reading the wrong file or the receiver-chain "
        "walk is broken and the empty reading above means nothing"
    )
