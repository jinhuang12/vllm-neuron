"""`inc-glm53f-112`: the REAL dense load path publishes the checkpoint's own 128 grid.

WHY THIS FILE EXISTS. Round 1's review found that every existing miniature of this path uses
extents that are NOT whole ``256`` blocks, so they all took the skip arm of
``_publish_compute_frame_operands`` and none of them ran the coarsening. The real checkpoint's
extents ARE whole ``256`` blocks, so the real load ran it -- coarsened the grid to
``(K//256, N//256)`` and then handed it to ``to_kernel_scale_layout``, which at 128 granularity
wants ``(K//128, N//128)`` and raises. A green miniature suite and a broken product is exactly the
gap a miniature is supposed to close.

So this file's fixture uses WHOLE-256 EXTENTS on purpose (``H = I = 512``, which is two whole 256
blocks and four whole 128 tiles) and drives the real methods in the real order the load path uses
them: attach the weights and the checkpoint's own grids, ``retile_checkpoint_scale_grids()``, then
``prepare_scale_operands`` built exactly as ``_run_load_time_preps`` builds it. If the publish and
the prep ever disagree about granularity again, item 1 raises where the product raises.

Nothing here needs a device or a checkpoint file: the operands are bound onto a real module, which
is what the loader does, and no forward is run.

THE LAST TWO ITEMS ARE ABOUT WHERE THE TRANSPOSE COPY IS TAKEN. A real 64-rank load raised
``Expected self.is_contiguous() to be true, but got false`` on the publisher's own transpose,
because the operands are on the device by then and a Neuron tensor refuses ``.contiguous()`` on a
strided view -- the rule the runner already writes down at
``vllm/worker/neuron_model_runner.py:4136``. Every relayout the load path takes now goes through
one helper that copies to the host first, and the second item wants ZERO that do not.

Because this file runs on the host, where a strided copy has always worked, neither item can prove
the refusal is gone. One holds the published frame's values and strides unchanged; the other keeps
the unguarded form from coming back. The proof that the load path clears is a real load.
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

#: WHOLE 256 BLOCKS ON PURPOSE -- this is the property every earlier miniature lacked. 512 is two
#: whole 256 blocks and four whole 128 tiles, 768 is three and six, so this fixture takes the SAME
#: arm of the publisher that the real checkpoint takes.
#:
#: NON-SQUARE ON PURPOSE TOO. The publish is followed by a transpose, and with H == I every shape
#: reading below would hold under a swapped axis pair. 512 by 768 makes the transpose visible.
HIDDEN = 512
INTERMEDIATE = 768

_LEAVES = ("gate_proj_weight", "up_proj_weight", "down_proj_weight")


def _emit(item: str, body: str) -> None:
    print(f"P112|{item}|{body}", flush=True)


def _grid_name(leaf: str) -> str:
    """The sibling grid attribute name, by the product's own rule."""
    return f"{leaf[: -len('_weight')]}_{FP8_SCALE_SUFFIX}"


def _leaf_shape(leaf: str) -> tuple[int, int]:
    """``[H, I]`` for gate and up, ``[I, H]`` for down -- the loader's own layout."""
    if leaf == "down_proj_weight":
        return (INTERMEDIATE, HIDDEN)
    return (HIDDEN, INTERMEDIATE)


def _module(*, coarse_grids: bool = False) -> nn.Module:
    """A shared-expert module with the three weights and their grids ATTACHED, as a load leaves it.

    ``coarse_grids=True`` attaches ``(rows//256, cols//256)`` grids instead of the checkpoint's
    own, which is the control: a grid built for a different consumer must be refused rather than
    reshaped.
    """
    torch.manual_seed(112)
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
    """The real path, in the real order, at extents the real checkpoint has.

    This is the item that fails if the publish and the prep ever disagree about granularity again.
    The prep call is built the way ``_run_load_time_preps`` builds it -- by keyword, from the
    module's OWN attributes after the publish -- so a wrong grid reaches it exactly as it would on
    a real load.
    """
    if dense_consumer_block_quant_size() != SCALE_BLOCK_SIZE:
        raise AssertionError(
            f"the load path's DENSE consumer granularity is "
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
        _emit(
            "I1_PUBLISH",
            f"leaf={leaf} loader_frame={tuple(record['loader_frame'])} "
            f"published={record['published']} retiled={record['retiled']} "
            f"checkpoint_grid={tuple(record['checkpoint_grid'])} "
            f"public_grid={tuple(record['public_grid'])} "
            f"compute_grid={tuple(record['compute_grid'])} "
            f"inexact_rescales={record['inexact_rescales']}",
        )
        assert record["published"] is True, (
            f"{leaf} was not published: {record.get('reason')}. At [{rows},{cols}] both extents "
            f"are whole {SCALE_BLOCK_SIZE} blocks, so a skip here means the step could not read "
            f"the extents it was given"
        )
        assert record["retiled"] is False, (
            f"{leaf} reports a retile. Since `inc-glm53f-112` the dense path coarsens nothing: a "
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
        # The publish is followed by the unconditional transpose, so what the module CARRIES is
        # the published grid transposed. That is the frame the prep reads two calls below.
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

    # THE CALL THE PRODUCT MAKES, built the way the product builds it.
    operands: dict[str, torch.Tensor] = {}
    for leaf in _LEAVES:
        operands[leaf] = getattr(module, leaf)
        operands[f"{leaf[: -len('_weight')]}_scale"] = getattr(module, _grid_name(leaf))
    built = module.prepare_scale_operands(**operands)
    prepared = getattr(module, Glm5NextSharedExperts.PREPARED_SCALE_OPERANDS_ATTR)
    _emit(
        "I1_PREP",
        f"built={built} prepared_keys={sorted(prepared)} "
        f"shapes={{{', '.join(f'{k}: {tuple(v.shape)}' for k, v in sorted(prepared.items()))}}}",
    )
    assert built == len(_LEAVES), f"the prep built {built} operands, not {len(_LEAVES)}"
    for leaf in _LEAVES:
        rows, cols = _leaf_shape(leaf)
        name = leaf[: -len("_weight")]
        # The prep reads the STORED weight's extents, and the publish transposed them, so the
        # kernel operand is declared against (cols, rows). The extents are deliberately unequal,
        # so this reading fails if the two frames are ever swapped.
        want = kernel_scale_shape(cols, rows)
        assert name in prepared, f"the prep built no operand named {name!r}: {sorted(prepared)}"
        assert tuple(prepared[name].shape) == want, (
            f"{name}'s prepared operand is {tuple(prepared[name].shape)}; "
            f"kernel_scale_shape({cols}, {rows}) declares {want}"
        )


def test_a_grid_at_any_other_granularity_is_refused_by_name() -> None:
    """The control: without it, item 1 would pass on a step that accepted anything.

    A ``(rows//256, cols//256)`` grid is what the OLD path published. Handing it to the step now
    must refuse by name rather than reshape it, because such a grid was built for a different
    consumer and every downstream shape check would pass it.
    """
    module = _module(coarse_grids=True)
    with pytest.raises(Glm5NextSharedExpertRouteError) as refusal:
        module.retile_checkpoint_scale_grids()
    message = str(refusal.value)
    _emit("I2_CONTROL_REFUSAL", f"message={message[:200]!r}")
    for phrase in ("publishes", str(CHECKPOINT_TILE), "different consumer"):
        assert phrase in message, (
            f"the refusal does not say {phrase!r}: {message[:300]}. A refusal that does not name "
            f"the granularity it wanted sends a reader to the wrong place"
        )


def test_both_dense_classes_route_to_one_publisher() -> None:
    """The dense MLP and the shared expert share ONE definition of this step.

    Item 1 drives the shared expert. This item is why that is enough: both classes' methods call
    the same module-level function, so a second copy cannot have drifted. Read off the code
    objects rather than asserted in prose.
    """
    names = {
        cls.__name__: cls.retile_checkpoint_scale_grids.__code__.co_names
        for cls in (Glm5NextSharedExperts, Glm5NextDenseMLP)
    }
    _emit("I3_ONE_PUBLISHER", f"co_names={names}")
    for cls_name, co_names in names.items():
        assert "_publish_compute_frame_operands" in co_names, (
            f"{cls_name}.retile_checkpoint_scale_grids does not call "
            f"_publish_compute_frame_operands; it has its own body, which is the drift the shared "
            f"definition exists to prevent"
        )


def test_the_published_frame_keeps_its_values_and_its_strides() -> None:
    """Taking the transpose copy on the host publishes the frame the device copy published.

    The publisher no longer materialises the transpose on the device, because a Neuron tensor
    refuses ``.contiguous()`` on a strided view. This item is the guard that the change is a
    relayout of WHERE the copy happens and nothing else: for every projection the published
    weight and grid must match, byte for byte and stride for stride, what the direct expression
    built from the same inputs. An oracle recomputed here from the fixture's own tensors, so
    nothing is compared against a stored golden value.
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
        # STEP 1 does not touch the weight, so its oracle is the raw tensor transposed. The grid
        # IS compensated first, so its oracle runs the same compensator the publisher runs.
        want_weight = raw_weight.t().contiguous()
        want_grid = compensate_block_scales(raw_grid).scale_inv.t().contiguous()
        got_weight = getattr(module, leaf).data
        got_grid = getattr(module, _grid_name(leaf))
        _emit(
            "I4_FRAME_IDENTITY",
            f"leaf={leaf} weight_shape={tuple(got_weight.shape)} "
            f"weight_stride={got_weight.stride()} weight_contiguous={got_weight.is_contiguous()} "
            f"grid_shape={tuple(got_grid.shape)} grid_stride={got_grid.stride()} "
            f"grid_contiguous={got_grid.is_contiguous()}",
        )
        # fp8 has no ordering, so the bytes are compared as int8. Both sides are contiguous by
        # the assertions below, which is what makes the view legal.
        assert got_weight.is_contiguous(), (
            f"{leaf}'s published weight is not contiguous, strides {got_weight.stride()}; the "
            f"kernel reads it as a dense buffer"
        )
        assert want_weight.is_contiguous()
        assert torch.equal(got_weight.view(torch.int8), want_weight.view(torch.int8)), (
            f"{leaf}'s published weight differs from the transpose of the tensor the fixture "
            f"bound; taking the copy on the host must move no byte"
        )
        assert got_weight.stride() == want_weight.stride(), (
            f"{leaf}'s published weight has strides {got_weight.stride()} where the direct "
            f"transpose gives {want_weight.stride()}"
        )
        assert got_weight.dtype is want_weight.dtype, (
            f"{leaf}'s published weight is {got_weight.dtype}, not {want_weight.dtype}"
        )
        assert got_grid.is_contiguous(), (
            f"{_grid_name(leaf)} is not contiguous, strides {got_grid.stride()}"
        )
        assert torch.equal(got_grid, want_grid), (
            f"{_grid_name(leaf)}'s published grid differs from the compensated grid transposed; "
            f"taking the copy on the host must change no number"
        )
        assert got_grid.stride() == want_grid.stride(), (
            f"{_grid_name(leaf)} has strides {got_grid.stride()} where the direct transpose "
            f"gives {want_grid.stride()}"
        )


def test_the_load_path_materialises_no_strided_view_on_a_device_tensor() -> None:
    """No prep the load path runs materialises a strided view on a device tensor. Want ZERO.

    A Neuron tensor refuses ``.contiguous()`` on a transposed, permuted, sliced or broadcast view,
    so every such copy has to be taken on a host copy. This item reads the source instead of
    running it.

    THE SCAN SET IS DERIVED, NOT TYPED, IN BOTH DIRECTIONS. ``_run_load_time_preps`` enrols a prep
    by ``hasattr(type(module), "<name>")``, so those literals ARE the load path's entry points; this
    item reads them out of that function and follows call edges to their transitive closure. The
    closure crosses module edges: a prep imports its producer inside its own body, and this item
    reads those import statements, so the kernel-operand builders and the retile producer are
    scanned where they are defined. A prep or a producer added later is therefore covered without
    editing this item.

    THE GUARD IS A STATEMENT, SO THE SCAN IS ORDER-AWARE. A prep takes host copies of its operands
    once and moves the finished operands to the device once; the relayouts in between carry no
    ``.cpu()`` of their own and are legal because they run after that hop. So this item walks each
    function in source order, tracks which names are host-bound at each point, and carries that flag
    across call edges -- a callee handed only host operands computes on the host wherever it lives.
    Reading receiver chains alone would both flag those relayouts and miss a device operand
    materialised before the hop.

    IT CARRIES THREE CONTROLS. The walk must leave this file; the predicate must find
    materialisations in the imported producers, or the widening did not reach them; and the same
    predicate run with no host boundary must still find the serve-time transpose in ``forward``,
    which is a different question and out of scope. Without them a ZERO would mean nothing.
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
    # These copy a repeated or broadcast view on their own, with no strided step in
    # the chain to give them away: torch builds `repeat_interleave` as an expand and
    # then a contiguous copy, which is the refused copy one call deeper.
    MATERIALISES_ALONE = {"repeat_interleave", "repeat", "tile"}
    TO_THE_HOST = {"_on_the_host", "cpu"}
    TO_THE_DEVICE = {"_on_the_device"}
    # A tensor read off a module is device-resident, so a name bound from `getattr`
    # or from an attribute of `self` starts on the device and must be hopped like a
    # parameter. Without this the walk would call a producer host-side because ONE of
    # its arguments had been hopped while another came straight off the module.
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

    # ---- ONE PASS PER FUNCTION, IN SOURCE ORDER, because the guard is a statement and
    # not a receiver chain any more. A prep now takes host copies of its operands once
    # and moves the finished operands to the device once, so the materialisations in
    # between carry no ``.cpu()`` of their own; what makes them legal is that they run
    # AFTER the hop. A scan that only looked at the receiver chain would read those as
    # unguarded and would equally miss a device operand materialised BEFORE the hop.
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
            # A prep's parameters ARE the device-resident operands -- that is the
            # convention every caller's pre-flight states -- so they start here and
            # leave only by being hopped.
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
                # ---- THE CALL EDGE, and the flag it carries. A callee handed only
                # host operands computes on the host, wherever it is defined, so its
                # own materialisations are legal there. That is how the imported
                # producers are covered without editing them.
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
                # A callee is entered on the host only when every operand it is handed
                # is there. One hopped argument beside one straight off the module is
                # exactly the shape of a dequant that copies on the device.
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
    _emit("I5_LOAD_PATH_ENTRIES", f"read_from_{ENTRY}={sorted(set(seeds_read))}")
    _emit("I5_SCAN_SET", f"functions={len(visited)} modules={scanned_modules}")
    _emit("I5_MATERIALISATIONS_SEEN", f"{dict(raw_by_module)}")
    _emit("I5_PRODUCERS_ENTERED_ON_THE_HOST", f"count={len(entered_on_the_host)} {entered_on_the_host}")
    _emit("I5_UNGUARDED_ON_THE_LOAD_PATH", f"count={len(unguarded)} {sorted(unguarded)}")

    assert seeds_read, (
        f"no dispatch name was read out of {ENTRY}, so the scan set is that one function and "
        f"this item would pass without looking at a single prep"
    )
    imported = [dotted for dotted in scanned_modules if dotted != ROOT_MODULE]
    assert imported, (
        "the walk never left this file, so the producers the preps import were not scanned at "
        "all and the ZERO below would only be about one module"
    )
    assert sum(count for dotted, count in raw_by_module.items() if dotted != ROOT_MODULE), (
        f"the predicate found no materialisation of a strided view in any imported module "
        f"({imported}), so the widened scan is not reading the producers it reached; the two "
        f"kernel-scale-operand builders and the scale-layout bridge each broadcast with "
        f"expand(...).contiguous() and must be seen"
    )
    assert entered_on_the_host, (
        "no imported producer is entered on the host, so the preps are still handing device "
        "operands to the retile producer and the kernel-operand builders; every one of them "
        "computes on the operand it is given"
    )
    assert unguarded == [], (
        f"the load path materialises a strided view on a device-resident operand, at "
        f"{sorted(unguarded)}. On a Neuron tensor that raises rather than copying, which is the "
        f"refusal a real 64-rank load hit; the prep must take its host copies before this point"
    )

    # ---- THE POSITIVE CONTROL, unchanged in purpose. The same predicate is run over
    # this module with no host boundary at all, where it must still find the serve-time
    # transpose in ``forward`` -- a different question and out of scope here. A scan
    # that finds nothing anywhere is broken rather than clean.
    everywhere = []
    for inner in ast.walk(modules[ROOT_MODULE][0]):
        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
            chain = receiver_chain(inner)
            if inner.func.attr in MATERIALISERS and STRIDED & set(chain) and "cpu" not in chain:
                everywhere.append((inner.lineno, ".".join(reversed(chain))))
    _emit("I5_PREDICATE_FIRES_IN_THIS_FILE", f"count={len(everywhere)} {sorted(everywhere)}")
    assert everywhere, (
        "the predicate finds no strided materialisation anywhere in this file, not even the "
        "serve-time one in `forward`. It is reading the wrong file or the chain walk is broken, "
        "and the ZERO above would mean nothing"
    )
