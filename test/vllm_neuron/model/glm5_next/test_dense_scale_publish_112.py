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
