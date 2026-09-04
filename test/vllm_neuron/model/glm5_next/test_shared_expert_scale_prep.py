# SPDX-License-Identifier: Apache-2.0
"""Acceptance for `inc-glm53f-090` -- the kernel scale operand is built once.

**Five items, one per declared conjunct, and no ``parametrize`` decorator in this
file** (campaign rule D1.2). Each item names the component whose behaviour it
certifies (D1.4).

WHY THIS FILE EXISTS. `inc-glm53f-026`'s dense bridge,
``to_kernel_scale_layout``, allocates and then scatters ONE ELEMENT AT A TIME
(``blockwise_fp8_mm.py:340-342`` and ``:343-347``), and `-026`'s seam called it on
every entry. The shared-expert path enters that seam three times per call, so at
this campaign's dense geometry the operand was rebuilt by hundreds of one-element
device writes per shared-expert call, on every layer that takes the route, every
forward step -- review finding B26-M2. A block scale never changes after a
checkpoint load, so `-090` builds the operand once, at load time, and hands it to
the seam by keyword.

THE READINGS ARE PIPE-DELIMITED AND PREFIXED ``SCALEPREP|``. That prefix is this
file's own: `-088`'s readings are bracketed (``[label]``) and `-089`'s carry
``PRODUCTION|``, and an extractor written for either would read zero here and a
round that predicted zero would pass while proving nothing.

WHAT THE COUNTED ZERO IS AND WHAT IT IS NOT (D1.5). The build counter counts
builds THE SEAM performs. ``prepare_scale_operands`` calls the bridge directly,
so its own one-time build is deliberately not counted -- the claim being settled
is about the PER-FORWARD path, and the control below moves the count to 3 by
withholding the operand at the seam while still running the prep. Withholding the
prep instead would raise, which is conjunct 3's subject and not a control.

Run::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
    pytest test/vllm_neuron/model/glm5_next/test_shared_expert_scale_prep.py \\
      --timeout 60 -p no:cacheprovider
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from vllm_neuron.functional.blockwise_fp8_mm import (
    BLOCK_QUANT_SIZE,
    TILE_SIZE,
    BlockwiseFp8MmError,
    blockwise_fp8_mm,
    dispatch_counters,
    kernel_scale_shape,
    reset_dispatch_counters,
    reset_scale_layout_builds,
    scale_grid_shape,
    scale_layout_builds,
    to_kernel_scale_layout,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

#: The pinned checkpoint fixture, and its registered digest. Checked before the
#: file is parsed, on ``test_experts.py:1266-1288``'s landed idiom: the
#: quantisation policy these call sites route on must be the campaign's
#: registered one and not a value this test invented.
FIXTURE_SHA256 = "5ed24d23a3e14a038352e1bdc21fd25fc90ff2291d3f6a310acf5d4036665a1d"


def _impl():
    """Import the modeling module INSIDE a test body, never at import time.

    This package's uniform convention (``test_experts.py:110-120``,
    ``test_kv_spec.py:157-161``). ``test_factory.py``'s C03 no longer measures the
    session -- `inc-glm53f-031` repaired it to a subprocess -- so a module-level
    import here would break nothing; the convention is kept because it costs four
    lines and a file that departs from it reads like an oversight. The functional
    imports above are NOT this module and stay at the top.
    """
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8

# --------------------------------------------------------------------------- #
# THE DECLARED GEOMETRY, and why it is not the production one.                 #
# --------------------------------------------------------------------------- #
# Every conjunct here measures a COUNT, an exact equality or a named refusal --
# none measures a number that varies with extent -- so the extents only have to
# be admissible, and a small admissible geometry runs the NKI simulator three
# times per case inside the declared 60-second timeout. The production geometry
# is recorded as a READING instead (the scatter count it implies), so the record
# carries what the repair is worth without the test paying for it.
HIDDEN = 2 * BLOCK_QUANT_SIZE  # H = 512
INTERMEDIATE = 2 * BLOCK_QUANT_SIZE  # I = 512
TOKENS = TILE_SIZE  # T = 128, a whole number of TILE_SIZE rows
FP8 = torch.float8_e4m3fn

#: The dense geometry the review priced, from `-022` part 1's shape 1
#: `moe288-top8` (``increments/evidence-022-part1.md:105``). Used ONLY to record
#: how many one-element scatters the repair removes per shared-expert call.
PRODUCTION_HIDDEN = 4096
PRODUCTION_INTERMEDIATE = 2048

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _pinned_fixture() -> dict:
    """The pinned checkpoint config, digest-checked before it is parsed."""
    fixture = Path(__file__).resolve().parent / "fixtures" / "config.json"
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    if digest != FIXTURE_SHA256:
        raise AssertionError(
            f"pinned fixture digest moved: {digest} != {FIXTURE_SHA256}. The "
            f"quantisation policy these call sites route on would no longer be "
            f"the campaign's registered one."
        )
    return json.loads(fixture.read_text())


def _fixture(seed: int = 0):
    """One shared-expert module and one set of operands, seeded.

    The weights are fp8 bytes and the scales are the PUBLIC ``[K//256, N//256]``
    grids ``scale_grid_shape`` declares. Values are seeded so every equality
    below is reproducible rather than incidental.

    The SwiGLU bound is left at the config's own default deliberately: no reading
    in this file measures it, and both arms of every differential below share
    whatever it is, so the bound cannot influence a single comparison. `-033`
    owns that value and its provenance test.
    """
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    torch.manual_seed(seed)
    text_config = Glm5NextTextConfig(
        hidden_size=HIDDEN,
        moe_intermediate_size=INTERMEDIATE,
        n_shared_experts=1,
    )
    module = _impl().Glm5NextSharedExperts(text_config)
    gu_grid = scale_grid_shape(HIDDEN, INTERMEDIATE)
    dn_grid = scale_grid_shape(INTERMEDIATE, HIDDEN)
    operands = {
        "hidden_states": torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16),
        "gate_proj_weight": torch.randn(HIDDEN, INTERMEDIATE).to(FP8),
        "up_proj_weight": torch.randn(HIDDEN, INTERMEDIATE).to(FP8),
        "down_proj_weight": torch.randn(INTERMEDIATE, HIDDEN).to(FP8),
        "gate_proj_scale": torch.rand(*gu_grid, dtype=torch.float32) + 0.5,
        "up_proj_scale": torch.rand(*gu_grid, dtype=torch.float32) + 0.5,
        "down_proj_scale": torch.rand(*dn_grid, dtype=torch.float32) + 0.5,
    }
    return module, operands


def _quant_config() -> object:
    """``Glm5NextQuantConfig`` for the pinned checkpoint -- nothing hand-fed.

    ``test_experts.py:1266-1291``'s landed form, carried rather than reinvented:
    the policy comes from the digest-checked fixture through
    ``Glm5NextConfig.from_configs``, so this file cannot route on a block shape
    the campaign never registered.
    """
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig

    return _impl().Glm5NextQuantConfig.from_model_config(
        Glm5NextConfig.from_configs(_pinned_fixture())
    )


def _run_prepared(module, operands, quant_config) -> torch.Tensor:
    """One shared-expert call with the operands built once, beforehand."""
    module.prepare_scale_operands(
        operands["gate_proj_weight"],
        operands["up_proj_weight"],
        operands["down_proj_weight"],
        operands["gate_proj_scale"],
        operands["up_proj_scale"],
        operands["down_proj_scale"],
    )
    return module.shared_expert_mm(
        hidden_states=operands["hidden_states"],
        gate_proj_weight=operands["gate_proj_weight"],
        up_proj_weight=operands["up_proj_weight"],
        down_proj_weight=operands["down_proj_weight"],
        gate_proj_scale=operands["gate_proj_scale"],
        up_proj_scale=operands["up_proj_scale"],
        down_proj_scale=operands["down_proj_scale"],
        quant_config=quant_config,
    )


def _withhold_at_the_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the seam build the operand itself, WITHOUT skipping the prep.

    THE CONTROL, and its shape is the point. ``shared_expert_mm`` imports the
    seam function-locally, so replacing the module attribute replaces what the
    three call sites reach. The replacement forwards every argument EXCEPT
    ``prebuilt_scale_t``, which is exactly the pre-`-090` call form -- so the
    control exercises the same three dispatches over the same operands and
    differs in one thing only: who builds the scale operand.

    THE MODULE IS FETCHED BY IMPORTLIB, not by ``import ... as``. The package
    re-exports the seam FUNCTION under the submodule's own name, so
    ``import vllm_neuron.functional.blockwise_fp8_mm as seam`` binds the function
    and ``seam.blockwise_fp8_mm`` raises ``AttributeError``. ``import_module``
    returns the module object, which is the same object
    ``shared_expert_mm``'s function-local ``from ... import ...`` reads its name
    out of -- so patching it is what actually reaches the three call sites.
    """
    import importlib

    seam = importlib.import_module("vllm_neuron.functional.blockwise_fp8_mm")
    real = seam.blockwise_fp8_mm

    def without_prebuilt(x, weight, weight_scale, *, prebuilt_scale_t=None):
        return real(x, weight, weight_scale)

    monkeypatch.setattr(seam, "blockwise_fp8_mm", without_prebuilt)


# --------------------------------------------------------------------------- #
# CONJUNCT 1 -- the per-forward build count, a counted zero with its control.   #
# --------------------------------------------------------------------------- #
def test_the_seam_builds_no_scale_operand_per_forward_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B26-M2 remedy (i): count layout builds per forward step.

    Certifies (D1.4): ``to_kernel_scale_layout`` and `-090`'s build counter.
    """
    print(
        "SCALEPREP|conjunct1_per_forward_build_count"
        "|certifies=to_kernel_scale_layout and the build counter"
    )
    quant_config = _quant_config()

    module, operands = _fixture()
    reset_scale_layout_builds()
    reset_dispatch_counters()
    _run_prepared(module, operands, quant_config)
    prepared_builds = scale_layout_builds()
    prepared_dispatch = dispatch_counters()

    control_module, control_operands = _fixture()
    with monkeypatch.context() as patched:
        _withhold_at_the_seam(patched)
        reset_scale_layout_builds()
        reset_dispatch_counters()
        _run_prepared(control_module, control_operands, quant_config)
        control_builds = scale_layout_builds()
        control_dispatch = dispatch_counters()

    gu_grid = scale_grid_shape(HIDDEN, INTERMEDIATE)
    dn_grid = scale_grid_shape(INTERMEDIATE, HIDDEN)
    scatters_here = 2 * gu_grid[0] * gu_grid[1] + dn_grid[0] * dn_grid[1]
    prod_gu = scale_grid_shape(PRODUCTION_HIDDEN, PRODUCTION_INTERMEDIATE)
    prod_dn = scale_grid_shape(PRODUCTION_INTERMEDIATE, PRODUCTION_HIDDEN)
    scatters_production = 2 * prod_gu[0] * prod_gu[1] + prod_dn[0] * prod_dn[1]

    print(
        f"SCALEPREP|conjunct1|builds_per_forward={prepared_builds}"
        f"|control_builds={control_builds}"
        f"|dispatch_prepared={prepared_dispatch}"
        f"|dispatch_control={control_dispatch}"
    )
    print(
        f"SCALEPREP|conjunct1|one_element_scatters_removed_per_call_here="
        f"{scatters_here}"
        f"|at_production_geometry={scatters_production}"
        f"|production_H={PRODUCTION_HIDDEN}|production_I={PRODUCTION_INTERMEDIATE}"
    )

    assert prepared_builds == 0, (
        "the prebuilt operand was supplied at all three sites, so the seam must "
        f"not have built one; it built {prepared_builds}"
    )
    # THE CONTROL MOVES THE COUNT (D1.5). Without this reading the zero above
    # would also be produced by a counter that never increments.
    assert control_builds == 3, (
        "withholding the operand at the seam must make the seam build one per "
        f"projection, so 3; it built {control_builds}"
    )
    # Both arms took the kernel route the same number of times, so the zero is
    # not a zero reached by skipping the seam.
    assert prepared_dispatch == control_dispatch == (3, 0)


# --------------------------------------------------------------------------- #
# CONJUNCT 2 -- the operand is the same bytes, and so is the output.           #
# --------------------------------------------------------------------------- #
def test_the_prebuilt_operand_and_the_output_are_the_same_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moving the build must not move one bit of arithmetic.

    Certifies (D1.4): ``shared_expert_mm``.
    """
    print(
        "SCALEPREP|conjunct2_same_bytes"
        "|certifies=shared_expert_mm"
    )
    quant_config = _quant_config()

    module, operands = _fixture()
    prepared_output = _run_prepared(module, operands, quant_config)

    # Every operand, against the bridge run directly on the same public grid.
    equal_rows = 0
    max_operand_diff = 0.0
    for name, (rows, cols) in (
        ("gate_proj", (HIDDEN, INTERMEDIATE)),
        ("up_proj", (HIDDEN, INTERMEDIATE)),
        ("down_proj", (INTERMEDIATE, HIDDEN)),
    ):
        want = to_kernel_scale_layout(operands[f"{name}_scale"], rows, cols)
        got = module._prepared_scale_operand(name)
        max_operand_diff = max(
            max_operand_diff, float((got - want).abs().max())
        )
        if torch.equal(got, want):
            equal_rows += 1

    control_module, control_operands = _fixture()
    with monkeypatch.context() as patched:
        _withhold_at_the_seam(patched)
        control_output = _run_prepared(
            control_module, control_operands, quant_config
        )

    output_max_abs_diff = float((prepared_output - control_output).abs().max())

    print(
        f"SCALEPREP|conjunct2|operands_element_equal={equal_rows}/3"
        f"|max_operand_abs_diff={max_operand_diff}"
        f"|operand_shape={tuple(module._prepared_scale_operand('gate_proj').shape)}"
        f"|declared_shape={kernel_scale_shape(HIDDEN, INTERMEDIATE)}"
    )
    print(
        f"SCALEPREP|conjunct2|output_max_abs_diff={output_max_abs_diff}"
        f"|output_shape={tuple(prepared_output.shape)}"
        f"|output_dtype={prepared_output.dtype}"
        f"|output_is_finite={bool(torch.isfinite(prepared_output).all())}"
    )

    # Non-vacuity: an all-zero output would satisfy a difference of zero.
    assert float(prepared_output.abs().max()) > 0.0
    assert equal_rows == 3, f"only {equal_rows} of 3 operands matched the bridge"
    assert max_operand_diff == 0.0
    # EXACT, and no tolerance is introduced: both paths compute one function
    # from one set of bytes, so anything but zero is a real defect.
    assert output_max_abs_diff == 0.0


# --------------------------------------------------------------------------- #
# CONJUNCT 3 -- two named refusals.                                            #
# --------------------------------------------------------------------------- #
def test_both_named_refusals_fire_by_name() -> None:
    """A caller that skips the prep, and one that prepares the wrong shape.

    Certifies (D1.4): ``_prepared_scale_operand``'s refusal and
    ``_checked_prebuilt_scale``'s refusal.
    """
    print(
        "SCALEPREP|conjunct3_named_refusals"
        "|certifies=the prep refusal and the seam's shape refusal"
    )
    quant_config = _quant_config()
    module, operands = _fixture()
    route_error = _impl().Glm5NextSharedExpertRouteError

    # REFUSAL 1 -- the prep did not run. The `_prepared_weight` form.
    with pytest.raises(route_error) as skipped:
        module.shared_expert_mm(
            hidden_states=operands["hidden_states"],
            gate_proj_weight=operands["gate_proj_weight"],
            up_proj_weight=operands["up_proj_weight"],
            down_proj_weight=operands["down_proj_weight"],
            gate_proj_scale=operands["gate_proj_scale"],
            up_proj_scale=operands["up_proj_scale"],
            down_proj_scale=operands["down_proj_scale"],
            quant_config=quant_config,
        )
    names_the_method = "prepare_scale_operands()" in str(skipped.value)

    # REFUSAL 2 -- an operand prepared for a different weight.
    wrong = torch.zeros(TILE_SIZE, 99, dtype=torch.float32)
    with pytest.raises(BlockwiseFp8MmError) as mis_shaped:
        blockwise_fp8_mm(
            operands["hidden_states"],
            operands["gate_proj_weight"],
            operands["gate_proj_scale"],
            prebuilt_scale_t=wrong,
        )
    names_the_shape_helper = "kernel_scale_shape" in str(mis_shaped.value)

    print(
        f"SCALEPREP|conjunct3|refusal1_type={type(skipped.value).__name__}"
        f"|names_the_method={names_the_method}"
        f"|refusal2_type={type(mis_shaped.value).__name__}"
        f"|names_kernel_scale_shape={names_the_shape_helper}"
        f"|refusals_fired=2/2"
    )

    assert names_the_method, str(skipped.value)
    assert names_the_shape_helper, str(mis_shaped.value)


# --------------------------------------------------------------------------- #
# CONJUNCT 4 -- `-026`'s landed acceptance, re-run whole and unchanged.        #
# --------------------------------------------------------------------------- #
def test_the_landed_dense_seam_acceptance_still_passes_whole() -> None:
    """The default path is unchanged, so `-026`'s own suite needs no edit.

    Certifies (D1.4): `-026`'s landed acceptance, ``test_blockwise_fp8_mm.py``.

    The item count is READ from the run and printed, never restated here: a
    number written into this file could agree with a suite that had silently
    lost an item.
    """
    print(
        "SCALEPREP|conjunct4_landed_suite"
        "|certifies=inc-glm53f-026's landed acceptance, re-run whole"
    )
    target = (
        Path(__file__).resolve().parents[3]
        / "vllm_neuron"
        / "functional"
        / "test_blockwise_fp8_mm.py"
    )
    assert target.is_file(), f"{target} is not a file"

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", str(target), "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    tail = completed.stdout.strip().splitlines()[-1] if completed.stdout else ""
    passed = 0
    for token in tail.replace(",", " ").split():
        if token.isdigit():
            passed = int(token)
            break

    print(
        f"SCALEPREP|conjunct4|exit={completed.returncode}"
        f"|items_passed_read_from_the_run={passed}"
        f"|summary_line={tail!r}"
    )

    assert completed.returncode == 0, completed.stdout[-3000:]
    # A floor, and labelled as one: the equality this conjunct owes is "no item
    # added or removed", which the exit code plus a positive count settles. The
    # exact number is recorded above for the next round to compare against.
    assert passed > 0, "the landed suite reported no passing item"


# --------------------------------------------------------------------------- #
# CONJUNCT 5 -- the route predicate (D13 form R-2).                            #
# --------------------------------------------------------------------------- #
def test_the_route_predicate_counts_one_dispatch_and_no_fallback_per_call() -> None:
    """An operand moved to load time that stopped reaching the kernel would
    satisfy every exact reading above. This is the reading that would break.

    Certifies (D1.4): the ``wrap_nki`` seam `-026` authors.
    """
    print(
        "SCALEPREP|conjunct5_route_predicate"
        "|certifies=the wrap_nki seam inc-glm53f-026 authors"
    )
    module, operands = _fixture()
    module.prepare_scale_operands(
        operands["gate_proj_weight"],
        operands["up_proj_weight"],
        operands["down_proj_weight"],
        operands["gate_proj_scale"],
        operands["up_proj_scale"],
        operands["down_proj_scale"],
    )

    # PER CALL, which is what the predicate declares.
    reset_dispatch_counters()
    blockwise_fp8_mm(
        operands["hidden_states"],
        operands["gate_proj_weight"],
        operands["gate_proj_scale"],
        prebuilt_scale_t=module._prepared_scale_operand("gate_proj"),
    )
    per_call = dispatch_counters()

    # And over one whole shared-expert call, the section's declared 3.
    reset_dispatch_counters()
    _run_prepared(module, operands, _quant_config())
    per_shared_expert_call = dispatch_counters()

    route_available = can_run_kernel(operands["hidden_states"])

    print(
        f"SCALEPREP|conjunct5|per_call_nki_dispatch={per_call[0]}"
        f"|per_call_torch_fallback={per_call[1]}"
        f"|per_shared_expert_call={per_shared_expert_call}"
        f"|can_run_kernel={route_available}"
    )

    assert route_available is True
    assert per_call == (1, 0), f"per call the seam read {per_call}, wanted (1, 0)"
    assert per_shared_expert_call == (3, 0)
