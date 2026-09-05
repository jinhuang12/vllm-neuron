# SPDX-License-Identifier: Apache-2.0
"""`inc-glm53f-023` acceptance — WP6 block-fp8 recognition (the dispatcher gap).

THE DECLARED PREDICATE (increment plan revision 29, block `#### inc-glm53f-023`),
quoted so nothing here can drift from it:

    the model resolves a block-fp8 method from `weight_block_size [128,128]` in
    **1/1** cases and reports `(block_h, block_w) == (128, 128)` exactly; an
    unsupported `weight_block_size` raises in **1/1** cases. Baseline recorded in
    the same test: `weight_block_size` has **0** occurrences in `vllm_neuron` at
    the pin (P47), so this increment is the whole recognition path.

Four counted conjuncts, therefore, one pytest item each (test layout rule 6: one
item per counted conjunct, NO `parametrize`, so the item count is derivable
before the run). A fifth item guards plan section 11 constraint B.6, the vendor
enum this campaign may not widen.

TWO FALSE-PASS DOORS, BOTH MEASURED AT THE UNMODIFIED PARENT `6affd98` BEFORE A
LINE OF SOURCE WAS WRITTEN (`increments/probe-023-parent-readings.py`, 71
readings, exit 0). Both are the reason this file asserts what it asserts:

1. **The SPEC already reported `(128, 128)` at the parent.** Parent reading
   `R4_DOOR_spec_pair_already_equals_128_128=True`: `quantization.py` landed
   before this increment and already parsed the checkpoint's block shape into a
   `QuantizationSpec`. So an assertion that reads the pair off the SPEC passes at
   the parent and certifies nothing. Every reading below is taken off the
   resolved **METHOD**, which the same probe shows did not exist at the parent
   (`R3_parent_quantconfig_constructed=False`).
2. **A malformed block size already raised at the parent.** Parent readings
   `R5_parent_three_dims` and `R5_parent_scalar` both RAISED, while
   `R5_parent_64_64='ACCEPTED weight_block_size=(64, 64)'`. So "unsupported"
   must be a WELL-FORMED shape with no authored path — `[64, 64]` — never a
   malformed one, or conjunct 3 measures the landed parser's shape validation
   instead of this increment's method resolution.

D1.4 is honoured per conjunct: each item names, in its docstring, the
file-plus-symbol whose behaviour its number certifies. D1.5 is honoured per
conjunct: each item drives a control that MOVES the reading when the property
under test is false, and prints both sides.

FIXTURE PROVENANCE: `fixtures/config.json`, the trimmed real checkpoint config
pinned by `inc-glm53f-008` and digest-pinned again here, derives from
HuggingFace `zai-org/GLM-5.3-Flash` revision `04c4e9e9…`. Its
`quantization_config` is the checkpoint's own — `quant_method "fp8"`,
`activation_scheme "dynamic"`, `weight_block_size [128, 128]`. This file
performs ZERO network access and touches no weight tensor.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from vllm_neuron.model.glm5_next import quantization as qz
from vllm_neuron.model.glm5_next.config import Glm5NextConfig


def _impl():
    """Import the modeling module INSIDE a test body, never at import time.

    This is `test_kv_spec.py:157-161`'s idiom, adopted for the same reason and
    NOT optional here. `test_factory.py`'s C03 asserts
    `"vllm_neuron.model.glm5_next.model_fp8" not in sys.modules`, which is what
    certifies `factory.py`'s lazy import -- the property that lets the arch class
    be looked up without allocating a 45-layer stack. This file sorts BEFORE
    `test_factory.py` (`b` < `f`), so a module-level import here populates
    `sys.modules` first and breaks that assertion for the whole directory.

    Measured, not assumed: with the import at module level, a whole-directory run
    reported `1 failed, 93 passed`, the failure being exactly that C03; with this
    helper, 94 passed. `quantization` stays a module-level import -- C03 names
    the modeling module only, and this file needs `qz` at class scope for
    `pytest.raises` and for the monkeypatch target.
    """
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------
#: repo/test/vllm_neuron/model/glm5_next/<this file> -> parents[4] is the repo.
REPO_ROOT = Path(__file__).resolve().parents[4]

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "config.json"

#: Pinned by `inc-glm53f-008`; repeated so a silent fixture edit fails loudly
#: here too rather than moving a declared value.
FIXTURE_SHA256 = "5ed24d23a3e14a038352e1bdc21fd25fc90ff2291d3f6a310acf5d4036665a1d"

#: The campaign's target base — `release-0.24.0.1.1.0`, DECISIONS section 1.
#: This is the revision P47's baseline is a statement ABOUT, so the baseline is
#: read out of git at this commit rather than from the working tree: the working
#: tree's count is nonzero the moment this increment lands, while the pin's is a
#: settled historical fact that stays true forever.
PIN = "f8abae640a43824c1dc73aed3cf2f67b83bce507"

#: The declared expected block shape, and the whole supported set.
EXPECTED_BLOCK_SHAPE = (128, 128)

#: Well-formed (two positive ints, so the landed parser accepts it) and
#: unsupported (not in the authored set). Both halves are load-bearing — see
#: false-pass door 2 in the module docstring.
UNSUPPORTED_BLOCK_SIZE = [64, 64]


def _raw() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def raw() -> dict:
    digest = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    assert digest == FIXTURE_SHA256, (
        f"fixture digest moved: {digest} != {FIXTURE_SHA256}"
    )
    return _raw()


def _config_from(raw_dict: dict) -> Glm5NextConfig:
    return Glm5NextConfig.from_configs(copy.deepcopy(raw_dict))


def _quant_config_from(raw_dict: dict):
    """The recognition path end to end, exactly as a call site would walk it.

    HF `quantization_config` -> `Glm5NextConfig` (`config.py`) ->
    `QuantizationSpec` (`quantization.py`) -> `Glm5NextQuantConfig`
    (`model_fp8.py`). Nothing is hand-fed at any hop.

    Return type deliberately unannotated: naming `Glm5NextQuantConfig` in the
    signature would need the modeling module at import time, which is exactly
    what `_impl` exists to avoid.
    """
    return _impl().Glm5NextQuantConfig.from_model_config(_config_from(raw_dict))


# ---------------------------------------------------------------------------
# Conjunct 1 — a block-fp8 METHOD resolves, 1/1
# ---------------------------------------------------------------------------
def test_block_quant_recognition_resolves_a_block_fp8_method(raw) -> None:
    """C01 — 1/1: the pinned checkpoint config resolves a block-fp8 method.

    D1.4 certifying component:
    `vllm_neuron/model/glm5_next/quantization.py::resolve_quant_method`, its
    `scheme is QuantScheme.FP8_BLOCK_DYNAMIC` arm, reached through
    `model_fp8.py::Glm5NextQuantConfig.from_model_config`. That pair, and nothing
    else, converts a parsed spec into a method.

    D1.5 moving control: the same config with `quantization_config` REMOVED. The
    resolved-method count must fall to 0, so "a method resolved" is not something
    this path returns unconditionally.
    """
    resolved = 0
    of = 0

    of += 1
    quant_config = _quant_config_from(raw)
    method = quant_config.get_quant_method(layer_index=3, prefix="mlp.experts")
    if method is not None:
        resolved += 1

    print(f"c01_resolved={resolved} c01_of={of}")
    print(f"c01_method_repr={method!r}")
    print(f"c01_method_type={type(method).__name__}")
    print(f"c01_scheme={None if method is None else method.scheme.value}")
    print(f"c01_is_block_quantized={quant_config.is_block_quantized}")

    assert of == 1
    assert resolved == 1, "the pinned checkpoint config must resolve a method"
    assert isinstance(method, qz.BlockFp8QuantMethod)
    assert method.scheme is qz.QuantScheme.FP8_BLOCK_DYNAMIC
    assert quant_config.is_block_quantized is True

    # --- moving control: no quantization_config at all -------------------
    unquantized_raw = copy.deepcopy(raw)
    unquantized_raw.pop("quantization_config", None)
    control_config = _quant_config_from(unquantized_raw)
    control_method = control_config.get_quant_method(
        layer_index=3, prefix="mlp.experts"
    )
    control_resolved = 0 if control_method is None else 1

    print(f"c01_control_resolved={control_resolved}")
    print(f"c01_control_spec_is_none={control_config.spec is None}")
    print(f"c01_control_is_block_quantized={control_config.is_block_quantized}")
    print(f"c01_control_MOVES={resolved} -> {control_resolved}")

    assert control_resolved == 0, "control did not move; C01 would be vacuous"
    assert control_config.spec is None
    assert control_config.is_block_quantized is False

    # --- second control: a spec whose SCHEME is NONE ---------------------
    # Reaches the `scheme is QuantScheme.NONE` arm rather than the absent-spec
    # arm, so the two early returns are separately exercised.
    none_spec = qz.QuantizationSpec(
        linear_scheme=qz.QuantScheme.NONE,
        kv_cache_scheme=qz.QuantScheme.NONE,
    )
    none_scheme_method = qz.resolve_quant_method(none_spec)
    print(f"c01_control_none_scheme_resolved={0 if none_scheme_method is None else 1}")
    assert none_scheme_method is None


# ---------------------------------------------------------------------------
# Conjunct 2 — the method reports (block_h, block_w) == (128, 128) exactly
# ---------------------------------------------------------------------------
def test_block_quant_recognition_reports_the_block_shape_exactly(
    raw, monkeypatch
) -> None:
    """C02 — `(block_h, block_w) == (128, 128)` exactly, off the METHOD.

    D1.4 certifying component:
    `quantization.py::BlockFp8QuantMethod.block_h` / `.block_w`, populated by
    `resolve_quant_method`'s `int(block[0])` / `int(block[1])` read of
    `spec.weight_block_size`.

    NOT off the spec. Parent reading `R4_DOOR_spec_pair_already_equals_128_128`
    was True at `6affd98`, so a spec-side assertion is vacuous — see false-pass
    door 1 in the module docstring. This item asserts the spec ALSO carries the
    pair only to record that the two agree, and never as the load-bearing
    reading.

    D1.5 moving control: with the supported set widened to admit `(256, 256)`,
    a `[256, 256]` config must report `(256, 256)`. That shows the pair is READ
    from the checkpoint rather than being a constant this arch prints — which is
    the only way a single-supported-shape equality can be non-vacuous.
    """
    quant_config = _quant_config_from(raw)
    method = quant_config.get_quant_method(layer_index=3, prefix="mlp.experts")
    assert method is not None

    pair = (method.block_h, method.block_w)
    print(f"c02_pair={pair}")
    print(f"c02_block_shape={method.block_shape}")
    print(f"c02_expected={EXPECTED_BLOCK_SHAPE}")
    print(f"c02_quant_config_block_shape={quant_config.block_shape}")
    print(f"c02_recorded_spec_pair={quant_config.spec.weight_block_size}")
    print(f"c02_fixture_declared={_raw()['quantization_config']['weight_block_size']}")

    assert pair == EXPECTED_BLOCK_SHAPE
    assert method.block_shape == EXPECTED_BLOCK_SHAPE
    assert quant_config.block_shape == EXPECTED_BLOCK_SHAPE
    assert isinstance(method.block_h, int) and isinstance(method.block_w, int)
    # Recorded, not load-bearing: the two views agree.
    assert tuple(quant_config.spec.weight_block_size) == EXPECTED_BLOCK_SHAPE

    # --- moving control: widen the supported set, move the checkpoint ----
    monkeypatch.setattr(
        qz,
        "SUPPORTED_WEIGHT_BLOCK_SIZES",
        frozenset({(128, 128), (256, 256)}),
    )
    widened_raw = copy.deepcopy(raw)
    widened_raw["quantization_config"]["weight_block_size"] = [256, 256]
    control_method = _quant_config_from(widened_raw).get_quant_method()
    control_pair = (control_method.block_h, control_method.block_w)

    print(f"c02_control_pair={control_pair}")
    print(f"c02_control_MOVES={pair} -> {control_pair}")

    assert control_pair == (256, 256), (
        "control did not move; the reported pair would be a constant"
    )
    assert control_pair != pair


# ---------------------------------------------------------------------------
# Conjunct 3 — an unsupported weight_block_size raises, 1/1
# ---------------------------------------------------------------------------
def test_block_quant_recognition_rejects_an_unsupported_block_size(raw) -> None:
    """C03 — 1/1: a well-formed but unsupported `weight_block_size` raises.

    D1.4 certifying component:
    `quantization.py::BlockFp8QuantMethod.__post_init__`, its
    `self.block_shape not in SUPPORTED_WEIGHT_BLOCK_SIZES` arm, raising
    `UnsupportedWeightBlockSize`. The check lives in `__post_init__` rather than
    in the resolver so no construction path can bypass it, and this item proves
    both entry points reach it.

    `[64, 64]` and not `[128, 128, 128]`: parent readings show the landed parser
    ALREADY raised on a malformed shape while ACCEPTING `[64, 64]`, so only a
    well-formed unsupported shape measures this increment (false-pass door 2).

    D1.5 moving control: the supported `[128, 128]` through the same call path
    must NOT raise. Without it, a resolver that raised unconditionally would
    also pass.
    """
    raised = 0
    of = 0

    of += 1
    unsupported_raw = copy.deepcopy(raw)
    unsupported_raw["quantization_config"]["weight_block_size"] = (
        UNSUPPORTED_BLOCK_SIZE
    )
    with pytest.raises(qz.UnsupportedWeightBlockSize) as excinfo:
        _quant_config_from(unsupported_raw)
    raised += 1

    print(f"c03_raised={raised} c03_of={of}")
    print(f"c03_block_size={UNSUPPORTED_BLOCK_SIZE}")
    print(f"c03_error_type={type(excinfo.value).__name__}")
    print(f"c03_error={excinfo.value}")

    assert of == 1
    assert raised == 1
    # A named error, and still a ValueError so existing callers keep working.
    assert isinstance(excinfo.value, ValueError)
    assert "64" in str(excinfo.value)

    # The landed parser stays permissive: the spec for the same config builds
    # fine, and the refusal is attributable to method resolution alone.
    permissive_spec = qz.QuantizationSpec.from_hf_quantization_config(
        unsupported_raw["quantization_config"]
    )
    print(f"c03_parser_still_accepts={permissive_spec.weight_block_size}")
    assert tuple(permissive_spec.weight_block_size) == (64, 64)

    # The direct constructor reaches the same guard.
    with pytest.raises(qz.UnsupportedWeightBlockSize):
        qz.BlockFp8QuantMethod(block_h=64, block_w=64, activation_scheme="dynamic")

    # --- moving control: the supported shape does NOT raise -------------
    control_raised = 0
    try:
        control_method = _quant_config_from(raw).get_quant_method()
    except qz.UnsupportedWeightBlockSize:
        control_raised = 1
        control_method = None

    print(f"c03_control_raised={control_raised}")
    control_pair = None if control_method is None else control_method.block_shape
    print(f"c03_control_pair={control_pair}")
    print(f"c03_control_MOVES={raised} -> {control_raised}")

    assert control_raised == 0, (
        "control did not move; the refusal would be unconditional"
    )
    assert control_method is not None


# ---------------------------------------------------------------------------
# Conjunct 4 — the P47 pin baseline: 0 occurrences at the pin
# ---------------------------------------------------------------------------
def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _matching_lines_at(rev: str, token: str) -> int:
    """Total `git grep -c` matching-line count for `token` under `vllm_neuron/`."""
    out = _git("grep", "-F", "-c", token, rev, "--", "vllm_neuron/")
    # exit 1 = no match (a legitimate zero); anything else is an instrument fault.
    assert out.returncode in (0, 1), (
        f"git grep failed at {rev}: rc={out.returncode} err={out.stderr!r}"
    )
    total = 0
    for line in out.stdout.splitlines():
        total += int(line.rsplit(":", 1)[1])
    return total


def test_block_quant_recognition_records_the_pin_baseline(raw) -> None:
    """C04 — `weight_block_size` has 0 occurrences in `vllm_neuron` at the pin.

    D1.4 certifying component: the PIN TREE itself
    (`f8abae64…` = `release-0.24.0.1.1.0`, DECISIONS section 1), read through
    `git grep -F -c weight_block_size <pin> -- vllm_neuron/`. The number
    certifies the absence of any recognition path in the fork's own package at
    the campaign's base, which is what makes this increment the whole path.

    Read out of git at the pin, NOT off the working tree: the working tree's
    count is nonzero the moment this increment lands, whereas the pin's is a
    settled historical fact.

    D1.5 controls, both required because a counted zero read from a scanner can
    be zero for the wrong reason:
      * SENSITIVITY — the same scanner, same revision, a token known present at
        the pin (`quant_method`) must read NON-ZERO. A scanner that reads 0 for
        everything is not measuring.
      * MOVEMENT — the same scanner at HEAD must read NON-ZERO. The property
        "no occurrences" is false at HEAD, and the count moves accordingly.
    """
    # The pin object must be present, and it must actually be this branch's base.
    assert _git("cat-file", "-e", f"{PIN}^{{commit}}").returncode == 0, (
        f"pin {PIN} is not in this repo's object store"
    )
    assert _git("merge-base", "--is-ancestor", PIN, "HEAD").returncode == 0, (
        f"pin {PIN} is not an ancestor of HEAD"
    )
    merge_base = _git("merge-base", "HEAD", PIN).stdout.strip()
    print(f"c04_pin={PIN}")
    print(f"c04_merge_base_head_pin={merge_base}")
    assert merge_base == PIN, "the pin is not this branch's base"

    py_at_pin = [
        p
        for p in _git("ls-tree", "-r", "--name-only", PIN, "--", "vllm_neuron/")
        .stdout.splitlines()
        if p.endswith(".py")
    ]
    baseline = _matching_lines_at(PIN, "weight_block_size")
    print(f"c04_pin_py_file_count={len(py_at_pin)}")
    print(f"c04_PIN_weight_block_size_lines={baseline}")

    assert len(py_at_pin) > 0, "the pin tree carries no vllm_neuron/*.py to scan"
    assert baseline == 0

    sensitivity = _matching_lines_at(PIN, "quant_method")
    print(f"c04_control_sensitivity_PIN_quant_method_lines={sensitivity}")
    assert sensitivity > 0, "scanner is insensitive; the zero is not a measurement"

    movement = _matching_lines_at("HEAD", "weight_block_size")
    print(f"c04_control_movement_HEAD_weight_block_size_lines={movement}")
    print(f"c04_control_MOVES={baseline} -> {movement}")
    assert movement > 0, "control did not move; the counted zero is decoration"


# ---------------------------------------------------------------------------
# Guard — plan section 11 constraint B.6: no vendor-enum widening
# ---------------------------------------------------------------------------
_VENDOR_ENUM = "Quantization" + "Type"  # split so this literal is not itself a hit


def _vendor_enum_code_refs(source: str) -> int:
    """Count real CODE references to the vendor quantisation enum.

    AST-based, so a comment or docstring naming the prohibition is not a hit —
    the `inc-glm53f-021` lesson, where a raw grep reported 2 hits that were both
    prose. Counts `Name` loads of the enum and attribute chains rooted at it.
    """
    hits = 0
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and node.id == _VENDOR_ENUM:
            hits += 1
        elif isinstance(node, ast.alias) and (node.name or "").split(".")[-1] == (
            _VENDOR_ENUM
        ):
            hits += 1
    return hits


def test_block_quant_recognition_adds_no_vendor_enum_member() -> None:
    """B.6 guard — 0 code references to the vendor quantisation enum, package-wide.

    D1.4 certifying component: every `.py` file in
    `vllm_neuron/model/glm5_next/`, parsed by `ast`. The route this increment
    takes is D5(b)'s direct inner-kernel call precisely BECAUSE the vendor enum
    carries no blockwise member at this pin and this campaign may not add one
    (plan section 11, B.6, a FORBIDDEN ACTION). This guard makes that constraint
    falsifiable in-tree rather than trusted.

    D1.5 moving control: the same counter over a synthetic module that DOES
    reference the enum must read non-zero. Parent reading
    `R2b_QuantizationType_in_glm5_next_at_parent=0` is the pre-change baseline.
    """
    pkg = REPO_ROOT / "vllm_neuron" / "model" / "glm5_next"
    files = sorted(pkg.glob("*.py"))
    per_file = {p.name: _vendor_enum_code_refs(p.read_text()) for p in files}
    total = sum(per_file.values())

    print(f"b6_files_scanned={len(files)}")
    print(f"b6_per_file={per_file}")
    print(f"b6_total_code_refs={total}")

    assert len(files) >= 5, "package scan found too few files to be the real tree"
    assert total == 0

    control = _vendor_enum_code_refs(
        "from nkilib.core.utils.common_types import "
        + _VENDOR_ENUM
        + "\nx = "
        + _VENDOR_ENUM
        + ".BLOCK_128\n"
    )
    print(f"b6_control_code_refs={control}")
    print(f"b6_control_MOVES={total} -> {control}")
    assert control > 0, "control did not move; the counted zero is decoration"
