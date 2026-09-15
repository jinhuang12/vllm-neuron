# SPDX-License-Identifier: Apache-2.0
"""``platform.py`` accepts a block-fp8 quantisation config (``inc-glm53f-019``).

Subject: ``NeuronPlatform._validate_quantization_config``
(``vllm_neuron/vllm/platform.py``, the declared surface -- ``@classmethod`` at
L495 and ``def`` at L496 at worktree HEAD ``ea633e8a``, the pin-anchored L487/488
of the increment block shifted by the uniform +8 that ``-075``/``-074`` landed
above L487).

**Three items, one per declared conjunct, no ``parametrize``** (test-layout rule
6), so the item count stays derivable before the run.

Conjunct 1 -- a config carrying ``weight_block_size [128,128]`` +
``activation_scheme dynamic`` validates in **1/1** cases.
Conjunct 2 -- a config carrying an MX quantisation method is **rejected** in
**1/1** cases.
Conjunct 3 -- **0** currently-accepted config in the pin's own matrix becomes
rejected.

**Parent readings, measured on the instrument BEFORE a source line was written**
(``increments/probe-019-parent-readings.py``, run at the unmodified parent
``ea633e8a``, ``PROBE_EXIT_CODE=0``):

* Conjunct 1 is **ACCEPT at the parent** -- declared green here rather than
  discovered later. ``quant_method="fp8"`` is not ``"compressed-tensors"``, so
  the parent returns without ever examining ``weight_block_size`` or
  ``activation_scheme``. Item 1 therefore carries the declared predicate as a
  **regression guard on this increment's own MX gate** (a careless gate breaks
  fp8 admission), and adds a **recorded differential** -- the admission log
  record, **0** at the parent and **1** here -- which is the reading that shows
  this increment's own code examined the config. The differential is recorded,
  and adds no conjunct.
* Conjunct 2 is **FALSE at the parent**: ``mxfp8``, ``mxfp4``, ``mxfp6``, ``mx``
  and all **4** mx-named methods in vLLM's registry each read **ACCEPT**. This
  is the conjunct that discriminates.
* Conjunct 3 reads **0 of 9** rejected at the parent, over the pin accept-path
  matrix below.

**Why conjuncts 2 and 3 do not collide, measured rather than argued.** Rejecting
a method that is currently accepted would violate conjunct 3. All **4** mx-named
methods in the instrument's registry are **already REFUSED** by
``Platform.verify_quantization`` against ``NeuronPlatform.supported_quantization``
(``-075``'s surface), so none of them is in the platform's currently-accepted
population, and **0** allowlisted method contains ``"mx"``. Item 2 measures both
halves on the instrument instead of asserting the reconciliation.

No item constructs an engine, loads a checkpoint, reaches a network or touches a
device. Nothing under any compile cache is read, written or relocated (P2).
"""

from __future__ import annotations

import contextlib
import json
import logging
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_neuron.vllm import platform as platform_module
from vllm_neuron.vllm.platform import NeuronPlatform

# The campaign's pinned checkpoint config, landed by inc-glm53f-008 and already
# the join ``-075`` item 4 asserts against. Resolved off ``__file__`` so the read
# cannot depend on the invocation's cwd.
FIXTURE_CONFIG = (
    Path(__file__).resolve().parent / "model" / "glm5_next" / "fixtures" / "config.json"
)

ADMISSION_MARKER = "Admitting block-scaled fp8 checkpoint"


def _record(label: str, value: object) -> None:
    """Emit an instrument-world reading into the run's own transcript."""
    warnings.warn(f"RECORDED {label}={value!r}", stacklevel=2)


def _cfg(neuron_config: dict | None = None, quant_cfg: object = "ABSENT"):
    """A stand-in carrying only the three attributes the method under test reads.

    ``"ABSENT"`` means ``hf_config`` has no ``quantization_config`` attribute at
    all, which is a distinct pin accept path from a falsy one.
    """
    hf_config = SimpleNamespace()
    if quant_cfg != "ABSENT":
        hf_config.quantization_config = quant_cfg
    return SimpleNamespace(
        additional_config=(
            {} if neuron_config is None else {"neuron_config": neuron_config}
        ),
        model_config=SimpleNamespace(hf_config=hf_config),
    )


def _verdict(vllm_config) -> str:
    """``"ACCEPT"`` or ``"REJECT"``, taken from the method, never from a flag."""
    try:
        NeuronPlatform._validate_quantization_config(vllm_config)
    except ValueError:
        return "REJECT"
    return "ACCEPT"


class _RecordingHandler(logging.Handler):
    """Collects formatted records off the module's own logger object."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def _capture_admissions():
    """Attach to ``platform.logger`` directly.

    Directly, not through ``caplog``: ``caplog``'s handler lives on the root
    logger, so a propagation setting anywhere in vLLM's logging configuration
    would make the differential read 0 for a reason that has nothing to do with
    the code under test.
    """
    handler = _RecordingHandler()
    target = platform_module.logger
    previous_level = target.level
    target.addHandler(handler)
    target.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        target.removeHandler(handler)
        target.setLevel(previous_level)


def _admission_count(handler: _RecordingHandler) -> int:
    return sum(1 for message in handler.messages if ADMISSION_MARKER in message)


def _fixture_quant_cfg() -> dict:
    assert FIXTURE_CONFIG.is_file(), f"pinned fixture unreachable: {FIXTURE_CONFIG}"
    return json.loads(FIXTURE_CONFIG.read_text())["quantization_config"]


def _pin_accept_matrix(fixture_quant_cfg: dict) -> list[tuple[str, object]]:
    """One row per ``return`` the PIN's method body can reach.

    Derived from the pinned repo bytes by a named method -- reading the accept
    paths off the parent's own branch structure -- so the population is the
    pin's and not a hand-picked sample. Every row read **ACCEPT** at the parent
    (probe reading R3: 0 of 9 rejected).
    """
    return [
        # A. the CPU-dequant waiver -- operator intent, the pin's L502-503.
        (
            "A_cpu_dequant_waiver_mxfp8",
            _cfg({"quantization": "mxfp8"}, {"quant_method": "mxfp8"}),
        ),
        # B. no quantization_config attribute at all (the getattr default).
        ("B_no_quantization_config_attr", _cfg()),
        # C. a falsy quantization_config.
        ("C_empty_quantization_config", _cfg(None, {})),
        # D. quant_method is not compressed-tensors -- the pinned fixture's own
        #    block-fp8 config, which is also conjunct 1's config.
        ("D_fp8_block_fixture", _cfg(None, dict(fixture_quant_cfg))),
        ("D2_modelopt", _cfg(None, {"quant_method": "modelopt"})),
        ("D3_neuron_quant", _cfg(None, {"quant_method": "neuron_quant"})),
        # E. compressed-tensors with no config_groups: the loop never runs.
        ("E_ct_no_config_groups", _cfg(None, {"quant_method": "compressed-tensors"})),
        # F. compressed-tensors, KV-cache-only -- input activations on an
        #    Attention target, the one input-activation shape the pin admits.
        (
            "F_ct_kv_cache_only_attention",
            _cfg(
                None,
                {
                    "quant_method": "compressed-tensors",
                    "config_groups": {
                        "group_0": {
                            "input_activations": {"num_bits": 8, "type": "float"},
                            "targets": ["Attention"],
                        }
                    },
                },
            ),
        ),
        # G. compressed-tensors, a group carrying none of the three guarded keys.
        (
            "G_ct_bare_group",
            _cfg(
                None,
                {
                    "quant_method": "compressed-tensors",
                    "config_groups": {"group_0": {"targets": ["Linear"]}},
                },
            ),
        ),
    ]


def _registry_mx_methods() -> list[str]:
    """The instrument's own mx-named methods, read rather than declared.

    D1.3's preferred form: this increment's rule is asserted as a **relation**
    against a set the instrument produces, so a vendor that adds an MX method
    widens the population instead of being absorbed by a literal list.
    """
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

    return sorted(m for m in QUANTIZATION_METHODS if "mx" in m.lower())


def test_block_fp8_config_validates() -> None:
    """Conjunct 1 -- 1/1 validates; the admission differential is RECORDED."""
    quant_cfg = _fixture_quant_cfg()
    _record("fixture quantization_config", quant_cfg)

    # The two declared carriers, read off the pinned fixture rather than typed
    # into this test, so a fixture change cannot pass silently.
    assert quant_cfg["weight_block_size"] == [128, 128]
    assert quant_cfg["activation_scheme"] == "dynamic"

    with _capture_admissions() as handler:
        verdict = _verdict(_cfg(None, dict(quant_cfg)))
        admissions = _admission_count(handler)

    # The declared predicate. Parent reading: ACCEPT -- green at the parent and
    # declared so; here it is a regression guard on this increment's MX gate.
    assert verdict == "ACCEPT", (
        "the block-fp8 config the campaign's pinned checkpoint carries must "
        "validate; a substring MX gate that also matched fp8 would fail here"
    )

    # RECORDED differential, adding no conjunct: 0 admission records at the
    # parent (which never examines the config), exactly 1 here.
    _record("admission_records_for_block_fp8", admissions)
    assert admissions == 1, (
        "the block-fp8 config was accepted without this increment's branch "
        "examining it -- an acceptance indistinguishable from the parent's "
        "fall-through"
    )

    # Control for that differential: per-tensor fp8 (no weight_block_size) is
    # still ACCEPTED and produces 0 admission records, so the record tracks the
    # BLOCK shape and not merely "fp8 was seen".
    with _capture_admissions() as handler:
        per_tensor_verdict = _verdict(
            _cfg(None, {"quant_method": "fp8", "activation_scheme": "dynamic"})
        )
        per_tensor_admissions = _admission_count(handler)
    _record("admission_records_for_per_tensor_fp8", per_tensor_admissions)
    assert per_tensor_verdict == "ACCEPT"
    assert per_tensor_admissions == 0


def test_mx_quantization_method_is_rejected() -> None:
    """Conjunct 2 -- 1/1 rejected, at the boundary and not only by the scan."""
    # The declared 1/1 case: an MX checkpoint WITHOUT the CPU-dequant path
    # selected. That is exactly the "fails loud" claim the pin's own
    # _cpu_dequant_quantizations comment makes, now measured.
    with pytest.raises(ValueError) as excinfo:
        NeuronPlatform._validate_quantization_config(
            _cfg(None, {"quant_method": "mxfp8", "weight_block_size": [32, 32]})
        )
    assert "mxfp8" in str(excinfo.value), (
        f"the refusal must name the method; message was {str(excinfo.value)!r}"
    )

    # Widened to the instrument's own population, as a relation (D1.3 form 1).
    # Parent reading: every one of these read ACCEPT.
    mx_methods = _registry_mx_methods()
    _record("registry mx-named methods", mx_methods)
    assert mx_methods, "the instrument reported no mx-named method at all"
    rejected = [
        m for m in mx_methods if _verdict(_cfg(None, {"quant_method": m})) == "REJECT"
    ]
    assert rejected == mx_methods, (
        f"mx-named methods not rejected: {sorted(set(mx_methods) - set(rejected))}"
    )

    # The RECONCILIATION with conjunct 3, measured: every mx-named method is
    # already refused at the platform allowlist, so none of them is in the
    # currently-accepted population and rejecting it here removes no acceptance.
    already_refused = []
    for method in mx_methods:
        with pytest.raises(ValueError):
            NeuronPlatform.verify_quantization(method)
        already_refused.append(method)
    _record("mx methods already refused by supported_quantization", already_refused)
    assert already_refused == mx_methods

    # Non-vacuity: the same config shape with a NON-MX method is not rejected,
    # so the arm reads the method name and not the shape it was handed.
    assert (
        _verdict(_cfg(None, {"quant_method": "fp8", "weight_block_size": [32, 32]}))
        == "ACCEPT"
    )


def test_no_currently_accepted_config_becomes_rejected() -> None:
    """Conjunct 3 -- 0 of the pin's accept paths flip, with the zero shown to move."""
    fixture_quant_cfg = _fixture_quant_cfg()
    matrix = _pin_accept_matrix(fixture_quant_cfg)
    verdicts = {name: _verdict(cfg) for name, cfg in matrix}
    _record("pin_accept_matrix_verdicts", verdicts)

    newly_rejected = sorted(n for n, v in verdicts.items() if v == "REJECT")
    _record("newly_rejected", newly_rejected)
    assert len(matrix) == 9, "the pin accept-path population changed size"
    assert newly_rejected == [], (
        f"configs the pin accepts are now rejected: {newly_rejected}"
    )

    # Every method the platform allowlists must still reach ACCEPT, which is the
    # guard that a widened MX rule cannot quietly close the fp8 door.
    allowlist = list(NeuronPlatform.supported_quantization)
    _record("supported_quantization", allowlist)
    allowlisted_rejected = [
        m for m in allowlist if _verdict(_cfg(None, {"quant_method": m})) == "REJECT"
    ]
    assert allowlisted_rejected == [], (
        f"allowlisted methods rejected by the validator: {allowlisted_rejected}"
    )

    # D1.5 CONTROL -- the zero MOVES. The property this zero tests is that the
    # MX gate runs AFTER the CPU-dequant waiver. Applying the identical
    # substring rule with the waiver IGNORED -- the one ordering mistake
    # available here -- rejects row A and the count reads 1, not 0.
    def _gate_ignoring_the_waiver(quant_cfg: object) -> str:
        if isinstance(quant_cfg, dict):
            method = quant_cfg.get("quant_method")
            if isinstance(method, str) and "mx" in method.lower():
                return "REJECT"
        return "ACCEPT"

    hoisted_rejections = sorted(
        name
        for name, cfg in matrix
        if _gate_ignoring_the_waiver(
            getattr(cfg.model_config.hf_config, "quantization_config", None)
        )
        == "REJECT"
    )
    _record("control_rejections_with_the_gate_hoisted", hoisted_rejections)
    assert hoisted_rejections == ["A_cpu_dequant_waiver_mxfp8"], (
        "the control did not move the zero, so this conjunct would read 0 "
        "whether or not the ordering property holds"
    )
