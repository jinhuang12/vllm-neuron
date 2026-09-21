# SPDX-License-Identifier: Apache-2.0
"""``NeuronPlatform._validate_quantization_config``.

A block-scaled ``fp8`` checkpoint (``weight_block_size`` plus
``activation_scheme``) must validate, an MX quantization method must be refused
loudly unless the operator selected the CPU-dequant path for it, and every
config shape the validator already accepts must keep validating.

No test constructs an engine, loads a checkpoint, reaches a network or touches a
device.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_neuron.vllm import platform as platform_module
from vllm_neuron.vllm.platform import NeuronPlatform

# Resolved off ``__file__`` so the read cannot depend on the invocation's cwd.
FIXTURE_CONFIG = (
    Path(__file__).resolve().parent / "model" / "glm5_next" / "fixtures" / "config.json"
)

ADMISSION_MARKER = "Admitting block-scaled fp8 checkpoint"


def _cfg(neuron_config: dict | None = None, quant_cfg: object = "ABSENT"):
    """A stand-in carrying only the three attributes the method reads.

    ``"ABSENT"`` means ``hf_config`` has no ``quantization_config`` attribute at
    all, which is a different accept path from a falsy one.
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

    Not through ``caplog``: its handler lives on the root logger, so a
    propagation setting anywhere in vLLM's logging configuration would make the
    record count read 0 for a reason unrelated to the code under test.
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
    assert FIXTURE_CONFIG.is_file(), f"fixture unreachable: {FIXTURE_CONFIG}"
    return json.loads(FIXTURE_CONFIG.read_text())["quantization_config"]


def _accepted_configs(fixture_quant_cfg: dict) -> list[tuple[str, object]]:
    """One row per ``return`` the method body can reach without raising."""
    return [
        # A. the CPU-dequant waiver: operator intent wins over the MX refusal,
        #    which is why the refusal has to run after the waiver.
        (
            "A_cpu_dequant_waiver_mxfp8",
            _cfg({"quantization": "mxfp8"}, {"quant_method": "mxfp8"}),
        ),
        # B. no quantization_config attribute at all (the getattr default).
        ("B_no_quantization_config_attr", _cfg()),
        # C. a falsy quantization_config.
        ("C_empty_quantization_config", _cfg(None, {})),
        # D. quant_method is not compressed-tensors.
        ("D_fp8_block_fixture", _cfg(None, dict(fixture_quant_cfg))),
        ("D2_modelopt", _cfg(None, {"quant_method": "modelopt"})),
        ("D3_neuron_quant", _cfg(None, {"quant_method": "neuron_quant"})),
        # E. compressed-tensors with no config_groups: the loop never runs.
        ("E_ct_no_config_groups", _cfg(None, {"quant_method": "compressed-tensors"})),
        # F. compressed-tensors, KV-cache-only: input activations on an
        #    Attention target, the one input-activation shape that is admitted.
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
    """The MX-named methods vLLM's registry carries, read rather than typed."""
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

    return sorted(m for m in QUANTIZATION_METHODS if "mx" in m.lower())


def test_block_fp8_config_validates() -> None:
    """The fixture's block-scaled fp8 config validates and is logged as admitted."""
    quant_cfg = _fixture_quant_cfg()

    # Read off the fixture rather than typed in, so a fixture change cannot pass
    # silently.
    assert quant_cfg["weight_block_size"] == [128, 128]
    assert quant_cfg["activation_scheme"] == "dynamic"

    with _capture_admissions() as handler:
        verdict = _verdict(_cfg(None, dict(quant_cfg)))
        admissions = _admission_count(handler)

    assert verdict == "ACCEPT", (
        "the block-fp8 config the fixture checkpoint carries must validate; a "
        "substring MX refusal that also matched fp8 would fail here"
    )

    # The log record shows the block-fp8 branch examined the config instead of
    # falling through the way a non-compressed-tensors method does.
    assert admissions == 1

    # Per-tensor fp8 (no weight_block_size) is still accepted and logs nothing,
    # so the record tracks the BLOCK shape and not merely "fp8 was seen".
    with _capture_admissions() as handler:
        per_tensor_verdict = _verdict(
            _cfg(None, {"quant_method": "fp8", "activation_scheme": "dynamic"})
        )
        per_tensor_admissions = _admission_count(handler)
    assert per_tensor_verdict == "ACCEPT"
    assert per_tensor_admissions == 0


def test_mx_quantization_method_is_rejected() -> None:
    """MX checkpoints are refused by name unless CPU dequant is selected."""
    with pytest.raises(ValueError) as excinfo:
        NeuronPlatform._validate_quantization_config(
            _cfg(None, {"quant_method": "mxfp8", "weight_block_size": [32, 32]})
        )
    assert "mxfp8" in str(excinfo.value), (
        f"the refusal must name the method; message was {str(excinfo.value)!r}"
    )

    # Stated as a relation over the registry's own population, so a vendor that
    # adds an MX method widens the check instead of escaping a literal list.
    mx_methods = _registry_mx_methods()
    assert mx_methods, "the registry reported no MX-named method at all"
    rejected = [
        m for m in mx_methods if _verdict(_cfg(None, {"quant_method": m})) == "REJECT"
    ]
    assert rejected == mx_methods, (
        f"MX-named methods not rejected: {sorted(set(mx_methods) - set(rejected))}"
    )

    # Every MX method is already refused at the platform allowlist, so refusing
    # it here removes no acceptance the platform used to have.
    for method in mx_methods:
        with pytest.raises(ValueError):
            NeuronPlatform.verify_quantization(method)

    # The same config shape with a non-MX method is not rejected, so the refusal
    # reads the method name and not the shape it was handed.
    assert (
        _verdict(_cfg(None, {"quant_method": "fp8", "weight_block_size": [32, 32]}))
        == "ACCEPT"
    )


def test_no_currently_accepted_config_becomes_rejected() -> None:
    """Every accept path, and every allowlisted method, still validates."""
    matrix = _accepted_configs(_fixture_quant_cfg())
    verdicts = {name: _verdict(cfg) for name, cfg in matrix}

    newly_rejected = sorted(n for n, v in verdicts.items() if v == "REJECT")
    assert newly_rejected == [], (
        f"configs that used to be accepted are now rejected: {newly_rejected}"
    )

    allowlist = list(NeuronPlatform.supported_quantization)
    allowlisted_rejected = [
        m for m in allowlist if _verdict(_cfg(None, {"quant_method": m})) == "REJECT"
    ]
    assert allowlisted_rejected == [], (
        f"allowlisted methods rejected by the validator: {allowlisted_rejected}"
    )
