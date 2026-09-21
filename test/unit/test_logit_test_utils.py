"""Checkpoint resolution in ``test/vllm_neuron/utils/logit_test_utils.py``.

``get_model_checkpoint`` is step 1 of every real-weight accuracy test
(``docs/design/accuracy/module_test_guidelines.md:273``), so a defect here
surfaces as an unrelated model test failing to find weights, or as a CPU-mode
unit run quietly reaching the HuggingFace hub. Covered here: the documented
zero-keyword call form resolves a checkpoint out of the local cache without
opening a socket, a pre-downloaded directory resolves directly, an empty
directory is not a checkpoint, the S3 leg stays inert until configured, and a
miss raises with every leg it tried.

The helper is loaded from its file path under the distinct module name
``fork_logit_test_utils``: ``test/`` has no ``__init__.py``, so pytest prepends
``<root>/test`` to ``sys.path`` and ``test/vllm_neuron/`` would otherwise be
importable under the plugin's own name. Loading it by path keeps the
import-hygiene checks unambiguous -- any ``vllm_neuron`` entry they see is a real
plugin import and never this tree.

``NXDI_CHECKPOINT_CACHE`` is set with ``monkeypatch.setenv`` because
``checkpoint_cache_root()`` reads it on every call, unlike the platform target
``test/conftest.py`` has to pin before collection.
"""

from __future__ import annotations

import importlib.util
import socket
import sys
from pathlib import Path
from typing import Any

import pytest

# --------------------------------------------------------------------------- #
# Load the helper under test from its path (see the module docstring).
# --------------------------------------------------------------------------- #

_HELPER_PATH = (
    Path(__file__).resolve().parents[1] / "vllm_neuron" / "utils" / "logit_test_utils.py"
)
_spec = importlib.util.spec_from_file_location("fork_logit_test_utils", _HELPER_PATH)
assert _spec is not None and _spec.loader is not None, f"cannot load {_HELPER_PATH}"
logit_test_utils = importlib.util.module_from_spec(_spec)
# Register before executing: with ``from __future__ import annotations`` in the
# helper, ``@dataclass`` resolves its ``KW_ONLY`` check through
# ``sys.modules[cls.__module__].__dict__`` (CPython 3.14 ``dataclasses._is_type``)
# and raises ``AttributeError`` if the module was never registered. The name is
# fork-distinct, so this adds no ``vllm_neuron`` entry.
sys.modules[_spec.name] = logit_test_utils


#: SDKs the helper's two remote legs use. Each one is optional, so importing any
#: of them at module scope would make this file fail to collect in a CPU venv.
OPTIONAL_SDKS = ("boto3", "botocore", "huggingface_hub", "safetensors", "transformers")


def _plugin_modules() -> set[str]:
    return {m for m in sys.modules if m == "vllm_neuron" or m.startswith("vllm_neuron.")}


def _loaded_sdks() -> set[str]:
    return {m for m in OPTIONAL_SDKS if m in sys.modules}


# Both readings are deltas around the helper's own import rather than absolute
# snapshots: another test in the same session may legitimately have imported the
# plugin or one of these SDKs already, and what the two checks below report is
# the helper's own doing.
_plugin_before_helper = _plugin_modules()
_sdks_before_helper = _loaded_sdks()
_spec.loader.exec_module(logit_test_utils)
PLUGIN_MODULES_THE_HELPER_IMPORTED = sorted(_plugin_modules() - _plugin_before_helper)
SDKS_THE_HELPER_IMPORTED = sorted(_loaded_sdks() - _sdks_before_helper)

CheckpointNotFound = logit_test_utils.CheckpointNotFound
get_model_checkpoint = logit_test_utils.get_model_checkpoint
resolve_model_checkpoint = logit_test_utils.resolve_model_checkpoint
is_checkpoint_dir = logit_test_utils.is_checkpoint_dir
CHECKPOINT_CACHE_ENV = logit_test_utils.CHECKPOINT_CACHE_ENV
S3_CHECKPOINTS_URI_ENV = logit_test_utils.S3_CHECKPOINTS_URI_ENV

MODEL_ID = "glm-org/glm-5.3-flash"
WEIGHT_FILE = "model.safetensors"


class NetworkTripwire:
    """Counts attempted network entries and raises on every one of them."""

    #: ``socket.socket`` alone is not enough: a resolver call or a convenience
    #: connect would slip past a constructor-only patch.
    PATCHED = ("socket", "create_connection", "getaddrinfo")

    def __init__(self) -> None:
        self.calls: list[str] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> NetworkTripwire:
        for name in self.PATCHED:
            monkeypatch.setattr(socket, name, self._stub(name))
        return self

    def _stub(self, name: str):
        def _raise(*args: Any, **kwargs: Any):
            self.calls.append(name)
            raise AssertionError(
                f"network call attempted through socket.{name}"
                f"({len(args)} args) -- this run must not reach the network"
            )

        return _raise

    @property
    def socket_calls(self) -> int:
        """Calls through ``socket.socket`` alone."""
        return self.calls.count("socket")

    @property
    def total_calls(self) -> int:
        return len(self.calls)


@pytest.fixture
def tripwire(monkeypatch: pytest.MonkeyPatch) -> NetworkTripwire:
    return NetworkTripwire().install(monkeypatch)


def _make_checkpoint(directory: Path, *, weight_bytes: int = 32) -> Path:
    """Write the smallest thing the helper is allowed to call a checkpoint."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / WEIGHT_FILE).write_bytes(b"\0" * weight_bytes)
    (directory / "config.json").write_text('{"architectures": ["Glm5NextForCausalLM"]}')
    return directory


def test_get_model_checkpoint_resolves_a_cached_checkpoint_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tripwire: NetworkTripwire,
) -> None:
    """The documented ``get_model_checkpoint(model_id)`` call form resolves offline."""
    cache_root = tmp_path / "checkpoint-cache"
    expected = _make_checkpoint(cache_root / "glm-org_glm-5.3-flash")
    monkeypatch.setenv(CHECKPOINT_CACHE_ENV, str(cache_root))
    monkeypatch.delenv(S3_CHECKPOINTS_URI_ENV, raising=False)

    returned = Path(get_model_checkpoint(MODEL_ID))

    # The leg is asserted too: "no network call" means little without evidence
    # that resolution stopped at a local leg rather than being skipped.
    resolution = resolve_model_checkpoint(MODEL_ID)

    assert returned == expected
    assert returned.is_dir()
    assert resolution.leg == "local_cache"
    assert tripwire.total_calls == 0, f"network calls attempted: {tripwire.calls}"


def test_direct_local_path_resolves_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tripwire: NetworkTripwire,
) -> None:
    """A pre-downloaded directory passed straight in resolves without the cache."""
    checkpoint = _make_checkpoint(tmp_path / "pre-downloaded")
    monkeypatch.setenv(CHECKPOINT_CACHE_ENV, str(tmp_path / "unused-cache"))

    resolution = resolve_model_checkpoint(str(checkpoint))

    assert resolution.leg == "local_path"
    assert Path(resolution.path) == checkpoint
    assert tripwire.total_calls == 0, f"network calls attempted: {tripwire.calls}"


def test_helper_import_pulls_in_no_optional_sdk() -> None:
    """The S3 and hub legs import lazily, so the local leg needs no SDK."""
    assert SDKS_THE_HELPER_IMPORTED == [], (
        f"helper import pulled in optional SDKs: {SDKS_THE_HELPER_IMPORTED}"
    )


def test_empty_directory_is_not_accepted_as_a_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tripwire: NetworkTripwire,
) -> None:
    """An interrupted download leaves a directory that exists but holds no weights."""
    # Accepting it would move the failure into AutoConfig.from_pretrained, where it
    # reads as a corrupt-model bug instead of a missing checkpoint.
    cache_root = tmp_path / "cache"
    (cache_root / "glm-org_glm-5.3-flash").mkdir(parents=True)
    monkeypatch.setenv(CHECKPOINT_CACHE_ENV, str(cache_root))
    monkeypatch.delenv(S3_CHECKPOINTS_URI_ENV, raising=False)

    assert not is_checkpoint_dir(cache_root / "glm-org_glm-5.3-flash")

    with pytest.raises(CheckpointNotFound) as excinfo:
        get_model_checkpoint(MODEL_ID, allow_remote=False)

    attempts = excinfo.value.attempts
    assert len(attempts) == 3, attempts
    assert attempts[0].startswith("local_path: no checkpoint directory")
    assert attempts[1].startswith("local_cache: no .safetensors file")
    assert attempts[2] == "s3, huggingface: skipped (allow_remote=False)"
    assert tripwire.total_calls == 0, f"network calls attempted: {tripwire.calls}"


def test_s3_leg_is_disabled_until_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tripwire: NetworkTripwire,
) -> None:
    """An empty S3 URI disables the leg, and the miss still enumerates every leg."""
    monkeypatch.setenv(CHECKPOINT_CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setenv(S3_CHECKPOINTS_URI_ENV, "")

    with pytest.raises(CheckpointNotFound) as excinfo:
        get_model_checkpoint(MODEL_ID)

    attempts = excinfo.value.attempts
    assert [a for a in attempts if a.startswith("s3:")] == [
        f"s3: disabled ({S3_CHECKPOINTS_URI_ENV} unset or empty)"
    ]
    # The hub leg is reached and reports itself; whether it is unavailable (no SDK
    # here) or fails (SDK present, tripwire raises) it must never be silent.
    hf_lines = [a for a in attempts if a.startswith("huggingface:")]
    assert len(hf_lines) == 1 and hf_lines[0].startswith("huggingface: "), hf_lines
    assert tripwire.socket_calls == 0, f"network calls attempted: {tripwire.calls}"


def test_no_plugin_module_is_imported() -> None:
    """The helper pulls in no ``vllm_neuron`` module, so it runs off-host."""
    assert PLUGIN_MODULES_THE_HELPER_IMPORTED == [], (
        f"importing the helper pulled in {PLUGIN_MODULES_THE_HELPER_IMPORTED}"
    )
    # The helper is loaded under a distinct module name, which is what makes the
    # reading above about the plugin rather than about this tree.
    assert logit_test_utils.__name__ == "fork_logit_test_utils"
