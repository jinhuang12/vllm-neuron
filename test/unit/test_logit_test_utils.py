"""The checkpoint-resolution helper is itself under test -- it is load-bearing.

``docs/design/accuracy/module_test_guidelines.md:273`` makes
``get_model_checkpoint`` step 1 of *every* real-weight accuracy test, so a
defect here surfaces as an unrelated model test failing to find weights, or --
much worse -- as a CPU-mode unit run quietly dialling the HuggingFace hub.

**One declared predicate**, the increment's acceptance:

    ``get_model_checkpoint`` resolves a temp-dir checkpoint and returns a path
    that **exists** (1/1), with **exactly 0** network calls, asserted by
    monkeypatching ``socket.socket`` to raise.

Everything else in this file is a **guard**, not a criterion: the tripwire's own
falsifiability (a counter that cannot report non-zero is not a measurement), the
direct-local-path leg, the "empty directory is not a checkpoint" negative, the
offline stop, the no-plugin-import obligation, and one item that *records*
overlay import-resolution state for a later increment without changing it.

**Three mechanical notes.**

*Import by path, on purpose.* ``test/`` has no ``__init__.py``, so pytest
prepends ``<root>/test`` to ``sys.path`` and ``test/vllm_neuron/`` becomes
importable as a top-level ``vllm_neuron`` -- the plugin's own name. Loading the
helper from its file path under the distinct name ``fork_logit_test_utils``
sidesteps that entirely and keeps the import-hygiene check unambiguous: any
``vllm_neuron`` entry in ``sys.modules`` is then a real plugin import and never
this tree. This is the mechanism ``test/unit/test_model_test_utils.py`` already
proved.

*The cache root is set through the environment, and that is not a D2 breach.*
``NEURON_PLATFORM_TARGET_OVERRIDE`` is forbidden to fixtures because
``FP8_CLAMP_MAX`` resolves at **import** time. ``NXDI_CHECKPOINT_CACHE`` is read
inside ``checkpoint_cache_root()`` on every **call**, so ``monkeypatch.setenv``
is exactly on time -- and it is what lets the declared predicate exercise the
documented zero-argument call form ``get_model_checkpoint(model_id)``
(``module_test_guidelines.md:330``) instead of a keyword the docs do not use.

*Measured values are written out.* pytest captures stdout for passing tests, so
each measured value also goes to a machine-readable JSON file
(``$VLLM_NEURON_INC003_RESULTS_JSON``, else a fixed path in the temp dir). The
assertions are the gate; the file makes the numbers auditable after a green run.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import sys
import tempfile
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
# Register before executing -- the documented order for a spec-based load, and
# load-bearing here rather than cosmetic: with ``from __future__ import
# annotations`` in the helper, ``@dataclass`` resolves its ``KW_ONLY`` check
# through ``sys.modules[cls.__module__].__dict__`` (CPython 3.14
# ``dataclasses._is_type``), which raises ``AttributeError: 'NoneType' object
# has no attribute '__dict__'`` if the module was never registered. The name is
# fork-distinct, so this adds no ``vllm_neuron`` entry and leaves the
# import-hygiene guard below untouched.
sys.modules[_spec.name] = logit_test_utils
_spec.loader.exec_module(logit_test_utils)

# Snapshots taken the instant the helper finished importing. Recorded here
# rather than inside a test so the observation cannot be polluted by a later
# import from another test item.
_OPTIONAL_SDKS = ("boto3", "botocore", "huggingface_hub", "safetensors", "transformers")
_SDKS_AFTER_HELPER_IMPORT = sorted(m for m in _OPTIONAL_SDKS if m in sys.modules)
_PLUGIN_MODULES_AFTER_HELPER_IMPORT = sorted(
    m for m in sys.modules if m == "vllm_neuron" or m.startswith("vllm_neuron.")
)

CheckpointNotFound = logit_test_utils.CheckpointNotFound
get_model_checkpoint = logit_test_utils.get_model_checkpoint
resolve_model_checkpoint = logit_test_utils.resolve_model_checkpoint
is_checkpoint_dir = logit_test_utils.is_checkpoint_dir
CHECKPOINT_CACHE_ENV = logit_test_utils.CHECKPOINT_CACHE_ENV
S3_CHECKPOINTS_URI_ENV = logit_test_utils.S3_CHECKPOINTS_URI_ENV

# --------------------------------------------------------------------------- #
# Measured-value sink.
# --------------------------------------------------------------------------- #

_RESULTS_PATH = Path(
    os.environ.get("VLLM_NEURON_INC003_RESULTS_JSON")
    or Path(tempfile.gettempdir()) / "vllm_neuron_inc003_predicates.json"
)
_RESULTS: dict[str, Any] = {}
_RESULTS_PATH.write_text("{}\n")  # truncate stale values from an earlier run


def _record(**values: Any) -> None:
    _RESULTS.update(values)
    _RESULTS_PATH.write_text(
        json.dumps(_RESULTS, indent=2, sort_keys=True, default=str) + "\n"
    )


# --------------------------------------------------------------------------- #
# The network tripwire: the declared instrument for "exactly 0 network calls".
# --------------------------------------------------------------------------- #

MODEL_ID = "glm-org/glm-5.3-flash"
WEIGHT_FILE = "model.safetensors"

#: ``socket.socket`` is the increment's declared instrument. The other two are
#: additional tripwires: a resolver call or a convenience connect would
#: otherwise slip past a socket-constructor-only patch. They strengthen the
#: declared predicate and are counted separately from it, never instead of it.
DECLARED_TRIPWIRE = "socket"
EXTRA_TRIPWIRES = ("create_connection", "getaddrinfo")


class NetworkTripwire:
    """Counts attempted network entries and raises on every one of them."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> NetworkTripwire:
        for name in (DECLARED_TRIPWIRE, *EXTRA_TRIPWIRES):
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
    def declared_calls(self) -> int:
        """Calls through ``socket.socket``, the declared instrument."""
        return self.calls.count(DECLARED_TRIPWIRE)

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


# --------------------------------------------------------------------------- #
# THE DECLARED PREDICATE
# --------------------------------------------------------------------------- #


def test_get_model_checkpoint_resolves_temp_dir_checkpoint_with_zero_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tripwire: NetworkTripwire,
) -> None:
    """Declared: path exists 1/1, network calls == 0.

    The call form is the documented one, ``get_model_checkpoint(model_id)`` with
    no keywords, reached by pointing the fork's own checkpoint-cache variable at
    a temp dir.
    """
    cache_root = tmp_path / "checkpoint-cache"
    expected = _make_checkpoint(cache_root / "glm-org_glm-5.3-flash")
    monkeypatch.setenv(CHECKPOINT_CACHE_ENV, str(cache_root))
    monkeypatch.delenv(S3_CHECKPOINTS_URI_ENV, raising=False)

    returned = get_model_checkpoint(MODEL_ID)

    returned_path = Path(returned)
    paths_checked = 1
    paths_existing = 1 if returned_path.exists() else 0

    # The leg is asserted too: "0 network calls" means little without evidence
    # that resolution stopped at a local leg rather than being skipped.
    resolution = resolve_model_checkpoint(MODEL_ID)

    _record(
        declared_model_id=MODEL_ID,
        declared_returned_path=returned,
        declared_paths_checked=paths_checked,
        declared_paths_existing=paths_existing,
        declared_returned_is_dir=returned_path.is_dir(),
        declared_returned_weight_files=sorted(
            p.name for p in returned_path.iterdir() if p.is_file()
        ),
        declared_resolution_leg=resolution.leg,
        declared_resolution_attempts=list(resolution.attempts),
        declared_network_calls_socket_socket=tripwire.declared_calls,
        declared_network_calls_all_tripwires=tripwire.total_calls,
        declared_tripwire_names=[DECLARED_TRIPWIRE, *EXTRA_TRIPWIRES],
    )

    assert paths_existing == paths_checked == 1, (
        f"declared predicate: expected 1/1 existing, got "
        f"{paths_existing}/{paths_checked} for {returned!r}"
    )
    assert returned_path == expected
    assert returned_path.is_dir()
    assert tripwire.declared_calls == 0, (
        f"declared predicate: expected exactly 0 socket.socket calls, got "
        f"{tripwire.declared_calls} ({tripwire.calls})"
    )
    assert tripwire.total_calls == 0, (
        f"expected 0 calls across all tripwires, got {tripwire.calls}"
    )
    assert resolution.leg == "local_cache"


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def test_network_tripwire_counts_and_raises_when_something_dials_out(
    tripwire: NetworkTripwire,
) -> None:
    """Falsifiability of the zero: the counter can report non-zero.

    Without this, ``0 network calls`` could be the value a broken tripwire
    always reports.
    """
    with pytest.raises(AssertionError, match="network call attempted"):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(AssertionError, match="network call attempted"):
        socket.getaddrinfo("example.invalid", 443)

    _record(
        guard_tripwire_declared_calls=tripwire.declared_calls,
        guard_tripwire_total_calls=tripwire.total_calls,
        guard_tripwire_call_names=list(tripwire.calls),
    )
    assert tripwire.declared_calls == 1
    assert tripwire.total_calls == 2


def test_direct_local_path_leg_resolves_with_zero_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tripwire: NetworkTripwire,
) -> None:
    """The doc's escape hatch (``module_test_guidelines.md:333-337``) works.

    Guard, not the declared predicate: a pre-downloaded directory passed
    straight in must resolve without consulting the cache at all.
    """
    checkpoint = _make_checkpoint(tmp_path / "pre-downloaded")
    monkeypatch.setenv(CHECKPOINT_CACHE_ENV, str(tmp_path / "unused-cache"))

    resolution = resolve_model_checkpoint(str(checkpoint))

    _record(
        guard_local_path_leg=resolution.leg,
        guard_local_path_exists=Path(resolution.path).exists(),
        guard_local_path_network_calls=tripwire.total_calls,
    )
    assert resolution.leg == "local_path"
    assert Path(resolution.path) == checkpoint
    assert tripwire.total_calls == 0


def test_helper_import_pulls_in_no_optional_sdk() -> None:
    """The S3 and HuggingFace legs are lazy, so the local leg needs no SDK.

    Recorded from a snapshot taken at helper-import time. If either SDK were
    imported at module scope, this whole file would fail to collect in a CPU
    instrument venv that has neither -- which is precisely the failure mode the
    lazy-import rule exists to prevent.
    """
    installed = sorted(
        name for name in _OPTIONAL_SDKS if importlib.util.find_spec(name) is not None
    )
    _record(
        guard_optional_sdks_watched=list(_OPTIONAL_SDKS),
        guard_optional_sdks_installed=installed,
        guard_optional_sdks_imported_by_helper=_SDKS_AFTER_HELPER_IMPORT,
    )
    assert _SDKS_AFTER_HELPER_IMPORT == [], (
        f"helper import pulled in optional SDKs: {_SDKS_AFTER_HELPER_IMPORT}"
    )


def test_empty_directory_is_not_accepted_as_a_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tripwire: NetworkTripwire,
) -> None:
    """"Returns a path that exists" is not satisfiable by an empty directory.

    An interrupted download leaves exactly this state; accepting it would move
    the failure into ``AutoConfig.from_pretrained`` where it reads as a
    corrupt-model bug.
    """
    cache_root = tmp_path / "cache"
    (cache_root / "glm-org_glm-5.3-flash").mkdir(parents=True)
    monkeypatch.setenv(CHECKPOINT_CACHE_ENV, str(cache_root))
    monkeypatch.delenv(S3_CHECKPOINTS_URI_ENV, raising=False)

    assert not is_checkpoint_dir(cache_root / "glm-org_glm-5.3-flash")

    with pytest.raises(CheckpointNotFound) as excinfo:
        get_model_checkpoint(MODEL_ID, allow_remote=False)

    attempts = excinfo.value.attempts
    _record(
        guard_empty_dir_attempt_count=len(attempts),
        guard_empty_dir_attempts=list(attempts),
        guard_empty_dir_network_calls=tripwire.total_calls,
    )
    assert len(attempts) == 3, attempts
    assert attempts[0].startswith("local_path: no checkpoint directory")
    assert attempts[1].startswith("local_cache: no .safetensors file")
    assert attempts[2] == "s3, huggingface: skipped (allow_remote=False)"
    assert tripwire.total_calls == 0


def test_s3_leg_is_disabled_until_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tripwire: NetworkTripwire,
) -> None:
    """Empty URI == disabled, the fork's own convention for its golden cache.

    With the S3 leg inert and no ``huggingface_hub`` reachable, the miss must
    still be a clean enumerated ``CheckpointNotFound`` -- and must not open a
    socket on the way there.
    """
    monkeypatch.setenv(CHECKPOINT_CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setenv(S3_CHECKPOINTS_URI_ENV, "")

    with pytest.raises(CheckpointNotFound) as excinfo:
        get_model_checkpoint(MODEL_ID)

    attempts = excinfo.value.attempts
    s3_lines = [a for a in attempts if a.startswith("s3:")]
    hf_lines = [a for a in attempts if a.startswith("huggingface:")]
    _record(
        guard_s3_disabled_attempts=list(attempts),
        guard_s3_lines=s3_lines,
        guard_hf_lines=hf_lines,
        guard_s3_disabled_network_calls=tripwire.total_calls,
    )
    assert s3_lines == [f"s3: disabled ({S3_CHECKPOINTS_URI_ENV} unset or empty)"]
    # The hub leg is reached and reports itself; whether it is unavailable (no
    # SDK here) or fails (SDK present, tripwire fires) it must never be silent.
    assert len(hf_lines) == 1 and hf_lines[0].startswith("huggingface: "), hf_lines
    assert tripwire.declared_calls == 0


def test_acceptance_imports_no_vllm_neuron_module() -> None:
    """The overlay obligation: this acceptance imports no plugin module.

    Two observations, one from helper-import time and one live, so an import by
    any other item in this run is caught too.
    """
    live = sorted(
        m for m in sys.modules if m == "vllm_neuron" or m.startswith("vllm_neuron.")
    )
    _record(
        guard_plugin_modules_after_helper_import=_PLUGIN_MODULES_AFTER_HELPER_IMPORT,
        guard_plugin_modules_live=live,
        guard_plugin_modules_live_count=len(live),
        guard_helper_module_name=logit_test_utils.__name__,
    )
    assert _PLUGIN_MODULES_AFTER_HELPER_IMPORT == []
    assert live == [], f"acceptance imported plugin modules: {live}"
    assert logit_test_utils.__name__ == "fork_logit_test_utils"


def test_records_overlay_import_resolution_state() -> None:
    """DATA for a later increment: what resolves where under this invocation.

    Recording, not fixing. ``find_spec`` is used deliberately -- it resolves a
    top-level name **without executing** it, so this item can report which
    ``vllm_neuron`` wins the name without importing the plugin and breaking the
    guard above. Nothing here is a criterion; the single assertion only keeps
    the record from being silently empty.
    """
    root = Path(__file__).resolve().parents[2]
    overlay = root / "test"

    plugin_spec = importlib.util.find_spec("vllm_neuron")
    plugin_origin = getattr(plugin_spec, "origin", None)
    search_locations = list(getattr(plugin_spec, "submodule_search_locations", ()) or ())
    winner = (
        "overlay"
        if plugin_origin and str(overlay) in plugin_origin
        else ("plugin" if plugin_origin else "unresolved")
    )
    # Does the overlay's shadow now cover the real plugin's ``utils`` package?
    shadowed_utils = [
        str(Path(loc) / "utils" / "__init__.py")
        for loc in search_locations
        if (Path(loc) / "utils" / "__init__.py").exists()
    ]

    # The doc prescribes ``from test.vllm_neuron.utils.logit_test_utils import
    # ...``; where does the bare name ``test`` actually resolve here?
    test_spec = importlib.util.find_spec("test")
    test_origin = getattr(test_spec, "origin", None)

    _record(
        overlay_root=str(overlay),
        overlay_has_init=(overlay / "__init__.py").exists(),
        overlay_sys_path_index=(
            sys.path.index(str(overlay)) if str(overlay) in sys.path else None
        ),
        repo_root_sys_path_index=(
            sys.path.index(str(root)) if str(root) in sys.path else None
        ),
        vllm_neuron_name_resolves_to=plugin_origin,
        vllm_neuron_name_winner=winner,
        vllm_neuron_search_locations=search_locations,
        vllm_neuron_utils_shadowed_at=shadowed_utils,
        bare_test_package_resolves_to=test_origin,
        doc_consumer_form_reaches_overlay=bool(
            test_origin and str(overlay) in str(test_origin)
        ),
    )
    assert plugin_origin is not None or winner == "unresolved"
