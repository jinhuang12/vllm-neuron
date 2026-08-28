"""Checkpoint resolution for the fork's logit / real-weight accuracy tests.

``docs/design/accuracy/module_test_guidelines.md`` already prescribes this
module and this symbol, but the repo never shipped either:

.. code-block:: python

    # module_test_guidelines.md:329-330
    from test.vllm_neuron.utils.logit_test_utils import get_model_checkpoint
    model_checkpoint = get_model_checkpoint(model_id)

The same doc names step 1 of every real-weight accuracy test as *"Resolve
checkpoint (get_model_checkpoint)"* (``:273``), describes the contract as
*"portable checkpoint resolution (local cache -> S3 -> HuggingFace)"*
(``:326``), and ``vllm_neuron/accuracy/accuracy_debugger/utils/api_utils.py:88``
records the same assumption from the other side (*"the test harness resolves it
via get_model_checkpoint before invoking the example"*).

Design rules this module holds to, each with the reason:

**1. Nothing here imports ``vllm_neuron``.** The overlay tree must stay
runnable off-host and must not drag the plugin's import graph into a CPU-mode
unit run -- ``vllm_neuron/utils/checkpoints.py:12`` imports ``huggingface_hub``
at module scope, so importing the plugin's checkpoint machinery would make an
SDK mandatory just to resolve a local directory. The plugin's constants are
therefore **mirrored as strings** with their source cited, exactly as
``test/vllm_neuron/model/utils.py`` mirrors ``"weight_loader"``.

**2. The S3 and HuggingFace legs import their SDKs lazily, inside the leg.**
This is the plugin's own convention, not an invention:
``vllm_neuron/utils/golden_cache.py:226`` does ``import boto3`` inside
``_get_s3_client()``, and ``vllm_neuron/model/llama3/eagle3_model.py:1095``
does ``from huggingface_hub import snapshot_download`` inside the function that
needs it. It also means the local-cache leg -- the one CPU-mode tests use --
needs no SDK and opens no socket.

**3. A miss raises with the legs enumerated.** ``CheckpointNotFound`` carries
one line per attempted leg and why it did not resolve, because "checkpoint not
found" without the attempt list is unactionable in CI.

**4. Remote legs are opt-out-able and S3 is opt-in.** ``allow_remote=False``
keeps a caller strictly offline; the S3 leg stays disabled unless a URI is
configured, which is ``golden_cache``'s own "empty = disabled" convention
(``golden_cache.py:110-117``, ``vllm_neuron/envs.py:174``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "CHECKPOINT_CACHE_ENV",
    "DEFAULT_CHECKPOINT_CACHE",
    "DEFAULT_WEIGHT_SUFFIXES",
    "S3_CHECKPOINTS_URI_ENV",
    "CheckpointNotFound",
    "CheckpointResolution",
    "cache_entry_name",
    "checkpoint_cache_root",
    "get_model_checkpoint",
    "is_checkpoint_dir",
    "resolve_model_checkpoint",
]

# --------------------------------------------------------------------------- #
# Plugin constants, mirrored by value with their source cited (rule 1).
# --------------------------------------------------------------------------- #

#: The OS environment variable the fork's own checkpoint-cache setting reads.
#: ``vllm_neuron/envs.py:166-168`` defines ``VLLM_NEURON_CHECKPOINT_CACHE`` as
#: ``os.getenv("NXDI_CHECKPOINT_CACHE", "/tmp/vllm_neuron-checkpoints")`` under
#: the comment *"Local cache directory for model checkpoints"*. The setting has
#: **no reader anywhere at this pin** -- this module is the consumer the docs
#: assumed, so it reads the same variable with the same default rather than
#: inventing a second cache location.
CHECKPOINT_CACHE_ENV = "NXDI_CHECKPOINT_CACHE"
DEFAULT_CHECKPOINT_CACHE = "/tmp/vllm_neuron-checkpoints"

#: Optional S3 base URI for the secondary leg. **No such variable exists at
#: this pin**; the name follows ``VLLM_NEURON_S3_GOLDENS_URI``
#: (``vllm_neuron/envs.py:174``) and the same "empty = disabled" semantics, so
#: the S3 leg is inert until someone configures it. Declared here rather than
#: silently assumed, so a later increment can promote it into ``envs.py`` as a
#: design decision.
S3_CHECKPOINTS_URI_ENV = "VLLM_NEURON_S3_CHECKPOINTS_URI"

#: What makes a directory a checkpoint. ``SafetensorsCheckpoint`` filters on
#: ``".safetensors"`` (``vllm_neuron/utils/checkpoints.py`` --
#: ``_LocalCheckpointSource`` keeps ``sorted(f for f in os.listdir(d) if
#: f.endswith(file_extension))``), so that is the default, overridable for the
#: ``.bin``/``.pt`` checkpoints the same factory also accepts.
DEFAULT_WEIGHT_SUFFIXES = (".safetensors",)


class CheckpointNotFound(FileNotFoundError):
    """No leg resolved the model id. Carries the attempted legs verbatim."""

    def __init__(self, model_id: str, attempts: tuple[str, ...]) -> None:
        self.model_id = model_id
        self.attempts = tuple(attempts)
        detail = "".join(f"\n  - {attempt}" for attempt in self.attempts)
        super().__init__(
            f"could not resolve a checkpoint for {model_id!r}; "
            f"{len(self.attempts)} leg(s) attempted:{detail}"
        )


@dataclass(frozen=True)
class CheckpointResolution:
    """Which leg produced the path, and what every other leg reported.

    ``leg`` is one of ``"local_path"``, ``"local_cache"``, ``"s3"`` or
    ``"huggingface"``. Tests assert on it so that "0 network calls" is
    corroborated by *which* leg ran, not only by a socket counter.
    """

    model_id: str
    path: str
    leg: str
    attempts: tuple[str, ...] = field(default_factory=tuple)


def checkpoint_cache_root(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the local cache root: explicit argument > env var > default.

    The precedence mirrors ``golden_cache``'s own resolution order for its S3
    URI (*"explicit cache_s3_uri > env var > disabled"*,
    ``golden_cache.py:53-54``).
    """
    if cache_dir is not None:
        return Path(cache_dir)
    return Path(
        os.path.expandvars(
            os.environ.get(CHECKPOINT_CACHE_ENV) or DEFAULT_CHECKPOINT_CACHE
        )
    )


def cache_entry_name(model_id: str) -> str:
    """Directory name for a model id inside the cache root.

    ``"/" -> "_"`` is ``golden_cache``'s own key sanitisation
    (``golden_cache.py:169``: ``key_config.get("model", "unknown").replace("/",
    "_")``), so a hub id and its cache entry stay mechanically related.
    """
    return model_id.replace("/", "_")


def is_checkpoint_dir(
    path: str | os.PathLike[str],
    weight_suffixes: tuple[str, ...] = DEFAULT_WEIGHT_SUFFIXES,
) -> bool:
    """True when ``path`` is a directory holding at least one weight file.

    Existence alone is deliberately **not** enough: an empty directory left
    behind by an interrupted download would otherwise resolve, and the caller
    would fail much later inside ``AutoConfig.from_pretrained``. The listing
    test is ``_LocalCheckpointSource``'s (``checkpoints.py``): a top-level file
    whose name ends with the extension.
    """
    directory = Path(path)
    if not directory.is_dir():
        return False
    return any(
        entry.endswith(weight_suffixes)
        for entry in os.listdir(directory)
        if (directory / entry).is_file()
    )


def _resolve_from_s3(
    model_id: str,
    s3_uri: str,
    destination: Path,
    weight_suffixes: tuple[str, ...],
) -> str | None:
    """Secondary leg: copy ``<s3_uri>/<cache_entry_name>/`` into the cache.

    ``boto3`` is imported **here**, not at module scope -- ``golden_cache.py:226``
    does the same inside ``_get_s3_client()``. The URI is split exactly as
    ``golden_cache._s3_key`` splits it (``golden_cache.py:165-170``).
    """
    import boto3  # noqa: PLC0415 -- lazy on purpose (rule 2)

    path = s3_uri.replace("s3://", "")
    bucket = path.split("/", 1)[0]
    prefix = path.split("/", 1)[1] if "/" in path else ""
    key_prefix = "/".join(part for part in (prefix, cache_entry_name(model_id)) if part)

    client = boto3.client("s3")
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{key_prefix}/"):
        for obj in page.get("Contents", ()):
            key = obj["Key"]
            name = key[len(key_prefix) + 1 :]
            if not name or name.endswith("/"):
                continue
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(target))
            downloaded += 1
    if downloaded and is_checkpoint_dir(destination, weight_suffixes):
        return str(destination)
    return None


def _resolve_from_hf(model_id: str, cache_root: Path) -> str:
    """Tertiary leg: a hub snapshot.

    ``snapshot_download`` is imported **here** -- the plugin's own form at
    ``vllm_neuron/model/llama3/eagle3_model.py:1095-1097``.
    """
    from huggingface_hub import snapshot_download  # noqa: PLC0415 -- lazy (rule 2)

    return snapshot_download(model_id, cache_dir=str(cache_root))


def resolve_model_checkpoint(
    model_id: str,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    s3_uri: str | None = None,
    weight_suffixes: tuple[str, ...] = DEFAULT_WEIGHT_SUFFIXES,
    allow_remote: bool = True,
) -> CheckpointResolution:
    """Walk local path -> local cache -> S3 -> HuggingFace and report the leg.

    Args:
        model_id: A hub id (``"company/model"``) or a local directory path.
        cache_dir: Local cache root; defaults to ``checkpoint_cache_root()``.
        s3_uri: S3 base URI; defaults to ``$VLLM_NEURON_S3_CHECKPOINTS_URI``.
            Empty or unset keeps the S3 leg disabled.
        weight_suffixes: What counts as a weight file.
        allow_remote: ``False`` stops after the two local legs -- the switch a
            CPU-mode test uses when reaching the hub would be a defect rather
            than a slow path.

    Returns:
        The resolution, including every leg that was tried.

    Raises:
        CheckpointNotFound: no leg resolved; the attempt list is attached.
    """
    attempts: list[str] = []
    cache_root = checkpoint_cache_root(cache_dir)

    # Leg 0 -- the id is already a checkpoint directory. This is the
    # discriminator ``checkpoints.py::_get_checkpoint_source`` uses
    # (``os.path.isdir(model_name_or_path)``), and the doc's own escape hatch
    # (*"Or use a local path directly"*, module_test_guidelines.md:333-337).
    if is_checkpoint_dir(model_id, weight_suffixes):
        attempts.append(f"local_path: hit at {model_id}")
        return CheckpointResolution(
            model_id=model_id,
            path=str(model_id),
            leg="local_path",
            attempts=tuple(attempts),
        )
    attempts.append(f"local_path: no checkpoint directory at {model_id}")

    # Leg 1 -- the local cache. No network, no SDK.
    cached = cache_root / cache_entry_name(model_id)
    if is_checkpoint_dir(cached, weight_suffixes):
        attempts.append(f"local_cache: hit at {cached}")
        return CheckpointResolution(
            model_id=model_id,
            path=str(cached),
            leg="local_cache",
            attempts=tuple(attempts),
        )
    attempts.append(
        f"local_cache: no {'/'.join(weight_suffixes)} file under {cached} "
        f"(cache root from {CHECKPOINT_CACHE_ENV} or {DEFAULT_CHECKPOINT_CACHE})"
    )

    if not allow_remote:
        attempts.append("s3, huggingface: skipped (allow_remote=False)")
        raise CheckpointNotFound(model_id, tuple(attempts))

    # Leg 2 -- S3, opt-in.
    resolved_s3_uri = s3_uri or os.environ.get(S3_CHECKPOINTS_URI_ENV, "")
    if resolved_s3_uri:
        try:
            from_s3 = _resolve_from_s3(
                model_id, resolved_s3_uri, cached, weight_suffixes
            )
        except ImportError as exc:
            attempts.append(f"s3: unavailable ({exc.__class__.__name__}: {exc})")
        except Exception as exc:  # noqa: BLE001 -- a dead leg must not mask leg 3
            attempts.append(f"s3: failed ({exc.__class__.__name__}: {exc})")
        else:
            if from_s3 is not None:
                attempts.append(f"s3: hit at {resolved_s3_uri}")
                return CheckpointResolution(
                    model_id=model_id,
                    path=from_s3,
                    leg="s3",
                    attempts=tuple(attempts),
                )
            attempts.append(f"s3: no objects under {resolved_s3_uri}")
    else:
        attempts.append(f"s3: disabled ({S3_CHECKPOINTS_URI_ENV} unset or empty)")

    # Leg 3 -- HuggingFace.
    try:
        from_hf = _resolve_from_hf(model_id, cache_root)
    except ImportError as exc:
        attempts.append(f"huggingface: unavailable ({exc.__class__.__name__}: {exc})")
    except Exception as exc:  # noqa: BLE001 -- reported, never swallowed silently
        attempts.append(f"huggingface: failed ({exc.__class__.__name__}: {exc})")
    else:
        attempts.append(f"huggingface: hit at {from_hf}")
        return CheckpointResolution(
            model_id=model_id,
            path=str(from_hf),
            leg="huggingface",
            attempts=tuple(attempts),
        )

    raise CheckpointNotFound(model_id, tuple(attempts))


def get_model_checkpoint(
    model_id: str,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    s3_uri: str | None = None,
    weight_suffixes: tuple[str, ...] = DEFAULT_WEIGHT_SUFFIXES,
    allow_remote: bool = True,
) -> str:
    """Return a local checkpoint directory for ``model_id``.

    The documented call form is ``get_model_checkpoint(model_id)``
    (``module_test_guidelines.md:330``) and stays callable verbatim; every
    keyword is optional and keyword-only, so the doc's signature cannot drift.
    """
    return resolve_model_checkpoint(
        model_id,
        cache_dir=cache_dir,
        s3_uri=s3_uri,
        weight_suffixes=weight_suffixes,
        allow_remote=allow_remote,
    ).path
