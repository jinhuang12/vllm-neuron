"""Checkpoint resolution for the logit and real-weight accuracy tests.

``docs/design/accuracy/module_test_guidelines.md`` prescribes this module as
step 1 of every real-weight accuracy test:

.. code-block:: python

    from test.vllm_neuron.utils.logit_test_utils import get_model_checkpoint
    model_checkpoint = get_model_checkpoint(model_id)

Resolution walks local path, then local cache, then S3, then HuggingFace, and
reports which leg produced the path. A miss raises ``CheckpointNotFound`` listing
every leg it tried, because "checkpoint not found" without that list is
unactionable in CI. Remote legs can be turned off with ``allow_remote=False``, and
the S3 leg stays disabled until a URI is configured, following
``golden_cache``'s own "empty means disabled" convention.

Nothing here imports ``vllm_neuron``: this tree has to stay runnable off-host, and
``vllm_neuron.utils.checkpoints`` imports ``huggingface_hub`` at module scope, so
reusing the plugin's checkpoint machinery would make an SDK mandatory just to
resolve a local directory. The plugin's constants are mirrored as strings, and the
S3 and HuggingFace legs import their SDKs inside the leg, so the local-cache leg
needs no SDK and opens no socket.
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
# Plugin constants, mirrored by value with their source cited.
# --------------------------------------------------------------------------- #

#: The OS variable the fork's own checkpoint-cache setting reads:
#: ``vllm_neuron/envs.py:166-168`` defines ``VLLM_NEURON_CHECKPOINT_CACHE`` as
#: ``os.getenv("NXDI_CHECKPOINT_CACHE", "/tmp/vllm_neuron-checkpoints")``. Read
#: here with the same name and default rather than inventing a second cache
#: location.
CHECKPOINT_CACHE_ENV = "NXDI_CHECKPOINT_CACHE"
DEFAULT_CHECKPOINT_CACHE = "/tmp/vllm_neuron-checkpoints"

#: Optional S3 base URI for the secondary leg. No such variable exists in
#: ``envs.py`` yet; the name follows ``VLLM_NEURON_S3_GOLDENS_URI``
#: (``vllm_neuron/envs.py:174``) and its "empty = disabled" semantics, so the S3
#: leg is inert until someone configures it.
S3_CHECKPOINTS_URI_ENV = "VLLM_NEURON_S3_CHECKPOINTS_URI"

#: What makes a directory a checkpoint. ``SafetensorsCheckpoint`` filters on
#: ``".safetensors"`` (``vllm_neuron/utils/checkpoints.py``), so that is the
#: default, overridable for the ``.bin``/``.pt`` checkpoints the same factory
#: also accepts.
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
    ``"huggingface"``, so a caller can tell a local hit from a download.
    """

    model_id: str
    path: str
    leg: str
    attempts: tuple[str, ...] = field(default_factory=tuple)


def checkpoint_cache_root(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the local cache root: explicit argument > env var > default."""
    # The precedence mirrors golden_cache's own order for its S3 URI
    # ("explicit cache_s3_uri > env var > disabled", golden_cache.py:53-54).
    if cache_dir is not None:
        return Path(cache_dir)
    return Path(
        os.path.expandvars(
            os.environ.get(CHECKPOINT_CACHE_ENV) or DEFAULT_CHECKPOINT_CACHE
        )
    )


def cache_entry_name(model_id: str) -> str:
    """Directory name for a model id inside the cache root."""
    # "/" -> "_" is golden_cache's own key sanitisation (golden_cache.py:169), so
    # a hub id and its cache entry stay mechanically related.
    return model_id.replace("/", "_")


def is_checkpoint_dir(
    path: str | os.PathLike[str],
    weight_suffixes: tuple[str, ...] = DEFAULT_WEIGHT_SUFFIXES,
) -> bool:
    """True when ``path`` is a directory holding at least one weight file."""
    # Existence alone is deliberately not enough: an empty directory left behind
    # by an interrupted download would otherwise resolve, and the caller would
    # fail much later inside AutoConfig.from_pretrained. The listing test is
    # _LocalCheckpointSource's (checkpoints.py): a top-level file whose name ends
    # with the extension.
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
    """Secondary leg: copy ``<s3_uri>/<cache_entry_name>/`` into the cache."""
    # boto3 is imported here, not at module scope -- golden_cache.py:226 does the
    # same inside _get_s3_client(). The URI is split exactly as
    # golden_cache._s3_key splits it (golden_cache.py:165-170).
    import boto3  # noqa: PLC0415 -- lazy on purpose

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
    """Tertiary leg: a hub snapshot."""
    # Imported here, the plugin's own form at
    # vllm_neuron/model/llama3/eagle3_model.py:1095-1097.
    from huggingface_hub import snapshot_download  # noqa: PLC0415 -- lazy

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
    """Return a local checkpoint directory for ``model_id``."""
    # The documented call form get_model_checkpoint(model_id)
    # (module_test_guidelines.md:330) stays callable verbatim: every keyword is
    # optional and keyword-only, so the doc's signature cannot drift.
    return resolve_model_checkpoint(
        model_id,
        cache_dir=cache_dir,
        s3_uri=s3_uri,
        weight_suffixes=weight_suffixes,
        allow_remote=allow_remote,
    ).path
