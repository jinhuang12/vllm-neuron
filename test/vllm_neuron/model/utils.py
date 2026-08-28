"""Host-side checkpoint-slice fakes the model-bringup docs assume.

``docs/design/framework/model_bringup.md`` (the "Key utilities in
``test/vllm_neuron/model/utils.py``" list) names three helpers and their
signatures, and the repo ships none of them. This module is those three:

- ``FakeSafeSlice(tensor)`` -- stand-in for a safetensors slice
- ``hf_state_to_fake_slices(state_dict, layer_idx)`` -- wraps an HF state dict
- ``load_weights_from_slices(module, slice_map, mappings, rank, device)`` --
  drives weight loaders on synthetic data

Together they let a credential-free test build synthetic HF weights in memory
and drive the fork's real weight loaders over them, with no checkpoint on disk
and nothing computed on an accelerator.

**Import hygiene, deliberate.** This module imports ``torch`` and nothing else.
It never imports ``vllm_neuron``, because the loader is an *input*: the fork
attaches its loader object to a parameter under the attribute name
``"weight_loader"`` (``vllm_neuron/utils/weight_loader.py``,
``_WEIGHT_LOADER_ATTR``) and the load call is
``loader.load(slices, rank)`` (``vllm_neuron/utils/checkpoints.py:261-265``).
Both are consumed here by DUCK TYPE off the module the caller passes in, so the
helper stays device-free and importable off-host while still driving the real
loader objects when a caller supplies them.

**What is mirrored, and from where.** ``load_weights_from_slices`` reproduces the
non-pipelined checkpoint path at ``vllm_neuron/utils/checkpoints.py:229-271``
step for step: ``mappings.get(name, name)``, a scalar key normalised to a
one-element list, missing keys reported per *parameter*, checkpoint keys nothing
references reported as unexpected, and ``tensor.to(device)`` on the loader's
result. The slice interface mirrors what the loaders actually call --
``get_shape()`` plus ``__getitem__`` with a tuple of slices
(``vllm_neuron/utils/weight_loader.py:216-218``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable, NamedTuple

import torch

__all__ = [
    "WEIGHT_LOADER_ATTR",
    "FakeSafeSlice",
    "LoadFromSlicesResult",
    "hf_state_to_fake_slices",
    "load_weights_from_slices",
]

#: Attribute name the fork attaches its loader under
#: (``vllm_neuron/utils/weight_loader.py``, ``_WEIGHT_LOADER_ATTR``). Duplicated
#: here as a string on purpose: importing the plugin to read it would pull the
#: device stack into a host-side test.
WEIGHT_LOADER_ATTR = "weight_loader"

_LAYER_KEY_PREFIX = "model.layers."


class FakeSafeSlice:
    """In-memory stand-in for a safetensors ``PySafeSlice``.

    Presents the checkpoint-slice interface the fork's weight loaders use --
    ``get_shape()`` plus ``__getitem__`` with slice / int / tuple keys -- over a
    torch tensor, so a loader can be driven on synthetic weights.

    Reads return FRESH memory, as a real slice read from a file does:
    ``__getitem__`` clones, so a loader that mutates or transposes its result
    can never write back into the caller's state dict.

    ``get_dtype()`` is deliberately not faked. No loader at this pin calls it,
    and the real method returns a safetensors dtype *string* rather than a
    ``torch.dtype``, so faking it would invent a contract instead of mirroring
    one. Use :attr:`dtype` for the backing tensor's dtype.
    """

    __slots__ = ("_tensor",)

    def __init__(self, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"FakeSafeSlice wraps a torch.Tensor, got {type(tensor).__name__}"
            )
        self._tensor = tensor

    def get_shape(self) -> list[int]:
        """Checkpoint shape as a list, matching ``PySafeSlice.get_shape()``."""
        return list(self._tensor.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._tensor.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    def __getitem__(self, key: Any) -> torch.Tensor:
        return self._tensor[key].clone()

    def __repr__(self) -> str:
        return f"FakeSafeSlice(shape={self.shape}, dtype={self._tensor.dtype})"


def hf_state_to_fake_slices(
    state_dict: Mapping[str, torch.Tensor], layer_idx: int | None
) -> dict[str, FakeSafeSlice]:
    """Wrap an HF state dict as the ``{checkpoint_key: slice}`` map a load needs.

    ``layer_idx`` is the decoder-layer index the state dict belongs to: each key
    is qualified to ``model.layers.<layer_idx>.<key>``, which is the checkpoint
    key form the fork's ``mappings`` dicts reference (see the mapping builder at
    ``vllm_neuron/model/llama3/model.py:1859-1893``). Pass ``layer_idx=None``
    for weights outside the layer stack (``model.embed_tokens.weight`` and
    friends); those are wrapped under their own keys.

    A key already qualified for ``layer_idx`` passes through unchanged, so
    wrapping is idempotent. A key qualified for a *different* layer raises
    rather than being re-prefixed into a key no mapping will ever ask for.
    """
    if layer_idx is not None and (
        isinstance(layer_idx, bool) or not isinstance(layer_idx, int) or layer_idx < 0
    ):
        raise ValueError(
            f"layer_idx must be None or a non-negative int, got {layer_idx!r}"
        )

    prefix = "" if layer_idx is None else f"{_LAYER_KEY_PREFIX}{layer_idx}."
    slices: dict[str, FakeSafeSlice] = {}
    for name, tensor in state_dict.items():
        if prefix and name.startswith(_LAYER_KEY_PREFIX) and not name.startswith(prefix):
            raise ValueError(
                f"state dict key {name!r} is qualified for another layer; "
                f"hf_state_to_fake_slices was asked for layer_idx={layer_idx}"
            )
        key = name if prefix and name.startswith(prefix) else f"{prefix}{name}"
        if key in slices:
            raise ValueError(f"two state dict keys qualify to the same key {key!r}")
        slices[key] = FakeSafeSlice(tensor)
    return slices


class LoadFromSlicesResult(NamedTuple):
    """Outcome of a fake load, mirroring ``CheckpointLoadResult``.

    Field meanings are that class's, verbatim
    (``vllm_neuron/utils/checkpoints.py:26-37``):

    - ``state_dict``: the tensors actually loaded, keyed by parameter name.
    - ``missing_keys``: parameter names whose checkpoint key(s) were absent.
    - ``unexpected_keys``: checkpoint keys no parameter's mapping referenced.
    """

    state_dict: dict[str, torch.Tensor]
    missing_keys: list[str]
    unexpected_keys: list[str]

    @property
    def num_loaded(self) -> int:
        """How many parameters received a tensor."""
        return len(self.state_dict)

    @property
    def unmatched_keys(self) -> list[str]:
        """Both directions of non-correspondence, in one list.

        A parameter nothing loaded into, and a checkpoint key nothing asked
        for, are both failures of the mapping under test, so a single count
        covers "0 unmatched".
        """
        return [*self.missing_keys, *self.unexpected_keys]


def _identity_load(slices: list[Any], rank: int) -> torch.Tensor:
    """Load a parameter that carries no loader: the first slice, as-is.

    Mirrors ``SafetensorsWeightLoader.load`` with ``transform=None``
    (``vllm_neuron/utils/weight_loader.py:66-74``), including its single-slice
    assertion and the ``.contiguous()`` on the way out.
    """
    if len(slices) != 1:
        raise AssertionError(
            "a parameter without a weight loader takes exactly one slice, "
            f"got {len(slices)}"
        )
    return slices[0][:].contiguous()


def _resolve_weight_loader(param: torch.nn.Parameter) -> Callable[[list[Any], int], torch.Tensor]:
    """Duck-type the loader attached to ``param``; never import the plugin.

    Accepts the fork's loader object (anything exposing
    ``load(slices, rank)``), a bare ``(slices, rank) -> tensor`` callable, or no
    loader at all. Anything else raises, so an attribute set to the wrong kind
    of object fails loudly instead of being skipped.
    """
    loader = getattr(param, WEIGHT_LOADER_ATTR, None)
    if loader is None:
        return _identity_load
    load = getattr(loader, "load", None)
    if callable(load):
        return load
    if callable(loader):
        return loader
    raise TypeError(
        f"{WEIGHT_LOADER_ATTR!r} is neither an object with .load(slices, rank) "
        f"nor a callable: {loader!r}"
    )


def load_weights_from_slices(
    module: torch.nn.Module,
    slice_map: Mapping[str, Any],
    mappings: Mapping[str, str | list[str]],
    rank: int,
    device: torch.device | str,
    *,
    strict: bool = True,
) -> LoadFromSlicesResult:
    """Drive each parameter's weight loader over ``slice_map`` and load ``module``.

    Args:
        module: the module whose ``named_parameters()`` are loaded. Loaders are
            read off the parameters themselves, so this is where the fork's real
            loader objects enter -- attached by the model code under test.
        slice_map: ``{checkpoint_key: slice}``, e.g. from
            :func:`hf_state_to_fake_slices`. Any object presenting
            ``get_shape()`` + ``__getitem__`` works.
        mappings: ``{parameter_name: checkpoint_key | [checkpoint_key, ...]}``.
            A parameter absent from ``mappings`` uses its own name as the key,
            exactly as the real path does (``checkpoints.py:242``).
        rank: tensor-parallel rank handed to every loader.
        device: where loaded tensors land (``tensor.to(device)``).
        strict: raise on a checkpoint key a mapping asks for but ``slice_map``
            lacks. ``False`` reports it in ``missing_keys`` instead. Default
            matches the real loader's.

    Returns:
        :class:`LoadFromSlicesResult`. Loaded tensors are also written into the
        parameters in place, so the module is ready to run.

    Raises:
        ValueError: a loader returned a tensor whose shape is not the
            parameter's. Silently accepting it would let a wrong loader pass a
            test, which is the whole failure this helper exists to catch.
    """
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise ValueError(f"rank must be a non-negative int, got {rank!r}")

    torch_device = torch.device(device)
    named_params = list(module.named_parameters())

    def keys_for(name: str) -> list[str]:
        checkpoint_keys = mappings.get(name, name)
        return (
            list(checkpoint_keys)
            if isinstance(checkpoint_keys, list)
            else [checkpoint_keys]
        )

    referenced_checkpoint_keys: set[str] = set()
    for name, _ in named_params:
        referenced_checkpoint_keys.update(keys_for(name))

    state_dict: dict[str, torch.Tensor] = {}
    missing_keys: list[str] = []

    for name, param in named_params:
        checkpoint_keys = keys_for(name)
        missing = [key for key in checkpoint_keys if key not in slice_map]
        if missing:
            if strict:
                raise RuntimeError(
                    f"Checkpoint key(s) not found for parameter '{name}': "
                    f"{missing}. Use strict=False to skip missing keys."
                )
            missing_keys.append(name)
            continue

        loader = _resolve_weight_loader(param)
        tensor = loader([slice_map[key] for key in checkpoint_keys], rank)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"weight loader for '{name}' returned "
                f"{type(tensor).__name__}, not a torch.Tensor"
            )
        if tuple(tensor.shape) != tuple(param.shape):
            raise ValueError(
                f"weight loader for '{name}' produced shape {tuple(tensor.shape)}, "
                f"parameter wants {tuple(param.shape)}"
            )

        tensor = tensor.to(torch_device)
        state_dict[name] = tensor
        param.data = tensor

    unexpected_keys = [
        key for key in slice_map if key not in referenced_checkpoint_keys
    ]
    return LoadFromSlicesResult(state_dict, missing_keys, unexpected_keys)
