# SPDX-License-Identifier: Apache-2.0
"""Load the 5938748 KDA decode kernels and the decode region that composed them.

The four kernel files next to this one are byte copies taken with
``git show 5938748:vllm_neuron/functional/kda/<name>.py``. ``SNAPSHOT_SHA256``
pins them, so an edit to a snapshot fails the load instead of moving the
baseline.

``decode_state.py`` and ``gate_clamp.py`` import their emit helpers from the live
``vllm_neuron.functional.kda.chunked_recurrence``. The load checks that the live
file is byte-identical to the snapshot of it, so those helpers are the 5938748
helpers.

:func:`old_decode_core` is the 5938748 decode region of
``Glm5NextKDAAttention.forward`` (``model_fp8.py`` lines 3406-3559 at that
commit), transcribed for one decode token per request. It is the only code in
this directory that is not a snapshot: the region lives inside a 9000-line model
file, so it is copied here line for line instead of loading that file.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch

SNAPSHOT_SHA256 = {
    "chunked_recurrence.py": "1f68efe16e3dd76da447eae1b1f1c1e00a00a1fda6d5d9efcb2eb8ec00c06017",
    "decode_state.py": "c7d08f597bbc2481a9006a2098fa51897a56c62280b69167c5bef0e6ffc2480f",
    "depthwise_conv1d.py": "dea2a6c74fe85ea3fbc807234b16a83bee7dc119059778ce356d40e8f8fd39a2",
    "gate_clamp.py": "2cf1fce64ee4f4deb9b84c0d449ec09c012ce1e8b89e9e69c1c9ab3b1dc594fa",
}

HERE = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load baseline snapshot {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_baseline(directory: Path | str = HERE) -> SimpleNamespace:
    """The three 5938748 decode kernels as fresh modules, plus the composition."""
    directory = Path(directory).resolve()
    for name, digest in SNAPSHOT_SHA256.items():
        got = _sha256(directory / name)
        if got != digest:
            raise ValueError(
                f"baseline snapshot {directory / name} hashes to {got}, not the "
                f"5938748 copy {digest}"
            )
    import vllm_neuron.functional.kda.chunked_recurrence as live_chunked

    live = _sha256(Path(live_chunked.__file__))
    if live != SNAPSHOT_SHA256["chunked_recurrence.py"]:
        raise ValueError(
            "the live chunked_recurrence.py differs from 5938748, so the snapshot "
            "decode_state/gate_clamp would import changed helpers"
        )
    tag = f"_kda_5938748_{abs(hash(str(directory)))}"
    return SimpleNamespace(
        directory=directory,
        depthwise_conv1d=_load(directory / "depthwise_conv1d.py", f"{tag}_conv"),
        gate_clamp=_load(directory / "gate_clamp.py", f"{tag}_gate"),
        decode_state=_load(directory / "decode_state.py", f"{tag}_decode"),
    )


# --------------------------------------------------------------------------- #
# The 5938748 decode region, transcribed.
# --------------------------------------------------------------------------- #
def _int64_scalar(value, device):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=torch.int64).reshape(())
    return torch.full((), int(value), dtype=torch.int64, device=device)


def _start_is_zero(start_position, device):
    return _int64_scalar(start_position, device) == 0


def old_decode_core(
    kernels: SimpleNamespace,
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    *,
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    q_conv1d_weight: torch.Tensor,
    k_conv1d_weight: torch.Tensor,
    v_conv1d_weight: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    gate_lower_bound: float,
    conv_state_dim_first: bool,
    start_position=0,
    real_tokens=None,
    row_mask=None,
) -> torch.Tensor:
    """One request's KDA decode region as 5938748 ran it; returns ``core``.

    ``q_in``..``raw_beta`` are the projections for this request's ``T`` tokens,
    ``conv_state`` is ``[R, C]`` (``[C, R]`` when ``conv_state_dim_first``) and
    ``recurrent_state`` is ``[H, V, K]``; both are written in place, as the model
    writes the bank views. Lines are the model's, with ``self.`` replaced by
    arguments.
    """
    depthwise_conv1d = kernels.depthwise_conv1d.depthwise_conv1d
    kda_gate_clamp = kernels.gate_clamp.kda_gate_clamp
    GATE_MAX_TILE = kernels.gate_clamp.MAX_TILE
    kda_decode_step = kernels.decode_state.kda_decode_step

    tokens = int(q_in.shape[0])
    heads, kdim = int(recurrent_state.shape[0]), int(recurrent_state.shape[2])
    width = heads * kdim
    state_rows = int(q_conv1d_weight.shape[-1]) - 1

    opening = _start_is_zero(start_position, conv_state.device)

    conv_in = torch.cat((q_in, k_in, v_in), dim=-1)
    rows = conv_state.t() if conv_state_dim_first else conv_state
    history = rows.to(torch.float32)
    history = torch.where(opening, torch.zeros_like(history), history)
    padded = torch.cat((history, conv_in), dim=0)
    channels = 3 * width
    img = padded.t().contiguous().reshape(1, channels, 1, state_rows + tokens)
    filt = torch.cat(
        (
            q_conv1d_weight.to(torch.float32).reshape(width, 1, 1, -1),
            k_conv1d_weight.to(torch.float32).reshape(width, 1, 1, -1),
            v_conv1d_weight.to(torch.float32).reshape(width, 1, 1, -1),
        ),
        dim=0,
    ).contiguous()
    conv_out = depthwise_conv1d(img, filt)
    conv_out = conv_out.reshape(channels, tokens).t()
    conv_out = torch.nn.functional.silu(conv_out)
    q_conv, k_conv, v_conv = conv_out.split(width, dim=-1)
    real_length = _int64_scalar(
        tokens if real_tokens is None else real_tokens, padded.device
    )
    history_index = torch.arange(state_rows, device=padded.device) + real_length
    stored = padded.index_select(0, history_index)
    value = stored.t() if conv_state_dim_first else stored
    conv_state.copy_(value.to(conv_state.dtype))

    a_log = A_log.to(torch.float32).reshape(-1)
    bias = dt_bias.to(torch.float32).reshape(-1)

    def clamp_one_head(h: int) -> torch.Tensor:
        span = slice(h * kdim, (h + 1) * kdim)
        tiles = [
            kda_gate_clamp(
                raw_gate[start : start + GATE_MAX_TILE, span].contiguous(),
                a_log[h],
                bias=bias[span],
                lower=gate_lower_bound,
            )
            for start in range(0, tokens, GATE_MAX_TILE)
        ]
        return tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=0)

    gate_parts = [clamp_one_head(h) for h in range(heads)]
    beta = torch.sigmoid(raw_beta)
    if row_mask is not None:
        gate_parts = [part * row_mask for part in gate_parts]
        beta = beta * row_mask

    core = torch.empty(tokens, width, dtype=torch.float32, device=q_conv.device)
    for h in range(heads):
        span = slice(h * kdim, (h + 1) * kdim)
        q_h = q_conv[:, span].contiguous()
        k_h = k_conv[:, span].contiguous()
        v_h = v_conv[:, span].contiguous()
        gk_h = gate_parts[h]
        beta_h = beta[:, h].contiguous()
        carried = recurrent_state[h].to(torch.float32)
        state = torch.where(opening, torch.zeros_like(carried), carried)
        for t in range(tokens):
            step = kda_decode_step(
                state,
                q_h[t : t + 1],
                k_h[t : t + 1],
                v_h[t : t + 1],
                beta_h[t].reshape(1, 1),
                gk_h[t : t + 1],
            )
            core[t : t + 1, span] = step.o.reshape(1, kdim)
            state = step.state
        recurrent_state[h] = state.to(recurrent_state.dtype)
    return core
