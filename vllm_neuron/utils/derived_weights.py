# SPDX-License-Identifier: Apache-2.0
"""Load-time derived weights, computed on the host and placed whole on the weights' device.

The Neuron device's copy refuses a source of another dtype, other sizes or a non-contiguous
layout, and every op it does not register runs through its CPU fallback, so a layout of loaded
weights reads them whole to the host, runs there, and places each result by one copy.
"""

import torch


def derive_on_host(fn, *weights: torch.Tensor):
    """Run ``fn`` on host reads of ``weights``; place each result on the first weight's device."""
    device = weights[0].device
    if device.type == "meta":
        # The device-free compile lays out the unloaded tree, which needs shapes and dtypes only.
        return fn(*weights)
    for w in weights:
        # A host weight, such as a rank's shard view, is read as it is.
        if w.device.type != "cpu" and not (w.is_contiguous() and w.storage_offset() == 0):
            raise ValueError(
                f"a derived weight reads whole device tensors, got shape {tuple(w.shape)} with "
                f"strides {w.stride()} at storage offset {w.storage_offset()}"
            )
    out = fn(*(w.detach().cpu() for w in weights))
    if isinstance(out, tuple):
        return tuple(o.contiguous().to(device) for o in out)
    return out.contiguous().to(device)
