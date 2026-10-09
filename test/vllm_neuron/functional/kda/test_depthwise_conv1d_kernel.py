# SPDX-License-Identifier: Apache-2.0
"""The KDA depthwise conv1d kernel against the definition, over every axis it tiles.

Each case runs the public :func:`depthwise_conv1d` in the NKI simulator, once on one
program and once on the ``LNC_SHARDS`` programs ``NEURON_LOGICAL_NC_CONFIG=2``
selects, and compares both with a direct sum computed here in float64 from the
definition of a padded, strided cross-correlation::

    out[n, c, q] = sum_s filt[c, s] * x_pad[n, c, q * stride_w + s]

where ``x_pad`` is the input with ``pad_left`` and ``pad_right`` zero columns. No
convolution operator is used on the expected side.

The bound is derived rather than tuned. The kernel forms each product and each
partial sum in float32, so the float32 error of one output is at most
``gamma_S * sum_s |filt[c, s] * x_pad[n, c, q * stride_w + s]|`` with
``gamma_S ~= S * 2**-24`` (S roundings deep: one per product, at most ``S - 1``
per sum); ``S * eps32`` is twice that. A non-float32 output then rounds once more,
by at most half an ulp of the output dtype, bounded here by one ``eps`` of it.

The geometries cover what the kernel tiles: channel tiles of
``PARTITION_MAX`` with and without a tail, column tiles of up to ``COL_TILE``
with several per program, the column split across programs (including a program
with no columns), batches, tap counts that leave the second accumulation chain
empty or one tap long, padding on either side and wider than the input, strides
that cap the input window, and the three float dtypes the wrap admits.
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.kda import depthwise_conv1d as conv
from vllm_neuron.functional.kda import depthwise_conv1d_kernel as kernel

PARTITIONS = kernel.PARTITION_MAX


def _case(batches, channels, width, taps, pads=(0, 0), stride_w=1, dtype=torch.float32):
    return {"batches": batches, "channels": channels, "width": width, "taps": taps,
            "pads": pads, "stride_w": stride_w, "dtype": dtype}


def _served_width(columns, taps=4):
    """The input width of a served call: ``columns`` outputs from ``taps - 1`` history."""
    return columns + taps - 1


#: ``id -> geometry``. Extents are written from the tiling constants, so a change in
#: ``COL_TILE`` or the partition count keeps each case on the boundary it names.
CASES = {
    # The served prefill shapes: one and two column tiles per program under LNC2.
    "served_1k": _case(1, 3 * PARTITIONS, _served_width(2 * kernel.COL_TILE), 4),
    "served_2k": _case(1, 3 * PARTITIONS, _served_width(4 * kernel.COL_TILE), 4),
    # A channel tail on an odd channel count, and fewer channels than one tile.
    "channel_tail": _case(1, PARTITIONS + 7, 40, 4),
    "few_channels": _case(1, 3, 20, 4),
    # Columns that split unevenly across programs and across column tiles.
    "uneven_columns": _case(1, PARTITIONS, _served_width(2 * kernel.COL_TILE + 273), 4),
    # Fewer output columns than programs: the second program has none.
    "one_column": _case(1, 8, 4, 4),
    "batches": _case(2, 2 * PARTITIONS, 33, 4),
    # One tap leaves the second chain empty; two give two one-tap chains.
    "taps_1": _case(1, PARTITIONS + 2, 17, 1),
    "taps_2": _case(1, PARTITIONS, 64, 2),
    "taps_3": _case(1, PARTITIONS, 64, 3),
    "taps_5": _case(1, PARTITIONS, 64, 5),
    "taps_9": _case(1, 64, 64, 9),
    "pad_left": _case(1, PARTITIONS + 1, 50, 4, (3, 0)),
    "pad_right": _case(1, PARTITIONS + 1, 50, 4, (0, 3)),
    "pad_both": _case(1, 64, 9, 4, (2, 5)),
    # Padding wider than the input: some windows hold no input column at all.
    "pad_wider_than_input": _case(1, 16, 2, 4, (3, 3)),
    "stride_2": _case(1, PARTITIONS + 2, 101, 4, (1, 2), 2),
    "stride_3": _case(1, 64, 64, 3, (0, 0), 3),
    # A stride whose window caps the column tile below COL_TILE, several tiles per program.
    "stride_window_cap": _case(1, 8, 3 * kernel.WINDOW_MAX, 4, (0, 0), 40),
    "bfloat16": _case(1, PARTITIONS + 8, 70, 4, (1, 1), 1, torch.bfloat16),
    "float16": _case(1, PARTITIONS + 8, 70, 4, (1, 1), 1, torch.float16),
}


def _operands(case, seed):
    generator = torch.Generator().manual_seed(seed)
    img = torch.randn(case["batches"], case["channels"], 1, case["width"], generator=generator)
    filt = torch.randn(case["channels"], 1, 1, case["taps"], generator=generator)
    return img.to(case["dtype"]), filt.to(case["dtype"])


def _direct_sum(img, filt, pads, stride_w):
    """``(exact, magnitude)`` in float64 from the definition, and the same sum over ``|.|``."""
    x = torch.nn.functional.pad(img.double()[:, :, 0, :], pads)
    taps = filt.shape[3]
    columns = (x.shape[-1] - taps) // stride_w + 1
    w = filt.double()[:, 0, 0, :]
    exact = torch.zeros(x.shape[0], x.shape[1], columns, dtype=torch.float64)
    magnitude = torch.zeros_like(exact)
    for s in range(taps):
        window = x[:, :, s:s + (columns - 1) * stride_w + 1:stride_w]
        exact += w[None, :, s:s + 1] * window
        magnitude += (w[None, :, s:s + 1] * window).abs()
    return exact[:, :, None, :], magnitude[:, :, None, :]


def _bound(magnitude, exact, taps, dtype):
    """The float32 accumulation bound, plus one output-dtype rounding off float32."""
    bound = taps * torch.finfo(torch.float32).eps * magnitude
    if dtype != torch.float32:
        bound = bound + torch.finfo(dtype).eps * exact.abs()
    return bound


def _run(case, img, filt, lnc, monkeypatch):
    if lnc is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    conv.reset_dispatch_counters()
    out = conv.depthwise_conv1d(img, filt, padding=((0, 0), case["pads"]),
                                stride=(1, case["stride_w"]))
    assert conv.dispatch_counters() == (1, 0), conv.dispatch_counters()
    return out


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_kernel_matches_the_definition_on_one_and_two_programs(name, monkeypatch):
    """Both launches agree with the direct sum within the bound, and bit for bit with each other."""
    case = CASES[name]
    img, filt = _operands(case, seed=sorted(CASES).index(name))
    exact, magnitude = _direct_sum(img, filt, case["pads"], case["stride_w"])
    bound = _bound(magnitude, exact, case["taps"], case["dtype"])

    one = _run(case, img, filt, None, monkeypatch)
    two = _run(case, img, filt, "2", monkeypatch)

    assert one.dtype == case["dtype"] and two.dtype == case["dtype"]
    assert tuple(one.shape) == tuple(exact.shape), (tuple(one.shape), tuple(exact.shape))
    for label, got in (("one program", one), ("two programs", two)):
        excess = (got.double() - exact).abs() - bound
        assert excess.max().item() <= 0.0, (
            f"{name}, {label}: {int((excess > 0).sum())} outputs outside the bound, worst by "
            f"{excess.max().item():.3e}")
    assert torch.equal(one.view(torch.uint8), two.view(torch.uint8)), (
        f"{name}: the column split across programs changed bits")


@pytest.mark.parametrize("name", ["served_1k", "pad_both", "stride_2", "bfloat16"])
def test_the_torch_reference_matches_the_definition(name):
    """The module's reference meets the same bound, so a kernel-to-reference comparison is fair."""
    case = CASES[name]
    img, filt = _operands(case, seed=97)
    exact, magnitude = _direct_sum(img, filt, case["pads"], case["stride_w"])
    reference = conv.depthwise_conv1d_torch_reference(
        img, filt, padding=((0, 0), case["pads"]), stride=(1, case["stride_w"]))
    assert reference.dtype == case["dtype"]
    excess = (reference.double() - exact).abs() - _bound(magnitude, exact, case["taps"],
                                                          case["dtype"])
    assert excess.max().item() <= 0.0, excess.max().item()


def test_the_bound_is_not_vacuous():
    """At the served shape the bound sits near float32 resolution, not near the outputs' size."""
    case = CASES["served_1k"]
    img, filt = _operands(case, seed=0)
    exact, magnitude = _direct_sum(img, filt, case["pads"], case["stride_w"])
    bound = _bound(magnitude, exact, case["taps"], case["dtype"])
    assert bound.max().item() < 1e-4 * exact.abs().max().item()
