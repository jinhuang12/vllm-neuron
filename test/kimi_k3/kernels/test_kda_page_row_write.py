# SPDX-License-Identifier: Apache-2.0
"""KDA page-row write kernel against the model's ``index_copy_`` write of the slot's row.

The buffer is the served one: 251 pages of 294,912 random bytes. Each case writes one
slot's conv rows [3, 1152] bf16 and state [3, 128, 128] fp32 into the buffer with the
kernel, and the same parts into a copy by an ``index_copy_`` of the slot's whole row,
the parts placed by the model's own helper at the runner's offsets. No byte may differ:
not in the row's parts, not in the row's tail and not in any other row. The odd-bytes
case holds NaN payloads, a signalling NaN, subnormals and negative zero.

With ``VLLM_NEURON_CPU_MODE=1`` and ``NKI_SIMULATOR=1`` the kernel runs on the NKI
simulator; otherwise on ``neuron:0``.
"""

import pytest
import torch

from vllm_neuron import envs
from vllm_neuron.functional.kda import page_row_write
from vllm_neuron.model.kimi_k3 import model as m
from vllm_neuron.utils.neuron_utils import can_run_kernel

ON_DEVICE = not envs.VLLM_NEURON_CPU_MODE
pytestmark = pytest.mark.skipif(
    not (ON_DEVICE or can_run_kernel("cpu")), reason="the kernel needs a Neuron device or NKI_SIMULATOR=1"
)

BLOCKS, PAGE, HEADS, DIM, WIDTH = 251, 294_912, 3, 128, 1152
CONV_BYTES = 3 * WIDTH * 2
STATE_END = CONV_BYTES + HEADS * DIM * DIM * 4
LAYOUTS = ((0, torch.bfloat16, (3, WIDTH)), (CONV_BYTES, torch.float32, (HEADS, DIM, DIM)))

_compiled = torch.compile(page_row_write.kda_page_row_write, backend=envs.get_compile_backend_name(),
                          fullgraph=True, dynamic=False) if ON_DEVICE else None


def _random_parts(gen):
    conv = torch.randn(1, 3, WIDTH, generator=gen).to(torch.bfloat16)
    return conv, torch.randn(1, HEADS, DIM, DIM, generator=gen)


def _odd_parts(gen):
    # bf16: quiet NaN with payload, signalling NaN, negative quiet NaN, subnormals, -0.
    odd16 = torch.tensor([0x7FC1, 0x7F81, -0x003D, 0x0001, -0x7F81, -0x8000], dtype=torch.int16)
    # fp32: the same classes.
    odd32 = torch.tensor([0x7FC00001, 0x7F800001, -0x003FFFFD, 0x00000001, -0x7F800001, -0x80000000],
                         dtype=torch.int32)
    conv = odd16.repeat(3 * WIDTH // 6).view(torch.bfloat16).view(1, 3, WIDTH)
    return conv, odd32.repeat(HEADS * DIM * DIM // 6).view(torch.float32).view(1, HEADS, DIM, DIM)


def _index_copy_write(pages, index, conv, state):
    """Writes the parts into ``pages`` by ``index_copy_`` of the slot's whole row."""
    parts = tuple(zip(LAYOUTS, (conv, state)))
    pages.index_copy_(0, index, m._with_parts(pages.index_select(0, index), parts))


def _kernel_write(pages, index, conv, state):
    """The buffer after the kernel's write, read back on the CPU."""
    if ON_DEVICE:
        on = [t.to("neuron:0") for t in (pages, index, conv, state)]
        _compiled(*on)
        return on[0].cpu()
    page_row_write.kda_page_row_write(pages, index, conv, state)
    return pages


@pytest.mark.parametrize("parts, slot", [(_random_parts, 1), (_random_parts, 7), (_random_parts, 250),
                                         (_odd_parts, 250)], ids=["random-1", "random-7", "random-250",
                                                                  "odd-250"])
def test_bytes_match_the_index_copy_write(parts, slot):
    """0 bytes differ over the row's parts, the row's tail and every other row."""
    gen = torch.Generator().manual_seed(slot)
    pages = torch.randint(-128, 128, (BLOCKS, PAGE), dtype=torch.int8, generator=gen)
    conv, state = parts(gen)
    index = torch.tensor([slot])
    expected = pages.clone()
    _index_copy_write(expected, index, conv, state)
    differ = _kernel_write(pages, index, conv, state) != expected
    counts = {"row's parts": differ[slot, :STATE_END].sum().item(),
              "row's tail": differ[slot, STATE_END:].sum().item(),
              "other rows": differ.sum().item() - differ[slot].sum().item()}
    print(f"slot {slot}, {parts.__name__}: bytes that differ {counts}")
    assert counts == dict.fromkeys(counts, 0), counts
