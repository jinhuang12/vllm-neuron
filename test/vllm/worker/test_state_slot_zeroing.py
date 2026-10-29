# SPDX-License-Identifier: Apache-2.0
"""A new request's state slots are zeroed by row copies, never by an op on a whole page buffer.

The runner is bare, with one attention group and one state group of 3 layers. The layers'
page buffers are random nonzero int8 CPU tensors. Four requests arrive: three at 0 computed
tokens with first state blocks 9, 4 and 5, and one with computed tokens at block 7. A torch
function mode stands in for the device backend's rule on the buffers. Run with
``VLLM_NEURON_CPU_MODE=1``.
"""

import ast
from pathlib import Path

import torch
from torch.overrides import TorchFunctionMode
from vllm.sampling_params import SamplingParams
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_neuron.vllm.worker import neuron_model_runner
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

BLOCKS, PAGE = 12, 24
LAYERS = ["state.0", "state.1", "state.2"]
NEW_BLOCKS = [4, 5, 9]
WHOLE_BUFFER_WRITES = {"index_fill_", "index_put_", "fill_", "zero_", "masked_fill_"}

METADATA = {"shape", "dtype", "device", "layout", "ndim", "size", "stride", "dim", "numel",
            "element_size", "is_contiguous", "storage_offset", "untyped_storage", "data_ptr"}
VIEW_OPS = {"detach", "select", "narrow", "view", "reshape", "as_strided"}


class DeviceRule(TorchFunctionMode):
    """The backend's rule on marked buffers: metadata reads, detach, views and one copy form.

    The allowed copy writes a contiguous marked view from a contiguous unmarked host tensor of
    the same dtype and size. Any other call with a marked argument is refused and recorded.
    """

    def __init__(self, buffers):
        super().__init__()
        self.storages = {b.untyped_storage().data_ptr() for b in buffers}
        self.refusals, self.copies = [], []

    def marked(self, x) -> bool:
        """True for a tensor on a marked storage, so every view of a buffer is marked."""
        return isinstance(x, torch.Tensor) and x.untyped_storage().data_ptr() in self.storages

    def __torch_function__(self, func, types_, args=(), kwargs=None):
        kwargs = kwargs or {}
        owner = getattr(func, "__self__", None)
        name = owner.__name__ if type(owner).__name__ == "getset_descriptor" else func.__name__
        flat = [x for a in (*args, *kwargs.values())
                for x in (a if isinstance(a, (list, tuple)) else [a])]
        marked = [x for x in flat if self.marked(x)]
        if not marked:
            return func(*args, **kwargs)
        if name in METADATA or name in VIEW_OPS:
            return func(*args, **kwargs)
        if name == "__getitem__" and isinstance(args[1], (int, slice)):
            return func(*args, **kwargs)
        if name == "copy_" and self._allowed_copy(marked, *args[:2]):
            self.copies.append(tuple(args[0].shape))
            return func(*args, **kwargs)
        self.refusals.append(name)
        raise RuntimeError(f"the device refuses {name} on a state page buffer")

    @staticmethod
    def _allowed_copy(marked, dst, src) -> bool:
        """Only ``dst`` is marked, both are contiguous, and the dtype and size match."""
        return (isinstance(src, torch.Tensor) and len(marked) == 1 and marked[0] is dst
                and dst.is_contiguous() and src.is_contiguous() and src.device.type == "cpu"
                and src.dtype == dst.dtype and src.shape == dst.shape)


def _runner():
    """A bare runner holding 3 random page buffers, and the buffers as they were."""
    gen = torch.Generator().manual_seed(0)
    pages = {name: torch.randint(1, 127, (BLOCKS, PAGE), dtype=torch.int8, generator=gen)
             for name in LAYERS}
    attention = FullAttentionSpec(block_size=8, num_kv_heads=1, head_size=8, dtype=torch.int8)
    state = MambaSpec(shapes=((3, 8),), dtypes=(torch.int8,), block_size=8, page_size_padded=PAGE)
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.kv_cache_config = KVCacheConfig(
        num_blocks=BLOCKS, kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["attention.0"], attention),
                         KVCacheGroupSpec(LAYERS, state)])
    runner._state_pages, runner._state_zero_rows = pages, {}
    runner.device = torch.device("cpu")
    return runner, {name: buffer.clone() for name, buffer in pages.items()}


def _requests() -> list[CachedRequestState]:
    """Three new requests at state blocks 9, 4 and 5, and one with computed tokens at block 7."""
    def request(name, block, computed):
        return CachedRequestState(
            req_id=name, prompt_token_ids=[1, 2, 3], mm_features=[],
            sampling_params=SamplingParams(), generator=None, block_ids=([1, 2], [block]),
            num_computed_tokens=computed, output_token_ids=[])
    return [request("a", 9, 0), request("b", 4, 0), request("c", 7, 3), request("d", 5, 0)]


def test_zeroing_passes_the_device_rule():
    """The zeroing completes with no refusal: only the two runs' rows are copied, per layer."""
    runner, _ = _runner()
    with DeviceRule(runner._state_pages.values()) as rule:
        runner._zero_new_state_slots(_requests())
    assert rule.refusals == []
    assert rule.copies == [(2, PAGE), (1, PAGE)] * len(LAYERS)


def test_zeroing_clears_the_new_rows_only():
    """The new requests' rows read 0 in every layer, and no other byte changes."""
    runner, before = _runner()
    runner._zero_new_state_slots(_requests())
    others = [b for b in range(BLOCKS) if b not in NEW_BLOCKS]
    for name, buffer in runner._state_pages.items():
        assert not buffer[NEW_BLOCKS].any(), name
        assert torch.equal(buffer[others], before[name][others]), name


def test_runner_source_has_no_whole_buffer_write():
    """The runner's source calls none of the ops that write a whole buffer or a masked part."""
    tree = ast.parse(Path(neuron_model_runner.__file__).read_text())
    calls = [(node.lineno, node.func.attr if isinstance(node.func, ast.Attribute)
              else getattr(node.func, "id", None))
             for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert [hit for hit in calls if hit[1] in WHOLE_BUFFER_WRITES] == []
