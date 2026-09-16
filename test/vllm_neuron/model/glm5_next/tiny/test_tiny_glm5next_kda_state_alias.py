"""Every recurrent layer that shares one raw KV tensor keeps its own state bytes.

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
        python -m pytest -s -rA test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_kda_state_alias.py

vLLM's hybrid allocator hands one raw tensor to one layer of every KV cache group
(``KVCacheTensor.shared_by``). The recurrent layers address their state by request slot, so
two recurrent layers over one raw tensor need distinct storage offsets, or the slot one layer
writes is the slot its sharer reads. Two readings over the first shared tensor of the KV cache
configuration vLLM builds from the runner's own specs: the storage identity of every recurrent
state view, and a write through one layer's slot 0 read back through its sharer's slot 0. The
tiny root declares its first two layers recurrent, the way the e2e item substitutes specs.
"""

from __future__ import annotations

import pytest
import torch
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    get_uniform_page_size,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

from vllm_neuron.model.kv_cache import KVSpec
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as landed
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_first_request as first
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as item

pytestmark = [pytest.mark.fast, pytest.mark.forked]


def _production_kv_cache_config(runner: NeuronModelRunner, vllm_config, specs) -> KVCacheConfig:
    """The configuration vLLM's allocator builds from the runner's specs, sized for the generation's blocks."""
    groups = get_kv_cache_groups(vllm_config, specs)
    page_size = get_uniform_page_size([group.kv_cache_spec for group in groups])
    group_size = max(len(group.layer_names) for group in groups)
    return get_kv_cache_config_from_groups(vllm_config, groups, page_size * group_size * (landed.E2E_BLOCKS + 1))


def _two_recurrent_then_attention(root) -> KVSpec:
    """The tiny root's spec with its first two layers declared recurrent, so the recurrent layers outnumber the attention layer and land in separate groups."""
    recurrent = landed._recurrent_spec(root).layers
    original = root.get_kv_spec().layers
    return KVSpec(layers=list(recurrent[:2]) + list(original[2:]))


def _identity(view: torch.Tensor) -> tuple[int, int]:
    """Where a view's bytes start: its storage address and its element offset into that storage."""
    return view.untyped_storage().data_ptr(), view.storage_offset()


@pytest.fixture
def shared_tensor(tmp_path, monkeypatch):
    """The recurrent sharers of the first raw tensor and the state views the runner bound for them."""
    landed._require_cpu_mode()
    config = first._engine_config()
    root = landed._fixture()["root"]
    spec = _two_recurrent_then_attention(root)
    monkeypatch.setattr(root, "get_kv_spec", lambda: spec)
    with first._parallel_state(tmp_path, config):
        runner = NeuronModelRunner(config, device=torch.device("cpu"))
        runner.model = root
        runner.vocab_size = item.STACK_VOCAB_SIZE
        specs = runner.get_kv_cache_spec()
        kv_cache_config = _production_kv_cache_config(runner, config, specs)
        kv = runner.initialize_kv_cache(kv_cache_config)
    tensors = kv_cache_config.kv_cache_tensors
    sharers = [name for name in tensors[0].shared_by if isinstance(specs[name], MambaSpec)]
    print(
        f"ALIAS|config|num_blocks={kv_cache_config.num_blocks}|groups={[len(g.layer_names) for g in kv_cache_config.kv_cache_groups]}"
        f"|tensors={len(tensors)}|tensor0_shared_by={tensors[0].shared_by}|recurrent_sharers={sharers}"
    )
    for name in sharers:
        for index, view in enumerate(kv[name]):
            print(
                f"ALIAS|view|{name}|state={index}|data_ptr={view.untyped_storage().data_ptr()}"
                f"|storage_offset={view.storage_offset()}|shape={tuple(view.shape)}|stride={view.stride()}"
            )
    assert len(sharers) >= 2, f"the first raw tensor is shared by fewer than two recurrent layers: {tensors[0].shared_by}"
    return sharers, kv


def test_recurrent_sharers_of_one_raw_tensor_have_distinct_storage_or_offset(shared_tensor):
    """No two recurrent layers over the first raw tensor start their first state view at the same bytes."""
    sharers, kv = shared_tensor
    identities = {name: _identity(kv[name][0]) for name in sharers}
    aliased = sorted(name for name, identity in identities.items() if list(identities.values()).count(identity) > 1)
    print(f"ALIAS|reading|distinct_identities={len(set(identities.values()))}|of={len(sharers)}|aliased={aliased}")
    assert not aliased, f"recurrent layers over one raw tensor start at the same storage and offset: {aliased}"


def test_a_slot_written_through_one_recurrent_layer_stays_zero_in_its_sharer(shared_tensor):
    """Filling slot 0 of the first sharer's first state leaves slot 0 of the second sharer's first state zero."""
    sharers, kv = shared_tensor
    a, b = sharers[0], sharers[1]
    before = kv[b][0][0].abs().sum().item()
    kv[a][0][0].fill_(1)
    after = kv[b][0][0].abs().sum().item()
    print(f"ALIAS|write|{a}[0][0].fill_(1)|{b}[0][0].abs().sum()|before={before}|after={after}")
    assert before == 0, f"{b} slot 0 was not zero before the write: {before}"
    assert after == 0, f"a slot written through {a} is read back through {b}: {after} nonzero"
