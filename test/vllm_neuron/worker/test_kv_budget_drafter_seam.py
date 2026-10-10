# SPDX-License-Identifier: Apache-2.0
"""The worker's need-sized KV pool admits vLLM's own requirement with a drafter.

``NeuronWorker._kv_cache_need_bytes`` is what ``determine_available_memory`` hands vLLM
as the available KV memory, and vLLM starts the engine only if its own requirement for
one request of ``max_model_len`` tokens fits in it: ``_check_enough_kv_cache_memory``
over the KV cache groups (``_max_memory_usage_bytes_from_groups``; the layer sum
``max_memory_usage_bytes`` is the same figure to within one page per group). A
speculative server changes both sides: its recurrent layers hold ``1 +
num_speculative_blocks`` state rows per request (``MambaSpec.max_memory_usage_bytes``;
the scheduler allocates them), and its drafter adds an attention layer, which regroups
the layers (12 attention + 34 recurrent -> 4 groups of 12 with 2 padding slots, against
the plain 5 groups of 11 with 10). The need must count ``k`` extra blocks per recurrent
group per request, or a spec-decoding serve does not start: with 3 drafts and one
4096-token request vLLM asks 520 pages of the real layers (528 by groups) and a pool
that ignores the draft rows offers 432 -- "0.06 GiB KV cache is needed, which is larger
than the available KV cache memory (0.05 GiB)".

Expectations are derived from the fixture's layer specs, vLLM's own grouping and vLLM's
own admission functions; no served figure is an expectation.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_kv_budget_drafter_seam.py
"""
from __future__ import annotations

import copy

import pytest
from vllm.utils.math_utils import cdiv

from test.vllm_neuron.worker import test_kv_budget_glm53f as kv
from test.vllm_neuron.worker.test_mtp_kv_budget import _mtp

K = 3
#: The served bs=1 line.
SEQS, LENGTH = 1, 4096
#: The bs=64 @ 8k line.
B64_SEQS, B64_LENGTH = 64, 8192
#: The drafter's attention layer, named as the MTP head's module is.
DRAFTER_LAYER = "mtp.layers.0.self_attn"


def _layers(*, drafter: bool):
    """The served model's layer specs, plus the drafter's attention layer when asked.

    An MTP head's attention layer has the trunk's attention geometry, so its spec is a
    trunk attention layer's under the head's name.
    """
    layers, text_config = kv.glm53f_layer_specs()
    if drafter:
        attention = next(
            layer for layer in layers if layer.kda_recurrent_state_shape is None
        )
        head = copy.copy(attention)
        head.name = DRAFTER_LAYER
        layers = [*layers, head]
    return layers, text_config


def _worker(*, drafter: bool, k: int, seqs: int, length: int):
    layers, text_config = _layers(drafter=drafter)
    runner = kv.fake_runner(
        layers, max_num_seqs=seqs, max_model_len=length, text_config=text_config
    )
    if k:
        _mtp(runner, k)
    return kv.fake_worker(runner)


def _groups(worker):
    """vLLM's own grouping of the worker's KV specs."""
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

    return get_kv_cache_groups(worker.vllm_config, worker.model_runner.get_kv_cache_spec())


def _layout(worker) -> tuple[int, int, int]:
    """(groups, layers per pool, padding layers) as the serve log states them."""
    groups = _groups(worker)
    layers_per_pool = max(len(group.layer_names) for group in groups)
    real = len(worker.model_runner.get_kv_cache_spec())
    return len(groups), layers_per_pool, len(groups) * layers_per_pool - real


def _recurrent_groups(worker) -> int:
    from vllm.v1.kv_cache_interface import MambaSpec

    return sum(isinstance(group.kv_cache_spec, MambaSpec) for group in _groups(worker))


def _page_size(worker) -> int:
    from vllm.v1.core.kv_cache_utils import get_uniform_page_size

    return get_uniform_page_size([group.kv_cache_spec for group in _groups(worker)])


def _vllm_admits(worker, need: int):
    """Run vLLM's own admission on ``need`` as the available KV memory and return the
    KV cache config it builds -- ``get_kv_cache_configs`` raises the serve-time
    "KV cache is needed, which is larger than the available KV cache memory" when the
    pool is short, so a returned config is admission."""
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

    (config,) = get_kv_cache_configs(
        worker.vllm_config, [worker.model_runner.get_kv_cache_spec()], [need]
    )
    return config


def _vllm_requirement(worker) -> int:
    """vLLM's bytes for one request of ``max_model_len`` tokens summed over the real
    layers; its group-based admission check (``_check_enough_kv_cache_memory``) rounds
    each group up to whole pages, so this is a floor of that check's figure."""
    from vllm.v1.core.kv_cache_utils import max_memory_usage_bytes

    return max_memory_usage_bytes(
        worker.vllm_config, worker.model_runner.get_kv_cache_spec().values()
    )


def test_the_plain_and_drafter_layouts_group_as_the_serve_logs_state() -> None:
    """A spec-off server of this model logs ``5 group(s), 11 layer(s) per pool`` with 10
    padding slots; with the drafter's layer it logs ``4 group(s), 12 layer(s) per pool``
    with 2. Both follow from vLLM's grouping of the fixture's specs."""
    assert _layout(_worker(drafter=False, k=0, seqs=SEQS, length=LENGTH)) == (5, 11, 10)
    assert _layout(_worker(drafter=True, k=K, seqs=SEQS, length=LENGTH)) == (4, 12, 2)


@pytest.mark.parametrize(
    "drafter, k",
    [(False, 0), (True, 0), (True, K)],
    ids=["plain", "drafter-layer-without-drafting", "spec3-as-served"],
)
def test_the_need_admits_vllms_own_requirement(drafter: bool, k: int) -> None:
    """The need the worker hands vLLM covers vLLM's requirement on the plain line, with
    the drafter's layer alone (the regrouping is not the gap), and on the served spec3
    line (the recurrent checkpoint blocks are) -- and vLLM's own admission, run on the
    need, builds the pool the worker priced: the attention pages plus ``1 + k`` blocks
    per recurrent group per request, plus the null block."""
    worker = _worker(drafter=drafter, k=k, seqs=SEQS, length=LENGTH)
    need = worker._kv_cache_need_bytes()
    assert need >= _vllm_requirement(worker)
    block_size = worker.vllm_config.cache_config.block_size
    assert _vllm_admits(worker, need).num_blocks == (
        (cdiv(LENGTH, block_size) + _recurrent_groups(worker) * (1 + k)) * SEQS + 1
    )


def test_the_served_spec3_line_is_priced_in_whole_blocks_of_the_pool() -> None:
    """On the served line (drafter, k = 3, one request of 4096 tokens) vLLM asks for the
    attention layers' pages plus ``1 + k`` pages per recurrent layer; the worker's pool
    holds, per request, the attention blocks plus ``1 + k`` blocks per recurrent group,
    plus the null block, each block one page per pool layer -- and that pool admits
    vLLM's figure."""
    worker = _worker(drafter=True, k=K, seqs=SEQS, length=LENGTH)
    page = _page_size(worker)
    block_size = worker.vllm_config.cache_config.block_size
    specs = worker.model_runner.get_kv_cache_spec().values()
    attention_layers = sum(layer.kda_recurrent_state_shape is None for layer in _layers(drafter=True)[0])
    recurrent_layers = len(specs) - attention_layers
    assert _vllm_requirement(worker) == (
        attention_layers * cdiv(LENGTH, block_size) + recurrent_layers * (1 + K)
    ) * page
    groups, layers_per_pool, _ = _layout(worker)
    blocks_per_request = cdiv(LENGTH, block_size) + _recurrent_groups(worker) * (1 + K)
    assert worker._kv_cache_need_bytes() == (blocks_per_request * SEQS + 1) * layers_per_pool * page
    assert worker._kv_cache_need_bytes() >= _vllm_requirement(worker)


def test_the_need_grows_by_k_blocks_per_recurrent_group_per_request() -> None:
    """From k = 0 to k = K on the bs=64 @ 8k line the need grows by exactly K blocks per
    recurrent group per request, each block ``page_size x layers_per_pool`` bytes -- the
    rows the scheduler hands a drafting request and nothing else."""
    worker = _worker(drafter=False, k=0, seqs=B64_SEQS, length=B64_LENGTH)
    plain = worker._kv_cache_need_bytes()
    _mtp(worker.model_runner, K)
    _, layers_per_pool, _ = _layout(worker)
    assert worker._kv_cache_need_bytes() - plain == (
        K * _recurrent_groups(worker) * B64_SEQS * _page_size(worker) * layers_per_pool
    )


def test_a_spec_class_the_need_cannot_price_is_refused_by_name(monkeypatch) -> None:
    """The need prices an attention group at its pages and a recurrent group at
    ``1 + num_speculative_blocks`` blocks per request. A KV spec of another class has no
    place in that arithmetic; pricing it as attention would hand vLLM a pool it may not
    fit, so the worker refuses it by the class's name instead of guessing."""
    from vllm.v1 import kv_cache_interface as kvi
    from vllm.v1.core import kv_cache_utils

    class ForeignSpec(kvi.KVCacheSpec):
        @property
        def type_id(self) -> str:
            return "foreign"

        @property
        def page_size_bytes(self) -> int:
            return 131072

        def max_memory_usage_bytes(self, vllm_config) -> int:
            return self.page_size_bytes

    worker = _worker(drafter=False, k=0, seqs=SEQS, length=LENGTH)
    foreign = kvi.KVCacheGroupSpec(
        layer_names=["layers.0.foreign"], kv_cache_spec=ForeignSpec(block_size=128)
    )
    monkeypatch.setattr(
        kv_cache_utils, "get_kv_cache_groups", lambda *args, **kwargs: [foreign]
    )
    with pytest.raises(ValueError, match="ForeignSpec"):
        worker._kv_cache_need_bytes()


def _recurrent_layers_at_two_geometries():
    """The served model's recurrent layers plus one more whose recurrent state has half
    the rows: layers of one KV type at differing geometries, which vLLM merges into one
    ``UniformTypeKVCacheSpecs`` group (the layout a drafter with its own KV geometry
    produces; vLLM PR 25101)."""
    layers, text_config = kv.glm53f_layer_specs()
    recurrent = [layer for layer in layers if layer.kda_recurrent_state_shape is not None]
    head = copy.copy(recurrent[0])
    head.name = "mtp.layers.0.linear_attn"
    heads, rows, columns = head.kda_recurrent_state_shape
    head.kda_recurrent_state_shape = (heads, rows // 2, columns)
    return [*recurrent, head], text_config


def test_a_uniform_type_group_is_priced_by_what_its_layers_hold() -> None:
    """vLLM merges layers of one KV type at differing geometries into a single
    ``UniformTypeKVCacheSpecs`` group, which the runner allocates (a drafter with its own
    KV geometry). The need prices that group by what its member layers hold -- recurrent
    layers here, so ``k`` draft blocks per request -- rather than refusing a class the
    server serves, and vLLM's own admission accepts the result."""
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    layers, text_config = _recurrent_layers_at_two_geometries()
    runner = kv.fake_runner(
        layers, max_num_seqs=SEQS, max_model_len=LENGTH, text_config=text_config
    )
    worker = kv.fake_worker(runner)
    (group,) = _groups(worker)
    assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    assert len(group.layer_names) == len(layers)
    plain = worker._kv_cache_need_bytes()
    _mtp(runner, K)
    drafting = worker._kv_cache_need_bytes()
    _, layers_per_pool, _ = _layout(worker)
    assert drafting - plain == K * SEQS * _page_size(worker) * layers_per_pool
    # vLLM sizes a uniform-type group's pool by the group's summed page, so it cuts
    # the need into more blocks than the worker priced; admission is the claim here.
    assert _vllm_admits(worker, drafting).num_blocks >= (1 + K) * SEQS + 1
