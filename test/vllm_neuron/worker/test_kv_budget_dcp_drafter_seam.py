# SPDX-License-Identifier: Apache-2.0
"""The worker's need-sized KV pool admits vLLM's own requirement under DCP with a drafter.

``test_kv_budget_drafter_seam.py`` pins the seam at decode context parallel size 1: a
speculative server's recurrent (KDA) layers hold ``1 + num_speculative_blocks`` state rows
per request (``MambaSpec.max_memory_usage_bytes``), so the need counts ``k`` extra blocks
per recurrent group per request, or the engine does not start. Under decode context
parallelism the need prices each group per rank (``NeuronWorker._kv_cache_need_bytes``,
the DCP branch) and must count the same draft blocks: a rank keeps every recurrent
layer's state rows, drafts included, at any DCP size.

The served GLM-5.3-Flash form here is the gate's DCP-2 line: 131072 tokens, one request,
``--no-enable-prefix-caching``, the recurrent block at ``max_model_len``. Without MTP its
serve log reads "KV cache need: ... (517 blocks of 131072 B, 516 allocator block(s) per
request, 5 group(s), 11 layer(s) per pool)"; with MTP k = 3 vLLM's admission must accept
the need. Expectations come from vLLM's own grouping and admission functions and the
fixture's layer specs; the 517 is the control serve log's, pinned so the control form's
need does not move.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_kv_budget_dcp_drafter_seam.py
"""
from __future__ import annotations

import pytest
from vllm.utils.math_utils import cdiv

from test.vllm_neuron.worker import test_kv_budget_drafter_seam as seam

K = 3
#: The served DCP-2 line: one request of 131072 tokens.
SEQS, LENGTH = 1, 131072
#: The control serve log's pool (stageB-df1-ctl-dcp2, server.log, the KV cache need line).
CONTROL_BLOCKS, CONTROL_LAYERS_PER_POOL = 517, 11


def _worker(*, dcp: int, drafter: bool, k: int, seqs: int = SEQS, length: int = LENGTH):
    worker = seam._worker(drafter=drafter, k=k, seqs=seqs, length=length)
    worker.vllm_config.parallel_config.decode_context_parallel_size = dcp
    worker.model_runner._dcp_size = dcp
    return worker


def _blocks_per_request(worker, *, dcp: int, k: int, length: int = LENGTH) -> int:
    """The attention group's pages on one rank plus ``1 + k`` blocks per recurrent group:
    the recurrent block is ``max_model_len`` on the served line, one block per group."""
    block_size = worker.vllm_config.cache_config.block_size
    return cdiv(length, block_size * dcp) + seam._recurrent_groups(worker) * (1 + k)


def test_the_dcp2_control_form_keeps_its_517_blocks() -> None:
    """No MTP at DCP 2: the need is the control serve log's pool, 517 blocks of 11 layers,
    and vLLM's admission builds exactly that pool."""
    worker = _worker(dcp=2, drafter=False, k=0)
    groups, layers_per_pool, _ = seam._layout(worker)
    assert (groups, layers_per_pool) == (5, CONTROL_LAYERS_PER_POOL)
    need = worker._kv_cache_need_bytes()
    assert need == CONTROL_BLOCKS * CONTROL_LAYERS_PER_POOL * seam._page_size(worker)
    assert seam._vllm_admits(worker, need).num_blocks == CONTROL_BLOCKS
    assert _blocks_per_request(worker, dcp=2, k=0) * SEQS + 1 == CONTROL_BLOCKS


@pytest.mark.parametrize("dcp", [2, 4])
def test_the_dcp_mtp_form_admits_vllms_own_requirement(dcp: int) -> None:
    """MTP k = 3 under DCP: the need covers vLLM's requirement and vLLM's admission, run on
    it, builds the pool the worker priced -- the rank's attention pages plus ``1 + k``
    blocks per recurrent group per request, plus the null block."""
    worker = _worker(dcp=dcp, drafter=True, k=K)
    need = worker._kv_cache_need_bytes()
    assert need >= seam._vllm_requirement(worker)
    assert seam._vllm_admits(worker, need).num_blocks == (
        _blocks_per_request(worker, dcp=dcp, k=K) * SEQS + 1
    )


def test_the_dcp_need_grows_by_k_blocks_per_recurrent_group_per_request() -> None:
    """From k = 0 to k = K with the drafter's layer at DCP 2 the need grows by exactly K
    blocks per recurrent group per request, each ``page_size x layers_per_pool`` bytes."""
    worker = _worker(dcp=2, drafter=True, k=0, seqs=4)
    plain = worker._kv_cache_need_bytes()
    seam._mtp(worker.model_runner, K)
    _, layers_per_pool, _ = seam._layout(worker)
    assert worker._kv_cache_need_bytes() - plain == (
        K * seam._recurrent_groups(worker) * 4 * seam._page_size(worker) * layers_per_pool
    )


@pytest.mark.parametrize("drafter, k", [(False, 0), (True, K)], ids=["plain", "spec3"])
def test_dcp_1_keeps_its_need(drafter: bool, k: int) -> None:
    """At DCP 1 the need is the whole sequence's pages plus ``1 + k`` blocks per recurrent
    group per request, plus the null block, as before, and vLLM admits it."""
    worker = _worker(dcp=1, drafter=drafter, k=k)
    _, layers_per_pool, _ = seam._layout(worker)
    blocks = _blocks_per_request(worker, dcp=1, k=k) * SEQS + 1
    need = worker._kv_cache_need_bytes()
    assert need == blocks * layers_per_pool * seam._page_size(worker)
    assert seam._vllm_admits(worker, need).num_blocks == blocks
