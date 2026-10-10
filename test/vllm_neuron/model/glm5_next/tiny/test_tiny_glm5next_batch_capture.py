# SPDX-License-Identifier: Apache-2.0
"""Decode graphs extract for every batch bucket up to 64 on the tiny root.

The worker warms and captures one decode graph per batch bucket, ``[1, 2, 4, ..., 64]``
for ``max_num_seqs=64``. Wave 1 served one request per decode step, and the batch-2
capture failed in the runner's token/request accounting: a synthetic step named one
request for a batch of two token rows. Each bucket is captured here through
``extract_decode_graphs`` with the stand-in capture backend, which calls the real root.
What is read for every bucket:

1. the root was entered once, with one token row and one logit row per sequence;
2. every sparse layer was handed one block-table column, one latent slot, one position
   and one ring/pooled-store view per sequence, and the views are disjoint (the graph's
   inputs must not alias, or a served batch, whose views are disjoint, would not match);
3. the decode attention took one launch per layer: the batched decode kernel at B > 1,
   the one-request sparse kernel at B = 1, the unchanged one-request path;
4. a synthetic step takes no slot claim and moves no cursor.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_batch_capture.py
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.attention import mla_decode as decode_seam
from vllm_neuron.functional.attention import mla_sparse as sparse_seam
from vllm_neuron.utils.bucket_utils import (
    get_default_num_seqs_buckets,
    validate_num_seqs_buckets,
)

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_capture_sites as sites
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The served concurrency bound and the buckets the worker compiles for it.
MAX_NUM_SEQS = 64
BUCKETS = [1, 2, 4, 8, 16, 32, 64]


def _capture_runner():
    root, caches = sites._bound_root()
    runner = sites._runner(root)
    runner.max_num_reqs = MAX_NUM_SEQS
    backend = sites._StandInBackend(runner)
    runner.capture_backend_model = backend
    return root, runner, backend


def test_the_bucket_list_is_the_one_the_validator_admits_for_sixty_four():
    assert get_default_num_seqs_buckets(MAX_NUM_SEQS) == BUCKETS
    assert validate_num_seqs_buckets(list(BUCKETS), MAX_NUM_SEQS) == BUCKETS


def test_decode_graphs_extract_for_every_batch_bucket_up_to_sixty_four(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_STATE_BANKS", "0")  # this test pins the per-request VIEW form
    root, runner, backend = _capture_runner()
    sparse_layers = [
        index
        for index, bank in enumerate(root.glm5next_layer_banks)
        if bank["family"] == "self_attn"
    ]
    assert sparse_layers, "the tiny root carries no sparse layer to read"
    for batch in BUCKETS:
        backend.seen.clear()
        decode_seam.reset_mla_decode_dispatch_counters()
        sparse_seam.reset_mla_sparse_dispatch_counters()

        runner.extract_decode_graphs(batch)

        kwargs = sites._assert_translated(f"decode B={batch}", backend.seen)
        assert int(kwargs["input_ids"].shape[0]) == batch
        sites._assert_finite_logits(f"decode B={batch}", backend.output, rows=batch)
        for index in sparse_layers:
            carrier = kwargs["layer_carriers"][index]
            assert "tail" in carrier and "slot_mapping" not in carrier, sorted(carrier)
            assert int(carrier["block_table_row"].shape[1]) == batch
            assert tuple(carrier["latent_slots"].shape) == (batch,)
            if batch == 1:
                # Wave 1's one-request carrier, unchanged.
                assert torch.is_tensor(carrier["tail"])
                assert carrier["position"].dim() == 0
                continue
            assert tuple(carrier["position"].shape) == (batch,)
            assert tuple(carrier["seq_lens"].shape) == (batch,)
            for key in ("tail", "pool_cache"):
                views = carrier[key]
                assert isinstance(views, tuple) and len(views) == batch, (key, type(views))
                starts = sorted(view.data_ptr() for view in views)
                assert len(set(starts)) == batch, f"B={batch}: two {key} views alias"
                span = views[0].numel() * views[0].element_size()
                assert all(b - a >= span for a, b in zip(starts, starts[1:])), (
                    f"B={batch}: {key} views overlap"
                )
        layers = len(sparse_layers)
        dense, selected, _ = decode_seam.mla_decode_route_counts()
        if batch == 1:
            assert sparse_seam.mla_sparse_dispatch_counters() == (layers, 0)
            assert dense + selected == 0
        else:
            # One batched launch per layer, never one per request.
            assert dense + selected == layers, (batch, dense, selected)
            assert sparse_seam.mla_sparse_dispatch_counters() == (0, 0)
        # Warmup is not a sequence step.
        assert not getattr(runner, "_glm5next_request_slot_table", {})
        assert not getattr(runner, "_glm5next_side_cache_positions", {})


def test_a_synthetic_decode_takes_distinct_slots_and_no_claim(monkeypatch):
    """The warmup's B rows are served from B distinct slots, none claimed."""
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_STATE_BANKS", "0")  # this test pins the per-request VIEW form
    _, runner, _ = _capture_runner()
    kwargs = runner._build_decode_synthetic_inputs(4, compiled_graph_input=True)
    converted = runner._glm5next_model_kwargs(kwargs)
    read = 0
    for index, carrier in enumerate(converted["layer_carriers"]):
        if "tail" not in carrier:
            continue
        ring = runner._glm5next_side_cache_set[index]["tail"]
        stride = ring[0].numel() * ring.element_size()
        slots = [(view.data_ptr() - ring.data_ptr()) // stride for view in carrier["tail"]]
        assert slots == [0, 1, 2, 3], slots
        read += 1
    assert read == tiny.STACK_LAYERS
    assert runner._glm5next_request_slot_table == {}
    assert runner._glm5next_side_cache_positions == {}
