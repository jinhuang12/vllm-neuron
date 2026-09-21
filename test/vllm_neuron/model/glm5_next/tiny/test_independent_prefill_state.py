# SPDX-License-Identifier: Apache-2.0
"""Compare MLA prefixes on one hybrid tiny model and its owned cache state."""

from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.model.glm5_next.config import KDA_LAYER_TYPE
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next import test_kda_layer as kda
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_request_tokens as padded

pytestmark = [pytest.mark.fast, pytest.mark.forked]

CONTEXT = 4096
SEGMENT = 1024
REQUEST = "independent-prefill"


def _hybrid_root():
    """Reuse the tiny root's weights; replace its middle attention with real KDA."""
    root = e2e._fixture()["root"]
    cfg = root.text_config
    cfg.layer_types[1] = KDA_LAYER_TYPE
    cfg.linear_attn_config = dict(cfg.linear_attn_config, num_heads=1)
    old = root.model.layers[1]
    layer = kda._impl().Glm5NextKDALayer(cfg, 1, world_size=1)
    layer.mlp = old.mlp
    layer.post_attention_layernorm_weight = old.post_attention_layernorm_weight
    for name in tiny._stack_leaf_shapes(cfg):
        setattr(layer, name, getattr(old, name))
    layer.bind_hyper_connection_sites(cfg, torch.device("cpu"))
    with torch.random.fork_rng():
        torch.manual_seed(kda.SEED)
        weights = kda._make_weights(
            int(cfg.hidden_size), 1, kda.KDA_HEAD_SIZE,
            kda.KDA_CONV_KERNEL_SIZE,
        )
    for name, value in weights.items():
        target = layer if name == "input_layernorm_weight" else layer.attention
        setattr(target, name, torch.nn.Parameter(value, requires_grad=False))
    root.model.layers[1] = layer
    return root


def _runner(root, *, caller_prefix=False):
    caches = e2e._runner_shaped_caches(root)
    for spec in root.get_kv_spec().layers:
        if spec.kda_recurrent_state_shape is None:
            caches[spec.name] = [
                torch.zeros((CONTEXT // tiny.MLA_PAGE_SIZE, *bank.shape[1:]), dtype=bank.dtype)
                for bank in caches[spec.name]
            ]
    root.bind_kv_cache(caches)
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = root
    runner.max_model_len = CONTEXT
    runner.max_num_reqs = 2  # Leave a second slot to detect writes outside this request.
    runner.input_batch = SimpleNamespace(req_ids=[REQUEST])
    if caller_prefix:
        runner.neuron_config = SimpleNamespace(num_batched_tokens_buckets=[128, 1024])
    return runner


def _step(runner, ids, *, width, cached, sampling_rows):
    real = len(ids)
    pages = e2e._blocks_for(cached + real)
    row = list(range(pages)) + [0] * (CONTEXT // tiny.MLA_PAGE_SIZE - pages)
    entry = e2e._entry(
        row=row, tokens=width, cached=cached, threshold=1,
        block_size=tiny.MLA_PAGE_SIZE,
    )
    entry["kv_segment_size"] = SEGMENT
    runner._glm5next_request_tokens = NeuronModelRunner._glm5next_request_token_counts(
        [REQUEST], {REQUEST: real}
    )
    kwargs = runner._glm5next_model_kwargs({
        "input_ids": torch.cat([ids, torch.zeros(width - real, dtype=ids.dtype)]),
        "positions": torch.arange(width, dtype=torch.long) + cached,
        "attn_metadata": {bank["name"]: entry for bank in runner.model.glm5next_layer_banks},
        "sampling_positions": torch.tensor(sampling_rows, dtype=torch.long),
        "sampling_params": None, "spec_decode_metadata": None,
        "rank": None, "logit_mask": None,
    })
    configured = hasattr(runner, "neuron_config")
    operator_rows = 1024 if configured and width > 1 else width
    assert kwargs["input_ids"].shape == (operator_rows,)
    assert entry["max_query_len"] == width, "translation changed caller metadata"
    for carrier in kwargs["layer_carriers"]:
        if "latent_cache" in carrier:
            prefix = 128 if configured and width == 128 else None
            assert carrier.get("active_mla_query_rows") == prefix
            # The carrier holds the whole bank, which this fixture allocates as
            # CONTEXT // page blocks of page rows, so its row count is CONTEXT on both
            # legs. The window the leg attends is the block table's width times the page,
            # which is where a captured graph's shape comes from.
            assert carrier["latent_cache"].shape[0] == CONTEXT
            window_rows = SEGMENT + operator_rows if width > 1 else CONTEXT
            assert (
                carrier["block_table_row"].shape[0] * int(carrier["page_size"])
                == window_rows
            )
            assert carrier["latent_slots"].shape[0] == operator_rows
            assert carrier["seq_lens"].shape[0] == operator_rows
        else:
            assert "active_mla_query_rows" not in carrier
            if width > 1:
                assert carrier["real_tokens"].tolist() == [[real]]
                assert carrier["row_mask"].shape == (1, operator_rows, 1)
    logits = runner.model(**kwargs).detach().float()
    assert torch.isfinite(logits).all()
    assert e2e._slot_position(runner, e2e._own_slot(runner, REQUEST)) == cached + real
    return logits


def _snapshot(runner, end):
    """Read this request's own state and reject writes outside its slot."""
    own = e2e._own_slot(runner, REQUEST)
    snapshots = {}
    banks = runner.model.glm5next_layer_banks
    side = runner._glm5next_live_side_caches(banks)
    for index, bank in enumerate(banks):
        if bank["family"] == "linear_attn":
            for name in ("conv_state", "recurrent_state"):
                value = bank[name]
                assert torch.count_nonzero(value[:own]) == 0
                assert torch.count_nonzero(value[own + 1:]) == 0
                snapshots[f"kda.{index}.{name}"] = value[own].clone()
                assert torch.count_nonzero(value[own]) > 0
        else:
            # The bank is read by physical row, and this request holds the row's first
            # pages, so its own slots are 0..end and a row past them carries no token of it.
            latent = bank["latent_cache"]
            assert torch.count_nonzero(latent[end:]) == 0, "padding wrote latent slots"
            snapshots[f"dsa.{index}.latent"] = latent[:end].clone()
            assert torch.count_nonzero(latent[:end]) > 0
            for name in ("pool_cache", "tail"):
                value = side[index][name]
                assert torch.count_nonzero(value[:own]) == 0
                assert torch.count_nonzero(value[own + 1:]) == 0
                owned = value[own]
                if name == "pool_cache":
                    pools = end // int(runner.model.text_config.index_kpool)
                    assert torch.count_nonzero(owned[pools:-1]) == 0, "padding wrote pools"
                    owned = owned[:pools]
                snapshots[f"dsa.{index}.{name}"] = owned.clone()
    assert all(torch.isfinite(value).all() for value in snapshots.values())
    return snapshots


def _compare(label, small, large):
    assert small.keys() == large.keys()
    for name, got in small.items():
        expected = large[name]
        gap = float((got.float() - expected.float()).abs().max())
        if name.startswith("kda."):
            torch.testing.assert_close(
                got, expected, rtol=kda.DECLARED_RTOL, atol=kda.DECLARED_ATOL,
                msg=f"{label}: {name}",
            )
        else:
            ratio, _, bound = padded._worst_offender(got, expected)
            assert ratio <= 1.0, f"{label}: {name} gap {gap} exceeds existing bound {bound}"


@pytest.mark.parametrize("real_tokens", [5, 127, 128])
@pytest.mark.parametrize("caller_prefix", [False, True], ids=["whole-query", "mla-prefix"])
def test_query_bucket_preserves_valid_logits_and_owned_state(
    real_tokens, caller_prefix, monkeypatch,
):
    e2e._require_cpu_mode()
    from vllm_neuron.functional.attention import mla_sparse

    sparse_calls = []
    original_sparse = mla_sparse.mla_sparse_attention

    def record_sparse(query, cache, indices, scale, *args, **kwargs):
        # The operand is the whole bank and the window the seam attends is the one its
        # block table names, so both are recorded: the bank's rows, the same at every
        # position, and the table's width times the page, which is the leg's bucket.
        window = int(kwargs["block_table_row"].shape[0]) * int(kwargs["page_size"])
        sparse_calls.append((query.shape[0], cache.shape[0], indices.shape[0], window))
        return original_sparse(query, cache, indices, scale, *args, **kwargs)

    monkeypatch.setattr(mla_sparse, "mla_sparse_attention", record_sparse)
    root = _hybrid_root()
    ids = torch.randint(
        0, tiny.STACK_VOCAB_SIZE, (real_tokens + 1,),
        generator=torch.Generator().manual_seed(tiny.SEED_STACK_IDS),
        dtype=torch.int64,
    )
    sampling_rows = [0, real_tokens // 2, real_tokens - 1]
    readings = {}
    for width in (128, 1024):
        runner = _runner(root, caller_prefix=caller_prefix)
        sparse_calls.clear()
        prefill = _step(runner, ids[:-1], width=width, cached=0, sampling_rows=sampling_rows)
        # Recorded per call: the query rows, the operand's row count, the index rows and the
        # window the block table names. The prefill's window is the segment plus the query
        # rows; the decode's is the whole table.
        window_rows = SEGMENT + (1024 if caller_prefix else width)
        assert sparse_calls == [(width, CONTEXT, width, window_rows)] * 2
        before_decode = _snapshot(runner, real_tokens)
        sparse_calls.clear()
        decode = _step(runner, ids[-1:], width=1, cached=real_tokens, sampling_rows=[0])
        assert sparse_calls == [(1, CONTEXT, 1, CONTEXT)] * 2
        readings[width] = (prefill, before_decode, decode, _snapshot(runner, real_tokens + 1))
    small, large = readings[128], readings[1024]
    for index in (0, 2):  # the prefill's logits, then the decode's
        # Same bound the tiny generation compares its reference logits at.
        torch.testing.assert_close(small[index], large[index], rtol=1e-2, atol=1e-5)
    _compare(f"real={real_tokens}/prefill", small[1], large[1])
    _compare(f"real={real_tokens}/decode", small[3], large[3])
