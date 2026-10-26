# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 taps on the real decoder layers, in the forward oracle's world.

The world is ``test_forward.py``'s: its 4-layer model, random weights, caches and inputs on
8 gloo ranks, TP4 by DP2, with its float64 reference, read from that file so the two cannot
drift. The forward taps the streams that layers 1, 2 and 3 read.

At the oracle's decode step each tap must be, bit for bit, the stream the oracle grades as the
previous layer's output. At a verify window of 5 tokens per sequence, which steps the KDA
layers through their verify mode, each tap must sit inside the oracle's end-to-end bound for
the previous layer on every TP rank, read per window position on its 8 rows against the
reference stepped token by token. The oracle's share item must pass at those rows first, and
its routing rule and caps hold per window position; a share outside its limits or a passed cap
fails with a reason that starts ``not adjudicable:``. The MLA layer's attention is a stand-in
that returns zeros: the taps are read before it.

Run with ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1`` and ``KIMI_K3_CHECKPOINT`` set, as the
oracle runs. Each rank uses 2 threads.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_neuron.model.kimi_k3 import model as m

_spec = importlib.util.spec_from_file_location(
    "forward_oracle", Path(__file__).with_name("test_forward.py")
)
oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oracle)

pytestmark = oracle.pytestmark

TP, DP, H, K, T, LAYERS = oracle.TP, oracle.DP, oracle.H, oracle.K, oracle.T, oracle.LAYERS
AUX, WINDOW = (1, 2, 3), 5
V = T * WINDOW  # verify tokens per replica, sequence-major


def _verify_caches(replica: int, tp: int) -> dict:
    """The oracle's caches, with 4 more KDA pages per sequence for the window's row pages."""
    caches, conv = oracle._caches(replica, tp), oracle.CONV[0] * oracle.CONV[1] * 2
    state = conv + oracle.STATE[0] * oracle.STATE[1] * oracle.STATE[2] * 4
    for i in range(3):
        name = f"layers.{i}.self_attn"
        old = caches[name][2]
        pages = torch.zeros(1 + T * WINDOW, old.shape[1], dtype=torch.int8)
        pages[: old.shape[0]] = old
        caches[name] = [pages[:, :conv].view(torch.bfloat16).view(-1, *oracle.CONV),
                        pages[:, conv:state].view(torch.float32).view(-1, *oracle.STATE), pages]
    return caches


def _verify_inputs(replica: int) -> tuple:
    """A replica's window token ids, their positions and per-layer verify metadata.

    Sequence n resumes from the oracle's state page n + 1, which is also its first row page.
    """
    gen, width = torch.Generator().manual_seed(100 * replica + 1), max(map(len, oracle.BLOCKS))
    rows = [[n + 1] + [T + 1 + (WINDOW - 1) * n + q for q in range(WINDOW - 1)] for n in range(T)]
    positions = torch.tensor([start + j for start in oracle.PREFIX for j in range(WINDOW)])
    slots = torch.tensor([own[p // oracle.BLOCK] * oracle.BLOCK + p % oracle.BLOCK
                          for own, start in zip(oracle.BLOCKS, oracle.PREFIX)
                          for p in range(start, start + WINDOW)])
    mla = torch.tensor([own + [-1] * (width - len(own)) for own in oracle.BLOCKS],
                       dtype=torch.int32)
    meta = {}
    for i in range(LAYERS):
        meta[f"layers.{i}.self_attn"] = {
            "block_table_tensor": torch.tensor(rows, dtype=torch.int32) if i < 3 else mla,
            "slot_mapping": slots, "max_query_len": WINDOW, "decode_token_threshold": WINDOW,
            **({"state_resume_rows": torch.zeros(T, dtype=torch.int32)} if i < 3 else {}),
        }
    return torch.randint(0, oracle.VOCAB, (V,), generator=gen), positions, meta


def _rank(rank: int, root: Path) -> None:
    """One gloo rank: the decode step's taps and mixes, then a verify window's taps and experts."""
    from vllm_neuron.parallel.neuron_parallel_state import (
        initialize_neuron_parallel_state, stamp_spmd_rank_tensors)

    torch.set_num_threads(2)
    replica, tp = divmod(rank, TP)
    dist.init_process_group("gloo", init_method=f"file://{root}/rdzv", rank=rank,
                            world_size=TP * DP)
    initialize_neuron_parallel_state(ep_degree=TP * DP, dp_size=DP, local_rank=rank,
                                     tp_global_ranks=[replica * TP + i for i in range(TP)])
    stamped = stamp_spmd_rank_tensors("cpu")
    model = oracle._model()
    oracle._load(model, torch.load(root / "weights.pt", mmap=True),
                 torch.load(root / "experts.pt"), tp)
    model.set_aux_hidden_state_layers(AUX)
    tp_rank = torch.tensor(tp, dtype=torch.int32)
    mix, select, verify = m.attn_res_decode, m.routed_experts_mx, m.kda_fused_verify
    calls, experts, windows = [], [], []

    def mixed(*args, **kwargs):
        a = oracle._arguments(mix, args, kwargs)
        out = mix(*args, **kwargs)
        calls.append({"bank": a["bank"][:, :a["num_sources"] - 1].clone(),
                      "new_stream": out.new_stream.clone()})
        return out

    def expert_call(*args, **kwargs):
        a = oracle._arguments(select, args, kwargs)
        experts.append({"routed": a["routed"].clone(), "bias": a["score_bias"].clone()})
        return select(*args, **kwargs)

    def verify_call(x, source, rows, *args, **kwargs):
        windows.append(rows.shape[1])
        return verify(x, source, rows, *args, **kwargs)

    mla = model.model.layers[3].self_attn

    def no_attention(latent, gate, meta):
        width = mla.output_weight.shape[0]
        return m._decode_layout(torch.zeros(latent.shape[0], width, dtype=torch.bfloat16))

    record = {"kernels": m.can_run_kernel()}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(m, "attn_res_decode", mixed)
        model.bind_kv_cache(oracle._caches(replica, tp))
        ids, positions, meta = oracle._inputs(replica)
        _, taps = model(input_ids=ids, positions=positions, attn_metadata=meta, rank=tp_rank)
        record["decode"] = {"taps": taps.clone(), "calls": calls}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(m, "routed_experts_mx", expert_call)
        patch.setattr(m, "kda_fused_verify", verify_call)
        patch.setattr(mla, "_attend", no_attention)
        model.bind_kv_cache(_verify_caches(replica, tp))
        ids, positions, meta = _verify_inputs(replica)
        _, taps = model(input_ids=ids, positions=positions, attn_metadata=meta, rank=tp_rank)
        record["verify"] = {"taps": taps.clone(), "experts": experts, "windows": windows}
    torch.save(record, root / f"rank{rank}.pt")
    del stamped
    dist.destroy_process_group()


def _window_reference(layers, embedding) -> list[list[list[dict]]]:
    """Per window position and layer, the 8 rows' reference taps, stepped token by token."""
    ends = [[[] for _ in range(LAYERS)] for _ in range(WINDOW)]
    for d in range(DP):
        ids = _verify_inputs(d)[0]
        for n, past in enumerate(oracle._pasts(layers, d)):
            for p in range(WINDOW):
                stream = embedding[ids[n * WINDOW + p]].double()
                bank = stream.new_zeros(1, 0, H)
                for i, layer in enumerate(layers):
                    tap = oracle._layer_run(layer, stream, bank, past)
                    stream, bank = tap["out"], tap["bank"]
                    ends[p][i].append(tap)
    return ends


def _chosen(verifies: list[dict], call: int, p: int) -> torch.Tensor:
    """Window position ``p``'s 8 rows' experts at the ``call``-th expert call, replica-major."""
    rows = []
    for d, record in enumerate(verifies):
        routed, bias = record["experts"][call]["routed"], record["experts"][call]["bias"]
        assert routed.shape[0] == DP * V, f"expert call holds {routed.shape[0]} rows"
        picked = m.router_select(routed[:, K:], bias, scaling=1.0)[0]
        rows.append(picked[d * V:(d + 1) * V][p::WINDOW])
    return torch.cat(rows).long().sort(-1).values


@pytest.fixture(scope="module")
def world(tmp_path_factory) -> SimpleNamespace:
    """Every rank's records of the decode step and the verify window, and the test's weights."""
    root = tmp_path_factory.mktemp("taps")
    weights, experts = oracle._weights(oracle._model()), oracle._experts()
    torch.save(weights, root / "weights.pt")
    torch.save(experts, root / "experts.pt")
    mp.spawn(_rank, args=(root,), nprocs=TP * DP, join=True)
    records = [torch.load(root / f"rank{r}.pt") for r in range(TP * DP)]
    return SimpleNamespace(weights=weights, experts=experts, records=records)


def test_decode_taps_equal_the_streams_the_oracle_grades(world):
    """At the oracle's decode step each tap is the stream graded as the layer before's output."""
    differing = {}
    for rank, record in enumerate(world.records):
        taps, calls = record["decode"]["taps"], record["decode"]["calls"]
        assert len(calls) == 2 * LAYERS and taps.shape == (T, len(AUX) * H)
        for slot, j in enumerate(AUX):
            graded = oracle._port_layer(calls, j - 1)[2]
            tap = taps[:, slot * H:(slot + 1) * H].contiguous()
            assert tap.dtype == graded.dtype and tap.shape == graded.shape
            count = int((tap.view(torch.int16) != graded.view(torch.int16)).sum())
            print(f"rank {rank} tap of layer {j}: {count} elements differ from layer {j - 1}'s "
                  f"graded output of {graded.numel()}")
            if count:
                differing[(rank, j)] = count
    assert not differing, f"taps differ from the graded streams: {differing}"


def test_verify_window_taps_inside_the_end_to_end_bounds(world):
    """At a 5-token verify window each tap is inside the oracle's end-to-end bound, per position."""
    for rank, record in enumerate(world.records):
        assert record["kernels"], f"rank {rank} ran without the kernels"
        assert record["verify"]["windows"] == [WINDOW] * 3, f"rank {rank}: {record['verify']}"
    layers, _ = oracle._reference_layers(world.weights, world.experts)
    with torch.no_grad():
        ends = _window_reference(layers, world.weights["model.embed_tokens.weight"])
    outside = []
    for p in range(WINDOW):
        print(f"window position {p}:")
        try:
            oracle.test_reference_shares_inside_the_model(SimpleNamespace(ends=ends[p]))
        except pytest.fail.Exception as error:
            outside.append(f"window position {p}: {error}")
    if outside:
        pytest.fail(f"not adjudicable: the share item fails: {'; '.join(outside)}")
    fails, withdrawn = [], []
    for tp in range(TP):
        verifies = [world.records[d * TP + tp]["verify"] for d in range(DP)]
        for p in range(WINDOW):
            routing = SimpleNamespace(end={
                j: oracle._routing(_chosen(verifies, n, p), ends[p][j], world.weights, j,
                                   oracle.TIE_END_TO_END[j])
                for n, j in enumerate(oracle.MOE_LAYERS)})
            print(f"TP rank {tp} window position {p} routing: "
                  + ", ".join(f"layer {j} {states.count('moved')} moved, {states.count('fail')} "
                              f"differ past the tie gap"
                              for j, (states, _, _) in routing.end.items()))
            for slot, j in enumerate(AUX):
                got = torch.cat([v["taps"][p::WINDOW, slot * H:(slot + 1) * H] for v in verifies])
                ref = torch.stack([tap["out"] for tap in ends[p][j - 1]])
                f, w = oracle._grade(
                    f"window position {p} tap of layer {j} TP rank {tp}", got, ref,
                    oracle._end_rows(routing, j - 1), oracle.END_TO_END[j - 1],
                    oracle._end_cap(routing, j - 1))
                fails, withdrawn = fails + f, withdrawn + w
    oracle._decide("verify window taps", fails, withdrawn)
