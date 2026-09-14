"""The scalar operands a traced path builds stay fake on the meta device.

Graph extraction under the CPU-compile arm traces the model on ``meta``. A 0-d tensor built
from a python number by ``torch.as_tensor`` is a REAL meta tensor even inside that trace, and
the next operator refuses a real tensor beside fake ones; a factory op such as ``torch.full``
is dispatched through the fake mode and comes back fake. The items here read torch's own
behaviour on both forms, trace the indexer's prefill seam and its whole prefill leg on meta at
2048 tokens, and require the int route and the tensor route of the seam to seed the same ring.

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 python -m pytest -s -rA \\
        test/vllm_neuron/model/glm5_next/test_traced_scalar_operands.py
"""

from __future__ import annotations

import os

import pytest
import torch
import torch._dynamo as dynamo
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from test.vllm_neuron.model.glm5_next import test_dsa_layer as layer_half

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The device the worker traces on under the CPU-compile arm (``neuron_worker.py:571-572``).
TRACED_DEVICE = "meta"

#: The text the fake mode refuses a real tensor with.
REFUSAL = "convert all Tensors to FakeTensors"

#: The prefill bucket the serving run extracts, and the geometry both traces are read at.
TOKENS = 2048

#: Slots per pool, the ring's length and every address's modulus, read from the fixture.
POOL = layer_half.POOL_SIZE
PAGE_SIZE = layer_half.PAGE_SIZE

#: ``(end_position, start_position)`` pairs item D compares the two routes at: a whole
#: chunk, an end that divides evenly, an end one short, and two padded chunks.
ROUTE_CASES = ((7, None), (8, None), (5, None), (11, 3), (10, 2))
CHUNK_ROWS = 12


def say(name: str, *values) -> None:
    """One reading per line, tagged so a launcher can anchor on it."""
    print("SCALARS|" + name + "|" + "|".join(str(value) for value in values))


def _require_cpu_mode() -> None:
    """The declared acceptance runs under VLLM_NEURON_CPU_MODE=1, so read it, not set it."""
    assert os.environ.get("VLLM_NEURON_CPU_MODE") == "1", (
        "the declared acceptance runs under VLLM_NEURON_CPU_MODE=1 and this process "
        "does not carry it, so nothing below would be measuring the declared mode"
    )


def _shape_only_indexer(device: str):
    """An indexer with no weight but its per-slot bias, the two seams' only module read."""
    indexer = layer_half._bare_indexer()
    indexer.index_kpool_compress_ape = torch.nn.Parameter(
        torch.zeros(POOL, int(indexer.index_head_dim), dtype=torch.bfloat16, device=device),
        requires_grad=False,
    )
    return indexer


def _chunk(indexer, *, tokens: int, device: str) -> dict:
    """One ring, one chunk of keys and one of gate scores."""
    head_dim = int(indexer.index_head_dim)
    return {
        "tail": torch.zeros(2, POOL, head_dim, dtype=torch.bfloat16, device=device),
        "key": torch.zeros(tokens, head_dim, dtype=torch.bfloat16, device=device),
        "gate": torch.zeros(tokens, head_dim, dtype=torch.bfloat16, device=device),
    }


class _Keep:
    """A compile backend that keeps every graph and runs none."""

    def __init__(self) -> None:
        self.graphs: list = []

    def __call__(self, graph_module, _example_inputs):
        self.graphs.append(graph_module)
        return graph_module.forward

    def nodes_targeting(self, *targets) -> int:
        return sum(
            1
            for graph_module in self.graphs
            for node in graph_module.graph.nodes
            if node.op == "call_function" and node.target in targets
        )


def _traced(fn, *args, **kwargs) -> tuple[_Keep, str]:
    """``fn`` through the runner's own compile flags; the refusal's message, or empty."""
    keep = _Keep()
    dynamo.reset()
    compiled = torch.compile(fn, backend=keep, fullgraph=True, dynamic=False)
    try:
        compiled(*args, **kwargs)
    except Exception as error:  # the trace's refusal is the reading, not a failure here
        return keep, " ".join(str(error).split())
    return keep, ""


def _refusal_reads(message: str) -> str:
    """The head of a refusal message, for a row and an assertion."""
    return message[:400] if message else "none"


def test_a_a_number_built_by_as_tensor_on_meta_is_real_under_the_fake_mode() -> None:
    """Torch's own behaviour on the two forms: the data-built one is refused, the factory one taken."""
    with FakeTensorMode():
        mask = torch.zeros(TOKENS, dtype=torch.bool, device=TRACED_DEVICE)
        index = torch.zeros(TOKENS, dtype=torch.int64, device=TRACED_DEVICE)
        as_tensor_built = torch.as_tensor(7, dtype=torch.int64, device=TRACED_DEVICE)
        full_built = torch.full((), 7, dtype=torch.int64, device=TRACED_DEVICE)
        through = torch.as_tensor(index[0], dtype=torch.int64, device=TRACED_DEVICE)

        refusals = {}
        for name, scalar in (("as_tensor", as_tensor_built), ("full", full_built)):
            try:
                torch.where(mask, index, scalar)
            except Exception as error:  # the fake mode's refusal is the reading
                refusals[name] = " ".join(str(error).split())
            else:
                refusals[name] = ""

    say("as_tensor_on_meta", f"is_fake={isinstance(as_tensor_built, FakeTensor)}",
        f"refused={bool(refusals['as_tensor'])}",
        f"carries_the_text={REFUSAL in refusals['as_tensor']}")
    say("full_on_meta", f"is_fake={isinstance(full_built, FakeTensor)}",
        f"refused={bool(refusals['full'])}")
    say("as_tensor_of_a_tensor", f"is_fake={isinstance(through, FakeTensor)}")

    assert not isinstance(as_tensor_built, FakeTensor), (
        "torch.as_tensor(<number>, device=meta) came back fake inside the fake mode, so this "
        "torch no longer has the meta shortcut the repair is for; re-read the pin"
    )
    assert refusals["as_tensor"] and REFUSAL in refusals["as_tensor"], (
        f"the data-built form was expected to be refused with {REFUSAL!r}; "
        f"read {_refusal_reads(refusals['as_tensor'])!r}"
    )
    assert isinstance(full_built, FakeTensor) and not refusals["full"], (
        f"the factory-built form was refused: {_refusal_reads(refusals['full'])!r}"
    )
    assert isinstance(through, FakeTensor), "as_tensor of a fake tensor did not stay fake"


def test_b_the_prefill_seam_traces_on_meta_with_the_chunk_end_as_a_tensor() -> None:
    """``seed_tail`` on meta at 2048 tokens, the end as a tensor, then the start as one too."""
    _require_cpu_mode()
    indexer = _shape_only_indexer(TRACED_DEVICE)
    end = torch.tensor(TOKENS, dtype=torch.int32, device=TRACED_DEVICE)
    start = torch.tensor(0, dtype=torch.int32, device=TRACED_DEVICE)
    refused = {}
    for form, positions in (("end", (end,)), ("end_and_start", (end, start))):
        chunk = _chunk(indexer, tokens=TOKENS, device=TRACED_DEVICE)
        keep, message = _traced(
            indexer.seed_tail, chunk["tail"], chunk["key"], chunk["gate"], *positions
        )
        say("seed_tail_trace", f"form={form}", f"refused={bool(message)}",
            f"graphs={len(keep.graphs)}",
            f"as_tensor_nodes={keep.nodes_targeting(torch.as_tensor)}",
            f"full_nodes={keep.nodes_targeting(torch.full)}",
            f"message={_refusal_reads(message)}")
        if message:
            refused[form] = message
    assert not refused, (
        f"the trace of seed_tail on meta was refused: {_refusal_reads(next(iter(refused.values())))!r}"
    )


def test_c_the_indexer_prefill_leg_traces_on_meta_at_2048_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole prefill leg through the indexer's forward, on meta, the kernel gate off.

    The simulator computes values and a meta trace has none, so the gate is turned off the
    way the runner's own switch does it; the seams' torch forms are what the trace reads.
    """
    _require_cpu_mode()
    monkeypatch.setenv("VLLM_NEURON_DISABLE_NKI_KERNELS", "1")
    stack, cfg, _gen = layer_half.build_layer_stack(layers=1)
    indexer = stack[0].attention.indexer.to(TRACED_DEVICE)
    head_dim = int(indexer.index_head_dim)
    max_seq_len = TOKENS + 1
    candidates = max_seq_len // POOL
    rows = ((candidates + 1 + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
    assert candidates > int(indexer.select_k()), (
        f"{candidates} candidate pool(s) is not more than the {int(indexer.select_k())} "
        f"selected, so the leg would take its causal bypass before the seam"
    )

    hidden = torch.zeros(TOKENS, int(cfg.hidden_size), dtype=torch.float32, device=TRACED_DEVICE)
    q_latent = torch.zeros(TOKENS, int(cfg.q_lora_rank), dtype=torch.float32, device=TRACED_DEVICE)
    pool_cache = torch.zeros(rows, head_dim, dtype=torch.bfloat16, device=TRACED_DEVICE)
    seq_lens = torch.arange(1, TOKENS + 1, dtype=torch.int32, device=TRACED_DEVICE)
    slot_mapping = layer_half.prefill_slot_mapping(TOKENS, POOL).to(TRACED_DEVICE)
    ring = torch.zeros(2, POOL, head_dim, dtype=torch.bfloat16, device=TRACED_DEVICE)
    end = torch.tensor(TOKENS, dtype=torch.int32, device=TRACED_DEVICE)

    keep, message = _traced(
        indexer,
        hidden,
        q_latent,
        pool_cache,
        seq_lens,
        max_seq_len=max_seq_len,
        page_size=PAGE_SIZE,
        slot_mapping=slot_mapping,
        prefill_tail=ring,
        prefill_end_position=end,
    )
    say("indexer_prefill_trace", f"tokens={TOKENS}", f"pool_rows={rows}",
        f"refused={bool(message)}", f"graphs={len(keep.graphs)}",
        f"as_tensor_nodes={keep.nodes_targeting(torch.as_tensor)}",
        f"full_nodes={keep.nodes_targeting(torch.full)}",
        f"names_the_seam={'_seed_tail_at' in message}",
        f"message={_refusal_reads(message)}")
    assert not message, (
        f"the trace of the indexer's prefill leg on meta was refused: {_refusal_reads(message)!r}"
    )
    assert len(keep.graphs) == 1, f"{len(keep.graphs)} graphs kept, not one"
    assert keep.nodes_targeting(torch.as_tensor) == 0, "the graph carries a torch.as_tensor node"


def test_d_the_int_route_and_the_tensor_route_seed_the_same_ring_for_padded_and_whole_chunks() -> None:
    """CPU values: by int and by tensor, equal ring bytes and equal counts at every case."""
    _require_cpu_mode()
    indexer = _shape_only_indexer("cpu")
    head_dim = int(indexer.index_head_dim)
    gen = torch.Generator().manual_seed(700_104)
    key = torch.randn(CHUNK_ROWS, head_dim, generator=gen).to(torch.bfloat16)
    gate = torch.randn(CHUNK_ROWS, head_dim, generator=gen).to(torch.bfloat16)
    disagreed = []
    for end, start in ROUTE_CASES:
        ring_int = torch.full((2, POOL, head_dim), -1.0, dtype=torch.bfloat16)
        ring_tensor = ring_int.clone()
        by_int = indexer.seed_tail(ring_int, key, gate, end, start)
        by_tensor = indexer.seed_tail(
            ring_tensor, key, gate,
            torch.tensor(end, dtype=torch.int32),
            None if start is None else torch.tensor(start, dtype=torch.int32),
        )
        same_ring = torch.equal(ring_tensor, ring_int)
        same_count = int(by_tensor) == int(by_int)
        say("routes_agree", f"end={end}", f"start={start}",
            f"real={CHUNK_ROWS if start is None else end - start}",
            f"remainder={end % POOL}", f"int_rows={int(by_int)}", f"tensor_rows={int(by_tensor)}",
            f"ring_differing={int((ring_tensor != ring_int).sum())}", f"same_count={same_count}")
        if not (same_ring and same_count):
            disagreed.append((end, start))
    assert not disagreed, f"the two routes seeded different rings or counts at {disagreed}"
