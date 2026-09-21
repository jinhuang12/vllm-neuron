"""The scalar operands a traced path builds stay fake on the meta device."""

from __future__ import annotations

import os

import pytest
import torch
import torch._dynamo as dynamo
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from vllm_neuron.functional.dsa.kpool_hadamard import dsa_hadamard128, dsa_kpool_hadamard
from vllm_neuron.utils.neuron_utils import can_run_kernel

from test.vllm_neuron.model.glm5_next import test_dsa_layer as layer_half

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The device the worker traces on under the CPU-compile arm (``neuron_worker.py``).
TRACED_DEVICE = "meta"

#: The text the fake mode refuses a real tensor with.
REFUSAL = "convert all Tensors to FakeTensors"


#: The prefill bucket the serving run extracts, and the geometry both traces are read at.
TOKENS = 2048

#: Slots per pool, the ring's length and every address's modulus, read from the fixture.
POOL = layer_half.POOL_SIZE
PAGE_SIZE = layer_half.PAGE_SIZE

#: ``(end_position, start_position)`` pairs test D compares the two routes at. A whole chunk's
#: end is at least its own length (the int route's own rule): one past a pool boundary, one
#: short of the next, exactly on one; then two padded chunks, whose real length is ``end - start``.
ROUTE_CASES = ((13, None), (15, None), (16, None), (11, 3), (10, 2))
CHUNK_ROWS = 12


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


class _Captured(Exception):
    """Raised in place of running the graph once it is kept, as the runner's capture backend does."""


class _Keep:
    """A compile backend that keeps every graph and runs none. """

    def __init__(self) -> None:
        self.graphs: list = []

    def __call__(self, graph_module, _example_inputs):
        self.graphs.append(graph_module)
        return self._stop

    @staticmethod
    def _stop(*_args, **_kwargs):
        raise _Captured()

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
    except _Captured:
        return keep, ""
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
        _keep, message = _traced(
            indexer.seed_tail, chunk["tail"], chunk["key"], chunk["gate"], *positions
        )
        if message:
            refused[form] = message
    assert not refused, (
        f"the trace of seed_tail on meta was refused: {_refusal_reads(next(iter(refused.values())))!r}"
    )


def _prefill_leg_on_meta() -> tuple[_Keep, str, int]:
    """The fixture indexer's whole prefill leg traced on meta at 2048 tokens; ``(keep, message, pool rows)``."""
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
    return keep, message, rows


def test_c_the_indexer_prefill_leg_traces_on_meta_at_2048_tokens() -> None:
    """The whole prefill leg through the indexer's forward, on meta, the kernel gate on.
    """
    _require_cpu_mode()
    assert can_run_kernel(), "the kernel gate reads off under the declared acceptance environment"
    keep, message, _rows = _prefill_leg_on_meta()
    assert not message, (
        f"the trace of the indexer's prefill leg on meta was refused: {_refusal_reads(message)!r}"
    )
    assert len(keep.graphs) == 1, f"{len(keep.graphs)} graphs kept, not one"
    assert keep.nodes_targeting(torch.as_tensor) == 0, "the graph carries a torch.as_tensor node"


def test_e_the_fallback_rotations_run_on_meta_with_their_matrix_on_the_activations_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rotation's two torch fallbacks under the fake mode on meta: one device, meta
    outputs.
    """
    _require_cpu_mode()
    monkeypatch.setenv("VLLM_NEURON_DISABLE_NKI_KERNELS", "1")
    assert not can_run_kernel(), "the kernel gate still reads on with the disabling switch set"
    indexer = _shape_only_indexer(TRACED_DEVICE)
    head_dim = int(indexer.index_head_dim)
    refused = {}
    with FakeTensorMode():
        rows = torch.zeros(TOKENS, head_dim, dtype=torch.bfloat16, device=TRACED_DEVICE)
        slot_k = torch.zeros(TOKENS // POOL, POOL, head_dim, dtype=torch.bfloat16, device=TRACED_DEVICE)
        ape = torch.zeros(POOL, head_dim, dtype=torch.float32, device=TRACED_DEVICE)
        for form, call in (
            ("stage", lambda: dsa_hadamard128(rows)),
            ("fused", lambda: dsa_kpool_hadamard(slot_k, slot_k.clone(), ape)),
        ):
            try:
                out = call()
            except Exception as error:  # the fake mode's refusal is the reading
                message = " ".join(str(error).split())
                refused[form] = message
                continue
            assert out.device.type == TRACED_DEVICE, f"the {form} rotation came back on {out.device}"
    assert not refused, (
        f"the fallback rotations on meta were refused: {_refusal_reads(next(iter(refused.values())))!r}"
    )


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
        if not (same_ring and same_count):
            disagreed.append((end, start))
    assert not disagreed, f"the two routes seeded different rings or counts at {disagreed}"
