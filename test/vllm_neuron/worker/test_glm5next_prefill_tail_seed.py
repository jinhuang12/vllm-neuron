# SPDX-License-Identifier: Apache-2.0
"""The prefill leg's ring seed survives the backend's passes on every DSA layer.

A prefill chunk ends inside a pool unless the sequence length divides by
``index_kpool``. The open pool's keys and gate scores exist only inside that
forward, so ``Glm5NextDSAIndexer.seed_tail`` writes them into the request's ring
(the carrier's ``prefill_tail``) for the first decode step to complete the pool
from. On the served path the chunk's end is a tensor, so the write is
``_seed_tail_at``'s.

The served graph is lowered by libtorch_neuronx_lite's ``AliasingOutputRewritePass``
and ``InPlaceToOutOfPlacePass``. A write through a view of a graph input, such as
``tail[0].copy_(...)``, is lost there. The aliasing pass does not trace an
integer-indexed view back to its placeholder, so the ring gets no aliased output.
The in-place pass rewrites only the later uses of the view itself, and nothing
reads them. The device then keeps the ring that the runner emptied, and the next
pool completion pools zeros for the prompt's last positions. A context within the
decode bypass (``index_topk + index_kpool`` tokens) never selects, so no
short-context accuracy gate sees the loss.

Part 1 traces one prefill step of the tiny root with Dynamo as one graph, runs the
two passes, executes the rewritten graph on CPU and copies each aliased output onto
its input, as the runtime does. The prompts have 5, 6 and 7 tokens (pool 4, so the
remainders are 1, 2 and 3). The draft head is off (``k = 0``) or on (``k = 3``,
and then the draft layer seeds its own ring too). Every seeding layer's ring must be
an aliased output and must equal the eager step's ring bit for bit, and the
captured graph must write each ring once, on the ring itself. An 8-token prompt is
the control: the sequence divides evenly, the eager step writes nothing, and the
served ring stays as the runner left it.

Part 2 is the eager write: the tensor route (``_seed_tail_at``) against the int
route (``seed_tail`` with host positions) on random rings, chunks and paddings, in
both 2-byte float dtypes the ring takes (the ring has the model dtype).

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_glm5next_prefill_tail_seed.py
"""

from __future__ import annotations

import operator

import pytest
import torch
import torch._dynamo as dynamo

from test.vllm_neuron.model.glm5_next import test_dsa_layer as layer_half
from test.vllm_neuron.model.glm5_next import test_shadow_draft_e2e as shadow

pytestmark = [pytest.mark.forked]

#: The tiny stack's pool size; every ring below has this many slots.
POOL = layer_half.POOL_SIZE
#: Prompt lengths that end inside a pool: remainders 1, 2 and 3.
OPEN_LENGTHS = (5, 6, 7)
#: A prompt length that divides evenly: no open pool, nothing to seed.
EVEN_LENGTH = 8
#: The draft head off (the standard arm) and at the served draft count.
DRAFT_COUNTS = (0, 3)
#: Random cases of the eager comparison, per dtype.
EAGER_CASES = 24
EAGER_SEED = 20261009


# ── part 1: one prefill step through the backend's own passes ────────────────


def _prompt(length: int) -> list[int]:
    assert length <= len(shadow.LONG_PROMPT), length
    return list(shadow.LONG_PROMPT[:length])


def _banks(world, kwargs: dict) -> dict[str, torch.Tensor]:
    """Every floating-point tensor the step can write, by name, one entry per storage view.

    The runner-shaped KV caches by cache name, then each carrier tensor by its path.
    """
    named: dict[str, torch.Tensor] = {}
    seen: set[tuple[int, int]] = set()

    def add(name: str, tensor: torch.Tensor) -> None:
        where = (tensor.untyped_storage().data_ptr(), tensor.storage_offset())
        if tensor.is_floating_point() and where not in seen:
            seen.add(where)
            named[name] = tensor

    for name, tensors in world.caches.items():
        for index, tensor in enumerate(tensors):
            add(f"{name}[{index}]", tensor)
    for index, carrier in enumerate(kwargs["layer_carriers"]):
        for key, value in carrier.items():
            if torch.is_tensor(value):
                add(f"layer_carriers[{index}][{key!r}]", value)
    return named


def _rings(kwargs: dict) -> dict[int, torch.Tensor]:
    """Each seeding layer's ring, by carrier index (the draft layer's is the last)."""
    return {
        index: carrier["prefill_tail"]
        for index, carrier in enumerate(kwargs["layer_carriers"])
        if "prefill_tail" in carrier
    }


def _restore(banks: dict[str, torch.Tensor], saved: dict[str, torch.Tensor]) -> None:
    for name, tensor in banks.items():
        tensor.copy_(saved[name])


def _input_index(inputs: list, tensor: torch.Tensor) -> int:
    """The one graph input that is ``tensor`` (same storage, offset and shape)."""
    matches = [
        index
        for index, value in enumerate(inputs)
        if torch.is_tensor(value)
        and value.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
        and value.storage_offset() == tensor.storage_offset()
        and tuple(value.shape) == tuple(tensor.shape)
    ]
    assert len(matches) == 1, matches
    return matches[0]


_VIEW_METHODS = {"view", "reshape", "select", "narrow", "squeeze", "unsqueeze",
                 "transpose", "permute", "expand", "flatten", "unflatten"}


def _writes_by_input(gm: torch.fx.GraphModule) -> dict[int, list[str]]:
    """The captured graph's in-place writes per placeholder index, before any pass.

    Each write is named by its op and whether it lands on the placeholder itself
    (``root``) or on a view of it (``view``). Only a root write is threaded to the
    graph's output by the in-place pass.
    """
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    position = {node: index for index, node in enumerate(placeholders)}

    def root_of(node):
        hops = 0
        while node not in position:
            is_view = (
                (node.op == "call_function" and node.target is operator.getitem)
                or (node.op == "call_method" and node.target in _VIEW_METHODS)
            )
            if not is_view or not node.args or not isinstance(node.args[0], torch.fx.Node):
                return None, hops
            node, hops = node.args[0], hops + 1
        return position[node], hops

    writes: dict[int, list[str]] = {}
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target is operator.setitem:
            name = "setitem"
        elif (
            node.op == "call_method"
            and node.target.endswith("_")
            and not node.target.startswith("__")
        ):
            name = node.target
        else:
            continue
        target = node.args[0]
        if not isinstance(target, torch.fx.Node):
            continue
        index, hops = root_of(target)
        if index is not None:
            writes.setdefault(index, []).append(f"{name}:{'root' if hops == 0 else 'view'}")
    return writes


def _lower(fn, kwargs: dict) -> dict:
    """Dynamo-trace ``fn(**kwargs)`` as ONE graph and run the backend's two rewrite passes.

    Returns the rewritten (out-of-place) graph, its ``io_map`` (aliased output index
    to graph input index), the original output count, the graph's inputs, and the
    captured graph's in-place writes per input before the passes ran. A graph break
    fails the trace (``fullgraph``): the served prefill step is one graph.
    """
    from libtorch_neuronx_lite.fx_passes.aliasing_pass import AliasingOutputRewritePass
    from libtorch_neuronx_lite.fx_passes.inplace_rewrite_pass import InPlaceToOutOfPlacePass

    kept: dict = {}

    def backend(gm, example_inputs):
        kept["writes"] = _writes_by_input(gm)
        rewritten, meta = AliasingOutputRewritePass().run(gm)
        rewritten, _ = InPlaceToOutOfPlacePass().run(rewritten)
        kept.update(
            gm=rewritten,
            io_map=dict(meta["io_map"]),
            outputs=int(meta["original_output_count"]),
            inputs=list(example_inputs),
        )
        return rewritten.forward

    dynamo.reset()
    torch.compile(fn, backend=backend, dynamic=False, fullgraph=True)(**kwargs)
    dynamo.reset()
    return kept


def _serve(lowered: dict) -> None:
    """Run the rewritten graph and copy each aliased output onto its input.

    This is the runtime's write-back.
    """
    outputs = lowered["gm"](*lowered["inputs"])
    outputs = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
    for output_index, input_index in lowered["io_map"].items():
        lowered["inputs"][input_index].copy_(outputs[output_index])


def _prefill_through_the_backend(monkeypatch, k: int, length: int):
    """One prefill of ``length`` prompt tokens, eager and through the passes.

    Returns, per seeding layer, the ring before the step, after the eager step and
    after the served step, with the lowered graph.
    """
    monkeypatch.setenv(shadow.KNOB, str(k))
    prompt = _prompt(length)
    world = shadow._world(prompt=prompt)
    assert (world.root.mtp is not None) == bool(k), (k, world.root.mtp)
    held: dict = {}

    def keep(converted):
        held["kwargs"] = converted
        held["banks"] = _banks(world, converted)
        held["before"] = {n: t.clone() for n, t in held["banks"].items()}
        held["rings"] = {i: ring.clone() for i, ring in _rings(converted).items()}

    # The converter moves the runner's ring bookkeeping, so the eager and the traced
    # step share one conversion; the eager step runs on the live banks here.
    shadow._step(world, prompt, cached=0, sampling=[length - 1], mutate=keep)
    kwargs, banks, before = held["kwargs"], held["banks"], held["before"]
    rings, rings_before = _rings(kwargs), held["rings"]
    eager = {index: ring.clone() for index, ring in rings.items()}
    _restore(banks, before)
    lowered = _lower(lambda **kw: world.root.forward(**kw), dict(kwargs))
    # The trace ran the out-of-place graph once; start the served step from the
    # same banks the eager step started from.
    _restore(banks, before)
    _serve(lowered)
    served = {index: ring.clone() for index, ring in rings.items()}
    expected_layers = sum(
        1 for bank in world.root.glm5next_layer_banks if bank["family"] == "self_attn"
    )
    return _Prefill(rings, rings_before, eager, served, lowered, expected_layers)


class _Prefill:
    """What one prefill through the backend left, per seeding layer."""

    def __init__(self, rings, before, eager, served, lowered, expected_layers):
        self.rings = rings
        self.before = before
        self.eager = eager
        self.served = served
        self.lowered = lowered
        self.expected_layers = expected_layers

    def report(self) -> list[str]:
        lines = []
        for index, ring in self.rings.items():
            graph_input = _input_index(self.lowered["inputs"], ring)
            lines.append(
                f"carrier {index}: aliased output "
                f"{graph_input in self.lowered['io_map'].values()}, "
                f"eager wrote {not torch.equal(self.eager[index], self.before[index])}, "
                f"served == eager {torch.equal(self.served[index], self.eager[index])}, "
                f"served == before {torch.equal(self.served[index], self.before[index])}, "
                f"writes {self.lowered['writes'].get(graph_input, [])}"
            )
        return lines


@pytest.mark.parametrize("length", OPEN_LENGTHS)
@pytest.mark.parametrize("k", DRAFT_COUNTS)
def test_every_layers_ring_seed_survives_the_backend_passes(monkeypatch, k, length):
    record = _prefill_through_the_backend(monkeypatch, k, length)
    report = "\n".join(record.report())
    assert len(record.rings) == record.expected_layers, (
        f"{len(record.rings)} carrier(s) seed a ring and the root has "
        f"{record.expected_layers} sparse layer(s) (draft layer included):\n{report}"
    )
    for index in record.rings:
        assert not torch.equal(record.eager[index], record.before[index]), (
            f"carrier {index}: the eager step left the ring as it was, so a "
            f"{length}-token prompt (remainder {length % POOL}) seeded nothing and "
            f"this comparison would be vacuous:\n{report}"
        )
    for index, ring in record.rings.items():
        graph_input = _input_index(record.lowered["inputs"], ring)
        assert graph_input in record.lowered["io_map"].values(), (
            f"carrier {index}'s ring is graph input {graph_input} and no output "
            f"aliases it (io_map={record.lowered['io_map']}), so the device keeps the "
            f"ring the runner emptied:\n{report}"
        )
        assert torch.equal(record.served[index], record.eager[index]), (
            f"carrier {index}: the served ring is not the eager step's:\n{report}"
        )


@pytest.mark.parametrize("k", DRAFT_COUNTS)
def test_the_captured_prefill_graph_writes_each_ring_once_on_the_ring_itself(monkeypatch, k):
    record = _prefill_through_the_backend(monkeypatch, k, OPEN_LENGTHS[1])
    report = "\n".join(record.report())
    for index, ring in record.rings.items():
        graph_input = _input_index(record.lowered["inputs"], ring)
        assert record.lowered["writes"].get(graph_input) == ["copy_:root"], (
            f"carrier {index}: the ring must take one write per graph, on the ring "
            f"itself (a view write is dropped by the in-place pass, and a second "
            f"root write is a second write of one bank in one graph):\n{report}"
        )


@pytest.mark.parametrize("k", DRAFT_COUNTS)
def test_a_prompt_that_fills_its_last_pool_leaves_every_ring_as_the_runner_left_it(
    monkeypatch, k
):
    record = _prefill_through_the_backend(monkeypatch, k, EVEN_LENGTH)
    report = "\n".join(record.report())
    assert record.rings, "no carrier seeds a ring, so this control checks nothing"
    for index in record.rings:
        assert torch.equal(record.eager[index], record.before[index]), (
            f"carrier {index}: the eager step wrote a ring at a length that divides "
            f"evenly, where the write mask is empty everywhere:\n{report}"
        )
        assert torch.equal(record.served[index], record.before[index]), (
            f"carrier {index}: the served step changed a ring the mask leaves "
            f"alone:\n{report}"
        )


# ── part 2: the eager write, tensor route against int route ──────────────────


def _eager_case(gen: torch.Generator, head_dim: int, dtype: torch.dtype) -> dict:
    """One random chunk: an end position, a start (or None), a padded width, a ring.

    Covers an empty window (the end divides evenly), the whole remainder in this
    chunk, and a chunk that holds only the remainder's tail (an earlier chunk began
    the open pool).
    """
    end = int(torch.randint(1, 6 * POOL, (1,), generator=gen))
    real = int(torch.randint(1, end + 1, (1,), generator=gen))
    padded = real + int(torch.randint(0, 3, (1,), generator=gen))
    # A padded chunk names its start, so its remainder is read from its real rows.
    unnamed = bool(torch.randint(0, 2, (1,), generator=gen)) and padded == real
    start = None if unnamed else end - real
    ring = torch.randn(2, POOL, head_dim, generator=gen).to(dtype)
    key = torch.randn(padded, head_dim, generator=gen).to(dtype)
    gate = torch.randn(padded, head_dim, generator=gen).to(dtype)
    return {"end": end, "start": start, "ring": ring, "key": key, "gate": gate, "real": real}


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_the_tensor_route_writes_the_int_routes_ring_bit_for_bit(dtype):
    indexer = layer_half._bare_indexer()
    head_dim = int(indexer.index_head_dim)
    assert int(indexer.index_kpool) == POOL
    gen = torch.Generator().manual_seed(EAGER_SEED)
    windows = set()
    for case in range(EAGER_CASES):
        c = _eager_case(gen, head_dim, dtype)
        by_int, by_tensor = c["ring"].clone(), c["ring"].clone()
        rows_int = indexer.seed_tail(by_int, c["key"], c["gate"], c["end"], c["start"])
        start = None if c["start"] is None else torch.tensor(c["start"], dtype=torch.int32)
        rows_tensor = indexer.seed_tail(
            by_tensor, c["key"], c["gate"], torch.tensor(c["end"], dtype=torch.int32), start
        )
        assert int(rows_tensor) == int(rows_int), (case, c["end"], c["start"])
        assert by_tensor.dtype == dtype
        assert torch.equal(by_tensor.view(torch.int16), by_int.view(torch.int16)), (
            f"case {case}: end {c['end']}, start {c['start']}, width "
            f"{int(c['key'].shape[0])}: the tensor route's ring differs from the int "
            f"route's"
        )
        rows = c["end"] % POOL
        windows.add("empty" if rows == 0 else "whole" if c["real"] >= rows else "partial")
    assert windows == {"empty", "whole", "partial"}, (
        f"the random cases covered only {sorted(windows)} of the three window kinds"
    )
