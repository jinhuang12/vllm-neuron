"""One compiled graph serves every expert group: the rank enters the forward as a tensor.

The expert bank selected this rank's experts from a python int, so the expert group was a
constant of the captured graph and an expert-parallel serve captured one graph per group.
The runner now hands the rank over as an int64 device tensor and the bank adds the rank's
offset to an ``arange`` from zero, so the group is an input of the graph. This file traces a
two-group tiny root at both ranks and requires one graph, with the python-int form as the
control that still gives two; it requires the tensor form to compute what the int form
computes; and it reads the model file for the casts that were removed.

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 \\
        python -m pytest -s -rA \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_one_graph.py
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch._dynamo as dynamo

import vllm_neuron.functional as functional_hub
import vllm_neuron.vllm.worker.neuron_model_runner as runner_module
from vllm_neuron.model.glm5_next import factory, model_fp8
from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_capture_sites as sites
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as landed
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as item
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_parallel_arguments as par

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The expert groups the tiny root is partitioned into here, and the ranks traced. Two
#: groups of eight keep the tiny top-8 inside one group's experts, as the served shape does.
GROUPS = 2
GROUP_RANKS = list(range(GROUPS))

#: The bank's three weight leaves, in the order the load-time prep takes them.
BANK_LEAVES = ("gate_proj_weight", "up_proj_weight", "down_proj_weight")

#: The form the runner hands the rank over in.
RANK_DTYPE = torch.int64
RANK_SHAPE = (1,)

_MODEL_SOURCE = Path(model_fp8.__file__)
_INT_CAST = "int(expert_parallel_rank)"
_INDICES_CALL = "self.local_expert_indices("
_CONTROL_LINE = "            owned = self.local_expert_indices(int(expert_parallel_rank))"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _pristine_fixtures(ranks) -> dict[int, dict]:
    """One tiny root per rank, all built before any stand-in state is installed.

    The shared fixture's load-time prep hands every bank the whole stack, so a bank built
    under a stand-in degree would refuse it. The roots are therefore built first, under the
    live degree-1 state, and partitioned afterwards.
    """
    landed._require_cpu_mode()
    return {rank: landed._fixture() for rank in ranks}


def _two_group_root(monkeypatch, rank: int, fixture: dict):
    """The tiny root partitioned into two expert groups, holding group ``rank``'s experts.

    ``fixture`` is one of ``_pristine_fixtures``. The two-group state is installed, then each
    routed bank is re-partitioned, re-bound to its group's slice of the stack operands and
    re-prepared, so its weights hold eight of the sixteen experts and the global-to-local
    mapping runs inside the forward.
    """
    root = fixture["root"]
    ep_rank, group = par._install_state(
        monkeypatch, world_size=GROUPS, ep_degree=GROUPS, rank=rank
    )
    assert ep_rank == rank and group.world_size == 1
    per_group = item.STACK_EXPERTS // GROUPS
    low, high = rank * per_group, (rank + 1) * per_group
    rebound = 0
    for index, layer in enumerate(root.model.layers):
        if not isinstance(layer.mlp, model_fp8.Glm5NextMoEBlock):
            continue
        bank = layer.mlp.experts
        assert int(bank.num_local_experts) == item.STACK_EXPERTS, (index, bank.num_local_experts)
        bank.ep_degree = GROUPS
        bank.expert_partition = factory.require_uniform_expert_partition(
            bank.num_routed_experts, GROUPS
        )
        bank.num_local_experts = bank.expert_partition.counts[0]
        assert int(bank.num_local_experts) == per_group, (index, bank.num_local_experts)
        operands = fixture["mlp_operands"][index]
        sliced = {
            leaf: (weight[low:high], grid[low:high]) for leaf, (weight, grid) in operands.items()
        }
        for leaf in BANK_LEAVES:
            item._attach(bank, leaf, *sliced[leaf])
        built = bank.prepare_scale_operands(
            *item._prep_operands_from_the_module(bank, BANK_LEAVES, sliced)
        )
        assert built == 4, built
        rebound += 1
    assert rebound == item.STACK_LAYERS - item.STACK_FIRST_K_DENSE
    root.bind_kv_cache(landed._runner_shaped_caches(root))
    runner = sites._runner(root)
    runner.input_batch = SimpleNamespace(req_ids=["req-0"])
    return root, runner, per_group


def _graph_text(root, translated: dict) -> tuple[str, int]:
    """The FX graphs a fresh trace of the root records for one step, joined, and their count."""
    graphs = []

    def keep(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    dynamo.reset()
    torch.compile(root.forward, backend=keep, dynamic=False)(**translated)
    return "\n".join(str(graph_module.graph) for graph_module in graphs), len(graphs)


def _as_int(translated: dict, rank: int) -> dict:
    """The same step with the rank as a python int, the form that bakes the group in."""
    return {**translated, "expert_parallel_rank": rank}


def test_the_runner_hands_the_rank_over_as_a_device_tensor(monkeypatch, caplog):
    """The rank is an int64 ``[1]`` tensor on the device asked for; the log row keeps the int."""
    ep_rank, _group = par._install_state(
        monkeypatch,
        world_size=par.SERVE_WORLD_SIZE,
        ep_degree=par.SERVE_EP_DEGREE,
        rank=par.SERVE_WORLD_SIZE - 1,
    )
    runner = par._shell()
    with caplog.at_level(logging.INFO, logger=runner_module.__name__):
        on_cpu = runner._glm5next_parallel_kwargs()["expert_parallel_rank"]
        on_meta = runner._glm5next_parallel_kwargs(device=torch.device("meta"))
    rows = [r.getMessage() for r in caplog.records if par.LOG_ROW.search(r.getMessage())]
    monkeypatch.setattr(par.parallel_state, "get_neuron_ep_degree", lambda: 1)
    below_two = par._shell()._glm5next_parallel_kwargs()["expert_parallel_rank"]
    print(
        f"ONEGRAPH|RESOLVER|type={type(on_cpu).__name__} dtype={on_cpu.dtype} "
        f"shape={tuple(on_cpu.shape)} value={int(on_cpu)} "
        f"meta_device={on_meta['expert_parallel_rank'].device.type} "
        f"below_two={type(below_two).__name__}:{int(below_two)} "
        f"log_rows={len(rows)} row={rows[0] if rows else None}"
    )
    assert isinstance(on_cpu, torch.Tensor), f"the rank is a {type(on_cpu).__name__}, not a tensor"
    assert on_cpu.dtype == RANK_DTYPE and tuple(on_cpu.shape) == RANK_SHAPE
    assert int(on_cpu) == ep_rank and ep_rank != 0
    assert on_meta["expert_parallel_rank"].device.type == "meta"
    assert isinstance(below_two, torch.Tensor) and int(below_two) == 0
    assert rows == [
        f"glm5next parallel arguments rank=0 ep_rank={ep_rank} "
        f"ep_degree={par.SERVE_EP_DEGREE} tp_degree={par.SERVE_WORLD_SIZE // par.SERVE_EP_DEGREE}"
    ]
    assert "tensor" not in rows[0]


def test_both_captures_and_a_prompt_step_hand_the_root_the_tensor(monkeypatch):
    """Every driven site reaches the root with the rank as a tensor on the batch's device."""
    fixture = _pristine_fixtures([GROUPS - 1])[GROUPS - 1]
    root, runner, _per_group = _two_group_root(monkeypatch, GROUPS - 1, fixture)
    reached: list[dict] = []
    original = type(root).forward

    def spy(self, *args, **kwargs):
        reached.append(dict(kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(root), "forward", spy)
    backend = sites._StandInBackend(runner)
    runner.capture_backend_model = backend
    runner.extract_prefill_graphs(sites.PREFILL_BUCKET, 0)
    runner.extract_decode_graphs(sites.DECODE_BATCH)
    root(**par._translated_prompt(runner))
    tensors = [
        call for call in reached if isinstance(call.get("expert_parallel_rank"), torch.Tensor)
    ]
    placed = [
        call for call in tensors
        if call["expert_parallel_rank"].device == call["input_ids"].device
        and int(call["expert_parallel_rank"]) == GROUPS - 1
    ]
    print(
        f"ONEGRAPH|SITES|forwards={len(reached)} tensors={len(tensors)} "
        f"on_batch_device={len(placed)} captures={len(backend.seen)} ep_rank={GROUPS - 1}"
    )
    assert len(backend.seen) == 2 and len(reached) == 3
    assert len(tensors) == len(reached), (
        f"{len(reached) - len(tensors)} of {len(reached)} root calls carried the rank as an int"
    )
    assert len(placed) == len(reached)


def test_one_graph_serves_both_expert_groups(monkeypatch):
    """The traced graph is the same at rank 0 and rank 1; the python-int form gives two."""
    texts: dict[str, dict[int, str]] = {"tensor": {}, "int": {}}
    counts: dict[int, int] = {}
    fixtures = _pristine_fixtures(GROUP_RANKS)
    for rank in GROUP_RANKS:
        root, runner, _per_group = _two_group_root(monkeypatch, rank, fixtures[rank])
        translated = par._translated_prompt(runner)
        assert int(translated["expert_parallel_rank"]) == rank
        texts["tensor"][rank], counts[rank] = _graph_text(root, translated)
        texts["int"][rank], _count = _graph_text(root, _as_int(translated, rank))
    tensor_hashes = [_digest(texts["tensor"][rank]) for rank in GROUP_RANKS]
    int_hashes = [_digest(texts["int"][rank]) for rank in GROUP_RANKS]
    one_graph = len(set(texts["tensor"].values())) == 1
    control_one_graph = len(set(texts["int"].values())) == 1
    print(
        f"ONEGRAPH|IDENTITY|graphs_per_trace={[counts[rank] for rank in GROUP_RANKS]} "
        f"tensor_hashes={tensor_hashes} one_graph={one_graph} "
        f"int_hashes={int_hashes} control_one_graph={control_one_graph}"
    )
    assert all(count > 0 for count in counts.values()), counts
    assert not control_one_graph, (
        "the python-int control traced one graph at both ranks, so this item cannot tell "
        "one graph from two"
    )
    assert one_graph, f"the graphs at rank 0 and rank 1 differ: {tensor_hashes}"


def test_the_tensor_form_computes_what_the_int_form_computes(monkeypatch):
    """At each group, the tensor form selects the group's experts and equals the int form."""
    fixtures = _pristine_fixtures(GROUP_RANKS)
    selected: list[list[int]] = []
    original = functional_hub.get_local_expert_affinities

    def record(expert_affinities, local_expert_indices):
        selected.append(local_expert_indices.tolist())
        return original(expert_affinities, local_expert_indices)

    # The bank imports the mapper from the functional package at call time, so the
    # package is where a stand-in has to sit; one stand-in serves both groups.
    monkeypatch.setattr(functional_hub, "get_local_expert_affinities", record)
    for rank in GROUP_RANKS:
        root, runner, per_group = _two_group_root(monkeypatch, rank, fixtures[rank])
        translated = par._translated_prompt(runner)
        selected.clear()
        tensor_form = root(**translated)
        mapped = len(selected)
        int_form = root(**_as_int(translated, rank))
        equal = torch.equal(tensor_form, int_form)
        owned = list(range(rank * per_group, (rank + 1) * per_group))
        print(
            f"ONEGRAPH|NUMERICS|rank={rank} mappings={mapped} "
            f"selected={selected[0] if selected else None} owned={owned} equal={equal}"
        )
        assert mapped == item.STACK_LAYERS - item.STACK_FIRST_K_DENSE, mapped
        assert all(indices == owned for indices in selected[:mapped]), selected
        assert equal, f"rank {rank}: max |tensor - int| = {(tensor_form - int_form).abs().max()}"


def test_the_tensor_form_through_the_translation_equals_the_int_form(monkeypatch):
    """On the narrow mesh, rank 0 and rank 15 through the translation equal the int form."""
    bound = {rank: par._bound_runner() for rank in (0, par.NARROW_RANK)}
    for rank, (root, _caches, runner) in bound.items():
        ep_rank, _group = par._install_state(
            monkeypatch,
            world_size=par.NARROW_WORLD_SIZE,
            ep_degree=par.NARROW_EP_DEGREE,
            rank=rank,
        )
        translated = par._translated_prompt(runner)
        tensor_form = root(**translated)
        int_form = root(**_as_int(translated, ep_rank))
        equal = torch.equal(tensor_form, int_form)
        carried = type(translated["expert_parallel_rank"]).__name__
        print(f"ONEGRAPH|NARROW|rank={ep_rank} carried={carried} equal={equal}")
        assert ep_rank == rank
        assert isinstance(translated["expert_parallel_rank"], torch.Tensor), (
            f"the translation carried the rank as {carried}"
        )
        assert equal, f"rank {ep_rank}: the tensor form and the int form disagree"


def test_the_model_file_reads_no_python_int_off_the_rank():
    """No ``int(expert_parallel_rank)`` cast and no expert-index call remain at any site."""
    lines = _MODEL_SOURCE.read_text().splitlines()
    casts = [number for number, line in enumerate(lines, 1) if _INT_CAST in line]
    calls = [number for number, line in enumerate(lines, 1) if _INDICES_CALL in line]
    control_flagged = _INT_CAST in _CONTROL_LINE and _INDICES_CALL in _CONTROL_LINE
    print(
        f"ONEGRAPH|SCAN|int_casts={len(casts)} index_calls={len(calls)} "
        f"control_flagged={control_flagged} lines={sorted(set(casts + calls))}"
    )
    assert control_flagged
    assert casts == [], f"int(expert_parallel_rank) remains at {casts}"
    assert calls == [], f"self.local_expert_indices( remains at {calls}"


def test_every_groups_first_expert_is_the_rank_times_the_group_size():
    """The offset form is right because the partition is uniform and contiguous."""
    experts = int(Glm5NextTextConfig().n_routed_experts)
    checked = []
    for total, groups in ((experts, par.SERVE_EP_DEGREE), (item.STACK_EXPERTS, GROUPS)):
        partition = factory.require_uniform_expert_partition(total, groups)
        per_group = total // groups
        agree = all(
            partition.local_expert_indices(rank)
            == tuple(range(rank * per_group, (rank + 1) * per_group))
            for rank in range(groups)
        )
        checked.append((total, groups, per_group, agree))
    print(f"ONEGRAPH|PARTITION|checked={checked}")
    assert all(agree for _total, _groups, _per_group, agree in checked), checked
