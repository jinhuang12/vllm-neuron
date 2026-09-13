"""The runner hands the root its three parallelism arguments, read from the parallel state.

The root's forward leaves ``moe_group``, ``tp_degree`` and ``expert_parallel_rank`` to its
caller and defaults them to the unsharded values, so a translation that supplied none made
every rank of an expert-parallel serve run the first expert group. This file drives the
runner's translation under a stand-in parallel state and requires the served values to
reach the root at every site the CPU lane can drive, and the expert bank behind it.

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 \\
        python -m pytest -s -rA \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_parallel_arguments.py
"""

from __future__ import annotations

import inspect
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import vllm_neuron.vllm.worker.neuron_model_runner as runner_module
from vllm_neuron.model.glm5_next import factory, model_fp8
from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
from vllm_neuron.parallel import neuron_parallel_state as parallel_state
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_capture_sites as sites
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as landed
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as item

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The three keyword arguments the resolver supplies, sorted.
PARALLEL_KWARG_KEYS = ["expert_parallel_rank", "moe_group", "tp_degree"]

#: The six keys a translated step carries: the three carriers and the three above.
TRANSLATED_KEYS = sorted(
    ["input_ids", "layer_carriers", "sampling_positions", *PARALLEL_KWARG_KEYS]
)

#: The values the root defaults the three arguments to, which the runner passes explicitly
#: below expert-parallel degree 2.
UNSHARDED = {"moe_group": None, "tp_degree": 1, "expert_parallel_rank": 0}

#: The mesh the fork serves this model on: 64 ranks in 16 expert groups of 4.
SERVE_WORLD_SIZE = 64
SERVE_EP_DEGREE = 16

#: The meshes every rank is resolved on: the parallel state's own docstring example, and the
#: served one.
MESHES = [(8, 2), (SERVE_WORLD_SIZE, SERVE_EP_DEGREE)]

#: A mesh whose expert groups are one rank wide: a bank the fixture built for one rank runs
#: unsharded under it, while the rank it receives is the last group's and not the default.
NARROW_WORLD_SIZE = 16
NARROW_EP_DEGREE = 16
NARROW_RANK = NARROW_WORLD_SIZE - 1

#: The model call sites the runner declares: three warmups, execute, the idle dummy step and
#: three graph captures.
RUNNER_MODEL_CALL_SITES = 8

#: The one row the first resolve on a runner logs, which a serve's log carries once per rank.
LOG_ROW = re.compile(
    r"glm5next parallel arguments rank=[0-9]+ ep_rank=[0-9]+ ep_degree=[0-9]+ tp_degree=[0-9]+"
)

_RUNNER_SOURCE = Path(runner_module.__file__)
_MODEL_CALL = re.compile(r"self\.(?:capture_backend_)?model\(\*\*(?P<splat>[^)]*)\)")
_TRANSLATED_SPLAT = "self._glm5next_model_kwargs("


class _StandInGroup:
    """The two fields the resolver and the bank read off a ``GroupCoordinator``."""

    def __init__(self, rank_in_group: int, world_size: int) -> None:
        self.rank_in_group = rank_in_group
        self.world_size = world_size


def _mesh_position(world_size: int, ep_degree: int, rank: int) -> tuple[int, int]:
    """``(ep_rank, column)`` for one global rank, from the package's own mesh builder."""
    rows, _columns = parallel_state._build_ep_group_ranks(world_size, ep_degree)
    for ep_rank, row in enumerate(rows):
        if rank in row:
            return ep_rank, row.index(rank)
    raise AssertionError(f"rank {rank} sits in no row of the {world_size}/{ep_degree} mesh")


def _install_state(monkeypatch, *, world_size: int, ep_degree: int, rank: int):
    """Stand in for one rank's parallel state on the module the resolver reads.

    Installed AFTER a fixture is built: the bank asks the same module for its degree when
    it is constructed, and this file measures a bank built for one rank.
    """
    ep_rank, column = _mesh_position(world_size, ep_degree, rank)
    group = _StandInGroup(column, world_size // ep_degree)
    monkeypatch.setattr(parallel_state, "get_neuron_ep_degree", lambda: ep_degree)
    monkeypatch.setattr(parallel_state, "get_neuron_ep_rank", lambda: ep_rank)
    monkeypatch.setattr(parallel_state, "get_neuron_ep_tp_group", lambda: group)
    return ep_rank, group


def _refuse():
    raise AssertionError("an expert-parallel getter was read below degree 2")


def _shell() -> NeuronModelRunner:
    """A runner shell; the resolver reads nothing off it."""
    return NeuronModelRunner.__new__(NeuronModelRunner)


def _bound_runner():
    """The tiny root, bound, on the capture harness's runner shell with a one-request batch."""
    root, caches = sites._bound_root()
    runner = sites._runner(root)
    runner.input_batch = SimpleNamespace(req_ids=["req-0"])
    return root, caches, runner


def _translated_prompt(runner) -> dict:
    """The end-to-end file's prompt step, translated by the converter under test."""
    prompt = torch.randint(
        0,
        item.STACK_VOCAB_SIZE,
        (item.STACK_TOKENS,),
        generator=torch.Generator().manual_seed(item.SEED_STACK_IDS),
        dtype=torch.int64,
    )
    return landed._model_kwargs(
        runner, input_ids=prompt, cached=0, sampling_row=item.STACK_TOKENS - 1
    )


def _served(call: dict, ep_rank: int, group: _StandInGroup) -> bool:
    """Whether one root call carries the stand-in state's values and not the defaults."""
    return (
        call.get("expert_parallel_rank") == ep_rank
        and call.get("tp_degree") == group.world_size
        and call.get("moe_group") is group
    )


def test_below_degree_two_the_defaults_are_handed_over_explicitly(monkeypatch):
    """Degree 1 hands the root the unsharded values by name and reads no group."""
    assert int(parallel_state.get_neuron_ep_degree()) == 1, (
        "this lane never initialises expert parallelism, so the live degree is 1"
    )
    live = _shell()._glm5next_parallel_kwargs()
    monkeypatch.setattr(parallel_state, "get_neuron_ep_degree", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_neuron_ep_rank", _refuse)
    monkeypatch.setattr(parallel_state, "get_neuron_ep_tp_group", _refuse)
    supplied = _shell()._glm5next_parallel_kwargs()
    print(
        f"PAR|DEFAULTS|keys={sorted(supplied)} moe_group={supplied['moe_group']} "
        f"tp_degree={supplied['tp_degree']} "
        f"expert_parallel_rank={supplied['expert_parallel_rank']} live_matches={live == supplied}"
    )
    assert sorted(supplied) == PARALLEL_KWARG_KEYS
    assert supplied == UNSHARDED
    assert live == UNSHARDED


def test_every_rank_reads_its_own_expert_rank_degree_and_group(monkeypatch):
    """Each rank of a mesh resolves its row's group, the row's width and its own row index."""
    for world_size, ep_degree in MESHES:
        tp_degree = world_size // ep_degree
        resolved: list[int] = []
        for rank in range(world_size):
            ep_rank, group = _install_state(
                monkeypatch, world_size=world_size, ep_degree=ep_degree, rank=rank
            )
            supplied = _shell()._glm5next_parallel_kwargs()
            assert sorted(supplied) == PARALLEL_KWARG_KEYS
            assert supplied["expert_parallel_rank"] == ep_rank, (rank, supplied)
            assert supplied["tp_degree"] == tp_degree, (rank, supplied)
            assert supplied["moe_group"] is group, (rank, supplied)
            resolved.append(supplied["expert_parallel_rank"])
        print(
            f"PAR|MESH|world={world_size} ep={ep_degree} tp_degree={tp_degree} "
            f"ranks={world_size} distinct_ep_ranks={len(set(resolved))} "
            f"ranks_per_group={resolved.count(0)}"
        )
        assert sorted(set(resolved)) == list(range(ep_degree))
        assert all(resolved.count(ep_rank) == tp_degree for ep_rank in range(ep_degree))


def test_the_first_resolve_logs_one_row_naming_the_rank(monkeypatch, caplog):
    """One INFO row per runner names the rank's group and degrees; a second resolve adds none."""
    ep_rank, group = _install_state(
        monkeypatch,
        world_size=SERVE_WORLD_SIZE,
        ep_degree=SERVE_EP_DEGREE,
        rank=SERVE_WORLD_SIZE - 1,
    )
    runner = _shell()
    with caplog.at_level(logging.INFO, logger=runner_module.__name__):
        runner._glm5next_parallel_kwargs()
        runner._glm5next_parallel_kwargs()
    rows = [record for record in caplog.records if LOG_ROW.search(record.getMessage())]
    row = rows[0].getMessage() if rows else None
    level = rows[0].levelname if rows else None
    print(f"PAR|LOG|rows={len(rows)} level={level} row={row}")
    assert len(rows) == 1
    assert row == (
        f"glm5next parallel arguments rank=0 ep_rank={ep_rank} ep_degree={SERVE_EP_DEGREE} "
        f"tp_degree={group.world_size}"
    )
    assert level == "INFO" and "'" not in row and '"' not in row


def test_the_translation_carries_the_three_arguments_and_the_root_binds_them(monkeypatch):
    """A translated step carries the six keys, with the served values, and the root accepts them."""
    root, _caches, runner = _bound_runner()
    ep_rank, group = _install_state(
        monkeypatch,
        world_size=SERVE_WORLD_SIZE,
        ep_degree=SERVE_EP_DEGREE,
        rank=SERVE_WORLD_SIZE - 1,
    )
    translated = _translated_prompt(runner)
    assert sorted(translated) == TRANSLATED_KEYS, (
        f"the translation carries {sorted(translated)}"
    )
    print(
        f"PAR|TRANSLATION|keys={sorted(translated)} "
        f"expert_parallel_rank={translated['expert_parallel_rank']} "
        f"tp_degree={translated['tp_degree']} "
        f"group_is_the_states={translated['moe_group'] is group}"
    )
    assert ep_rank != 0 and translated["expert_parallel_rank"] == ep_rank
    assert translated["tp_degree"] == SERVE_WORLD_SIZE // SERVE_EP_DEGREE
    assert translated["moe_group"] is group
    inspect.signature(root.forward).bind(**translated)
    assert sites.CARRIER_KWARG_KEYS == TRANSLATED_KEYS, "the capture harness declares the six"


def test_no_driven_site_reaches_the_root_with_the_defaults(monkeypatch):
    """Both graph captures and a real prompt step reach the root with the served values."""
    root, _caches, runner = _bound_runner()
    ep_rank, group = _install_state(
        monkeypatch, world_size=NARROW_WORLD_SIZE, ep_degree=NARROW_EP_DEGREE, rank=NARROW_RANK
    )
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
    root(**_translated_prompt(runner))
    with_defaults = [call for call in reached if not _served(call, ep_rank, group)]
    print(
        f"PAR|SITES|driven=3 forwards={len(reached)} with_defaults={len(with_defaults)} "
        f"ep_rank={ep_rank} captures={len(backend.seen)}"
    )
    assert len(backend.seen) == 2 and len(reached) == 3
    assert len(with_defaults) == 0, (
        f"{len(with_defaults)} of {len(reached)} root calls carried the defaults"
    )


def test_a_forward_through_the_translation_hands_the_bank_the_rank(monkeypatch):
    """The bank receives the served rank at every routed layer, and the logits are the reference's.

    The reference is the same fixture stepped under the live degree-1 state; the candidate is
    a second copy of that fixture under the narrow mesh, where the bank runs unsharded and
    receives the last group's rank. The fixture's bank holds every expert, so the rank
    selects nothing and the two logits are equal exactly.
    """
    reference_root, _caches, reference_runner = _bound_runner()
    reference = reference_root(**_translated_prompt(reference_runner))
    root, _caches, runner = _bound_runner()
    ep_rank, group = _install_state(
        monkeypatch, world_size=NARROW_WORLD_SIZE, ep_degree=NARROW_EP_DEGREE, rank=NARROW_RANK
    )
    handed: list[dict] = []
    original = model_fp8.Glm5NextRoutedExperts.block_quant_expert_mm

    def spy(self, *args, **kwargs):
        handed.append(dict(kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(model_fp8.Glm5NextRoutedExperts, "block_quant_expert_mm", spy)
    candidate = root(**_translated_prompt(runner))
    routed_layers = int(root.text_config.num_hidden_layers) - int(
        root.text_config.first_k_dense_replace
    )
    with_defaults = [call for call in handed if not _served(call, ep_rank, group)]
    equal = torch.equal(candidate, reference)
    print(
        f"PAR|BANK|routed_layers={routed_layers} bank_calls={len(handed)} "
        f"with_defaults={len(with_defaults)} ep_rank={ep_rank} tp_degree={group.world_size} "
        f"equal_to_reference={equal}"
    )
    assert routed_layers > 0 and len(handed) == routed_layers
    assert len(with_defaults) == 0, (
        f"{len(with_defaults)} of {len(handed)} bank calls carried the defaults"
    )
    assert equal, f"max |candidate - reference| = {(candidate - reference).abs().max()}"


def test_each_expert_group_owns_a_contiguous_slice_of_the_experts():
    """Group ``r`` of the served mesh owns experts ``[E/16 * r, E/16 * (r + 1))``."""
    experts = int(Glm5NextTextConfig().n_routed_experts)
    per_group = experts // SERVE_EP_DEGREE
    partition = factory.require_uniform_expert_partition(experts, SERVE_EP_DEGREE)
    for rank in range(SERVE_EP_DEGREE):
        assert partition.local_expert_indices(rank) == tuple(
            range(per_group * rank, per_group * (rank + 1))
        ), rank
    root = landed._fixture()["root"]
    tiny_experts = int(root.text_config.n_routed_experts)
    bank = model_fp8.Glm5NextRoutedExperts(
        root.text_config, world_size=tiny_experts, ep_degree=tiny_experts
    )
    print(
        f"PAR|PARTITION|experts={experts} groups={SERVE_EP_DEGREE} per_group={per_group} "
        f"last_group={partition.local_expert_indices(SERVE_EP_DEGREE - 1)[0]}.."
        f"{partition.local_expert_indices(SERVE_EP_DEGREE - 1)[-1]} "
        f"tiny_experts={tiny_experts} tiny_bank_last={bank.local_expert_indices(tiny_experts - 1)}"
    )
    assert experts % SERVE_EP_DEGREE == 0 and per_group > 1
    assert bank.num_local_experts == 1
    assert all(bank.local_expert_indices(rank) == (rank,) for rank in range(tiny_experts))


def test_every_model_call_in_the_runner_goes_through_the_translation():
    """Every ``self.model(**...)`` and capture-backend call in the runner splats the translation."""
    control = _MODEL_CALL.search("        model_output = self.model(**kwargs)")
    assert control is not None and not control.group("splat").startswith(_TRANSLATED_SPLAT)
    calls = []
    for number, line in enumerate(_RUNNER_SOURCE.read_text().splitlines(), 1):
        found = _MODEL_CALL.search(line)
        if found:
            calls.append((number, found.group("splat")))
    bypassing = [number for number, splat in calls if not splat.startswith(_TRANSLATED_SPLAT)]
    print(
        f"PAR|CENSUS|model_calls={len(calls)} bypassing={len(bypassing)} "
        f"control_flagged={not control.group('splat').startswith(_TRANSLATED_SPLAT)} "
        f"lines={[number for number, _splat in calls]}"
    )
    assert len(calls) == RUNNER_MODEL_CALL_SITES, calls
    assert bypassing == []
