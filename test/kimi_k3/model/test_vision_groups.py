# SPDX-License-Identifier: Apache-2.0
"""Encoder rank groups at world 64: text TP32 by DP2, encoder TP4.

Each of the 64 ranks runs the worker's distributed init through vLLM's world
and model-parallel setup and the Neuron group build. Only process-group
creation is replaced: the stand-ins record every rank list a group is built
from. A text replica is a text TP group as vLLM builds it (PP is 1).
"""

from types import SimpleNamespace

import torch.distributed as dist
import vllm.distributed.parallel_state as vllm_ps
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config

import vllm_neuron.parallel.neuron_parallel_state as neuron_ps
from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

TEXT_TP, TEXT_DP, ENCODER_TP = 32, 2, 4
WORLD = TEXT_TP * TEXT_DP
ENCODER_GROUPS = ("neuron_encoder_tp", "neuron_encoder_dp")
VLLM_STATE = ("_WORLD", "_TP", "_DCP", "_PCP", "_PP", "_DP", "_EP", "_EPLB")


def _init_rank(monkeypatch, config, rank, built):
    """Run one worker's distributed init; add each group's rank lists to built."""
    pg = {}

    def group(*args, **kwargs):
        ranks = kwargs.get("group_ranks", args[0] if args else None)
        built.setdefault(kwargs.get("group_name"), set()).update(map(tuple, ranks))
        mine = next(list(r) for r in ranks if pg["rank"] in r)
        return SimpleNamespace(
            ranks=mine,
            world_size=len(mine),
            rank_in_group=mine.index(pg["rank"]),
            local_rank=rank,
            backend="gloo",
            cpu_group=None,
            device_group=None,
        )

    def init_process_group(*args, world_size, rank, **kwargs):
        pg.update(world_size=world_size, rank=rank)

    for name in VLLM_STATE + ("_INNER_DP_WORLD", "_NODE_COUNT"):
        monkeypatch.setattr(vllm_ps, name, None)
    monkeypatch.setattr(dist, "is_initialized", lambda: bool(pg))
    monkeypatch.setattr(dist, "init_process_group", init_process_group)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: pg["world_size"])
    monkeypatch.setattr(dist, "get_rank", lambda group=None: pg["rank"])
    monkeypatch.setattr(vllm_ps, "_node_count", lambda group: 1)
    for module in (vllm_ps, neuron_ps):
        monkeypatch.setattr(module, "init_model_parallel_group", group)
        monkeypatch.setattr(
            module, "init_world_group", lambda r, lr, b: group([r], group_name="world")
        )

    worker = NeuronWorker.__new__(NeuronWorker)
    worker.vllm_config = config
    with set_current_vllm_config(config):
        worker._init_neuron_distributed_environment_and_runtime(
            config, rank, "tcp://127.0.0.1:29500", rank
        )


def test_no_encoder_group_crosses_a_text_replica(monkeypatch):
    """Every encoder TP and DP group lies inside one text replica's 32 ranks."""
    for name in vars(neuron_ps):
        if name.startswith(("_NEURON_", "_SAMPLING_", "_WIDE_EP")):
            monkeypatch.setattr(neuron_ps, name, getattr(neuron_ps, name))
    monkeypatch.setattr(vllm_ps, "in_the_same_node_as", vllm_ps.in_the_same_node_as)
    monkeypatch.setattr(neuron_ps, "_patch_getters", lambda: None)
    monkeypatch.setattr(neuron_ps, "_patch_destroy", lambda: None)

    built = {}
    for dp_rank in range(TEXT_DP):
        parallel = ParallelConfig(
            tensor_parallel_size=TEXT_TP,
            data_parallel_size=TEXT_DP,
            data_parallel_size_local=TEXT_DP,
            data_parallel_rank=dp_rank,
        )
        config = VllmConfig(
            parallel_config=parallel,
            additional_config={"encoder_neuron_config": {"tp_size": ENCODER_TP}},
        )
        for rank in range(TEXT_TP):
            _init_rank(monkeypatch, config, rank, built)

    replicas = [set(r) for r in built["tp"]]
    for name in ENCODER_GROUPS:
        covered = set().union(*built.get(name, [()]))
        assert covered == set(range(WORLD)), f"{name} covers {sorted(covered)}"
    crossing = sorted(
        (name, g)
        for name in ENCODER_GROUPS
        for g in built[name]
        if not any(set(g) <= r for r in replicas)
    )
    assert crossing == [], f"{len(crossing)} groups cross replicas: {crossing[:2]}"
