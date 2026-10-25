# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 built by the model runner from the checkpoint's own config, its vision section kept.

The world is ``test_dspark_proposer.py``'s, read from that file so the two cannot drift: the
target's config cut to 4 layers by the same override, DSpark with 4 drafts on the checkpoint's
draft, TP 2 over gloo, one thread per rank, and the same stand-ins and seeded random weight
load. Here the platform's encoder config for the vision section stays, and the text-only flag
is not set. The runner's ``load_model`` then builds the image-enabled model: the vision tower's
class, with the runner's encoder config, and the runner's neuron config in its text config.

Set ``KIMI_K3_CHECKPOINT`` and ``KIMI_K3_DSPARK_CHECKPOINT`` as that file reads them. Run with
``VLLM_NEURON_CPU_MODE=1``.
"""

import importlib.util
import os
import socket
import traceback
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

_spec = importlib.util.spec_from_file_location(
    "dspark_world", Path(__file__).with_name("test_dspark_proposer.py")
)
proposer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(proposer)

pytestmark = proposer.pytestmark

TP = proposer.TP
VISION_CLASS = "vllm_neuron.model.kimi_k3.vision_encoder.KimiK3VisionForConditionalGeneration"


def _engine_config():
    """The proposer test's DSpark engine config, with the encoder config the platform gives."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform

    spec = {"method": "dspark", "model": str(proposer.DRAFT),
            "num_speculative_tokens": proposer.K}
    config = EngineArgs(
        model=str(proposer.TARGET), tensor_parallel_size=TP, max_model_len=proposer.MAX_LEN,
        max_num_seqs=proposer.SEQS, enforce_eager=True, skip_tokenizer_init=True,
        enable_prefix_caching=False, hf_overrides=proposer._cut, speculative_config=spec,
        async_scheduling=False, additional_config={"neuron_config": dict(proposer.NEURON)},
    ).create_engine_config()
    current_platform.update_block_size_for_backend(config)
    config.speculative_config.draft_model_config.hf_config.target_layer_ids = [0, 1, 2]
    return config


def _rank(rank: int, port: int, root: str) -> None:
    """One TP rank: the runner's load, then what it built, or the error it raised."""
    import torch.distributed as dist
    from vllm import ModelRegistry
    from vllm.config import set_current_vllm_config

    from vllm_neuron.model import registry
    from vllm_neuron.parallel.neuron_parallel_state import initialize_neuron_parallel_state
    from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=TP)
    initialize_neuron_parallel_state(tp_global_ranks=list(range(TP)), local_rank=rank)
    config = _engine_config()
    for arch, model_cls in registry.get_models():
        ModelRegistry.register_model(arch, model_cls)
    out = {"encoder_config": config.additional_config.get("encoder_neuron_config"),
           "error": None}
    with pytest.MonkeyPatch.context() as patch, torch.no_grad(), set_current_vllm_config(config):
        proposer._stand_ins(patch)
        try:
            runner = NeuronModelRunner(config, torch.device("cpu"))
            out["runner_has_encoder_config"] = runner.encoder_neuron_config is not None
            runner.load_model()
        except Exception as error:
            frames = traceback.extract_tb(error.__traceback__)[-2:]
            out["error"] = (f"{type(error).__name__}: {error} (at "
                            + ", ".join(f"{Path(f.filename).name}:{f.lineno} in {f.name}"
                                        for f in frames) + ")")
        else:
            model = getattr(runner.model, "_orig_mod", runner.model)
            out.update(
                model_class=f"{type(model).__module__}.{type(model).__qualname__}",
                encoder_config_is_the_runners=runner.encoder_neuron_config is not None
                and getattr(model, "encoder_neuron_config", None) is runner.encoder_neuron_config,
                neuron_config_is_the_runners=model.text_config.neuron_config
                is runner.neuron_config,
            )
    torch.save(out, Path(root) / f"rank{rank}.pt")
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def loads(tmp_path_factory) -> list[dict]:
    """Each rank's readings from one two-rank load."""
    root = tmp_path_factory.mktemp("vision_load")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    saved, cwd = os.environ.get("VLLM_RPC_BASE_PATH"), os.getcwd()
    # vLLM's broadcaster binds an IPC socket under this path, which must stay short.
    os.environ["VLLM_RPC_BASE_PATH"] = "."
    os.chdir(root)
    try:
        mp.spawn(_rank, args=(port, str(root)), nprocs=TP, join=True)
    finally:
        os.chdir(cwd)
        if saved is None:
            os.environ.pop("VLLM_RPC_BASE_PATH")
        else:
            os.environ["VLLM_RPC_BASE_PATH"] = saved
    return [torch.load(root / f"rank{r}.pt", weights_only=False) for r in range(TP)]


def test_runner_builds_the_vision_model_from_the_checkpoint_config(loads):
    """The runner builds the image-enabled model from both configs, with no error."""
    for rank, out in enumerate(loads):
        print(f"rank {rank}: {out}")
    given = [out["encoder_config"] is not None and out.get("runner_has_encoder_config", False)
             for out in loads]
    assert all(given), f"the platform gave no encoder config on some rank: {given}"
    errors = [f"rank {rank}: {out['error']}" for rank, out in enumerate(loads) if out["error"]]
    assert errors == [], errors
    for rank, out in enumerate(loads):
        assert out["model_class"] == VISION_CLASS, (rank, out["model_class"])
        assert out["encoder_config_is_the_runners"], f"rank {rank}: not the runner's encoder config"
        assert out["neuron_config_is_the_runners"], f"rank {rank}: not the runner's neuron config"
