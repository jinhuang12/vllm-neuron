# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 DSpark drafter: every tensor rank traces one graph, because the rank is a tensor.

Both items run one world of two ranks over gloo, with the rank tensors stamped as the runner does.
The lowering item builds the proposer at TP 2 and DCP 2 under the device-free compile, as
``test_dspark_pages.py``'s trace item does, and stops the trace where the native backend turns
StableHLO into a NEFF. The two ranks' StableHLO texts, with their location records and dump paths
removed, must be byte-equal, and the location records must name every call of the drafter's rank
slice, so the texts are the drafter's. The values item checks that each rank's slice is its share
of the last dim, bit for bit, for 2-D and 3-D inputs.

Set ``KIMI_K3_DSPARK_CHECKPOINT`` to the draft directory and ``KIMI_K3_CHECKPOINT`` to the target
directory, as ``test_dspark_pages.py`` does; the items read their configs only. Run with
``VLLM_NEURON_CPU_MODE=1``.
"""

import importlib.util
import re
import traceback
from pathlib import Path

import pytest
import torch

import test_dspark_pages as pages

RANKS = pages.RANKS
# Location records: their definitions, their uses on ops, and the dump path of the module.
LOCATION_DEFINITIONS = re.compile(rb"^#loc\d* = loc\(.*\)\n?", re.M)
LOCATIONS = re.compile(rb" loc\((?:[^()]|\((?:[^()]|\([^()]*\))*\))*\)")
DUMP_PATH = re.compile(rb'debug_dump_path = "[^"]*"')


def _trace(config) -> None:
    """The proposer's load and warm-up under the device-free compile, stopped before the NEFF."""
    import torch._functorch.config as functorch_config
    from libtorch_neuronx_lite._compiler.neuron_dynamo_backend import backend
    from libtorch_neuronx_lite.compile.lite_compile_only import lite_compile_only
    from libtorch_neuronx_lite.compile.native_backend import native_precompile
    from vllm.config import set_current_vllm_config

    from vllm_neuron.vllm.spec_decode.dspark import DSparkProposer

    meta, columns = torch.device("meta"), pages.MAX_LEN // pages.BLOCK
    with pytest.MonkeyPatch.context() as patch, set_current_vllm_config(config), torch.no_grad():
        patch.setattr(functorch_config, "enable_autograd_cache", False)
        patch.setattr(backend, "_lower_stablehlo_to_neff", pages._stop_at_lowering)
        proposer = DSparkProposer(config, meta)
        proposer.load_model(target_hidden_size=config.model_config.hf_text_config.hidden_size)
        draft = proposer.model._orig_mod
        latent = draft.config.kv_lora_rank + draft.config.qk_rope_head_dim
        draft.bind_kv_cache({layer.cache_name: [torch.empty(columns + 1, 1, pages.BLOCK, latent,
                                                            dtype=torch.bfloat16, device=meta)]
                             for layer in draft.layers})
        proposer.alias_target_weights(pages._target_head(draft))
        table = torch.empty(1, columns, dtype=torch.int32, device=meta)
        metadata = {layer.cache_name: {"block_table_tensor": table, "block_size": pages.BLOCK,
                                       "max_blocks_per_seq": columns}
                    for layer in draft.layers}
        with lite_compile_only(), native_precompile():
            proposer.warmup(pages.TOKENS, 1, metadata)


def _lower_rank(rank: int, port: int, root: str) -> None:
    """One trace rank: the trace, or the error that stopped it."""
    import torch.distributed as dist
    from vllm import ModelRegistry

    from vllm_neuron.model import registry
    from vllm_neuron.parallel.neuron_parallel_state import (
        initialize_neuron_parallel_state,
        stamp_spmd_rank_tensors,
    )

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=RANKS)
    initialize_neuron_parallel_state(dcp_size=RANKS, tp_global_ranks=list(range(RANKS)),
                                     local_rank=rank)
    try:
        # The config comes first: vLLM checks the registered class when it builds one.
        config = pages._engine_config()
        for arch, model_cls in registry.get_models():
            ModelRegistry.register_model(arch, model_cls)
        stamp_spmd_rank_tensors("meta")  # the rank tensors of meta inputs
        _trace(config)
        out = {"stopped": False}
    except Exception as e:  # the reading names the stop
        out = {"stopped": isinstance(getattr(e, "inner_exception", None), pages._AtLowering),
               "error": traceback.format_exc(limit=-4)}
    torch.save(out, Path(root) / f"rank{rank}.pt")
    dist.destroy_process_group()


def _bare(text: bytes) -> bytes:
    """The StableHLO text without its location records and dump path."""
    return LOCATIONS.sub(b"", LOCATION_DEFINITIONS.sub(b"", DUMP_PATH.sub(b"", text)))


def _call_lines() -> set[int]:
    """The lines of dspark.py that call the rank slice."""
    package = importlib.util.find_spec("vllm_neuron").submodule_search_locations[0]
    source = (Path(package) / "model" / "kimi_k3" / "dspark.py").read_text().splitlines()
    return {n for n, line in enumerate(source, 1) if "self._rank_slice(" in line}


@pytest.mark.skipif(
    not ((pages.TARGET / "config.json").is_file() and (pages.DRAFT / "config.json").is_file()),
    reason=f"Kimi K3 config at {pages.TARGET} or DSpark config at {pages.DRAFT} not found",
)
def test_both_ranks_lower_the_same_draft_graph(tmp_path):
    if not pages._fits_the_window_kernel():
        pytest.skip(f"the DSpark config at {pages.DRAFT} has too many heads for the window kernel")
    caches = {"NEURON_LIBTORCH_CACHE_ROOT": "libtorch", "NKI_COMPILE_CACHE_URL": "nki",
              "NKI_TRACE_CACHE_URL": "nki-trace", "TORCHINDUCTOR_CACHE_DIR": "inductor"}
    venue = {**pages.VENUE, "TORCH_NEURONX_PRESERVE_COMPILATION_ARTIFACTS": "1",
             "TORCH_NEURONX_DEBUG_DIR": str(tmp_path / "debug"),
             **{name: str(tmp_path / sub) for name, sub in caches.items()}}
    for rank, reading in enumerate(pages._world(tmp_path, _lower_rank, (), venue)):
        assert reading["stopped"], (rank, reading.get("error", "the trace was not stopped"))
    texts = []
    for rank in range(RANKS):
        (path,) = (tmp_path / "debug").glob(f"torch_neuronx/rank{rank}/*/stablehlo/*.mlir")
        texts.append(path.read_bytes())
    first, second = (_bare(text).splitlines() for text in texts)
    differ = [(a, b) for a, b in zip(first, second) if a != b]
    assert first == second, (len(differ), differ[:1])
    named = {int(n) for n in re.findall(rb'kimi_k3/dspark\.py":(\d+):', texts[0])}
    calls = _call_lines()
    assert calls and calls <= named, (calls, sorted(named))


def _values_rank(rank: int, port: int, inputs: list, root: str) -> None:
    """One value rank: this rank's slice of each input."""
    import torch.distributed as dist

    from vllm_neuron.model.kimi_k3.dspark import K3DSparkForCausalLM
    from vllm_neuron.parallel.neuron_parallel_state import (
        initialize_neuron_parallel_state,
        stamp_spmd_rank_tensors,
    )

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=RANKS)
    initialize_neuron_parallel_state(tp_global_ranks=list(range(RANKS)), local_rank=rank)
    stamp_spmd_rank_tensors("cpu")
    config = pages._draft_config()
    with torch.device("meta"):
        draft = K3DSparkForCausalLM.from_configs(
            config=config, start_layer_idx=config["target_num_hidden_layers"],
            max_positions=pages.MAX_LEN,
        )
    torch.save([draft._rank_slice(x) for x in inputs], Path(root) / f"rank{rank}.pt")
    dist.destroy_process_group()


@pytest.mark.skipif(
    not (pages.DRAFT / "config.json").is_file(), reason=f"DSpark config not found at {pages.DRAFT}"
)
def test_each_rank_takes_its_share_of_the_last_dim_in_bits(tmp_path):
    g = torch.Generator().manual_seed(0)
    inputs = [torch.randn(5, 8, generator=g), torch.randn(2, 3, 8, generator=g)]
    world = pages._world(tmp_path, _values_rank, (inputs,), {"VLLM_RPC_BASE_PATH": "."})
    for rank, picks in enumerate(world):
        for x, got in zip(inputs, picks):
            width = x.shape[-1] // RANKS
            want = x[..., rank * width : (rank + 1) * width]
            assert got.shape == want.shape, (rank, got.shape, want.shape)
            assert torch.equal(got.view(torch.int32), want.contiguous().view(torch.int32)), rank
