# SPDX-License-Identifier: Apache-2.0
"""With prefix caching on, GLM-5.3-Flash hashes a block only if it ends inside its prompt.

In vLLM's "align" mode the KV cache manager hashes every full block, in a KDA group
and in the latent (MLA) group, a block that a decode step fills included, and a later
request whose prompt repeats the earlier prompt and output would hit it. A block that
ends at or before its request's prompt covers prompt tokens only, which a prefill
computes; a GLM-5.3-Flash hit is limited to such blocks.

``vllm_neuron.vllm.patches.prefix_cache_mask`` gives every group of the engine's
``KVCacheManager`` a block mask: vLLM's own mask AND "the block ends at or before
``request.num_prompt_tokens``". These tests step vLLM's own manager with the mask
installed by the product function, so a vLLM that stops calling the mask, or passes
it other arguments, fails them.

Each chain serves three requests on one manager:

* A: a prompt of ``P`` tokens, then 300 decode tokens;
* B: A's prompt, A's output and 300 new tokens, then 1 decode token;
* C: B's prompt and 100 new tokens, then 1 decode token.

Prefill legs follow vLLM's own ``Scheduler._mamba_block_aligned_split`` with a
1024-token budget, as the scheduler cuts them in "align" mode.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from test.vllm_neuron.worker import test_kv_budget_glm53f as kv

BLOCK = kv.BLOCK_SIZE_TOKENS
LEG_BUDGET = kv.MAX_NUM_BATCHED_TOKENS
MAX_MODEL_LEN = 8192

PROBES = [(657, 0), (657, 3), (2048, 0), (2048, 3)]
PROBE_IDS = ["P657", "P657-mtp3", "P2048", "P2048-mtp3"]

#: Hits of B and C with the mask. Each is the last prefill leg end that lies inside the
#: prompt of the request that cached the block; with MTP the scheduler ends the forced
#: leg one block earlier, so a hit can be one block shorter.
MASKED_HITS = {
    (657, 0): (640, 1152),
    (657, 3): (512, 1024),
    (2048, 0): (2048, 2560),
    (2048, 3): (2048, 2432),
}
#: B's hit without the mask: A's prompt and the blocks A's decode filled.
UNMASKED_B_HIT = {(657, 0): 896, (657, 3): 896, (2048, 0): 2304, (2048, 3): 2304}


class _Engine:
    """vLLM's own ``KVCacheManager`` in "align" mode over the GLM-5.3-Flash KV specs."""

    def __init__(self, draft_tokens: int) -> None:
        from vllm.utils.hashing import sha256
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.core.kv_cache_utils import (
            get_kv_cache_configs,
            get_request_block_hasher,
            init_none_hash,
            resolve_kv_cache_block_sizes,
        )
        from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec

        self.draft_tokens = draft_tokens
        eagle = bool(draft_tokens)
        layers, text_config = kv.glm53f_layer_specs()
        runner = kv.fake_runner(layers, max_num_seqs=2, max_model_len=MAX_MODEL_LEN,
                                gate_knobs=False, text_config=text_config)
        runner.vllm_config.cache_config.mamba_cache_mode = "align"
        if eagle:
            runner.speculative_config = SimpleNamespace(
                method="mtp", num_speculative_tokens=draft_tokens, use_eagle=lambda: True)
        worker = kv.fake_worker(runner)
        config = get_kv_cache_configs(runner.vllm_config, [runner.get_kv_cache_spec()],
                                      [worker._kv_cache_need_bytes(log=False)])[0]
        scheduler_block, hash_block = resolve_kv_cache_block_sizes(config, runner.vllm_config)
        self.manager = KVCacheManager(
            kv_cache_config=config, max_model_len=MAX_MODEL_LEN,
            scheduler_block_size=scheduler_block, hash_block_size=hash_block,
            max_num_batched_tokens=LEG_BUDGET, enable_caching=True, use_eagle=eagle)
        groups = config.kv_cache_groups
        self.kda = [i for i, g in enumerate(groups) if isinstance(g.kv_cache_spec, MambaSpec)]
        self.latent = [i for i, g in enumerate(groups)
                       if isinstance(g.kv_cache_spec, MLAAttentionSpec)]
        self.groups = self.kda + self.latent
        assert len(self.kda) == 4 and len(self.latent) == 1 and len(self.groups) == len(groups)
        init_none_hash(sha256)
        self._hasher = get_request_block_hasher(hash_block, sha256)
        self._split = SimpleNamespace(cache_config=SimpleNamespace(block_size=BLOCK),
                                      use_eagle=eagle)
        #: (group, block id, block hash) -> (request id, block index, prompt tokens)
        self.hashed: dict[tuple, tuple[str, int, int]] = {}

    def _note(self, request) -> None:
        managers = self.manager.coordinator.single_type_managers
        for group in self.groups:
            for index, block in enumerate(managers[group].req_to_blocks[request.request_id]):
                if block.is_null or block.block_hash is None:
                    continue
                key = (group, block.block_id, block.block_hash)
                self.hashed.setdefault(
                    key, (request.request_id, index, request.num_prompt_tokens))

    def serve(self, request_id: str, prompt: list[int], decode: int):
        """Serve one request to completion; return its hit, hit blocks and output ids."""
        from vllm.sampling_params import SamplingParams
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.request import Request

        request = Request(request_id=request_id, prompt_token_ids=prompt,
                          sampling_params=SamplingParams(max_tokens=decode),
                          pooling_params=None, block_hasher=self._hasher)
        hit_blocks, hit = self.manager.get_computed_blocks(request)
        first = True
        while request.num_computed_tokens < len(prompt):
            computed = request.num_computed_tokens + (hit if first else 0)
            tokens = Scheduler._mamba_block_aligned_split(
                self._split, request, min(LEG_BUDGET, len(prompt) - computed),
                hit if first else 0)
            extra = ({"num_new_computed_tokens": hit, "new_computed_blocks": hit_blocks}
                     if first else {})
            self.manager.new_step_starts()
            assert self.manager.allocate_slots(
                request, tokens, num_lookahead_tokens=self.draft_tokens, **extra) is not None
            request.num_computed_tokens += tokens + (hit if first else 0)
            self._note(request)
            first = False
        for token in range(decode):
            request.append_output_token_ids(token)
            self.manager.new_step_starts()
            assert self.manager.allocate_slots(
                request, 1, num_lookahead_tokens=self.draft_tokens) is not None
            request.num_computed_tokens += 1
            self._note(request)
        output = list(request.output_token_ids)
        self.manager.free(request)
        return hit, hit_blocks, output

    def chain(self, prompt_len: int):
        """Serve A, B and C; return the three hits and C's hit blocks."""
        prompt_a = list(range(prompt_len))
        hit_a, _, out_a = self.serve("A", prompt_a, 300)
        prompt_b = prompt_a + out_a + list(range(10_000, 10_300))
        hit_b, _, _ = self.serve("B", prompt_b, 1)
        hit_c, blocks_c, _ = self.serve("C", prompt_b + list(range(20_000, 20_100)), 1)
        return (hit_a, hit_b, hit_c), blocks_c


def _masked_engine(draft_tokens: int) -> _Engine:
    from vllm_neuron.vllm.patches import prefix_cache_mask

    engine = _Engine(draft_tokens)
    prefix_cache_mask.install(engine.manager)
    return engine


# ---------------------------------------------------------------------------
# Condition 1: a change of upstream's parameter names is refused by name
# ---------------------------------------------------------------------------


def test_the_mask_targets_match_this_vllm() -> None:
    from vllm_neuron.vllm.patches import prefix_cache_mask

    prefix_cache_mask.check_targets()


@pytest.mark.parametrize("manager_class", ["MambaManager", "FullAttentionManager"])
def test_a_renamed_mask_parameter_is_refused_by_name(monkeypatch, manager_class) -> None:
    from vllm.v1.core import single_type_kv_cache_manager as managers

    from vllm_neuron.vllm.patches import prefix_cache_mask

    engine = _Engine(0)

    def renamed(cls, start_block, end_block, alignment_tokens, kv_cache_spec, use_eagle,
                retention_interval=None, prompt_tokens=None):
        return None

    monkeypatch.setattr(getattr(managers, manager_class), "reachable_block_mask",
                        classmethod(renamed))
    with pytest.raises(prefix_cache_mask.PrefixCacheMaskTargetError,
                       match=rf"{manager_class}\.reachable_block_mask.*prompt_tokens"):
        prefix_cache_mask.install(engine.manager)
    # Refused before any manager was touched.
    for manager in engine.manager.coordinator.single_type_managers:
        assert "reachable_block_mask" not in vars(manager)


# ---------------------------------------------------------------------------
# Condition 2: vLLM's own manager hashes no block past its prompt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("prompt_len", "draft_tokens"), PROBES, ids=PROBE_IDS)
def test_vllms_manager_hashes_no_block_that_ends_past_its_prompt(prompt_len, draft_tokens):
    engine = _masked_engine(draft_tokens)
    (hit_a, hit_b, hit_c), blocks_c = engine.chain(prompt_len)

    assert hit_a == 0
    assert (hit_b, hit_c) == MASKED_HITS[(prompt_len, draft_tokens)]
    # Every group hashed blocks, and none of them ends past its own request's prompt.
    assert {group for group, _, _ in engine.hashed} == set(engine.groups)
    for (group, _, _), (request_id, index, prompt_tokens) in engine.hashed.items():
        assert (index + 1) * BLOCK <= prompt_tokens, (group, request_id, index)
    # Each latent block of C's hit was filled by a prefill.
    (latent,) = engine.latent
    hit_latent = [block for block in blocks_c.blocks[latent] if not block.is_null]
    assert len(hit_latent) == hit_c // BLOCK
    for block in hit_latent:
        _, index, prompt_tokens = engine.hashed[(latent, block.block_id, block.block_hash)]
        assert (index + 1) * BLOCK <= prompt_tokens


# ---------------------------------------------------------------------------
# Condition 3: the hit loss
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("prompt_len", "draft_tokens"), PROBES, ids=PROBE_IDS)
def test_a_later_turn_reuses_the_earlier_prompt_not_the_earlier_output(prompt_len, draft_tokens):
    """B repeats A's prompt and output. Its hit ends inside A's prompt, at the last
    prefill leg end there, so B prefills A's output again. Without the mask the hit
    also covers the blocks A's decode filled."""
    masked, _ = _masked_engine(draft_tokens).chain(prompt_len)
    unmasked, _ = _Engine(draft_tokens).chain(prompt_len)

    assert masked[1] == MASKED_HITS[(prompt_len, draft_tokens)][0]
    assert masked[1] <= prompt_len < unmasked[1]
    assert unmasked[1] == UNMASKED_B_HIT[(prompt_len, draft_tokens)]


# ---------------------------------------------------------------------------
# Scope: this model, prefix caching on, this engine's manager
# ---------------------------------------------------------------------------


def test_the_install_reaches_one_managers_groups_and_not_vllms_classes() -> None:
    from vllm.v1.core import single_type_kv_cache_manager as managers

    engine = _masked_engine(0)
    other = _Engine(0)

    for manager in engine.manager.coordinator.single_type_managers:
        assert "reachable_block_mask" in vars(manager)
    for manager in other.manager.coordinator.single_type_managers:
        assert "reachable_block_mask" not in vars(manager)
    assert "reachable_block_mask" not in vars(managers.FullAttentionManager)
    assert isinstance(vars(managers.MambaManager)["reachable_block_mask"], classmethod)


class _Stop(Exception):
    """Ends a scheduler ``__init__`` right after the step under test."""


MASK_MODULE = "vllm_neuron.vllm.patches.prefix_cache_mask"


@pytest.mark.parametrize("scheduler_class", ["NeuronScheduler", "NeuronAsyncScheduler"])
@pytest.mark.parametrize(
    ("architectures", "prefix_caching", "installs"),
    [
        (["Glm5NextForConditionalGeneration"], True, True),
        (["Glm5NextForConditionalGeneration"], False, False),
        (["Qwen3NextForCausalLM"], True, False),
        ([], True, False),
    ],
    ids=["glm5next-on", "glm5next-off", "other-model-on", "no-architecture-on"],
)
def test_the_neuron_schedulers_install_the_mask_on_this_model_with_prefix_caching_only(
    monkeypatch, scheduler_class, architectures, prefix_caching, installs
) -> None:
    """The schedulers install the mask on their own manager for this model with prefix
    caching on. Any other engine does not even import the mask module."""
    from vllm.v1.core.sched.scheduler import Scheduler

    from vllm_neuron.vllm import patches
    from vllm_neuron.vllm.core import scheduler as neuron_scheduler
    from vllm_neuron.vllm.patches import prefix_cache_mask

    manager = object()
    installed: list[object] = []

    def base_init(self, **_kwargs) -> None:
        self.kv_cache_manager = manager
        self.num_spec_tokens = 0
        self.parallel_config = SimpleNamespace(pipeline_parallel_size=1)

    def stop(self, vllm_config) -> None:
        raise _Stop

    monkeypatch.setattr(Scheduler, "__init__", base_init)
    monkeypatch.setattr(neuron_scheduler.NeuronScheduler,
                        "_cap_encoder_budget_to_vision_bucket", stop)
    if installs:
        monkeypatch.setattr(prefix_cache_mask, "install", installed.append)
    else:
        monkeypatch.delitem(sys.modules, MASK_MODULE)
        monkeypatch.delattr(patches, "prefix_cache_mask")
    config = SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN, model="model",
                                     architectures=architectures),
    )

    with pytest.raises(_Stop):
        getattr(neuron_scheduler, scheduler_class)(
            vllm_config=config, kv_cache_config=None, structured_output_manager=None,
            block_size=BLOCK)

    assert installed == ([manager] if installs else [])
    assert (MASK_MODULE in sys.modules) is installs


def test_importing_the_neuron_schedulers_does_not_import_the_mask_module() -> None:
    """A fresh interpreter that imports the schedulers holds no mask module, so an
    engine the mask does not apply to never loads it."""
    child = (
        "import sys\n"
        "from vllm_neuron.vllm.core.scheduler import NeuronAsyncScheduler, NeuronScheduler\n"
        f"print('MASK_MODULE_LOADED=' + str({MASK_MODULE!r} in sys.modules))\n"
    )
    completed = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True,
                               timeout=300)

    assert completed.returncode == 0, completed.stderr[-3000:]
    assert "MASK_MODULE_LOADED=False" in completed.stdout.splitlines(), completed.stdout[-3000:]
