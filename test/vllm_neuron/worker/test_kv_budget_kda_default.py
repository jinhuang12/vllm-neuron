# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash defaults to prefix caching off and one KDA block per request.

With prefix caching on (vLLM's default for this model: upstream does not flag it
hybrid, so ``HybridAttentionMambaModelConfig`` never runs) each recurrent (KDA)
group keeps the 128-token attention block, and vLLM's mamba "none" mode reserves
``cdiv(max_model_len, 128)`` of them per request in every one of the four KDA
groups. The runner keeps the KDA state in a bank addressed by request slot, so
those blocks hold no data, and it refuses a prefix-cache hit on a recurrent stack
(``kv_spec_patch.recurrent_spec_block_size``). At 1 x 1M that is 56.38 GiB per rank
for 12.38 GiB of real need.

``NeuronPlatform.check_and_update_config`` therefore sets, for this architecture:

* ``enable_prefix_caching=False`` unless ``--enable-prefix-caching`` or
  ``--no-enable-prefix-caching`` is on the command line, on every form;
* on the hybrid KV cache, ``mamba_block_size=max_model_len`` (upstream's own
  default for a hybrid model with prefix caching off) unless a mamba block size is
  set, or prefix caching is on, where vLLM's hybrid coordinator needs the 128-token
  block.

Each default yields to its own flag. One log line always says what was applied and
why. An explicit ``--enable-prefix-caching`` is then refused at startup, on every
form: the runner has no restore of a request's KDA state and DSA side caches for a
cached prefix.
With that refusal set aside, the worker prices every KDA block and one log line
names that footprint next to the one-block footprint.
"""

from __future__ import annotations

import logging
import math

import pytest

from test.vllm_neuron import test_platform_hybrid_config as hybrid
from test.vllm_neuron.worker import test_kv_budget_glm53f as kv
from vllm_neuron.vllm.platform import NeuronPlatform

GIB = 1024**3
ONE_M = 1024 * 1024
#: The worker's footprint at 1 x 1M before and after the default.
ONE_M_PC_ON_FOOTPRINT_GIB = 56.38
ONE_M_ONE_BLOCK_FOOTPRINT_GIB = 12.38
#: The 1 x 1M point with prefix caching on in vLLM's "align" KDA cache mode.
ONE_M_ALIGN_FOOTPRINT_GIB = 12.389

DEFAULTS_MARKER = "Glm5NextForConditionalGeneration KV cache defaults"
PC_FLAG_ON = "--enable-prefix-caching"
PC_FLAG_OFF = "--no-enable-prefix-caching"
MAMBA_FLAG = "--mamba-block-size"


@pytest.fixture(autouse=True)
def _device_path(monkeypatch):
    for name in (
        "VLLM_NEURON_CPU_MODE",
        "VLLM_NEURON_CPU_COMPILE",
        "VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB",
        "VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="module")
def model_specs():
    return kv.glm53f_layer_specs()


def _resolve(monkeypatch, argv: list[str], *, max_model_len: int = 8192, **build):
    """Run ``check_and_update_config`` as ``vllm serve <argv>`` would reach it.

    vLLM's ``EngineArgs`` has already turned the flags into the cache config by
    then; the config is built to match: ``--no-enable-prefix-caching`` leaves
    prefix caching off and ``--mamba-block-size N`` leaves N.
    """
    monkeypatch.setattr("sys.argv", ["vllm", "serve", "model", *argv])
    cfg = hybrid._build_config(max_model_len=max_model_len, **build)
    if PC_FLAG_OFF in argv:
        cfg.cache_config.enable_prefix_caching = False
    for index, arg in enumerate(argv):
        if arg == MAMBA_FLAG:
            cfg.cache_config.mamba_block_size = int(argv[index + 1])
        elif arg.startswith(MAMBA_FLAG + "="):
            cfg.cache_config.mamba_block_size = int(arg.split("=", 1)[1])
    with hybrid._isolated_platform_class_state(), hybrid._capture_platform_log() as handler:
        NeuronPlatform.check_and_update_config(cfg)
    return cfg, hybrid._marker_records(handler, DEFAULTS_MARKER)


def _resolve_past_the_refusal(monkeypatch, argv: list[str], **kwargs):
    """``_resolve`` with the startup refusal of prefix caching set aside.

    The platform refuses an explicit ``--enable-prefix-caching`` for this model
    (``NeuronPlatform._refuse_glm5next_prefix_caching``). The worker still prices the
    configuration the default leaves under that flag, so these tests reach it this way.
    """
    monkeypatch.setattr(
        NeuronPlatform,
        "_refuse_glm5next_prefix_caching",
        classmethod(lambda cls, vllm_config: None),
        raising=False,
    )
    return _resolve(monkeypatch, argv, **kwargs)


# ---------------------------------------------------------------------------
# The platform default
# ---------------------------------------------------------------------------


def test_without_either_flag_prefix_caching_is_off_and_a_kda_block_spans_the_sequence(
    monkeypatch,
) -> None:
    cfg, records = _resolve(monkeypatch, ["--max-model-len", "8192"])

    assert cfg.cache_config.enable_prefix_caching is False
    assert cfg.cache_config.mamba_block_size == 8192
    assert len(records) == 1
    line = records[0]
    assert "prefix caching defaulted off" in line
    assert "no --enable-prefix-caching given" in line
    assert "mamba_block_size defaulted to max_model_len=8192" in line
    assert "no --mamba-block-size given" in line


@pytest.mark.parametrize(
    "build",
    [{}, {"tp": hybrid.UNSUPPORTED_TP_DEGREE}],
    ids=["hybrid-kv-cache", "no-hybrid-kv-cache"],
)
@pytest.mark.parametrize("spelling", [PC_FLAG_ON, "--enable_prefix_caching"])
def test_an_explicit_prefix_caching_on_is_refused_at_startup(
    monkeypatch, spelling, build
) -> None:
    """The refusal comes from the platform's config hook, before an engine or a device exists.

    The runner refuses a prefix-cache hit on every form of this architecture, with or
    without the hybrid KV cache, so the refusal covers both. It names the flag, the
    state that has no restore, and the way to serve.
    """
    with pytest.raises(ValueError) as refused:
        _resolve(monkeypatch, [spelling], **build)

    message = str(refused.value)
    assert "--enable-prefix-caching" in message
    assert "KDA recurrent state" in message
    assert "DSA indexer side caches" in message
    assert "--no-enable-prefix-caching" in message


def test_the_refusal_follows_the_default_that_keeps_the_128_token_kda_block(
    monkeypatch,
) -> None:
    """With the refusal set aside, the default still keeps an explicit flag and the attention block.

    The worker prices that configuration (below), so the default's own outcome stays pinned.
    """
    cfg, records = _resolve_past_the_refusal(monkeypatch, [PC_FLAG_ON])

    assert cfg.cache_config.enable_prefix_caching is True
    assert cfg.cache_config.mamba_block_size is None
    assert cfg.cache_config.block_size == hybrid.HYBRID_BLOCK_SIZE
    assert len(records) == 1
    assert "prefix caching on as given" in records[0]
    assert "cdiv(8192, 128) = 64 blocks per request" in records[0]


def test_an_explicit_prefix_caching_off_still_gets_the_one_block_default(
    monkeypatch,
) -> None:
    """The two defaults are independent: only the mamba flag skips the mamba one."""
    cfg, records = _resolve(monkeypatch, [PC_FLAG_OFF])

    assert cfg.cache_config.enable_prefix_caching is False
    assert cfg.cache_config.mamba_block_size == 8192
    assert len(records) == 1
    assert "prefix caching off as given" in records[0]
    assert "mamba_block_size defaulted to max_model_len=8192" in records[0]


@pytest.mark.parametrize(
    "argv",
    [[MAMBA_FLAG, "8192"], [MAMBA_FLAG + "=8192"]],
    ids=["space", "equals"],
)
def test_an_explicit_mamba_block_size_is_kept(monkeypatch, argv) -> None:
    cfg, records = _resolve(monkeypatch, argv)

    assert cfg.cache_config.mamba_block_size == 8192
    assert cfg.cache_config.enable_prefix_caching is False
    assert len(records) == 1
    assert "mamba_block_size=8192 as given" in records[0]


def test_the_bs64_line_resolves_byte_identically(monkeypatch) -> None:
    """``--no-enable-prefix-caching --mamba-block-size 8192``: the default moves nothing.

    The control is the same call with the default removed, which is the config
    82bee3b resolved.
    """
    argv = [PC_FLAG_OFF, MAMBA_FLAG, "8192", "--max-num-seqs", "64"]
    with_default, records = _resolve(monkeypatch, argv)
    monkeypatch.setattr(
        NeuronPlatform,
        "_default_glm5next_recurrent_blocks",
        classmethod(lambda cls, vllm_config: None),
    )
    without_default, _ = _resolve(monkeypatch, argv)

    assert hybrid._cache_snapshot(with_default) == hybrid._cache_snapshot(without_default)
    assert len(records) == 1
    assert "prefix caching off as given" in records[0]
    assert "mamba_block_size=8192 as given" in records[0]


def test_other_architectures_keep_vllms_defaults(monkeypatch) -> None:
    """Even on the hybrid KV cache, another architecture keeps vLLM's settings."""
    other = hybrid._other_archs()[0]
    cfg, records = _resolve(
        monkeypatch, [], arch=other, neuron_config={"enable_hybrid_kv_cache": True}
    )

    assert cfg.cache_config.enable_prefix_caching is True
    assert cfg.cache_config.mamba_block_size is None
    assert records == []


def test_without_the_hybrid_cache_prefix_caching_is_still_defaulted_off(monkeypatch) -> None:
    """GLM at a degree without the hybrid KV cache: prefix caching off, mamba block kept.

    The runner refuses a prefix-cache hit there too, so the prefix-caching default holds.
    The one-block mamba default is the hybrid KV cache's, so it is not applied.
    """
    cfg, records = _resolve(monkeypatch, [], tp=hybrid.UNSUPPORTED_TP_DEGREE)

    assert cfg.cache_config.enable_prefix_caching is False
    assert cfg.cache_config.mamba_block_size is None
    assert len(records) == 1
    assert "prefix caching defaulted off" in records[0]
    assert "mamba_block_size left unset (the hybrid KV cache is off)" in records[0]


# ---------------------------------------------------------------------------
# What the default buys: pricing on both prefix-caching paths
# ---------------------------------------------------------------------------


def _priced_worker(model_specs, cfg, *, seqs: int, length: int):
    """The worker fake at ``seqs x length``, carrying the cache config ``cfg`` resolved."""
    layers, text_config = model_specs
    runner = kv.fake_runner(layers, max_num_seqs=seqs, max_model_len=length, text_config=text_config)
    cache = runner.vllm_config.cache_config
    cache.enable_prefix_caching = cfg.cache_config.enable_prefix_caching
    cache.mamba_block_size = cfg.cache_config.mamba_block_size
    return kv.fake_worker(
        runner,
        param_bytes=kv.TIP_PARAM_BYTES,
        resident_bytes=kv.TIP_RESIDENT_BYTES,
        runtime_used_bytes=int(7.06 * GIB),
    )


def _kda_slot_bytes(worker) -> list[int]:
    from vllm.v1.kv_cache_interface import MambaSpec

    from vllm_neuron.vllm.patches.kv_spec_patch import recurrent_state_slot_bytes

    return [
        recurrent_state_slot_bytes(spec)
        for spec in worker.model_runner.get_kv_cache_spec().values()
        if isinstance(spec, MambaSpec)
    ]


def test_one_sequence_at_1m_needs_its_attention_pages_plus_one_kda_state_per_layer(
    model_specs, monkeypatch
) -> None:
    """The default's 1 x 1M point: 12.38 GiB per rank, from 56.38 GiB.

    Pool: one latent layer per pool tensor (11), each holding the request's
    ``cdiv(L, 128)`` attention pages, one block per KDA group (4) and the null
    block. Banks: one KDA state per KDA layer (34). Plus the DSA side caches.
    """
    from vllm_neuron.vllm.worker.neuron_worker import _indexer_side_cache_bytes

    cfg, _ = _resolve(monkeypatch, ["--max-model-len", str(ONE_M)], max_model_len=ONE_M)
    worker = _priced_worker(model_specs, cfg, seqs=1, length=ONE_M)
    page = kv.BLOCK_SIZE_TOKENS * kv.LATENT_BYTES_PER_TOKEN
    kda_groups = math.ceil(kv.KDA_LAYERS / kv.MLA_LAYERS)
    attention_pages = math.ceil(ONE_M / kv.BLOCK_SIZE_TOKENS)

    need = worker._kv_cache_need_bytes(log=False)
    states = _kda_slot_bytes(worker)
    footprint = worker._kv_cache_footprint_bytes(need)

    assert need == kv.MLA_LAYERS * page * (attention_pages + kda_groups * 1 + 1)
    assert len(states) == kv.KDA_LAYERS
    assert sum(worker._kv_cache_allocation_sizes(need)) == need + sum(states)
    assert footprint == need + sum(states) + _indexer_side_cache_bytes(
        worker.vllm_config, worker.model_runner
    )
    assert round(footprint / GIB, 2) == ONE_M_ONE_BLOCK_FOOTPRINT_GIB


def test_prefix_caching_on_prices_every_kda_block(model_specs, monkeypatch) -> None:
    """Explicit prefix caching at 1 x 1M: each KDA group reserves cdiv(L, 128) blocks."""
    cfg, _ = _resolve_past_the_refusal(monkeypatch, [PC_FLAG_ON], max_model_len=ONE_M)
    worker = _priced_worker(model_specs, cfg, seqs=1, length=ONE_M)
    page = kv.BLOCK_SIZE_TOKENS * kv.LATENT_BYTES_PER_TOKEN
    kda_groups = math.ceil(kv.KDA_LAYERS / kv.MLA_LAYERS)
    pages = math.ceil(ONE_M / kv.BLOCK_SIZE_TOKENS)

    need = worker._kv_cache_need_bytes(log=False)

    assert need == kv.MLA_LAYERS * page * (pages * (1 + kda_groups) + 1)
    assert round(worker._kv_cache_footprint_bytes(need) / GIB, 2) == ONE_M_PC_ON_FOOTPRINT_GIB


def _in_cache_mode(worker, mode: str, *, draft_blocks: int = 0):
    """The worker, its runner reporting every KDA spec in vLLM's ``mode`` cache mode."""
    import dataclasses

    from vllm.v1.kv_cache_interface import MambaSpec

    specs = {
        name: dataclasses.replace(spec, mamba_cache_mode=mode, num_speculative_blocks=draft_blocks)
        if isinstance(spec, MambaSpec)
        else spec
        for name, spec in worker.model_runner.get_kv_cache_spec().items()
    }
    worker.model_runner.get_kv_cache_spec = lambda: specs
    return worker


@pytest.mark.parametrize("draft_blocks", [0, 3], ids=["plain", "mtp-3"])
def test_align_mode_prices_two_kda_blocks_and_the_draft_blocks_per_request(
    model_specs, monkeypatch, draft_blocks
) -> None:
    """vLLM's "align" mode keeps a request's running KDA block, the block before it and
    its draft blocks in each KDA group (``MambaSpec.max_memory_usage_bytes``), however
    long the sequence. At 1 x 1M that is the one-block footprint plus one block per KDA
    group; the per-slot banks and the side caches do not change."""
    cfg, _ = _resolve_past_the_refusal(monkeypatch, [PC_FLAG_ON], max_model_len=ONE_M)
    worker = _in_cache_mode(_priced_worker(model_specs, cfg, seqs=1, length=ONE_M), "align",
                            draft_blocks=draft_blocks)
    page = kv.BLOCK_SIZE_TOKENS * kv.LATENT_BYTES_PER_TOKEN
    kda_groups = math.ceil(kv.KDA_LAYERS / kv.MLA_LAYERS)
    pages = math.ceil(ONE_M / kv.BLOCK_SIZE_TOKENS)

    need = worker._kv_cache_need_bytes(log=False)

    assert need == kv.MLA_LAYERS * page * (pages + kda_groups * (2 + draft_blocks) + 1)
    if not draft_blocks:
        default_cfg, _ = _resolve(monkeypatch, [], max_model_len=ONE_M)
        one_block = _priced_worker(model_specs, default_cfg, seqs=1, length=ONE_M)
        one_block_bytes = one_block._kv_cache_footprint_bytes(
            one_block._kv_cache_need_bytes(log=False))
        footprint = worker._kv_cache_footprint_bytes(need)
        assert footprint == one_block_bytes + kda_groups * kv.MLA_LAYERS * page
        assert round(footprint / GIB, 3) == ONE_M_ALIGN_FOOTPRINT_GIB


@pytest.mark.parametrize("mode", ["none", "all"])
def test_the_other_cache_modes_keep_every_kda_block_priced(model_specs, monkeypatch, mode):
    cfg, _ = _resolve_past_the_refusal(monkeypatch, [PC_FLAG_ON], max_model_len=ONE_M)
    worker = _in_cache_mode(_priced_worker(model_specs, cfg, seqs=1, length=ONE_M), mode)
    assert round(worker._kv_cache_footprint_bytes(worker._kv_cache_need_bytes(log=False))
                 / GIB, 2) == ONE_M_PC_ON_FOOTPRINT_GIB


@pytest.mark.parametrize("draft_blocks", [0, 3], ids=["plain", "mtp-3"])
def test_vllms_align_mode_manager_never_holds_more_kda_blocks_than_priced(
    model_specs, monkeypatch, draft_blocks
) -> None:
    """vLLM's own manager, stepped through a chunked prefill, a prefix hit and decode
    across block boundaries: in each KDA group a request holds at most the blocks the
    worker prices per request, 2 + k, and it does reach that figure, so the price is
    the allocation."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import (
        get_kv_cache_configs,
        get_request_block_hasher,
        init_none_hash,
        resolve_kv_cache_block_sizes,
    )
    from vllm.utils.hashing import sha256
    from vllm.v1.kv_cache_interface import MambaSpec
    from vllm.v1.request import Request

    from vllm_neuron.vllm.worker.neuron_worker import pool_blocks_per_request

    length, block = 2048, kv.BLOCK_SIZE_TOKENS
    cfg, _ = _resolve_past_the_refusal(monkeypatch, [PC_FLAG_ON], max_model_len=length)
    worker = _in_cache_mode(_priced_worker(model_specs, cfg, seqs=2, length=length), "align",
                            draft_blocks=draft_blocks)
    vllm_config = worker.vllm_config
    vllm_config.cache_config.mamba_cache_mode = "align"
    specs = worker.model_runner.get_kv_cache_spec()
    config = get_kv_cache_configs(vllm_config, [specs], [worker._kv_cache_need_bytes(log=False)])[0]
    scheduler_block, hash_block = resolve_kv_cache_block_sizes(config, vllm_config)
    manager = KVCacheManager(
        kv_cache_config=config, max_model_len=length, scheduler_block_size=scheduler_block,
        hash_block_size=hash_block, max_num_batched_tokens=kv.MAX_NUM_BATCHED_TOKENS,
        enable_caching=True, use_eagle=bool(draft_blocks),
    )
    kda = [i for i, g in enumerate(config.kv_cache_groups) if isinstance(g.kv_cache_spec, MambaSpec)]
    managers = manager.coordinator.single_type_managers
    init_none_hash(sha256)
    hasher = get_request_block_hasher(hash_block, sha256)
    held: list[int] = []
    hits: list[int] = []

    def step(request, tokens: int, **kwargs) -> None:
        manager.new_step_starts()  # as the scheduler does before each schedule
        assert manager.allocate_slots(request, tokens, num_lookahead_tokens=draft_blocks,
                                      **kwargs) is not None
        held.extend(sum(not b.is_null for b in managers[g].req_to_blocks[request.request_id])
                    for g in kda)
        request.num_computed_tokens += tokens + kwargs.get("num_new_computed_tokens", 0)

    def serve(request_id: str, prompt: list[int], decode: int) -> None:
        request = Request(request_id=request_id, prompt_token_ids=prompt,
                          sampling_params=SamplingParams(max_tokens=decode),
                          pooling_params=None, block_hasher=hasher)
        hit_blocks, hit = manager.get_computed_blocks(request)
        hits.append(hit)
        left = len(prompt) - hit
        first = True
        while left:  # legs of whole blocks, then the tail
            tokens = min(left, 4 * block) if left >= block else left
            tokens = tokens // block * block or tokens
            extra = {"num_new_computed_tokens": hit, "new_computed_blocks": hit_blocks} if first else {}
            step(request, tokens, **extra)
            first, left = False, left - tokens
        for index in range(decode):
            request.append_output_token_ids(index)
            step(request, 1)
        manager.free(request)

    prompt = list(range(5 * block + 17))
    serve("cold", prompt, decode=2 * block)
    serve("hit", prompt + list(range(3 * block)), decode=block)

    priced = {pool_blocks_per_request(config.kv_cache_groups[g].kv_cache_spec, length, "KDA")
              for g in kda}
    assert priced == {2 + draft_blocks}
    assert max(held) == 2 + draft_blocks
    # The second request hits 7 blocks, and its last KDA block was filled by the first
    # request's decode: in this mode vLLM caches such a block as it caches the end of a
    # prefill leg, so it must hold its checkpoint before it is a hit source.
    assert hits == [0, 7 * block]
    assert 7 * block > len(prompt)


def test_prefix_caching_on_names_both_footprints_in_one_line(
    model_specs, monkeypatch, caplog
) -> None:
    """The budget path says what the 128-token KDA blocks cost, and the alternative."""
    default_cfg, _ = _resolve(monkeypatch, [], max_model_len=ONE_M)
    one_block = _priced_worker(model_specs, default_cfg, seqs=1, length=ONE_M)
    one_block_bytes = one_block._kv_cache_footprint_bytes(one_block._kv_cache_need_bytes(log=False))
    cfg, _ = _resolve_past_the_refusal(monkeypatch, [PC_FLAG_ON], max_model_len=ONE_M)
    worker = _priced_worker(model_specs, cfg, seqs=1, length=ONE_M)
    pc_on_bytes = worker._kv_cache_footprint_bytes(worker._kv_cache_need_bytes(log=False))

    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError):
        worker.determine_available_memory()

    lines = [r.getMessage() for r in caplog.records if "recurrent (KDA) group" in r.getMessage()]
    assert len(lines) == 1
    line = lines[0]
    assert "4 recurrent (KDA) group(s)" in line
    assert "8192 block(s) per request in each" in line
    assert f"footprint {pc_on_bytes / GIB:.3f} GiB" in line
    assert round(pc_on_bytes / GIB, 2) == ONE_M_PC_ON_FOOTPRINT_GIB
    assert f"it is {one_block_bytes / GIB:.3f} GiB" in line
    assert round(one_block_bytes / GIB, 2) == ONE_M_ONE_BLOCK_FOOTPRINT_GIB
    assert "--no-enable-prefix-caching" in line
    # The configuration is left as it was priced.
    assert worker.cache_config.enable_prefix_caching is True
    assert worker.cache_config.mamba_block_size is None


def test_align_mode_names_two_kda_blocks_per_request_in_the_line(
    model_specs, monkeypatch, caplog
) -> None:
    cfg, _ = _resolve_past_the_refusal(monkeypatch, [PC_FLAG_ON], max_model_len=ONE_M)
    worker = _in_cache_mode(_priced_worker(model_specs, cfg, seqs=1, length=ONE_M), "align",
                            draft_blocks=3)
    footprint = worker._kv_cache_footprint_bytes(worker._kv_cache_need_bytes(log=False))

    with caplog.at_level(logging.INFO):
        worker._log_recurrent_blocks(footprint)

    (line,) = [r.getMessage() for r in caplog.records if "recurrent (KDA) group" in r.getMessage()]
    assert "4 recurrent (KDA) group(s) keeps 128-token blocks, 2 block(s) per request" in line
    assert f"footprint {footprint / GIB:.3f} GiB" in line


def test_one_block_per_group_logs_no_kda_line(model_specs, monkeypatch, caplog) -> None:
    cfg, _ = _resolve(monkeypatch, [])
    worker = _priced_worker(model_specs, cfg, seqs=64, length=8192)
    worker._graph_need = lambda *_a, **_k: kv_graph(int(1.31 * GIB))

    with caplog.at_level(logging.INFO):
        assert worker.determine_available_memory() == 6276120576

    assert not [r for r in caplog.records if "recurrent (KDA) group" in r.getMessage()]


def kv_graph(need_bytes: int):
    from vllm_neuron.vllm.worker.neuron_worker import GraphNeed

    return GraphNeed(need_bytes, "fixture")
