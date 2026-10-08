# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash defaults to prefix caching off and one KDA block per request.

With prefix caching on (vLLM's default for this model: upstream does not flag it
hybrid, so ``HybridAttentionMambaModelConfig`` never runs) each recurrent (KDA)
group keeps the 128-token attention block, and vLLM's mamba "none" mode reserves
``cdiv(max_model_len, 128)`` of them per request in every one of the four KDA
groups. The runner keeps the KDA state in a bank addressed by request slot, so
those blocks hold no data, and it refuses a prefix-cache hit on a recurrent stack
(``kv_spec_patch.recurrent_spec_block_size``). At 1 x 1M that is 55.69 GiB per rank
for 11.70 GiB of real need.

``NeuronPlatform.check_and_update_config`` therefore sets, for this architecture on
the hybrid KV cache:

* ``enable_prefix_caching=False`` unless ``--enable-prefix-caching`` or
  ``--no-enable-prefix-caching`` is on the command line;
* ``mamba_block_size=max_model_len`` (upstream's own default for a hybrid model
  with prefix caching off) unless a mamba block size is set, or prefix caching is
  on, where vLLM's hybrid coordinator needs the 128-token block.

Each default yields to its own flag. One log line always says what was applied and
why. With prefix caching on, the worker prices every KDA block and one log line
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
#: The 1 x 1M point before and after the default (``kvbudget-logs/derived.json``).
ONE_M_PC_ON_FOOTPRINT_GIB = 55.69
ONE_M_ONE_BLOCK_FOOTPRINT_GIB = 11.70

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


@pytest.mark.parametrize("spelling", [PC_FLAG_ON, "--enable_prefix_caching"])
def test_an_explicit_prefix_caching_on_keeps_the_128_token_kda_block(
    monkeypatch, spelling
) -> None:
    """The operator's prefix caching stands; the mamba block is not set under it."""
    cfg, records = _resolve(monkeypatch, [spelling])

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


def test_the_default_needs_the_architecture_and_the_hybrid_cache(monkeypatch) -> None:
    """Other architectures, and GLM at a degree without the hybrid cache, keep vLLM's."""
    other = hybrid._other_archs()[0]
    for build in ({"arch": other, "neuron_config": {"enable_hybrid_kv_cache": True}},
                  {"tp": hybrid.UNSUPPORTED_TP_DEGREE}):
        cfg, records = _resolve(monkeypatch, [], **build)

        assert cfg.cache_config.enable_prefix_caching is True, build
        assert cfg.cache_config.mamba_block_size is None, build
        assert records == [], build


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
    """The default's 1 x 1M point: 11.70 GiB per rank, from 55.69 GiB.

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
    cfg, _ = _resolve(monkeypatch, [PC_FLAG_ON], max_model_len=ONE_M)
    worker = _priced_worker(model_specs, cfg, seqs=1, length=ONE_M)
    page = kv.BLOCK_SIZE_TOKENS * kv.LATENT_BYTES_PER_TOKEN
    kda_groups = math.ceil(kv.KDA_LAYERS / kv.MLA_LAYERS)
    pages = math.ceil(ONE_M / kv.BLOCK_SIZE_TOKENS)

    need = worker._kv_cache_need_bytes(log=False)

    assert need == kv.MLA_LAYERS * page * (pages * (1 + kda_groups) + 1)
    assert round(worker._kv_cache_footprint_bytes(need) / GIB, 2) == ONE_M_PC_ON_FOOTPRINT_GIB


def test_prefix_caching_on_names_both_footprints_in_one_line(
    model_specs, monkeypatch, caplog
) -> None:
    """The budget path says what the 128-token KDA blocks cost, and the alternative."""
    default_cfg, _ = _resolve(monkeypatch, [], max_model_len=ONE_M)
    one_block = _priced_worker(model_specs, default_cfg, seqs=1, length=ONE_M)
    one_block_bytes = one_block._kv_cache_footprint_bytes(one_block._kv_cache_need_bytes(log=False))
    cfg, _ = _resolve(monkeypatch, [PC_FLAG_ON], max_model_len=ONE_M)
    worker = _priced_worker(model_specs, cfg, seqs=1, length=ONE_M)

    with caplog.at_level(logging.INFO), pytest.raises(RuntimeError):
        worker.determine_available_memory()

    lines = [r.getMessage() for r in caplog.records if "recurrent (KDA) group" in r.getMessage()]
    assert len(lines) == 1
    line = lines[0]
    assert "4 recurrent (KDA) group(s)" in line
    assert "8192 block(s) per request in each" in line
    assert f"footprint {ONE_M_PC_ON_FOOTPRINT_GIB:.2f}" in line
    assert f"it is {one_block_bytes / GIB:.3f} GiB" in line
    assert round(one_block_bytes / GIB, 2) == ONE_M_ONE_BLOCK_FOOTPRINT_GIB
    assert "--no-enable-prefix-caching" in line
    # The configuration is left as it was priced.
    assert worker.cache_config.enable_prefix_caching is True
    assert worker.cache_config.mamba_block_size is None


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
