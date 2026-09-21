"""Graph capture hands this family the translated kwargs, exactly as warmup does.

This model takes its caches as forward arguments, so the runner translates its generic
mapping at every model call site (``_glm5next_model_kwargs``). A capture that splatted the
generic mapping straight at the capture backend would hand the root ``positions``, which
it does not declare. Both capture entry points are driven over the tiny fixture here, and
the model must see the translated mapping and nothing else.

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_capture_sites.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from libtorch_neuronx_lite.compile.capture_backend import CaptureComplete
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

# The fixture and its dials come from the forward file; the cache dict, the CPU-mode gate
# and the route predicate come from the end-to-end file. Imported rather than
# re-implemented, so every file reads the same tree.
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny

pytestmark = [pytest.mark.fast, pytest.mark.forked]

# The six keyword arguments the translation hands the root for a threaded step, and the
# only ones it may hand it: the three carriers and the three parallelism arguments. Sorted,
# so a key that appeared or vanished fails rather than reorders.
CARRIER_KWARG_KEYS = [
    "expert_parallel_rank", "input_ids", "layer_carriers", "moe_group",
    "sampling_positions", "tp_degree",
]

# The prefill bucket the capture drives is the fixture's own stack length.
PREFILL_BUCKET = tiny.STACK_TOKENS

# One request, one token: the shape the decode capture builds, and the only decode shape
# this file drives (a multi-token decode refuses by name).
DECODE_BATCH = 1


def _bound_root():
    """The tiny root with the runner's own cache dict bound, and that dict."""
    e2e._require_cpu_mode()
    root = e2e._fixture()["root"]
    caches = e2e._runner_shaped_caches(root)
    root.bind_kv_cache(caches)
    return root, caches


def _runner(root) -> NeuronModelRunner:
    """A runner shell carrying only the attributes the two capture entry points read.

    The real ``load_model`` needs a full engine config, weights and the neuron compile
    backends, none of which exist in CPU mode, so the shell supplies the values those
    methods read and nothing else; every method under test runs unaltered on it.

    The KV-cache group is real and its geometry is read off the spec the model produced, so
    the page the metadata builder reports cannot drift from the page the banks were
    allocated at. The carrier builder refuses such a drift, which is the check this file
    would otherwise pass by construction.
    """
    layer = root.get_kv_spec().layers[0]
    names = [bank["name"] for bank in root.glm5next_layer_banks]
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = root
    runner.max_model_len = e2e.E2E_MAX_SEQ_LEN
    # The converter sizes its per-sequence caches by the engine's concurrent-sequence
    # bound; it is read from the file this harness shares its shapes with, not restated.
    runner.max_num_reqs = e2e.E2E_MAX_NUM_SEQS
    runner.device = torch.device("cpu")
    runner.rank_tensor = torch.zeros(1, dtype=torch.int32)
    runner.uses_mrope = False
    runner.supports_mm_inputs = False
    runner.enable_prompt_embeds = False
    runner.vision_neuron_config = None
    runner.drafter = None
    runner.speculative_config = None
    runner.use_async_scheduling = False
    runner.on_device_sampling = False
    runner._dcp_size = 1
    runner.cp_world_size = 1
    # The backing field, not ``cp_rank``: that name is a read-only property and assigning
    # to it would refuse.
    runner._cp_rank = 0
    runner.neuron_config = SimpleNamespace(
        enable_structured_outputs=False,
        kv_segment_size_buckets=None,
        decode_context_length_buckets=None,
    )
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(
                names,
                FullAttentionSpec(
                    block_size=tiny.MLA_PAGE_SIZE,
                    num_kv_heads=int(layer.num_kv_heads),
                    head_size=int(layer.head_size),
                    dtype=layer.dtype,
                    sliding_window=None,
                    attention_chunk_size=None,
                ),
            )
        ]
    )
    return runner


class _StandInBackend:
    """The capture backend's contract: call the model, then raise ``CaptureComplete``.

    The real backend is a ``torch.compile`` wrapper whose neuron sidecar traces the graph
    and then raises that exception, which the capture entry points swallow as the success
    signal. This stand-in keeps both halves of the contract and calls the model for real,
    so a signature the root does not accept surfaces here as it would on a device.
    """

    def __init__(self, runner: NeuronModelRunner) -> None:
        self.runner = runner
        self.seen: list[dict] = []
        self.output = None

    def __call__(self, **kwargs):
        self.seen.append(kwargs)
        self.output = self.runner.model(**kwargs)
        raise CaptureComplete


def _assert_translated(label: str, seen: list[dict]) -> dict:
    """The model was entered once, with the translated mapping and nothing else."""
    assert len(seen) == 1, (
        f"the {label} capture entered the backend {len(seen)} time(s); this configuration "
        f"has no drafter and no speculative config, so it captures exactly one graph"
    )
    kwargs = seen[0]
    assert sorted(kwargs) == CARRIER_KWARG_KEYS, (
        f"the {label} capture handed the root {sorted(kwargs)}; the root declares "
        f"{CARRIER_KWARG_KEYS} for a threaded step, and the generic keys belong to "
        f"features this model does not implement"
    )
    assert len(kwargs["layer_carriers"]) == tiny.STACK_LAYERS, (
        f"the {label} capture built {len(kwargs['layer_carriers'])} carriers for a "
        f"{tiny.STACK_LAYERS}-layer stack"
    )
    return kwargs


def _assert_wrote_the_banks(label: str, root, caches: dict, before: list) -> None:
    """Every latent bank the caller allocated has changed, so the carriers were its own."""
    for bank, prior in zip(root.glm5next_layer_banks, before):
        assert not torch.equal(caches[bank["name"]][0], prior), (
            f"the {label} capture left the allocated latent bank for {bank['name']} "
            f"unchanged, so the carrier it built pointed at something else"
        )


def _assert_finite_logits(label: str, logits, rows: int) -> None:
    """The forward produced one finite logit row per sampling position."""
    assert logits.shape[0] == rows
    assert logits.shape[-1] == tiny.STACK_VOCAB_SIZE
    assert torch.isfinite(logits.to(torch.float32)).all()


def test_prefill_graph_capture_hands_the_root_its_carriers():
    """One prefill capture, translated, through to the root and into the caller's banks."""
    root, caches = _bound_root()
    runner = _runner(root)
    backend = _StandInBackend(runner)
    runner.capture_backend_model = backend
    before_banks = [caches[bank["name"]][0].clone() for bank in root.glm5next_layer_banks]

    tiny._reset_seam_counters()
    before_seams = tiny._read_seam_counters()

    runner.extract_prefill_graphs(PREFILL_BUCKET, 0)

    after_seams = tiny._read_seam_counters()
    kwargs = _assert_translated("prefill", backend.seen)
    assert int(kwargs["input_ids"].shape[0]) == PREFILL_BUCKET
    _assert_finite_logits("prefill", backend.output, rows=1)
    _assert_wrote_the_banks("prefill", root, caches, before_banks)
    e2e._assert_route_predicate_r3("prefill-capture", before_seams, after_seams)


def test_prefill_capture_with_independent_query_and_kv_lengths():
    """Capture selects MLA Q128 while model operators and KV keep their full width."""
    e2e._require_cpu_mode()
    root = e2e._fixture()["root"]
    caches = e2e._runner_shaped_caches(root)
    context_length = 4096
    segment_size = 1024
    # Reserve the full context for the synthetic block table. Recurrent state
    # remains per request and does not grow with the query or KV window.
    for spec in root.get_kv_spec().layers:
        if spec.kda_recurrent_state_shape is None:
            caches[spec.name] = [
                torch.zeros(
                    (context_length // tiny.MLA_PAGE_SIZE + 1, *bank.shape[1:]),
                    dtype=bank.dtype,
                )
                for bank in caches[spec.name]
            ]
    root.bind_kv_cache(caches)
    runner = _runner(root)
    runner.max_model_len = context_length
    runner.neuron_config.kv_segment_size_buckets = [segment_size]
    runner.neuron_config.num_batched_tokens_buckets = [PREFILL_BUCKET, 1024]
    backend = _StandInBackend(runner)
    runner.capture_backend_model = backend

    runner.extract_prefill_graphs(PREFILL_BUCKET, segment_size)

    kwargs = _assert_translated("independent-prefill", backend.seen)
    assert kwargs["input_ids"].shape[0] == 1024
    assert torch.count_nonzero(kwargs["input_ids"][PREFILL_BUCKET:]) == 0
    _assert_finite_logits("independent-prefill", backend.output, rows=1)
    # The carrier builder walks the bound banks in order, so a carrier pairs with its bank
    # positionally and the rows below are read against the bank the caller allocated.
    for carrier, bank in zip(kwargs["layer_carriers"], root.glm5next_layer_banks):
        if "latent_cache" in carrier:
            # The carrier holds the whole latent bank, so its row count is the bank's own
            # and says nothing about where the request sits. What a captured graph is
            # compiled for is the block table's width times the page, read below.
            assert carrier["latent_cache"].shape[0] == int(bank["slots"])
            assert carrier["latent_cache"] is bank["latent_cache"]
            assert (
                carrier["block_table_row"].shape[0] * int(carrier["page_size"])
                == segment_size + 1024
            )
            assert carrier["latent_slots"].shape[0] == 1024
            assert carrier["active_mla_query_rows"] == PREFILL_BUCKET
            assert carrier["max_seq_len"] == context_length
        else:
            assert carrier["real_tokens"].tolist() == [[PREFILL_BUCKET]]
            assert carrier["row_mask"].shape == (1, 1024, 1)


def test_decode_graph_capture_hands_the_root_its_carriers():
    """One decode capture, translated, and classified as the decode leg it is."""
    root, caches = _bound_root()
    runner = _runner(root)
    backend = _StandInBackend(runner)
    runner.capture_backend_model = backend
    before_banks = [caches[bank["name"]][0].clone() for bank in root.glm5next_layer_banks]

    tiny._reset_seam_counters()
    before_seams = tiny._read_seam_counters()

    runner.extract_decode_graphs(DECODE_BATCH)

    after_seams = tiny._read_seam_counters()
    kwargs = _assert_translated("decode", backend.seen)
    assert int(kwargs["input_ids"].shape[0]) == DECODE_BATCH
    carrier = kwargs["layer_carriers"][0]
    assert "prefill_tail" not in carrier, (
        "the decode capture built a prefill carrier, so the decode graph is not the graph "
        "that was captured"
    )
    _assert_finite_logits("decode", backend.output, rows=DECODE_BATCH)
    _assert_wrote_the_banks("decode", root, caches, before_banks)
    e2e._assert_route_predicate_r3("decode-capture", before_seams, after_seams)


def test_an_untranslated_capture_is_refused_by_the_roots_signature(monkeypatch):
    """With the translation an identity, both captures raise ``TypeError`` at the root.

    Only the translation is removed; everything else stands. The refusal has to arrive from
    the model call itself, which the recorded backend entries show.
    """
    root, _ = _bound_root()
    runner = _runner(root)
    backend = _StandInBackend(runner)
    runner.capture_backend_model = backend
    monkeypatch.setattr(
        NeuronModelRunner, "_glm5next_model_kwargs", lambda self, kwargs: kwargs
    )
    refusal = "unexpected keyword argument 'positions'"

    with pytest.raises(TypeError, match=refusal) as prefill:
        runner.extract_prefill_graphs(PREFILL_BUCKET, 0)
    with pytest.raises(TypeError, match=refusal) as decode:
        runner.extract_decode_graphs(DECODE_BATCH)

    for label, caught in (("prefill", prefill), ("decode", decode)):
        assert "Glm5NextForConditionalGeneration.forward()" in str(caught.value), (
            f"the {label} capture failed somewhere other than the root's signature"
        )
    assert len(backend.seen) == 2, (
        f"the backend was entered {len(backend.seen)} time(s) across the two captures; a "
        f"refusal raised before the model call would prove nothing"
    )
