"""Graph capture must hand this family the translated kwargs, exactly as warmup does.

This model takes its caches as forward ARGUMENTS, so the runner translates its generic
mapping at every model call site (``neuron_model_runner.py``'s
``_glm5next_model_kwargs``). The three GRAPH-CAPTURE calls did not translate: they
splatted the generic mapping straight at the capture backend, and the root's forward was
handed ``positions``, which it does not declare. Every rank of a 64-rank serve died there.

This file drives both capture entry points over the tiny fixture and requires the model to
see the translated mapping and nothing else. Its control strikes the translation and
requires the original refusal back, by name, so a file that passed on a stand-in that
never reached the model cannot look green.

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 \\
        python -m pytest -s -rA \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_capture_sites.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from libtorch_neuronx_lite.compile.capture_backend import CaptureComplete
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

# The landed tiny files, imported rather than re-implemented: the fixture and its dials
# come from the forward file, and the cache dict, the CPU-lane gate and the registered
# route predicate come from the end-to-end file, so this file measures the same tree
# those items measure and can disagree with neither about what was built.
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as landed
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as item

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The three keyword arguments the root's forward declares for a threaded step, which are
#: the only ones the translation may hand it. Sorted, because the assertions compare sets
#: of keys and a key that appeared or vanished must redden rather than reorder.
CARRIER_KWARG_KEYS = ["input_ids", "layer_carriers", "sampling_positions"]

#: The prefill bucket the capture drives: the fixture's own stack length, so the block run
#: the translation slices is the run the landed prefill items already measure.
PREFILL_BUCKET = item.STACK_TOKENS

#: One request, one token -- the shape the decode capture builds, and the only decode shape
#: this half threads (a multi-token decode refuses by name).
DECODE_BATCH = 1


def _bound_root():
    """The tiny root with the runner's own cache dict bound, and that dict."""
    landed._require_cpu_mode()
    root = landed._fixture()["root"]
    caches = landed._runner_shaped_caches(root)
    root.bind_kv_cache(caches)
    return root, caches


def _runner(root) -> NeuronModelRunner:
    """A runner shell carrying only the attributes the two capture entry points read.

    THE SHELL IS THE LANDED CONVENTION and it is deliberate: the real ``load_model`` needs
    a full engine config, weights and the neuron compile backends, none of which exist in
    the CPU lane. Every method under test here -- the two ``extract_*_graphs``, both
    synthetic-input builders, the warmup metadata builder and the translation -- runs
    VERBATIM on this shell; only the values it reads are supplied.

    THE KV-CACHE GROUP IS REAL AND ITS GEOMETRY IS THE MODEL'S OWN, read off the spec the
    model produced, so the page the metadata builder reports cannot drift from the page the
    banks were allocated at. A drift there is refused by the carrier builder, which is the
    check this file would otherwise be passing by construction.
    """
    layer = root.get_kv_spec().layers[0]
    names = [bank["name"] for bank in root.glm5next_layer_banks]
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = root
    runner.max_model_len = landed.E2E_MAX_SEQ_LEN
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
    # The backing field, not ``cp_rank``: that name is a read-only property
    # (``neuron_model_runner.py:995``) and assigning to it would refuse.
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
                    block_size=item.MLA_PAGE_SIZE,
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
    and then throws that exception, which the capture entry points swallow as the success
    signal (``neuron_model_runner.py:4575``). This stand-in keeps both halves of that
    contract, and it CALLS THE MODEL FOR REAL: the defect under repair is a ``TypeError``
    raised by the root's own signature, so a stand-in that only recorded its arguments
    could not produce it and the control below would be inert.
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
    print(f"CAPTURESITE|{label}|keys={sorted(kwargs)}")
    assert sorted(kwargs) == CARRIER_KWARG_KEYS, (
        f"the {label} capture handed the root {sorted(kwargs)}; the root declares "
        f"{CARRIER_KWARG_KEYS} for a threaded step, and the generic keys belong to "
        f"features this model does not implement"
    )
    assert len(kwargs["layer_carriers"]) == item.STACK_LAYERS, (
        f"the {label} capture built {len(kwargs['layer_carriers'])} carriers for a "
        f"{item.STACK_LAYERS}-layer stack"
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
    print(f"CAPTURESITE|{label}|logits={tuple(logits.shape)}:{logits.dtype}")
    assert logits.shape[0] == rows
    assert logits.shape[-1] == item.STACK_VOCAB_SIZE
    assert torch.isfinite(logits.to(torch.float32)).all()


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 1. the prefill capture reaches the root with carriers, on the route it declares.
# ══════════════════════════════════════════════════════════════════════════════════════


def test_prefill_graph_capture_hands_the_root_its_carriers():
    """One prefill capture, translated, through to the root and into the caller's banks."""
    root, caches = _bound_root()
    runner = _runner(root)
    backend = _StandInBackend(runner)
    runner.capture_backend_model = backend
    before_banks = [caches[bank["name"]][0].clone() for bank in root.glm5next_layer_banks]

    item._reset_seam_counters()
    before_seams = item._read_seam_counters()

    runner.extract_prefill_graphs(PREFILL_BUCKET, 0)

    after_seams = item._read_seam_counters()
    kwargs = _assert_translated("prefill", backend.seen)
    assert int(kwargs["input_ids"].shape[0]) == PREFILL_BUCKET
    _assert_finite_logits("prefill", backend.output, rows=1)
    _assert_wrote_the_banks("prefill", root, caches, before_banks)
    landed._assert_route_predicate_r3("prefill-capture", before_seams, after_seams)


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 2. the decode capture does the same, and is captured as a decode.
# ══════════════════════════════════════════════════════════════════════════════════════


def test_decode_graph_capture_hands_the_root_its_carriers():
    """One decode capture, translated, and classified as the decode leg it is."""
    root, caches = _bound_root()
    runner = _runner(root)
    backend = _StandInBackend(runner)
    runner.capture_backend_model = backend
    before_banks = [caches[bank["name"]][0].clone() for bank in root.glm5next_layer_banks]

    item._reset_seam_counters()
    before_seams = item._read_seam_counters()

    runner.extract_decode_graphs(DECODE_BATCH)

    after_seams = item._read_seam_counters()
    kwargs = _assert_translated("decode", backend.seen)
    assert int(kwargs["input_ids"].shape[0]) == DECODE_BATCH
    carrier = kwargs["layer_carriers"][0]
    print(f"CAPTURESITE|decode|carrier_keys={sorted(carrier)}")
    assert "prefill_tail" not in carrier, (
        "the decode capture built a PREFILL carrier, so this item is measuring the wrong "
        "leg and the decode graph is not the graph that was captured"
    )
    _assert_finite_logits("decode", backend.output, rows=DECODE_BATCH)
    _assert_wrote_the_banks("decode", root, caches, before_banks)
    landed._assert_route_predicate_r3("decode-capture", before_seams, after_seams)


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 3. THE CONTROL. Strike the translation and the measured refusal comes back.
# ══════════════════════════════════════════════════════════════════════════════════════


def test_striking_the_translation_returns_the_refusal_the_ranks_hit(monkeypatch):
    """With the translation an identity, both captures raise the served defect by name.

    THIS IS WHAT MAKES THE TWO ITEMS ABOVE MEAN SOMETHING. They would pass unchanged
    against a stand-in that quietly dropped whatever it could not use; this arm removes
    only the translation, leaves everything else standing, and requires the exact
    ``TypeError`` the 64 ranks reported. The refusal must arrive FROM the model call, which
    the recorded entries measure.
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
        print(f"CAPTURESITE|control|{label}|{caught.value}")
        assert "Glm5NextForConditionalGeneration.forward()" in str(caught.value), (
            f"the {label} control failed somewhere other than the root's signature, so it "
            f"is not the defect this file repairs"
        )
    assert len(backend.seen) == 2, (
        f"the backend was entered {len(backend.seen)} time(s) across the two controls; a "
        f"refusal raised before the model call would leave this arm proving nothing"
    )
