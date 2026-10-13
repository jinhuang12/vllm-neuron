# SPDX-License-Identifier: Apache-2.0
"""At max_num_seqs 4, a step that prefills an image request carries that request alone.

The runner gathers multimodal embeddings for one request per step
(``NeuronModelRunner._gather_mm_embeddings`` asserts it), so admitting image requests at a
batch above 1 needs the scheduler to give vLLM's base scheduler one prefill request per
step and no decode beside it. ``NeuronScheduler`` already does that for every prefill
(``max_prefills_per_batch`` 1, decodes hidden during a prefill step); these tests pin it
for requests that carry an image (``mm_features``), in each phase of their prefill.

They drive the real ``schedule()`` of both scheduler classes on an instance built without
a ``VllmConfig`` (``test_scheduler_hybrid._build``), with vLLM's base ``schedule`` replaced
by a recorder, and read what the base scheduler was handed.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/test_scheduler_one_prefill_with_images.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_neuron.vllm.core.scheduler import NeuronAsyncScheduler, NeuronScheduler

from test.vllm_neuron.test_scheduler_hybrid import _BasePatch, _build, _fake_request

pytestmark = [pytest.mark.fast]

BATCH = 4
PROMPT = 4096  # tokens; several 1024-token chunks
DECODED = 64  # a decode request: every prompt token computed
IMAGE_TOKENS = 512


def _request(rid: str, *, prompt: int, computed: int, image: bool = False):
    request = _fake_request(rid, prompt=prompt, computed=computed)
    request.mm_features = (
        [SimpleNamespace(modality="image", identifier=f"{rid}-image", mm_position=None)]
        if image
        else []
    )
    return request


def _decodes(count: int):
    return [_request(f"dec{i}", prompt=DECODED, computed=DECODED) for i in range(count)]


def _schedule(cls, *, running, holdback) -> dict:
    """One real ``schedule()``; what the base scheduler was handed."""
    scheduler = _build(cls, kv_window=BATCH, num_blocks=1, groups=[], max_num_seqs=BATCH)
    scheduler.running.extend(running)
    scheduler.holdback_queue.extend(holdback)
    scheduler._apply_padding_and_log_stats = lambda output: output
    scheduler.get_grammar_bitmask = lambda output: None
    seen: dict = {}
    with _BasePatch(seen):
        scheduler.schedule()
    running_ids = seen["at_delegate_running"]
    waiting_ids = seen["at_delegate_waiting"]
    return {
        "prefills": [r for r in running_ids if not r.startswith("dec")] + waiting_ids,
        "decodes": [r for r in running_ids if r.startswith("dec")],
    }


SCHEDULERS = pytest.mark.parametrize(
    "cls", [NeuronScheduler, NeuronAsyncScheduler], ids=lambda cls: cls.__name__
)


@SCHEDULERS
def test_a_queued_image_request_is_prefilled_alone(cls):
    """Three decodes running, the image request and a text prompt queued."""
    seen = _schedule(
        cls,
        running=_decodes(BATCH - 1),
        holdback=[
            _request("img0", prompt=PROMPT, computed=0, image=True),
            _request("txt0", prompt=PROMPT, computed=0),
        ],
    )
    assert seen == {"prefills": ["img0"], "decodes": []}


@SCHEDULERS
def test_an_image_request_mid_prefill_is_prefilled_alone(cls):
    """The image request has computed one chunk; two decodes run beside it."""
    seen = _schedule(
        cls,
        running=[_request("img0", prompt=PROMPT, computed=1024, image=True), *_decodes(2)],
        holdback=[
            _request("img1", prompt=PROMPT, computed=0, image=True),
            _request("txt0", prompt=PROMPT, computed=0),
        ],
    )
    assert seen == {"prefills": ["img0"], "decodes": []}


@SCHEDULERS
def test_a_queued_image_request_waits_for_a_text_prefill(cls):
    seen = _schedule(
        cls,
        running=[_request("txt0", prompt=PROMPT, computed=1024), *_decodes(2)],
        holdback=[_request("img0", prompt=PROMPT, computed=0, image=True)],
    )
    assert seen == {"prefills": ["txt0"], "decodes": []}


@SCHEDULERS
def test_decodes_run_together_once_the_image_request_has_prefilled(cls):
    """After its last chunk the image request decodes beside the others."""
    image_done = _request(
        "dec_img0", prompt=PROMPT + IMAGE_TOKENS, computed=PROMPT + IMAGE_TOKENS, image=True
    )
    seen = _schedule(cls, running=[image_done, *_decodes(BATCH - 1)], holdback=[])
    assert seen["prefills"] == []
    assert sorted(seen["decodes"]) == ["dec0", "dec1", "dec2", "dec_img0"]
