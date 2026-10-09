# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.3-Flash multi-token-prediction (MTP) draft, served from the target graph.

``--speculative-config '{"method": "mtp", "num_speculative_tokens": k}'`` names the
draft the checkpoint carries itself: the decoder layer past the stack, which the
root model builds as ``Glm5NextForConditionalGeneration.mtp``
(``vllm_neuron/model/glm5_next/mtp.py``) and runs inside its own forward.
Every decode step of the root therefore already returns ``(sampled_or_accepted,
draft_ids)``: on the one-token decode leg the ``[B, k]`` ids drafted from the row
it sampled, on the verify leg (``1 + k`` rows per request) the ``[B, k + 1]``
accepted ids of the on-device greedy rejection sampler and the ``[B, k]`` ids
drafted from the last accepted row. There is no second model, no second graph and
no second launch.

So this proposer, unlike :class:`~vllm_neuron.vllm.spec_decode.eagle.EagleProposer`,
owns nothing the worker has to load, warm, capture or bind:

* ``model`` is the root's own head, bound by :meth:`load_model`, so the worker's
  storage-identity footprint walk sees the module the root already counts.
  :data:`shares_target_parameters` says the head's bytes are the root's and must
  not be summed a second time.
* :meth:`warmup` and :meth:`graph_extract` are no-ops: the runner's target decode
  warmups and captures (the ``1 + k``-row verify shape and the one-row shape)
  trace the draft with the target.
* :meth:`take_drafts` turns the root's ``[B, k]`` int32 rows into the lists the
  scheduler takes. The root's prefill leg drafts nothing and returns rows of
  ``-1``; those become "no drafts", and the first decode after a prefill runs as a
  one-token step whose in-graph draft opens the verify loop.

Served with synchronous scheduling only. Under async scheduling the accepted count
reaches the host one step late, and the indexer-ring cursor and the recurrent-state
commit (``NeuronModelRunner._update_states_after_model_execute``) are corrected on
the host from that count, so that path is a second series and is refused here by
name rather than served with stale state.
"""

from __future__ import annotations

import logging

import torch
from vllm.config import VllmConfig

from vllm_neuron import envs

logger = logging.getLogger(__name__)

#: The root's draft rows carry this where no draft was made (the prefill leg).
NO_DRAFT = -1


class MtpProposer:
    """The runner's handle on the root's own draft head; see the module docstring."""

    #: The head is a submodule of the target root, so its parameters are already in
    #: ``model_runner.model.parameters()``; a footprint that sums the drafter's
    #: parameters on top would count them twice.
    shares_target_parameters = True

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        on_device_sampling: bool = True,
    ):
        """Fix ``k`` from the speculative config; refuse by name what this proposer cannot serve.

        Args:
            vllm_config: the engine config; ``speculative_config.method`` must be "mtp".
            device: where the drafts the root returns live.
            on_device_sampling: the runner's sampling mode; the verify step samples and
                drafts inside the target graph, so a host sampler is refused.

        Raises:
            ValueError: method other than "mtp"; host sampling; async scheduling with
                ``VLLM_NEURON_GLM5NEXT_MTP_ASYNC`` unset (the accepted count reaches the
                host one step late and the synchronous drafter corrects on the host);
                the knob set with synchronous scheduling (it would be ignored) or with
                ``max_num_seqs > 1`` (the async drafter serves one sequence).
        """
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        if self.speculative_config is None or self.speculative_config.method != "mtp":
            method = None if self.speculative_config is None else self.speculative_config.method
            raise ValueError(
                f"MtpProposer serves speculative method 'mtp' and was built for {method!r}"
            )
        if not on_device_sampling:
            raise ValueError(
                "speculative method 'mtp' on GLM-5.3-Flash needs on-device sampling: the "
                "verify step's rejection sampling and the draft from the last accepted "
                "row run inside the target graph, which a host sampler cannot feed; "
                "serve with an on_device_sampling_config"
            )
        async_scheduling = bool(vllm_config.scheduler_config.async_scheduling)
        async_knob = bool(envs.VLLM_NEURON_GLM5NEXT_MTP_ASYNC)
        if async_scheduling and not async_knob:
            raise ValueError(
                "speculative method 'mtp' on GLM-5.3-Flash is served with synchronous "
                "scheduling only: under async scheduling the accepted count reaches the "
                "host one step late and the indexer-ring cursor and recurrent-state "
                "commit would run on stale counts; serve with --no-async-scheduling, or "
                "set VLLM_NEURON_GLM5NEXT_MTP_ASYNC=1 for the async drafter, which "
                "corrects on device (one sequence)"
            )
        if async_knob and not async_scheduling:
            raise ValueError(
                "VLLM_NEURON_GLM5NEXT_MTP_ASYNC=1 selects the async drafter and the "
                "scheduler is synchronous, so the knob would change nothing; unset it "
                "or pass --async-scheduling"
            )
        max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
        if async_knob and max_num_seqs > 1:
            raise ValueError(
                f"VLLM_NEURON_GLM5NEXT_MTP_ASYNC=1 serves one sequence (its on-device "
                f"corrections carry one request's positions between steps) and the server "
                f"has max_num_seqs={max_num_seqs}; serve with --max-num-seqs 1"
            )
        self.device = device
        self.on_device_sampling = on_device_sampling
        #: True when the async drafter serves: positions are corrected on device one
        #: step late and the runner feeds the next step from the previous one's output.
        self.async_steps = async_knob
        self.num_speculative_tokens = int(self.speculative_config.num_speculative_tokens)
        #: The root's head once :meth:`load_model` ran; ``None`` before.
        self.model = None

    def load_model(self, target) -> None:
        """Bind the target root's own draft head; refuse a root that built none.

        The root builds the head when the head's reader of ``k``
        (``mtp.shadow_draft_k``, contract C1) is above 0, and it drafts that many
        tokens per step; the speculative config's ``num_speculative_tokens`` must be
        the same number, or the scheduler would reserve one count and the graph
        return another.
        """
        head = getattr(target, "mtp", None)
        if head is None:
            raise ValueError(
                f"speculative method 'mtp' needs the target root's draft head and "
                f"{type(target).__name__} built none (its 'mtp' attribute is None); the "
                f"head is built when the head's reader of k "
                f"(vllm_neuron.model.glm5_next.mtp.shadow_draft_k) reads "
                f"k={self.num_speculative_tokens} at construction"
            )
        from vllm_neuron.model.glm5_next import mtp as head_module

        built_k = int(head_module.shadow_draft_k())
        if built_k != self.num_speculative_tokens:
            raise ValueError(
                f"the target root's draft head drafts k={built_k} token(s) per step and "
                f"the speculative config asks for num_speculative_tokens="
                f"{self.num_speculative_tokens}; the scheduler reserves the config's count "
                f"and the graph returns the head's, so the two must be one number"
            )
        self.model = head
        logger.info(
            "Speculative method 'mtp': drafting k=%d token(s) per step from the target's "
            "own head; no draft model is loaded.",
            self.num_speculative_tokens,
        )

    def warmup(self, **_unused) -> None:
        """No-op: the target's decode warmups trace the draft with the target."""
        return None

    def graph_extract(self, **_unused) -> None:
        """No-op: the target's decode captures hold the draft graph."""
        return None

    @staticmethod
    def take_drafts(draft_ids: torch.Tensor) -> list[list[int]]:
        """The root's ``[B, k]`` int32 draft rows as per-request lists.

        A row is read up to its first :data:`NO_DRAFT`: the prefill leg returns whole
        rows of it (nothing drafted), and a request with no drafts is scheduled as a
        one-token decode. O(B * k) host integers, no device read: the caller hands a
        host tensor.
        """
        rows: list[list[int]] = []
        for row in draft_ids.to(torch.int64).tolist():
            kept: list[int] = []
            for value in row:
                if value == NO_DRAFT:
                    break
                kept.append(int(value))
            rows.append(kept)
        return rows
