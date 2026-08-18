# SPDX-License-Identifier: Apache-2.0
"""Proposer for DeepSeek-V4's DSpark block-parallel drafter.

WHY THIS IS A SUBCLASS OF :class:`~.eagle.EagleProposer` AND NOT A SIBLING

The runner identifies its drafter by type in three places -- the async
spec-decode transition path, the drafter KV-cache bind, and the KV-spec merge
(``neuron_model_runner.py:7491``, ``:7786``, ``:7854``, the last an outright
``assert isinstance(self.drafter, EagleProposer)``). A sibling class would need
all three widened, i.e. a framework-layer change to a path every Eagle3 model in
the repo already uses. Subclassing satisfies them as written. That is the
recorded decision, and it is why nothing below touches those guards.

WHAT ACTUALLY DIFFERS, AND WHAT DELIBERATELY DOES NOT

``EagleProposer.propose`` is generic: shift input ids by one, build the no-op
async-correction tensors, filter ``attn_metadata`` down to
``self.attn_layer_names``, call the draft model with ten fixed kwargs, unpack a
3-tuple. Every one of those steps is right for DSpark, so ``propose`` is NOT
overridden -- re-implementing it would be a second copy of the async-correction
contract, which is the kind of duplication that drifts. Only three things are
Eagle3-specific, and each is overridden below:

1. the ``method == "eagle3"`` assertion in ``__init__``,
2. the ``Eagle3``-prefixed registry lookup in ``compile_and_load_draft_model``,
3. the ``layers.{i}.self_attn`` KV layer names in ``load_model``.

Plus one hardening: ``_build_synthetic_inputs`` hardcodes the aux-hidden width
as ``hidden_size * 3``, which is coincidentally correct for this checkpoint's
three target layers. It is overridden to read the width from config so a
config change cannot silently produce a warmup graph whose input width differs
from runtime's.
"""

import logging

import torch
from torch import nn

from vllm_neuron.vllm.spec_decode.eagle import EagleProposer

logger = logging.getLogger(__name__)

#: The speculative-decoding method string this proposer serves. Upstream vLLM
#: 0.21.0 already accepts it (``SpeculativeConfig.use_eagle()`` lists ``"mtp"``,
#: which is what makes the runner's KV-spec merge and the drafter bind fire for
#: this proposer without a framework change), and it is the string the pinned
#: checkpoint's speculative config carries.
DSPARK_METHOD = "mtp"


class DSparkProposer(EagleProposer):
    """Drives one DSpark BLOCK of drafted tokens per target step.

    The block is ``config.dspark_block_size`` tokens wide and is produced by a
    single forward through all three draft stages -- not by
    ``num_speculative_tokens`` sequential draft passes. The proposer does not
    need to know that: it makes one model call either way, and the model returns
    the whole block. What the proposer DOES need to agree on is the count, which
    :meth:`load_model` checks.
    """

    def __init__(self, vllm_config, device: torch.device, on_device_sampling: bool = True):
        # The parent asserts ``method == "eagle3"``. Rather than duplicate the
        # parent's fifteen lines of setup to skip one line, temporarily present
        # the method it accepts and restore the real one immediately -- so any
        # future field the parent adds is still initialized here.
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        real_method = speculative_config.method
        if real_method != DSPARK_METHOD:
            raise ValueError(
                f"DSparkProposer serves speculative method {DSPARK_METHOD!r}; "
                f"the config declares {real_method!r}."
            )
        try:
            speculative_config.method = "eagle3"
            super().__init__(vllm_config, device, on_device_sampling)
        finally:
            speculative_config.method = real_method
        self.method = real_method

    # ── Draft-model resolution ──────────────────────────────────────────────
    def compile_and_load_draft_model(
        self, start_layer_idx: int, target_hidden_size: int
    ) -> nn.Module:
        """Resolve the DSpark drafter DIRECTLY, bypassing the registry.

        The parent looks the draft architecture up in the model registry after
        prepending ``Eagle3`` to it. Neither half applies here: there is no
        ``Eagle3DeepSeekV4...`` implementation, and the drafter is deliberately
        not a registry entry of its own -- it is not independently servable, it
        only ever exists beside a DeepSeek-V4 target. Importing the port class
        by name is therefore both simpler and more honest than adding a registry
        row that nothing else may use.

        The rest of the parent's body -- meta-device construction,
        ``num_speculative_tokens`` assignment, full vs lite weight loading, the
        graph-capture backend selection, the two ``torch.compile`` calls -- is
        reused verbatim by delegating to it with the registry step satisfied.

        Args:
            start_layer_idx: the target's ``num_hidden_layers``; the drafter's
                ``from_configs`` checks it against its own config.
            target_hidden_size: the target's (possibly padded) hidden size. The
                drafter's hidden size must equal it, since the drafter reuses the
                target's dimensions; a mismatch means a padded target, which
                this family does not produce.
        """
        draft_hf_config = self.draft_model_config.hf_config
        if target_hidden_size != draft_hf_config.hidden_size:
            raise ValueError(
                "DSpark shares the target's dimensions, so the target's hidden "
                f"size ({target_hidden_size}) must equal the draft config's "
                f"({draft_hf_config.hidden_size}). A padded target hidden size "
                "is not a configuration this family produces."
            )

        from vllm_neuron.model.deepseek_v4.factory import DeepSeekV4MTP

        # Skipping the ``Eagle3`` prefix is not enough on its own: the target and
        # the drafter SHARE one config file, so ``draft_model_config.architecture``
        # is ``"DeepseekV4ForCausalLM"`` -- a registry entry that resolves to the
        # TARGET. A plain lookup would therefore build a second full target as
        # the draft model: 43 layers, right runner contract, wrong weights, and
        # no error until the ``mtp.*`` keys came up missing. Resolving the port
        # class by import is what avoids that.
        return self._compile_and_load_with_model_cls(
            DeepSeekV4MTP, start_layer_idx, target_hidden_size
        )

    def _compile_and_load_with_model_cls(
        self, model_cls, start_layer_idx: int, target_hidden_size: int
    ) -> nn.Module:
        """The parent's body with the registry lookup replaced by ``model_cls``.

        Implemented by temporarily installing a one-entry registry view that the
        parent's lookup resolves against, so the load, compile and graph-capture
        logic -- meta-device construction, ``num_speculative_tokens``
        assignment, full vs lite weight loading, the graph-capture backend
        selection, both ``torch.compile`` calls -- stays in ONE place and this
        subclass inherits any change to it. Copying that body instead would put
        a second, drifting copy of the CPU-compile and warmup contract in the
        tree.

        The swap is on a module attribute and is restored in ``finally``. It is
        safe because ``load_model`` runs exactly once per runner during startup,
        single-threaded, before any draft step -- there is no second proposer
        loading concurrently to observe the swapped view.
        """
        import vllm_neuron.vllm.spec_decode.eagle as eagle_mod

        arch = self.draft_model_config.architecture
        original_get_models = eagle_mod.get_models
        original_method = self.method
        try:
            eagle_mod.get_models = lambda: [(arch, model_cls)]
            # ``method`` only steers the ``Eagle3``-prefix decision in the
            # parent; with the registry keyed on the bare architecture the
            # prefix must NOT be applied, which any non-"eagle3" value achieves.
            self.method = DSPARK_METHOD
            return super().compile_and_load_draft_model(
                start_layer_idx=start_layer_idx,
                target_hidden_size=target_hidden_size,
            )
        finally:
            eagle_mod.get_models = original_get_models
            self.method = original_method

    # ── KV layer names ──────────────────────────────────────────────────────
    def load_model(self, target_hidden_size: int) -> None:
        """Load and compile the drafter, then name its KV layers correctly.

        The parent derives ``layers.{i}.self_attn`` for ``i`` past the target's
        depth, which is the Eagle3 drafter's naming. DSpark's stages live in the
        checkpoint's ``mtp.*`` namespace and each declares exactly ONE
        sliding-window leg, so the names come from the drafter itself
        (:meth:`~vllm_neuron.model.deepseek_v4.dspark_model.DeepseekV4DSparkDrafter.expected_kv_layer_names`)
        rather than being re-derived here. Those names are the ones the TARGET
        declared to the runner -- see
        ``DeepseekV4ForCausalLM._drafter_kv_layer_specs`` for why the target owns
        the declaration -- and they are what ``propose`` filters
        ``attn_metadata`` down to, so a mismatch here surfaces as a KeyError on
        the first draft step rather than as silent wrong attention.
        """
        target_num_layers = self.vllm_config.model_config.hf_config.num_hidden_layers
        self.model = self.compile_and_load_draft_model(
            start_layer_idx=target_num_layers,
            target_hidden_size=target_hidden_size,
        )
        logger.info("Completed compilation for the DSpark draft model.")

        inner = getattr(self.model, "_orig_mod", self.model)
        self.attn_layer_names = list(inner.expected_kv_layer_names())
        logger.info("DSpark draft KV layers: %s", self.attn_layer_names)

        block_size = inner.config.dspark_block_size
        if self.num_speculative_tokens != block_size:
            raise ValueError(
                "DSpark drafts a whole block per step, so "
                f"num_speculative_tokens ({self.num_speculative_tokens}) must "
                f"equal config.dspark_block_size ({block_size}). The runner "
                "sizes its rejection-sampler tensors from the former and the "
                "model emits the latter; a mismatch is a shape error deep in "
                "the accepted-token path."
            )

    # ── Warmup input width ──────────────────────────────────────────────────
    def _build_synthetic_inputs(
        self, num_tokens: int, num_reqs: int, device: torch.device | None = None
    ) -> dict:
        """Fix the aux-hidden width to what config says, not ``hidden_size * 3``.

        The parent's literal 3 happens to be right for this checkpoint (three
        entries in ``dspark_target_layer_ids``), so this override changes no
        number today. It exists because the failure mode if that ever stops
        holding is a warmup graph traced at one input width and a runtime call
        at another -- a recompile under ``fail_on_recompile``, or worse a
        silently wrong ``main_proj`` if the widths happen to be compatible.
        """
        inputs = super()._build_synthetic_inputs(num_tokens, num_reqs, device=device)
        inner = getattr(self.model, "_orig_mod", self.model)
        width = inner.config.dspark_main_hidden_size
        current = inputs["target_hidden_states"]
        if current.shape[-1] != width:
            inputs["target_hidden_states"] = torch.ones(
                current.shape[0], width, dtype=current.dtype, device=current.device
            )
        return inputs
