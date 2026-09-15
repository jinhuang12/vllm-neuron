"""GLM-5.3-Flash multi-token-prediction (MTP) draft head.

``inc-glm53f-063``, WP9. This module is the whole source Surface of that
increment: the draft head that proposes speculative tokens, and nothing else.

WHAT THIS MODULE OWNS. The four MTP tensors the checkpoint ships at
``model.language_model.layers.45.*`` -- ``enorm.weight``, ``hnorm.weight``,
``eh_proj.weight`` and ``shared_head.norm.weight`` -- plus the arithmetic that
uses them: masking the embedding at position zero, normalising both inputs,
projecting their concatenation down from ``2H`` to ``H``, adding the residual
the block returns, applying the shared-head norm, and resolving which draft
layer a speculative step reads. The layer index the head starts at is
``num_hidden_layers`` (45 in this checkpoint), one past the main stack.

WHAT THIS MODULE DELIBERATELY DOES NOT OWN, AND WHY THAT IS NOT A GAP.

* **The decoder block.** The head takes its block as a constructor argument
  rather than building one, so this module imports no model tree at all. The
  checkpoint's MTP layer is ``deepseek_sparse_attention``-typed, which is
  ``Glm5NextDSALayer`` -- ``inc-glm53f-051``'s D14 section. D14 tells an
  implementer whose increment would have to touch a class outside its own
  section to raise the widening rather than take it
  (``model_fp8.py:4632-4640`` states the same rule for a shared norm body), so
  the block arrives already built and its keyword arguments pass through
  untouched. ``_build_layer`` is the intended builder; the caller runs it.
* **The head projection.** Layer 45 ships no head tensor of its own -- the
  top-level ``lm_head.weight`` is shared -- so the logits step takes that weight
  as an argument instead of declaring a fifth parameter it would then duplicate.
* **Config plumbing and weight loading.** ``num_nextn_predict_layers`` is a
  constructor argument because ``Glm5NextTextConfig`` declares no such field at
  this base and the weight map carries no MTP key. Both land with
  ``inc-glm53f-064`` as a recorded Surface rider; hard-coding the count here
  would type geometry the checkpoint declares, which this fork's house form
  argues against in terms.

REFERENCE. Upstream vLLM ``model_executor/models/glm4_moe_mtp.py:60-190``
(``Glm4MoeMultiTokenPredictorLayer`` and ``Glm4MoeMultiTokenPredictor``), read
from source at ``6a9c69fa85``. The arithmetic order is that reference's; the
parameter form is this fork's own flat declared-parameter house form rather than
upstream's ``nn`` submodules, so the map's paths flatten onto the module.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .config import Glm5NextTextConfig


def _declare_parameters(module: nn.Module, *names: str) -> None:
    """Reserve parameter attribute paths on ``module`` without allocating.

    DUPLICATED FROM ``model_fp8.py:135-150`` RATHER THAN IMPORTED, and the
    duplication is forced rather than lazy. That helper is private to its module
    and has no cross-file caller anywhere in the tree; importing a private name
    across modules is not this fork's convention -- ``model_fp8.py:4775-4790``
    states the same reasoning for ``_FP8_DTYPE``, which three shipped modules
    each declare for themselves. Importing it would also pull the whole model
    tree into this module's import graph, which the head does not otherwise need.
    The body is identical on purpose: a variant form would put a second
    declaration convention in one package.
    """
    for name in names:
        module.register_parameter(name, None)
    module.declared_param_names = (
        *getattr(module, "declared_param_names", ()),
        *names,
    )


class Glm5NextMTPLayer(nn.Module):
    """One MTP draft layer: two input norms, the fused projection, one block.

    The parameter names are the checkpoint's own paths flattened, the same way
    every landed decoder layer in this tree flattens them: ``enorm.weight``
    becomes ``enorm_weight`` and ``shared_head.norm.weight`` becomes
    ``shared_head_norm_weight``.
    """

    def __init__(
        self,
        text_config: Glm5NextTextConfig,
        layer_idx: int,
        mtp_block: nn.Module,
    ) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        _declare_parameters(
            self,
            "enorm_weight",
            "hnorm_weight",
            "eh_proj_weight",
            "shared_head_norm_weight",
        )
        self.mtp_block = mtp_block
        self.hidden_size = int(text_config.hidden_size)
        # Resolved at construction, on the ground the landed layers state: the
        # value is the checkpoint's and does not change per call.
        self.rms_norm_eps = float(text_config.rms_norm_eps)

    def _rms_norm(self, x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        """``x / sqrt(mean(x**2) + eps) * gain``, accumulated in fp32.

        THE SAME BODY AS ``Glm5NextDSALayer._input_norm``
        (``model_fp8.py:4643-4646``) for the same reason that method gives for
        repeating its KDA sibling: sharing it would move a landed class this
        increment does not own. Written once here and taken three gains.
        """
        promoted = x.to(torch.float32)
        variance = promoted.pow(2).mean(dim=-1, keepdim=True)
        normed = promoted * torch.rsqrt(variance + self.rms_norm_eps)
        normed = normed * gain.to(torch.float32)
        return normed.to(x.dtype)

    @staticmethod
    def _mask_first_position(
        inputs_embeds: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Zero the embedding wherever the absolute position is zero.

        The first position has no previous token to draft from, so its embedding
        contributes nothing. Upstream ``glm4_moe_mtp.py:172``.
        """
        return torch.where(
            positions.unsqueeze(-1) == 0,
            torch.zeros((), dtype=inputs_embeds.dtype, device=inputs_embeds.device),
            inputs_embeds,
        )

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        positions: torch.Tensor,
        **block_kwargs: object,
    ) -> torch.Tensor:
        """Norm both inputs, project their concatenation, run the block, norm out.

        THE BLOCK'S KEYWORD ARGUMENTS PASS THROUGH UNCHANGED. This head knows
        nothing about caches, slot mappings or index tails; whichever decoder
        family the caller injected owns those, and threading them verbatim is
        what keeps the two increments' sections apart.

        THE CONCATENATION ORDER IS LOAD-BEARING: the embedding half first, the
        previous hidden state second, because ``eh_proj_weight``'s columns are
        the checkpoint's in that order. Reversing it silently produces plausible
        numbers, which is why its test carries a firing control.
        """
        embeds = self._mask_first_position(inputs_embeds, positions)
        embeds = self._rms_norm(embeds, self.enorm_weight)
        previous = self._rms_norm(previous_hidden_states, self.hnorm_weight)
        joined = torch.cat([embeds, previous], dim=-1)
        hidden_states = torch.nn.functional.linear(joined, self.eh_proj_weight)
        block_out = self.mtp_block(hidden_states, **block_kwargs)
        return self._rms_norm(block_out, self.shared_head_norm_weight)


class Glm5NextMultiTokenPredictor(nn.Module):
    """The draft head. Indexes ONE draft layer per speculative step, never loops.

    ``num_nextn_predict_layers`` is how many draft layers the checkpoint ships,
    and it is also the width of one step's proposal: at 1 -- this checkpoint's
    declared value -- every step resolves to the single layer and emits exactly
    one draft token per position. The repetition across steps belongs to the
    caller, not to this class. Upstream does the same, resolving the step by
    modulo rather than iterating (``glm4_moe_mtp.py:161-178``), and
    ``mimo_mtp.py:200`` turns the single-layer case into an outright assert.
    """

    def __init__(
        self,
        text_config: Glm5NextTextConfig,
        num_nextn_predict_layers: int,
        mtp_blocks: Sequence[nn.Module],
    ) -> None:
        super().__init__()
        count = int(num_nextn_predict_layers)
        if count < 1:
            raise ValueError(
                f"num_nextn_predict_layers must be at least 1, got {count!r}; "
                "the checkpoint declares it and a draft head with no layer "
                "cannot propose"
            )
        blocks = list(mtp_blocks)
        if len(blocks) != count:
            raise ValueError(
                f"the draft head needs one block per draft layer: "
                f"num_nextn_predict_layers is {count} but {len(blocks)} "
                f"block(s) were given"
            )
        self.text_config = text_config
        self.num_mtp_layers = count
        # One past the main stack, which is where the checkpoint puts the MTP
        # layer. Read from the config rather than typed.
        self.mtp_start_layer_idx = int(text_config.num_hidden_layers)
        # Keyed by ABSOLUTE layer index as a string, the same key the weight
        # paths use, so a loader never has to translate.
        self.layers = nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx + offset): Glm5NextMTPLayer(
                    text_config, self.mtp_start_layer_idx + offset, block
                )
                for offset, block in enumerate(blocks)
            }
        )

    def resolve_step_layer_index(self, spec_step_index: int) -> int:
        """The ABSOLUTE layer index a speculative step reads.

        Modulo the draft-layer count, so a caller that keeps stepping past the
        last draft layer wraps rather than indexing off the end. At a count of 1
        every step resolves to the one layer.
        """
        step = int(spec_step_index)
        if step < 0:
            raise ValueError(f"spec_step_index must not be negative, got {step!r}")
        return self.mtp_start_layer_idx + step % self.num_mtp_layers

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        positions: torch.Tensor,
        spec_step_index: int = 0,
        **block_kwargs: object,
    ) -> torch.Tensor:
        """The normed hidden state one speculative step produces."""
        layer = self.layers[str(self.resolve_step_layer_index(spec_step_index))]
        return layer(
            inputs_embeds=inputs_embeds,
            previous_hidden_states=previous_hidden_states,
            positions=positions,
            **block_kwargs,
        )

    def compute_draft_logits(
        self, hidden_states: torch.Tensor, lm_head_weight: torch.Tensor
    ) -> torch.Tensor:
        """Draft logits against the SHARED head weight.

        Layer 45 ships no head tensor, so the weight is the root's
        ``lm_head.weight`` and it arrives as an argument. Kept separate from
        :meth:`forward` because a verifier may want the hidden state without
        paying for a vocabulary-wide projection.
        """
        return torch.nn.functional.linear(hidden_states, lm_head_weight)

    @property
    def draft_window_width(self) -> int:
        """How many draft tokens a full proposal covers -- the checkpoint's count.

        One STEP reads one draft layer and emits one token per position; the
        window is how many such steps fill a complete proposal. At this
        checkpoint's declared ``num_nextn_predict_layers == 1`` the two coincide,
        which is exactly why a head that quietly emitted a second token per step
        would still look right on a shape check of the wrong axis.
        """
        return self.num_mtp_layers

    def propose_draft_tokens(
        self,
        *,
        inputs_embeds: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        positions: torch.Tensor,
        lm_head_weight: torch.Tensor,
        spec_step_index: int = 0,
        **block_kwargs: object,
    ) -> torch.Tensor:
        """One step's draft tokens, shaped ``(num_positions,)``.

        ONE STEP EMITS ONE TOKEN PER POSITION, because a step resolves exactly
        one draft layer and a layer produces one hidden state per position. That
        is a property of the returned shape rather than of a comment, and it does
        NOT multiply by the layer count -- filling a wider draft window is the
        caller's loop over ``spec_step_index``, as it is upstream. Greedy
        ``argmax``, which is what a draft proposal is before the verifier sees it.
        """
        hidden_states = self.forward(
            inputs_embeds=inputs_embeds,
            previous_hidden_states=previous_hidden_states,
            positions=positions,
            spec_step_index=spec_step_index,
            **block_kwargs,
        )
        logits = self.compute_draft_logits(hidden_states, lm_head_weight)
        return logits.argmax(dim=-1)
