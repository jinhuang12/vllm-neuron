# SPDX-License-Identifier: Apache-2.0
"""DSA "lightning indexer" index-logit scoring + top-k selection (DeepSeek-V4).

WHY this is real torch math and not a kernel wrapper
----------------------------------------------------
The port plan cites the NKI kernel ``sparse_attention_indexer`` (ladder row
LD-06) for this step. That kernel is **ABSENT from the installed nkilib wheel**
(family interface contract Section 0), so this row takes its recorded
same-row torch-composition rung: the index-logit math is composed here from
traceable static-shape torch ops, and the only kernel reuse is the *existing*
rotational top-k (via :func:`vllm_neuron.functional.topk.topk`, which owns the
authoritative dry-run feasibility gate). Nothing here imports ``nkilib``
directly, and this file carries no NxDI dependency of any kind.

(The forbidden distribution's package name is deliberately NOT spelled out
anywhere in this tree: the R6 gate is a mechanical text scan over
``vllm_neuron/``, so even a comment asserting the absence of that import
registers as a hit. State the property, never the string.)

Math source of truth: DeepSeek's OWN reference implementation
------------------------------------------------------------
``dsv4_ref/model.py`` (shipped in the pinned checkpoint repo, revision
``7872f01b1d1fe23eabc4c98b48bffcef5a386062``) is PRIMARY evidence and outranks
the derived spec reports wherever they disagree. ``Indexer.forward``, lines
408-439, is the whole of this op::

    417  q = self.wq_b(qr)
    418  q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
    419  apply_rotary_emb(q[..., -rd:], freqs_cis)
    420  q = rotate_activation(q)
    422  fp4_act_quant(q, fp4_block_size, True)
    423  self.compressor(x, start_pos)
    424  weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
    426  index_score = torch.einsum("bshd,btd->bsht", q,
    426                             self.kv_cache[:bsz, :end_pos // ratio])
    427  index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    431  mask = arange(seqlen // ratio).repeat(seqlen, 1) >= arange(1, seqlen+1)[:, None] // ratio
    432  index_score += torch.where(mask, float("-inf"), 0)
    433  topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
    435  mask = topk_idxs >= arange(1, seqlen + 1).unsqueeze(1) // ratio
    436  topk_idxs = torch.where(mask, -1, topk_idxs + offset)

Settled by those lines (each was previously an assumption):

* **ReLU is real.** ``index_score.relu_()`` is applied to the raw per-head dot
  product BEFORE the per-head weighting and the sum over heads (ref:427). This
  refutes dataflow-shapes GAPS-3 ("plain scaled dot-product logits"), so
  ``head_activation`` defaults to ``"relu"`` here. It also means the per-head
  weights CANNOT be folded into Q (see ``_index_scores``).
* **The weight fold is exactly** ``weights_proj(x) * (head_dim**-0.5 *
  n_heads**-0.5)`` (ref:424, with ``softmax_scale = head_dim ** -0.5`` at
  ref:401) -- no softplus, no sigmoid, no abs, applied to ``hidden_states`` and
  not to the q latent. ``weights_proj`` is bf16/unquantized (ref:400).
* **Q RoPE** is applied to the LAST ``rope_head_dim=64`` dims of each 128-wide
  index head (ref:419) using the interleaved/GPT-J pairing
  (``apply_rotary_emb`` -> ``view_as_complex(x.unflatten(-1, (-1, 2)))``,
  ref:238-249, i.e. ``is_neox_style=False``). The index Q gets NO per-head
  RMSNorm (contrast the main attention's ``q *= rsqrt(...)``, ref:503).
* **Index keys are the indexer's own compressor output** read out of its
  ``kv_cache`` (ref:405, ref:426): gated pooling + RMSNorm(128) + RoPE + quant
  inside ``Compressor.forward`` (ref:322-385). This op therefore consumes keys,
  it does not build them.
* **Causal cap** ``column < (pos + 1) // compress_ratio`` (ref:431), applied
  twice: as ``-inf`` on the score (ref:432) and again as an exact index test
  after the top-k (ref:435). Both are reproduced.
* **The sentinel is -1** and valid indices are shifted by the caller's pool
  ``offset`` (ref:436, ``offset = kv.size(1)`` at prefill / ``window_size`` at
  decode, ref:519) -> ``pad_index=-1`` and ``index_offset`` here.
* **Positions inside the sliding window are NOT excluded** from the compressed
  top-k: the caller simply concatenates the window index list with this one
  (ref:521-522). Early queries with ``(pos+1)//ratio == 0`` get an all-sentinel
  row and attend to the window only -- which is what the static-shape padding
  below produces.
* No collective here: the reference all-reduces ``index_score`` (ref:428-429)
  only because it column-shards the 64 index heads. Contract Section 2
  REPLICATES indexer ``wq_b``/``weights_proj``, so all 64 heads are local and
  the sum over heads is already global.

Deliberately NOT reproduced: ``rotate_activation`` (randomized Hadamard) +
``fp4_act_quant`` QAT simulation on Q (ref:420-422) and on the indexer
compressor's keys (``Compressor(..., rotate=True)``, ref:404 -> ref:379-381).
``hadamard_transform(x, scale=d**-0.5)`` is ORTHOGONAL, so
``(Hq) . (Hk) == q . k``: dropping it on BOTH sides leaves every index score
unchanged apart from the dropped fp4 rounding noise. WARNING: it must be
dropped on both sides or neither -- if the port's indexer compressor ever
starts Hadamard-rotating its keys, Q must be rotated here too.

Traceability rules obeyed (contract Section 0)
----------------------------------------------
No ``.item()``, no ``.tolist()``, no ``nonzero()``, no boolean-mask indexing,
no data-dependent shapes, no Python ``if`` on tensor values. Every Python
``if``/``for`` below branches on a *static* shape, dtype or Python argument.
Invalid keys are neutralised with ``torch.where`` (never by shrinking), and the
returned ``indices`` always has the static shape ``[T, topk]``.

Sentinel convention (settled by the reference, load-bearing for the consumer)
----------------------------------------------------------------------------
Slots with no admissible key -- a query whose causal compressed pool is
shorter than ``topk``, a padded cache slot (``key_slot_ids < 0``), or the
column padding used when the candidate pool itself is smaller than ``topk`` --
are filled with ``pad_index``, whose default ``-1`` is the reference's own
sentinel (``torch.where(mask, -1, topk_idxs + offset)``, ``dsv4_ref/model.py``
line 436; it is also vLLM's ``slot_mapping`` "no slot" convention). ``-1`` is
NOT a usable gather index: the consumer (``mla_sparse_attention``) must either
clamp it and mask the resulting logit, or carry its own ``topk_length``, exactly
as upstream's ``combine_topk_swa_indices`` does with ``topk_len``.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["sparse_indexer_topk"]

# Score written into masked-out (non-causal / padded / invalid) key columns.
# A large finite negative rather than -inf so that (a) the rotational NKI top-k
# kernel, whose pad sentinel is the dtype minimum, never has to reason about
# inf/NaN, and (b) the "was this slot real?" test below is a plain finite
# comparison that stays inside the traceable subset.
_MASKED_SCORE: float = -1.0e30
# Anything at or below this came from _MASKED_SCORE, not from real data.
_MASKED_SCORE_THRESHOLD: float = _MASKED_SCORE / 2.0

# Default "no key" sentinel written into unfilled top-k slots.
DEFAULT_PAD_INDEX: int = -1


def sparse_indexer_topk(
    x: Tensor,
    wq_b: Tensor,
    wk: Tensor | None,
    weights_proj: Tensor | None,
    index_k_cache: Tensor | None,
    topk: int,
    *,
    wq_b_scale: Tensor | None = None,
    hidden_states: Tensor | None = None,
    head_weights: Tensor | None = None,
    index_k: Tensor | None = None,
    key_states: Tensor | None = None,
    key_slot_ids: Tensor | None = None,
    key_scale: Tensor | None = None,
    key_valid: Tensor | None = None,
    positions: Tensor | None = None,
    rope_cos: Tensor | None = None,
    rope_sin: Tensor | None = None,
    rope_tables_pregathered: bool = False,
    q_scale: Tensor | None = None,
    n_index_heads: int = 64,
    index_head_dim: int = 128,
    rope_head_dim: int = 64,
    compress_ratio: int = 4,
    head_activation: str = "relu",
    index_offset: int | Tensor = 0,
    pad_index: int = DEFAULT_PAD_INDEX,
    accum_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype = torch.int32,
) -> Tensor:
    """Select the top-``topk`` compressed KV slots per query token (DSA indexer).

    The six leading parameters are the plan-fixed positional signature
    (contract Section 5, row ``functional/attention/sparse_indexer.py``); every
    other input the torch-composition rung needs is keyword-only with a
    default, so the recorded call form stays valid and this signature is a
    superset of it.

    Args:
        x: The q latent ``qr`` **after** ``q_norm``, ``[T, q_lora_rank=1024]``.
            This is the same tensor the main attention's ``wq_b`` consumes --
            the indexer has no separate q-down path
            (``deepseek_v4_attention.py:1197``).
        wq_b: Indexer query up-projection, ``[n_index_heads*index_head_dim,
            q_lora_rank]`` = ``[8192, 1024]``, stored ``[out, in]`` like every
            weight in this checkpoint (contract Section 1). fp8_e4m3 when
            ``wq_b_scale`` is given, else bf16.
        wk: Indexer key projection, ``[coff*index_head_dim, hidden_size]`` =
            ``[256, 4096]`` (checkpoint ``indexer.compressor.wkv.weight``).
            Used ONLY by the fallback key path (see ``index_k``); may be
            ``None`` when keys are supplied directly. See the ``key_states``
            note about parity.
        weights_proj: Unquantized per-head weight projection, ``[n_index_heads,
            hidden_size]`` = ``[64, 4096]``. Ignored when ``head_weights`` is
            supplied.
        index_k_cache: The paged indexer K cache,
            ``[num_blocks, 1, block_size, index_head_dim]`` (contract Section 4,
            ``layers.{i}.self_attn.indexer`` k_cache, head_size 128). Read with
            ``key_slot_ids``. May be ``None`` when keys are supplied directly.
        topk: Static number of slots to return per query token
            (``config.index_topk`` = 512). Also the static width of the result.

    Keyword Args:
        wq_b_scale: fp32 block scales for ``wq_b``, ``[ceil(8192/128),
            ceil(1024/128)]`` = ``[64, 8]``. When not ``None`` the projection
            goes through ``NF.block_fp8_linear``; when ``None`` it is a plain
            ``torch.matmul(x, wq_b.t())`` (the bf16 debug/test path).
        hidden_states: ``[T, hidden_size]`` -- the input ``weights_proj``
            consumes. Required unless ``head_weights`` is given.
        head_weights: Pre-computed raw per-head weights ``[T, n_index_heads]``
            (upstream computes these on an aux stream,
            ``deepseek_v4_attention.py:384-387``). The scale fold below is
            still applied to them.
        index_k: Materialised index-key pool, ``[S, index_head_dim]`` (shared
            pool) or ``[T, S, index_head_dim]`` (per-query pool). This is the
            PARITY path: pass the indexer compressor's output (gathered cache +
            this step's fresh compressed slots) here.
        key_states: ``[S, hidden_size]`` for the ``wk`` fallback path. NOT a
            parity path (see the ``wk`` note in the key-resolution helper).
        key_slot_ids: int slot ids into ``index_k_cache`` flattened as
            ``block_idx * block_size + offset`` (dataflow-shapes Section D
            addressing), shape ``[S]`` or ``[T, S]``. Negative entries mark
            padding and are masked out (never indexed).
        key_scale: Optional dequant scale broadcast-multiplied onto the gathered
            keys (the indexer cache stores fp8; its scales live in the paired
            ``index_v_cache``). Group-wise scales must be expanded by the
            caller.
        key_valid: Extra caller-supplied validity mask, broadcastable to
            ``[T, S]``. AND-ed with the causal and slot-padding masks.
        positions: ``[T]`` int positions. Drives the Q RoPE table lookup and
            the causal cap ``(pos + 1) // compress_ratio``.
        rope_cos: GPT-J RoPE cosine table, one row per POSITION by default and
            indexed by ``positions``. Trailing dim ``rope_head_dim // 2`` (or
            ``rope_head_dim``, in which case the first half is used).
        rope_sin: Matching sine table.
        rope_tables_pregathered: set True when the tables already carry one row
            per TOKEN, so the position lookup is skipped. Dropping ``positions``
            is NOT an alternative here -- it also drives the causal cap
            ``(pos + 1) // compress_ratio``, so a caller holding per-token
            tables has no way to opt out of the lookup without this flag, and
            feeding them with the lookup on is silently correct only for a
            single sequence prefilled from position 0. Same flag and same
            reason as :func:`~vllm_neuron.functional.attention.mla_qkv.mla_qkv`;
            the two ops share the convention so a caller cannot get one right
            and the other wrong. DeepSeek-V4's ``_cos_sin`` is per-token, so its
            callers pass True.
        q_scale: Optional per-token fp8 dequant scale for Q, ``[T]`` or
            ``[T, 1]``, folded into the head weights exactly as
            ``fused_indexer_q.py:165`` does.
        n_index_heads: ``config.index_n_heads`` (64).
        index_head_dim: ``config.index_head_dim`` (128).
        rope_head_dim: ``config.qk_rope_head_dim`` (64).
        compress_ratio: Compression ratio of this layer (4 on every indexer
            layer); only used for the causal cap.
        head_activation: ``"relu"`` (default, PARITY -- ``dsv4_ref/model.py``
            line 427 ``index_score.relu_()``) or ``"none"`` (NOT parity; a
            cheaper linear form kept only for ablation, since without the ReLU
            the per-head weights fold into Q and the score collapses to one
            ``[T, D] x [D, S]`` matmul).
        index_offset: Added to every SURVIVING index, exactly as
            ``topk_idxs + offset`` at ``dsv4_ref/model.py`` line 436 (the
            reference passes the base offset of the compressed region inside the
            concatenated KV buffer: ``kv.size(1)`` at prefill, ``window_size``
            at decode, line 519). Sentinel slots are never offset. Accepts an
            int or a ``[T, 1]``-broadcastable tensor.
        pad_index: Sentinel written into unfilled result slots (default -1,
            the reference's sentinel, ``dsv4_ref/model.py`` line 436).
        accum_dtype: Accumulation dtype for the logits (fp32, matching
            upstream's fp32 QK reference, ``rocm_aiter_mla_sparse.py:929-933``).
        out_dtype: Integer dtype of the returned indices (int32).

    Returns:
        ``indices``: ``[T, topk]`` ``out_dtype`` tensor of candidate key slots
        (columns of the candidate pool, plus ``index_offset``), descending by
        index score, with ``pad_index`` in every slot that has no admissible
        key. Static shape, always.
    """
    if head_activation not in ("none", "relu"):
        raise ValueError(
            f"head_activation must be 'none' or 'relu', got {head_activation!r}"
        )
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")

    num_tokens = x.shape[0]

    # 1. Index queries: x [T, 1024] -> [T, 64, 128], then GPT-J RoPE.
    q = _index_queries(x, wq_b, wq_b_scale, n_index_heads, index_head_dim)
    q = _apply_gptj_rope(
        q,
        rope_cos,
        rope_sin,
        None if rope_tables_pregathered else positions,
        rope_head_dim,
    )

    # 2. Index keys: [S, 128] or [T, S, 128], plus their padding validity.
    keys, slot_valid = _index_keys(
        wk=wk,
        index_k_cache=index_k_cache,
        index_k=index_k,
        key_states=key_states,
        key_slot_ids=key_slot_ids,
        key_scale=key_scale,
        index_head_dim=index_head_dim,
        accum_dtype=accum_dtype,
    )

    # 3. Per-head weights with the upstream scale fold, then the weighted sum
    #    over the 64 index heads -> one score per (query, key).
    weights = _head_weights(
        weights_proj=weights_proj,
        hidden_states=hidden_states,
        head_weights=head_weights,
        q_scale=q_scale,
        num_tokens=num_tokens,
        n_index_heads=n_index_heads,
        index_head_dim=index_head_dim,
        accum_dtype=accum_dtype,
    )
    scores = _index_scores(
        q.to(accum_dtype), keys, weights, head_activation, n_index_heads
    )

    # 4. Mask non-causal / padded keys, then static-shape top-k.
    scores = _mask_scores(
        scores,
        slot_valid=slot_valid,
        key_valid=key_valid,
        positions=positions,
        compress_ratio=compress_ratio,
    )
    return _select_topk(
        scores,
        topk,
        pad_index,
        out_dtype,
        positions=positions,
        compress_ratio=compress_ratio,
        index_offset=index_offset,
    )


# ---------------------------------------------------------------------------
# Stage 1: index queries
# ---------------------------------------------------------------------------


def _index_queries(
    x: Tensor,
    wq_b: Tensor,
    wq_b_scale: Tensor | None,
    n_index_heads: int,
    index_head_dim: int,
) -> Tensor:
    """``q = wq_b(qr).view(T, n_index_heads, index_head_dim)``.

    ``dsv4_ref/model.py`` 417-418: ``q = self.wq_b(qr)`` then
    ``q.unflatten(-1, (n_local_heads, head_dim))`` -- no bias, no per-head norm.

    WHY the lazy ``NF`` import: ``NF.block_fp8_linear`` (ladder row LD-11) is
    authored by a different node in this same port. Referencing it inside the
    body -- never at module import -- keeps ``import
    vllm_neuron.functional.attention.sparse_indexer`` working before that op
    lands, so this module can be compiled and reviewed independently.
    """
    if wq_b_scale is not None:
        import vllm_neuron.functional as NF  # noqa: PLC0415 (see docstring)

        # Exact call form pinned by contract Section 5.
        q = NF.block_fp8_linear(
            x,
            wq_b,
            wq_b_scale,
            block_size=(128, 128),
            act_group_size=128,
            accum_dtype=torch.float32,
            out_dtype=torch.bfloat16,
            bias=None,
        )
    else:
        # Weights are stored [out, in] in this checkpoint (contract Section 1),
        # hence the transpose. Unquantized/bf16 path only.
        q = torch.matmul(x, wq_b.t())
    return q.view(-1, n_index_heads, index_head_dim)


def _apply_gptj_rope(
    q: Tensor,
    rope_cos: Tensor | None,
    rope_sin: Tensor | None,
    positions: Tensor | None,
    rope_head_dim: int,
) -> Tensor:
    """GPT-J (interleaved) RoPE on the LAST ``rope_head_dim`` dims of each head.

    Mirrors ``fused_indexer_q.py:113-119``::

        x_even = q[NOPE + 2*i]; x_odd = q[NOPE + 2*i + 1]
        r_even = x_even*cos - x_odd*sin
        r_odd  = x_odd*cos  + x_even*sin

    ``is_neox_style=False`` for this model (dataflow-shapes B20), which is
    exactly this even/odd pairing. No-op when no table is supplied (unit-test
    and RoPE-free debug path).
    """
    if rope_cos is None or rope_sin is None:
        return q
    if rope_head_dim <= 0:
        return q

    half = rope_head_dim // 2
    cos = rope_cos
    sin = rope_sin
    if positions is not None:
        # Table lookup by position. index_select keeps the shape static and is
        # traceable; positions must be an integer tensor.
        idx = positions.reshape(-1).to(torch.int64)
        cos = cos.index_select(0, idx)
        sin = sin.index_select(0, idx)
    # Some caches store cos/sin duplicated to the full rope width; take the
    # first half. This is a STATIC shape test, not a value test.
    if cos.shape[-1] >= rope_head_dim:
        cos = cos[..., :half]
        sin = sin[..., :half]

    nope_dim = q.shape[-1] - rope_head_dim
    q_nope = q[..., :nope_dim]
    q_rope = q[..., nope_dim:]

    pairs = q_rope.reshape(q.shape[0], q.shape[1], half, 2)
    even = pairs[..., 0]
    odd = pairs[..., 1]
    cos = cos.reshape(q.shape[0], 1, half).to(q.dtype)
    sin = sin.reshape(q.shape[0], 1, half).to(q.dtype)
    rot_even = even * cos - odd * sin
    rot_odd = odd * cos + even * sin
    # stack+reshape re-interleaves (even, odd) back into the head's tail.
    rotated = torch.stack((rot_even, rot_odd), dim=-1).reshape(
        q.shape[0], q.shape[1], rope_head_dim
    )
    return torch.cat((q_nope, rotated), dim=-1)


# ---------------------------------------------------------------------------
# Stage 2: index keys
# ---------------------------------------------------------------------------


def _index_keys(
    *,
    wk: Tensor | None,
    index_k_cache: Tensor | None,
    index_k: Tensor | None,
    key_states: Tensor | None,
    key_slot_ids: Tensor | None,
    key_scale: Tensor | None,
    index_head_dim: int,
    accum_dtype: torch.dtype,
) -> tuple[Tensor, Tensor | None]:
    """Resolve the candidate key pool and its padding-validity mask.

    Three sources, in precedence order:

    1. ``index_k`` -- already-materialised indexer keys. This is the PARITY
       path: the reference scores against its indexer compressor's own
       ``kv_cache`` (``dsv4_ref/model.py`` 405, 415, 426), whose contents are
       the gated pooling ``(kv_state * score_state.softmax(dim)).sum(dim)`` plus
       RMSNorm(128) plus RoPE plus quant of ``Compressor.forward``
       (``dsv4_ref/model.py`` 322-385).
    2. ``index_k_cache`` + ``key_slot_ids`` -- paged read of cached compressed
       slots. Flattening the cache to ``[num_blocks*block_size, head_dim]``
       makes ``block_idx * block_size + offset`` a single ``index_select``,
       which is the traceable static-shape form of the addressing in
       dataflow-shapes Section D. Negative slot ids are clamped to 0 for the
       gather and reported invalid, so padding is masked -- never indexed and
       never used to shrink a shape.
    3. ``key_states @ wk[:head_dim].t()`` -- a RAW key projection. NOT parity:
       the real indexer key is the compressor's *gated pooled* value (softmax
       over ``wgate(x) + ape[pos % ratio]`` across the
       ``(1 + overlap) * compress_ratio`` window, then RMSNorm, RoPE, quant --
       ``dsv4_ref/model.py`` 322-385). That pipeline lives in
       ``DeepseekV4KVCompressor``, not here, and ``wk`` is only the ``wkv`` half
       of it (checkpoint ``indexer.compressor.wkv.weight`` ``[256, 4096]`` =
       ``coff(2) * index_head_dim(128)``; the first ``index_head_dim`` rows are
       the non-overlap window). This branch exists so the op is exercisable
       standalone; production must pass ``index_k``.
    """
    if index_k is not None:
        keys = index_k.to(accum_dtype)
        # A caller that pre-gathered the pool itself (plan §19.2 gather-first:
        # reads trace before the cache write, in-flight rows overlaid in the
        # POOL) still needs the padded-slot backstop that the cache branch
        # derives, so validity comes from ``key_slot_ids`` whenever the caller
        # provides it. Callers passing ``index_k`` alone are unchanged.
        slot_valid = None if key_slot_ids is None else key_slot_ids >= 0
    elif index_k_cache is not None and key_slot_ids is not None:
        flat_cache = index_k_cache.reshape(-1, index_head_dim)
        slot_valid = key_slot_ids >= 0
        safe_slots = torch.where(
            slot_valid, key_slot_ids, torch.zeros_like(key_slot_ids)
        ).to(torch.int64)
        gathered = flat_cache.index_select(0, safe_slots.reshape(-1))
        keys = gathered.reshape(*key_slot_ids.shape, index_head_dim).to(accum_dtype)
    elif wk is not None and key_states is not None:
        keys = torch.matmul(
            key_states.to(accum_dtype), wk[:index_head_dim, :].t().to(accum_dtype)
        )
        slot_valid = None
    else:
        raise ValueError(
            "no index-key source: pass index_k=..., or "
            "(index_k_cache=..., key_slot_ids=...), or (wk=..., key_states=...)"
        )

    if key_scale is not None:
        keys = keys * key_scale.to(accum_dtype)
    if keys.dim() not in (2, 3):
        raise ValueError(
            f"index keys must be [S, D] or [T, S, D], got shape {tuple(keys.shape)}"
        )
    return keys, slot_valid


# ---------------------------------------------------------------------------
# Stage 3: per-head weights and the weighted head sum
# ---------------------------------------------------------------------------


def _head_weights(
    *,
    weights_proj: Tensor | None,
    hidden_states: Tensor | None,
    head_weights: Tensor | None,
    q_scale: Tensor | None,
    num_tokens: int,
    n_index_heads: int,
    index_head_dim: int,
    accum_dtype: torch.dtype,
) -> Tensor:
    """Per-token per-head index weights ``[T, n_index_heads]`` with the fold.

    Reference fold, ``dsv4_ref/model.py`` line 424 (with ``softmax_scale =
    head_dim ** -0.5``, line 401)::

        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)

    The fp8/fp4 Q dequant scale is folded into the same weight on the kernel
    path (``fused_indexer_q.py:153-167``), which is why ``q_scale`` multiplies
    here; it is 1 (i.e. omitted) on this bf16-Q composition. There is
    deliberately NO "uniform weights" fallback: the per-head weighting decides
    which keys win the top-k, so a missing input raises instead of silently
    changing the selection.
    """
    if head_weights is not None:
        raw = head_weights.to(accum_dtype)
    else:
        if weights_proj is None or hidden_states is None:
            raise ValueError(
                "per-head index weights need either head_weights=..., or both "
                "weights_proj and hidden_states=... (weights_proj consumes "
                "hidden_states, not the q latent)"
            )
        # weights_proj is [n_index_heads, hidden_size], stored [out, in].
        raw = torch.matmul(
            hidden_states.to(accum_dtype), weights_proj.t().to(accum_dtype)
        )
    raw = raw.reshape(num_tokens, n_index_heads)

    softmax_scale = float(index_head_dim) ** -0.5  # self.softmax_scale
    head_scale = float(n_index_heads) ** -0.5  # self.n_head**-0.5
    weights = raw * (softmax_scale * head_scale)
    if q_scale is not None:
        weights = weights * q_scale.reshape(num_tokens, 1).to(accum_dtype)
    return weights


def _index_scores(
    q: Tensor,
    keys: Tensor,
    weights: Tensor,
    head_activation: str,
    n_index_heads: int,
) -> Tensor:
    """One score per (query, key). Parity form (``dsv4_ref/model.py`` 426-427)::

        index_score = einsum("bshd,btd->bsht", q, k)
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

    i.e. ``score[t, s] = sum_h weights[t, h] * relu(q[t, h] . k[s])``. The ReLU
    is inside the head sum, so the weights CANNOT be folded into Q; the heads
    are accumulated in a Python loop over a STATICALLY known count (allowed by
    contract Section 0), which keeps peak live memory at ``[T, S]`` instead of
    the ``[T, 64, S]`` block the reference's einsum materialises (2 GiB fp32 at
    T=512, S=16384).

    ``head_activation="none"`` drops the ReLU -- NOT parity, kept only for
    ablation. Without it the head sum is linear in ``q``, so the weights fold
    into Q and the score collapses to a single ``[T, D] x [D, S]`` matmul.
    """
    per_token_pool = keys.dim() == 3
    if head_activation == "none":
        # sum_h w_h * (q_h . k) == (sum_h w_h * q_h) . k
        q_folded = (q * weights.unsqueeze(-1)).sum(dim=1)  # [T, D]
        if per_token_pool:
            return torch.bmm(q_folded.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)
        return torch.matmul(q_folded, keys.t())

    scores = None
    for head in range(n_index_heads):
        q_h = q[:, head, :]
        if per_token_pool:
            logits_h = torch.bmm(q_h.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)
        else:
            logits_h = torch.matmul(q_h, keys.t())
        term = weights[:, head : head + 1] * torch.relu(logits_h)
        scores = term if scores is None else scores + term
    assert scores is not None  # n_index_heads >= 1
    return scores


# ---------------------------------------------------------------------------
# Stage 4: masking and static-shape top-k
# ---------------------------------------------------------------------------


def _mask_scores(
    scores: Tensor,
    *,
    slot_valid: Tensor | None,
    key_valid: Tensor | None,
    positions: Tensor | None,
    compress_ratio: int,
) -> Tensor:
    """Push non-admissible keys to ``_MASKED_SCORE`` (never shrink the pool).

    Causal cap, ``dsv4_ref/model.py`` 431-432::

        mask = arange(seqlen // ratio).repeat(seqlen, 1) >= arange(1, seqlen+1)[:, None] // ratio
        index_score += torch.where(mask, float("-inf"), 0)

    i.e. a query at ``pos`` may only see compressed slots
    ``[0, (pos + 1) // compress_ratio)``. The reference's ``min(index_topk,
    end_pos // ratio)`` (line 433) is a data-dependent ``k``; here the ``k`` is
    static and the same cap is enforced by masking plus the post-top-k index
    test in ``_select_topk``. Masking uses a large finite negative instead of
    ``-inf`` and every mask feeds ``torch.where`` -- no boolean-mask indexing.
    """
    mask: Tensor | None = None
    if slot_valid is not None:
        mask = slot_valid
    if key_valid is not None:
        mask = key_valid if mask is None else (mask & key_valid)
    if positions is not None and compress_ratio > 0:
        num_slots = scores.shape[-1]
        limit = torch.div(
            positions.reshape(-1, 1) + 1, compress_ratio, rounding_mode="floor"
        )
        columns = torch.arange(num_slots, device=scores.device).reshape(1, num_slots)
        causal = columns < limit
        mask = causal if mask is None else (mask & causal)
    if mask is None:
        return scores
    masked = torch.full_like(scores, _MASKED_SCORE)
    return torch.where(mask.expand_as(scores), scores, masked)


def _select_topk(
    scores: Tensor,
    topk: int,
    pad_index: int,
    out_dtype: torch.dtype,
    *,
    positions: Tensor | None,
    compress_ratio: int,
    index_offset: int | Tensor,
) -> Tensor:
    """Static-shape top-``topk`` over the key axis, sentinel-padded.

    LADDER REUSE (LD-06): selection goes through
    ``vllm_neuron.functional.topk.topk``, which owns the authoritative dry-run
    feasibility gate for the rotational NKI top-k kernel
    (``_rotational_topk_config_compiles``) and falls back to ``torch.topk``
    for any shape the kernel cannot compile. Calling that entry point instead
    of re-deriving a gate here means this row cannot drift from the kernel's
    real envelope, and the choice is made at trace time on the target device
    (``can_run_kernel`` is False on CPU, so host tests always take
    ``torch.topk``). The gate cannot be evaluated off-instance (nkilib/nki are
    not installed on the planning host), so the branch for the production shape
    (rows = T up to the prefill chunk, ``k = 512``, pool =
    ``ceil(max_model_len / 4)``) is decided there; both branches return the
    same static ``[T, k]`` shape and sorted-descending order, so correctness
    does not depend on which one wins.

    The import is lazy because ``vllm_neuron.functional.topk`` imports
    ``nki.language`` at module scope, which is unavailable off-instance.

    Post-selection invalidation mirrors ``dsv4_ref/model.py`` 435-436
    (``mask = topk_idxs >= arange(1, seqlen+1)[:, None] // ratio``;
    ``topk_idxs = where(mask, -1, topk_idxs + offset)``): the reference decides
    on the INDEX, not on the score, so that exact test is used whenever
    ``positions`` is known. The score test is kept as the backstop that also
    catches padded cache slots (``key_slot_ids < 0``), which the reference has no
    equivalent of because it has no paged cache.
    """
    num_tokens, num_slots = scores.shape
    # Static: bounded by the candidate pool. This is the static-shape stand-in
    # for the reference's data-dependent min(index_topk, end_pos // ratio)
    # (dsv4_ref/model.py:433). torch.topk rejects k > pool, and the rotational
    # gate requires 0 < k < pool.
    k_eff = min(topk, num_slots)

    from vllm_neuron.functional.topk import topk as _gated_topk  # noqa: PLC0415

    values, indices = _gated_topk(scores, k_eff, dim=-1, gather_dim=-1)

    # Slots that only won because everything else was masked out.
    # 0-dim tensor threshold, not a bare Python float: the ``Tensor > float``
    # form lowers the scalar at F64 and upcasts, which neuronx-cc rejects
    # ([NCC_ESPP004], StableHLOToPythonPrinter.cc:824). Same construct class and
    # the same literal (-5e+29) as swa_attention.py:289. ep11 iteration 11 proved
    # by a backward HLO walk (P6) that this site contributed ZERO F64
    # instructions to the failing graph — 23 of 23 clusters were additive-mask-
    # like with no sort/custom-call ancestor. It is corrected as measured-class
    # prophylaxis, since the site is on the production path
    # (model/deepseek_v4/attention.py:1657) and the serve graph is unmeasured.
    _vthr = torch.full(
        (), _MASKED_SCORE_THRESHOLD, dtype=values.dtype, device=values.device
    )
    valid = values > _vthr
    if positions is not None and compress_ratio > 0:
        limit = torch.div(
            positions.reshape(-1, 1) + 1, compress_ratio, rounding_mode="floor"
        ).to(indices.dtype)
        valid = valid & (indices < limit)
    if isinstance(index_offset, Tensor):
        shifted = indices + index_offset.reshape(-1, 1).to(indices.dtype)
    else:
        shifted = indices + index_offset
    pad = torch.full_like(shifted, pad_index)
    indices = torch.where(valid, shifted, pad).to(out_dtype)

    if k_eff < topk:
        # Pool smaller than topk: pad COLUMNS with the sentinel so the result
        # keeps its static [T, topk] width instead of shrinking.
        tail = torch.full(
            (num_tokens, topk - k_eff),
            pad_index,
            dtype=indices.dtype,
            device=indices.device,
        )
        indices = torch.cat((indices, tail), dim=-1)
    return indices
