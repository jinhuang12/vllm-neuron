# SPDX-License-Identifier: Apache-2.0
"""Independent reference for ``NF.kv_cache_write`` (triad LD-79).

Plain torch, OUTSIDE ``vllm_neuron`` (the harness refuses a reference that
resolves inside the port package). A second, independent spelling of the
declared write — never the fallback's:

* the triad's fallback (``_torch_kv_cache_write``, the incumbent
  ``_masked_scatter_rows`` composition) redirects sentinel destinations to
  slot 0 via ``clamp`` and no-op-writes that slot's own content back through
  ``index_select`` + ``where`` + in-place ``index_put_``;
* THIS reference boolean-selects the valid rows only, zero-pads them into an
  fp32 staging block, and merges OUT-OF-PLACE with non-mutating
  ``Tensor.index_put`` — no clamp, no redirect, no ``where``, no
  ``index_select``, no in-place op.

Bit-exactness of the fp32 round trip (why 0/0 tolerances are attainable):
``float8_e4m3fn -> float32`` is exact (every fp8 value is a binary32 value),
and casting an exactly-representable value back rounds to itself, so
untouched slots survive ``fp8 -> f32 -> fp8`` byte-identically. Written slots
carry ``rows.to(cache.dtype)`` — the SAME single cast the incumbent performs
(prereg Z0 §D2: the cast is part of the declared math, not a leg detail) —
then ride the same exact round trip. Zero pad in fp32 casts to fp8 zero
bytes, which are ``torch.zeros(dtype=fp8)`` bytes (``_pad_columns``'s).
randn-scale inputs cannot mint fp8 NaN/Inf (declaration
``_tolerance_derivation``), so no non-roundtripping encoding exists in any
declared case.

Domain note (declaration ``_domain_notes``): duplicate VALID destinations are
outside the op contract — order-unspecified in the incumbent's own
``index_put_`` — and no declared or probe case constructs one; sentinel
duplicates are no-ops on every spelling.
"""

import torch
from torch import Tensor


def kv_cache_write_reference(
    cache: Tensor,
    slot_ids: Tensor,
    rows: Tensor,
    *,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Out-of-place reference: returns a NEW tensor with the declared write
    applied; ``cache`` is never mutated (the harness rebuilds inputs per leg
    from seeds, so mutation-vs-not cannot leak between legs either way)."""
    num_blocks, num_kv_heads, block_size, width = cache.shape

    # The incumbent's single write cast (Z0 §D2) — part of the declared math.
    payload = rows.to(cache.dtype)

    keep = slot_ids.to(torch.int64) >= 0
    if valid_mask is not None:
        keep = keep & valid_mask
    dest = slot_ids.to(torch.int64)[keep]

    # fp32 staging block: selected rows left-aligned, zero pad to width.
    src32 = torch.zeros(dest.shape[0], width, dtype=torch.float32)
    src32[:, : payload.shape[-1]] = payload[keep].to(torch.float32)

    flat32 = cache.reshape(
        num_blocks * num_kv_heads * block_size, width
    ).to(torch.float32)
    merged = flat32.index_put((dest,), src32)  # OUT-OF-PLACE, non-mutating

    return merged.reshape(cache.shape).to(cache.dtype)
