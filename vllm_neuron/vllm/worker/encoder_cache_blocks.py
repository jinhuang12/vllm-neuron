# SPDX-License-Identifier: Apache-2.0
"""On-device block-based encoder cache for multimodal embeddings.

One mm_item per block (no cross-item sharing), pre-allocated contiguous
device buffer. The vision encoder graph writes directly into this buffer
via input-output aliasing (same pattern as KV cache). Eviction returns
blocks to free queue (no device-side memory ops).

Buffer layout::

    buffer: [num_cache_blocks, block_size, fat_dim]

    ┌──────────────┬──────────────┬──────────────┬───┬──────────────┐
    │ block 0      │ block 1      │ block 2      │...│ block N-1    │
    │[block_size,D]│[block_size,D]│[block_size,D]│   │ (scratch)    │
    └──────────────┴──────────────┴──────────────┴───┴──────────────┘

    slot_map (mm_hash → block_ids):
      "img_hash_A" → [0]         (1 image = 1 block)
      "img_hash_B" → [2]         (1 image = 1 block)
      "video_hash" → [3, 4, 5]   (1 video = multiple blocks, frames packed within)

    block N-1 is the scratch block (absorbs padding writes, never allocated).

Writes into the buffer happen inside the vision encoder graph via
input-output aliasing (same pattern as KV cache). Reads happen outside
the graph in eager mode via zero-copy views (``buffer[block_id]``)
passed as a tuple to the prefill graph.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class SlotEntry:
    """Bookkeeping for one cached mm_item."""

    block_ids: list[int]
    tokens_per_block: list[int]
    write_time: float
    # Real (non-pad) merged *tokens* physically stored in each block, one entry
    # per block_id. For videos, when frames don't tile blocks exactly, a
    # non-last block has a trailing pad; for images, this is the dense expansion
    # [block_size, ..., remainder]. Readers use this to skip pad and enforce the
    # invariant: sum(tokens_per_block) == total valid tokens, len(tokens_per_block)
    # == len(block_ids) == block count.


class EncoderCacheBlocks:
    """Block-based on-device encoder cache (one mm_item per block).

    Pre-allocates a contiguous buffer [num_cache_blocks, block_size, fat_dim]
    on device at init time. Each mm_item occupies one or more blocks (one item
    per block, no cross-item packing).

    This class manages **allocation only** — it tracks which blocks are assigned
    to which mm_hash and maintains the free queue. Actual data writes happen
    inside the vision encoder graph (which takes `self.buffer` as an
    input-output alias and scatter-writes merged embeddings at the allocated
    block positions). Reads happen outside the prefill graph: the model runner
    builds zero-copy views (``buffer[i]`` per block) and passes them as a tuple
    to the prefill graph, which uses ``torch.stack`` to assemble them inside
    the compiled graph.

    Args:
        num_blocks: Total number of cache blocks.
        block_size: Tokens per block (post-merger, matches vision encoder output block size).
        fat_dim: Embedding width = ``out_hidden_size * (1 + num_deepstack_levels)``.
            Concatenates the main visual embedding with deepstack residuals.
        dtype: Data type for the buffer.
        device: Device to allocate buffer on.
        min_hold_time_ms: Minimum time (ms) a block must stay allocated after
            write before it can be freed. Prevents premature reuse when a
            remote reader hasn't finished pulling. 0 = instant free.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        fat_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
        min_hold_time_ms: float = 100.0,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.fat_dim = fat_dim
        self.dtype = dtype
        self._device = device
        self.min_hold_time_ms = min_hold_time_ms

        # Last block is reserved as scratch — absorbs padding writes from
        # vision encoder bucket padding. Never allocated to real items.
        self.scratch_block_id = num_blocks - 1

        self.buffer = torch.zeros(
            (num_blocks, block_size, fat_dim), dtype=dtype, device=device
        )

        self._slot_map: dict[str, SlotEntry] = {}
        # Free queue excludes the scratch block
        self._free_queue: deque[int] = deque(range(num_blocks - 1))
        # mm_hash → (block_ids, write_time) for blocks in hold-time window
        self._held_blocks: dict[str, tuple[list[int], float]] = {}

        logger.debug(
            "[EncoderCacheBlocks] init: num_blocks=%d (allocatable=%d, scratch=%d), "
            "block_size=%d, fat_dim=%d, min_hold_time_ms=%.0f, "
            "buffer=%.0f MB, device=%s",
            num_blocks,
            num_blocks - 1,
            self.scratch_block_id,
            block_size,
            fat_dim,
            min_hold_time_ms,
            num_blocks * block_size * fat_dim * 2 / (1024 * 1024),
            device,
        )

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_queue)

    def contains(self, mm_hash: str) -> bool:
        return mm_hash in self._slot_map

    @staticmethod
    def dense_tokens_per_block(num_tokens: int, block_size: int) -> list[int]:
        """Dense per-block token counts for a contiguous item: [block_size, ..., remainder]."""
        n = math.ceil(num_tokens / block_size)
        return [min(block_size, num_tokens - j * block_size) for j in range(n)]

    def allocate(
        self,
        mm_hash: str,
        tokens_per_block: list[int],
    ) -> list[int]:
        """Reserve blocks for an mm_item. No data is written.

        The caller passes the returned block_ids to the vision encoder graph
        as `write_block_ids` so it can scatter-write directly into
        `self.buffer` at those positions.

        Self-healing: if free blocks are insufficient, attempts to reclaim
        expired held blocks. If still insufficient and held blocks exist
        that haven't expired yet, waits (bounded by min_hold_time_ms)
        for them to expire before giving up.

        Args:
            mm_hash: Unique identifier for this mm_item.
            tokens_per_block: Real (non-pad) merged tokens per reserved block,
                one entry per block; its length sets the block count, which can
                exceed the dense ceil(num_tokens / block_size) when a video's
                whole-frame packing leaves per-block pad. Stored on the
                SlotEntry so readers skip each block's pad tail.

        Returns:
            List of allocated block_ids.
            If mm_hash is already allocated, returns existing block_ids.

        Raises:
            RuntimeError: If the cache cannot satisfy the request after
                reclaiming all held blocks.
        """
        t0 = time.monotonic()
        existing = self._slot_map.get(mm_hash)
        if existing is not None:
            logger.debug(
                "[EncoderCacheBlocks] allocate(%s): cache hit, returning existing block_ids=%s",
                mm_hash[:12],
                existing.block_ids,
            )
            return existing.block_ids

        # If this mm_hash was recently freed and is in held state, reclaim
        # its blocks immediately — re-allocating the same item means the
        # hold is no longer needed (we're about to overwrite the data).
        held = self._held_blocks.pop(mm_hash, None)
        if held is not None:
            held_block_ids, _ = held
            for block_id in held_block_ids:
                self._free_queue.append(block_id)
            logger.debug(
                "[EncoderCacheBlocks] allocate(%s): reclaimed %d held blocks for same hash",
                mm_hash[:12],
                len(held_block_ids),
            )

        num_needed = len(tokens_per_block)

        if num_needed > len(self._free_queue):
            reclaimed = self.reclaim_held()
            logger.debug(
                "[EncoderCacheBlocks] allocate(%s): reclaimed %d expired held blocks",
                mm_hash[:12],
                reclaimed,
            )

        if num_needed > len(self._free_queue) and self._held_blocks:
            logger.warning(
                "[EncoderCacheBlocks] cache pressure: need %d blocks but only %d free. "
                "Waiting up to %.0fms for %d held blocks to expire. "
                "This stalls the inference thread and degrades TTFT. "
                "Cause: scheduler's token-based budget doesn't account for "
                "per-block padding waste in the physical buffer layout.",
                num_needed,
                len(self._free_queue),
                self.min_hold_time_ms,
                len(self._held_blocks),
            )
            self._wait_and_reclaim(num_needed)

        if num_needed > len(self._free_queue):
            raise RuntimeError(
                f"Encoder cache full: cannot allocate mm_hash={mm_hash} "
                f"({sum(tokens_per_block)} tokens, {num_needed} blocks needed, "
                f"{len(self._free_queue)} free, "
                f"{len(self._slot_map)} active, "
                f"{len(self._held_blocks)} held). "
                f"Scheduler should have evicted before dispatching."
            )

        block_ids = [self._free_queue.popleft() for _ in range(num_needed)]

        self._slot_map[mm_hash] = SlotEntry(
            block_ids=block_ids,
            tokens_per_block=tokens_per_block,
            write_time=0.0,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug(
            "[EncoderCacheBlocks] allocate(%s): %d tokens → %d blocks, free=%d, active=%d, held=%d, took=%.1fms",
            mm_hash[:12],
            sum(tokens_per_block),
            num_needed,
            len(self._free_queue),
            len(self._slot_map),
            len(self._held_blocks),
            elapsed_ms,
        )
        return block_ids

    def mark_written(self, mm_hash: str) -> None:
        """Update write_time after data is valid in the buffer.

        Called after embed_multimodal() completes — the vision encoder graph
        has finished writing. This ensures the hold-time guard measures from
        when data is actually available, not from when blocks were reserved.
        """
        entry = self._slot_map.get(mm_hash)
        if entry is not None:
            entry.write_time = time.monotonic()
            logger.debug(
                "[EncoderCacheBlocks] mark_written(%s): block_ids=%s",
                mm_hash[:12],
                entry.block_ids,
            )

    def _wait_and_reclaim(self, num_needed: int) -> None:
        """Wait for held blocks to expire until enough are free.

        Sleeps in short intervals until either enough blocks are reclaimed
        or all held blocks have been reclaimed (and it's still not enough).
        Total wait is bounded by min_hold_time_ms.
        """
        deadline = time.monotonic() + self.min_hold_time_ms / 1000
        poll_interval = 0.001  # 1ms

        while len(self._free_queue) < num_needed and self._held_blocks:
            now = time.monotonic()
            if now >= deadline:
                break
            # Sleep until the next earliest expiry or poll_interval
            earliest_expiry = min(
                wt + self.min_hold_time_ms / 1000
                for _, (_, wt) in self._held_blocks.items()
            )
            sleep_time = min(earliest_expiry - now, poll_interval)
            if sleep_time > 0:
                time.sleep(sleep_time)
            self.reclaim_held()

    def get_block_ids(self, mm_hash: str) -> list[int] | None:
        """Get allocated block IDs for a cached mm_item.

        Used by the model runner to build zero-copy views from the
        cache buffer for the prefill graph input.
        """
        entry = self._slot_map.get(mm_hash)
        if entry is None:
            return None
        return entry.block_ids

    def get_num_tokens(self, mm_hash: str) -> int | None:
        """Get the number of valid tokens stored for an mm_item.

        Used to compute vision_positions and determine how many tokens
        to gather from each block.
        """
        entry = self._slot_map.get(mm_hash)
        if entry is None:
            return None
        return sum(entry.tokens_per_block)

    def get_tokens_per_block(self, mm_hash: str) -> list[int] | None:
        """Real (non-pad) merged tokens stored in each block for an mm_item.

        For videos, when frames don't tile blocks exactly, non-last blocks may
        have pad; for images, this is the dense expansion [block_size, ..., remainder].
        Readers use this to skip pad. Always an explicit list; sum == get_num_tokens.
        """
        entry = self._slot_map.get(mm_hash)
        if entry is None:
            return None
        return entry.tokens_per_block

    def free(self, mm_hash: str) -> None:
        """Free blocks for an mm_hash, returning them to the free queue.

        If min_hold_time_ms > 0 and the entry was written recently, the
        blocks are held and will be reclaimed on a later reclaim_held() call.
        """
        entry = self._slot_map.pop(mm_hash, None)
        if entry is None:
            return

        if self.min_hold_time_ms > 0:
            elapsed_ms = (time.monotonic() - entry.write_time) * 1000
            if elapsed_ms < self.min_hold_time_ms:
                self._held_blocks[mm_hash] = (entry.block_ids, entry.write_time)
                logger.debug(
                    "[EncoderCacheBlocks] free(%s): HELD (elapsed=%.0fms < hold=%.0fms), blocks=%s, free=%d, held=%d",
                    mm_hash[:12],
                    elapsed_ms,
                    self.min_hold_time_ms,
                    entry.block_ids,
                    len(self._free_queue),
                    len(self._held_blocks),
                )
                return

        elapsed_ms_free = (
            (time.monotonic() - entry.write_time) * 1000
            if entry.write_time > 0.0
            else None
        )
        for block_id in entry.block_ids:
            self._free_queue.append(block_id)
        logger.debug(
            "[EncoderCacheBlocks] free(%s): FREED immediately (elapsed=%s), blocks=%s, free=%d",
            mm_hash[:12],
            f"{elapsed_ms_free:.0f}ms" if elapsed_ms_free is not None else "unwritten",
            entry.block_ids,
            len(self._free_queue),
        )

    def reclaim_held(self) -> int:
        """Reclaim blocks whose hold time has expired.

        Returns:
            Number of blocks reclaimed.
        """
        if not self._held_blocks:
            return 0

        reclaimed = 0
        expired = []
        now = time.monotonic()
        for mm_hash, (block_ids, write_time) in self._held_blocks.items():
            if (now - write_time) * 1000 >= self.min_hold_time_ms:
                expired.append(mm_hash)

        for mm_hash in expired:
            block_ids, _ = self._held_blocks.pop(mm_hash)
            for block_id in block_ids:
                self._free_queue.append(block_id)
            reclaimed += len(block_ids)

        return reclaimed

    def clear(self) -> None:
        """Reset all state, returning all blocks to free queue."""
        self._slot_map.clear()
        self._held_blocks.clear()
        self._free_queue = deque(range(self.num_blocks - 1))  # exclude scratch
        self.buffer.zero_()

    # ------------------------------------------------------------------
    # Read/write helpers for snapshot, serialization, and prompt_embeds.
    # ------------------------------------------------------------------

    def get(self, mm_hash: str) -> torch.Tensor | None:
        """Read cached entry by gathering from buffer.

        Returns [num_tokens, fat_dim] or None. Used by snapshot/serialization
        path — not the hot path (prefill reads via zero-copy views).
        """
        entry = self._slot_map.get(mm_hash)
        if entry is None:
            return None

        blocks = []
        for block_id, chunk_size in zip(entry.block_ids, entry.tokens_per_block):
            blocks.append(self.buffer[block_id, :chunk_size])

        if len(blocks) == 1:
            return blocks[0]
        return torch.cat(blocks, dim=0)

    def put(self, mm_hash: str, tensor: torch.Tensor) -> None:
        """Allocate blocks and write tensor data into the buffer.

        Used by the prompt_embeds passthrough path (non-vision modalities
        that store embeddings directly). For vision, the encoder graph writes
        directly via index_put_ — this method is not on the vision hot path.
        """
        if mm_hash in self._slot_map:
            return
        num_tokens = tensor.shape[0]
        block_ids = self.allocate(
            mm_hash, self.dense_tokens_per_block(num_tokens, self.block_size)
        )
        tokens_written = 0
        for block_id in block_ids:
            chunk_size = min(self.block_size, num_tokens - tokens_written)
            self.buffer[block_id, :chunk_size] = tensor[
                tokens_written : tokens_written + chunk_size
            ]
            if chunk_size < self.block_size:
                self.buffer[block_id, chunk_size:] = 0
            tokens_written += chunk_size
        self.mark_written(mm_hash)

    def items(self):
        """Iterate (mm_hash, tensor) pairs. Used by snapshot/serialization."""
        for mm_hash in self._slot_map:
            yield mm_hash, self.get(mm_hash)
