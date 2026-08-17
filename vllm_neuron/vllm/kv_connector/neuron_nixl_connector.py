# SPDX-License-Identifier: Apache-2.0
"""
NeuronNixlConnector — standalone NIXL connector for Neuron with DCP support.

Replaces the monkey-patch approach. Users specify this connector directly:

    --kv-transfer-config '{
        "kv_connector": "NeuronNixlConnector",
        "kv_connector_module_path": "vllm_neuron.vllm.kv_connector.neuron_nixl_connector",
        ...
    }'

Supports all DCP DI topologies via unified head_ratio / seq_ratio math:
  - DCP prefill → TP decode
  - DCP prefill → DCP decode (same or different DCP degrees)
  - TP prefill → DCP decode
  - Standard TP → TP (passthrough)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
    NixlConnectorMetadata,
    NixlHandshakePayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.scheduler import (
    NixlConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
    TPMapping,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorRole,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class NeuronNixlAgentMetadata(NixlAgentMetadata):
    """Extended metadata with Neuron DCP info.

    The prefill advertises its DCP degree so the decode can compute
    correct remote rank mappings without user config.
    """

    dcp_size: int = 0
    physical_blocks_per_logical_kv_block: int = 1
    attn_backend_name: str = "neuron"


class NeuronNixlConnectorWorker(NixlConnectorWorker):
    """NIXL connector worker with DCP-aware KV transfer.

    Uses unified head_ratio / seq_ratio logic to handle all topologies:
    - head_ratio > 1: split (remote block has more heads, read a subset)
    - head_ratio < 1: merge (remote block has fewer heads, read multiple)
    - seq_ratio > 1: read from multiple remote DCP ranks
    - seq_ratio <= 1: read from subset of one remote DCP rank's blocks
    """

    def __init__(self, *args, **kwargs):
        # NixlBaseConnectorWorker._nixl_handshake (in nixl.base_worker) builds
        # msgspec.msgpack.Decoder(NixlAgentMetadata), resolving NixlAgentMetadata
        # from base_worker's own module namespace, where it was bound at import
        # via `from ...metadata import NixlAgentMetadata`. Rebinding the attribute
        # on the metadata module alone does NOT change that already-imported name,
        # so the decoder keeps using the base class and our `dcp_size` field is
        # dropped. Rebind it on base_worker (where the decoder resolves it) so the
        # prefill's dcp_size is decoded, and also on the metadata module for any
        # consumer that resolves it there.
        import vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata as _nixl_mod
        import vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker as _nixl_base_worker

        if _nixl_mod.NixlAgentMetadata is not NeuronNixlAgentMetadata:
            _nixl_mod.NixlAgentMetadata = NeuronNixlAgentMetadata
        if _nixl_base_worker.NixlAgentMetadata is not NeuronNixlAgentMetadata:
            _nixl_base_worker.NixlAgentMetadata = NeuronNixlAgentMetadata

        super().__init__(*args, **kwargs)
        self._local_dcp_size: int = 1
        self._local_dcp_rank: int = 0
        self._remote_dcp: dict[str, int] = {}
        self._head_ratio: dict[str, float] = {}
        self._remote_tp_size: dict[str, int] = {}
        self._inverse_handle_cache: dict[str, list] = {}
        # Per-engine remote geometry for our DCP paths. add_remote_agent fully
        # replaces the base (no super()), so we own these dicts and their
        # cleanup (see _cleanup_remote_engine).
        self._tp_size: dict[str, int] = {}
        self._block_size: dict[str, int] = {}

    # ── Override: register_kv_caches ─────────────────────────────────────

    def register_kv_caches(self, kv_caches):
        import msgspec

        super().register_kv_caches(kv_caches)

        dcp = self.vllm_config.parallel_config.decode_context_parallel_size
        is_prefill = self.kv_transfer_config.kv_role == "kv_producer"

        if dcp > 1 and is_prefill:
            agent_meta = NeuronNixlAgentMetadata(
                engine_id=self.engine_id,
                agent_metadata=self.nixl_wrapper.get_agent_metadata(),
                device_id=self.device_id,
                kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id][
                    self.tp_rank
                ],
                num_blocks=self.num_blocks,
                block_lens=self.block_len_per_layer,
                kv_cache_layout=self.kv_cache_layout
                if not self.use_host_buffer
                else self.host_buffer_kv_cache_layout,
                block_size=self.block_size,
                ssm_sizes=self._mamba_ssm_size,
                attn_backend_name="neuron",
                physical_blocks_per_logical_kv_block=1,
                dcp_size=dcp,
            )
            encoder = msgspec.msgpack.Encoder()
            self.xfer_handshake_metadata = NixlHandshakePayload(
                compatibility_hash=self.compat_hash,
                agent_metadata_bytes=encoder.encode(agent_meta),
            )
            logger.info("DCP NIXL: prefill metadata tagged with dcp_size=%d", dcp)
        else:
            # Decode: store local DCP rank for block filtering.
            from vllm.distributed.parallel_state import get_dcp_group

            g = get_dcp_group()
            self._local_dcp_size = g.world_size
            self._local_dcp_rank = g.rank_in_group

        # Patch tp_ratio to always return 1, preventing upstream from
        # applying its own head splitting logic. We handle all DCP
        # topologies in our _read_blocks_for_req / add_remote_agent.
        if self.transfer_topo is not None:
            self.transfer_topo.tp_ratio = lambda remote_tp_size: 1

        if self._local_dcp_size > 1:
            logger.info(
                "DCP NIXL: local decode dcp_size=%d, dcp_rank=%d, tp_rank=%d",
                self._local_dcp_size,
                self._local_dcp_rank,
                self.tp_rank,
            )

    # ── Override: _validate_remote_agent_handshake ────────────────────────

    def _validate_remote_agent_handshake(self, nixl_agent_meta, remote_tp_size):
        """Skip upstream strict validation for all DCP topologies as the upstream doesn't consider DCP configs."""
        assert self.transfer_topo is not None
        # both should have num_kv_cache_groups entres
        assert len(nixl_agent_meta.kv_caches_base_addr) == len(
            self.block_len_per_layer
        ), "KV group count mismatch between P and D"
        remote_block_len = nixl_agent_meta.block_lens[0]
        local_block_len = self.block_len_per_layer[0]
        assert all(bl == remote_block_len for bl in nixl_agent_meta.block_lens), (
            f"Remote block_lens not uniform: {nixl_agent_meta.block_lens}"
        )
        assert all(bl == local_block_len for bl in self.block_len_per_layer), (
            f"Local block_lens not uniform: {self.block_len_per_layer}"
        )
        ratio = remote_block_len / local_block_len
        inv_ratio = local_block_len / remote_block_len
        assert ratio == int(ratio) or inv_ratio == int(inv_ratio), (
            f"Block lens must be integer-divisible: "
            f"remote_block_len={remote_block_len}, local_block_len={local_block_len}"
        )

    # ── Override: add_remote_agent ────────────────────────────────────────

    def add_remote_agent(self, nixl_agent_meta, remote_tp_rank=0, remote_tp_size=1):
        """Register remote agent with head_ratio-aware descriptors."""
        eid = nixl_agent_meta.engine_id
        remote_dcp = getattr(nixl_agent_meta, "dcp_size", 0) or 1
        remote_block_len = nixl_agent_meta.block_lens[0]
        local_block_len = self.block_len_per_layer[0]

        head_ratio = remote_block_len / local_block_len

        if eid not in self._remote_dcp:
            self._remote_dcp[eid] = remote_dcp
            self._head_ratio[eid] = head_ratio
            self._remote_tp_size[eid] = remote_tp_size
            # Seed an inert tp_mappings[eid] (we never read it) so the base's
            # unguarded `del tp_mappings[eid]` on TTL eviction doesn't KeyError —
            # our add_remote_agent replaces the base and never populated it.
            self.tp_mappings[eid] = TPMapping(
                source_ranks_per_group=(),
                all_source_ranks=(),
                rank_to_attention_slot={},
                rank_offset_factor=0,
            )
            logger.info(
                "DCP NIXL: engine %s head_ratio=%.2f, remote_dcp=%d, "
                "remote_block_len=%d, local_block_len=%d, "
                "P_TP=%d, D_TP=%d, D_DCP=%d",
                eid,
                head_ratio,
                remote_dcp,
                remote_block_len,
                local_block_len,
                remote_tp_size,
                self.world_size,
                self._local_dcp_size,
            )

        # For split/merge/match we handle descriptor registration ourselves.
        # Match case (head_ratio=1) also goes through our path to avoid
        # upstream's _group_spec_types / tp_mapping logic which breaks for DCP.
        if eid not in self.dst_num_blocks:
            self.dst_num_blocks[eid] = (
                nixl_agent_meta.num_blocks
            )  # total capacity (all slots in the KV cache), not specific to a certain request

        self.kv_caches_base_addr[eid][remote_tp_rank] = (
            nixl_agent_meta.kv_caches_base_addr
        )

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )
        # Don't write self._remote_agents here. The base publishes the full
        # per-rank dict atomically (from our return value) once the background
        # handshake finishes; an incremental write flips its completion guard
        # early and reads start mid-handshake -> KeyError on unregistered ranks.

        # Register with TransferTopology so _read_blocks can resolve get_engine_info()
        assert self.transfer_topo is not None
        if eid not in self.transfer_topo._engines:
            from vllm.distributed.kv_transfer.kv_connector.utils import (
                EngineTransferInfo,
            )

            self.transfer_topo.register_remote_engine(
                eid,
                EngineTransferInfo(
                    remote_tp_size=remote_tp_size,
                    remote_block_len=nixl_agent_meta.block_lens[0],
                    remote_block_size=nixl_agent_meta.block_size,
                    remote_physical_blocks_per_logical=1,
                ),
            )

        if head_ratio > 1:
            # Split: remote block has more heads than local.
            num_kv_heads = self.vllm_config.model_config.get_total_num_kv_heads()
            D_ranks_per_dcp_group = self.world_size // self._local_dcp_size
            D_num_kv_replica = max(1, D_ranks_per_dcp_group // num_kv_heads)
            D_kv_head_rank = self.tp_rank // (self._local_dcp_size * D_num_kv_replica)
            head_idx = D_kv_head_rank % int(head_ratio)
            head_offset = head_idx * local_block_len
            self._register_remote_descs(
                eid,
                nixl_agent_meta,
                remote_agent_name,
                remote_tp_rank,
                desc_len=local_block_len,
                offset=head_offset,
            )
        else:
            # Merge: register full remote block.
            self._register_remote_descs(
                eid,
                nixl_agent_meta,
                remote_agent_name,
                remote_tp_rank,
                desc_len=remote_block_len,
                offset=0,
            )

        return remote_agent_name

    def _register_remote_descs(
        self,
        eid,
        nixl_agent_meta,
        remote_agent_name,
        remote_tp_rank,
        desc_len,
        offset,
    ):
        """Register NIXL descriptors for a remote agent's blocks."""
        kv_addrs = nixl_agent_meta.kv_caches_base_addr
        blocks_data = []
        for kv_group_idx, base_addr in enumerate(kv_addrs):
            page_size = nixl_agent_meta.block_lens[kv_group_idx]
            for block_id in range(nixl_agent_meta.num_blocks):
                addr = base_addr + block_id * page_size + offset
                blocks_data.append((addr, desc_len, nixl_agent_meta.device_id))
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        handle = self.nixl_wrapper.prep_xfer_dlist(remote_agent_name, descs)
        self.dst_xfer_side_handles[eid][remote_tp_rank] = handle

    # ── Override: _cleanup_remote_engine ──────────────────────────────────

    def _cleanup_remote_engine(self, engine_id, *args, **kwargs):
        """Tear down per-engine state on TTL eviction / shutdown.

        Since we own add_remote_agent, we own the matching cleanup: release our
        own NIXL dlist handles (_inverse_handle_cache — the base doesn't know
        about them, so they'd leak), delegate the base-owned state to super(),
        then clear our extra geometry dicts so a re-registered engine starts
        clean. Handle release goes first in a try/finally so it always happens
        even if super() raises.
        """
        try:
            for handle in self._inverse_handle_cache.get(engine_id, []):
                self.nixl_wrapper.release_dlist_handle(handle)
        finally:
            super()._cleanup_remote_engine(engine_id, *args, **kwargs)
            self._inverse_handle_cache.pop(engine_id, None)
            self._remote_dcp.pop(engine_id, None)
            self._head_ratio.pop(engine_id, None)
            self._remote_tp_size.pop(engine_id, None)
            self._tp_size.pop(engine_id, None)
            self._block_size.pop(engine_id, None)

    # ── Override: _nixl_handshake ─────────────────────────────────────────

    def _nixl_handshake(self, host, port, remote_tp_size, expected_engine_id):
        """Connect to all remote ranks unconditionally.

        We cannot know the remote's DCP degree before the handshake, so
        always register all remote ranks. The read logic selects the
        correct subset at transfer time.
        """
        self.transfer_topo.handshake_target_ranks = lambda rtp: list(range(rtp))
        try:
            result = super()._nixl_handshake(
                host, port, remote_tp_size, expected_engine_id
            )
        finally:
            if "handshake_target_ranks" in self.transfer_topo.__dict__:
                del self.transfer_topo.__dict__["handshake_target_ranks"]
        return result

    # ── Override: _send_heartbeats ────────────────────────────────────────

    def _send_heartbeats(self, metadata: NixlConnectorMetadata) -> None:
        """Send heartbeats, tolerating concurrent remote-agent registration.

        Upstream iterates ``self._remote_agents[engine_id].values()`` directly.
        Handshakes run on a background ``_handshake_initiation_executor`` thread
        whose done-callback assigns ``self._remote_agents[eid] = ...``, so a
        handshake completing mid-iteration raises ``RuntimeError: dictionary
        changed size during iteration``, which kills the decode EngineCore and
        surfaces to the client as a 500. This is most likely with the wide
        remote-TP producers used in DCP DI (e.g. tp32_dcp4). Snapshotting the
        agent list before iterating closes that race; a heartbeat that misses a
        just-registered agent is picked up on the next cycle.
        """
        for engine_id, hb_info in metadata.heartbeat_by_engine.items():
            # Proactive handshake so the next heartbeat for this remote can go
            # through; skip sending while it is still pending.
            if (
                self._ensure_handshake(
                    engine_id, hb_info.host, hb_info.port, hb_info.tp_size
                )
                is not None
            ):
                continue

            hb_msg = ("HB:" + ",".join(hb_info.req_ids)).encode()
            for agent_name in list(self._remote_agents[engine_id].values()):
                try:
                    self.nixl_wrapper.send_notif(agent_name, notif_msg=hb_msg)
                except Exception:
                    logger.debug(
                        "Failed to send heartbeat to engine %s",
                        engine_id,
                        exc_info=True,
                    )

    # ── Override: _read_blocks_for_req ────────────────────────────────────

    def _read_blocks_for_req(self, req_id, meta):
        """Unified DCP-aware block reading using head topology and seq_ratio."""
        assert meta.remote is not None and self.transfer_topo is not None
        eid = meta.remote.engine_id

        P_DCP = self._remote_dcp.get(eid, 1)
        D_DCP = self._local_dcp_size
        head_ratio = self._head_ratio.get(eid, 1.0)

        P_TP = self._remote_tp_size[eid]
        seq_ratio = P_DCP / D_DCP

        D_rank = self.tp_rank
        D_DCP_rank = self._local_dcp_rank

        num_kv_heads = self.vllm_config.model_config.get_total_num_kv_heads()
        D_ranks_per_dcp_group = self.world_size // D_DCP
        P_ranks_per_dcp_group = P_TP // P_DCP
        D_num_kv_replica = max(1, D_ranks_per_dcp_group // num_kv_heads)
        P_num_kv_replica = max(1, P_ranks_per_dcp_group // num_kv_heads)
        D_kv_head_rank = D_rank // (D_DCP * D_num_kv_replica)

        # --- Determine which prefill KV head ranks to read from ---
        if head_ratio >= 1:
            P_kv_head_ranks_to_read = [D_kv_head_rank // int(head_ratio)]
            split_local_handles = None
            remote_block_size = self.transfer_topo.get_engine_info(
                eid
            ).remote_block_size
            local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                remote_block_size
            ]
        else:
            inverse_head_ratio = int(1 / head_ratio)
            P_kv_head_ranks_to_read = [
                D_kv_head_rank * inverse_head_ratio + j
                for j in range(inverse_head_ratio)
            ]
            split_local_handles = self._get_or_create_inverse_handles(
                eid, inverse_head_ratio
            )

        # --- Determine which prefill DCP ranks and block mappings ---
        # Anchor num_groups on the LOCAL side (matching upstream): a DI pull can
        # arrive with zero local blocks (prefix-cache hit), and anchoring on the
        # remote count would then IndexError the empty local.
        num_groups = len(meta.local_physical_block_ids)
        if num_groups == 0:
            # Nothing to write, but still notify the prefill rank(s) this decode
            # rank maps to so the producer releases the KV (same p_rank_id as the
            # real path). Empty _read_blocks issues the notif without recording a
            # transfer; recording one would trip the scheduler's req-ownership
            # assert for a request this engine no longer owns (DP>1).
            if seq_ratio > 1:
                p_dcp_ranks = [D_DCP_rank + i * D_DCP for i in range(int(seq_ratio))]
            else:
                p_dcp_ranks = [D_DCP_rank % P_DCP]
            for p_dcp_rank in p_dcp_ranks:
                for p_kv_head_rank in P_kv_head_ranks_to_read:
                    p_rank_id = p_kv_head_rank * (P_DCP * P_num_kv_replica) + p_dcp_rank
                    self._read_blocks(
                        read_spec=ReadSpec(
                            remote_rank=p_rank_id,
                            local_block_ids=[],
                            remote_block_ids=[],
                        ),
                        dst_engine_id=eid,
                        request_id=req_id,
                        remote_request_id=meta.remote.request_id,
                        local_xfer_side_handle=(
                            split_local_handles[0]
                            if split_local_handles is not None
                            else local_xfer_side_handle
                        ),
                        remote_xfer_side_handle=self.dst_xfer_side_handles[eid][
                            p_rank_id
                        ],
                    )
            return

        if seq_ratio > 1:
            # DCP>1 isn't used by GPT-OSS today (always DCP=1, the branch below),
            # so this path still front-aligns and has the same prefix-cache-hit bug
            # we fix there. Doing it right here means handling the round-robin block
            # interleave across DCP ranks; leaving it until we have a config to test.
            int_seq_ratio = int(seq_ratio)
            P_DCP_ranks_to_read = [D_DCP_rank + i * D_DCP for i in range(int_seq_ratio)]
            remote_blk_id_indices_per_dcp_rank = []
            local_blk_id_indices_per_dcp_rank = []
            for k in range(int_seq_ratio):
                remote_indices_per_group = []
                local_indices_per_group = []
                for g in range(num_groups):
                    num_remote_blks = len(meta.remote.block_ids[g])
                    num_local_blks = len(meta.local_physical_block_ids[g])
                    local_indices = [
                        k + i * int_seq_ratio
                        for i in range(num_remote_blks)
                        if k + i * int_seq_ratio < num_local_blks
                    ]
                    remote_indices_per_group.append(list(range(len(local_indices))))
                    local_indices_per_group.append(local_indices)
                remote_blk_id_indices_per_dcp_rank.append(remote_indices_per_group)
                local_blk_id_indices_per_dcp_rank.append(local_indices_per_group)
        else:
            P_DCP_ranks_to_read = [D_DCP_rank % P_DCP]
            step = D_DCP // P_DCP
            offset = D_DCP_rank // P_DCP
            remote_indices_per_group = []
            local_indices_per_group = []
            for g in range(num_groups):
                num_local_blks = len(meta.local_physical_block_ids[g])
                num_remote_blks = len(meta.remote.block_ids[g])
                # This rank's slice of the remote blocks ([0..N-1] when DCP=1).
                strided = list(range(offset, num_remote_blks, step))
                # On a partial prefix-cache hit the decode side already has the
                # leading blocks, so only the tail is left to pull -- read the last
                # num_local blocks, matching upstream _apply_prefix_caching
                # (remote[-num_local:]). Reading from the front would copy the
                # cached prefix into the tail slots and decode the wrong tokens.
                # A full pull is unchanged.
                if num_local_blks <= 0:
                    remote_indices = []
                elif num_local_blks < len(strided):
                    remote_indices = strided[-num_local_blks:]
                else:
                    remote_indices = strided
                remote_indices_per_group.append(remote_indices)
                local_indices_per_group.append(list(range(len(remote_indices))))
            remote_blk_id_indices_per_dcp_rank = [remote_indices_per_group]
            local_blk_id_indices_per_dcp_rank = [local_indices_per_group]

        def get_p_rank_id(p_dcp_rank, p_kv_head_rank):
            # Prefill now uses the SAME full-TP + token-interleaved-KV layout
            # as decode: each KV-head group is a block of consecutive ranks,
            # with dcp_rank the inner (fastest-varying) dimension:
            #   rank = kv_head_rank * (P_DCP * P_num_kv_replica)
            #          + replica_idx * P_DCP + dcp_rank
            # Pick replica 0 (replicas hold identical KV). For P_DCP == 1 this
            # reduces to the standard-TP mapping kv_head_rank * P_num_kv_replica.
            return p_kv_head_rank * (P_DCP * P_num_kv_replica) + p_dcp_rank

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Unified read: req=%s, D_rank=%d, D_DCP_rank=%d/%d, "
                "D_kv_head_rank=%d, seq_ratio=%.2f, "
                "P_kv_head_ranks=%s, P_DCP_ranks=%s",
                req_id,
                D_rank,
                D_DCP_rank,
                D_DCP,
                D_kv_head_rank,
                seq_ratio,
                P_kv_head_ranks_to_read,
                P_DCP_ranks_to_read,
            )

        # --- Issue reads ---
        for i, p_dcp_rank in enumerate(P_DCP_ranks_to_read):
            remote_indices_per_group = remote_blk_id_indices_per_dcp_rank[i]
            local_indices_per_group = local_blk_id_indices_per_dcp_rank[i]

            if not any(local_indices_per_group):
                continue

            remote_block_ids = [
                [meta.remote.block_ids[g][idx] for idx in remote_indices_per_group[g]]
                for g in range(num_groups)
            ]
            local_block_ids = [
                [
                    meta.local_physical_block_ids[g][idx]
                    for idx in local_indices_per_group[g]
                ]
                for g in range(num_groups)
            ]

            for j, p_kv_head_rank in enumerate(P_kv_head_ranks_to_read):
                p_rank_id = get_p_rank_id(p_dcp_rank, p_kv_head_rank)
                remote_xfer_side_handle = self.dst_xfer_side_handles[eid][p_rank_id]

                if split_local_handles is not None:
                    cur_local_handle = split_local_handles[j]
                else:
                    cur_local_handle = local_xfer_side_handle

                self._read_blocks(
                    read_spec=ReadSpec(
                        remote_rank=p_rank_id,
                        local_block_ids=local_block_ids,
                        remote_block_ids=remote_block_ids,
                    ),
                    dst_engine_id=eid,
                    request_id=req_id,
                    remote_request_id=meta.remote.request_id,
                    local_xfer_side_handle=cur_local_handle,
                    remote_xfer_side_handle=remote_xfer_side_handle,
                )

        # When D_DCP > P_DCP a short prompt can leave some decode DCP ranks with
        # no blocks to read (empty stride above), so they post no transfer and
        # create no _recving_transfers key. get_finished then never reports the
        # req from those ranks, and since completion needs all TP ranks, the req
        # hangs until TTL eviction. Register an empty entry (reported done
        # immediately) for reqs this engine owns. The _recving_metadata gate +
        # the num_groups==0 early-return above keep this off the disowned-abort
        # path, so it can't reintroduce the DP>1 assert this used to trip.
        if req_id in self._recving_metadata and req_id not in self._recving_transfers:
            self._recving_transfers[req_id] = []

    def _get_or_create_inverse_handles(self, eid, inverse_head_ratio):
        """Create local split handles for merge (inverse_head_ratio > 1)."""
        if eid in self._inverse_handle_cache:
            return self._inverse_handle_cache[eid]

        remote_block_len = self.block_len_per_layer[0] // inverse_head_ratio
        local_base_addrs = self.kv_caches_base_addr[self.engine_id][self.tp_rank]
        split_handles = []
        for hp in range(inverse_head_ratio):
            blocks_data = []
            for kv_group_idx, base_addr in enumerate(local_base_addrs):
                local_page_size = self.block_len_per_layer[kv_group_idx]
                for block_id in range(self.num_blocks):
                    addr = (
                        base_addr + block_id * local_page_size + hp * remote_block_len
                    )
                    blocks_data.append((addr, remote_block_len, self.device_id))
            descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
            handle = self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs)
            split_handles.append(handle)
        self._inverse_handle_cache[eid] = split_handles
        return split_handles


class NeuronNixlConnectorScheduler(NixlConnectorScheduler):
    """Scheduler-side connector carrying the DI sliding-window fix.

    On a hybrid SWA model the decode node admits a DI request at the full prompt
    length ``N`` but recomputes the last token at ``N - 1``, so the sliding-window
    skip (anchored at ``N``) evicts the window's oldest block one step too early
    at the ``N % block_size == block_size - 1`` boundary; that block is never
    transferred and decode reads uninitialised K -> garbled / empty output.

    The fix makes both nodes operate at ``N - 1``: the prefill node
    (``do_remote_decode``) pops the last prompt token, the decode node
    (``do_remote_prefill``) lowers the admitted count by one (prompt kept, since
    the decoder recomputes ``N - 1``). Both then reserve the same block count and
    the eviction lines up. It lives here, not in the old ``allocate_slots``
    monkeypatch, so it touches no block-manager internals.

    If a future vLLM does the N-1 dance itself, both sides self-protect: the
    decode gate stops firing once the base returns ``N - 1``, and the prefill pop
    has a tripwire that raises if the base also mutates the prompt length.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        from vllm.v1.kv_cache_interface import SlidingWindowSpec

        # True iff this model has a sliding-window group (the DI over-eviction).
        self._has_swa = any(
            isinstance(g.kv_cache_spec, SlidingWindowSpec)
            for g in kv_cache_config.kv_cache_groups
        )
        # vLLM reserves block_id 0 as the null block and HMA marks SWA-skipped
        # blocks with it at the front of a group; we only see ids here, so keep
        # the named constant. Deferred import matches the SlidingWindowSpec above.
        from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

        self._null_block_id = NULL_BLOCK_ID
        if self._has_swa:
            logger.info(
                "NeuronNixlConnectorScheduler: sliding-window DI truncation active "
                "(%d KV-cache group(s)).",
                len(kv_cache_config.kv_cache_groups),
            )

    def on_new_request(self, request: "Request") -> None:
        """Prefill node: drop the last prompt token at add time so it ships
        ``h(0 .. N-2)``, matching the decode side's N-1 admission.

        Must run here, not in get_num_new_matched_tokens: the base sizes the
        prefix-cache hit before calling that, so popping later could leave
        ``num_new_tokens == 0`` and trip its assert (a prefill crash at
        ``N % block_size == 1``). Idempotent via the ``_swa_di_truncated`` marker.
        """
        params = request.kv_transfer_params
        if (
            self._has_swa
            and params is not None
            and params.get("do_remote_decode")
            and not params.get("_swa_di_truncated")
            and request.num_prompt_tokens > 1
            and request.prompt_token_ids is not None
        ):
            request.prompt_token_ids.pop()
            request._all_token_ids.pop()
            request.num_prompt_tokens -= 1
            params["_swa_di_truncated"] = True
        # Preserve the base's heartbeat tracking (no-op for do_remote_decode reqs).
        super().on_new_request(request)

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        params = request.kv_transfer_params

        # The prefill-side pop now happens in on_new_request (before the base's
        # prefix-cache hit is computed), so nothing to truncate here.
        prompt_len_before = request.num_prompt_tokens
        count, load_async = super().get_num_new_matched_tokens(
            request, num_computed_tokens
        )

        # Tripwire: we popped in on_new_request; if a future vLLM also truncates
        # this prompt, super() would take it to N-2. Fail loud (explicit raise so
        # it survives ``python -O``).
        if (
            self._has_swa
            and params is not None
            and params.get("do_remote_decode")
            and params.get("_swa_di_truncated")
            and request.num_prompt_tokens != prompt_len_before
        ):
            raise RuntimeError(
                "NeuronNixlConnectorScheduler: the base scheduler mutated the "
                f"prompt length ({prompt_len_before} -> {request.num_prompt_tokens})"
                " after this subclass already truncated it for the sliding-window "
                "DI fix. vLLM may now truncate these requests itself; the fix "
                "double-applies. Remove this subclass or re-validate."
            )

        # Decode node: admit N-1 (prompt intact) to retain the window-low block.
        # The ``>= num_prompt_tokens`` guard decrements only a full-prompt pull
        # (a partial prefix-cache pull is untouched), and is False if the base
        # already returns N-1, so we never stack to N-2.
        if (
            self._has_swa
            and params is not None
            and params.get("do_remote_prefill")
            and count > 0
            and num_computed_tokens + count >= request.num_prompt_tokens
        ):
            count -= 1
            if count == 0:
                # Nothing left to pull; load_async=True with count 0 would trip
                # the base's ``assert num_external_computed_tokens > 0``.
                load_async = False

        return count, load_async

    def _strip_leading_null_blocks(self, blocks: list[int]) -> list[int]:
        """Drop the leading run of null blocks from one SWA group's clipped list.

        HMA front-pads SWA groups with null blocks and the base keeps the last
        ``blocks_per_sw`` entries — exact only for a full window. Under N-1
        truncation the window is one block short at ``N % block_size == 1``, so the
        base slice keeps a leading null that the worker mispairs (dropping the
        newest real block). Only the front is ever null, so a full window is
        unchanged (strict no-op).
        """
        n_leading_null = 0
        for block_id in blocks:
            if block_id == self._null_block_id:
                n_leading_null += 1
            else:
                break
        return blocks[n_leading_null:] if n_leading_null else blocks

    def get_sw_clipped_blocks(self, block_ids: "BlockIds") -> "BlockIds":
        """Base SWA unpad, then strip any leading null it left behind (see
        ``_strip_leading_null_blocks``). No-op for full-attn groups and the
        untruncated path, which have no leading null.
        """
        clipped = super().get_sw_clipped_blocks(block_ids)
        if not self._has_swa:
            return clipped
        return tuple(self._strip_leading_null_blocks(list(group)) for group in clipped)


class NeuronNixlConnector(NixlConnector):
    """Neuron NIXL connector with DCP support."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: "KVConnectorRole",
        kv_cache_config: "KVCacheConfig",
    ):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
            KVConnectorRole,
        )

        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.kv_cache_config = kv_cache_config
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self.kv_transfer_config = vllm_config.kv_transfer_config

        if role == KVConnectorRole.SCHEDULER:
            # Subclass carries the DI sliding-window fix. See
            # NeuronNixlConnectorScheduler.
            self.connector_scheduler = NeuronNixlConnectorScheduler(
                vllm_config, self.engine_id, kv_cache_config
            )
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = NeuronNixlConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )
