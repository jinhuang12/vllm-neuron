# SPDX-License-Identifier: Apache-2.0
from .argsort_unstable import argsort_unstable
from .spec_decode_correction import (
    correct_spec_decode_positions_and_slot_mapping as correct_spec_decode_positions_and_slot_mapping,
)  # noqa: F401
from .attention.attention_decode_mask import gen_attention_decode_mask
from .attention.attention_decode import attention_decode
from .attention.qkv import qkv_proj
from .attention.o_proj import o_proj
from .attention.attention_cte import flash_attention
from .attention.attention_segmented_cte import (
    segmented_attention,
    segmented_attention_cp,
)
# <-- deepseek_v4: MLA / sliding-window / DSA-indexer ops. Each is a torch
# composition, not an nkilib wrapper: every kernel the port plan's ladder
# cited for these rows is absent from the installed neuron wheel, so each
# row takes its recorded same-row torch-composition rung. Without these
# re-exports `NF.<op>` does not resolve at call time.
from .attention.mla_qkv import mla_qkv
from .attention.mla_oproj import mla_grouped_oproj
from .attention.mla_sparse_attention import mla_sparse_attention
from .attention.mla_decode import mla_decode_attention
from .attention.swa_attention import swa_attention
from .attention.sparse_indexer import sparse_indexer_topk
from .collectives.all_to_all import all_to_all
from .collectives.all_to_all_v import all_to_all_v
from .embedding import embedding
from .expert_parallel import (
    calculate_local_expert_indices,
    validate_expert_parallelism_config,
    get_local_expert_affinities,
)
from .mlp import mlp
from .moe.build_all2all_combine_metadata import build_all2all_combine_metadata
from .moe.build_all2all_dispatch_metadata import build_all2all_dispatch_metadata
from .moe.moe_blockwise import build_blockwise_mapping
from .moe.moe_block_tkg import moe_block_tkg
from .moe.moe_tkg import moe_tkg
from .moe.moe_cte import moe_cte
from .moe.permute_routed_tokens import permute_routed_tokens
from .moe.rmsnorm_router_topk_tkg import rmsnorm_router_topk_tkg
from .moe.router import router
from .moe.topk_reduce import topk_reduce
from .rmsnorm_quant import rmsnorm_quant
from .sampling import sample
from .cumsum import cumsum
from .prompt_embeds import merge_prompt_embeds
from .process_groups import create_row_col_groups, get_group_slice_indices


# Alphabetical
__all__ = [
    "argsort_unstable",
    "all_to_all",
    "all_to_all_v",
    "attention_decode",
    "build_all2all_combine_metadata",
    "build_all2all_dispatch_metadata",
    "build_blockwise_mapping",
    "calculate_local_expert_indices",
    "create_row_col_groups",
    "cumsum",
    "embedding",
    "flash_attention",
    "gen_attention_decode_mask",
    "get_group_slice_indices",
    "get_local_expert_affinities",
    "merge_prompt_embeds",
    "mla_decode_attention",
    "mla_grouped_oproj",
    "mla_qkv",
    "mla_sparse_attention",
    "mlp",
    "moe_block_tkg",
    "moe_cte",
    "moe_tkg",
    "o_proj",
    "permute_routed_tokens",
    "qkv_proj",
    "rmsnorm_quant",
    "rmsnorm_router_topk_tkg",
    "router",
    "sample",
    "segmented_attention",
    "segmented_attention_cp",
    "sparse_indexer_topk",
    "swa_attention",
    "topk_reduce",
    "validate_expert_parallelism_config",
]
