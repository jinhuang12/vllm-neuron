# SPDX-License-Identifier: Apache-2.0
"""ld68_witness functional API — ep18/LD-68 in-execution b2e witness.

INSTRUMENT, not a compute op (port-plan SS15.3). Numerical contract:
IDENTITY PASS-THROUGH of the ``block_to_expert`` stream
(``numerics/ld68_witness.declaration.json``, rtol=atol=0.0 on int32 —
a witness that sanitizes the values it exists to observe is the defect
class the declaration kills). The instrument value — three
``nl.device_print`` emissions per MoE layer instance — is a side effect
the numerics venues cannot observe; its transport/watchdog-survival/volume
proof venue is the LD-69 P-B hardware probe
(``FAMILY19-RAW-P611A3-ld69-pb.txt`` leg b1, ``FAMILY19-RAW-P611A4-ld69-pb.txt``
legs b2/b3): prints retrieved host-side via NEURON_RT_DEBUG_OUTPUT_DIR dump
files AND stdout-formatted lines; a pre-wedge print SURVIVES the default
30,000 ms NRT watchdog kill (NRT_TIMEOUT(5)); 129 literal print sites
(43 layers x 3) arrive complete with zero ring drops.

Parser-frontend constraints honored (measured on the deployed nki 0.5.0,
FAMILY19-RAW-P611A3/A4): NO inner function definitions, NO string
formatting inside kernel source (the parser folds ``%`` into an NKI
arithmetic ``mod`` op) — hence the generated per-layer branch blocks with
LITERAL prefixes below; ``layer_idx`` is a compile-time int per kernel
specialization, so exactly one branch survives dead-code folding in each
compiled instance (production-precedent fold: config flags throughout
nkilib).

Reading contract (port-plan SS15.3 W-table): sentinel payload is the
compile-time constant ``7300.0 + layer_idx`` replicated x4 — constants
clean vs dynamic values garbled is the in-instrument W2 discriminator.
The 11 b2e values are printed TWICE from two INDEPENDENT dma_copy reads
(read-consistency cell). The kernel returns the read-A tile pass-through;
the whole kernel is one custom call, so a consumer of its output forces
the entire witness (prints included) to execute before the flooding
weight-gather consumers in every legal schedule.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert

import torch
from vllm_neuron.nki.nki_hop import can_run_kernel, wrap_nki

_NUM_MOE_LAYERS = 43  # deepseek_v4 config.num_hidden_layers; layer_idx in [0, 42]
_MAX_WITNESS_N = 128  # print volume proven at N=11; (1,128) int32 is ~512 B, far under the 256 KiB printf buffer


def ld68_witness(
    block_to_expert: torch.Tensor,
    layer_idx: int,
    sentinel_only: bool = False,
) -> torch.Tensor:
    """Witness the b2e stream in-execution; return it bit-identical.

    Args:
        block_to_expert (torch.Tensor): 1D [N] int32 block->expert mapping.
        layer_idx (int): static per-instance MoE layer index in [0, 42];
            baked into the printed prefixes at trace time.
        sentinel_only (bool): when True, print only the sentinel line
            (the pre-committed reduced-volume variant; unused at the
            all-43 budget B3 verified).

    Returns:
        torch.Tensor: the same values, same shape, same dtype (identity).
    """
    _validate_inputs(block_to_expert, layer_idx)

    if _can_use_kernel(block_to_expert, layer_idx):
        wrapped = wrap_nki(_ld68_witness_nki)
        return wrapped[2](
            b2e=block_to_expert, layer_idx=layer_idx, sentinel_only=sentinel_only
        )
    else:
        return _torch_ld68_witness(block_to_expert)


def _validate_inputs(block_to_expert: torch.Tensor, layer_idx: int) -> None:
    """Validate inputs for ld68_witness."""
    assert block_to_expert.ndim == 1, (
        f"ld68_witness expects a 1D [N] tensor, got shape {block_to_expert.shape}"
    )
    assert isinstance(layer_idx, int), (
        f"ld68_witness expects a compile-time int layer_idx, got {type(layer_idx)}"
    )


def _can_use_kernel(block_to_expert: torch.Tensor, layer_idx: int) -> bool:
    """Check if the NKI witness kernel can be used.

    Constraints (the P-B-proven envelope):
    - int32 1D input, N <= 128
    - layer_idx within the generated literal-prefix range [0, 42]
    """
    if not can_run_kernel(block_to_expert):
        return False
    if block_to_expert.dtype != torch.int32:
        return False
    if block_to_expert.shape[0] > _MAX_WITNESS_N:
        return False
    if not (0 <= layer_idx < _NUM_MOE_LAYERS):
        return False
    return True


def _torch_ld68_witness(block_to_expert: torch.Tensor) -> torch.Tensor:
    """Numerically matched fallback: identity clone, no prints.

    A fallback that printed nothing but returned anything other than the
    exact input would make the witness a lying discriminator; the clone is
    the whole contract (declaration reference: ``inputs[0].clone()``).
    """
    return block_to_expert.clone()


@nki.jit
def _ld68_witness_nki(b2e, layer_idx=0, sentinel_only=False):
    """Print sentinel + two independent b2e reads; return b2e pass-through.

    Dimensions:
        N: number of blocks (production family shape: 11). 1 <= N <= 128.

    Args:
        b2e (nl.ndarray): [N] int32 block->expert mapping in HBM.
        layer_idx (int): compile-time MoE layer index in [0, 42]; selects
            the literal print prefixes via dead-code branch folding.
        sentinel_only (bool): compile-time; when True, skip the value prints.

    Returns:
        out (nl.ndarray): [N] int32, bit-identical pass-through of b2e.
    """
    # In-kernel checks use LITERALS and single comparisons only — the
    # exact shapes the in-repo specimen kernel proves on this parser
    # (``argsort_unstable.py:178-179``); the module-level constants guard
    # the host-side gate instead.
    N = b2e.shape[0]
    kernel_assert(N >= 1, f"Expected N >= 1, got {N}")
    kernel_assert(N <= 128, f"Expected N <= 128, got {N}")
    kernel_assert(layer_idx >= 0, f"Expected layer_idx >= 0, got {layer_idx}")
    kernel_assert(layer_idx <= 42, f"Expected layer_idx <= 42, got {layer_idx}")
    x2 = b2e.reshape((1, N))

    # (a) unconditional sentinel FIRST: tag + compile-time constant +
    # static layer index. Payload = 7300 + layer_idx (folds at trace),
    # so layer identity rides in BOTH the prefix and the data.
    sent = nl.ndarray((1, 4), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=sent, value=7300.0 + layer_idx)

    # (b) two INDEPENDENT reads of the same logical stream.
    read_a = nl.ndarray((1, N), dtype=b2e.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=read_a, src=x2)
    read_b = nl.ndarray((1, N), dtype=b2e.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=read_b, src=x2)

    # Generated literal-prefix blocks: exactly one survives folding per
    # specialization (layer_idx is compile-time). No string ops in kernel
    # source — parser constraint, measured (FAMILY19-RAW-P611A3 leg b3).
    if layer_idx == 0:
        nl.device_print("LD68_SENT_L00", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L00", read_a)
            nl.device_print("LD68_B2E_B_L00", read_b)
    elif layer_idx == 1:
        nl.device_print("LD68_SENT_L01", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L01", read_a)
            nl.device_print("LD68_B2E_B_L01", read_b)
    elif layer_idx == 2:
        nl.device_print("LD68_SENT_L02", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L02", read_a)
            nl.device_print("LD68_B2E_B_L02", read_b)
    elif layer_idx == 3:
        nl.device_print("LD68_SENT_L03", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L03", read_a)
            nl.device_print("LD68_B2E_B_L03", read_b)
    elif layer_idx == 4:
        nl.device_print("LD68_SENT_L04", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L04", read_a)
            nl.device_print("LD68_B2E_B_L04", read_b)
    elif layer_idx == 5:
        nl.device_print("LD68_SENT_L05", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L05", read_a)
            nl.device_print("LD68_B2E_B_L05", read_b)
    elif layer_idx == 6:
        nl.device_print("LD68_SENT_L06", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L06", read_a)
            nl.device_print("LD68_B2E_B_L06", read_b)
    elif layer_idx == 7:
        nl.device_print("LD68_SENT_L07", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L07", read_a)
            nl.device_print("LD68_B2E_B_L07", read_b)
    elif layer_idx == 8:
        nl.device_print("LD68_SENT_L08", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L08", read_a)
            nl.device_print("LD68_B2E_B_L08", read_b)
    elif layer_idx == 9:
        nl.device_print("LD68_SENT_L09", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L09", read_a)
            nl.device_print("LD68_B2E_B_L09", read_b)
    elif layer_idx == 10:
        nl.device_print("LD68_SENT_L10", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L10", read_a)
            nl.device_print("LD68_B2E_B_L10", read_b)
    elif layer_idx == 11:
        nl.device_print("LD68_SENT_L11", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L11", read_a)
            nl.device_print("LD68_B2E_B_L11", read_b)
    elif layer_idx == 12:
        nl.device_print("LD68_SENT_L12", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L12", read_a)
            nl.device_print("LD68_B2E_B_L12", read_b)
    elif layer_idx == 13:
        nl.device_print("LD68_SENT_L13", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L13", read_a)
            nl.device_print("LD68_B2E_B_L13", read_b)
    elif layer_idx == 14:
        nl.device_print("LD68_SENT_L14", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L14", read_a)
            nl.device_print("LD68_B2E_B_L14", read_b)
    elif layer_idx == 15:
        nl.device_print("LD68_SENT_L15", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L15", read_a)
            nl.device_print("LD68_B2E_B_L15", read_b)
    elif layer_idx == 16:
        nl.device_print("LD68_SENT_L16", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L16", read_a)
            nl.device_print("LD68_B2E_B_L16", read_b)
    elif layer_idx == 17:
        nl.device_print("LD68_SENT_L17", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L17", read_a)
            nl.device_print("LD68_B2E_B_L17", read_b)
    elif layer_idx == 18:
        nl.device_print("LD68_SENT_L18", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L18", read_a)
            nl.device_print("LD68_B2E_B_L18", read_b)
    elif layer_idx == 19:
        nl.device_print("LD68_SENT_L19", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L19", read_a)
            nl.device_print("LD68_B2E_B_L19", read_b)
    elif layer_idx == 20:
        nl.device_print("LD68_SENT_L20", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L20", read_a)
            nl.device_print("LD68_B2E_B_L20", read_b)
    elif layer_idx == 21:
        nl.device_print("LD68_SENT_L21", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L21", read_a)
            nl.device_print("LD68_B2E_B_L21", read_b)
    elif layer_idx == 22:
        nl.device_print("LD68_SENT_L22", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L22", read_a)
            nl.device_print("LD68_B2E_B_L22", read_b)
    elif layer_idx == 23:
        nl.device_print("LD68_SENT_L23", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L23", read_a)
            nl.device_print("LD68_B2E_B_L23", read_b)
    elif layer_idx == 24:
        nl.device_print("LD68_SENT_L24", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L24", read_a)
            nl.device_print("LD68_B2E_B_L24", read_b)
    elif layer_idx == 25:
        nl.device_print("LD68_SENT_L25", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L25", read_a)
            nl.device_print("LD68_B2E_B_L25", read_b)
    elif layer_idx == 26:
        nl.device_print("LD68_SENT_L26", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L26", read_a)
            nl.device_print("LD68_B2E_B_L26", read_b)
    elif layer_idx == 27:
        nl.device_print("LD68_SENT_L27", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L27", read_a)
            nl.device_print("LD68_B2E_B_L27", read_b)
    elif layer_idx == 28:
        nl.device_print("LD68_SENT_L28", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L28", read_a)
            nl.device_print("LD68_B2E_B_L28", read_b)
    elif layer_idx == 29:
        nl.device_print("LD68_SENT_L29", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L29", read_a)
            nl.device_print("LD68_B2E_B_L29", read_b)
    elif layer_idx == 30:
        nl.device_print("LD68_SENT_L30", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L30", read_a)
            nl.device_print("LD68_B2E_B_L30", read_b)
    elif layer_idx == 31:
        nl.device_print("LD68_SENT_L31", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L31", read_a)
            nl.device_print("LD68_B2E_B_L31", read_b)
    elif layer_idx == 32:
        nl.device_print("LD68_SENT_L32", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L32", read_a)
            nl.device_print("LD68_B2E_B_L32", read_b)
    elif layer_idx == 33:
        nl.device_print("LD68_SENT_L33", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L33", read_a)
            nl.device_print("LD68_B2E_B_L33", read_b)
    elif layer_idx == 34:
        nl.device_print("LD68_SENT_L34", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L34", read_a)
            nl.device_print("LD68_B2E_B_L34", read_b)
    elif layer_idx == 35:
        nl.device_print("LD68_SENT_L35", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L35", read_a)
            nl.device_print("LD68_B2E_B_L35", read_b)
    elif layer_idx == 36:
        nl.device_print("LD68_SENT_L36", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L36", read_a)
            nl.device_print("LD68_B2E_B_L36", read_b)
    elif layer_idx == 37:
        nl.device_print("LD68_SENT_L37", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L37", read_a)
            nl.device_print("LD68_B2E_B_L37", read_b)
    elif layer_idx == 38:
        nl.device_print("LD68_SENT_L38", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L38", read_a)
            nl.device_print("LD68_B2E_B_L38", read_b)
    elif layer_idx == 39:
        nl.device_print("LD68_SENT_L39", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L39", read_a)
            nl.device_print("LD68_B2E_B_L39", read_b)
    elif layer_idx == 40:
        nl.device_print("LD68_SENT_L40", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L40", read_a)
            nl.device_print("LD68_B2E_B_L40", read_b)
    elif layer_idx == 41:
        nl.device_print("LD68_SENT_L41", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L41", read_a)
            nl.device_print("LD68_B2E_B_L41", read_b)
    elif layer_idx == 42:
        nl.device_print("LD68_SENT_L42", sent)
        if not sentinel_only:
            nl.device_print("LD68_B2E_A_L42", read_a)
            nl.device_print("LD68_B2E_B_L42", read_b)

    # (c) pass-through return FROM READ A: the consumer's data dependence
    # on this output pulls the whole custom call (prints included) ahead
    # of the flooding weight gathers in every legal schedule.
    out = nl.ndarray((1, N), dtype=b2e.dtype, buffer=nl.shared_hbm, name="ld68_b2e_hbm")
    nisa.dma_copy(dst=out, src=read_a)
    return out.reshape((N,))
