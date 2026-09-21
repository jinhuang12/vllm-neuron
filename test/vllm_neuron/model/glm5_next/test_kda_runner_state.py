# SPDX-License-Identifier: Apache-2.0
"""The runner carries a KDA layer's conv and recurrent state through a step.

One call site, the call order, and the state a cleared bank fails to reproduce.
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_neuron.vllm.worker import neuron_model_runner as runner_module
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next import test_kda_layer as layer_half

# ---------------------------------------------------------------------------
# Declared values. Every one is either imported from the layer half or derived
# from something the runner or the config states.
# ---------------------------------------------------------------------------

#: The hook this block repairs, and the method that calls it.
HOOK_NAME = "_update_states_after_model_execute"
CALLER_NAME = "sample_tokens"

#: One call site, in one method. Both numbers are the reading, not a guess: the
#: module is parsed and every call is found.
DECLARED_HOOK_CALL_SITES = 1

#: The state bank the runner allocates has a leading slot dimension the layer
#: half's bank does not, because the runner's carrier builder slices it
#: (``neuron_model_runner.py``). One slot is enough for one sequence,
#: and the builder refuses a slot outside the bank it was handed.
DECLARED_STATE_SLOTS = 1

#: The state slot this file's one request resolves to. The converter does not read a slot
#: off the block row -- it hands the request the slot its own table owns -- so this value
#: is the table's first hand-out, and the tests below read the table rather than this
#: constant, which cannot tell the two sources apart.
DECLARED_STATE_SLOT = 0

#: How many sequences the modelled engine admits at once. One, because this file
#: runs one request, and the bank above holds exactly that many slots.
DECLARED_MAX_NUM_SEQS = 1

#: The page the metadata declares. The linear branch of the carrier builder never
#: reads it -- only the sparse branch cross-checks paging -- so this is the
#: hybrid block size the pin admits and nothing depends on its value.
DECLARED_PAGE_SIZE = layer_half.HYBRID_BLOCK_SIZE

#: The decode-token threshold the converter reads to decide the leg
#: (``max_query_len > decode_token_threshold``). One means a single-token step is
#: a decode and the 17-token prompt is a prefill.
DECLARED_DECODE_THRESHOLD = 1

#: Carried from the layer half, which reads each of them off its own origin.
DECLARED_STACK_LAYERS = layer_half.DECLARED_STACK_LAYERS
DECLARED_DECODE_STEPS = layer_half.DECLARED_DECODE_STEPS
DECLARED_PREFILL_TOKENS = layer_half.DECLARED_PREFILL_TOKENS
DECLARED_CHUNK = layer_half.DECLARED_CHUNK
DECLARED_DECODE_DISPATCHES = layer_half.DECLARED_DECODE_DISPATCHES
DECLARED_CHUNKED_DISPATCHES_ON_DECODE = (
    layer_half.DECLARED_CHUNKED_DISPATCHES_ON_DECODE
)
DECLARED_FALLBACKS = layer_half.DECLARED_FALLBACKS
DECLARED_RTOL = layer_half.DECLARED_RTOL
DECLARED_ATOL = layer_half.DECLARED_ATOL
TP_WORLD_SIZE = layer_half.TP_WORLD_SIZE
SEED = layer_half.SEED

#: The keys a linear layer's carrier holds, in the model's own declared spelling
#: (``model_fp8.py``). The carrier is splatted straight into the layer, so
#: this set is asserted rather than assumed: an extra key would be a TypeError and
#: a missing one would be served as a default. ``start_position`` joined the set
#: when the layer learned to continue a segmented prompt's recurrence; the
#: sparse family's carrier has carried the same key from its own beginning.
#: ``real_tokens`` and ``row_mask`` joined it when the layer learned to scan only the
#: rows that carry a token, which is the change that moved this pin.
DECLARED_CARRIER_KEYS = {
    "conv_state",
    "recurrent_state",
    "is_prefill",
    "start_position",
    "real_tokens",
    "row_mask",
}


class VacuousControlError(AssertionError):
    """A control that cannot discriminate is a failure, never a pass. """


# ---------------------------------------------------------------------------
# The runner-level drive.
# ---------------------------------------------------------------------------
def _banks(layers) -> list[dict]:
    """The bank mapping ``bind_kv_cache`` leaves on the model, built the runner's way.
    """
    banks = []
    for index, layer in enumerate(layers):
        attention = layer.attention
        banks.append(
            {
                "name": f"model.layers.{index}.attention",
                "family": "linear_attn",
                "state_slots": DECLARED_STATE_SLOTS,
                "conv_state": torch.zeros(
                    (DECLARED_STATE_SLOTS, *attention.kda_conv_state_shape),
                    dtype=attention.kda_conv_state_dtype,
                ),
                "recurrent_state": torch.zeros(
                    (DECLARED_STATE_SLOTS, *attention.kda_recurrent_state_shape),
                    dtype=attention.kda_recurrent_state_dtype,
                ),
            }
        )
    return banks


def _runner(text_config, banks) -> NeuronModelRunner:
    """A runner carrying only what the converter reads. """
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    # The converter keys per-request cache state on the engine's own request ids and
    # refuses a real step that carries none, so a harness that models a runner must model
    # its batch too.
    runner.input_batch = SimpleNamespace(req_ids=["req-0"])
    runner.model = SimpleNamespace(
        text_config=text_config, glm5next_layer_banks=banks
    )
    # Only the side-cache allocator reads this, and with no sparse bank in the
    # stack it allocates nothing; it must still be positive.
    runner.max_model_len = DECLARED_PREFILL_TOKENS + DECLARED_DECODE_STEPS + 1
    # The slot axis is the engine's concurrency bound, so the harness carries it.
    runner.max_num_reqs = DECLARED_MAX_NUM_SEQS
    return runner


def _metadata(banks, *, tokens: int, cached: int) -> dict:
    """The runner's attention-metadata mapping: one entry per layer name. """
    span = cached + tokens
    blocks = max(1, -(-span // DECLARED_PAGE_SIZE))
    table = torch.tensor(
        [[DECLARED_STATE_SLOT + offset for offset in range(blocks)]],
        dtype=torch.int32,
    )
    entry = {
        "block_table_tensor": table,
        "full_block_table_tensor": table,
        "slot_mapping": torch.arange(tokens, dtype=torch.int32) + cached,
        "max_query_len": int(tokens),
        "block_size": int(DECLARED_PAGE_SIZE),
        "max_blocks_per_seq": int(table.shape[1]),
        "decode_token_threshold": DECLARED_DECODE_THRESHOLD,
        "cached_seq_len": torch.tensor([cached], dtype=torch.int32),
        "host_block_table": table,
        "host_num_computed_tokens": [int(cached)],
        "kv_segment_size": int(table.shape[1]) * int(DECLARED_PAGE_SIZE),
    }
    return {str(bank["name"]): entry for bank in banks}


def _carriers(runner, banks, *, tokens: int, cached: int) -> list[dict]:
    """One step's carriers, produced by the converter under test. """
    generic = {
        "input_ids": torch.zeros(tokens, dtype=torch.long),
        "positions": torch.arange(tokens, dtype=torch.long) + cached,
        "attn_metadata": _metadata(banks, tokens=tokens, cached=cached),
        "sampling_positions": torch.tensor([tokens - 1], dtype=torch.long),
        "sampling_params": None,
        "spec_decode_metadata": None,
        "rank": None,
        "logit_mask": None,
    }
    converted = runner._glm5next_model_kwargs(generic)
    assert sorted(converted) == [
        "expert_parallel_rank",
        "input_ids",
        "layer_carriers",
        "moe_group",
        "sampling_positions",
        "tp_degree",
    ], f"the converter returned {sorted(converted)}"
    carriers = converted["layer_carriers"]
    assert len(carriers) == len(banks), (
        f"the converter built {len(carriers)} carrier(s) for {len(banks)} bank(s); "
        f"the model's own forward refuses a count that disagrees with the stack"
    )
    for carrier in carriers:
        assert set(carrier) == DECLARED_CARRIER_KEYS, (
            f"a linear carrier holds {sorted(carrier)}, not "
            f"{sorted(DECLARED_CARRIER_KEYS)}"
        )
    return carriers


def _world(*, clear_state_before_decode: bool) -> SimpleNamespace:
    """One prefill and two decode steps, every carrier built by the runner. """
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    text_config = Glm5NextTextConfig()
    hidden = int(text_config.hidden_size)
    linear_attn = text_config.linear_attn_config
    heads = int(linear_attn["num_heads"]) // TP_WORLD_SIZE
    head_dim = int(linear_attn["head_dim"])
    kernel = int(linear_attn["short_conv_kernel_size"])
    eps = float(text_config.rms_norm_eps)
    total = DECLARED_PREFILL_TOKENS + DECLARED_DECODE_STEPS

    # The same seeds the layer half uses, so this file's stack and its reference
    # are the same world that file measured -- one comparator, two drives.
    torch.manual_seed(SEED)
    weights = [
        layer_half._make_weights(hidden, heads, head_dim, kernel)
        for _ in range(DECLARED_STACK_LAYERS)
    ]
    torch.manual_seed(SEED + 1)
    tokens = torch.randn(total, hidden, dtype=torch.float32)

    layers = []
    for index, layer_weights in enumerate(weights):
        layer = layer_half._impl().Glm5NextKDALayer(
            text_config, index, TP_WORLD_SIZE
        )
        for name, tensor in layer_weights.items():
            target = layer if name == "input_layernorm_weight" else layer.attention
            setattr(target, name, nn.Parameter(tensor.clone(), requires_grad=False))
        layers.append(layer)

    banks = _banks(layers)
    runner = _runner(text_config, banks)

    def drive(rows: torch.Tensor, *, cached: int) -> torch.Tensor:
        carriers = _carriers(
            runner, banks, tokens=int(rows.shape[0]), cached=cached
        )
        out = rows
        for layer, carrier in zip(layers, carriers):
            out = layer(out, **carrier, chunk_size=DECLARED_CHUNK)
        return out, carriers

    prefill_out, prefill_carriers = drive(
        tokens[:DECLARED_PREFILL_TOKENS], cached=0
    )
    prefill_state = [bank["recurrent_state"].clone() for bank in banks]

    if clear_state_before_decode:
        for bank in banks:
            bank["recurrent_state"].zero_()

    # Reset here, not before the prefill. This test reads the decode arm only, so the
    # counters are cleared immediately before the decode loop and read immediately
    # after it. No prefill count is captured, because none is registered: the
    # prefill arm's own per-seam numbers are the layer half's, and
    # taking a second reading of them here would register a criterion this block
    # does not declare.
    layer_half._reset_counters()
    decode_rows = []
    decode_carriers = []
    for step in range(DECLARED_DECODE_STEPS):
        index = DECLARED_PREFILL_TOKENS + step
        out, carriers = drive(
            tokens[index : index + 1], cached=index
        )
        decode_rows.append(out)
        decode_carriers.append(carriers)
    decode_counts = layer_half._read_counters()
    decode_out = torch.cat(decode_rows, dim=0)

    # The reference, driven in the same three calls with its own carriers, in the
    # layer half's form.
    conv_carrier_dtype = layers[0].attention.kda_conv_state_dtype
    histories = [
        torch.zeros(kernel - 1, 3 * heads * head_dim, dtype=torch.float32)
        for _ in range(DECLARED_STACK_LAYERS)
    ]
    states: list[list[torch.Tensor] | None] = [None] * DECLARED_STACK_LAYERS

    def reference(rows: torch.Tensor) -> torch.Tensor:
        out = rows
        for index, layer_weights in enumerate(weights):
            out, histories[index], states[index] = layer_half._reference_layer(
                out,
                layer_weights,
                histories[index],
                states[index],
                heads=heads,
                head_dim=head_dim,
                eps=eps,
                conv_carrier_dtype=conv_carrier_dtype,
            )
        return out

    reference_prefill = reference(tokens[:DECLARED_PREFILL_TOKENS])
    reference_decode = torch.cat(
        [
            reference(tokens[DECLARED_PREFILL_TOKENS + step :][:1])
            for step in range(DECLARED_DECODE_STEPS)
        ],
        dim=0,
    )

    return SimpleNamespace(
        banks=banks,
        layers=layers,
        prefill_out=prefill_out,
        prefill_state=prefill_state,
        prefill_carriers=prefill_carriers,
        decode_out=decode_out,
        decode_counts=decode_counts,
        decode_carriers=decode_carriers,
        reference_prefill=reference_prefill,
        reference_decode=reference_decode,
        # The slot the table handed this request, read off the runner rather than
        # assumed: a view assertion against a declared 0 cannot tell a slot the table
        # owns from a slot read off a block row, because the first hand-out is 0 too.
        owned_slot=dict(runner._glm5next_request_slot_table)[
            runner.input_batch.req_ids[0]
        ],
    )


@pytest.fixture(scope="module")
def run() -> SimpleNamespace:
    """The shared world run for both arms: the state is carried."""
    return _world(clear_state_before_decode=False)


# ---------------------------------------------------------------------------
# Conjunct 1 -- the structural reading. No torch, no model, no serving loop.
# ---------------------------------------------------------------------------
def _hook_sites() -> SimpleNamespace:
    """Parse the runner module and locate the hook's definition and every call. """
    source = inspect.getsource(runner_module)
    tree = ast.parse(source)
    definition = None
    calls = []
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for node in functions:
        if node.name == HOOK_NAME:
            definition = node
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == HOOK_NAME
        ):
            enclosing = [
                function
                for function in functions
                if function.lineno <= node.lineno <= function.end_lineno
                and function.name != HOOK_NAME
            ]
            # The innermost enclosing function is the caller.
            enclosing.sort(key=lambda function: function.end_lineno - function.lineno)
            calls.append(
                SimpleNamespace(
                    line=node.lineno,
                    caller=enclosing[0].name if enclosing else None,
                    arguments=[
                        argument.attr
                        if isinstance(argument, ast.Attribute)
                        else getattr(argument, "id", None)
                        for argument in node.args
                    ],
                )
            )
    return SimpleNamespace(definition=definition, calls=calls)


def test_one_call_site_and_it_is_in_sample_tokens() -> None:
    """Conjunct 1 (i). """
    sites = _hook_sites()
    assert sites.definition is not None, f"{HOOK_NAME} is not defined in the module"
    assert len(sites.calls) == DECLARED_HOOK_CALL_SITES, (
        f"the module holds {len(sites.calls)} call(s) to {HOOK_NAME}, expected "
        f"{DECLARED_HOOK_CALL_SITES}"
    )
    assert sites.calls[0].caller == CALLER_NAME, (
        f"the call is inside {sites.calls[0].caller!r}, not {CALLER_NAME!r}; "
        f"check 1's claim names the caller"
    )


def test_the_call_order_equals_the_definition_order() -> None:
    """Conjunct 1 (ii), and the base-failing control. """
    sites = _hook_sites()
    # The call-site reading, repeated so that this test states its own precondition rather
    # than raising an IndexError about a list a different test is responsible for.
    assert sites.definition is not None and len(sites.calls) == (
        DECLARED_HOOK_CALL_SITES
    ), (
        f"this test reads one call site's order and the module holds "
        f"{len(sites.calls)}; the call-site test reports on the count"
    )
    parameters = [argument.arg for argument in sites.definition.args.args][1:]
    arguments = sites.calls[0].arguments
    assert None not in arguments, (
        f"a positional argument at the call site is neither a name nor an "
        f"attribute ({arguments}), so this test cannot read its order"
    )
    assert len(arguments) == len(parameters), (
        f"the call passes {len(arguments)} positional argument(s) and the "
        f"definition takes {len(parameters)}"
    )
    assert arguments == parameters, (
        f"the call passes {arguments} while the definition takes {parameters}; "
        f"a body added to this hook would read its arguments swapped"
    )


# ---------------------------------------------------------------------------
# Conjunct 2 -- the runner carries the state through the layer.
# ---------------------------------------------------------------------------
def test_the_runner_carries_the_state_through_the_layer(
    run: SimpleNamespace,
) -> None:
    """Conjunct 2. """
    # The leg is the runner's own decision, read on both arms. The converter
    # decides prefill against decode by ``max_query_len > decode_token_threshold``
    # (``neuron_model_runner.py``), so nothing here sets the flag: 17
    # tokens over a threshold of 1 is a prefill, one token is a decode. If this
    # file set the flag itself, every reading below would be about a leg this file
    # chose rather than the leg the runner derives while serving.
    for index, carrier in enumerate(run.prefill_carriers):
        assert carrier["is_prefill"] is True, (
            f"bank {index} received is_prefill=False for the "
            f"{DECLARED_PREFILL_TOKENS}-token prompt"
        )

    # The carrier the runner handed each layer is a view of that layer's own bank
    # slot. If it were a copy, the layer's in-place advance would be written to
    # something the next step never reads, and the carry would silently restart.
    # Every step's carriers, not just the first: a converter that handed a view on
    # step 1 and a fresh tensor on step 2 would still pass a first-step-only reading
    # while dropping the carry the test is about.
    assert len(run.decode_carriers) == DECLARED_DECODE_STEPS
    for step, carriers in enumerate(run.decode_carriers):
        for index, carrier in enumerate(carriers):
            bank = run.banks[index]
            assert carrier["recurrent_state"][0].data_ptr() == (
                bank["recurrent_state"][run.owned_slot].data_ptr()
            ), (
                f"step {step}: bank {index}'s recurrent carrier is not a view of "
                f"its own slot, so the layer's in-place advance is written where "
                f"the next step will not read it"
            )
            assert carrier["conv_state"][0].data_ptr() == (
                bank["conv_state"][run.owned_slot].data_ptr()
            ), f"step {step}: bank {index}'s conv carrier is not a view of its own slot"
            assert carrier["is_prefill"] is False, (
                f"step {step}: bank {index} received is_prefill=True for a "
                f"single-token step"
            )

    # It advanced, and the advance is in the bank the runner owns.
    for index, bank in enumerate(run.banks):
        recurrent = bank["recurrent_state"][run.owned_slot]
        attention = run.layers[index].attention
        assert tuple(recurrent.shape) == tuple(attention.kda_recurrent_state_shape)
        assert recurrent.dtype is attention.kda_recurrent_state_dtype
        assert not torch.equal(
            bank["recurrent_state"], run.prefill_state[index]
        ), (
            f"bank {index} holds the state the prefill left; two decode steps "
            f"through the runner's carriers did not advance it"
        )
        assert float(recurrent.abs().max()) > 0.0

    # The values. A state carried wrongly changes these, which is what makes the
    # carry a measurement instead of a bookkeeping claim.
    torch.testing.assert_close(
        run.decode_out,
        run.reference_decode,
        rtol=DECLARED_RTOL,
        atol=DECLARED_ATOL,
    )

    # The prefill leg, through the same runner-built carriers. It is asserted here
    # rather than left uncompared because a prefill served from the wrong slot, or
    # from a carrier that is a copy, would leave the bank right and the output
    # wrong -- and the decode comparison above continues from this leg's state, so a
    # silent prefill error would arrive as an unexplained decode failure.
    torch.testing.assert_close(
        run.prefill_out,
        run.reference_prefill,
        rtol=DECLARED_RTOL,
        atol=DECLARED_ATOL,
    )


def test_the_decode_arm_reads_the_declared_seam_counts(
    run: SimpleNamespace,
) -> None:
    """The decode arm takes the kernel route, over two steps."""
    counts = run.decode_counts
    for seam in ("conv", "gate", "decode"):
        dispatches, fallbacks = counts[seam]
        assert dispatches == DECLARED_DECODE_DISPATCHES, (
            f"{seam} read {dispatches}, expected {DECLARED_DECODE_DISPATCHES} -- "
            f"{DECLARED_STACK_LAYERS} per step over {DECLARED_DECODE_STEPS} steps"
        )
        assert fallbacks == DECLARED_FALLBACKS, (
            f"{seam} reported {fallbacks} torch fallback(s); the kernel route admits none"
        )
    for seam in ("intra", "inter"):
        dispatches, fallbacks = counts[seam]
        assert dispatches == DECLARED_CHUNKED_DISPATCHES_ON_DECODE, (
            f"{seam} read {dispatches} dispatch(es) on a decode arm; a "
            f"single-token decode step must never enter a chunked seam"
        )
        assert fallbacks == DECLARED_FALLBACKS


def test_a_cleared_bank_misses_the_reference() -> None:
    """The non-vacuity control on the carried-state arm."""
    world = _world(clear_state_before_decode=True)
    for index, bank in enumerate(world.banks):
        # The cleared world still advanced from zero, so the failure below is a
        # wrong state and not an untouched bank.
        assert float(bank["recurrent_state"].abs().max()) > 0.0, (
            f"bank {index} is still all zeros after two decode steps, so this "
            f"control is measuring a dead drive rather than a dropped carry"
        )
    gap = float((world.decode_out - world.reference_decode).abs().max())
    tolerance = DECLARED_ATOL + DECLARED_RTOL * float(
        world.reference_decode.abs().max()
    )
    if gap <= tolerance:
        raise VacuousControlError(
            f"clearing the recurrent state moved the decode outputs by {gap:.3e}, "
            f"inside the declared tolerance {tolerance:.3e}; that arm's pass therefore "
            f"does not discriminate a carried state from a dropped one"
        )
