# SPDX-License-Identifier: Apache-2.0
"""``inc-glm53f-038b`` acceptance -- WP3: KDA runner state plumbing.

THE DECLARED ACCEPTANCE, the block's Tier N harness as ``inc-glm53f-025``:

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 \\
      NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python -m pytest \\
      test/vllm_neuron/model/glm5_next/test_kda_runner_state.py -s -rA \\
      -p no:randomly -p no:cacheprovider

Five items, one test each, no ``parametrize``.

WHAT CONJUNCT 1 OBSERVES, AND WHAT IT DOES NOT
----------------------------------------------
Conjunct 1 is a STRUCTURAL reading of the state hook, and the wording matters
because an earlier draft of this block asked for an invocation count instead.

* It observes that the module contains exactly ONE call to
  ``_update_states_after_model_execute``, that the call is inside
  ``sample_tokens``, and that the call's positional argument order equals the
  definition's parameter order.
* It does NOT observe the serving loop calling the hook once per decode step.
  Nothing here drives ``sample_tokens``.

The invocation count was RETIRED rather than deferred (ruling LEAD-LOG §1276).
It was designed for a hook that was to hold the state-advance body; that body is
not owed, because the KDA carrier keys already reach the layer
(``neuron_model_runner.py:4985-4987``) and the landed layer advances both states
in place (``model_fp8.py:3964``, ``:4225``). Calling a ``pass`` hook twice and
counting two observes nothing about the model, so it is not counted here and is
not described as "fires once per decode step". The per-step property belongs to
the serving chain's own end-to-end run.

THE FIVE ITEMS
--------------
* B01 -- conjunct 1 (i). One call to the hook in the module, and it is inside
  ``sample_tokens``; no other function in the module calls it.
* B02 -- conjunct 1 (ii), AND THE BASE-FAILING CONTROL. The call's positional
  argument order equals the definition's parameter order. This item FAILS at base
  ``0a1888a9`` -- the definition declared ``(scheduler_output,
  sampled_token_ids)`` while the call passes ``(sampler_output.sampled_token_ids,
  scheduler_output)`` -- and passes at this increment's candidate. It is the item
  that makes this increment's code change measurable rather than asserted.
* B03 -- conjunct 2. One prefill and two decode steps, every carrier built by the
  RUNNER's own converter, and both legs' outputs value-compared against the torch
  reference at the carried tolerance pair. The leg flag each layer receives is the
  runner's own derivation from the batch, not a flag this file set, and each state
  carrier is checked to be a VIEW of the bank slot the runner owns -- a copy would
  send the layer's in-place advance somewhere the next step never reads.
* B04 -- the D13 route predicate, form R-2, decode arm only, read off B03's own
  world run: three seams at six each, both chunked seams at exactly zero, and the
  module's torch-fallback counter at zero.
* B05 -- the non-vacuity control on B03. A second world whose recurrent state is
  cleared between the prefill and the first decode step must MISS the reference by
  more than the tolerance. Without it, B03 would read the same green whether the
  runner carried the state or dropped it.

WHY THIS IS THE RUNNER HALF, AND WHY IT IS A DIFFERENT FILE
----------------------------------------------------------
``inc-glm53f-038a``'s landed ``test_kda_layer.py`` drives the layer directly, one
bank tensor per call. This file never hands a layer a bank tensor: every
``conv_state``, ``recurrent_state`` and ``is_prefill`` a layer receives here comes
out of ``NeuronModelRunner._glm5next_model_kwargs``, which is the code this block
is about. The block requires the two halves to live in separate files so that
neither half's counted predicate can be satisfied by the other half's items.

WHAT IS IMPORTED FROM THE LAYER HALF, AND WHY
--------------------------------------------
The reference, the seam counters and the declared values are imported from
``test_kda_layer`` rather than copied. The comparator is the retired ``-038``
Acceptance line's own pair carried byte-for-byte (P9), so importing it makes the
carry literal: a copy could drift from the value the plan registered while still
looking right. A test-to-test import is this campaign's landed idiom
(``test_tiny_glm5next_e2e.py:60``); the D11 rule it must respect is that nothing
under ``test/`` is imported INTO ``vllm_neuron/``, which this does not do.

CONVENTIONS THIS FILE FOLLOWS
-----------------------------
``model_fp8`` is never imported at module level -- ``test_factory.py:318-319`` is
a landed assertion that it stays out of ``sys.modules`` -- so it is reached
through the layer half's ``_impl()`` inside a body or fixture. The runner is
stood up with ``__new__`` and given only the attributes the converter reads,
which is this campaign's landed harness shape
(``test_get_kv_cache_spec_hybrid.py:189-201``, and ten uses in
``test_tiny_glm5next_e2e.py``).
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
#: (``neuron_model_runner.py:4985-4986``). One slot is enough for one sequence,
#: and the builder refuses a slot outside the bank it was handed.
DECLARED_STATE_SLOTS = 1

#: The state slot every geometry in this file resolves to. The converter reads it
#: as the FIRST block id of the row (``neuron_model_runner.py:5151``), so the row
#: below starts at 0 and this is that 0, named rather than implied.
DECLARED_STATE_SLOT = 0

#: The page the metadata declares. The linear branch of the carrier builder never
#: reads it -- only the sparse branch cross-checks paging -- so this is the
#: hybrid block size the pin admits and nothing depends on its value.
DECLARED_PAGE_SIZE = layer_half.REGISTERED_HYBRID_BLOCK_SIZE

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
REGISTERED_TP_WORLD_SIZE = layer_half.REGISTERED_TP_WORLD_SIZE
SEED = layer_half.SEED

#: The keys a linear layer's carrier holds, in the model's own declared spelling
#: (``model_fp8.py:4149``). The carrier is splatted straight into the layer, so
#: this set is asserted rather than assumed: an extra key would be a TypeError and
#: a missing one would be served as a default. ``start_position`` joined the set
#: when the layer learned to continue a segmented prompt's recurrence; the
#: sparse family's carrier has carried the same key from its own beginning.
DECLARED_CARRIER_KEYS = {
    "conv_state",
    "recurrent_state",
    "is_prefill",
    "start_position",
}


class VacuousControlError(AssertionError):
    """A control that cannot discriminate is a failure, never a pass.

    Raised when the world a control is built on does not have the property the
    control needs in order to mean anything -- for example a "state was carried"
    control whose reference decode is identical to its prefill.
    """


# ---------------------------------------------------------------------------
# The runner-level drive.
# ---------------------------------------------------------------------------
def _banks(layers) -> list[dict]:
    """The bank mapping ``bind_kv_cache`` leaves on the model, built the runner's way.

    Each bank carries the four keys the carrier builder reads for a linear layer
    (``neuron_model_runner.py:4977-4990``): the family, the slot count it will
    range-check, and the two state tensors with a leading slot dimension. The
    shapes and dtypes come from the layer's own ``kda_*`` fields, which is what
    ``get_kv_spec`` reports, so this helper cannot disagree with the model about
    what was asked for.
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
    """A runner carrying ONLY what the converter reads.

    The converter reads ``self.model``, ``self.model.text_config``,
    ``self.model.glm5next_layer_banks``, ``self.max_model_len`` and its own two
    ``_glm5next_*`` helpers, and nothing else
    (``neuron_model_runner.py:5051-5293``). Building the object with ``__new__``
    keeps every other attribute absent, so a converter that started reading
    something new would raise here instead of quietly finding a stand-in value.
    """
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = SimpleNamespace(
        text_config=text_config, glm5next_layer_banks=banks
    )
    # Only the side-cache allocator reads this, and with no sparse bank in the
    # stack it allocates nothing; it must still be positive.
    runner.max_model_len = DECLARED_PREFILL_TOKENS + DECLARED_DECODE_STEPS + 1
    return runner


def _metadata(banks, *, tokens: int, cached: int) -> dict:
    """The runner's attention-metadata mapping: one entry per layer name.

    THE KEYING AND THE KEYS ARE THE RUNNER'S OWN (``neuron_model_runner.py``
    ``:4417-4441`` for the entry, ``:4256-4257`` for the keying). The converter
    reads five of them and derives each bank's geometry from the HOST-SIDE block
    table and cached length, and the page; the device copies of those two numbers
    are present, unread, because the runner's own mapping carries both.
    """
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
    """One step's carriers, produced by the converter under test.

    THIS IS THE WHOLE POINT OF THE FILE. Nothing below hands a layer a bank
    tensor; the layer receives what the runner decided to give it.
    """
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
        "input_ids",
        "layer_carriers",
        "sampling_positions",
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
    """One prefill and two decode steps, every carrier built by the runner.

    ``clear_state_before_decode`` builds B05's world instead of B03's: the
    recurrent state is zeroed after the prefill, which is what a runner that
    allocated a fresh bank per step, or handed the layer the wrong slot, would
    leave behind.
    """
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    text_config = Glm5NextTextConfig()
    hidden = int(text_config.hidden_size)
    linear_attn = text_config.linear_attn_config
    heads = int(linear_attn["num_heads"]) // REGISTERED_TP_WORLD_SIZE
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
            text_config, index, REGISTERED_TP_WORLD_SIZE
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

    # RESET HERE, NOT BEFORE THE PREFILL. B04 reads the DECODE arm only, so the
    # counters are cleared immediately before the decode loop and read immediately
    # after it. No prefill count is captured, because none is registered: the
    # prefill arm's own per-seam numbers are the layer half's A01 reading, and
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
    # layer half's landed form.
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
    )


@pytest.fixture(scope="module")
def run() -> SimpleNamespace:
    """B03's and B04's shared world run: the state IS carried.

    Module-scoped because B04 reads the seam counts of the same run B03 value-
    compares. Two readings over one world run, never two runs.
    """
    return _world(clear_state_before_decode=False)


# ---------------------------------------------------------------------------
# Conjunct 1 -- the structural reading. No torch, no model, no serving loop.
# ---------------------------------------------------------------------------
def _hook_sites() -> SimpleNamespace:
    """Parse the runner module and locate the hook's definition and every call.

    Parsing rather than grepping: a grep counts a mention in a docstring or a
    comment as a call site, and this item's whole claim is about calls.
    """
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


def test_kda_runner_state_b01_one_call_site_and_it_is_in_sample_tokens() -> None:
    """Conjunct 1 (i). One call, inside ``sample_tokens``, and nowhere else.

    This is what replaces the retired invocation count. It says where the hook is
    called from and that nothing else calls it; it does not say how often the
    serving loop reaches it.
    """
    sites = _hook_sites()
    assert sites.definition is not None, f"{HOOK_NAME} is not defined in the module"
    print(
        f"B01_DEFINITION={HOOK_NAME}:"
        f"{sites.definition.lineno}-{sites.definition.end_lineno}"
    )
    for site in sites.calls:
        print(f"B01_CALL_SITE={site.caller}:{site.line} args={site.arguments}")
    assert len(sites.calls) == DECLARED_HOOK_CALL_SITES, (
        f"the module holds {len(sites.calls)} call(s) to {HOOK_NAME}, expected "
        f"{DECLARED_HOOK_CALL_SITES}"
    )
    assert sites.calls[0].caller == CALLER_NAME, (
        f"the call is inside {sites.calls[0].caller!r}, not {CALLER_NAME!r}; "
        f"conjunct 1's claim names the caller"
    )


def test_kda_runner_state_b02_the_call_order_equals_the_definition_order() -> None:
    """Conjunct 1 (ii), and THE BASE-FAILING CONTROL.

    At base ``0a1888a9`` the definition declared ``(scheduler_output,
    sampled_token_ids)`` while the only call passes
    ``(sampler_output.sampled_token_ids, scheduler_output)``, so this item fails
    there and passes at this increment's candidate. Upstream is self-consistent --
    it declares ``(output_token_ids, scheduler_output)`` and calls in that same
    order at ``vllm/v1/worker/gpu_model_runner.py:1497`` and ``:4473``, tag
    ``v0.24.0``, the version this fork pins -- so the definition was the side that
    disagreed and the definition is the side that moved.

    THE READER TAKES THE ARGUMENT'S OWN NAME, not its expression: the call passes
    an attribute, ``sampler_output.sampled_token_ids``, whose attribute name is
    the parameter it fills. Comparing the whole expression would never match any
    parameter name and the item would fail on correct code.
    """
    sites = _hook_sites()
    # B01's reading, repeated so that this item states its own precondition rather
    # than raising an IndexError about a list a different item is responsible for.
    assert sites.definition is not None and len(sites.calls) == (
        DECLARED_HOOK_CALL_SITES
    ), (
        f"this item reads one call site's order and the module holds "
        f"{len(sites.calls)}; B01 is the item that reports on the count"
    )
    parameters = [argument.arg for argument in sites.definition.args.args][1:]
    arguments = sites.calls[0].arguments
    print(f"B02_DEFINITION_ORDER={parameters}")
    print(f"B02_CALL_ORDER={arguments}")
    assert None not in arguments, (
        f"a positional argument at the call site is neither a name nor an "
        f"attribute ({arguments}), so this item cannot read its order"
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
# Conjunct 2 -- the runner carries the state through the landed layer.
# ---------------------------------------------------------------------------
def test_kda_runner_state_b03_the_runner_carries_the_state_through_the_layer(
    run: SimpleNamespace,
) -> None:
    """Conjunct 2. Two decode steps, runner-built carriers, values compared.

    Every carrier here came out of the converter, and the bank it slices is the
    one the runner allocated. So this item reads the runner's plumbing, not the
    layer's arithmetic -- the layer half already measured that.
    """
    # THE LEG IS THE RUNNER'S OWN DECISION, read on both arms. The converter
    # decides prefill against decode by ``max_query_len > decode_token_threshold``
    # (``neuron_model_runner.py:5155-5158``), so nothing here sets the flag: 17
    # tokens over a threshold of 1 is a prefill, one token is a decode. If this
    # file set the flag itself, every reading below would be about a leg this file
    # chose rather than the leg the runner derives while serving.
    for index, carrier in enumerate(run.prefill_carriers):
        assert carrier["is_prefill"] is True, (
            f"bank {index} received is_prefill=False for the "
            f"{DECLARED_PREFILL_TOKENS}-token prompt"
        )

    # The carrier the runner handed each layer IS a view of that layer's own bank
    # slot. If it were a copy, the layer's in-place advance would be written to
    # something the next step never reads, and the carry would silently restart.
    # EVERY step's carriers, not just the first: a converter that handed a view on
    # step 1 and a fresh tensor on step 2 would still pass a first-step-only reading
    # while dropping the carry the item is about.
    assert len(run.decode_carriers) == DECLARED_DECODE_STEPS
    for step, carriers in enumerate(run.decode_carriers):
        for index, carrier in enumerate(carriers):
            bank = run.banks[index]
            assert carrier["recurrent_state"].data_ptr() == (
                bank["recurrent_state"][DECLARED_STATE_SLOT].data_ptr()
            ), (
                f"step {step}: bank {index}'s recurrent carrier is not a view of "
                f"its own slot, so the layer's in-place advance is written where "
                f"the next step will not read it"
            )
            assert carrier["conv_state"].data_ptr() == (
                bank["conv_state"][DECLARED_STATE_SLOT].data_ptr()
            ), f"step {step}: bank {index}'s conv carrier is not a view of its own slot"
            assert carrier["is_prefill"] is False, (
                f"step {step}: bank {index} received is_prefill=True for a "
                f"single-token step"
            )

    # It ADVANCED, and the advance is in the bank the runner owns.
    for index, bank in enumerate(run.banks):
        recurrent = bank["recurrent_state"][DECLARED_STATE_SLOT]
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
        print(f"B03_BANK{index}_REC_MAXABS={float(recurrent.abs().max()):.3e}")

    # The values. A state carried wrongly changes these, which is what makes the
    # carry a measurement instead of a bookkeeping claim.
    torch.testing.assert_close(
        run.decode_out,
        run.reference_decode,
        rtol=DECLARED_RTOL,
        atol=DECLARED_ATOL,
    )
    print(
        "B03_DECODE_MAXABS="
        f"{float((run.decode_out - run.reference_decode).abs().max()):.3e}"
    )

    # The prefill leg, through the same runner-built carriers. It is asserted here
    # rather than left uncompared because a prefill served from the wrong slot, or
    # from a carrier that is a copy, would leave the bank right and the OUTPUT
    # wrong -- and the decode comparison above continues from this leg's state, so a
    # silent prefill error would arrive as an unexplained decode failure.
    torch.testing.assert_close(
        run.prefill_out,
        run.reference_prefill,
        rtol=DECLARED_RTOL,
        atol=DECLARED_ATOL,
    )
    print(
        "B03_PREFILL_MAXABS="
        f"{float((run.prefill_out - run.reference_prefill).abs().max()):.3e}"
    )


def test_kda_runner_state_b04_the_decode_arm_reads_the_declared_seam_counts(
    run: SimpleNamespace,
) -> None:
    """The D13 route predicate, form R-2: decode arm only, one arm, two steps.

    Three seams at six each -- three layers over two steps -- and both chunked
    seams at exactly zero, because neither accepts an entering state and a decode
    driven through one would restart the recurrence on every step. The two zeros
    are the readings this item exists for.
    """
    counts = run.decode_counts
    print(f"B04_DECODE_COUNTS={counts}")
    for seam in ("conv", "gate", "decode"):
        dispatches, fallbacks = counts[seam]
        assert dispatches == DECLARED_DECODE_DISPATCHES, (
            f"{seam} read {dispatches}, expected {DECLARED_DECODE_DISPATCHES} -- "
            f"{DECLARED_STACK_LAYERS} per step over {DECLARED_DECODE_STEPS} steps"
        )
        assert fallbacks == DECLARED_FALLBACKS, (
            f"{seam} reported {fallbacks} torch fallback(s); P13 admits none"
        )
    for seam in ("intra", "inter"):
        dispatches, fallbacks = counts[seam]
        assert dispatches == DECLARED_CHUNKED_DISPATCHES_ON_DECODE, (
            f"{seam} read {dispatches} dispatch(es) on a decode arm; a "
            f"single-token decode step must never enter a chunked seam"
        )
        assert fallbacks == DECLARED_FALLBACKS


def test_kda_runner_state_b05_a_cleared_bank_misses_the_reference() -> None:
    """The non-vacuity control on B03 (D1.5).

    B03 asserts the decode outputs match the reference. That reading only means
    "the runner carried the state" if the same comparison FAILS when the state is
    not carried. Here the recurrent state is cleared after the prefill -- what a
    runner that allocated a fresh bank per step, or sliced the wrong slot, would
    leave behind -- and the outputs must then miss the reference by more than the
    tolerance.

    The guard below is the reason this control cannot pass vacuously: if the
    reference's own decode were insensitive to the entering state, the comparison
    would fail for a reason that has nothing to do with the carry.
    """
    world = _world(clear_state_before_decode=True)
    for index, bank in enumerate(world.banks):
        # The cleared world still ADVANCED from zero, so the failure below is a
        # wrong state and not an untouched bank.
        assert float(bank["recurrent_state"].abs().max()) > 0.0, (
            f"bank {index} is still all zeros after two decode steps, so this "
            f"control is measuring a dead drive rather than a dropped carry"
        )
    gap = float((world.decode_out - world.reference_decode).abs().max())
    print(f"B05_CLEARED_DECODE_MAXABS={gap:.3e}")
    tolerance = DECLARED_ATOL + DECLARED_RTOL * float(
        world.reference_decode.abs().max()
    )
    if gap <= tolerance:
        raise VacuousControlError(
            f"clearing the recurrent state moved the decode outputs by {gap:.3e}, "
            f"inside the declared tolerance {tolerance:.3e}; B03's pass therefore "
            f"does not discriminate a carried state from a dropped one"
        )
