"""The layer dump: configured it writes the stack's own named tensors, unset it does nothing.

WHAT THIS FILE MEASURES, on the tiny stack and through the runner's own converter:

  1. ``VLLM_NEURON_DUMP_LAYER_STREAMS=<dir>`` set -> the converter hands the model
     ``collect_layer_streams``, the root returns the dump's tensors after its logits, and
     ``NeuronModelRunner._take_layer_stream_dump`` writes one file per name the model
     declares. An ``after_layer_<i>.pt`` is bit-equal to the tensor the NEXT layer was
     handed; the last of them is checked against the graph's own logits instead, because no
     layer follows it. The tapped layer's ``mlp_input1`` and ``mlp_output`` are bit-equal to
     what hooks on that layer's own expert block saw.
  2. The variable unset -> the converter hands the model no such keyword, the forward returns
     a bare tensor, no file is written, and the logits are bit-equal to the configured run's.
  3. Neither module this dump touches imports the NxDI stack.

The dump exists to be read on a device serve, so the save happens OUTSIDE the traced forward
and the model is told what to collect through an argument. Nothing here writes to the model's
environment for it: the runner reads the variable once, where its production copy reads it.

HOW TO RUN IT, print rows included:

    VLLM_NEURON_CPU_MODE=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
        python -m pytest -s -rA -q --timeout 60 -p no:cacheprovider \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_layer_stream_dump.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.model.glm5_next.model_fp8 import (
    DUMP_STREAM_LAYERS,
    dump_tap_layer,
    layer_dump_names,
)
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

# The landed tiny fixtures, imported rather than re-built: the stack, its seeds and its dials
# come from the forward item's file and the runner-shaped caches from the threading item's.
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as item

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The environment variable that configures the dump. Named once here, and read by the runner.
DUMP_VARIABLE = "VLLM_NEURON_DUMP_LAYER_STREAMS"

#: The file the dump writes per kept layer, the name the depth comparison reads.
DUMP_NAME = "after_layer_{index}.pt"

#: The tapped layer's two inner tensors, in the order the stack appends them.
TAP_SUFFIXES = ("mlp_input1", "mlp_output")

#: The modules the dump touches, relative to the repository root. The import scan below reads
#: these whole rather than a diff, which covers every line the increment added and the lines
#: around them too.
SCANNED = (
    "vllm_neuron/model/glm5_next/model_fp8.py",
    "vllm_neuron/vllm/worker/neuron_model_runner.py",
)

#: The vendor package this platform does not import. Spelled in two pieces on purpose: a
#: scan of a diff for the token would otherwise read this scanner's own needle as an
#: import of it, and a check that flags its own checker teaches nobody anything.
NXDI = "neuronx" + "_distributed"


def _require_cpu_mode() -> None:
    """The CPU lane's own flag, from the process environment where the seams read it."""
    if os.environ.get("VLLM_NEURON_CPU_MODE") != "1":
        raise item.VacuousControlError(
            f"VLLM_NEURON_CPU_MODE is "
            f"{os.environ.get('VLLM_NEURON_CPU_MODE')!r}; the seams read it at import "
            f"time, so a fixture that set it here would measure the wrong backend"
        )
    print(f"TINYDUMP|venue|cpu_mode=1|nki_simulator={os.environ.get('NKI_SIMULATOR')!r}"
          f"|target={os.environ.get('NEURON_PLATFORM_TARGET_OVERRIDE')!r}")


def _bound_root():
    """A tiny root with its caches bound, ready for one prefill through the converter."""
    fixture = e2e._fixture()
    root = fixture["root"]
    root.bind_kv_cache(e2e._runner_shaped_caches(root))
    return fixture, root


def _runner_for(root):
    """A runner that models the four attributes the converter reads and the dump's three.

    ``__init__`` is not run -- no engine is built to measure a translation -- so the dump's
    directory is resolved through the very reader the constructor calls, which is what makes
    this stand-in read the environment the same way the production runner does.
    """
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=["req-0"])
    runner.model = root
    runner.max_model_len = e2e.E2E_MAX_SEQ_LEN
    runner.max_num_reqs = e2e.E2E_MAX_NUM_SEQS
    runner._layer_stream_dump_dir = NeuronModelRunner._layer_stream_dump_dir_from_env()
    runner._layer_streams_dumped = False
    runner.rank_tensor = torch.tensor(0, dtype=torch.int32)
    return runner


def _prompt() -> torch.Tensor:
    """The forward item's own prompt: its length is the stack's token count and its seed is that file's."""
    return torch.randint(
        0,
        item.STACK_VOCAB_SIZE,
        (item.STACK_TOKENS,),
        generator=torch.Generator().manual_seed(item.SEED_STACK_IDS),
        dtype=torch.int64,
    )


def _prefill(runner, root, prompt):
    """One prefill step, translated by the converter and run through the root's ``__call__``."""
    converted = e2e._model_kwargs(
        runner, input_ids=prompt, cached=0, sampling_row=item.STACK_TOKENS - 1
    )
    return converted, root(**converted)


def _record_the_streams_each_layer_was_handed(layers) -> tuple[dict, list]:
    """Pre-hooks that clone every layer's first positional argument, keyed by layer index.

    A CLONE, not the object: the comparison below is a value comparison, and recording the
    live tensor would pass on identity alone.
    """
    seen: dict[int, torch.Tensor] = {}
    handles = []

    def _record(index):
        def _hook(module, args, kwargs):
            seen[index] = args[0].detach().clone()

        return _hook

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(_record(index), with_kwargs=True))
    return seen, handles


def _record_the_tapped_block(layer) -> tuple[dict, list]:
    """Hooks that clone the tapped layer's expert-block input after its norm, and its output."""
    seen: dict[str, torch.Tensor] = {}

    def _record_input(module, args, kwargs):
        normed = kwargs.get("normed_hidden_states", args[1] if len(args) > 1 else None)
        seen["mlp_input1"] = normed.detach().clone()

    def _record_output(module, args, output):
        seen["mlp_output"] = output.detach().clone()

    handles = [
        layer.mlp.register_forward_pre_hook(_record_input, with_kwargs=True),
        layer.mlp.register_forward_hook(_record_output),
    ]
    return seen, handles


def _nxdi_import_lines(text: str) -> list[str]:
    """Every line of ``text`` that imports the NxDI stack, in either import form."""
    return [
        line.strip()
        for line in text.splitlines()
        if re.match(rf"\s*(import|from)\s+{NXDI}", line)
    ]


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 1. the configured dump writes one file per layer, each the next layer's own input.
# ══════════════════════════════════════════════════════════════════════════════════════


def test_a_configured_dump_writes_one_file_per_layer_bit_equal_to_the_streams(
    tmp_path, monkeypatch
):
    """Exactly ``num_layers`` files, each bit-equal to the stream that layer handed on."""
    _require_cpu_mode()
    save_dir = tmp_path / "layer-streams"
    monkeypatch.setenv(DUMP_VARIABLE, str(save_dir))
    fixture, root = _bound_root()
    runner = _runner_for(root)
    layers = list(root.model.layers)
    names = layer_dump_names(root)
    tap = dump_tap_layer(layers)
    kept = min(DUMP_STREAM_LAYERS, len(layers))
    seen, handles = _record_the_streams_each_layer_was_handed(layers)
    assert tap is not None, (
        f"this stack of {len(layers)} layer(s) holds no sparse-attention layer with an "
        f"expert block, so the taps below would measure nothing"
    )
    tapped, tap_handles = _record_the_tapped_block(layers[tap])
    handles += tap_handles
    try:
        converted, output = _prefill(runner, root, _prompt())
    finally:
        for handle in handles:
            handle.remove()

    print(f"TINYDUMP|configured|dir={save_dir}|keyword_in_model_kwargs="
          f"{converted.get('collect_layer_streams')}|outputs={len(output)}"
          f"|layers={len(layers)}|kept_streams={kept}|tap_layer={tap}|names={list(names)}"
          f"|pre_hooks_recorded={sorted(seen)}|tapped={sorted(tapped)}")
    assert converted.get("collect_layer_streams") is True, (
        f"the converter handed the model {sorted(converted)}; a configured dump needs the "
        f"collection keyword on the one mapping every call site goes through"
    )
    assert names == tuple(
        [f"after_layer_{index}" for index in range(kept)]
        + [f"layer{tap}_{suffix}" for suffix in TAP_SUFFIXES]
    ), f"the model names its dump {list(names)}, which is not the order this item reads"
    assert isinstance(output, tuple) and len(output) == 1 + len(names), (
        f"the root returned {type(output).__name__} of "
        f"{len(output) if isinstance(output, tuple) else 1}; a collecting forward returns "
        f"the logits and one tensor per declared name, flat"
    )

    logits = NeuronModelRunner._take_layer_stream_dump(runner, output, is_prefill=True)
    files = sorted(save_dir.glob("*.pt"))
    print(f"TINYDUMP|files|count={len(files)}|want={len(names)}"
          f"|names={[f.name for f in files]}|logits_rows={tuple(logits.shape)}")
    assert [f.name for f in files] == sorted(f"{name}.pt" for name in names), (
        f"the dump wrote {[f.name for f in files]} for the names {list(names)}"
    )
    assert torch.is_tensor(logits), (
        f"the dump handed {type(logits).__name__} back to the step; every branch after it "
        f"reads the logits tensor"
    )

    table = root.model.embed_tokens_weight
    for index in range(kept):
        dumped = torch.load(
            save_dir / DUMP_NAME.format(index=index), map_location="cpu", weights_only=True
        )
        assert dumped.dtype is torch.float32, (
            f"after_layer_{index}.pt is {dumped.dtype}; the comparison reads fp32"
        )
        if index + 1 < len(layers):
            want = seen[index + 1].float()
            same = torch.equal(dumped, want)
            print(f"TINYDUMP|after_layer|{index}|shape={tuple(dumped.shape)}"
                  f"|bit_equal_to_the_input_of_layer_{index + 1}={same}")
            assert same, (
                f"after_layer_{index}.pt is not the tensor layer {index + 1} was handed; "
                f"max abs delta {float((dumped - want).abs().max())}"
            )
        else:
            # THE LAST FILE HAS NO NEXT LAYER, so it is checked against the graph's own
            # logits: collapsed and normed the way the stack collapses and norms, and
            # projected through the head, it has to reproduce them exactly.
            collapsed = dumped.to(table.dtype).mean(dim=1).to(table.dtype)
            normed = root.model._rms_norm(collapsed, root.model.norm_weight)
            rows = torch.index_select(
                normed, dim=0, index=converted["sampling_positions"]
            )
            replayed = torch.nn.functional.linear(rows, root._head_weight())
            same = torch.equal(replayed, logits)
            print(f"TINYDUMP|after_layer|{index}|shape={tuple(dumped.shape)}"
                  f"|replays_the_logits_through_the_final_norm={same}")
            assert same, (
                f"after_layer_{index}.pt is not the tensor the final norm consumed; the "
                f"replayed logits differ by {float((replayed - logits).abs().max())}"
            )

    # THE TAPS ARE READ AGAINST THE BLOCK'S OWN HOOKS. The expert block returns the seams'
    # dtype and the feed-forward half casts it back to the collapsed stream's, which is the
    # table's, so the output oracle carries that cast and the input oracle needs none.
    for suffix in TAP_SUFFIXES:
        dumped = torch.load(
            save_dir / f"layer{tap}_{suffix}.pt", map_location="cpu", weights_only=True
        )
        want = tapped[suffix]
        want = want.to(table.dtype) if suffix == "mlp_output" else want
        same = torch.equal(dumped, want.float())
        print(f"TINYDUMP|tap|layer{tap}_{suffix}|shape={tuple(dumped.shape)}"
              f"|dtype_on_the_hook={want.dtype}|bit_equal_to_the_hook={same}")
        assert same, (
            f"layer{tap}_{suffix}.pt is not what the hook on layer {tap}'s expert block "
            f"saw; max abs delta {float((dumped - want.float()).abs().max())}"
        )


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 2. unset, the dump is absent from the signature, writes nothing and changes nothing.
# ══════════════════════════════════════════════════════════════════════════════════════


def test_an_unset_dump_writes_no_file_and_leaves_the_logits_bit_equal(tmp_path, monkeypatch):
    """No keyword, no file, and the same logits the configured run produced."""
    _require_cpu_mode()
    monkeypatch.delenv(DUMP_VARIABLE, raising=False)
    plain_fixture, plain_root = _bound_root()
    plain_runner = _runner_for(plain_root)
    assert plain_runner._layer_stream_dump_dir is None
    plain_converted, plain_output = _prefill(plain_runner, plain_root, _prompt())
    plain_logits = NeuronModelRunner._take_layer_stream_dump(
        plain_runner, plain_output, is_prefill=True
    )
    wrote = sorted(str(p) for p in tmp_path.rglob("*.pt"))
    print(f"TINYDUMP|unset|keyword_in_model_kwargs="
          f"{'collect_layer_streams' in plain_converted}|returned="
          f"{type(plain_output).__name__}|files={len(wrote)}|kwargs={sorted(plain_converted)}")
    assert "collect_layer_streams" not in plain_converted, (
        "the converter added the collection keyword with no directory configured, so an "
        "ordinary serve would trace a signature it never traced before"
    )
    assert torch.is_tensor(plain_output), (
        f"the root returned {type(plain_output).__name__} with no dump configured; the "
        f"ungated signature is one tensor"
    )
    assert plain_logits is plain_output
    assert wrote == [], f"a run with the variable unset wrote {wrote}"

    # THE SAME WEIGHTS, CONFIGURED. Both fixtures are seeded, so a difference in the logits
    # is the dump's and nothing else. A second tree rather than a second forward on the
    # first: this way no bank or ring carries the earlier step's state into the comparison.
    save_dir = tmp_path / "layer-streams"
    monkeypatch.setenv(DUMP_VARIABLE, str(save_dir))
    gated_fixture, gated_root = _bound_root()
    assert torch.equal(plain_fixture["table"], gated_fixture["table"]), (
        "the two fixtures hold different embedding tables, so their logits cannot be "
        "compared bit for bit"
    )
    gated_runner = _runner_for(gated_root)
    _, gated_output = _prefill(gated_runner, gated_root, _prompt())
    gated_logits = NeuronModelRunner._take_layer_stream_dump(
        gated_runner, gated_output, is_prefill=True
    )
    same = torch.equal(gated_logits, plain_logits)
    files = sorted(save_dir.glob("*.pt"))
    print(f"TINYDUMP|logits_unchanged|bit_equal={same}"
          f"|max_abs_delta={float((gated_logits - plain_logits).abs().max())}"
          f"|files_written_by_the_configured_run={len(files)}")
    assert same, "the configured run changed the logits the ungated run produced"
    assert len(files) == len(layer_dump_names(gated_root))


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 3. the modules this dump touches import no NxDI stack.
# ══════════════════════════════════════════════════════════════════════════════════════


def test_the_modules_the_dump_touches_import_no_nxdi_stack():
    """Zero NxDI imports in either module, with the scanner shown to fire on a planted one."""
    root = Path(__file__).resolve().parents[5]
    hits = {}
    for relative in SCANNED:
        path = root / relative
        assert path.is_file(), f"{relative} is not in this checkout at {path}"
        hits[relative] = _nxdi_import_lines(path.read_text(encoding="utf-8"))
    planted = _nxdi_import_lines(f"import torch\nimport {NXDI}_inference as nxdi\n")
    print(f"TINYDUMP|nxdi_scan|"
          f"{'|'.join(f'{name}={len(found)}' for name, found in hits.items())}"
          f"|scanner_fires_on_a_planted_import={len(planted)}")
    assert planted == [f"import {NXDI}_inference as nxdi"], (
        f"the scanner did not flag a planted NxDI import, so a zero from it means nothing: "
        f"{planted}"
    )
    for relative, found in hits.items():
        assert found == [], f"{relative} imports the NxDI stack: {found}"
