"""The layer dump: configured it writes the stack's own named tensors, unset it does nothing.

Read on the tiny stack, through the runner's own converter:

1. With ``VLLM_NEURON_DUMP_LAYER_STREAMS=<dir>`` set, the converter hands the model
   ``collect_layer_streams``, the root returns the dump's tensors after its logits, and
   ``NeuronModelRunner._take_layer_stream_dump`` writes one file per name the model
   declares. An ``after_layer_<i>.pt`` is bit-equal to the tensor the next layer was
   handed; the last of them has no next layer, so it is replayed through the final norm
   and the head instead. Every tapped layer's own tensors are bit-equal to what hooks on
   that layer saw, and its token mapping shows each real token reaching exactly its top-k
   experts.
2. With the variable unset, the converter hands the model no such keyword, the forward
   returns a bare tensor, no file is written, and the logits are bit-equal to the
   configured run's.
3. Neither module the dump touches imports the NxDI stack.
4. The latent-cache taps: the five names sit between the index rows and the projection in
   the order the attention half appends them; the cache file holds the bank as it stood at
   the tap even after a later write into the caller's bank; and the rows the write named
   carry the values the write wrote, which a planted row offset breaks.

The dump exists to be read on a device serve, so the save happens outside the traced
forward and the model is told what to collect through an argument. Nothing here writes the
model's environment for it: the runner reads the variable where its production copy reads
it.

    VLLM_NEURON_CPU_MODE=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python -m pytest \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_layer_stream_dump.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.model.glm5_next import model_fp8
from vllm_neuron.model.glm5_next.model_fp8 import (
    DUMP_DSA_TAP_NAMES,
    DUMP_STREAM_LAYERS,
    dump_tap_layers,
    dump_tap_names,
    layer_dump_names,
)
from vllm_neuron.functional.attention import mla_sparse as mla_sparse_module
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

# The tiny fixtures, imported rather than re-built: the stack, its seeds and its dials come
# from the forward file and the runner-shaped caches from the end-to-end file.
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny

pytestmark = [pytest.mark.fast, pytest.mark.forked]

# The environment variable that configures the dump. Named once here; the runner reads it.
DUMP_VARIABLE = "VLLM_NEURON_DUMP_LAYER_STREAMS"

# The file the dump writes per kept layer, the name the depth comparison reads.
DUMP_NAME = "after_layer_{index}.pt"

# The modules the dump touches, relative to the repository root. The import scan below
# reads them whole.
SCANNED = (
    "vllm_neuron/model/glm5_next/model_fp8.py",
    "vllm_neuron/vllm/worker/neuron_model_runner.py",
)

# The vendor package this platform does not import. Spelled in two pieces so a scan for the
# token does not read this scanner's own needle as an import of it.
NXDI = "neuronx" + "_distributed"


def _require_cpu_mode() -> None:
    """Refuse unless CPU mode is set in the process environment, where the seams read it."""
    if os.environ.get("VLLM_NEURON_CPU_MODE") != "1":
        raise tiny.VacuousControlError(
            f"VLLM_NEURON_CPU_MODE is "
            f"{os.environ.get('VLLM_NEURON_CPU_MODE')!r}; the seams read it at import "
            f"time, so a fixture that set it here would measure the wrong backend"
        )


def _bound_root():
    """A tiny root with its caches bound, ready for one prefill through the converter."""
    fixture = e2e._fixture()
    root = fixture["root"]
    root.bind_kv_cache(e2e._runner_shaped_caches(root))
    return fixture, root


def _runner_for(root):
    """A runner shell carrying the attributes the converter and the dump read.

    ``__init__`` is not run, so the dump's directory is resolved through the same reader the
    constructor calls and the shell reads the environment the way the real runner does.
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
    """One prompt as long as the stack's token count, from the stack's own seed."""
    return torch.randint(
        0,
        tiny.STACK_VOCAB_SIZE,
        (tiny.STACK_TOKENS,),
        generator=torch.Generator().manual_seed(tiny.SEED_STACK_IDS),
        dtype=torch.int64,
    )


def _prefill(runner, root, prompt):
    """One prefill step, translated by the converter and run through the root's ``__call__``."""
    converted = e2e._model_kwargs(
        runner, input_ids=prompt, cached=0, sampling_row=tiny.STACK_TOKENS - 1
    )
    return converted, root(**converted)


def _record_the_streams_each_layer_was_handed(layers) -> tuple[dict, list]:
    """Pre-hooks that clone every layer's first positional argument, keyed by layer index.

    A clone and not the object: the comparison below is by value, and recording the live
    tensor would pass on identity alone.
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


def _record_one_tapped_layer(layer, monkeypatch) -> tuple[dict, list]:
    """Clones of the tensors one tapped layer makes, recorded where the layer makes them.

    The attention output, the expert block's two ends and the routed half come from module
    hooks. ``route_tokens`` and ``shared_expert_mm`` are methods called directly, so each is
    wrapped and its return recorded on the way out. The token mapping needs no reference of
    its own: it is read against its own property below.
    """
    seen: dict[str, torch.Tensor] = {}

    def _record_attention(module, args, output):
        seen["attention_output"] = output.detach().clone()

    def _record_input(module, args, kwargs):
        normed = kwargs.get("normed_hidden_states", args[1] if len(args) > 1 else None)
        seen["mlp_input1"] = normed.detach().clone()

    def _record_output(module, args, output):
        seen["mlp_output"] = output.detach().clone()

    def _record_routed(module, args, output):
        seen["routed_output"] = output.detach().clone()

    bank = layer.mlp.experts
    routed = bank.route_tokens

    def _record_router(*args, **kwargs):
        logits, index, affinities = routed(*args, **kwargs)
        seen["router_logits"] = logits.detach().clone()
        seen["router_topk_indices"] = index.detach().clone()
        seen["router_topk_weights"] = torch.gather(
            affinities, -1, index.long()
        ).detach().clone()
        seen["router_affinities"] = affinities.detach().clone()
        return logits, index, affinities

    monkeypatch.setattr(bank, "route_tokens", _record_router)
    shared = getattr(layer.mlp, "shared_experts", None)
    if shared is not None:
        shared_mm = shared.shared_expert_mm

        def _record_shared(*args, **kwargs):
            output = shared_mm(*args, **kwargs)
            seen["shared_output"] = output.detach().clone()
            return output

        monkeypatch.setattr(shared, "shared_expert_mm", _record_shared)
    handles = [
        layer.attention.register_forward_hook(_record_attention),
        layer.mlp.register_forward_pre_hook(_record_input, with_kwargs=True),
        layer.mlp.register_forward_hook(_record_output),
        bank.register_forward_hook(_record_routed),
    ]
    indexer = getattr(layer.attention, "indexer", None)
    if indexer is not None:
        norm = layer._input_norm

        def _record_norm(single_stream):
            normed = norm(single_stream)
            seen["attn_hc_collapsed"] = single_stream.detach().clone()
            seen["attn_input_normed"] = normed.detach().clone()
            return normed

        def _record_index_rows(module, args, output):
            seen["index_rows"] = output[:5].detach().clone()

        project = layer.attention.project_output

        def _record_projection(attn_out, collector=None):
            # The projection reads the flattened head axis, so the reference for its input
            # is this same tensor reshaped -- a view, not a second arithmetic.
            # ``o_proj_reduced`` needs no reference of its own: it is the attention module's
            # return, which the hook above already holds.
            seen["o_proj_input"] = (
                attn_out.detach().to(torch.float32).reshape(int(attn_out.shape[0]), -1)
            )
            return project(attn_out, collector)

        # The references for the latent taps come from either side of the cache and not
        # from the tap sites: the projection that makes the latent, and the seam the bank is
        # handed to. The clamp is recorded rather than re-derived, because a padded chunk
        # gathers its rows and the raw latent would then not be what the write carried.
        project_latent = layer.attention.project_query_and_latent

        def _record_latent(hidden_states):
            query, kv_latent = project_latent(hidden_states)
            seen["latent_written"] = kv_latent.detach().clone()
            return query, kv_latent

        attend = layer.attention.attend

        def _record_attend(hidden_states, latent_cache, *args, **kwargs):
            # ``_slots`` is the bank's row count, because the layer is handed the bank itself
            # rather than a window of it. Which of those rows belong to the request is the
            # table's answer, so the table and the page size are recorded beside it. Both
            # are read out of the keywords rather than defaulted: the layer requires them,
            # so an absent one raises here instead of reading as None.
            seen["_clamped"] = kwargs.get("prefill_end_position") is not None
            seen["_slots"] = int(latent_cache.shape[0])
            seen["_pages"] = [
                int(entry) for entry in kwargs["block_table_row"].flatten().tolist()
            ]
            seen["_page_size"] = int(kwargs["page_size"])
            return attend(hidden_states, latent_cache, *args, **kwargs)

        seam = mla_sparse_module.mla_sparse_attention

        def _record_seam(q_lift, c_kv, *args, **kwargs):
            seen["q_lift"] = q_lift.detach().clone()
            seen["cache_rows"] = c_kv.detach().clone()
            attended = seam(q_lift, c_kv, *args, **kwargs)
            seen["attended_latent"] = attended.detach().clone()
            return attended

        monkeypatch.setattr(layer, "_input_norm", _record_norm)
        monkeypatch.setattr(layer.attention, "project_output", _record_projection)
        monkeypatch.setattr(layer.attention, "project_query_and_latent", _record_latent)
        monkeypatch.setattr(layer.attention, "attend", _record_attend)
        monkeypatch.setattr(mla_sparse_module, "mla_sparse_attention", _record_seam)
        handles.append(indexer.register_forward_hook(_record_index_rows))
    return seen, handles


def _nxdi_import_lines(text: str) -> list[str]:
    """Every line of ``text`` that imports the NxDI stack, in either import form."""
    return [
        line.strip()
        for line in text.splitlines()
        if re.match(rf"\s*(import|from)\s+{NXDI}", line)
    ]


def test_a_configured_dump_writes_one_file_per_declared_name_bit_equal_to_the_layers(
    tmp_path, monkeypatch
):
    """One file per declared name: the kept streams, then every tapped layer's tensors."""
    _require_cpu_mode()
    save_dir = tmp_path / "layer-streams"
    monkeypatch.setenv(DUMP_VARIABLE, str(save_dir))
    _, root = _bound_root()
    runner = _runner_for(root)
    layers = list(root.model.layers)
    names = layer_dump_names(root)
    taps = dump_tap_layers(layers)
    kept = min(DUMP_STREAM_LAYERS, len(layers))
    seen, handles = _record_the_streams_each_layer_was_handed(layers)
    assert taps, (
        f"this stack of {len(layers)} layer(s) holds no layer with an expert block, so there "
        f"would be nothing to tap"
    )
    tapped = {}
    for tap in taps:
        one, tap_handles = _record_one_tapped_layer(layers[tap], monkeypatch)
        tapped[tap] = one
        handles += tap_handles
    try:
        converted, output = _prefill(runner, root, _prompt())
    finally:
        for handle in handles:
            handle.remove()

    assert converted.get("collect_layer_streams") is True, (
        f"the converter handed the model {sorted(converted)}; a configured dump needs the "
        f"collection keyword on the one mapping every call site goes through"
    )
    expected = [f"after_layer_{index}" for index in range(kept)]
    for tap in taps:
        expected += [f"layer{tap}_{suffix}" for suffix in dump_tap_names(layers[tap])]
    assert names == tuple(expected), (
        f"the model names its dump {list(names)}, which is not the order read below"
    )
    assert isinstance(output, tuple) and len(output) == 1 + len(names), (
        f"the root returned {type(output).__name__} of "
        f"{len(output) if isinstance(output, tuple) else 1}; a collecting forward returns "
        f"the logits and one tensor per declared name, flat"
    )

    logits = NeuronModelRunner._take_layer_stream_dump(runner, output, is_prefill=True)
    files = sorted(save_dir.glob("*.pt"))
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
            expected = seen[index + 1].float()
            assert torch.equal(dumped, expected), (
                f"after_layer_{index}.pt is not the tensor layer {index + 1} was handed; "
                f"max abs delta {float((dumped - expected).abs().max())}"
            )
        else:
            # The last file has no next layer, so it is checked against the graph's own
            # logits: collapsed and normed the way the stack does, then projected through
            # the head, it has to reproduce them exactly.
            collapsed = dumped.to(table.dtype).mean(dim=1).to(table.dtype)
            normed = root.model._rms_norm(collapsed, root.model.norm_weight)
            rows = torch.index_select(
                normed, dim=0, index=converted["sampling_positions"]
            )
            replayed = torch.nn.functional.linear(rows, root._head_weight())
            same = torch.equal(replayed, logits)
            assert same, (
                f"after_layer_{index}.pt is not the tensor the final norm consumed; the "
                f"replayed logits differ by {float((replayed - logits).abs().max())}"
            )

    # The taps are read where the layer makes them. The expert block returns the seams'
    # dtype and the feed-forward half casts it back to the collapsed stream's, which is the
    # table's, so the reference for that output carries the cast and no other one needs it.
    # An integer tensor is saved as it stands, so indices are compared as choices.
    top_k = int(root.model.text_config.num_experts_per_tok)
    for tap in taps:
        for suffix in dump_tap_names(layers[tap]):
            dumped = torch.load(
                save_dir / f"layer{tap}_{suffix}.pt", map_location="cpu", weights_only=True
            )
            if suffix == "token_position_to_id":
                # The mapping's own property, which needs no reference tensor: every real
                # token position appears exactly top-k times, once per expert it was routed
                # to. Padded positions carry the mapping's filler and are not counted.
                real = dumped[dumped >= 0]
                counts = torch.bincount(real.flatten().long(), minlength=tiny.STACK_TOKENS)
                counts = counts[: tiny.STACK_TOKENS]
                assert torch.equal(counts, torch.full_like(counts, top_k)), (
                    f"layer{tap} routed its real tokens {counts.tolist()} times against the "
                    f"{top_k} experts each was given"
                )
                continue
            if suffix == "o_proj_partial":
                # This fixture resolves no tensor-parallel coordinator, so nothing is added
                # between the partial and the whole and the two differ by the output cast
                # alone. A reduce that adds is read further down, under a stand-in group.
                reduced = _dumped(save_dir, tap, "o_proj_reduced")
                cast = dumped.to(table.dtype).float()
                same = torch.equal(cast, reduced)
                assert same, (
                    f"layer{tap}_o_proj_partial.pt is not the reduced whole under the output "
                    f"cast, and no coordinator added anything; they differ by "
                    f"{float((cast - reduced).abs().max())}"
                )
                continue
            if suffix == "write_rows":
                # Nothing here restates the write's arithmetic: an unpadded prefill starting
                # at slot zero writes consecutive slots, one per token, and this fixture's
                # table names its pages in ascending order from block zero, so the physical
                # rows are those same consecutive numbers.
                slots = int(tapped[tap]["_slots"])
                page = int(tapped[tap]["_page_size"])
                named = [entry for entry in tapped[tap]["_pages"] if entry >= 0]
                written_pages = sorted({int(row) // page for row in dumped.tolist()})
                consecutive = torch.arange(
                    int(dumped[0]), int(dumped[0]) + int(dumped.numel()), dtype=dumped.dtype
                )
                assert dumped.dtype is torch.int32, (
                    f"layer{tap}_write_rows.pt is {dumped.dtype}; the slots are choices and a "
                    f"float there is a number where a slot was asked for"
                )
                assert torch.equal(dumped, consecutive), (
                    f"layer{tap}_write_rows.pt is {dumped.tolist()[:8]}...; this fixture prefills "
                    f"unpadded, so the write names consecutive slots"
                )
                assert int(dumped.max()) < slots and int(dumped.min()) >= 0, (
                    f"layer{tap}_write_rows.pt names bank row {int(dumped.max())} in a bank of "
                    f"{slots} row(s); a row outside the bank is no slot at all"
                )
                # The bank's row count does not separate this request's rows from another's;
                # the table does. A written row belongs to a block the request's own table
                # names, or it is a page this request does not hold.
                assert set(written_pages) <= set(named), (
                    f"layer{tap}_write_rows.pt writes page(s) {written_pages[:8]} and the table "
                    f"this step was handed names {named[:8]}; a row outside the request's own "
                    f"pages reaches another sequence's rows"
                )
                continue
            if suffix == "latent_written":
                # The reference is the projection's own return, which is what the write
                # carried only while no row was collapsed onto another. A binding clamp
                # repeats the last real slot, so the slots the write named are consecutive
                # exactly when nothing was gathered; that is read from the rows themselves,
                # because the clamp argument can be present and still bind nothing.
                named = _dumped(save_dir, tap, "write_rows")
                gathered = not torch.equal(
                    named,
                    torch.arange(
                        int(named[0]), int(named[0]) + int(named.numel()), dtype=named.dtype
                    ),
                )
                assert not gathered, (
                    f"the write collapsed rows onto one another, so the projection's own "
                    f"return is no longer the tensor it wrote and this comparison would "
                    f"read a gather as a defect"
                )
            expected = tapped[tap]["attention_output" if suffix == "o_proj_reduced" else suffix]
            expected = expected.to(table.dtype) if suffix == "mlp_output" else expected
            expected = expected.float() if expected.is_floating_point() else expected
            assert torch.equal(dumped, expected), (
                f"layer{tap}_{suffix}.pt is not the tensor layer {tap} made; it holds "
                f"{tuple(dumped.shape)} of {dumped.dtype} against {tuple(expected.shape)} "
                f"of {expected.dtype}"
            )


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

    # The same weights, configured. Both fixtures are seeded, so a difference in the logits
    # is the dump's and nothing else. A second fixture rather than a second forward on the
    # first, so no bank or ring carries the earlier step's state into the comparison.
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
    files = sorted(save_dir.glob("*.pt"))
    assert torch.equal(gated_logits, plain_logits), (
        "the configured run changed the logits the ungated run produced"
    )
    assert len(files) == len(layer_dump_names(gated_root))


def test_the_modules_the_dump_touches_import_no_nxdi_stack():
    """Neither module the dump touches imports the NxDI stack."""
    root = Path(__file__).resolve().parents[5]
    for relative in SCANNED:
        path = root / relative
        assert path.is_file(), f"{relative} is not in this checkout at {path}"
        found = _nxdi_import_lines(path.read_text(encoding="utf-8"))
        assert found == [], f"{relative} imports the NxDI stack: {found}"


# The attention-side taps. Each test below runs one dumping prefill and reads the files
# against a tensor recorded where the layer itself makes it, so nothing re-derives the
# model's arithmetic.


def _dsa_tap_layer(layers):
    """The first tapped layer whose attention carries the indexer, or None."""
    for tap in dump_tap_layers(layers):
        if getattr(getattr(layers[tap], "attention", None), "indexer", None) is not None:
            return tap
    return None


def _dumped(save_dir, tap, suffix):
    """One tap file, loaded."""
    return torch.load(
        save_dir / f"layer{tap}_{suffix}.pt", map_location="cpu", weights_only=True
    )


def _run_a_dumping_prefill(save_dir, monkeypatch):
    """A dumping prefill on the tiny stack. Returns the layers, the tapped index and the runner."""
    monkeypatch.setenv(DUMP_VARIABLE, str(save_dir))
    _, root = _bound_root()
    runner = _runner_for(root)
    layers = list(root.model.layers)
    tap = _dsa_tap_layer(layers)
    if tap is None:
        pytest.skip("this fixture taps no layer whose attention carries the sparse indexer")
    return layers, tap, runner, root


def test_the_attention_half_taps_are_the_collapsed_stream_and_its_norm(tmp_path, monkeypatch):
    """``attn_hc_collapsed`` is what the half was handed and ``attn_input_normed`` is its norm."""
    _require_cpu_mode()
    save_dir = tmp_path / "layer-streams"
    layers, tap, runner, root = _run_a_dumping_prefill(save_dir, monkeypatch)
    layer = layers[tap]
    # The norm itself is the reference: it receives exactly the collapsed stream and returns
    # exactly the tensor the attention module is handed, so neither side is re-derived.
    seen = {}
    norm = layer._input_norm

    def _record_norm(single_stream):
        normed = norm(single_stream)
        seen["collapsed"] = single_stream.detach().clone()
        seen["normed"] = normed.detach().clone()
        return normed

    monkeypatch.setattr(layer, "_input_norm", _record_norm)
    _, output = _prefill(runner, root, _prompt())
    NeuronModelRunner._take_layer_stream_dump(runner, output, is_prefill=True)

    collapsed = _dumped(save_dir, tap, "attn_hc_collapsed")
    normed = _dumped(save_dir, tap, "attn_input_normed")
    assert torch.equal(collapsed, seen["collapsed"].float()), (
        f"layer{tap}_attn_hc_collapsed.pt is not the stream the attention half was handed; "
        f"max abs delta {float((collapsed - seen['collapsed'].float()).abs().max())}"
    )
    assert torch.equal(normed, seen["normed"].float()), (
        f"layer{tap}_attn_input_normed.pt is not that stream normalised; max abs delta "
        f"{float((normed - seen['normed'].float()).abs().max())}"
    )


def test_the_index_rows_tap_holds_the_rows_the_indexer_emitted(tmp_path, monkeypatch):
    """``index_rows`` is the indexer's first rows, unaltered, and every real column is causal."""
    _require_cpu_mode()
    save_dir = tmp_path / "layer-streams"
    layers, tap, runner, root = _run_a_dumping_prefill(save_dir, monkeypatch)
    seen = {}

    def _record_indexer(module, args, output):
        seen["emitted"] = output.detach().clone()

    handle = layers[tap].attention.indexer.register_forward_hook(_record_indexer)
    try:
        output = _prefill(runner, root, _prompt())[1]
    finally:
        handle.remove()
    NeuronModelRunner._take_layer_stream_dump(runner, output, is_prefill=True)

    rows = _dumped(save_dir, tap, "index_rows")
    emitted = seen["emitted"]
    assert rows.dtype is emitted.dtype, (
        f"layer{tap}_index_rows.pt is {rows.dtype} and the indexer emitted {emitted.dtype}; "
        f"a cast would turn the -1 sentinel into a number the comparison reads as a column"
    )
    assert int(rows.shape[0]) == min(5, int(emitted.shape[0])), (
        f"the tap holds {int(rows.shape[0])} row(s); it declares the first five, or every row "
        f"of a shorter batch ({int(emitted.shape[0])} here)"
    )
    assert torch.equal(rows, emitted[: int(rows.shape[0])]), (
        f"layer{tap}_index_rows.pt is not the indexer's own first rows"
    )
    # The precondition the index rows have to meet for ``index_expand``: a real column of
    # row ``i`` names a token inside that row's own sequence, and ``-1`` names no token. A
    # fresh prefill starts at position 0, so row ``i``'s own position is ``i``.
    for row in range(int(rows.shape[0])):
        real = rows[row][rows[row] >= 0]
        if not int(real.numel()):
            continue
        assert int(real.max()) <= row, (
            f"row {row} selects token {int(real.max())}, which is past its own position"
        )


def test_the_output_projection_taps_are_the_input_the_partial_and_the_whole(tmp_path, monkeypatch):
    """The partial is this rank's own share, so a reduce that adds shows in one file only."""
    _require_cpu_mode()
    save_dir = tmp_path / "layer-streams"
    layers, tap, runner, root = _run_a_dumping_prefill(save_dir, monkeypatch)
    added = 7.0

    class _AddingGroup:
        """A stand-in for the tensor-parallel coordinator: it reduces by adding in place."""

        @staticmethod
        def all_reduce(tensor):
            tensor.add_(added)

    monkeypatch.setattr(model_fp8, "_resolve_tp_group", lambda: _AddingGroup)
    seen = {}
    project = layers[tap].attention.project_output

    def _record_projection(attn_out, collector=None):
        seen["handed"] = attn_out.detach().clone()
        return project(attn_out, collector)

    monkeypatch.setattr(layers[tap].attention, "project_output", _record_projection)
    _, output = _prefill(runner, root, _prompt())
    NeuronModelRunner._take_layer_stream_dump(runner, output, is_prefill=True)

    entered = _dumped(save_dir, tap, "o_proj_input")
    partial = _dumped(save_dir, tap, "o_proj_partial")
    whole = _dumped(save_dir, tap, "o_proj_reduced")
    handed = seen["handed"]
    delta = (whole - partial).abs()
    assert entered.shape[0] == handed.shape[0] and entered.ndim == 2, (
        f"layer{tap}_o_proj_input.pt is {tuple(entered.shape)}; the projection reads its input "
        f"as [tokens, this rank's value width] and was handed {tuple(handed.shape)}"
    )
    assert partial.shape == whole.shape, (
        f"the partial is {tuple(partial.shape)} and the reduced whole is {tuple(whole.shape)}; "
        f"a reduction changes values and not shape"
    )
    # The cast is part of the claim, so the reference carries it rather than a tolerance
    # absorbing it: the method adds in float32 and hands the whole back in the input's
    # dtype, so the partial plus the added amount, under that same cast, is the file.
    expected = (partial + added).to(handed.dtype).float()
    assert torch.equal(whole, expected), (
        f"the reduce added {added} to every element, so the reduced file must be the partial "
        f"plus that under the output cast; they differ by "
        f"{float((whole - expected).abs().max())} and the raw gap is "
        f"{float(delta.min())}..{float(delta.max())}, which is what a partial holding the "
        f"sum itself would give"
    )


LATENT_TAP_NAMES = (
    "latent_written",
    "write_rows",
    "cache_rows",
    "attended_latent",
    "q_lift",
)


def test_the_latent_taps_are_declared_between_the_index_rows_and_the_projection():
    """The five latent names sit between the index rows and the projection, once each."""
    names = list(DUMP_DSA_TAP_NAMES)
    between = tuple(names[names.index("index_rows") + 1 : names.index("o_proj_input")])
    assert between == LATENT_TAP_NAMES, (
        f"the names between the index rows and the projection are {list(between)}; the "
        f"attention half appends {list(LATENT_TAP_NAMES)} in that order, and the file names are "
        f"positional, so any other order labels the wrong tensor"
    )
    assert [names.count(name) for name in LATENT_TAP_NAMES] == [1] * len(LATENT_TAP_NAMES), (
        f"a latent tap name is declared more than once in {names}"
    )


def test_the_latent_cache_taps_hold_the_write_its_rows_and_the_bank(tmp_path, monkeypatch):
    """The cache file survives a later bank write, and the written rows carry the written values."""
    _require_cpu_mode()
    save_dir = tmp_path / "layer-streams"
    layers, tap, runner, root = _run_a_dumping_prefill(save_dir, monkeypatch)
    attention = layers[tap].attention
    attend = attention.attend
    seen = {}

    def _record_attend(hidden_states, latent_cache, *args, **kwargs):
        # The carrier is the whole bank, so this records the bank itself; the keywords the
        # layer requires travel on untouched.
        seen["carrier"] = latent_cache
        return attend(hidden_states, latent_cache, *args, **kwargs)

    monkeypatch.setattr(attention, "attend", _record_attend)
    _, output = _prefill(runner, root, _prompt())
    carrier = seen["carrier"]
    bank_at_the_tap = carrier[:, 0, :].detach().clone()
    # A later write into the caller's bank, after the forward and before the files are
    # written, is the bank a decode step would leave behind. The tap holds a clone, so the
    # file must still read the bank as it stood at the tap. The second assertion below shows
    # this write really landed, so the first cannot pass by the write never happening.
    sentinel = -9.0
    with torch.no_grad():
        carrier.fill_(sentinel)
    NeuronModelRunner._take_layer_stream_dump(runner, output, is_prefill=True)

    written = _dumped(save_dir, tap, "latent_written")
    rows = _dumped(save_dir, tap, "write_rows")
    cached = _dumped(save_dir, tap, "cache_rows")
    attended = _dumped(save_dir, tap, "attended_latent")
    lifted = _dumped(save_dir, tap, "q_lift")
    bank_dtype = carrier.dtype
    assert rows.dtype is torch.int32 and rows.ndim == 1, (
        f"write_rows is {tuple(rows.shape)} of {rows.dtype}; the slots the write named are one "
        f"integer per token, and a float there is a number where a slot was asked for"
    )
    assert written.ndim == 2 and int(written.shape[0]) == int(rows.numel()), (
        f"latent_written is {tuple(written.shape)} against {int(rows.numel())} written row(s); "
        f"the tap holds one latent per row the write names"
    )
    assert cached.shape == bank_at_the_tap.shape and int(cached.shape[1]) == int(
        written.shape[1]
    ), (
        f"cache_rows is {tuple(cached.shape)} and the bank the layer was handed is "
        f"{tuple(bank_at_the_tap.shape)}; the tap holds the whole bank at the latent's own "
        f"width, which is what the seam gathers its pages out of"
    )
    for name, tensor in (("attended_latent", attended), ("q_lift", lifted)):
        assert tensor.ndim == 3 and int(tensor.shape[0]) == int(written.shape[0]), (
            f"{name} is {tuple(tensor.shape)}; the seam works in [tokens, heads, latent] and "
            f"this prefill wrote {int(written.shape[0])} token(s)"
        )
        assert tensor.dtype is torch.float32, f"{name} is {tensor.dtype}; the dump widens floats"

    filled = torch.full_like(cached, sentinel)
    assert torch.equal(cached, bank_at_the_tap.float()), (
        f"cache_rows was written from a view of the bank, so the later write reached the "
        f"file: max abs delta "
        f"{float((cached - bank_at_the_tap.float()).abs().max())} against the bank at the "
        f"tap. This tap has to clone"
    )
    assert not torch.equal(cached, filled), (
        f"the later write into the bank did not land, so nothing here reads the clone"
    )

    # The cast is part of the identity: the write hands the bank ``kv_latent`` in the bank's
    # own dtype, so the rows read back equal the written values under that cast, not exactly.
    expected = written.to(bank_dtype).float()
    got = cached[rows.long()]
    assert torch.equal(got, expected), (
        f"the rows the write named do not carry the values it wrote: max abs delta "
        f"{float((got - expected).abs().max())} under the bank's own cast. Either the write "
        f"did not land or the rows are not the ones it used"
    )

    # Shift the rows by one and the identity has to break, or it was reading a bank whose
    # rows are indistinguishable and would pass on the wrong slots too.
    shifted = (rows.long() + 1).clamp(max=int(cached.shape[0]) - 1)
    offset_got = cached[shifted]
    assert not torch.equal(offset_got, expected), (
        f"the identity still held with every row shifted by one, so it does not depend on "
        f"the slots the write named and cannot witness a wrong offset"
    )


class _ViewThatDefersItsWrite:
    """A bank view whose in-place write is only recorded, so the bank keeps its old rows."""

    def __init__(self, view, pending):
        self._view = view
        self._pending = pending

    def index_copy_(self, dim, index, source):
        self._pending.append((dim, index.detach().clone(), source.detach().clone()))
        return self


class _BankThatDefersItsWrite:
    """A bank double for a device where an in-place write is not visible in its own graph.

    The first view it hands out only records its write; every later one is the bank's own,
    unwritten. The layer takes one view to write this step's rows through and one to hand
    the seam, in that order, so the seam is given a bank that does not carry them yet. Both
    counts are asserted, so a layer that took the views in the other order fails on the
    count instead of the deferral passing silently.
    """

    def __init__(self, bank):
        self._bank = bank
        self.pending: list = []
        self.views = 0

    def __getattr__(self, name):
        return getattr(self._bank, name)

    def __getitem__(self, key):
        self.views += 1
        if self.views == 1:
            return _ViewThatDefersItsWrite(self._bank[key], self.pending)
        return self._bank[key]

    def let_the_deferred_writes_land(self):
        """Apply what the forward only recorded, as a step boundary would."""
        with torch.no_grad():
            for dim, index, source in self.pending:
                self._bank[:, 0, :].index_copy_(dim, index, source)


def test_the_seam_reads_this_steps_rows_even_when_the_bank_write_lands_later(tmp_path, monkeypatch):
    """This step's rows ride beside the bank, so the seam attends them before the write lands.

    The seam is handed the bank itself, plus ``written`` and ``write_offset``, and overlays
    the rows on the window it assembles. The bank operand is therefore expected to be stale
    here, and what carries this step is read where it travels:

    * the bank the seam was handed holds none of this step's rows, which the sentinel makes
      unmistakable;
    * the rows it was handed beside the bank are the values the write carried;
    * and the same dispatch, replayed once the deferred write has landed, returns what it
      returned over the stale bank -- so the overlay, and not the state of the bank, is what
      the seam attends.
    """
    _require_cpu_mode()
    save_dir = tmp_path / "layer-streams"
    layers, tap, runner, root = _run_a_dumping_prefill(save_dir, monkeypatch)
    attention = layers[tap].attention
    attend = attention.attend
    seam = mla_sparse_module.mla_sparse_attention
    stale = -3.0
    seen = {}
    dispatches: list[tuple] = []

    def _hand_over_a_deferring_bank(hidden_states, latent_cache, *args, **kwargs):
        # The old rows are made recognisable first, so a stale read cannot be mistaken for a
        # written one; the layer then works against the double instead of the bank itself.
        with torch.no_grad():
            latent_cache.fill_(stale)
        seen["carrier"] = latent_cache
        seen["double"] = _BankThatDefersItsWrite(latent_cache)
        seen["inside"] = True
        try:
            return attend(hidden_states, seen["double"], *args, **kwargs)
        finally:
            seen["inside"] = False

    def _record_the_dispatch(q_lift, c_kv, *args, **kwargs):
        # Every sparse layer of the stack enters this seam and only the tapped layer's bank
        # is the deferring one, so the flag above keeps this record the tapped layer's.
        attended = seam(q_lift, c_kv, *args, **kwargs)
        if seen.get("inside"):
            dispatches.append((q_lift, c_kv, args, dict(kwargs), attended.detach().clone()))
        return attended

    monkeypatch.setattr(attention, "attend", _hand_over_a_deferring_bank)
    monkeypatch.setattr(mla_sparse_module, "mla_sparse_attention", _record_the_dispatch)
    _, output = _prefill(runner, root, _prompt())
    carrier, double = seen["carrier"], seen["double"]
    NeuronModelRunner._take_layer_stream_dump(runner, output, is_prefill=True)

    written = _dumped(save_dir, tap, "latent_written")
    rows = _dumped(save_dir, tap, "write_rows")
    cached = _dumped(save_dir, tap, "cache_rows")
    bank_dtype = carrier.dtype
    expected = written.to(bank_dtype).float()
    handed = cached[rows.long()]
    assert double.pending, (
        f"the double recorded no write, so the layer never wrote through the bank and a "
        f"write that lands late is not what is being read"
    )
    assert double.views == 2, (
        f"the layer took {double.views} view(s) of the bank; it takes one to write this step's "
        f"rows through and one to hand the seam, so any other count means the deferral did not "
        f"cover the write or the operand is not the bank"
    )
    assert len(dispatches) == 1, (
        f"the tapped layer entered the seam {len(dispatches)} time(s); one dispatch assembles "
        f"one window, and the replay below reads that one dispatch"
    )
    assert torch.equal(handed, torch.full_like(handed, stale)), (
        f"the bank the seam was handed already carries this step's rows, so the deferral did not "
        f"hold and an overlay that carried nothing would pass here too"
    )
    assert not torch.equal(handed, expected), (
        f"the rows the bank still holds are the written ones, so a seam that read the bank "
        f"alone would pass too and the two reads cannot be told apart"
    )

    # What carries this step instead: the values the write carried, handed to the seam beside
    # the bank under its own keyword. The cast is part of the identity for the same reason as
    # above -- the write hands the bank its latent in the bank's dtype.
    q_lift, operand, args, kwargs, attended = dispatches[0]
    overlaid = kwargs["written"].detach().to(bank_dtype).float()
    assert torch.equal(overlaid, expected), (
        f"the rows the seam was handed beside the bank are not the values the write carried: "
        f"max abs delta {float((overlaid - expected).abs().max())} under the bank's own cast"
    )
    assert int(kwargs["write_offset"].flatten()[0]) == 0, (
        f"this fixture prefills from position 0, so the row the overlay sits at is 0; the seam "
        f"was handed {kwargs['write_offset'].flatten().tolist()}"
    )

    double.let_the_deferred_writes_land()
    landed = carrier[:, 0, :].float()[rows.long()]
    replayed = seam(q_lift, operand, *args, **kwargs)
    assert torch.equal(landed, expected), (
        f"once the recorded write was applied the bank still does not carry the written rows: "
        f"max abs delta {float((landed - expected).abs().max())}. The in-place write is what "
        f"later steps read"
    )
    assert torch.equal(replayed, attended), (
        f"the same dispatch attended something else once the bank carried this step's rows, "
        f"so what it read over the stale bank was not this step: max abs delta "
        f"{float((replayed - attended).abs().max())}. The overlay has to carry the rows"
    )
