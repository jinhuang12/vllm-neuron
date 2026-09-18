"""The layer dump: configured it writes the stack's own named tensors, unset it does nothing.

WHAT THIS FILE MEASURES, on the tiny stack and through the runner's own converter:

  1. ``VLLM_NEURON_DUMP_LAYER_STREAMS=<dir>`` set -> the converter hands the model
     ``collect_layer_streams``, the root returns the dump's tensors after its logits, and
     ``NeuronModelRunner._take_layer_stream_dump`` writes one file per name the model declares.
     An ``after_layer_<i>.pt`` is bit-equal to the tensor the NEXT layer was handed, and the
     last of them is checked against the graph's own logits instead, because no layer follows
     it. Every tapped layer's own tensors are bit-equal to what hooks on that layer saw, and
     its token mapping shows each real token reaching exactly its top-k experts.
  2. The variable unset -> the converter hands the model no such keyword, the forward returns
     a bare tensor, no file is written, and the logits are bit-equal to the configured run's.
  3. Neither module this dump touches imports the NxDI stack.
  4. The latent-cache taps: the five names sit between the index rows and the projection in the
     order the attention half appends them; the window file holds the window as it stood at the
     tap even after a later write into the caller's bank; and the rows the write named carry the
     values the write wrote, which a planted row offset breaks.

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

# The landed tiny fixtures, imported rather than re-built: the stack, its seeds and its dials
# come from the forward item's file and the runner-shaped caches from the threading item's.
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as item

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The environment variable that configures the dump. Named once here, and read by the runner.
DUMP_VARIABLE = "VLLM_NEURON_DUMP_LAYER_STREAMS"

#: The file the dump writes per kept layer, the name the depth comparison reads.
DUMP_NAME = "after_layer_{index}.pt"

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
    """The forward item's own prompt: its length is the stack's token count, its seed that file's."""
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


def _record_one_tapped_layer(layer, monkeypatch) -> tuple[dict, list]:
    """Clones of the tensors one tapped layer claims, read where the layer itself makes them.

    The attention output, the expert block's two ends and the routed half come from module
    hooks. Two do not: ``route_tokens`` and ``shared_expert_mm`` are methods called directly,
    so each is wrapped for the item and its return recorded on the way out. The token mapping
    has no oracle here and needs none -- it is read against its own declared property below.
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
            # THE WIDTH THE PROJECTION READS is the flattened head axis, so the input's
            # oracle is this same tensor reshaped -- a view, not a second arithmetic.
            # ``o_proj_reduced`` needs no oracle of its own: it IS the attention module's
            # return, which the hook above already holds, and the equality proves it.
            seen["o_proj_input"] = (
                attn_out.detach().to(torch.float32).reshape(int(attn_out.shape[0]), -1)
            )
            return project(attn_out, collector)

        # THE LATENT TAPS' ORACLES COME FROM EITHER SIDE OF THE CACHE, not from the tap sites:
        # the projection that makes the latent, and the seam that reads the window back. The
        # clamp is recorded rather than re-derived, because a padded chunk gathers its rows and
        # the raw latent would then not be what the write carried.
        project_latent = layer.attention.project_query_and_latent

        def _record_latent(hidden_states):
            query, kv_latent = project_latent(hidden_states)
            seen["latent_written"] = kv_latent.detach().clone()
            return query, kv_latent

        attend = layer.attention.attend

        def _record_attend(hidden_states, latent_cache, *args, **kwargs):
            seen["_clamped"] = kwargs.get("prefill_end_position") is not None
            seen["_slots"] = int(latent_cache.shape[0])
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


# ══════════════════════════════════════════════════════════════════════════════════════
# ITEM 1. the configured dump writes the model's declared tensors, each one its layer's own.
# ══════════════════════════════════════════════════════════════════════════════════════


def test_a_configured_dump_writes_one_file_per_declared_name_bit_equal_to_the_layers(
    tmp_path, monkeypatch
):
    """One file per declared name: the kept streams, then every tapped layer's own tensors."""
    _require_cpu_mode()
    save_dir = tmp_path / "layer-streams"
    monkeypatch.setenv(DUMP_VARIABLE, str(save_dir))
    fixture, root = _bound_root()
    runner = _runner_for(root)
    layers = list(root.model.layers)
    names = layer_dump_names(root)
    taps = dump_tap_layers(layers)
    kept = min(DUMP_STREAM_LAYERS, len(layers))
    seen, handles = _record_the_streams_each_layer_was_handed(layers)
    assert taps, (
        f"this stack of {len(layers)} layer(s) holds no layer with an expert block, so the "
        f"taps below would measure nothing"
    )
    # TWO LAYERS ARE TAPPED ON A STACK THAT HOLDS TWO, and this fixture may hold one. The
    # served model taps the first expert layer and the one after it; a stack whose first expert
    # layer is also its last has no second to tap, and the row below says which case ran.
    print(f"TINYDUMP|tap_layers|first={taps[0]}|tapped={list(taps)}|layers={len(layers)}"
          f"|the_layer_after_the_first_holds_experts={len(taps) == 2}")
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

    print(f"TINYDUMP|configured|dir={save_dir}|keyword_in_model_kwargs="
          f"{converted.get('collect_layer_streams')}|outputs={len(output)}"
          f"|layers={len(layers)}|kept_streams={kept}|tap_layers={list(taps)}"
          f"|names={list(names)}|pre_hooks_recorded={sorted(seen)}")
    assert converted.get("collect_layer_streams") is True, (
        f"the converter handed the model {sorted(converted)}; a configured dump needs the "
        f"collection keyword on the one mapping every call site goes through"
    )
    expected = [f"after_layer_{index}" for index in range(kept)]
    for tap in taps:
        expected += [f"layer{tap}_{suffix}" for suffix in dump_tap_names(layers[tap])]
    assert names == tuple(expected), (
        f"the model names its dump {list(names)}, which is not the order this item reads"
    )
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

    # THE TAPS ARE READ WHERE THE LAYER MAKES THEM. The expert block returns the seams' dtype
    # and the feed-forward half casts it back to the collapsed stream's, which is the table's,
    # so the output oracle carries that cast and no other oracle needs one. An integer tensor
    # is saved as it stands, so indices and positions are compared as choices, not as numbers.
    top_k = int(root.model.text_config.num_experts_per_tok)
    for tap in taps:
        for suffix in dump_tap_names(layers[tap]):
            dumped = torch.load(
                save_dir / f"layer{tap}_{suffix}.pt", map_location="cpu", weights_only=True
            )
            if suffix == "token_position_to_id":
                # THE MAPPING'S OWN PROPERTY, which is the reading and needs no oracle: every
                # real token position must appear exactly top-k times, once per expert it was
                # routed to. Padded positions carry the mapping's own filler and are not
                # counted here.
                real = dumped[dumped >= 0]
                counts = torch.bincount(real.flatten().long(), minlength=item.STACK_TOKENS)
                counts = counts[: item.STACK_TOKENS]
                print(f"TINYDUMP|tap|layer{tap}_{suffix}|shape={tuple(dumped.shape)}"
                      f"|dtype={dumped.dtype}|top_k={top_k}"
                      f"|count_per_real_token={counts.tolist()}")
                assert torch.equal(counts, torch.full_like(counts, top_k)), (
                    f"layer{tap} routed its real tokens {counts.tolist()} times against the "
                    f"{top_k} experts each was given"
                )
                continue
            if suffix == "o_proj_partial":
                # THE PARTIAL'S OWN PROPERTY, and the only reading of it from outside the
                # method: this fixture resolves no coordinator, so nothing is added between
                # the partial and the whole and the two differ by the output cast alone.
                # The reduce itself is read where a coordinator exists, in its own item.
                reduced = _dumped(save_dir, tap, "o_proj_reduced")
                cast = dumped.to(table.dtype).float()
                same = torch.equal(cast, reduced)
                print(f"TINYDUMP|tap|layer{tap}_{suffix}|shape={tuple(dumped.shape)}"
                      f"|dtype={dumped.dtype}|is_the_whole_under_the_output_cast={same}")
                assert same, (
                    f"layer{tap}_o_proj_partial.pt is not the reduced whole under the output "
                    f"cast, and no coordinator added anything; they differ by "
                    f"{float((cast - reduced).abs().max())}"
                )
                continue
            if suffix == "write_rows":
                # THE SLOTS' OWN PROPERTY, and no oracle restates the write's arithmetic here:
                # an unpadded prefill starting at slot zero writes consecutive slots, one per
                # token, and every one of them lies inside the window it was handed.
                slots = int(tapped[tap]["_slots"])
                consecutive = torch.arange(
                    int(dumped[0]), int(dumped[0]) + int(dumped.numel()), dtype=dumped.dtype
                )
                print(f"TINYDUMP|tap|layer{tap}_{suffix}|shape={tuple(dumped.shape)}"
                      f"|dtype={dumped.dtype}|first={int(dumped[0])}|last={int(dumped[-1])}"
                      f"|slots={slots}|consecutive={torch.equal(dumped, consecutive)}"
                      f"|clamped={bool(tapped[tap]['_clamped'])}")
                assert dumped.dtype is torch.int32, (
                    f"layer{tap}_write_rows.pt is {dumped.dtype}; the slots are choices and a "
                    f"float there is a number where a slot was asked for"
                )
                assert torch.equal(dumped, consecutive), (
                    f"layer{tap}_write_rows.pt is {dumped.tolist()[:8]}...; this fixture prefills "
                    f"unpadded, so the write names consecutive slots"
                )
                assert int(dumped.max()) < slots and int(dumped.min()) >= 0, (
                    f"layer{tap}_write_rows.pt names slot {int(dumped.max())} in a window of "
                    f"{slots}; a write outside the window reaches another sequence's rows"
                )
                continue
            if suffix == "latent_written":
                # THE ORACLE IS THE PROJECTION'S OWN RETURN, which is what the write carried only
                # while no row was collapsed onto another. A BINDING clamp repeats the last real
                # slot, so the slots the write named are consecutive exactly when nothing was
                # gathered -- and that is read from the sibling file rather than from the clamp
                # argument, which can be present and still bind nothing at all.
                named = _dumped(save_dir, tap, "write_rows")
                gathered = not torch.equal(
                    named,
                    torch.arange(
                        int(named[0]), int(named[0]) + int(named.numel()), dtype=named.dtype
                    ),
                )
                print(f"TINYDUMP|latent_written_precondition|layer={tap}"
                      f"|rows_gathered={gathered}|clamp_argument_given="
                      f"{bool(tapped[tap]['_clamped'])}")
                assert not gathered, (
                    f"the write collapsed rows onto one another, so the projection's own return "
                    f"is no longer the tensor it wrote and this comparison would read a gather "
                    f"as a defect"
                )
            want = tapped[tap]["attention_output" if suffix == "o_proj_reduced" else suffix]
            want = want.to(table.dtype) if suffix == "mlp_output" else want
            want = want.float() if want.is_floating_point() else want
            same = torch.equal(dumped, want)
            print(f"TINYDUMP|tap|layer{tap}_{suffix}|shape={tuple(dumped.shape)}"
                  f"|dtype={dumped.dtype}|bit_equal_to_the_layers_own_tensor={same}")
            assert same, (
                f"layer{tap}_{suffix}.pt is not the tensor layer {tap} made; it holds "
                f"{tuple(dumped.shape)} of {dumped.dtype} against {tuple(want.shape)} of "
                f"{want.dtype}"
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


# ---------------------------------------------------------------------------
# THE ATTENTION-SIDE TAPS
# ---------------------------------------------------------------------------
# Each item below runs one dumping prefill and reads the files against an oracle recorded
# where the layer itself makes the tensor, so nothing here re-derives the model's arithmetic.


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
    fixture, root = _bound_root()
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
    # THE ORACLE IS THE NORM ITSELF: it receives exactly the collapsed stream and returns
    # exactly the tensor the attention module is handed, so neither side is re-derived here.
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
    print(f"TINYDUMP|attn_half_taps|layer={tap}|collapsed={tuple(collapsed.shape)}"
          f"|normed={tuple(normed.shape)}|oracle_collapsed={tuple(seen['collapsed'].shape)}"
          f"|equal_collapsed={torch.equal(collapsed, seen['collapsed'].float())}"
          f"|equal_normed={torch.equal(normed, seen['normed'].float())}")
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
    print(f"TINYDUMP|index_rows|layer={tap}|file={tuple(rows.shape)}|dtype={rows.dtype}"
          f"|emitted={tuple(emitted.shape)}|sentinels={int((rows < 0).sum())}"
          f"|max_index={int(rows.max())}")
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
    # THE CALLER PRECONDITION THE ROWS MUST MEET (``index_expand.py:48``): a real column of
    # row ``i`` names a token inside that row's own sequence, and ``-1`` names no token.
    # A fresh prefill starts at position 0, so row ``i``'s own position IS ``i``.
    for row in range(int(rows.shape[0])):
        real = rows[row][rows[row] >= 0]
        if not int(real.numel()):
            continue
        assert int(real.max()) <= row, (
            f"row {row} selects token {int(real.max())}, which is past its own position"
        )


def test_the_output_projection_taps_are_the_input_the_partial_and_the_whole(tmp_path, monkeypatch):
    """The partial is this rank's own share: a reduce that adds is visible in one file only."""
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
    print(f"TINYDUMP|o_proj_taps|layer={tap}|input={tuple(entered.shape)}"
          f"|partial={tuple(partial.shape)}|reduced={tuple(whole.shape)}"
          f"|handed={tuple(handed.shape)}|added={added}|delta_min={float(delta.min())}"
          f"|delta_max={float(delta.max())}")
    assert entered.shape[0] == handed.shape[0] and entered.ndim == 2, (
        f"layer{tap}_o_proj_input.pt is {tuple(entered.shape)}; the projection reads its input "
        f"as [tokens, this rank's value width] and was handed {tuple(handed.shape)}"
    )
    assert partial.shape == whole.shape, (
        f"the partial is {tuple(partial.shape)} and the reduced whole is {tuple(whole.shape)}; "
        f"a reduction changes values and not shape"
    )
    # THE CAST IS PART OF THE CLAIM, so the oracle carries it rather than a tolerance
    # absorbing it: the method adds in float32 and hands the whole back in the input's
    # dtype, so the partial plus the added amount, under that same cast, IS the file.
    want = (partial + added).to(handed.dtype).float()
    same = torch.equal(whole, want)
    print(f"TINYDUMP|o_proj_reduce|layer={tap}|cast_to={handed.dtype}"
          f"|reduced_is_the_partial_plus_{added}_under_the_cast={same}")
    assert same, (
        f"the reduce added {added} to every element, so the reduced file must be the partial "
        f"plus that under the output cast; they differ by "
        f"{float((whole - want).abs().max())} and the raw gap is "
        f"{float(delta.min())}..{float(delta.max())}. A partial that carries the sum is the "
        f"clone this tap needs, missing"
    )


LATENT_TAP_NAMES = (
    "latent_written",
    "write_rows",
    "cache_rows",
    "attended_latent",
    "q_lift",
)


def test_the_latent_taps_are_declared_between_the_index_rows_and_the_projection():
    """The five names, in the order the attention half appends them, and nowhere else."""
    names = list(DUMP_DSA_TAP_NAMES)
    between = tuple(names[names.index("index_rows") + 1 : names.index("o_proj_input")])
    print(f"TINYDUMP|latent_tap_census|declared={len(names)}|between={list(between)}"
          f"|once_each={[names.count(name) for name in LATENT_TAP_NAMES]}")
    assert between == LATENT_TAP_NAMES, (
        f"the names between the index rows and the projection are {list(between)}; the "
        f"attention half appends {list(LATENT_TAP_NAMES)} in that order, and the file names are "
        f"positional, so any other order labels the wrong tensor"
    )
    assert [names.count(name) for name in LATENT_TAP_NAMES] == [1] * len(LATENT_TAP_NAMES), (
        f"a latent tap name is declared more than once in {names}"
    )


def test_the_latent_cache_taps_hold_the_write_its_rows_and_the_window(tmp_path, monkeypatch):
    """The window file survives a later bank write, and the written rows carry the written values."""
    _require_cpu_mode()
    save_dir = tmp_path / "layer-streams"
    layers, tap, runner, root = _run_a_dumping_prefill(save_dir, monkeypatch)
    attention = layers[tap].attention
    attend = attention.attend
    seen = {}

    def _record_attend(hidden_states, latent_cache, *args, **kwargs):
        seen["carrier"] = latent_cache
        return attend(hidden_states, latent_cache, *args, **kwargs)

    monkeypatch.setattr(attention, "attend", _record_attend)
    _, output = _prefill(runner, root, _prompt())
    carrier = seen["carrier"]
    window = carrier[:, 0, :].detach().clone()
    # A LATER WRITE INTO THE CALLER'S BANK, after the forward and before the files are written:
    # that is the window a decode step would leave behind. The tap holds a clone, so the file
    # must still read the window as it stood at the tap -- and the second assertion below shows
    # this write really landed, so the first one cannot pass by the write never happening.
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
    print(f"TINYDUMP|latent_taps|layer={tap}|written={tuple(written.shape)}/{written.dtype}"
          f"|rows={tuple(rows.shape)}/{rows.dtype}|cached={tuple(cached.shape)}/{cached.dtype}"
          f"|attended={tuple(attended.shape)}/{attended.dtype}"
          f"|q_lift={tuple(lifted.shape)}/{lifted.dtype}|bank_dtype={bank_dtype}"
          f"|window={tuple(window.shape)}|rows_first={rows[:4].tolist()}")
    assert rows.dtype is torch.int32 and rows.ndim == 1, (
        f"write_rows is {tuple(rows.shape)} of {rows.dtype}; the slots the write named are one "
        f"integer per token, and a float there is a number where a slot was asked for"
    )
    assert written.ndim == 2 and int(written.shape[0]) == int(rows.numel()), (
        f"latent_written is {tuple(written.shape)} against {int(rows.numel())} written row(s); "
        f"the tap holds one latent per row the write names"
    )
    assert cached.shape == window.shape and int(cached.shape[1]) == int(written.shape[1]), (
        f"cache_rows is {tuple(cached.shape)} and the window the layer was handed is "
        f"{tuple(window.shape)}; the tap holds the whole window at the latent's own width"
    )
    for name, tensor in (("attended_latent", attended), ("q_lift", lifted)):
        assert tensor.ndim == 3 and int(tensor.shape[0]) == int(written.shape[0]), (
            f"{name} is {tuple(tensor.shape)}; the seam works in [tokens, heads, latent] and "
            f"this prefill wrote {int(written.shape[0])} token(s)"
        )
        assert tensor.dtype is torch.float32, f"{name} is {tensor.dtype}; the dump widens floats"

    filled = torch.full_like(cached, sentinel)
    kept = torch.equal(cached, window.float())
    print(f"TINYDUMP|cache_rows_clone|layer={tap}|equals_the_window_at_the_tap={kept}"
          f"|equals_the_later_write={torch.equal(cached, filled)}|sentinel={sentinel}")
    assert kept, (
        f"cache_rows was written from a view of the bank, so the later write reached the file: "
        f"max abs delta {float((cached - window.float()).abs().max())} against the window at the "
        f"tap. This tap must clone"
    )
    assert not torch.equal(cached, filled), (
        f"the later write into the bank did not land, so nothing here tests the clone"
    )

    # THE CAST IS PART OF THE IDENTITY: the write hands the bank ``kv_latent`` in the bank's own
    # dtype, so the rows read back equal the written values UNDER that cast and not exactly.
    want = written.to(bank_dtype).float()
    got = cached[rows.long()]
    same = torch.equal(got, want)
    print(f"TINYDUMP|latent_write_identity|layer={tap}|cast_to={bank_dtype}|rows={int(rows.numel())}"
          f"|equal_under_the_cast={same}|max_abs={float((got - want).abs().max())}")
    assert same, (
        f"the rows the write named do not carry the values it wrote: max abs delta "
        f"{float((got - want).abs().max())} under the bank's own cast. Either the write did not "
        f"land or the rows are not the ones it used"
    )

    # THE FIXED POINT: shift the rows by one and the identity must break, or it was reading a
    # window whose rows are indistinguishable and would pass on the wrong slots too.
    shifted = (rows.long() + 1).clamp(max=int(cached.shape[0]) - 1)
    offset_got = cached[shifted]
    broken = not torch.equal(offset_got, want)
    print(f"TINYDUMP|latent_write_control|layer={tap}|offset=1|identity_breaks={broken}"
          f"|max_abs={float((offset_got - want).abs().max())}")
    assert broken, (
        f"the identity still held with every row shifted by one, so it does not depend on the "
        f"slots the write named and cannot witness a wrong offset"
    )

