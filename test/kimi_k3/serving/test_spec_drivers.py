# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 speculative-decoding serving drivers: cells, prompt files, TPOT reducer,
accepted-tokens reader and agreement scorer.

Set ``HF_HOME`` to the offline cache that holds MMLU-Pro at its pinned revision,
``KIMI_K3_CHECKPOINT`` to the checkpoint folder (default ``~/models/moonshotai/Kimi-K3``)
and ``KIMI_K3_REFERENCE_RUNS`` to the reference runs folder that holds ``seed-0`` and
``seed-1``.
"""

import json
import os
import random
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

SERVING = Path(__file__).resolve().parent
CHECKPOINT = os.environ.get("KIMI_K3_CHECKPOINT",
                            str(Path.home() / "models" / "moonshotai" / "Kimi-K3"))
REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
CKPT = "/home/ubuntu/models/moonshotai/Kimi-K3"
DRAFT_CONFIG = ('{"method": "dspark", "model": "/home/ubuntu/models/Inferact/Kimi-K3-DSpark-Block5", '
                '"num_speculative_tokens": 4, "draft_sample_method": "greedy"}')
PROFILER = '{"profiler": "cuda", "delay_iterations": 0, "max_iterations": 5}'


def _run(script, *args):
    return subprocess.run([sys.executable, str(SERVING / script)] + [str(a) for a in args],
                          capture_output=True, text=True)


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _form(d, max_len, buckets):
    return {"tensor_parallel_size": 32, "data_parallel_size": 2,
            "decode_context_parallel_size": d, "max_model_len": max_len,
            "decode_context_length_buckets": buckets,
            "serve_flags": ["--decode-context-parallel-size", str(d), "--max-model-len",
                            str(max_len), "--gpu-memory-utilization", "0.90",
                            "--reasoning-parser", "kimi_k3", "--no-enable-prefix-caching",
                            "--served-model-name", "us.moonshotai.kimi-k3", CKPT,
                            "--trust-remote-code", "--async-scheduling",
                            "--max-num-seqs", "16",
                            "--profiler-config", PROFILER],
            "neuron_profiler": {"activities": ["device_profile", "system_profile"],
                                "neuron_cores": [0],
                                "output_dir": f"<measurements folder>/profiles/dcp{d}",
                                "sys_trace_max_events_per_nc": None},
            "speculative_flags": ["--speculative-config", DRAFT_CONFIG]}


def test_cells_and_gates():
    """Item 1: 3 cells with their gates, forms, lengths, temperature 0 and serve flags; each
    cell's input + output + 2(k + 1) fits the 8,192 bucket."""
    doc = json.loads((SERVING / "spec_cells.json").read_text())
    faster = {"metric": "tpot", "rule": "comparator_over_candidate_at_least", "value": 1.3}
    accepted = {"metric": "accepted_tokens_per_step", "rule": "at_least", "value": 2.3}
    no_regression = {"metric": "tpot", "rule": "candidate_over_comparator_at_most", "value": 1.1}
    expected = [("8K c1", 1, 8, "DCP 4", 3, 3, [faster]),
                ("8K c8", 8, 8, "DCP 4", 3, 3, [faster, accepted]),
                ("8K c32", 32, 32, "DCP 16", 2, 3, [no_regression])]
    cells = [{"name": n, "context": 8192, "concurrency": c, "num_prompts": p, "form": f,
              "slots": s, "comparator_slots": cs, "k": 4, "input_len": 8048, "output_len": 128,
              "temperature": 0, "gates": g} for n, c, p, f, s, cs, g in expected]
    forms = {"DCP 4": _form(4, 32768, [2048, 8192, 16384, 28672]),
             "DCP 16": _form(16, 131072, [2048, 8192, 16384, 28672, 45056, 77824])}
    assert [c for c, e in zip(doc["cells"], cells) if c != e] == [] and len(doc["cells"]) == 3
    assert [name for name in forms if doc["forms"].get(name) != forms[name]] == []
    assert sorted(doc["forms"]) == sorted(forms)
    assert (doc["prompt_file"], doc["max_spread"]) == ("spec_prompts.jsonl", 0.1)
    over = []
    for c in doc["cells"]:
        form = doc["forms"][c["form"]]
        need = c["input_len"] + c["output_len"] + 2 * (c["k"] + 1)
        bucket = min(b for b in form["decode_context_length_buckets"] + [form["max_model_len"]]
                     if b >= need)
        if (need, bucket) != (8186, 8192):
            over.append((c["name"], need, bucket))
    assert over == []


def _reference_inputs(dst, flip):
    """Lay out the builder's two sources in ``dst``, one byte changed in ``flip``."""
    rows = dst / "mmlu_pro_500.json"
    split = next(Path(os.environ["HF_HOME"], "datasets").glob(f"*/*/*/{REVISION}"))
    split_copy = dst / "hf" / split.relative_to(Path(os.environ["HF_HOME"])) / "mmlu-pro-test.arrow"
    split_copy.parent.mkdir(parents=True)
    shutil.copyfile(SERVING / "mmlu_pro_500.json", rows)
    shutil.copyfile(split / "mmlu-pro-test.arrow", split_copy)
    target = rows if flip == "rows" else split_copy
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0x01
    target.write_bytes(bytes(data))
    return rows, dst / "hf", target


def test_prompt_files(tmp_path):
    """Item 2: inputs checked, outputs reproduced; every prompt renders to its file's length;
    no two prompts share their first 512 tokens; a source with one changed byte fails."""
    assert os.environ.get("HF_HOME"), "set HF_HOME to the offline cache"
    r = _run("spec_prompts.py", "--check")
    assert r.returncode == 0, r.stderr
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
    for name, length in (("spec_prompts.jsonl", 8048), ("spec_prompts_profile.jsonl", 7168)):
        prompts = [p["prompt"] for p in _jsonl(SERVING / name)]
        assert len(prompts) == 256
        ids = [tok.apply_chat_template([{"role": "user", "content": p}],
                                       add_generation_prompt=True, continue_final_message=False,
                                       return_dict=False, tokenize=True) for p in prompts]
        assert [i for i, x in enumerate(ids) if len(x) != length] == [], name
        heads = Counter(tuple(x[:512]) for x in ids)
        assert sum(n * (n - 1) // 2 for n in heads.values()) == 0, name
    for flip in ("rows", "split"):
        out = tmp_path / f"out-{flip}"
        out.mkdir()
        rows, hf_home, changed = _reference_inputs(tmp_path / f"in-{flip}", flip)
        r = _run("spec_prompts.py", "--rows", rows, "--hf-home", hf_home, "--out", out)
        assert r.returncode == 1, (flip, r.stdout, r.stderr)
        assert f"{changed}: sha256" in r.stderr, r.stderr
        assert list(out.iterdir()) == [], flip


def _write(path, result):
    path.write_text(json.dumps(result))
    return path


def _metrics(reading, path):
    r = _run("spec_metrics.py", reading, path)
    return r.returncode, json.loads(r.stdout)


def test_tpot_reducer(tmp_path):
    """Item 3: (last - first) / (tokens - 1) on multi-token chunks, the mean gap on one token
    per chunk, exit 2 on a request with fewer than 2 tokens."""
    # Chunk gaps in ms whose sums are 1,651, 1,905 and 1,524 ms over 127 token steps.
    gaps_ms = [[63.5] * 26, [76.2] * 25, [76.2] * 20]
    multi = {"itls": [[g / 1000.0 for g in req] for req in gaps_ms], "output_lens": [128] * 3,
             "median_tpot_ms": 13.4}
    code, rec = _metrics("tpot", _write(tmp_path / "multi.json", multi))
    reading = rec["files"][str(tmp_path / "multi.json")]
    assert code == 0
    assert reading["requests_ms"] == pytest.approx([13.0, 15.0, 12.0], rel=1e-12)
    assert reading["median_ms"] == pytest.approx(13.0, rel=1e-12)

    one = [30.0 + (i % 7) * 1.5 for i in range(127)]
    single = {"itls": [[g / 1000.0 for g in one]], "output_lens": [128]}
    code, rec = _metrics("tpot", _write(tmp_path / "single.json", single))
    assert code == 0
    assert rec["files"][str(tmp_path / "single.json")]["requests_ms"] == pytest.approx(
        [statistics.fmean(one)], rel=1e-12)

    short = {"itls": [[0.0635] * 26, []], "output_lens": [128, 1]}
    code, rec = _metrics("tpot", _write(tmp_path / "short.json", short))
    assert code == 2
    assert "request 1" in rec["invalid"][str(tmp_path / "short.json")]


def _counters(drafts, accepted, k=4):
    out = {"spec_decode_num_drafts": drafts, "spec_decode_draft_tokens": k * drafts,
           "spec_decode_accepted_tokens": accepted,
           "spec_decode_per_position_acceptance_rates": [0.5] * k}
    if drafts:
        out["spec_decode_acceptance_length"] = 1 + accepted / drafts
        out["spec_decode_acceptance_rate"] = 100.0 * accepted / (k * drafts)
    return out


def test_accepted_reader(tmp_path):
    """Item 4: 1 + accepted / drafts exactly; 0 drafts or no speculative fields exit 2."""
    for accepted in (1300, 1299):
        path = _write(tmp_path / f"run-{accepted}.json", _counters(1000, accepted))
        code, rec = _metrics("accepted", path)
        assert code == 0
        assert rec["files"][str(path)]["accepted_tokens_per_step"] == 1 + accepted / 1000
    zero = _counters(0, 0)
    zero["spec_decode_acceptance_length"] = 0.0
    code, rec = _metrics("accepted", _write(tmp_path / "zero.json", zero))
    assert code == 2 and "0 drafts" in rec["invalid"][str(tmp_path / "zero.json")]
    code, rec = _metrics("accepted", _write(tmp_path / "none.json", {"completed": 8}))
    assert code == 2 and "no speculative fields" in rec["invalid"][str(tmp_path / "none.json")]


def test_agreement_scorer(tmp_path):
    """Item 5: the reference's two GSM8K seeds, first 200 items, give 191, 198 and 128; a
    shuffled log fails the item check."""
    runs = os.environ.get("KIMI_K3_REFERENCE_RUNS")
    assert runs, "set KIMI_K3_REFERENCE_RUNS to the reference runs folder"
    seed0, seed1 = Path(runs, "seed-0", "gsm8k_cot"), Path(runs, "seed-1", "gsm8k_cot")
    r = _run("agreement.py", seed0, seed1)
    assert r.returncode == 0, r.stdout
    rec = json.loads(r.stdout)
    assert (rec["items"], rec["same_answer"], rec["same_answer_as_scored"],
            rec["same_reply_text"]) == (200, 191, 198, 128)

    log = next(seed1.rglob("samples_gsm8k_cot_*.jsonl"))
    lines = log.read_text(encoding="utf-8").splitlines(keepends=True)
    random.Random(0).shuffle(lines)
    shuffled = tmp_path / log.name
    shuffled.write_text("".join(lines), encoding="utf-8")
    r = _run("agreement.py", seed0, shuffled)
    assert r.returncode == 2, r.stdout
    assert "second log entry" in json.loads(r.stdout)["reason"]
