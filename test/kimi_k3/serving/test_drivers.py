# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 serving drivers: prompt and reference files, scorer, lm-eval launch, bench shapes,
steady TPOT reducer and per-item sample check.

Set ``HF_HOME`` to the offline cache that holds the datasets at their registered commits.
"""

import hashlib
import json
import os
import random
import shlex
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

import build_prompt_files

SERVING = Path(__file__).resolve().parent
SEED = 20261023
NAME = "us.moonshotai.kimi-k3"
TASK_DIR = SERVING / "tasks" / "humaneval_plus_chat"

# (landed path, path in the reference folder, sha256)
REFERENCE = (
    ("mmlu_pro_500.json", "samples/mmlu_pro_500.json",
     "199075dab36703c3fb0fb71020e12da85157f464fdf1921b0227de3c4a4e03a2"),
    ("tasks/humaneval_plus_chat/humaneval_plus_chat.yaml", "tasks/humaneval_plus_chat.yaml",
     "4265e34808fac8515ef6206bc7ce18e8192bdd1fc222f7be97aa4de10a02a218"),
    ("tasks/humaneval_plus_chat/utils.py", "tasks/utils.py",
     "d230583fcd8f8838bb5d9e5076859e6804920c0b921e58f5e71ab844be518b1d"),
)

# Draw order: (source, dataset, commit, split, count)
DRAW = (
    ("gsm8k", "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "test", 20),
    ("mmlu_pro", "TIGER-Lab/MMLU-Pro", "b189ec765aa7ed75c8acfea42df31fdae71f97be", "test", 15),
    ("humanevalplus", "evalplus/humanevalplus", "d32357cf319e50e9c8d8dab5ea876c72b0fd321b",
     "test", 10),
    ("ifeval", "google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84", "train", 5),
)


def _run(script, *args, **kwargs):
    cmd = (["bash"] if script.endswith(".sh") else [sys.executable]) + [str(SERVING / script)]
    return subprocess.run(cmd + [str(a) for a in args], capture_output=True, text=True,
                          **kwargs)


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _hf_home():
    hf_home = os.environ.get("HF_HOME")
    assert hf_home, "set HF_HOME to the offline cache"
    return hf_home


def _split(hf_home, commit, split):
    rows = build_prompt_files.load_split(hf_home, "", commit, split)
    info = next(Path(hf_home, "datasets").glob(f"*/*/*/{commit}/dataset_info.json"))
    assert len(rows) == json.loads(info.read_text())["splits"][split]["num_examples"]
    return rows


def _prompt_text(source, row):
    if source == "gsm8k":
        return row["question"]
    if source == "mmlu_pro":
        return row["question"] + "".join(
            f"\n{'ABCDEFGHIJ'[i]}. {opt}" for i, opt in enumerate(row["options"]))
    return row["prompt"]


def _reference_copy(dst, flip=None):
    """Lay the landed reference copies out as a reference folder, one byte changed in ``flip``."""
    for landed, rel, _ in REFERENCE:
        p = Path(dst, rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = bytearray((SERVING / landed).read_bytes())
        if rel == flip:
            data[len(data) // 2] ^= 0x01
        p.write_bytes(bytes(data))


def test_prompt_file_draw():
    """Item 1: 50 prompts, 20/15/10/5 in order, rows of one seeded draw over the splits."""
    hf_home = _hf_home()
    prompts = _jsonl(SERVING / "prompts_rung1.jsonl")
    rng = random.Random(SEED)
    expected = []
    for source, _, commit, split, k in DRAW:
        rows = _split(hf_home, commit, split)
        for row_id in rng.sample(range(len(rows)), k):
            expected.append((source, commit, split, row_id, _prompt_text(source, rows[row_id])))
    assert len(prompts) == 50
    assert [(p["source"], p["revision"], p["split"], p["row"], p["prompt"])
            for p in prompts] == expected


def test_reference_files_fail_closed(tmp_path):
    """Item 2: --check passes on the landed copies; one changed byte makes --reference write nothing."""
    r = _run("build_prompt_files.py", "--check")
    assert r.returncode == 0, r.stderr
    for _, rel, _ in REFERENCE:
        ref, out = tmp_path / f"ref-{Path(rel).name}", tmp_path / f"out-{Path(rel).name}"
        out.mkdir()
        _reference_copy(ref, flip=rel)
        r = _run("build_prompt_files.py", "--reference", ref, "--out", out, "--hf-home",
                 _hf_home())
        assert r.returncode == 1, (rel, r.stdout, r.stderr)
        assert f"{rel}: sha256" in r.stderr, r.stderr
        assert list(out.rglob("*")) == [], rel


def test_scorer_known_bad():
    """Item 3: each known-bad reply fails its own check, with a nonzero exit."""
    r = _run("score_coherence.py", "--replies", SERVING / "known_bad.jsonl", "--checks", "1,2")
    assert r.returncode != 0
    assert json.loads(r.stdout)["failed"] == {"1": ["repeat-32"], "2": ["length-300"]}


def test_scorer_known_good():
    """Item 4: the 20 GSM8K gold answers and 10 HumanEval+ canonical solutions pass checks 1 and 2."""
    hf_home = _hf_home()
    rng = random.Random(SEED)
    expected = []
    for source, _, commit, split, k in DRAW[:3]:
        rows = _split(hf_home, commit, split)
        for row_id in rng.sample(range(len(rows)), k):
            row = rows[row_id]
            if source == "gsm8k":
                expected.append((f"gsm8k-{row_id}", row["question"], row["answer"]))
            elif source == "humanevalplus":
                expected.append((f"humanevalplus-{row_id}", row["prompt"],
                                 row["prompt"] + row["canonical_solution"]))
    good = _jsonl(SERVING / "known_good.jsonl")
    assert [(g["id"], g["prompt"], g["content"]) for g in good] == expected
    r = _run("score_coherence.py", "--replies", SERVING / "known_good.jsonl", "--checks", "1,2")
    assert r.returncode == 0, r.stdout
    assert json.loads(r.stdout)["passed"] == {"1": 30, "2": 30}


def test_sheet_planted_row(tmp_path):
    """Item 5: 51 rows, one planted reply to another prompt at the seeded position, key apart."""
    replies = [{"id": f"r{i:02d}", "prompt": f"Prompt {i:02d}: name a colour.", "reasoning": "",
                "content": f"Reply {i:02d}: blue.", "finish_reason": "stop",
                "completion_tokens": 5} for i in range(50)]
    _write_jsonl(tmp_path / "replies.jsonl", replies)
    sheet, key_file = tmp_path / "sheet.md", tmp_path / "key.json"
    r = _run("score_coherence.py", "--replies", tmp_path / "replies.jsonl", "--checks", "1,2",
             "--sheet", sheet, "--key", key_file, "--sheet-seed", 7)
    assert r.returncode == 0, r.stderr
    rng = random.Random(7)
    position = rng.randrange(51)
    a = rng.randrange(50)
    b = rng.choice([i for i in range(50) if i != a])
    key = json.loads(key_file.read_text())
    assert key == {"seed": 7, "planted_row": position + 1, "prompt_of": f"r{a:02d}",
                   "reply_of": f"r{b:02d}"}
    text = sheet.read_text()
    blocks = text.split("\n## Row ")[1:]
    assert [blk.split("\n", 1)[0] for blk in blocks] == [str(n) for n in range(1, 52)]
    genuine = iter(range(50))
    for n, blk in enumerate(blocks, start=1):
        p, q = (a, b) if n == position + 1 else (next(genuine),) * 2
        assert f"\nPrompt {p:02d}: name a colour.\n" in blk and f"\nReply {q:02d}: blue.\n" in blk
    assert "planted" not in text.lower() and "seed" not in text.lower()


def test_lm_eval_launch_print_only(tmp_path):
    """Item 6: the four registered commands, their per-item checks, the serve line; fail closed."""
    args = ["--print-only", "--client-venv", "/venvs/client", "--base-url",
            "http://127.0.0.1:8000", "--out", "/runs/out", "--reference-runs", "/ref/seed-0",
            "--hf-home", "/hf"]
    r = _run("run_lm_eval.sh", *args)
    assert r.returncode == 0, r.stderr
    rows = [line.split("\t", 1) for line in r.stdout.splitlines()]
    tok = {label: shlex.split(cmd) for label, cmd in rows}
    assert [label for label, _ in rows] == [
        "serve", "serve-absent", "gsm8k_cot-200", "gsm8k_cot-200:check", "gsm8k_cot",
        "gsm8k_cot:check", "mmlu_pro", "mmlu_pro:check", "humaneval_plus_chat",
        "humaneval_plus_chat:check"]

    py = "/venvs/client/bin/python"
    env = ["env", "HF_HOME=/hf", "HF_DATASETS_OFFLINE=1", "HF_HUB_OFFLINE=1",
           "OPENAI_API_KEY=EMPTY"]
    model_args = (f"model={NAME},base_url=http://127.0.0.1:8000/v1/chat/completions,"
                  "tokenized_requests=False,tokenizer_backend=None,num_concurrent=8,"
                  "timeout=7200,max_length=16384,max_retries=10")
    gen = "max_gen_toks=2048,temperature=0,do_sample=false,reasoning_effort=none"

    def lm_eval(task, out, limit=(), extra=(), env_extra=()):
        return [*env, *env_extra, py, "-m", "lm_eval", "--model", "local-chat-completions",
                "--tasks", task, "--model_args", model_args, "--log_samples", "--output_path",
                f"/runs/out/{out}", *limit, "--apply_chat_template", "--fewshot_as_multiturn",
                "--gen_kwargs", gen, "--seed", "0,1234,1234,1234", *extra]

    def check(out, ref, limit=()):
        return [py, str(SERVING / "check_samples.py"), "--candidate", f"/runs/out/{out}",
                "--reference", f"/ref/seed-0/{ref}/out", *limit]

    limit = ["--limit", "200"]
    assert tok["gsm8k_cot-200"] == lm_eval("gsm8k_cot", "gsm8k_cot-200", limit)
    assert tok["gsm8k_cot-200:check"] == check("gsm8k_cot-200", "gsm8k_cot", limit)
    assert tok["gsm8k_cot"] == lm_eval("gsm8k_cot", "gsm8k_cot")
    assert tok["gsm8k_cot:check"] == check("gsm8k_cot", "gsm8k_cot")
    assert tok["mmlu_pro"] == lm_eval("mmlu_pro", "mmlu_pro",
                                      extra=["--samples", str(SERVING / "mmlu_pro_500.json")])
    assert tok["mmlu_pro:check"] == check("mmlu_pro", "mmlu_pro")
    assert tok["humaneval_plus_chat"] == lm_eval(
        "humaneval_plus_chat", "humaneval_plus_chat",
        extra=["--include_path", str(TASK_DIR), "--confirm_run_unsafe_code"],
        env_extra=["HF_ALLOW_CODE_EVAL=1"])
    assert tok["humaneval_plus_chat:check"] == check("humaneval_plus_chat",
                                                     "humaneval_plus_chat")
    assert hashlib.sha256((SERVING / "mmlu_pro_500.json").read_bytes()).hexdigest() == \
        REFERENCE[0][2]
    assert sorted(p.name for p in TASK_DIR.iterdir() if p.name != "__pycache__") == [
        "humaneval_plus_chat.yaml", "utils.py"]
    assert tok["serve"] == ["--reasoning-parser", "kimi_k3", "--max-model-len", "16384",
                            "--served-model-name", NAME, "--no-enable-prefix-caching"]
    assert tok["serve-absent"] == ["--chat-template", "--default-chat-template-kwargs",
                                   "--generation-config"]

    for _, rel, _ in REFERENCE:
        files = tmp_path / f"files-{Path(rel).name}"
        for landed, rel2, _ in REFERENCE:
            p = files / landed
            p.parent.mkdir(parents=True, exist_ok=True)
            data = bytearray((SERVING / landed).read_bytes())
            if rel2 == rel:
                data[len(data) // 2] ^= 0x01
            p.write_bytes(bytes(data))
        r = _run("run_lm_eval.sh", *args, "--files", files)
        assert (r.returncode, r.stdout) == (1, ""), rel

    # A per-item check that exits 2 makes the run invalid.
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    shim = venv / "bin" / "python"
    shim.write_text(f'#!/bin/sh\n[ "$1" = "-m" ] && [ "$2" = "lm_eval" ] && exit 0\n'
                    f'exec {shlex.quote(sys.executable)} "$@"\n')
    shim.chmod(0o755)
    ref_log = tmp_path / "ref" / "gsm8k_cot" / "out" / NAME
    ref_log.mkdir(parents=True)
    _write_jsonl(ref_log / "samples_gsm8k_cot_2026-10-23T00-00-00.000000.jsonl",
                 [{"doc_id": 0, "filter": "flexible-extract", "arguments": {}, "prompt_hash": "h",
                   "exact_match": 1.0}])
    r = _run("run_lm_eval.sh", "--rung", "2", "--client-venv", venv, "--out", tmp_path / "out",
             "--reference-runs", tmp_path / "ref", "--hf-home", tmp_path)
    assert r.returncode == 2, r.stderr
    assert "invalid" in r.stderr


def test_bench_shapes():
    """Item 7: the 9 shapes, S - 128 in, 128 out, N = c or 8, 6-hour timeout, 45-minute deadline."""
    doc = json.loads((SERVING / "bench_shapes.json").read_text())
    pairs = [(2048, 1), (2048, 8), (2048, 32), (8192, 1), (8192, 8), (8192, 32), (32768, 1),
             (32768, 8), (131072, 1)]
    assert [(s["S"], s["concurrency"]) for s in doc["shapes"]] == pairs
    for s in doc["shapes"]:
        assert s["input_len"] == s["S"] - 128 and s["output_len"] == 128
        assert s["num_prompts"] == (8 if s["concurrency"] == 1 else s["concurrency"])
    assert doc["shapes"][4]["input_len"] == 8064
    assert (doc["client_timeout_s"], doc["run_deadline_s"]) == (6 * 3600, 45 * 60)


def _cell(dst, gap_ms, b_gap_ms=None, first_gap_ms=None, spike=None, failed_run=None):
    """Write 6 timed result files of 8 requests: processes A and B, runs 1 to 3."""
    dst.mkdir()
    for process in "AB":
        for run in (1, 2, 3):
            gap = gap_ms if process == "A" or b_gap_ms is None else b_gap_ms
            itls = [[gap / 1000.0] * 127 for _ in range(8)]
            if first_gap_ms is not None:
                for req in itls:
                    req[0] = first_gap_ms / 1000.0
            if spike and (process, run) == spike[0]:
                req, pos, ms = spike[1]
                itls[req][pos] = ms / 1000.0
            result = {"num_prompts": 8, "completed": 8,
                      "failed": 1 if failed_run == (process, run) else 0,
                      "input_lens": [8064] * 8, "output_lens": [128] * 8, "ttfts": [1.0] * 8,
                      "itls": itls, "median_tpot_ms": gap, "tip": "t", "process": process,
                      "run": str(run)}
            (dst / f"{process}-{run}.json").write_text(json.dumps(result))
    return dst


def _reduce(path):
    r = _run("steady_tpot.py", path)
    return r.returncode, json.loads(r.stdout)


def test_reducer_known_bad(tmp_path):
    """Item 8: a 400 ms gap at position 3 stays in its request's mean, guard 1, exit 3."""
    code, rec = _reduce(_cell(tmp_path / "c", 37.6, spike=(("A", 1), (2, 3, 400.0))))
    assert code == 3
    assert rec["runs"]["A/1"]["requests_ms"][2] == pytest.approx((400.0 + 125 * 37.6) / 126)
    assert rec["guard_counts"]["A/1"] == [0, 0, 1, 0, 0, 0, 0, 0] and rec["guard_total"] == 1


def test_reducer_known_good(tmp_path):
    """Item 9: flat 37.6 ms passes, also with a 4 s first gap; 42.02 passes, 42.22 fails; invalid; unstable."""
    code, rec = _reduce(_cell(tmp_path / "flat", 37.6))
    assert (code, rec["guard_total"]) == (0, 0)
    assert rec["value_of_record_ms"] == pytest.approx(37.6)
    code, rec = _reduce(_cell(tmp_path / "first", 37.6, first_gap_ms=4000.0))
    assert code == 0 and rec["value_of_record_ms"] == pytest.approx(37.6)
    assert _reduce(_cell(tmp_path / "slow", 42.22))[0] == 1
    code, rec = _reduce(_cell(tmp_path / "under", 42.02))
    assert (code, rec["bar_ms"]) == (0, 42.12)
    assert _reduce(_cell(tmp_path / "failed", 37.6, failed_run=("B", 2)))[0] == 2
    code, rec = _reduce(_cell(tmp_path / "spread", 30.0, b_gap_ms=34.0))
    assert code == 3 and rec["spread"] == pytest.approx(4.0 / 32.0)


def _samples(root, n_docs, flex):
    log = Path(root) / NAME
    log.mkdir(parents=True)
    rows = []
    for filt in ("strict-match", "flexible-extract"):
        for d in range(n_docs):
            rows.append({"doc_id": d, "filter": filt,
                         "arguments": {"gen_args_0": {"arg_0": [f"question {d}", {"seed": 1234}]}},
                         "prompt_hash": f"hash-{d}",
                         "exact_match": flex(d) if filt == "flexible-extract" else 0.0})
    return log / "samples_gsm8k_cot_2026-10-23T00-00-00.000000.jsonl", rows


def test_per_item_check(tmp_path):
    """Item 10: 200 of 250 reference docs pair, mean of docs 0-199; changed or missing entries exit 2."""
    def flex(d):
        return 1.0 if d >= 200 or d % 4 else 0.0

    ref_file, ref_rows = _samples(tmp_path / "ref", 250, flex)
    _write_jsonl(ref_file, ref_rows)
    cand_rows = [dict(e) for e in ref_rows if e["doc_id"] < 200]

    def check(rows, name):
        cand_file, _ = _samples(tmp_path / name, 0, flex)
        _write_jsonl(cand_file, rows)
        r = _run("check_samples.py", "--candidate", tmp_path / name, "--reference",
                 tmp_path / "ref", "--limit", 200)
        return r.returncode, json.loads(r.stdout)

    code, rep = check(cand_rows, "equal")
    assert (code, rep["pairs"]) == (0, 400)
    assert rep["reference_mean_flexible_extract"] == pytest.approx(
        100.0 * statistics.fmean(flex(d) for d in range(200)))

    changed = [dict(e) for e in cand_rows]
    changed[257]["arguments"] = {"gen_args_0": {"arg_0": ["another question", {"seed": 1234}]}}
    code, rep = check(changed, "arguments")
    assert code == 2 and rep["first_difference"]["doc_id"] == changed[257]["doc_id"]

    changed = [dict(e) for e in cand_rows]
    changed[123]["prompt_hash"] = "another-hash"
    code, rep = check(changed, "prompt-hash")
    assert code == 2 and rep["first_difference"]["doc_id"] == 123

    code, rep = check(cand_rows[:-1], "missing")
    assert code == 2 and rep["first_difference"]["reason"] == "missing in candidate"
