# SPDX-License-Identifier: Apache-2.0
"""Build the speculative-decoding bench prompt files from the pinned MMLU-Pro rows.

Message i joins the question texts of rows i, i + 1 and onward of the 500 rows that
``mmlu_pro_500.json`` pins, in that file's order, and is cut so the checkpoint's
tokenizer renders it, as the server's chat endpoint does, to exactly the file's
length: 8,048 tokens in ``spec_prompts.jsonl`` (timed runs) and 7,168 tokens in
``spec_prompts_profile.jsonl`` (profiler captures), 256 messages each.

Every input's sha256 is checked before any other work; on a mismatch or a missing
input it exits 1 and writes no file. ``--check`` rebuilds both files in memory and
compares them with the landed copies and their recorded sha256.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import build_prompt_files

SERVING_DIR = Path(__file__).resolve().parent
DATASET = "TIGER-Lab/MMLU-Pro"
REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
ROWS_SHA256 = "199075dab36703c3fb0fb71020e12da85157f464fdf1921b0227de3c4a4e03a2"
SPLIT_SHA256 = {
    "mmlu-pro-test.arrow": "4c741944e3b53719b433be6e7916e4ab9e8ff33f70e9c365614b4e317c0476bb",
}
TOKENIZER_SHA256 = {
    "tiktoken.model": "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
    "tokenizer_config.json": "5d0803c94db9cd78763499e0956c95fd5a225c14a727e5a6cf5db3f96f010a6e",
    "tokenization_kimi.py": "f28ea66e2d862a2a5814970b2ce40c2f7d8296ff09aed90a7e7def689b906944",
    "encoding_k3.py": "49ff03305fdc4be26867972788d36150b67f8a9e852e62bb7959d87482223676",
}
# (file, rendered prompt tokens)
FILES = (("spec_prompts.jsonl", 8048), ("spec_prompts_profile.jsonl", 7168))
RECORDED_SHA256 = {
    "spec_prompts.jsonl": "457adbee90b90c1c4dcb9a3465fd70fbe604a9a0aff7ca922a0228ce9a573fb6",
    "spec_prompts_profile.jsonl": "aac98377243dff1a4ec935a4f440b1395cea27a723bd9464776b7064acbe842a",
}
MESSAGES = 256
SEPARATOR = "\n\n"
# The arguments vLLM's Kimi K3 chat renderer passes to the tokenizer for a request
# that sets no chat template arguments and no reasoning effort.
RENDER_KWARGS = {"add_generation_prompt": True, "continue_final_message": False,
                 "return_dict": False, "tokenize": True}


def default_checkpoint():
    """Return the checkpoint folder: ``KIMI_K3_CHECKPOINT`` or ``~/models/moonshotai/Kimi-K3``."""
    return os.environ.get("KIMI_K3_CHECKPOINT",
                          str(Path.home() / "models" / "moonshotai" / "Kimi-K3"))


def split_dir(hf_home):
    """Return the offline cache folder of the pinned dataset revision, or None."""
    dirs = [d for d in Path(hf_home, "datasets").glob(f"*/*/*/{REVISION}") if d.is_dir()]
    return dirs[0] if len(dirs) == 1 else None


def check_inputs(rows_path, hf_home, checkpoint):
    """Return the problems with the inputs' sha256; empty when every input matches."""
    expected = [(Path(rows_path), ROWS_SHA256)]
    folder = split_dir(hf_home) if hf_home else None
    if folder is None:
        return [f"{DATASET} at {REVISION}: not exactly one cache folder under {hf_home}"]
    expected += [(folder / name, digest) for name, digest in SPLIT_SHA256.items()]
    expected += [(Path(checkpoint, name), digest) for name, digest in TOKENIZER_SHA256.items()]
    problems = []
    for path, digest in expected:
        if not path.is_file():
            problems.append(f"{path}: missing")
        elif build_prompt_files.sha256(path) != digest:
            problems.append(f"{path}: sha256 {build_prompt_files.sha256(path)}, expected {digest}")
    return problems


def pinned_questions(rows_path, hf_home):
    """Return the question texts of the pinned rows, in the row file's order."""
    rows = build_prompt_files.load_split(hf_home, DATASET, REVISION, "test")
    by_task = {}
    for i, row in enumerate(rows):
        by_task.setdefault("mmlu_pro_" + row["category"].replace(" ", "_"), []).append(i)
    picks = json.loads(Path(rows_path).read_text())
    return [rows[by_task[task][pos]]["question"] for task, positions in picks.items()
            for pos in positions]


def load_tokenizer(checkpoint):
    """Load the checkpoint's own tokenizer class, as the server does."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)


def rendered_tokens(tokenizer, text):
    """Return the prompt token count of one user message as the chat endpoint renders it."""
    return len(tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                             **RENDER_KWARGS))


def build_message(tokenizer, questions, sizes, template, start, target):
    """Return (text, row count): rows from ``start`` joined and cut to ``target`` tokens."""
    # The rows' own token counts and the template's pick how many rows to join before the
    # first render.
    parts, estimate = [], template
    while True:
        if len(parts) == len(questions):
            raise ValueError(f"message {start}: all rows render below {target} tokens")
        j = (start + len(parts)) % len(questions)
        parts.append(questions[j])
        estimate += sizes[j] + 1
        if estimate < target:
            continue
        text = SEPARATOR.join(parts)
        if rendered_tokens(tokenizer, text) >= target:
            break
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if rendered_tokens(tokenizer, text[:mid]) >= target:
            hi = mid
        else:
            lo = mid + 1
    # A character can add two tokens and step over the target; take the nearest exact cut.
    # The cut may end in whitespace: a space before a number is its own token.
    for cut in list(range(lo, min(lo + 64, len(text) + 1))) + list(range(lo - 1, lo - 65, -1)):
        if rendered_tokens(tokenizer, text[:cut]) == target:
            starts = [len(SEPARATOR.join(parts[:k])) + (len(SEPARATOR) if k else 0)
                      for k in range(len(parts))]
            return text[:cut], sum(1 for x in starts if x < cut)
    raise ValueError(f"message {start}: no cut renders to exactly {target} tokens")


def build(questions, tokenizer):
    """Return {file name: file bytes} for both prompt files."""
    template = rendered_tokens(tokenizer, "")
    sizes = [rendered_tokens(tokenizer, q) - template for q in questions]
    out = {}
    for name, target in FILES:
        lines = []
        for i in range(MESSAGES):
            text, count = build_message(tokenizer, questions, sizes, template, i, target)
            lines.append(json.dumps({"id": i, "first_row": i, "row_count": count,
                                     "prompt_tokens": target, "prompt": text},
                                    ensure_ascii=False) + "\n")
        out[name] = "".join(lines).encode("utf-8")
    return out


def main(argv=None):
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", default=str(SERVING_DIR / "mmlu_pro_500.json"),
                    help="the pinned 500-row file (default: this folder's copy)")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME"),
                    help="offline Hugging Face home (default $HF_HOME)")
    ap.add_argument("--checkpoint", default=default_checkpoint(),
                    help="checkpoint folder holding the tokenizer files")
    ap.add_argument("--out", default=str(SERVING_DIR), help="landing folder (default: this one)")
    ap.add_argument("--check", action="store_true",
                    help="rebuild in memory and compare with the landed files")
    args = ap.parse_args(argv)
    problems = check_inputs(args.rows, args.hf_home, args.checkpoint)
    if not problems:
        try:
            files = build(pinned_questions(args.rows, args.hf_home),
                          load_tokenizer(args.checkpoint))
        except (build_prompt_files.MissingDataset, ValueError) as e:
            problems = [str(e)]
    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        return 1
    digests = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    if args.check:
        for name, data in files.items():
            landed = Path(args.out, name)
            if not landed.is_file() or landed.read_bytes() != data:
                problems.append(f"{name}: the landed file differs from the rebuild")
            if digests[name] != RECORDED_SHA256[name]:
                problems.append(f"{name}: rebuild sha256 {digests[name]}, "
                                f"recorded {RECORDED_SHA256[name]}")
    else:
        for name, data in files.items():
            Path(args.out, name).write_bytes(data)
    print(json.dumps({"sha256": digests, "result": "fail" if problems else "ok"}))
    for p in problems:
        print(p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
