# SPDX-License-Identifier: Apache-2.0
"""Score coherence replies on the machine checks and write the on-topic sheet.

Each reply is one JSON line: ``id``, ``prompt``, ``reasoning``, ``content``,
``finish_reason`` and ``completion_tokens``. The checked text is the reasoning text,
then the answer text.

- Check 1, repetition: no sequence of 32 whitespace-separated words occurs twice.
- Check 2, ending: ``finish_reason`` is ``stop``, or ``length`` at 512 completion tokens.
- Check 3, language: langdetect with seed 0 gives the prompt and the checked text the
  same language code.

``--sheet`` writes the 51-row on-topic sheet: the 50 replies and one reply to a
different prompt at a seeded position, whose key goes to ``--key``. Exit 0 when every
reply passes every selected check, 1 otherwise.
"""

import argparse
import json
import random
import secrets
import sys
from pathlib import Path

NGRAM = 32
MAX_TOKENS = 512
SHEET_ROWS = 50


def checked_text(reply):
    """Return the reasoning text, then the answer text."""
    parts = [reply.get("reasoning") or "", reply.get("content") or ""]
    return "\n".join(p for p in parts if p)


def check_repetition(reply):
    """Check 1: no 32-word sequence occurs twice."""
    words = checked_text(reply).split()
    seen = set()
    for i in range(len(words) - NGRAM + 1):
        gram = tuple(words[i:i + NGRAM])
        if gram in seen:
            return False
        seen.add(gram)
    return True


def check_ending(reply):
    """Check 2: the reply stopped, or ran to the 512-token cap."""
    reason = reply.get("finish_reason")
    return reason == "stop" or (reason == "length"
                                and reply.get("completion_tokens") == MAX_TOKENS)


def check_language(reply):
    """Check 3: the prompt and the checked text get the same language code."""
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0
    try:
        return detect(reply["prompt"]) == detect(checked_text(reply))
    except LangDetectException:
        return False


CHECKS = {1: check_repetition, 2: check_ending, 3: check_language}


def score(replies, checks):
    """Return the report of the selected checks over the replies."""
    failed = {c: [r["id"] for r in replies if not CHECKS[c](r)] for c in checks}
    return {"replies": len(replies), "checks": list(checks),
            "passed": {str(c): len(replies) - len(failed[c]) for c in checks},
            "failed": {str(c): failed[c] for c in checks}}


def _fence(text):
    run, longest = 0, 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    return "`" * max(3, longest + 1)


def write_sheet(replies, sheet_path, key_path, seed=None):
    """Write the 51-row sheet and its key; return the key."""
    if len(replies) != SHEET_ROWS:
        raise ValueError(f"the sheet takes {SHEET_ROWS} replies, got {len(replies)}")
    if seed is None:
        seed = secrets.randbits(32)
    rng = random.Random(seed)
    position = rng.randrange(SHEET_ROWS + 1)
    prompt_of = rng.randrange(SHEET_ROWS)
    reply_of = rng.choice([i for i in range(SHEET_ROWS) if i != prompt_of])
    rows = [(r["prompt"], checked_text(r)) for r in replies]
    rows.insert(position, (replies[prompt_of]["prompt"], checked_text(replies[reply_of])))
    lines = ["# On-topic sheet", "",
             f"Mark each of the {len(rows)} rows on-topic or off-topic.", ""]
    for n, (prompt, text) in enumerate(rows, start=1):
        fp, ft = _fence(prompt), _fence(text)
        lines += [f"## Row {n}", "", "Prompt:", "", fp, prompt, fp, "", "Reply:", "",
                  ft, text, ft, "", "On-topic: yes / no", ""]
    Path(sheet_path).write_text("\n".join(lines), encoding="utf-8")
    key = {"seed": seed, "planted_row": position + 1, "prompt_of": replies[prompt_of]["id"],
           "reply_of": replies[reply_of]["id"]}
    Path(key_path).write_text(json.dumps(key) + "\n", encoding="utf-8")
    return key


def main(argv=None):
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replies", required=True, help="replies, one JSON object per line")
    ap.add_argument("--checks", default="1,2,3", help="comma-separated check numbers")
    ap.add_argument("--sheet", help="write the on-topic sheet here")
    ap.add_argument("--key", help="write the sheet's key here (required with --sheet)")
    ap.add_argument("--sheet-seed", type=int, default=None, help="seed of the planted row")
    args = ap.parse_args(argv)
    checks = [int(c) for c in args.checks.split(",")]
    if any(c not in CHECKS for c in checks):
        ap.error(f"--checks takes numbers from {sorted(CHECKS)}")
    replies = [json.loads(line) for line in Path(args.replies).read_text(
        encoding="utf-8").splitlines() if line.strip()]
    report = score(replies, checks)
    if args.sheet:
        if not args.key or Path(args.key).resolve() == Path(args.sheet).resolve():
            ap.error("--sheet needs a separate --key file")
        write_sheet(replies, args.sheet, args.key, args.sheet_seed)
        report["sheet_rows"] = SHEET_ROWS + 1
    print(json.dumps(report))
    return 0 if all(not ids for ids in report["failed"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
