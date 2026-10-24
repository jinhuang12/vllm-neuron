# SPDX-License-Identifier: Apache-2.0
"""Count the item-by-item agreement of two lm-eval GSM8K CoT sample logs.

Each log is a ``samples_gsm8k_cot_*.jsonl`` file, or a folder that holds exactly one.
The first ``--items`` entries of the ``--filter`` filter, in file order, are paired.
Item check: in both logs those entries are documents 0 to items - 1 in order, and each
pair has one ``doc_hash`` and one ``prompt_hash``. Three levels are counted: the same
reply text, the same extracted answer, and the same answer as scored, which compares
the answers after GSM8K's exact-match clean-up (drop commas, dollar signs, all text up
to ``#### `` and a final period, then lower case).

Standard library only. Prints one JSON record. Exit 0 when the item check passes,
2 when it fails or a log is missing.
"""

import argparse
import json
import re
import sys
from pathlib import Path

OK, INVALID = 0, 2
CLEAN_UP = (",", r"\$", r"(?s).*#### ", r"\.$")


def scored(answer):
    """Return an answer after the GSM8K exact-match clean-up."""
    if not isinstance(answer, str):
        return answer
    for rx in CLEAN_UP:
        answer = re.sub(rx, "", answer)
    return answer.lower()


def log_file(path):
    """Return the one sample log at ``path``; raise ValueError when there is not one."""
    p = Path(path)
    if p.is_file():
        return p
    found = sorted(p.rglob("samples_gsm8k_cot_*.jsonl")) if p.is_dir() else []
    if len(found) != 1:
        raise ValueError(f"{path}: {len(found)} gsm8k_cot sample logs, expected 1")
    return found[0]


def entries(path, flt, items):
    """Return the first ``items`` entries of filter ``flt`` in one log, in file order."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                x = json.loads(line)
                if x["filter"] == flt:
                    out.append(x)
                    if len(out) == items:
                        break
    return out


def answer(entry):
    """Return the entry's extracted answer."""
    a = entry["filtered_resps"][0]
    return a[0] if isinstance(a, list) else a


def score(a_path, b_path, flt, items):
    """Return the agreement record of two logs; raise ValueError when the item check fails."""
    a, b = entries(a_path, flt, items), entries(b_path, flt, items)
    for name, log in (("first", a), ("second", b)):
        if len(log) != items:
            raise ValueError(f"{name} log holds {len(log)} {flt} entries, expected {items}")
        for i, x in enumerate(log):
            if x["doc_id"] != i:
                raise ValueError(f"{name} log entry {i} is document {x['doc_id']}, expected {i}")
    for i, (x, y) in enumerate(zip(a, b)):
        for key in ("doc_hash", "prompt_hash"):
            if x[key] != y[key]:
                raise ValueError(f"document {i}: {key} differs")
    same = {"same_reply_text": [x["resps"][0][0] == y["resps"][0][0] for x, y in zip(a, b)],
            "same_answer": [answer(x) == answer(y) for x, y in zip(a, b)],
            "same_answer_as_scored": [scored(answer(x)) == scored(answer(y)) for x, y in zip(a, b)]}
    record = {"filter": flt, "items": items}
    record.update({level: sum(v) for level, v in same.items()})
    record["differing_documents"] = {level: [i for i, s in enumerate(v) if not s]
                                     for level, v in same.items()}
    return record


def main(argv=None):
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("first", help="a sample log, or a folder holding one")
    ap.add_argument("second", help="the log to pair with it")
    ap.add_argument("--filter", default="flexible-extract", help="the filter to pair")
    ap.add_argument("--items", type=int, default=200, help="entries to pair from the start")
    args = ap.parse_args(argv)
    try:
        record = score(log_file(args.first), log_file(args.second), args.filter, args.items)
    except (ValueError, KeyError, OSError) as e:
        print(json.dumps({"result": "invalid", "reason": str(e)}))
        return INVALID
    record["result"] = "ok"
    print(json.dumps(record))
    return OK


if __name__ == "__main__":
    sys.exit(main())
