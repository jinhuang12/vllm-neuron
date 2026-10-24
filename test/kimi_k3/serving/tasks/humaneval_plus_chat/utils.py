"""Helpers for the HumanEval+ chat candidates. pass_at_k and build_predictions_instruct are the
lm-eval 0.4.13 humaneval functions, re-exported unchanged."""
import re

from lm_eval.tasks.humaneval.utils import build_predictions_instruct, pass_at_k  # noqa: F401

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)(?:\n[ \t]*```|\Z)", re.DOTALL)


def extract_first_code_block(text: str) -> str:
    """First fenced block, preferring a python/py/untagged one. No fence: the whole reply."""
    blocks = _FENCE.findall(text or "")
    for lang, body in blocks:
        if lang.lower() in ("", "python", "py", "python3"):
            return body
    return blocks[0][1] if blocks else (text or "")


def build_predictions_chat(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    # The prompt goes first, so its imports and helpers exist. A full redefinition of the
    # entry point in the reply replaces the stub; a bare body continues the stub.
    return [[doc["prompt"] + "\n" + extract_first_code_block(r) for r in resp]
            for resp, doc in zip(resps, docs)]
