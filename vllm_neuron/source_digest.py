# SPDX-License-Identifier: Apache-2.0
"""Digests of Python source that see its code and not its comments or docstrings.

Kernel source is digested in two places: the compile cache keys
(``vllm_neuron.compile_cache_key`` folds a digest of the kernel files a graph
reaches) and the ``SOURCE_DIGEST`` int some kernel files hand their kernels as a
trace-time argument (so it lands in the FX graph text and in the NKI kernel key).
Both read a file through :func:`code_digest`, the one owner of the rule:

* the file is parsed with :mod:`ast` and every statement that is a lone string
  literal is dropped: the docstrings of the module, its classes and functions, and
  the attribute docstrings written after an assignment. Such a statement runs no
  code; only a leading one sets a ``__doc__``, which no kernel reads;
* the rest is digested as ``ast.dump(tree, include_attributes=False)``, which keeps
  every code token (names, literal values, operators, decorators, defaults) and
  holds no comment, blank line, line number or layout.

So a comment, docstring or formatting edit moves no compile cache key and no
``SOURCE_DIGEST``, and so recompiles nothing; any code edit moves both. The dump
format is the interpreter's: another Python minor version may digest the same code
differently, which costs a recompile and never serves a stale kernel. A file that
does not parse is digested as its raw bytes, and a WARNING names it.
"""

from __future__ import annotations

import ast
import hashlib
import logging
from os import PathLike
from pathlib import Path

logger = logging.getLogger(__name__)

#: Leads the digested form of a file that parses: its docstring-free syntax tree.
CODE_FORM = b"python-ast\0"
#: Leads the digested form of a file that does not parse: its raw bytes.
RAW_FORM = b"raw-bytes\0"

#: Hex digits of a ``SOURCE_DIGEST``: a trace-time int argument small enough for
#: every kernel signature that takes one.
SOURCE_DIGEST_HEX = 7


class _DropDocstrings(ast.NodeTransformer):
    """Removes every statement that is a lone ``str`` literal. An f-string statement
    stays: it evaluates the expressions in it."""

    def visit_Expr(self, node: ast.Expr) -> ast.Expr | None:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)


def code_form(source: bytes, name: str) -> bytes:
    """What a digest of ``source`` reads: its code without docstrings, comments or
    layout, or, when it does not parse, its bytes. ``name`` is for the log line."""
    try:
        tree = ast.parse(source, filename=name)
    except (SyntaxError, ValueError) as exc:
        logger.warning(
            "source digest of %s reads its raw bytes: it does not parse as Python (%s), "
            "so a comment or docstring edit to it moves the digest",
            name,
            exc,
        )
        return RAW_FORM + source
    tree = _DropDocstrings().visit(tree)
    return CODE_FORM + ast.dump(tree, include_attributes=False).encode()


#: :func:`code_digest` by source bytes: the keys and a kernel file's ``SOURCE_DIGEST``
#: read the same file under different names, and parsing is most of a digest's cost.
_CODE_DIGESTS: dict[bytes, bytes] = {}


def code_digest(source: bytes, name: str) -> bytes:
    """The 32-byte sha256 of :func:`code_form`, computed once per process for each
    source text (so a file that does not parse is logged once, under its first name)."""
    digest = _CODE_DIGESTS.get(source)
    if digest is None:
        digest = _CODE_DIGESTS[source] = hashlib.sha256(code_form(source, name)).digest()
    return digest


def source_digest(*paths: str | PathLike[str]) -> int:
    """A ``SOURCE_DIGEST``: the code of ``paths``, in order, as a 7-hex int."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(code_digest(Path(path).read_bytes(), str(path)))
    return int(digest.hexdigest()[:SOURCE_DIGEST_HEX], 16)
