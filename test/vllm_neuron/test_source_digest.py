# SPDX-License-Identifier: Apache-2.0
"""The kernel source digests read a file's code, not its comments or docstrings.

Two digests read kernel source. ``vllm_neuron.compile_cache_key`` folds the digest of
the kernel files a graph reaches (or of every kernel file) into the compile cache keys.
A kernel file's ``SOURCE_DIGEST`` int reaches its kernels as a trace-time argument, so
it is in the FX graph text and in the NKI kernel key. Both read a file through
``vllm_neuron.source_digest``: its syntax tree without docstrings, comments or layout.
A comment or docstring edit therefore moves no key and no ``SOURCE_DIGEST``; a code edit
moves both.

Pinned on copies of the real kernel files, for every file that defines a
``SOURCE_DIGEST``:

* (a) an added comment line, a trailing comment and a reworded module, function and
  attribute docstring move neither digest;
* (b) a renamed identifier, a changed number and a changed operator each move both;
* (c) a file that does not parse is read as its bytes, and a log line says so;
* (d) on a synthetic two-kernel tree, through the resolver the keys use: a comment edit
  moves no digest, and a code edit moves exactly the digests whose kernels reach the
  edited file.

``test_compile_cache_key_pergraph.py`` checks the same through the installed graph key
function on two real kernel graphs.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import logging
import shutil
import sys
import tokenize
from pathlib import Path

import pytest

import vllm_neuron
from vllm_neuron import compile_cache_key as ck

PACKAGE_ROOT = Path(vllm_neuron.__file__).resolve().parent

#: Every kernel file that hands its kernels a digest of its own source, and the names
#: it binds. A new one joins this table and so every test below.
SOURCE_DIGEST_FILES = {
    "functional/attention/dcp_merge.py": ("SOURCE_DIGEST",),
    "functional/attention/mla_decode.py": ("SOURCE_DIGEST",),
    "functional/attention/mla_dense_window.py": ("SOURCE_DIGEST",),
    "functional/attention/mla_sparse.py": ("SOURCE_DIGEST",),
    "functional/dsa/dcp_candidate_merge.py": ("SOURCE_DIGEST",),
    "functional/dsa/decode_batch.py": ("SOURCE_DIGEST",),
    "functional/dsa/decode_select.py": ("SOURCE_DIGEST",),
    "functional/dsa/decode_tail_update.py": ("ROWS_SOURCE_DIGEST",),
    "functional/dsa/decode_trow.py": ("SOURCE_DIGEST",),
    "functional/moe/expert_decode.py": ("_SOURCE_DIGEST",),
}

ADDED_COMMENT = "# An added comment line.\n"
TRAILING_COMMENT = "  # an added trailing comment"
REWORDED = " Reworded."


# ------------------------------------------------------------------ the edits


def _splice(text: str, edits: list[tuple[int, int, str, int]]) -> str:
    """Apply ``(line, column, replacement, removed)`` edits (1-based lines), last first."""
    lines = text.splitlines(keepends=True)
    for line, col, replacement, removed in sorted(edits, reverse=True):
        row = lines[line - 1]
        lines[line - 1] = row[:col] + replacement + row[col + removed:]
    return "".join(lines)


def _docstring_nodes(tree: ast.Module) -> list[ast.Constant]:
    """The module docstring, the first function docstring and the first attribute
    docstring (a lone string after an assignment), those the file has."""
    found = []
    if ast.get_docstring(tree, clean=False) is not None:
        found.append(tree.body[0].value)
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and ast.get_docstring(node, clean=False) is not None):
            found.append(node.body[0].value)
            break
    for before, node in zip(tree.body, tree.body[1:]):
        if (isinstance(before, (ast.Assign, ast.AnnAssign)) and isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            found.append(node.value)
            break
    assert found, "no docstring to reword"
    return found


def _first_in_functions(tree: ast.Module, wanted) -> ast.AST:
    """The first node inside a function body for which ``wanted(node)`` holds."""
    for function in ast.walk(tree):
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for statement in function.body:
                for node in ast.walk(statement):
                    if wanted(node):
                        return node
    raise AssertionError("no such node inside a function")


def reword_comments_and_docstrings(text: str) -> str:
    """Add a comment line and a trailing comment; reword two docstrings. No code moves."""
    tree = ast.parse(text)
    edits = [(1, 0, ADDED_COMMENT, 0)]
    for node in _docstring_nodes(tree):
        quotes = 3 if text.splitlines()[node.end_lineno - 1][
            node.end_col_offset - 3:node.end_col_offset] in ('"""', "'''") else 1
        edits.append((node.end_lineno, node.end_col_offset - quotes, REWORDED, 0))
    assign = next(node for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign)))
    end_line = text.splitlines()[assign.end_lineno - 1]
    edits.append((assign.end_lineno, len(end_line), TRAILING_COMMENT, 0))
    return _splice(text, edits)


def rename_identifier(text: str) -> str:
    """Rename the first module-level function, at its definition and every use."""
    tree = ast.parse(text)
    name = next(node.name for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    edits = [(token.start[0], token.start[1], f"{name}_renamed", len(name))
             for token in tokenize.generate_tokens(io.StringIO(text).readline)
             if token.type == tokenize.NAME and token.string == name]
    return _splice(text, edits)


def change_number(text: str) -> str:
    """Add one to the first integer literal inside a function body."""
    node = _first_in_functions(ast.parse(text), lambda n: isinstance(n, ast.Constant)
                               and type(n.value) is int)
    segment = ast.get_source_segment(text, node)
    return _splice(text, [(node.lineno, node.col_offset, str(node.value + 1), len(segment))])


def change_operator(text: str) -> str:
    """Turn the first ``a + b`` inside a function body into ``a - b``."""
    node = _first_in_functions(ast.parse(text), lambda n: isinstance(n, ast.BinOp)
                               and isinstance(n.op, ast.Add))
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if (token.type == tokenize.OP and token.string == "+"
                and (node.left.end_lineno, node.left.end_col_offset) <= token.start
                and token.end <= (node.right.lineno, node.right.col_offset)):
            return _splice(text, [(token.start[0], token.start[1], "-", 1)])
    raise AssertionError("no + token between the operands")


CODE_EDITS = {
    "identifier": rename_identifier,
    "number": change_number,
    "operator": change_operator,
}


# ------------------------------------------------------------ the digests


def _copy_tree(dst_parent: Path) -> Path:
    """Copy every file the digest covers into ``dst_parent/vllm_neuron``."""
    dst = dst_parent / "vllm_neuron"
    for rel in ck.digest_files(PACKAGE_ROOT):
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PACKAGE_ROOT / rel, target)
    return dst


def _literals(path: Path, rel: str) -> dict[str, int]:
    """The ``SOURCE_DIGEST`` values the kernel module at ``path`` binds when it runs.

    The copy runs as a sibling of the real module, so its imports (relative ones too)
    resolve to the real package and its digest reads the copy's own ``__file__``.
    """
    package = "vllm_neuron." + rel[: -len(".py")].replace("/", ".").rpartition(".")[0]
    name = f"{package}._digest_copy_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return {binding: getattr(module, binding) for binding in SOURCE_DIGEST_FILES[rel]}


def _digests(root: Path, rel: str) -> dict:
    """Every digest an edit to ``rel`` can move: the package digest, the file's digest
    in the per-graph form, and the ``SOURCE_DIGEST`` literals the file binds."""
    resolver = ck.KernelDigestResolver(root)
    return {
        "package": ck.kernel_digest(root),
        "file": resolver.digest_of({rel}),
        "literal": _literals(root / rel, rel),
    }


def _edited(root: Path, rel: str, edit) -> dict:
    path = root / rel
    original = path.read_text()
    edited = edit(original)
    assert edited != original
    ast.parse(edited)  # still Python
    path.write_text(edited)
    try:
        return _digests(root, rel)
    finally:
        path.write_text(original)


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    return _copy_tree(tmp_path_factory.mktemp("kernels"))


@pytest.fixture(scope="module")
def originals(tree):
    return {rel: _digests(tree, rel) for rel in SOURCE_DIGEST_FILES}


def test_the_table_names_every_source_digest_in_the_kernel_files():
    """A kernel file that binds a ``*SOURCE_DIGEST`` from its own source is in the table."""
    found = {}
    for path in (PACKAGE_ROOT / ck.KERNEL_SOURCE_DIR).rglob("*.py"):
        for node in ast.parse(path.read_bytes()).body:
            if isinstance(node, ast.Assign) and not isinstance(node.value, (ast.Name,
                                                                            ast.Attribute)):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)
                         and t.id.endswith("SOURCE_DIGEST")]
                if names:
                    found.setdefault(path.relative_to(PACKAGE_ROOT).as_posix(), []).extend(names)
    assert {rel: tuple(names) for rel, names in found.items()} == SOURCE_DIGEST_FILES


def test_every_source_digest_is_computed_by_the_shared_rule():
    """Each ``SOURCE_DIGEST`` is ``source_digest(...)`` of its files, so the literal and
    the keys read a file the same way."""
    raw = []
    for rel, names in SOURCE_DIGEST_FILES.items():
        for node in ast.parse((PACKAGE_ROOT / rel).read_bytes()).body:
            if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in names):
                call = node.value
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                        and call.func.id == "source_digest"):
                    raw.append(f"{rel}:{node.lineno} {ast.unparse(node)[:100]}")
    assert raw == []


@pytest.mark.parametrize("rel", SOURCE_DIGEST_FILES)
def test_a_comment_or_docstring_edit_moves_no_digest(tree, originals, rel):
    """(a) The package digest, the file's per-graph digest and its ``SOURCE_DIGEST``."""
    assert _edited(tree, rel, reword_comments_and_docstrings) == originals[rel]


@pytest.mark.parametrize("edit", CODE_EDITS)
@pytest.mark.parametrize("rel", SOURCE_DIGEST_FILES)
def test_a_code_token_edit_moves_every_digest(tree, originals, rel, edit):
    """(b) One renamed identifier, one changed number or one changed operator."""
    got = _edited(tree, rel, CODE_EDITS[edit])
    want = originals[rel]
    moved = {kind: got[kind] != want[kind] for kind in want}
    literals = {name: got["literal"][name] != want["literal"][name]
                for name in want["literal"]}
    assert moved == {"package": True, "file": True, "literal": True}
    assert all(literals.values()), literals


# --------------------------------------------------- a file that does not parse


def _write(root: Path, files: dict[str, str]) -> Path:
    for rel, source in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return root


def test_a_file_that_does_not_parse_is_digested_as_its_bytes(tmp_path, caplog):
    """(c) The digest still covers the file, byte for byte, and the log names it."""
    caplog.set_level(logging.INFO)
    root = _write(tmp_path / "vllm_neuron", {
        "functional/kernel.py": '"""A kernel."""\n\ndef kernel(x):\n    return x + 1\n',
        "functional/broken.py": "def kernel(:\n    return 1\n",
    })
    base = ck.kernel_digest(root)
    assert any("functional/broken.py" in record.getMessage()
               and "does not parse" in record.getMessage() for record in caplog.records), (
        [record.getMessage() for record in caplog.records])

    broken = root / "functional/broken.py"
    broken.write_text(broken.read_text() + "# a comment\n")
    assert ck.kernel_digest(root) != base


def test_the_shared_rule_reads_a_file_that_does_not_parse_as_its_bytes(tmp_path, caplog):
    """(c) ``source_digest`` takes the same fallback as the keys, and logs it."""
    from vllm_neuron.source_digest import source_digest

    caplog.set_level(logging.INFO)
    path = tmp_path / "broken.py"
    path.write_text("def kernel(:\n")
    first = source_digest(path)
    path.write_text("def kernel(:\n# a comment\n")
    second = source_digest(path)

    assert first != second
    assert 0 <= first < 16 ** 7 and 0 <= second < 16 ** 7
    assert any(str(path) in record.getMessage() and "does not parse" in record.getMessage()
               for record in caplog.records)


# ----------------------------------------------- (d) a synthetic two-kernel tree


TWO_KERNELS = {
    "functional/__init__.py": "",
    "functional/helper.py": (
        '"""A helper both kernels could call."""\n\n'
        "SCALE = 2\n"
        '"""The factor ``scaled`` applies."""\n\n\n'
        "def scaled(x):\n"
        '    """Scale ``x``."""\n'
        "    return x * SCALE + 1\n"
    ),
    "functional/kernel_a.py": (
        '"""Kernel A calls the helper."""\n\n'
        "from vllm_neuron.functional.helper import scaled\n\n\n"
        "def kernel_a(x):\n"
        '    """Kernel A."""\n'
        "    return scaled(x) + 3\n"
    ),
    "functional/kernel_b.py": (
        '"""Kernel B stands alone."""\n\n'
        "LIMIT = 7\n\n\n"
        "def kernel_b(x):\n"
        '    """Kernel B."""\n'
        "    return x + LIMIT\n"
    ),
}

LIBRARY_KEY = "0123456789abcdef0123456789abcdef"


def _keys(root: Path) -> dict[str, str]:
    """The folded key of a graph calling each kernel, and the whole-package digest."""
    resolver = ck.KernelDigestResolver(root)
    keys = {"package": resolver.package_digest}
    for kernel in ("kernel_a", "kernel_b"):
        resolution = resolver.resolve(
            [ck.Reference(ck.RefKind.MODULE, f"vllm_neuron.functional.{kernel}")])
        assert resolution.per_graph, resolution.reasons
        keys[kernel] = ck.fold_key(LIBRARY_KEY, resolver.digest(resolution))
    return keys


def test_a_two_kernel_tree_moves_only_the_keys_a_code_edit_reaches(tmp_path):
    """(d) A comment or docstring edit to the helper moves no key; a code edit moves
    kernel A's key and the package digest, not kernel B's."""
    root = _write(tmp_path / "vllm_neuron", TWO_KERNELS)
    assert ck.kernel_digest(root) == _keys(root)["package"]
    before = _keys(root)
    helper = root / "functional/helper.py"
    original = helper.read_text()

    helper.write_text(reword_comments_and_docstrings(original))
    reworded = _keys(root)
    for name, edit in CODE_EDITS.items():
        helper.write_text(edit(original))
        moved = {kind: key != before[kind] for kind, key in _keys(root).items()}
        assert moved == {"package": True, "kernel_a": True, "kernel_b": False}, name

    assert reworded == before
