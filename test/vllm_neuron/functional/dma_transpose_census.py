# SPDX-License-Identifier: Apache-2.0
"""Read every DMA transpose a kernel module traces, through the helpers that issue them.

For each ``nisa.dma_transpose`` and ``nl.load_transpose2d`` in a module, and for each call of
a helper that transposes into one of its parameters, the reading is: the source rows each
DMA moves, the source width, and every byte offset the destination can start at, per
element width the destination can have. Values come from the module's own size arithmetic
evaluated at the geometry the caller names; a value that arithmetic cannot produce is
``None`` and never a default, so a census that reads nothing is a failed census.
"""

from __future__ import annotations

import ast
import copy
import itertools
from dataclasses import dataclass

#: The most loop-value combinations one bound is read at before it is called unreadable.
_COMBINATIONS = 16384

#: The runtime's rule for a transpose destination: it starts on a 32-byte line.
LINE = 32

#: Element widths per NKI dtype name; a tensor's own ``.dtype`` takes the caller's widths.
_ITEMSIZES = {"bfloat16": 2, "float16": 2, "float32": 4, "int32": 4, "uint32": 4}

_RANGES = ("range", "affine_range", "sequential_range", "static_range")


@dataclass
class Site:
    """One transpose's reading at one geometry."""

    line: int
    via: str
    source: str
    rows: tuple[int, ...] | None
    width: tuple[int, ...] | None
    offsets: dict[int, tuple[int, ...] | None]

    @property
    def misaligned(self) -> bool | None:
        """``None`` when an offset is unreadable, else whether one starts off the line."""
        if any(o is None for o in self.offsets.values()):
            return None
        return any(b % LINE for o in self.offsets.values() for b in o)

    def host_shaped(self, step: int) -> bool:
        """Whether a DMA here misses the device's own shape: rows not ``step``, width not a tile."""
        if self.rows is None or self.width is None:
            return True
        return any(r != step for r in self.rows) or any(w <= 0 or w % 128 for w in self.width)


def _value(node: ast.expr | None, names: dict):
    """One expression's trace-time value, or ``None`` when it is not readable."""
    if node is None:
        return None
    try:
        return eval(compile(ast.Expression(body=node), "<size>", "eval"), dict(names))
    except Exception:
        return None


def _eval(node: ast.expr | None, names: dict) -> int | None:
    """One size expression's trace-time integer, or ``None`` when it is not one."""
    value = _value(node, names)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _live(body: list, names: dict):
    """The statements a trace visits: an ``if`` whose test reads as a constant keeps one branch."""
    for node in body:
        if isinstance(node, ast.If):
            test = _value(node.test, names)
            if test is True or test is False:
                yield from _live(node.body if test else node.orelse, names)
                continue
            yield node
            yield from _live(node.body, names)
            yield from _live(node.orelse, names)
        else:
            yield node
            for field in ("body", "orelse", "finalbody"):
                inner = getattr(node, field, None)
                if isinstance(inner, list):
                    yield from _live(inner, names)


def _calls(fn: ast.FunctionDef, names: dict):
    """Every call the trace visits in ``fn``, dead branches pruned."""
    for stmt in _live(fn.body, names):
        for node in ast.walk(stmt) if not isinstance(stmt, (ast.If, ast.For, ast.While, ast.With)) else [stmt]:
            if isinstance(node, ast.Call):
                yield node
        if isinstance(stmt, (ast.For, ast.While, ast.With, ast.If)):
            for field in ("iter", "test", "items"):
                part = getattr(stmt, field, None)
                for node in ast.walk(part) if isinstance(part, ast.AST) else []:
                    if isinstance(node, ast.Call):
                        yield node


def _names(fn: ast.FunctionDef, base: dict) -> dict:
    """The caller's names, then the body's own simple assignments in source order."""
    names = dict(base)
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id in names:
            continue
        value = _value(node.value, names)
        if value is not None:
            names[target.id] = value
    return names


def _range_values(call: ast.expr, names: dict) -> tuple[int, ...] | None:
    """The values a ``range``-family iterator takes, or ``None`` when its bounds are unreadable."""
    if not isinstance(call, ast.Call):
        return None
    name = getattr(call.func, "id", "") or getattr(call.func, "attr", "")
    if name not in _RANGES:
        return None
    args = [_eval(a, names) for a in call.args]
    if not args or any(a is None for a in args):
        return None
    start, stop, step = (0, args[0], 1) if len(args) == 1 else (args[0], args[1], args[2] if len(args) > 2 else 1)
    return tuple(range(start, stop, step))


def _loops(fn: ast.FunctionDef, names: dict) -> dict[str, tuple[int, ...] | None]:
    """Per loop target, the values it takes; an unreadable bound reads ``None``."""
    found: dict[str, tuple[int, ...] | None] = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            found[node.target.id] = _range_values(node.iter, names)
    return found


def _itemsizes(dtype: ast.expr | None, source_itemsizes: tuple[int, ...]) -> tuple[int, ...] | None:
    """The element widths a tile of ``dtype`` can have."""
    if dtype is None:
        return None
    if isinstance(dtype, ast.Attribute) and dtype.attr == "dtype":
        return source_itemsizes
    size = _ITEMSIZES.get(getattr(dtype, "attr", getattr(dtype, "id", "")))
    return (size,) if size else None


#: Element widths per SBUF tile maker of the kernels read here.
_MAKERS = {"_sbuf": (4,), "_sbuf_u32": (4,), "_sbuf_i32": (4,)}


def _helpers(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    """The module's functions by name: a tile made through one is read through its body."""
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


class _Inline(ast.NodeTransformer):
    """Put a caller's argument expressions in place of a helper's parameter names."""

    def __init__(self, binding: dict[str, ast.expr]):
        self.binding = binding

    def visit_Name(self, node: ast.Name) -> ast.expr:
        return copy.deepcopy(self.binding.get(node.id, node))


def _returned(fn: ast.FunctionDef, call: ast.Call) -> ast.expr | None:
    """What a single-return helper returns for ``call``, with the call's arguments in place."""
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
    params = [a.arg for a in fn.args.args]
    if len(returns) != 1 or len(call.args) > len(params):
        return None
    binding: dict[str, ast.expr] = dict(zip(params, call.args))
    binding.update({kw.arg: kw.value for kw in call.keywords if kw.arg})
    return _Inline(binding).visit(copy.deepcopy(returns[0].value))


def _from_base(sub: ast.Subscript) -> bool:
    """Whether a subscript's every slice starts at its tile's own base."""
    parts = sub.slice.elts if isinstance(sub.slice, ast.Tuple) else [sub.slice]
    return all(isinstance(p, ast.Slice) and (p.lower is None or _eval(p.lower, {}) == 0) for p in parts)


def _shape(expr: ast.expr, helpers: dict, depth: int = 0):
    """``(slice-count node or None, width node, element widths or dtype node)`` of an SBUF tile expression.

    ``None`` when the expression makes no SBUF tile. A base-anchored view of a tile is the
    tile; a helper that returns one is read through its body with the caller's arguments in
    place of its parameters, so a tile's row is read where the tile is made.
    """
    while isinstance(expr, (ast.Subscript, ast.IfExp)):
        if isinstance(expr, ast.Subscript) and not _from_base(expr):
            return None
        expr = expr.value if isinstance(expr, ast.Subscript) else expr.body
    if not isinstance(expr, ast.Call) or depth > 8:
        return None
    maker = getattr(expr.func, "id", "") or getattr(expr.func, "attr", "")
    if maker in _MAKERS and len(expr.args) in (2, 3):
        return (expr.args[1] if len(expr.args) == 3 else None, expr.args[-1], _MAKERS[maker])
    if maker == "ndarray":
        shape = expr.args[0] if expr.args else None
        buffer = next((kw.value for kw in expr.keywords if kw.arg == "buffer"), None)
        if not isinstance(shape, ast.Tuple) or len(shape.elts) < 2:
            return None
        if buffer is not None and getattr(buffer, "attr", "") != "sbuf":
            return None
        dtype = next((kw.value for kw in expr.keywords if kw.arg == "dtype"), None)
        return (shape.elts[1] if len(shape.elts) == 3 else None, shape.elts[-1], dtype)
    if maker in helpers and maker not in _MAKERS:
        inlined = _returned(helpers[maker], expr)
        return None if inlined is None else _shape(inlined, helpers, depth + 1)
    return None


def _sizes(kind, source_itemsizes: tuple[int, ...]) -> tuple[int, ...] | None:
    """The element widths behind a shape reading: a maker's own, or those of a dtype node."""
    return kind if isinstance(kind, tuple) else _itemsizes(kind, source_itemsizes)


def _tile_of(call: ast.expr, names: dict, source_itemsizes: tuple[int, ...], helpers: dict):
    """``(slices, width, itemsizes)`` for a tile-making expression, or ``None`` when it is not one."""
    shape = _shape(call, helpers)
    if shape is None:
        return None
    slices = 1 if shape[0] is None else _eval(shape[0], names)
    width, sizes = _eval(shape[1], names), _sizes(shape[2], source_itemsizes)
    if width is None or slices is None or sizes is None:
        return None
    return (slices, width, sizes)


def _declarations(fn: ast.FunctionDef) -> list[tuple[int, str, ast.expr]]:
    """Every ``name = <expr>`` and ``name.append(<expr>)`` in a body, as ``(line, name, expr)``."""
    found = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            found.append((node.lineno, node.targets[0].id, node.value))
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) \
                and getattr(node.value.func, "attr", "") == "append" and node.value.args \
                and isinstance(node.value.func.value, ast.Name):
            found.append((node.lineno, node.value.func.value.id, node.value.args[0]))
    return found


def _tiles(fn: ast.FunctionDef, names: dict, source_itemsizes: tuple[int, ...], helpers: dict) -> dict:
    """Per tile name in a body: slice count, slice width and the element widths it can have.

    A name built by ``name.append(<tile>)`` is a list of such tiles; ``name[i]`` is one of them.
    """
    found: dict = {}
    for _, name, expr in _declarations(fn):
        tile = _tile_of(expr, names, source_itemsizes, helpers)
        if tile is not None:
            found[name] = tile
    return found


def _values(node: ast.expr | None, names: dict, loops: dict) -> tuple[int, ...] | None:
    """The values a bound takes: a constant, or an expression over one loop target."""
    if node is None:
        return (0,)
    loop_names = _loop_names([node], names, loops)
    if not loop_names:
        value = _eval(node, names)
        return None if value is None else (value,)
    combinations = _combinations(loop_names, loops)
    if combinations is None:
        return None
    values = set()
    for combo in combinations:
        value = _eval(node, _bind(names, dict(zip(loop_names, combo))))
        if value is None:
            return None
        values.add(value)
    return tuple(sorted(values))


def _base(node: ast.expr) -> ast.expr:
    """The name under a chain of subscripts."""
    while isinstance(node, ast.Subscript):
        node = node.value
    return node


def _elements(dst: ast.expr, tiles: dict, names: dict, loops: dict):
    """Where a destination expression starts, in elements from its tile's base, with its widths.

    Returns ``(offsets, itemsizes)``; ``offsets`` is ``None`` for a destination the arithmetic
    cannot place, including any name the body never declared as a tile.
    """
    base = _base(dst)
    if not isinstance(base, ast.Name) or base.id not in tiles:
        return None, None
    slices, width, sizes = tiles[base.id]
    if not isinstance(dst, ast.Subscript):
        return (0,), sizes
    parts = dst.slice.elts if isinstance(dst.slice, ast.Tuple) else [dst.slice]
    if len(parts) == 1 and not isinstance(parts[0], ast.Slice):
        return (0,), sizes                       # one tile out of a list of tiles
    if len(parts) == 3:
        index = _values(parts[1], names, loops) if not isinstance(parts[1], ast.Slice) else tuple(range(slices))
        lowers = _values(parts[2].lower, names, loops) if isinstance(parts[2], ast.Slice) else (0,)
        if index is None or lowers is None:
            return None, sizes
        return tuple(sorted({i * width + lo for i in index for lo in lowers})), sizes
    if len(parts) == 2:
        lowers = _values(parts[1].lower, names, loops) if isinstance(parts[1], ast.Slice) else (0,)
        return (None if lowers is None else tuple(sorted(set(lowers)))), sizes
    return None, sizes


def _source_param(call: ast.Call) -> str:
    """The name a ``dma_transpose`` reads its source from (``NAME.ap(...)``)."""
    src = next((kw.value for kw in call.keywords if kw.arg == "src"), None)
    if isinstance(src, ast.Call) and isinstance(src.func, ast.Attribute):
        return getattr(_base(src.func.value), "id", "")
    return getattr(_base(src), "id", "") if src is not None else ""


def _source_name(call: ast.Call) -> str:
    """The source tensor a direct ``dma_transpose`` reads."""
    return _source_param(call)


def _shape_params(helper: ast.FunctionDef, inner: ast.Call, inner_dst: ast.expr) -> set[str]:
    """The helper parameters its rows, width, loop bounds and destination slice depend on.

    A source offset is not among them, so a caller's source arithmetic never widens the census.
    """
    rows_node, width_node = _pattern(inner)
    roots = [n for n in (rows_node, width_node) if n is not None]
    if isinstance(inner_dst, ast.Subscript):
        roots.append(inner_dst.slice)
    for node in ast.walk(helper):
        if isinstance(node, ast.For):
            roots.append(node.iter)
    assigns = _assigns(helper)
    seen, todo = set(), set().union(*(_referenced(r) for r in roots)) if roots else set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in assigns:
            todo |= _referenced(assigns[name])
    return seen


def _pattern(call: ast.Call):
    """The ``[[stride, ROWS], [1, WIDTH]]`` pattern nodes of a ``dma_transpose`` source."""
    src = next((kw.value for kw in call.keywords if kw.arg == "src"), None)
    if not isinstance(src, ast.Call) or getattr(src.func, "attr", "") != "ap":
        return None, None
    pattern = next((kw.value for kw in src.keywords if kw.arg == "pattern"), None)
    if not isinstance(pattern, ast.List) or len(pattern.elts) != 2:
        return None, None
    rows, width = pattern.elts
    if not all(isinstance(p, ast.List) and len(p.elts) == 2 for p in (rows, width)):
        return None, None
    return rows.elts[1], width.elts[1]


def _direct(call: ast.Call, names: dict, loops: dict, r0: str | None):
    """A direct ``dma_transpose``'s rows per DMA and width, over the loop that steps its destination."""
    rows_node, width_node = _pattern(call)
    if rows_node is None:
        return None, None
    width = _values(width_node, names, loops)
    if r0 is None:
        rows = _values(rows_node, names, loops)
        return rows, width
    steps = loops.get(r0)
    if steps is None:
        return None, width
    rows = []
    for v in steps:
        row = _eval(rows_node, _names_with(names, r0, v))
        if row is None:
            return None, width
        rows.append(row)
    return tuple(rows), width


def _referenced(node: ast.expr) -> set[str]:
    """Every name an expression reads."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _loop_names(nodes: list, names: dict, loops: dict) -> list[str]:
    """The loop targets some expressions depend on, directly or through the body's assignments."""
    assigns = names.get("__assigns__", {})
    seen, todo = set(), set().union(*(_referenced(n) for n in nodes)) if nodes else set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in assigns:
            todo |= _referenced(assigns[name])
    return sorted(n for n in seen if n in loops)


def _bind(names: dict, values: dict) -> dict:
    """The names with loop targets bound, and the assignments that depend on them re-read."""
    bound = {**names, **values}
    assigns = names.get("__assigns__", {})
    for _ in range(3):
        for name, node in assigns.items():
            if name in values:
                continue
            v = _value(node, bound)
            if v is not None:
                bound[name] = v
    return bound


def _names_with(names: dict, target: str, value: int) -> dict:
    """The names with one loop target bound."""
    return _bind(names, {target: value})


def _combinations(loop_names: list[str], loops: dict):
    """Every combination of the named loops' values, or ``None`` when one is unreadable or too many."""
    ranges = [loops.get(n) for n in loop_names]
    if any(r is None for r in ranges):
        return None
    total = 1
    for r in ranges:
        total *= max(len(r), 1)
    if total > _COMBINATIONS:
        return None
    return list(itertools.product(*ranges))


def _assigns(fn: ast.FunctionDef) -> dict[str, ast.expr]:
    """The body's single-name assignments, by name, so a loop-dependent one can be re-read."""
    return {n.targets[0].id: n.value for n in ast.walk(fn)
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}


def _step_target(dst: ast.expr, loops: dict) -> str | None:
    """The loop target a destination slice is stepped by, if any."""
    if not isinstance(dst, ast.Subscript):
        return None
    for n in ast.walk(dst.slice):
        if isinstance(n, ast.Name) and n.id in loops:
            return n.id
    return None


def census(source: str, at: dict, arithmetic: dict, source_itemsizes: tuple[int, ...] = (2, 4)) -> list[Site]:
    """Every transpose site of ``source`` read at geometry ``at`` with the module's own ``arithmetic``."""
    tree = ast.parse(source)
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    makers = _helpers(tree)
    base = {"__builtins__": {"min": min, "max": max, "len": len, "range": range}, **arithmetic, **at}
    sites: list[Site] = []
    helpers: dict[str, tuple[ast.FunctionDef, ast.Call, ast.expr, int]] = {}
    for fn in functions:
        params = [a.arg for a in fn.args.args]
        names = _names(fn, base)
        names["__assigns__"] = _assigns(fn)
        loops, tiles = _loops(fn, names), _tiles(fn, names, source_itemsizes, makers)
        for node in _calls(fn, names):
            op = getattr(node.func, "attr", "")
            if op == "load_transpose2d" and node.args and isinstance(node.args[0], ast.Subscript):
                sub = node.args[0]
                parts = sub.slice.elts if isinstance(sub.slice, ast.Tuple) else [sub.slice]
                extents = []
                for p in parts[:2]:
                    if not isinstance(p, ast.Slice):
                        extents.append(None)
                        continue
                    span = ast.BinOp(left=p.upper, op=ast.Sub(), right=p.lower or ast.Constant(0))
                    ast.copy_location(span, p)
                    extents.append(_values(ast.fix_missing_locations(span), names, loops))
                source = getattr(_base(sub), "id", "")
                sites.append(Site(node.lineno, "", source, extents[0], extents[1] if len(extents) > 1 else None,
                                  {s: (0,) for s in source_itemsizes}))
            elif op == "dma_transpose":
                dst = next((kw.value for kw in node.keywords if kw.arg == "dst"), None)
                if dst is None:
                    continue
                dst_base = _base(dst)
                if isinstance(dst_base, ast.Name) and dst_base.id in params:
                    helpers[fn.name] = (fn, node, dst, params.index(dst_base.id))
                    continue                     # read once per caller, below
                rows, width = _direct(node, names, loops, _step_target(dst, loops))
                offsets, sizes = _elements(dst, tiles, names, loops)
                sizes = sizes or source_itemsizes
                sites.append(Site(node.lineno, "", _source_name(node), rows, width,
                                  {s: (None if offsets is None else tuple(o * s for o in offsets)) for s in sizes}))
    for fn in functions:
        names = _names(fn, base)
        names["__assigns__"] = _assigns(fn)
        loops, tiles = _loops(fn, names), _tiles(fn, names, source_itemsizes, makers)
        for node in _calls(fn, names):
            if getattr(node.func, "id", "") not in helpers:
                continue
            helper, inner, inner_dst, dst_index = helpers[node.func.id]
            params = [a.arg for a in helper.args.args]
            relevant = _shape_params(helper, inner, inner_dst)
            args = [(params[i], a) for i, a in enumerate(node.args)
                    if i < len(params) and i != dst_index and params[i] in relevant]
            args += [(kw.arg, kw.value) for kw in node.keywords if kw.arg in relevant]
            loop_names = _loop_names([a for _, a in args], names, loops)
            combinations = _combinations(loop_names, loops) if loop_names else [()]
            bindings = [] if combinations is None else \
                [_bind(names, dict(zip(loop_names, combo))) for combo in combinations]
            rows_seen, width_seen, inner_lowers = set(), set(), set()
            readable = bool(bindings)
            for caller_names in bindings:
                bound = dict(base)
                for name, arg in args:
                    v = _eval(arg, caller_names)
                    if v is not None:
                        bound[name] = v
                hnames = _names(helper, bound)
                hnames["__assigns__"] = _assigns(helper)
                hloops = _loops(helper, hnames)
                r, w = _direct(inner, hnames, hloops, _step_target(inner_dst, hloops))
                lowers = None
                if isinstance(inner_dst, ast.Subscript):
                    parts = inner_dst.slice.elts if isinstance(inner_dst.slice, ast.Tuple) else [inner_dst.slice]
                    last = parts[-1]
                    lowers = _values(last.lower, hnames, hloops) if isinstance(last, ast.Slice) else (0,)
                if r is None or w is None or lowers is None:
                    readable = False
                    break
                rows_seen.update(r)
                width_seen.update(w)
                inner_lowers.update(lowers)
            rows = tuple(sorted(rows_seen)) if readable else None
            width = tuple(sorted(width_seen)) if readable else None
            inner_lowers = tuple(sorted(inner_lowers)) if readable else None
            caller_dst = node.args[dst_index] if dst_index < len(node.args) else None
            offsets, sizes = (None, None) if caller_dst is None else _elements(caller_dst, tiles, names, loops)
            sizes = sizes or source_itemsizes
            if offsets is None or inner_lowers is None:
                combined = None
            else:
                combined = tuple(sorted({o + i for o in offsets for i in inner_lowers}))
            src_index = next((i for i, a in enumerate(helper.args.args) if a.arg == _source_param(inner)), None)
            source = getattr(_base(node.args[src_index]), "id", "") if src_index is not None and src_index < len(node.args) else ""
            sites.append(Site(node.lineno, node.func.id, source, rows, width,
                              {s: (None if combined is None else tuple(c * s for c in combined)) for s in sizes}))
    return sorted(sites, key=lambda s: s.line)


def _bindings(fn: ast.FunctionDef, functions: list, base: dict) -> list[dict]:
    """The name sets a body is read under: one per call of it in the module, else the base alone."""
    params = [a.arg for a in fn.args.args]
    found = []
    for caller in functions:
        if caller is fn:
            continue
        caller_names = None
        for node in ast.walk(caller):
            if not isinstance(node, ast.Call) or getattr(node.func, "id", "") != fn.name:
                continue
            caller_names = _names(caller, base) if caller_names is None else caller_names
            bound = dict(base)
            pairs = list(zip(params, node.args)) + [(kw.arg, kw.value) for kw in node.keywords if kw.arg]
            for param, arg in pairs:
                value = _eval(arg, caller_names)
                if value is not None:
                    bound[param] = value
            found.append(bound)
    return found or [base]


def tile_rows(source: str, at: dict, arithmetic: dict, source_itemsizes: tuple[int, ...] = (2, 4)) -> list[tuple[int, str, int | None]]:
    """Every SBUF tile a module declares, as ``(line, name, bytes per partition)``, at geometry ``at``.

    A body with parameters is read once per call of it in the module, its parameters bound to the
    caller's values; a helper that only returns a tile is read at its callers instead. A row the
    arithmetic cannot size reads ``None``. A tile declared in a loop reads once per row size it takes.
    """
    tree = ast.parse(source)
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    makers = _helpers(tree)
    base = {"__builtins__": {"min": min, "max": max, "len": len, "range": range}, **arithmetic, **at}
    found: set = set()
    for fn in functions:
        returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
        if len(returns) == 1 and _shape(returns[0].value, makers) is not None:
            continue
        for bound in _bindings(fn, functions, base):
            names = _names(fn, bound)
            loops = _loops(fn, names)
            for line, name, expr in _declarations(fn):
                shape = _shape(expr, makers)
                if shape is None:
                    continue
                slices = (1,) if shape[0] is None else _values(shape[0], names, loops)
                widths = _values(shape[1], names, loops)
                sizes = _sizes(shape[2], source_itemsizes)
                if slices is None or widths is None or sizes is None:
                    found.add((line, name, None))
                    continue
                found.update((line, name, sl * w * s) for sl in slices for w in widths for s in sizes)
    return sorted(found, key=lambda row: (row[0], row[1], -1 if row[2] is None else row[2]))


def narrow_tiles(source: str, at: dict, arithmetic: dict, source_itemsizes: tuple[int, ...] = (2, 4)) -> list[tuple[int, str, int | None]]:
    """The rows of :func:`tile_rows` that are not a whole line, or that the arithmetic cannot size.

    A reading for the allocator's placement, which no arithmetic here controls: packed tile
    bases stay on the line only when every earlier tile's row is a whole number of lines.
    """
    return [row for row in tile_rows(source, at, arithmetic, source_itemsizes)
            if row[2] is None or row[2] % LINE]
