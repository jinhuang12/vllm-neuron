# SPDX-License-Identifier: Apache-2.0
"""Small fused kernels that replace the XLA glue between the GLM-5.3-Flash NKI kernels.

Each module here takes one run of elementwise and small-matmul torch ops and does it in
one NKI launch, on the operands' served dtypes, so the casts and transposes that the
torch spelling implies are not issued. Each entry point has an ``*_admits`` predicate,
and a call site keeps its torch route whenever the predicate says no.

Which kernel serves which call: ``VLLM_NEURON_GLUE_FUSED``
------------------------------------------------------------
The switch is registered in :mod:`vllm_neuron.envs`. It names four sites
(:data:`KERNELS`):

* ``mhc_pre``: ``mhc_pre.py``, the mHC pre-mix at both mHC sites of every layer.
* ``kda_projections``: ``kda_projections.py``, the six KDA input projections.
* ``kda_output``: ``kda_output.py``, the KDA gated norm and output projection.
* ``mhc_post``: the bf16 epilogue of ``functional/mhc/hyper_connection.py``. When it is
  selected, the mHC combine takes the bf16 streams as they are, not fp32 copies, and
  rounds its fp32 result to bf16 once in the kernel, not in an XLA convert after it.
  The values are the same in the simulator; on the device the two roundings can
  differ by one bf16 step in some elements.

Values:

* ``0``: nothing is fused. Every site takes its torch route.
* ``1`` or unset: :data:`vllm_neuron.envs.DEFAULT_GLUE_FUSED_SPEC`, which is prefill
  only. Its docstring gives the measurement it rests on.
* ``all``: every kernel at every phase and row count.
* otherwise a comma list of rules ``kernel[:phase][@rows]``:

  - ``phase`` is ``prefill``, ``decode`` or ``all`` (the default).
  - ``rows`` bounds the call's row count (decode: the padded batch B; prefill: the
    chunk's T): ``N``, ``N-M``, ``N-`` or ``-M``, inclusive, ``N >= 1``.
  - Rules add up: a call is fused when any rule selects it.
  - Example: ``mhc_post:prefill,mhc_post:decode@64,kda_output:decode@2-64``.

An unknown kernel or phase, a bad row range or an empty rule raises ``ValueError``
that names the switch, at the first site that reads it, so a typo cannot serve a
different network. The switch is read when a graph is traced: a compiled graph keeps
the routes it was traced with.

Phase. The KDA layer passes the step's phase (``is_prefill``) to kda_projections and
kda_output. An mHC site sees only ``[T, S, H]`` streams, so its layer
(``Glm5NextHyperConnection``) tells the phases apart by row count, against the largest
decode batch its config declares (``neuron_config.num_seqs_buckets``), and refuses at
construction a value that the row count cannot follow (:func:`require_rows_tell_phase`).
A call whose phase is not known (``phase=None``) is selected only by rules without a
phase.

Every kernel is also bounded by its own shape rules in its ``*_admits`` predicate,
whatever the switch selects. A kernel may declare its own row bound (for example
``kda_projections.KDA_PROJECTIONS_MAX_TOKENS``); mhc_pre has none, it walks the rows
in token tiles.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from vllm_neuron import envs

#: The switch for every kernel in this package (registered in :mod:`vllm_neuron.envs`).
GLUE_FUSED_ENV = "VLLM_NEURON_GLUE_FUSED"

#: The fused sites, by the names the switch takes.
KERNELS = ("mhc_pre", "kda_projections", "kda_output", "mhc_post")

#: The two step phases a rule can name.
PHASES = ("prefill", "decode")

@dataclass(frozen=True)
class _Rule:
    """One ``kernel[:phase][@rows]`` rule. ``phase`` None: both. ``hi`` None: no bound."""

    kernel: str
    phase: str | None
    lo: int
    hi: int | None

    def selects(self, kernel: str, tokens: int, phase: str | None) -> bool:
        return (kernel == self.kernel and (self.phase is None or phase == self.phase)
                and tokens >= self.lo and (self.hi is None or tokens <= self.hi))


@dataclass(frozen=True)
class GlueSelection:
    """A parsed ``VLLM_NEURON_GLUE_FUSED`` value: the set of rules it names."""

    rules: frozenset[_Rule]

    def selects(self, kernel: str, tokens: int, phase: str | None = None) -> bool:
        """True when ``kernel`` is fused for a call of ``tokens`` rows.

        Args:
            kernel: one of :data:`KERNELS`.
            tokens: the call's row count (decode: the padded batch; prefill: the chunk).
            phase: ``"prefill"`` or ``"decode"``. None when the call site does not know
                it; then only a rule without a phase selects the call.

        Raises:
            ValueError: ``kernel`` or ``phase`` is not one this package defines.
        """
        if kernel not in KERNELS:
            raise ValueError(f"{GLUE_FUSED_ENV}: no glue kernel {kernel!r}; the kernels "
                             f"are {', '.join(KERNELS)}")
        if phase is not None and phase not in PHASES:
            raise ValueError(f"{GLUE_FUSED_ENV}: phase {phase!r} is not one of {PHASES}")
        return any(rule.selects(kernel, int(tokens), phase) for rule in self.rules)


def _refuse(spec: str, why: str) -> ValueError:
    return ValueError(
        f"{GLUE_FUSED_ENV}={spec!r}: {why}. Use 0, 1, all, or a comma list of "
        f"kernel[:phase][@rows] with kernel in {{{', '.join(KERNELS)}}}, phase in "
        f"{{prefill, decode, all}} and rows N, N-M, N- or -M (N >= 1)")


def _bound(spec: str, text: str) -> int:
    if not text.isdigit() or int(text) < 1:
        raise _refuse(spec, f"row bound {text!r} is not an integer >= 1")
    return int(text)


def _parse_rule(spec: str, entry: str) -> _Rule:
    head, at, rows = entry.partition("@")
    kernel, colon, phase = head.partition(":")
    if kernel not in KERNELS:
        raise _refuse(spec, f"unknown kernel {kernel!r}")
    if colon and phase not in PHASES + ("all",):
        raise _refuse(spec, f"unknown phase {phase!r} for {kernel}")
    lo, hi = 1, None
    if at:
        low, dash, high = rows.partition("-")
        if not dash:
            lo = hi = _bound(spec, low)
        else:
            if not low and not high:
                raise _refuse(spec, f"empty row range in {entry!r}")
            lo = _bound(spec, low) if low else 1
            hi = _bound(spec, high) if high else None
        if hi is not None and hi < lo:
            raise _refuse(spec, f"row range {rows!r} is empty")
    return _Rule(kernel, phase if colon and phase != "all" else None, lo, hi)


def _parse(spec: str) -> GlueSelection:
    spec = spec.strip()
    if spec == "0":
        return GlueSelection(frozenset())
    if spec == "1":
        return _parse(envs.DEFAULT_GLUE_FUSED_SPEC)
    if spec == "all":
        return GlueSelection(frozenset(_Rule(k, None, 1, None) for k in KERNELS))
    entries = [part.strip() for part in spec.split(",")]
    if not any(entries):
        raise _refuse(spec, "no rule")
    rules = []
    for entry in entries:
        if not entry:
            raise _refuse(spec, "an empty rule between commas")
        rules.append(_parse_rule(spec, entry))
    return GlueSelection(frozenset(rules))


def glue_selection(spec: str | None = None) -> GlueSelection:
    """``spec``, or the switch's current value when None, as a :class:`GlueSelection`.

    Raises:
        ValueError: the value is malformed or names an unknown kernel or phase.
    """
    return _parse(envs.VLLM_NEURON_GLUE_FUSED if spec is None else spec)


def glue_selected(kernel: str, tokens: int, phase: str | None = None) -> bool:
    """True when the switch fuses ``kernel`` for a call of ``tokens`` rows in ``phase``.

    ``phase`` None: the call site does not know it (see :meth:`GlueSelection.selects`).
    """
    return glue_selection().selects(kernel, tokens, phase)


def require_rows_tell_phase(kernels: Iterable[str], max_decode_rows: int,
                            prefill_rows: Iterable[int], spec: str | None = None) -> None:
    """Refuse a value that a site which tells the phase by row count cannot follow.

    Such a site calls a call of at most ``max_decode_rows`` rows (its largest decode
    batch) decode, and a larger one prefill. A prefill chunk that is no larger looks
    like a decode batch there. So when the value routes one of ``kernels`` differently
    in the two phases at the row count of such a chunk, the site would serve the chunk
    the decode route.

    Args:
        kernels: the kernels the site serves, names in :data:`KERNELS`.
        max_decode_rows: the largest row count of a decode call at the site.
        prefill_rows: the row counts a prefill chunk can have there (the prefill
            buckets).
        spec: the value to check; None checks the switch's current value.

    Raises:
        ValueError: naming the switch, the kernel and the row count, when a prefill
            chunk of at most ``max_decode_rows`` rows is routed by phase.
    """
    value = envs.VLLM_NEURON_GLUE_FUSED if spec is None else spec
    selection = glue_selection(value)
    for rows in sorted({int(r) for r in prefill_rows if int(r) <= max_decode_rows}):
        for kernel in kernels:
            if selection.selects(kernel, rows, "prefill") != selection.selects(
                    kernel, rows, "decode"):
                raise ValueError(
                    f"{GLUE_FUSED_ENV}={value!r} routes {kernel} at {rows} rows by "
                    f"phase, but at a site that sees only the row count a {rows}-row "
                    f"prefill chunk looks like a decode batch (decode batches reach "
                    f"{max_decode_rows} rows here). Use a value whose {kernel} rules do "
                    f"not depend on the phase at {max_decode_rows} rows or fewer, for "
                    f"example 0 or all")
