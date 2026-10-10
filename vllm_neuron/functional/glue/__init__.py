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
* ``1`` or unset: :data:`vllm_neuron.envs.DEFAULT_GLUE_FUSED_SPEC`: prefill buckets, and
  mhc_pre at the verify step. Its docstring gives the measurements it rests on.
* ``all``: every kernel at every phase and row count.
* otherwise a comma list of rules ``kernel[:phase][@rows]``:

  - ``phase`` is ``prefill``, ``decode``, ``verify`` or ``all`` (the default).
    ``verify`` is the speculative verify step at the mHC sites (below), and only
    ``mhc_pre`` and ``mhc_post`` take it (:data:`VERIFY_KERNELS`).
  - ``rows`` bounds the call's row count (decode: the padded batch B; verify: B times
    ``1 + k``; prefill: the chunk's T): ``N``, ``N-M``, ``N-`` or ``-M``, inclusive,
    ``N >= 1``.
  - Rules add up: a call is fused when any rule selects it.
  - Example: ``mhc_post:prefill,mhc_post:decode@64,kda_output:decode@2-64``.

An unknown kernel or phase, a bad row range or an empty rule raises ``ValueError``
that names the switch, at the first site that reads it, so a typo cannot serve a
different network. The switch is read when a graph is traced: a compiled graph keeps
the routes it was traced with.

Phase. The KDA layer passes the step's phase (``is_prefill``) to kda_projections and
kda_output; it calls the verify step decode. An mHC site sees only ``[T, S, H]``
streams, so its layer (``Glm5NextHyperConnection``) tells the phases apart by row count
(:func:`phase_of_rows`): the verify step's row counts (:func:`verify_rows`, the decode
buckets times ``1 + k`` under speculative method "mtp", :func:`verify_draft_k`) are
``verify``, other calls of up to the largest decode batch its config declares
(``neuron_config.num_seqs_buckets``) are ``decode``, and larger calls are ``prefill``. At
construction it refuses a value that the row count cannot follow, or whose
``kernel:verify`` rule leaves out one of the verify step's row counts
(:func:`require_rows_tell_phase`). A call whose phase is not known (``phase=None``) is
selected only by rules without a phase.

Every kernel is also bounded by its own shape rules in its ``*_admits`` predicate,
whatever the switch selects. A kernel may declare its own row bound (for example
``kda_projections.KDA_PROJECTIONS_MAX_TOKENS``); mhc_pre has none, it walks the rows
in token tiles.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass

from vllm_neuron import envs

#: The switch for every kernel in this package (registered in :mod:`vllm_neuron.envs`).
GLUE_FUSED_ENV = "VLLM_NEURON_GLUE_FUSED"

#: The fused sites, by the names the switch takes.
KERNELS = ("mhc_pre", "kda_projections", "kda_output", "mhc_post")

#: The step phases a rule can name.
PHASES = ("prefill", "decode", "verify")

#: The kernels whose sites tell the verify step apart (the mHC sites, by row count). The
#: KDA layer calls the verify step decode, so a ``verify`` rule for its kernels would
#: never select a call and is refused.
VERIFY_KERNELS = ("mhc_pre", "mhc_post")

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
            tokens: the call's row count (decode: the padded batch; verify: the batch
                times ``1 + k``; prefill: the chunk).
            phase: ``"prefill"``, ``"decode"`` or ``"verify"``. None when the call site
                does not know it; then only a rule without a phase selects the call.

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
        f"{{prefill, decode, verify, all}} and rows N, N-M, N- or -M (N >= 1)")


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
    if colon and phase == "verify" and kernel not in VERIFY_KERNELS:
        raise _refuse(spec, f"{kernel} has no verify phase: the KDA layer calls the "
                            f"verify step decode, and only {', '.join(VERIFY_KERNELS)} "
                            f"take verify")
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


def verify_draft_k(draft_k: int) -> int:
    """``k`` of the speculative verify step: ``draft_k`` under speculative method "mtp",
    otherwise 0.

    ``draft_k`` is the model's draft count, from its one reader (``mtp.shadow_draft_k``,
    contract C1), which the mHC layer passes in. The shadow draft (that knob with no
    speculative config) runs no verify step, so it is 0 here. The speculative config is
    the current vLLM config's, which the worker sets around ``load_model``, where the
    layers are built; outside that context there is none, and this is 0.
    """
    from vllm.config import get_current_vllm_config_or_none

    config = get_current_vllm_config_or_none()
    speculative = None if config is None else config.speculative_config
    if speculative is None or speculative.method != "mtp":
        return 0
    return int(draft_k)


def verify_rows(decode_buckets: Iterable[int], draft_k: int) -> frozenset[int]:
    """The row counts of the speculative verify step: each decode bucket times ``1 + k``.

    Under speculative method "mtp" the runner runs a verify step of ``1 + k`` rows per
    request (``_decode_token_threshold``), so a decode bucket of ``B`` requests is a
    ``B * (1 + k)``-row call at an mHC site. ``draft_k`` is the verify step's ``k``
    (:func:`verify_draft_k`); 0 (no speculation, or the shadow draft) has no verify step.

    Raises:
        ValueError: ``draft_k`` is negative, or a bucket is not a positive row count.
    """
    k = int(draft_k)
    if k < 0:
        raise ValueError(f"{GLUE_FUSED_ENV}: draft count k={k} is negative; the verify "
                         f"step has 1 + k rows per request, k >= 0")
    buckets = [int(b) for b in decode_buckets]
    if any(b < 1 for b in buckets):
        raise ValueError(f"{GLUE_FUSED_ENV}: decode buckets {buckets} hold a row count "
                         f"below 1")
    if k == 0:
        return frozenset()
    return frozenset(b * (1 + k) for b in buckets)


def phase_of_rows(rows: int, max_decode_rows: int, verify: Collection[int] = ()) -> str:
    """The phase a site that sees only the row count gives a call of ``rows`` rows.

    A verify-step row count (``verify``, from :func:`verify_rows`) is ``verify``, also
    where a decode step of one row per request has it: a speculative server also runs
    that graph (the first decode step, a mixed step), and at ``B * (1 + k)`` requests it
    has the row count of the ``B``-request verify step. The row count cannot tell the two
    apart, so such a decode batch takes the verify route; the fused kernel and the torch
    route serve either phase, so this choice changes the route only. Otherwise a call of
    at most ``max_decode_rows`` rows (the largest decode batch) is ``decode``, and a
    larger one is a ``prefill`` chunk.
    """
    rows = int(rows)
    if rows in verify:
        return "verify"
    return "decode" if rows <= int(max_decode_rows) else "prefill"


def require_rows_tell_phase(kernels: Iterable[str], max_decode_rows: int,
                            prefill_rows: Iterable[int], spec: str | None = None,
                            verify: Iterable[int] = ()) -> None:
    """Refuse a value that a site which tells the phase by row count cannot follow.

    Such a site gives each call the phase :func:`phase_of_rows` names. A prefill chunk
    whose row count is also a decode batch's, or the verify step's, looks like that call
    there. So when the value routes one of ``kernels`` differently in the two phases at
    the row count of such a chunk, the site would serve the chunk the other route.

    A value that names the verify phase for one of ``kernels`` (a ``kernel:verify`` rule)
    must select it at every verify-step row count: a row count with no entry would keep
    the torch route at that verify step without a word, for example ``mhc_pre:verify@4``
    on a server whose 2 drafts make the verify step 3 rows per request.

    Args:
        kernels: the kernels the site serves, names in :data:`KERNELS`.
        max_decode_rows: the largest row count of a decode call at the site.
        prefill_rows: the row counts a prefill chunk can have there (the prefill
            buckets).
        spec: the value to check; None checks the switch's current value.
        verify: the verify step's row counts there (:func:`verify_rows`); empty on a
            server without speculation.

    Raises:
        ValueError: naming the switch, the kernel and the row count, when a prefill
            chunk is routed by phase at a row count the site gives another phase, or
            when a ``kernel:verify`` rule leaves a verify-step row count out.
    """
    value = envs.VLLM_NEURON_GLUE_FUSED if spec is None else spec
    selection = glue_selection(value)
    verify = frozenset(int(r) for r in verify)
    for rows in sorted({int(r) for r in prefill_rows}):
        seen_as = phase_of_rows(rows, max_decode_rows, verify)
        if seen_as == "prefill":
            continue
        for kernel in kernels:
            if selection.selects(kernel, rows, "prefill") != selection.selects(
                    kernel, rows, seen_as):
                like = (f"a verify step (verify steps have {sorted(verify)} rows here: "
                        f"each decode bucket times 1 + k)" if seen_as == "verify" else
                        f"a decode batch (decode batches reach {max_decode_rows} rows "
                        f"here)")
                raise ValueError(
                    f"{GLUE_FUSED_ENV}={value!r} routes {kernel} at {rows} rows by "
                    f"phase, but at a site that sees only the row count a {rows}-row "
                    f"prefill chunk looks like {like}. Use a value whose {kernel} rules "
                    f"do not depend on the phase at {rows} rows: for example "
                    f"{kernel}@{rows} in place of its phase rules there, which fuses "
                    f"both, or 0, or all")
    for kernel in kernels:
        if not any(rule.kernel == kernel and rule.phase == "verify"
                   for rule in selection.rules):
            continue
        missing = sorted(r for r in verify if not selection.selects(kernel, r, "verify"))
        if missing:
            raise ValueError(
                f"{GLUE_FUSED_ENV}={value!r} fuses {kernel} at the verify step, but no "
                f"{kernel}:verify rule selects its {missing} rows here (verify steps "
                f"have {sorted(verify)} rows: each decode bucket times 1 + k). Name "
                f"every one, or use {kernel}:verify without a row bound")
