# SPDX-License-Identifier: Apache-2.0
"""Per-graph compiler-argument helpers.

vLLM-Neuron compiles one graph per warmup call. Prefill, decode, jump-forward,
the encoder, and the spec-verify target share one ``torch.compile``.
They differ only by input bucket, so each can want its own neuronx-cc
arguments (for example a higher ``-O`` level for the plain decode graph and the
vision graph).

The runner resolves the per-graph arguments eagerly through the model's
``compiler_args_for`` policy and carries the concrete list to the compiler: on
the parallel-trace path it rides on the trace job's third element; on the
in-process path the site opens a ``scoped_compiler_args`` scope from libtorch.
This module keeps the pure helpers those policies use to edit the arguments.
"""

from typing import List, Sequence

_HLO2TENSORIZER_FLAG = "--internal-hlo2tensorizer-options="


def with_optimization_level(args: Sequence[str], level: int) -> List[str]:
    """Return a copy of ``args`` with the neuronx-cc ``-O`` level set to ``level``.

    Replace an existing ``-O<n>`` token. Append ``-O<level>`` when the list has
    no ``-O`` token. The input list stays unchanged.

    Args:
        args: The base compiler arguments, for example the caller default args.
        level: The neuronx-cc optimization level, for example ``3``.

    Returns:
        A new argument list at the wanted level.

    Example:
        >>> from vllm_neuron.utils.compiler_args_ctx import with_optimization_level
        >>> with_optimization_level(["--auto-cast=none", "-O1"], 3)
        ['--auto-cast=none', '-O3']
        >>> with_optimization_level(["--auto-cast=none"], 2)
        ['--auto-cast=none', '-O2']
    """
    flag = f"-O{level}"
    out = [flag if a.startswith("-O") and a[2:].isdigit() else a for a in args]
    if flag not in out:
        out.append(flag)
    return out


def with_hlo2tensorizer_option(args: Sequence[str], option: str) -> List[str]:
    """Return a copy of ``args`` with ``option`` added to the hlo2tensorizer options.

    neuronx-cc keeps only the last ``--internal-hlo2tensorizer-options`` entry, so a
    second entry would drop the options of the first. Extend that last entry in
    place instead. Append an entry when the list has none. Leave the list as it is
    when the entry already holds ``option``. The input list stays unchanged.

    Args:
        args: The base compiler arguments, for example the caller default args.
        option: One hlo2tensorizer option, for example
            ``"--partitioner-max-disj-runs=16"``.

    Returns:
        A new argument list with exactly one entry that holds ``option``.

    Example:
        >>> from vllm_neuron.utils.compiler_args_ctx import with_hlo2tensorizer_option
        >>> with_hlo2tensorizer_option(
        ...     ["-O1", "--internal-hlo2tensorizer-options=--a=1"], "--b=2"
        ... )
        ['-O1', '--internal-hlo2tensorizer-options=--a=1 --b=2']
        >>> with_hlo2tensorizer_option(["-O1"], "--b=2")
        ['-O1', '--internal-hlo2tensorizer-options=--b=2']
    """
    out = list(args)
    for i in reversed(range(len(out))):
        if out[i].startswith(_HLO2TENSORIZER_FLAG):
            existing = out[i][len(_HLO2TENSORIZER_FLAG) :]
            if option not in existing.split():
                joined = f"{existing} {option}" if existing else option
                out[i] = f"{_HLO2TENSORIZER_FLAG}{joined}"
            return out
    out.append(f"{_HLO2TENSORIZER_FLAG}{option}")
    return out
