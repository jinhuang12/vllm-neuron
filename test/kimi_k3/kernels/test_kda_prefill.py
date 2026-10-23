# SPDX-License-Identifier: Apache-2.0
"""KDA prefill kernel against an fp64 token-by-token recurrence.

The oracle follows the checkpoint's ``KimiDeltaAttention`` chunk path: q and k
L2-normed with eps 1e-6 inside the root, query scale ``K ** -0.5``, beta by
sigmoid, gate ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``, then per
token ``S *= exp(gate)``, ``S += k (beta (v - k S))``, ``o = q S``. The state
must stay within 1e-4 relative Frobenius error of the oracle, per head.

The tests run under ``torch.compile`` on core 0 of ``NEURON_RT_VISIBLE_CORES``
(the runtime needs ``NEURON_RT_ROOT_COMM_ID`` set), or on the NKI simulator when
``VLLM_NEURON_CPU_MODE=1`` and ``NKI_SIMULATOR=1`` are set.
"""

import os

import numpy as np
import pytest
import torch

from vllm_neuron.functional.kda.prefill import kda_prefill

H, D, T = 3, 128, 2048
LOWER_BOUND = -5.0
TOLERANCE = 1e-4
SIMULATOR = os.environ.get("VLLM_NEURON_CPU_MODE") == "1" and os.environ.get("NKI_SIMULATOR") == "1"


_compiled = None


def _call(*args):
    global _compiled
    if SIMULATOR:
        return kda_prefill(*args, lower_bound=LOWER_BOUND)
    if _compiled is None:
        from vllm_neuron import envs

        torch.classes.neuron.Runtime().initialize()
        _compiled = torch.compile(kda_prefill, backend=envs.get_compile_backend_name(), fullgraph=True)
    device = torch.device("neuron", 0)
    dev = [a.to(device) if isinstance(a, torch.Tensor) else a for a in args]
    out, state = _compiled(*dev, lower_bound=LOWER_BOUND)
    return out.cpu(), state.cpu()


def _run(inputs, state=None):
    """Run the kernel on fp32 copies of ``inputs``; return float64 numpy arrays."""
    args = [torch.from_numpy(np.ascontiguousarray(x, np.float32)) for x in inputs]
    seed = None if state is None else torch.from_numpy(np.array(state, np.float32))
    out, final = _call(*args, seed)
    return out.double().numpy(), final.double().numpy()


def _oracle(q, k, v, g, beta, a_log, dt_bias, state=None):
    """Return the fp64 token-recurrence output and final state."""
    q, k, v, g, beta = (np.asarray(x, np.float64) for x in (q, k, v, g, beta))
    S = np.zeros((H, D, D)) if state is None else np.array(state, np.float64)
    out = np.zeros(v.shape)
    with np.errstate(invalid="ignore", over="ignore"):
        qn = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) * D**-0.5
        kn = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
        decay = np.exp(LOWER_BOUND / (1 + np.exp(-np.exp(a_log)[:, None, None] * (g + dt_bias[:, None, :]))))
        b = 1 / (1 + np.exp(-beta))
        for t in range(q.shape[1]):
            S *= decay[:, t, :, None]
            u = b[:, t, None] * (v[:, t] - np.einsum("hk,hkv->hv", kn[:, t], S))
            S += kn[:, t, :, None] * u[:, None, :]
            out[:, t] = np.einsum("hk,hkv->hv", qn[:, t], S)
    return out, S


def _inputs(rng, n=T):
    """Draw raw projections at the served widths: q, k, v, g, beta, A_log, dt_bias."""
    return [rng.standard_normal((H, n, D)), rng.standard_normal((H, n, D)), rng.standard_normal((H, n, D)),
            rng.standard_normal((H, n, D)), rng.standard_normal((H, n)),
            np.log(rng.uniform(1, 16, H)), 0.1 * rng.standard_normal((H, D))]


def _assert_class_f(got, want):
    err = {name: [float(np.linalg.norm(a[h] - b[h]) / np.linalg.norm(b[h])) for h in range(H)]
           for name, a, b in zip(("out", "state"), got, want)}
    assert max(max(e) for e in err.values()) <= TOLERANCE, f"relative Frobenius error per head {err}"


def _rounded(inputs):
    return [np.asarray(x, np.float32).astype(np.float64) for x in inputs]


def test_correlated_rows_with_one_dominant_channel():
    rng = np.random.default_rng(1)
    x = _inputs(rng)
    for i in range(4):
        x[i] = 0.9 * rng.standard_normal((H, 1, D)) + 0.1 * x[i]
    for i in range(3):
        x[i][..., 7] *= 100.0
    x = _rounded(x)
    _assert_class_f(_run(x), _oracle(*x))


def test_repeated_tokens():
    x = _inputs(np.random.default_rng(2))
    x = _rounded([np.repeat(a[:, :1], T, axis=1) for a in x[:5]] + x[5:])
    _assert_class_f(_run(x), _oracle(*x))


def test_near_one_gates_and_saturated_beta_from_a_carried_state():
    rng = np.random.default_rng(3)
    x = _inputs(rng)
    x[6] = np.zeros((H, D))
    # exp(A_log) * g <= -12 keeps every decay above 1 - 2**-12.
    x[3] = -(12.0 + 4.0 * rng.random((H, T, D))) / np.exp(x[5])[:, None, None]
    x[4] = np.where(np.arange(T) % 2 == 0, 12.0, -12.0)[None, :].repeat(H, 0)
    x = _rounded(x)
    seed = 0.05 * rng.standard_normal((H, D, D)).astype(np.float32)
    _assert_class_f(_run(x, seed), _oracle(*x, state=seed))


def test_long_carried_state_over_four_calls():
    rng = np.random.default_rng(4)
    x = _rounded(_inputs(rng, 4 * T))
    state = 0.05 * rng.standard_normal((H, D, D)).astype(np.float32)
    want = _oracle(*x, state=state)
    outs = []
    for c in range(4):
        part = [a[:, c * T:(c + 1) * T] for a in x[:5]] + x[5:]
        out, state = _run(part, state)
        outs.append(out)
    _assert_class_f((np.concatenate(outs, axis=1), state), want)


def test_one_nan_token_reaches_no_earlier_token():
    x = _rounded(_inputs(np.random.default_rng(5)))
    clean_out, _ = _run(x)
    t = 1000
    for a in x[:5]:
        a[:, t] = np.nan
    out, state = _run(x)
    reached = np.flatnonzero(np.isnan(out[:, :t]).any(axis=(0, 2)))
    assert reached.size == 0, f"NaN reaches {reached.size} earlier tokens, from token {reached[:1]}"
    np.testing.assert_array_equal(out[:, :t], clean_out[:, :t])
    assert np.isnan(out[:, t:]).all() and np.isnan(state).all()
