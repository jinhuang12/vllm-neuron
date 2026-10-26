# SPDX-License-Identifier: Apache-2.0
"""fp64 oracle test for the vision tower's attention through ``flash_attention``.

The call is the tower's: one rank's 3 heads of 128, bidirectional, with per-query
key bounds, at scale 1.0. q and k arrive multiplied by the square root of the softmax
scale in fp32 and then rounded once to bf16, as the QKV call delivers them. The
oracle is the checkpoint's vision attention (``modeling_kimi_k3.py``) run per image
in fp64 from the bf16 inputs the call receives. Pass rule on every real row: each
element within 2^-7 of its value plus 2^-7 of the row RMS, and row cosine at least
0.9999. The 33,536-patch bucket allows twice that element bound, because the kernel
stores its exponentials in bf16 for the second matmul; its reading against the plain
bound is reported beside. Every item fails if the wrapper takes its torch branch.

On a Neuron device the wrapper runs under ``torch.compile`` on core 0 of
``NEURON_RT_VISIBLE_CORES``; the runtime needs ``NEURON_RT_ROOT_COMM_ID`` set to load the
program. With ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1`` it runs in the NKI simulator.
"""

import importlib
import math
import os

import pytest
import torch
import torch.nn.functional as F

from vllm_neuron import envs

if envs.VLLM_NEURON_CPU_MODE and os.environ.get("NKI_SIMULATOR") != "1":
    pytest.skip("the kernel runs on the host only under NKI_SIMULATOR=1", allow_module_level=True)

wrapper = importlib.import_module("vllm_neuron.functional.attention.attention_cte")

HEADS, HEAD_DIM = 3, 128
BUCKET, REAL = 5504, 5476
SLICE_FIRST, SLICE_REAL_ROWS, SLICE_ROWS = 7 * 688, 688, 768
LARGE_BUCKET, LARGE_REAL = 33536, 33488
PACKED = ((0, 2288), (2288, 5360))
SEED = 20261024
ELEMENT_TOL = 2.0**-7
ROW_RMS_TOL = 2.0**-7
MIN_ROW_COSINE = 0.9999
MAX_SCORE = 22.0
SQRT_SCALE = HEAD_DIM**-0.25
LARGE_BUCKET_FACTOR = 2.0

_compiled = None
_runtime_ready = False


@pytest.fixture(autouse=True)
def kernel_only(monkeypatch):
    """Fail any call that reaches the wrapper's torch branch."""

    def torch_branch(*args, **kwargs):
        raise AssertionError("flash_attention took its torch branch")

    monkeypatch.setattr(wrapper, "_torch_attention_impl", torch_branch)


def _device():
    """The device this venue runs the kernel on."""
    global _runtime_ready
    if envs.VLLM_NEURON_CPU_MODE:
        return torch.device("cpu")
    if not _runtime_ready:
        torch.classes.neuron.Runtime().initialize()
        _runtime_ready = True
    return torch.device("neuron", 0)


def _attention(q, k, v, bound_min, bound_max):
    """The tower's call."""
    return wrapper.flash_attention(
        q, k, v, scale=1.0, causal_mask=False, tp_q=True, tp_k=True,
        bound_min=bound_min, bound_max=bound_max,
    )


def _run(case):
    """Run the wrapper on this venue and return its output on the CPU."""
    global _compiled
    args = (case["q"], case["k"], case["v"], case["bound_min"], case["bound_max"])
    if envs.VLLM_NEURON_CPU_MODE:
        return _attention(*args)
    device = _device()
    if _compiled is None:
        _compiled = torch.compile(
            _attention, backend=envs.get_compile_backend_name(), fullgraph=True, dynamic=False
        )
    return _compiled(*(t.to(device) for t in args)).cpu()


def _case(q, k, v, q_tokens, images):
    """Bounds from each query row's token: its image's keys, or none for a pad row."""
    bound_min = torch.zeros(HEADS, q.shape[1], 1, dtype=torch.int32)
    bound_max = torch.zeros(HEADS, q.shape[1], 1, dtype=torch.int32)
    for start, end in images:
        rows = (q_tokens >= start) & (q_tokens < end)
        bound_min[:, rows] = start
        bound_max[:, rows] = end
    return {"q": q, "k": k, "v": v, "q_tokens": q_tokens, "images": images,
            "bound_min": bound_min, "bound_max": bound_max}


def _full(q, k, v):
    return _case(q, k, v, torch.arange(BUCKET), ((0, REAL),))


def _slice(q, k, v, gen):
    """The last of the 8 slices of 688 rows, padded to 768 query rows."""
    pad = _folded(gen, SLICE_ROWS - SLICE_REAL_ROWS)
    q_slice = torch.cat((q[:, SLICE_FIRST:SLICE_FIRST + SLICE_REAL_ROWS], pad), dim=1)
    tokens = torch.cat((torch.arange(SLICE_FIRST, SLICE_FIRST + SLICE_REAL_ROWS),
                        torch.full((SLICE_ROWS - SLICE_REAL_ROWS,), -1)))
    return _case(q_slice.contiguous(), k, v, tokens, ((0, REAL),))


def _shaped(shape, q, k, v, gen):
    return _full(q, k, v) if shape == "full" else _slice(q, k, v, gen)


def _normal(gen, rows):
    return torch.randn(HEADS, rows, HEAD_DIM, generator=gen).bfloat16()


def _fold(x):
    """Scale fp32 q or k rows by the square root of the softmax scale, then round once."""
    return (x.float() * SQRT_SCALE).bfloat16()


def _folded(gen, rows):
    return _fold(torch.randn(HEADS, rows, HEAD_DIM, generator=gen))


def _qkv(gen, rows):
    return _folded(gen, rows), _folded(gen, rows), _normal(gen, rows)


def _gen(n):
    return torch.Generator().manual_seed(SEED + n)


def _reference(case):
    """fp64 transcription of the reference attention at scale 1.0, one image at a time."""
    q, k, v, tokens = case["q"], case["k"], case["v"], case["q_tokens"]
    out = torch.zeros(HEADS, q.shape[1], HEAD_DIM, dtype=torch.float64)
    for start, end in case["images"]:
        rows = ((tokens >= start) & (tokens < end)).nonzero().squeeze(1)
        keys, values = k[:, start:end].double(), v[:, start:end].double()
        for chunk in rows.split(512):
            scores = q[:, chunk].double() @ keys.transpose(1, 2)
            out[:, chunk] = torch.softmax(scores, dim=-1) @ values
    return out


def _real_rows(case, images=None):
    tokens = case["q_tokens"]
    rows = torch.zeros_like(tokens, dtype=torch.bool)
    for start, end in images or case["images"]:
        rows |= (tokens >= start) & (tokens < end)
    return rows


def _assert_class_b(case, out, record_property, factor=1.0):
    rows = _real_rows(case)
    ref = _reference(case)[:, rows]
    got = out[:, rows].double()
    assert out.dtype == torch.bfloat16 and out.shape == case["q"].shape
    error = (got - ref).abs()
    bound = ELEMENT_TOL * ref.abs() + ROW_RMS_TOL * ref.pow(2).mean(-1, keepdim=True).sqrt()
    excess = error / bound
    worst = torch.nan_to_num(excess, nan=math.inf).argmax()
    head, row, channel = (int(i) for i in torch.unravel_index(worst, excess.shape))
    token = int(case["q_tokens"][rows][row])
    record_property("worst_error_over_bound", float(excess[head, row, channel]))
    record_property("worst_at_head_token_channel", f"{head},{token},{channel}")
    record_property("elements_over_bound", int((~(error <= bound)).sum()))
    over = ~(error <= factor * bound)
    record_property("bound_factor", factor)
    record_property("elements_over_bound_times_factor", int(over.sum()))
    assert not over.any(), (
        f"{int(over.sum())} elements outside {factor} times the bound; worst "
        f"{float(excess[head, row, channel]):.3f} of the bound at head {head}, token {token}, "
        f"channel {channel}"
    )
    cosine = F.cosine_similarity(got, ref, dim=-1)
    record_property("min_row_cosine", float(cosine.min()))
    assert float(cosine.min()) >= MIN_ROW_COSINE, f"min row cosine {float(cosine.min()):.6f}"


def _bits(x):
    return x.view(torch.int16)


@pytest.mark.parametrize(
    "shape, expected",
    [((HEADS, BUCKET, HEAD_DIM), True), ((HEADS, BUCKET, HEAD_DIM), True),
     ((HEADS, LARGE_BUCKET, HEAD_DIM), True), ((HEADS, BUCKET, HEAD_DIM + 1), False)],
    ids=["full", "slice_v", "large", "head_129"],
)
def test_kernel_check(shape, expected):
    """The wrapper picks the kernel at the tower's shapes and refuses a head of 129."""
    v = torch.empty(shape, dtype=torch.bfloat16, device=_device())
    assert wrapper._can_use_flash_attention_kernel(v) is expected


@pytest.mark.parametrize("shape, n", [("full", 1), ("slice", 2)])
def test_correlated_rows(shape, n, record_property):
    """Correlated q and k with one channel at 100x, scaled to the largest measured score."""
    gen = _gen(n)
    u = torch.randn(HEADS, 1, HEAD_DIM, generator=gen)

    def rows():
        x = 0.9 * u + 0.1 * torch.randn(HEADS, BUCKET, HEAD_DIM, generator=gen)
        x[..., 37] *= 100.0
        return x

    q, k = rows(), rows()
    peak = max(float((q[h, :REAL].double() @ k[h, :REAL].double().t()).max()) for h in range(HEADS))
    scale = math.sqrt(MAX_SCORE * math.sqrt(HEAD_DIM) / peak)
    q, k = _fold(q * scale), _fold(k * scale)
    case = _shaped(shape, q, k, _normal(gen, BUCKET), gen)
    _assert_class_b(case, _run(case), record_property)


@pytest.mark.parametrize("shape, n", [("full", 3), ("slice", 4)])
def test_repeated_tokens(shape, n, record_property):
    """One token's q, k and v on every real row."""
    gen = _gen(n)
    q, k, v = _qkv(gen, BUCKET)
    for x, draw in ((q, _folded), (k, _folded), (v, _normal)):
        x[:, :REAL] = draw(gen, 1)
    case = _shaped(shape, q, k, v, gen)
    _assert_class_b(case, _run(case), record_property)


@pytest.mark.parametrize("shape, n", [("full", 5), ("slice", 6)])
def test_equal_scores(shape, n, record_property):
    """One key on every real row, so each query's scores are all equal."""
    gen = _gen(n)
    q, k, v = _qkv(gen, BUCKET)
    k[:, :REAL] = _folded(gen, 1)
    case = _shaped(shape, q, k, v, gen)
    _assert_class_b(case, _run(case), record_property)


def test_largest_bucket(record_property):
    """The 33,536-patch bucket with the largest drawn image, at twice the element bound."""
    gen = _gen(7)
    q, k, v = _qkv(gen, LARGE_BUCKET)
    case = _case(q, k, v, torch.arange(LARGE_BUCKET), ((0, LARGE_REAL),))
    _assert_class_b(case, _run(case), record_property, factor=LARGE_BUCKET_FACTOR)


@pytest.mark.parametrize("shape, n, token", [("full", 8, 1000), ("slice", 9, 4916)])
def test_nan_query_stays_in_its_row(shape, n, token):
    """A NaN query gives a NaN row and leaves every other row's bits."""
    gen = _gen(n)
    q, k, v = _qkv(gen, BUCKET)
    case = _shaped(shape, q, k, v, gen)
    clean = _run(case)
    row = int((case["q_tokens"] == token).nonzero())
    case["q"] = case["q"].clone()
    case["q"][:, row] = float("nan")
    out = _run(case)
    others = torch.arange(out.shape[1]) != row
    assert torch.isnan(out[:, row]).all(), "the NaN query's row is not all NaN"
    assert torch.equal(_bits(out[:, others]), _bits(clean[:, others])), (
        "a NaN query changed another row"
    )


@pytest.mark.parametrize("shape, n, token", [("full", 10, 1000), ("slice", 11, 4916)])
def test_nan_token_reaches_its_image(shape, n, token):
    """A NaN token's key and value reach every query of its image."""
    gen = _gen(n)
    q, k, v = _qkv(gen, BUCKET)
    for x in (q, k, v):
        x[:, token] = float("nan")
    case = _shaped(shape, q, k, v, gen)
    out = _run(case)
    assert torch.isnan(out[:, _real_rows(case)]).all(), "a query of the image is not all NaN"


def test_two_packed_images(record_property):
    """Two images in one bucket, each against its own keys."""
    gen = _gen(12)
    q, k, v = _qkv(gen, BUCKET)
    case = _case(q, k, v, torch.arange(BUCKET), PACKED)
    _assert_class_b(case, _run(case), record_property)


def test_nan_token_reaches_first_image_alone():
    """A NaN token near the first packed image's last key tile, in a call of that image alone."""
    gen = _gen(13)
    q, k, v = _qkv(gen, BUCKET)
    for x in (q, k, v):
        x[:, 2280] = float("nan")
    case = _case(q, k, v, torch.arange(BUCKET), PACKED[:1])
    out = _run(case)
    assert torch.isnan(out[:, _real_rows(case)]).all(), "a query of the image is not all NaN"


@pytest.mark.parametrize("shape, n", [("full", 14), ("slice", 15)])
def test_pad_queries_finite(shape, n):
    """Pad query rows with an empty bound give finite outputs."""
    gen = _gen(n)
    q, k, v = _qkv(gen, BUCKET)
    case = _shaped(shape, q, k, v, gen)
    out = _run(case)
    assert torch.isfinite(out[:, ~_real_rows(case)]).all(), "a pad query row is not finite"


@pytest.mark.parametrize("shape, n", [("full", 16), ("slice", 17)])
def test_pad_keys_change_nothing(shape, n):
    """Real rows keep their bits when the pad keys and values change."""
    gen = _gen(n)
    q, k, v = _qkv(gen, BUCKET)
    case = _shaped(shape, q, k, v, gen)
    first = _run(case)
    for name, scale in (("k", SQRT_SCALE), ("v", 1.0)):
        case[name] = case[name].clone()
        pad = 100.0 * torch.randn(HEADS, BUCKET - REAL, HEAD_DIM, generator=gen)
        case[name][:, REAL:] = (pad * scale).bfloat16()
    second = _run(case)
    rows = _real_rows(case)
    assert torch.equal(_bits(second[:, rows]), _bits(first[:, rows])), (
        "a pad key changed a real row"
    )
