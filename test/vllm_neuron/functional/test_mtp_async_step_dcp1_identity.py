# SPDX-License-Identifier: Apache-2.0
"""``mtp_async_correct`` at DCP 1 is the b5b516b3 correction, value for value.

The DCP slots (``test_mtp_async_step_dcp.py``) add two keywords to the correction, ``dcp_size``
and ``cp_rank``; at ``dcp_size == 1`` every field must be what the base tree built. Two
references:

* ``FROZEN``: 13 cases captured on the untouched b5b516b3 tree before the change, both routes
  (the torch route and the NKI kernel on the simulator, asserted equal at capture), with and
  without a page table. The cases: kept counts on both sides of a page edge, the first
  verify step after a prefill, the one-row step after a verify step, 2 / 4 / 64 requests,
  bucket padding, two blocks spanned with the next page unallocated, and the served
  column height (``max_model_len`` 131072 over pages of 128 = 1024 pages). Each case's
  tables are rebuilt from its ``seed`` / ``allocated`` / ``window`` by :func:`_tables`
  (a formula, no RNG). Generator: ``reports/mtp-dcp/async/RECORD.md`` names it and its md5.
* the b5b516b3 module itself, loaded from git (``reference_at_commit``), on the DCP file's
  DCP-1 grid.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/functional/test_mtp_async_step_dcp1_identity.py
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.mtp import async_step

from test.vllm_neuron.functional.reference_at_commit import load_reference, needs_reference

BASE = "b5b516b3"
MODULE = "vllm_neuron/functional/mtp/async_step.py"
PAGE = 128
T = 4
NULL = 0


def _tables(seed: int, batch: int, pages: int) -> list[list[int]]:
    """Distinct non-null block ids by formula (no RNG): 4093 is prime and 37 is a unit mod it."""
    ids = [1 + (seed * 131 + i * 37) % 4093 for i in range(batch * pages)]
    return [ids[r * pages:(r + 1) * pages] for r in range(batch)]


def _column(tables: list[list[int]], window: int, padding: int) -> torch.Tensor:
    rows = [list(row) for row in tables] + [[NULL]] * padding
    return torch.tensor([[row[p] if p < len(row) else -1 for row in rows] for p in range(window)],
                        dtype=torch.int32)


def _operands(case: dict):
    inputs = case["inputs"]
    batch = len(inputs["starts"])
    tables = _tables(inputs["seed"], batch, inputs["allocated"])
    return (
        torch.tensor(inputs["prev_rows"], dtype=torch.int32),
        torch.tensor(inputs["starts"], dtype=torch.int32),
        _column(tables, inputs["window"], inputs["padding"]),
        {key: inputs[key] for key in ("prev_width", "width", "page_size", "padding")},
    )


def _as_frozen(correction) -> dict:
    return {
        name: None if getattr(correction, name) is None
        else (getattr(correction, name).tolist(), str(getattr(correction, name).dtype))
        for name in correction._fields
    }


# ── captured at b5b516b3 by the generator RECORD.md names; do not edit by hand ──────────
FROZEN_AT = 'b5b516b3a9bcaa9fc9dca5cb478ffd03a6f3d077'
FROZEN = {'first_verify_after_prefill': {'inputs': {'allocated': 4,
                                           'padding': 0,
                                           'page_size': 128,
                                           'prev_rows': [0],
                                           'prev_width': 1,
                                           'seed': 12,
                                           'starts': [37],
                                           'width': 4,
                                           'window': 4},
                                'outputs': {'checkpoint_rows': ([0], 'torch.int32'),
                                            'latent_slots': ([201381, 201382, 201383, 201384], 'torch.int64'),
                                            'linear_start': ([37], 'torch.int32'),
                                            'seq_lens': ([38, 39, 40, 41], 'torch.int32'),
                                            'start_position': ([37], 'torch.int32')},
                                'outputs_no_table': {'checkpoint_rows': ([0], 'torch.int32'),
                                                     'latent_slots': None,
                                                     'linear_start': ([37], 'torch.int32'),
                                                     'seq_lens': ([38, 39, 40, 41], 'torch.int32'),
                                                     'start_position': ([37], 'torch.int32')}},
 'four_requests': {'inputs': {'allocated': 4,
                              'padding': 0,
                              'page_size': 128,
                              'prev_rows': [0, 1, 2, 3],
                              'prev_width': 4,
                              'seed': 15,
                              'starts': [5, 127, 128, 382],
                              'width': 4,
                              'window': 4},
                   'outputs': {'checkpoint_rows': ([0, 1, 2, 3], 'torch.int32'),
                               'latent_slots': ([251650, 251651, 251652, 251653, 270717, 270718, 270719, 275328, 289663,
                                                 294272, 294273, 294274, 318078, 318079, 322688, 322689],
                                                'torch.int64'),
                               'linear_start': ([2, 125, 127, 382], 'torch.int32'),
                               'seq_lens': ([3, 4, 5, 6, 126, 127, 128, 129, 128, 129, 130, 131, 383, 384, 385, 386],
                                            'torch.int32'),
                               'start_position': ([2, 125, 127, 382], 'torch.int32')},
                   'outputs_no_table': {'checkpoint_rows': ([0, 1, 2, 3], 'torch.int32'),
                                        'latent_slots': None,
                                        'linear_start': ([2, 125, 127, 382], 'torch.int32'),
                                        'seq_lens': ([3, 4, 5, 6, 126, 127, 128, 129, 128, 129, 130, 131, 383, 384, 385,
                                                      386],
                                                     'torch.int32'),
                                        'start_position': ([2, 125, 127, 382], 'torch.int32')}},
 'one_row_after_verify': {'inputs': {'allocated': 4,
                                     'padding': 0,
                                     'page_size': 128,
                                     'prev_rows': [1],
                                     'prev_width': 4,
                                     'seed': 13,
                                     'starts': [257],
                                     'width': 1,
                                     'window': 4},
                          'outputs': {'checkpoint_rows': ([1], 'torch.int32'),
                                      'latent_slots': ([222975], 'torch.int64'),
                                      'linear_start': ([255], 'torch.int32'),
                                      'seq_lens': ([256], 'torch.int32'),
                                      'start_position': ([255], 'torch.int32')},
                          'outputs_no_table': {'checkpoint_rows': ([1], 'torch.int32'),
                                               'latent_slots': None,
                                               'linear_start': ([255], 'torch.int32'),
                                               'seq_lens': ([256], 'torch.int32'),
                                               'start_position': ([255], 'torch.int32')}},
 'served_window_1024': {'inputs': {'allocated': 1016,
                                   'padding': 0,
                                   'page_size': 128,
                                   'prev_rows': [2],
                                   'prev_width': 4,
                                   'seed': 19,
                                   'starts': [130000],
                                   'width': 4,
                                   'window': 1024},
                        'outputs': {'checkpoint_rows': ([2], 'torch.int32'),
                                    'latent_slots': ([410703, 410704, 410705, 410706], 'torch.int64'),
                                    'linear_start': ([129999], 'torch.int32'),
                                    'seq_lens': ([130000, 130001, 130002, 130003], 'torch.int32'),
                                    'start_position': ([129999], 'torch.int32')},
                        'outputs_no_table': {'checkpoint_rows': ([2], 'torch.int32'),
                                             'latent_slots': None,
                                             'linear_start': ([129999], 'torch.int32'),
                                             'seq_lens': ([130000, 130001, 130002, 130003], 'torch.int32'),
                                             'start_position': ([129999], 'torch.int32')}},
 'sixty_four_requests': {'inputs': {'allocated': 4,
                                    'padding': 0,
                                    'page_size': 128,
                                    'prev_rows': [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2,
                                                  3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
                                                  2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                                    'prev_width': 4,
                                    'seed': 16,
                                    'starts': [132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132,
                                               132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132,
                                               132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132,
                                               132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132, 132,
                                               132, 132, 132, 132, 132, 132, 132, 132],
                                    'width': 4,
                                    'window': 4},
                         'outputs': {'checkpoint_rows': ([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0,
                                                          1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
                                                          2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2,
                                                          3],
                                                         'torch.int32'),
                                     'latent_slots': ([273153, 273154, 273155, 273156, 292098, 292099, 292100, 292101,
                                                       311043, 311044, 311045, 311046, 329988, 329989, 329990, 329991,
                                                       348929, 348930, 348931, 348932, 367874, 367875, 367876, 367877,
                                                       386819, 386820, 386821, 386822, 405764, 405765, 405766, 405767,
                                                       424705, 424706, 424707, 424708, 443650, 443651, 443652, 443653,
                                                       462595, 462596, 462597, 462598, 481540, 481541, 481542, 481543,
                                                       500481, 500482, 500483, 500484, 519426, 519427, 519428, 519429,
                                                       14467, 14468, 14469, 14470, 33412, 33413, 33414, 33415, 52353,
                                                       52354, 52355, 52356, 71298, 71299, 71300, 71301, 90243, 90244,
                                                       90245, 90246, 109188, 109189, 109190, 109191, 128129, 128130,
                                                       128131, 128132, 147074, 147075, 147076, 147077, 166019, 166020,
                                                       166021, 166022, 184964, 184965, 184966, 184967, 203905, 203906,
                                                       203907, 203908, 222850, 222851, 222852, 222853, 241795, 241796,
                                                       241797, 241798, 260740, 260741, 260742, 260743, 279681, 279682,
                                                       279683, 279684, 298626, 298627, 298628, 298629, 317571, 317572,
                                                       317573, 317574, 336516, 336517, 336518, 336519, 355457, 355458,
                                                       355459, 355460, 374402, 374403, 374404, 374405, 393347, 393348,
                                                       393349, 393350, 412292, 412293, 412294, 412295, 431233, 431234,
                                                       431235, 431236, 450178, 450179, 450180, 450181, 469123, 469124,
                                                       469125, 469126, 488068, 488069, 488070, 488071, 507009, 507010,
                                                       507011, 507012, 2050, 2051, 2052, 2053, 20995, 20996, 20997,
                                                       20998, 39940, 39941, 39942, 39943, 58881, 58882, 58883, 58884,
                                                       77826, 77827, 77828, 77829, 96771, 96772, 96773, 96774, 115716,
                                                       115717, 115718, 115719, 134657, 134658, 134659, 134660, 153602,
                                                       153603, 153604, 153605, 172547, 172548, 172549, 172550, 191492,
                                                       191493, 191494, 191495, 210433, 210434, 210435, 210436, 229378,
                                                       229379, 229380, 229381, 248323, 248324, 248325, 248326, 267268,
                                                       267269, 267270, 267271, 286209, 286210, 286211, 286212, 305154,
                                                       305155, 305156, 305157, 324099, 324100, 324101, 324102, 343044,
                                                       343045, 343046, 343047, 361985, 361986, 361987, 361988, 380930,
                                                       380931, 380932, 380933, 399875, 399876, 399877, 399878, 418820,
                                                       418821, 418822, 418823],
                                                      'torch.int64'),
                                     'linear_start': ([129, 130, 131, 132, 129, 130, 131, 132, 129, 130, 131, 132, 129,
                                                       130, 131, 132, 129, 130, 131, 132, 129, 130, 131, 132, 129, 130,
                                                       131, 132, 129, 130, 131, 132, 129, 130, 131, 132, 129, 130, 131,
                                                       132, 129, 130, 131, 132, 129, 130, 131, 132, 129, 130, 131, 132,
                                                       129, 130, 131, 132, 129, 130, 131, 132, 129, 130, 131, 132],
                                                      'torch.int32'),
                                     'seq_lens': ([130, 131, 132, 133, 131, 132, 133, 134, 132, 133, 134, 135, 133, 134,
                                                   135, 136, 130, 131, 132, 133, 131, 132, 133, 134, 132, 133, 134, 135,
                                                   133, 134, 135, 136, 130, 131, 132, 133, 131, 132, 133, 134, 132, 133,
                                                   134, 135, 133, 134, 135, 136, 130, 131, 132, 133, 131, 132, 133, 134,
                                                   132, 133, 134, 135, 133, 134, 135, 136, 130, 131, 132, 133, 131, 132,
                                                   133, 134, 132, 133, 134, 135, 133, 134, 135, 136, 130, 131, 132, 133,
                                                   131, 132, 133, 134, 132, 133, 134, 135, 133, 134, 135, 136, 130, 131,
                                                   132, 133, 131, 132, 133, 134, 132, 133, 134, 135, 133, 134, 135, 136,
                                                   130, 131, 132, 133, 131, 132, 133, 134, 132, 133, 134, 135, 133, 134,
                                                   135, 136, 130, 131, 132, 133, 131, 132, 133, 134, 132, 133, 134, 135,
                                                   133, 134, 135, 136, 130, 131, 132, 133, 131, 132, 133, 134, 132, 133,
                                                   134, 135, 133, 134, 135, 136, 130, 131, 132, 133, 131, 132, 133, 134,
                                                   132, 133, 134, 135, 133, 134, 135, 136, 130, 131, 132, 133, 131, 132,
                                                   133, 134, 132, 133, 134, 135, 133, 134, 135, 136, 130, 131, 132, 133,
                                                   131, 132, 133, 134, 132, 133, 134, 135, 133, 134, 135, 136, 130, 131,
                                                   132, 133, 131, 132, 133, 134, 132, 133, 134, 135, 133, 134, 135, 136,
                                                   130, 131, 132, 133, 131, 132, 133, 134, 132, 133, 134, 135, 133, 134,
                                                   135, 136, 130, 131, 132, 133, 131, 132, 133, 134, 132, 133, 134, 135,
                                                   133, 134, 135, 136],
                                                  'torch.int32'),
                                     'start_position': ([129, 130, 131, 132, 129, 130, 131, 132, 129, 130, 131, 132,
                                                         129, 130, 131, 132, 129, 130, 131, 132, 129, 130, 131, 132,
                                                         129, 130, 131, 132, 129, 130, 131, 132, 129, 130, 131, 132,
                                                         129, 130, 131, 132, 129, 130, 131, 132, 129, 130, 131, 132,
                                                         129, 130, 131, 132, 129, 130, 131, 132, 129, 130, 131, 132,
                                                         129, 130, 131, 132],
                                                        'torch.int32')},
                         'outputs_no_table': {'checkpoint_rows': ([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
                                                                   2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                                                                   0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
                                                                   2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                                                                  'torch.int32'),
                                              'latent_slots': None,
                                              'linear_start': ([129, 130, 131, 132, 129, 130, 131, 132, 129, 130, 131,
                                                                132, 129, 130, 131, 132, 129, 130, 131, 132, 129, 130,
                                                                131, 132, 129, 130, 131, 132, 129, 130, 131, 132, 129,
                                                                130, 131, 132, 129, 130, 131, 132, 129, 130, 131, 132,
                                                                129, 130, 131, 132, 129, 130, 131, 132, 129, 130, 131,
                                                                132, 129, 130, 131, 132, 129, 130, 131, 132],
                                                               'torch.int32'),
                                              'seq_lens': ([130, 131, 132, 133, 131, 132, 133, 134, 132, 133, 134, 135,
                                                            133, 134, 135, 136, 130, 131, 132, 133, 131, 132, 133, 134,
                                                            132, 133, 134, 135, 133, 134, 135, 136, 130, 131, 132, 133,
                                                            131, 132, 133, 134, 132, 133, 134, 135, 133, 134, 135, 136,
                                                            130, 131, 132, 133, 131, 132, 133, 134, 132, 133, 134, 135,
                                                            133, 134, 135, 136, 130, 131, 132, 133, 131, 132, 133, 134,
                                                            132, 133, 134, 135, 133, 134, 135, 136, 130, 131, 132, 133,
                                                            131, 132, 133, 134, 132, 133, 134, 135, 133, 134, 135, 136,
                                                            130, 131, 132, 133, 131, 132, 133, 134, 132, 133, 134, 135,
                                                            133, 134, 135, 136, 130, 131, 132, 133, 131, 132, 133, 134,
                                                            132, 133, 134, 135, 133, 134, 135, 136, 130, 131, 132, 133,
                                                            131, 132, 133, 134, 132, 133, 134, 135, 133, 134, 135, 136,
                                                            130, 131, 132, 133, 131, 132, 133, 134, 132, 133, 134, 135,
                                                            133, 134, 135, 136, 130, 131, 132, 133, 131, 132, 133, 134,
                                                            132, 133, 134, 135, 133, 134, 135, 136, 130, 131, 132, 133,
                                                            131, 132, 133, 134, 132, 133, 134, 135, 133, 134, 135, 136,
                                                            130, 131, 132, 133, 131, 132, 133, 134, 132, 133, 134, 135,
                                                            133, 134, 135, 136, 130, 131, 132, 133, 131, 132, 133, 134,
                                                            132, 133, 134, 135, 133, 134, 135, 136, 130, 131, 132, 133,
                                                            131, 132, 133, 134, 132, 133, 134, 135, 133, 134, 135, 136,
                                                            130, 131, 132, 133, 131, 132, 133, 134, 132, 133, 134, 135,
                                                            133, 134, 135, 136],
                                                           'torch.int32'),
                                              'start_position': ([129, 130, 131, 132, 129, 130, 131, 132, 129, 130, 131,
                                                                  132, 129, 130, 131, 132, 129, 130, 131, 132, 129, 130,
                                                                  131, 132, 129, 130, 131, 132, 129, 130, 131, 132, 129,
                                                                  130, 131, 132, 129, 130, 131, 132, 129, 130, 131, 132,
                                                                  129, 130, 131, 132, 129, 130, 131, 132, 129, 130, 131,
                                                                  132, 129, 130, 131, 132, 129, 130, 131, 132],
                                                                 'torch.int32')}},
 'span_two_blocks_next_unallocated': {'inputs': {'allocated': 2,
                                                 'padding': 0,
                                                 'page_size': 128,
                                                 'prev_rows': [0],
                                                 'prev_width': 4,
                                                 'seed': 17,
                                                 'starts': [129],
                                                 'width': 4,
                                                 'window': 8},
                                      'outputs': {'checkpoint_rows': ([0], 'torch.int32'),
                                                  'latent_slots': ([285310, 285311, 289920, 289921], 'torch.int64'),
                                                  'linear_start': ([126], 'torch.int32'),
                                                  'seq_lens': ([127, 128, 129, 130], 'torch.int32'),
                                                  'start_position': ([126], 'torch.int32')},
                                      'outputs_no_table': {'checkpoint_rows': ([0], 'torch.int32'),
                                                           'latent_slots': None,
                                                           'linear_start': ([126], 'torch.int32'),
                                                           'seq_lens': ([127, 128, 129, 130], 'torch.int32'),
                                                           'start_position': ([126], 'torch.int32')}},
 'span_two_blocks_next_unallocated_padded': {'inputs': {'allocated': 2,
                                                        'padding': 1,
                                                        'page_size': 128,
                                                        'prev_rows': [3],
                                                        'prev_width': 4,
                                                        'seed': 18,
                                                        'starts': [130],
                                                        'width': 4,
                                                        'window': 8},
                                             'outputs': {'checkpoint_rows': ([3, 0], 'torch.int32'),
                                                         'latent_slots': ([306690, 306691, 306692, 306693, 0, 0, 0, 0],
                                                                          'torch.int64'),
                                                         'linear_start': ([130, 1], 'torch.int32'),
                                                         'seq_lens': ([131, 132, 133, 134, 1, 2, 3, 4], 'torch.int32'),
                                                         'start_position': ([130, 0], 'torch.int32')},
                                             'outputs_no_table': {'checkpoint_rows': ([3, 0], 'torch.int32'),
                                                                  'latent_slots': None,
                                                                  'linear_start': ([130, 1], 'torch.int32'),
                                                                  'seq_lens': ([131, 132, 133, 134, 1, 2, 3, 4],
                                                                               'torch.int32'),
                                                                  'start_position': ([130, 0], 'torch.int32')}},
 'two_requests_two_padding': {'inputs': {'allocated': 4,
                                         'padding': 2,
                                         'page_size': 128,
                                         'prev_rows': [3, 0],
                                         'prev_width': 4,
                                         'seed': 14,
                                         'starts': [131, 258],
                                         'width': 4,
                                         'window': 4},
                              'outputs': {'checkpoint_rows': ([3, 0, 0, 0], 'torch.int32'),
                                          'latent_slots': ([239619, 239620, 239621, 239622, 258687, 263296, 263297,
                                                            263298, 0, 0, 0, 0, 0, 0, 0, 0],
                                                           'torch.int64'),
                                          'linear_start': ([131, 255, 1, 1], 'torch.int32'),
                                          'seq_lens': ([132, 133, 134, 135, 256, 257, 258, 259, 1, 2, 3, 4, 1, 2, 3, 4],
                                                       'torch.int32'),
                                          'start_position': ([131, 255, 0, 0], 'torch.int32')},
                              'outputs_no_table': {'checkpoint_rows': ([3, 0, 0, 0], 'torch.int32'),
                                                   'latent_slots': None,
                                                   'linear_start': ([131, 255, 1, 1], 'torch.int32'),
                                                   'seq_lens': ([132, 133, 134, 135, 256, 257, 258, 259, 1, 2, 3, 4, 1,
                                                                 2, 3, 4],
                                                                'torch.int32'),
                                                   'start_position': ([131, 255, 0, 0], 'torch.int32')}},
 'verify_at_page': {'inputs': {'allocated': 4,
                               'padding': 0,
                               'page_size': 128,
                               'prev_rows': [0],
                               'prev_width': 4,
                               'seed': 11,
                               'starts': [128],
                               'width': 4,
                               'window': 4},
                    'outputs': {'checkpoint_rows': ([0], 'torch.int32'),
                                'latent_slots': ([184701, 184702, 184703, 189312], 'torch.int64'),
                                'linear_start': ([125], 'torch.int32'),
                                'seq_lens': ([126, 127, 128, 129], 'torch.int32'),
                                'start_position': ([125], 'torch.int32')},
                    'outputs_no_table': {'checkpoint_rows': ([0], 'torch.int32'),
                                         'latent_slots': None,
                                         'linear_start': ([125], 'torch.int32'),
                                         'seq_lens': ([126, 127, 128, 129], 'torch.int32'),
                                         'start_position': ([125], 'torch.int32')}},
 'verify_kept1_page_edge': {'inputs': {'allocated': 4,
                                       'padding': 0,
                                       'page_size': 128,
                                       'prev_rows': [0],
                                       'prev_width': 4,
                                       'seed': 7,
                                       'starts': [131],
                                       'width': 4,
                                       'window': 4},
                            'outputs': {'checkpoint_rows': ([0], 'torch.int32'),
                                        'latent_slots': ([122240, 122241, 122242, 122243], 'torch.int64'),
                                        'linear_start': ([128], 'torch.int32'),
                                        'seq_lens': ([129, 130, 131, 132], 'torch.int32'),
                                        'start_position': ([128], 'torch.int32')},
                            'outputs_no_table': {'checkpoint_rows': ([0], 'torch.int32'),
                                                 'latent_slots': None,
                                                 'linear_start': ([128], 'torch.int32'),
                                                 'seq_lens': ([129, 130, 131, 132], 'torch.int32'),
                                                 'start_position': ([128], 'torch.int32')}},
 'verify_kept2_page_edge': {'inputs': {'allocated': 4,
                                       'padding': 0,
                                       'page_size': 128,
                                       'prev_rows': [1],
                                       'prev_width': 4,
                                       'seed': 8,
                                       'starts': [131],
                                       'width': 4,
                                       'window': 4},
                            'outputs': {'checkpoint_rows': ([1], 'torch.int32'),
                                        'latent_slots': ([139009, 139010, 139011, 139012], 'torch.int64'),
                                        'linear_start': ([129], 'torch.int32'),
                                        'seq_lens': ([130, 131, 132, 133], 'torch.int32'),
                                        'start_position': ([129], 'torch.int32')},
                            'outputs_no_table': {'checkpoint_rows': ([1], 'torch.int32'),
                                                 'latent_slots': None,
                                                 'linear_start': ([129], 'torch.int32'),
                                                 'seq_lens': ([130, 131, 132, 133], 'torch.int32'),
                                                 'start_position': ([129], 'torch.int32')}},
 'verify_kept3_page_edge': {'inputs': {'allocated': 4,
                                       'padding': 0,
                                       'page_size': 128,
                                       'prev_rows': [2],
                                       'prev_width': 4,
                                       'seed': 9,
                                       'starts': [131],
                                       'width': 4,
                                       'window': 4},
                            'outputs': {'checkpoint_rows': ([2], 'torch.int32'),
                                        'latent_slots': ([155778, 155779, 155780, 155781], 'torch.int64'),
                                        'linear_start': ([130], 'torch.int32'),
                                        'seq_lens': ([131, 132, 133, 134], 'torch.int32'),
                                        'start_position': ([130], 'torch.int32')},
                            'outputs_no_table': {'checkpoint_rows': ([2], 'torch.int32'),
                                                 'latent_slots': None,
                                                 'linear_start': ([130], 'torch.int32'),
                                                 'seq_lens': ([131, 132, 133, 134], 'torch.int32'),
                                                 'start_position': ([130], 'torch.int32')}},
 'verify_kept4_page_edge': {'inputs': {'allocated': 4,
                                       'padding': 0,
                                       'page_size': 128,
                                       'prev_rows': [3],
                                       'prev_width': 4,
                                       'seed': 10,
                                       'starts': [131],
                                       'width': 4,
                                       'window': 4},
                            'outputs': {'checkpoint_rows': ([3], 'torch.int32'),
                                        'latent_slots': ([172547, 172548, 172549, 172550], 'torch.int64'),
                                        'linear_start': ([131], 'torch.int32'),
                                        'seq_lens': ([132, 133, 134, 135], 'torch.int32'),
                                        'start_position': ([131], 'torch.int32')},
                            'outputs_no_table': {'checkpoint_rows': ([3], 'torch.int32'),
                                                 'latent_slots': None,
                                                 'linear_start': ([131], 'torch.int32'),
                                                 'seq_lens': ([132, 133, 134, 135], 'torch.int32'),
                                                 'start_position': ([131], 'torch.int32')}}}
# ── end of the capture ────────────────────────────────────────────────────────────────


ROUTES = {"torch": async_step.mtp_async_correct_torch, "kernel": async_step.mtp_async_correct}
#: The keywords a DCP-1 caller may hand: none (the base tree's call), or DCP 1 named.
FORMS = {"default": {}, "explicit": {"dcp_size": 1, "cp_rank": 0}}


@pytest.mark.parametrize("form", sorted(FORMS))
@pytest.mark.parametrize("route", sorted(ROUTES))
@pytest.mark.parametrize("name", sorted(FROZEN))
def test_dcp_one_reproduces_the_frozen_base_outputs(name, route, form):
    prev, optimistic, column, kwargs = _operands(FROZEN[name])
    got = ROUTES[route](prev, optimistic, column, **kwargs, **FORMS[form])
    assert _as_frozen(got) == FROZEN[name]["outputs"]
    got = ROUTES[route](prev, optimistic, None, **kwargs, **FORMS[form])
    assert _as_frozen(got) == FROZEN[name]["outputs_no_table"]


def test_the_frozen_tables_are_the_ones_the_base_saw():
    """The rebuilt tables are the capture's: the frozen slots of each one-request case are
    ``table[p // 128] * 128 + p % 128`` over its true positions."""
    for name, case in FROZEN.items():
        inputs = case["inputs"]
        if len(inputs["starts"]) != 1 or inputs["padding"]:
            continue
        (table,) = _tables(inputs["seed"], 1, inputs["allocated"])
        starts = case["outputs"]["start_position"][0]
        positions = [starts[0] + t for t in range(inputs["width"])]
        want = [table[p // PAGE] * PAGE + p % PAGE for p in positions]
        assert case["outputs"]["latent_slots"][0] == want, name


@needs_reference(BASE, MODULE)
def test_dcp_one_is_the_base_module_on_the_dcp_grid():
    """The DCP file's DCP-1 grid, both routes, against b5b516b3's module."""
    from test.vllm_neuron.functional import test_mtp_async_step_dcp as grid

    base = load_reference(BASE, MODULE, "async_step_at_b5b516b3")
    for prev_width, width in grid.WIDTHS:
        for padding in (0, 1):
            starts = grid._true_starts(1)
            _, prev, optimistic, column = grid._step(starts, dcp=1, prev_width=prev_width,
                                                     width=width, padding=padding)
            kwargs = dict(prev_width=prev_width, width=width, page_size=PAGE, padding=padding)
            for ours, theirs in (
                (async_step.mtp_async_correct_torch, base.mtp_async_correct_torch),
                (async_step.mtp_async_correct, base.mtp_async_correct),
            ):
                want = _as_frozen(theirs(prev, optimistic, column, **kwargs))
                assert _as_frozen(ours(prev, optimistic, column, **kwargs)) == want
                assert _as_frozen(ours(prev, optimistic, column, **kwargs, dcp_size=1, cp_rank=0)) == want
