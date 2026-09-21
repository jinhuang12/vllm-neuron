"""The scheduler's initialized row names the class that initialized.

    python -m pytest -s -rA test/vllm_neuron/test_scheduler_init_row.py

Both scheduler classes are read through the one row their constructor logs, and the module
source is read for the call site and for the absence of a fixed class name in that row.
"""

from __future__ import annotations

import inspect
import logging

import pytest

from vllm_neuron.vllm.core import scheduler as scheduler_module
from vllm_neuron.vllm.core.scheduler import NeuronAsyncScheduler, NeuronScheduler

pytestmark = [pytest.mark.fast]


def _rows_logged_by(cls) -> list[str]:
    """The INFO rows the initialized-row method logs for an instance of ``cls`` built without its constructor."""
    rows: list[str] = []

    class _Rows(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            rows.append(record.getMessage())

    logger = logging.getLogger(scheduler_module.__name__)
    handler = _Rows(level=logging.INFO)
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        cls._log_initialized(cls.__new__(cls))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
    return rows


@pytest.mark.parametrize("cls", [NeuronScheduler, NeuronAsyncScheduler], ids=lambda cls: cls.__name__)
def test_the_initialized_row_names_the_class_that_initialized(cls):
    """The row carries the instance's own class name, for the sync class and for the async class."""
    rows = _rows_logged_by(cls)
    print(f"SCHEDROW|{cls.__name__}|rows={rows}")
    assert rows == [f"Initialized {cls.__name__} for Neuron platform"], rows


def test_the_constructor_logs_the_row_through_the_method():
    """``__init__`` calls the method once, and no row in the module spells a class name out."""
    calls = inspect.getsource(NeuronScheduler.__init__).count("self._log_initialized()")
    module_source = inspect.getsource(scheduler_module)
    spelled_out = sum(
        module_source.count(f"Initialized {name} for Neuron platform")
        for name in ("NeuronScheduler", "NeuronAsyncScheduler")
    )
    print(f"SCHEDROW|source|calls={calls}|spelled_out_rows={spelled_out}")
    assert calls == 1 and spelled_out == 0
