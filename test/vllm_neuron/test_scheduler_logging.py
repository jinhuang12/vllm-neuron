"""The row each Neuron scheduler class logs once it has initialized."""

from __future__ import annotations

import logging

import pytest

from vllm_neuron.vllm.core import scheduler as scheduler_module
from vllm_neuron.vllm.core.scheduler import NeuronAsyncScheduler, NeuronScheduler

pytestmark = [pytest.mark.fast]


def _rows_logged_by(cls) -> list[str]:
    """The INFO rows ``cls._log_initialized`` emits for a bare instance."""
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
        # __new__ rather than the constructor: a real instance needs a full
        # VllmConfig, and the row is the only subject here.
        cls._log_initialized(cls.__new__(cls))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
    return rows


@pytest.mark.parametrize(
    "cls", [NeuronScheduler, NeuronAsyncScheduler], ids=lambda cls: cls.__name__
)
def test_the_initialized_row_names_the_class_that_initialized(cls):
    """The row carries the instance's own class name, not a fixed one."""
    rows = _rows_logged_by(cls)
    assert rows == [f"Initialized {cls.__name__} for Neuron platform"], rows
