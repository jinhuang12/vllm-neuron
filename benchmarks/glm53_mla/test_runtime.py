import json

import pytest

from .runtime import summarize_trace


def event(phase, tracking, launch, core, time):
    return {"event_type": "nc_exec_running", "tracking_id": tracking,
            "phase": phase, "data": {"exec_id": launch, "device_core_idx": core,
            "nc_timestamp_ns": time}}


def test_timing_groups_cores_by_launch_without_common_clock_assumption():
    events = [event("start", 1, 7, 0, 100), event("start", 2, 7, 1, 3000),
              event("stop", 1, 7, 0, 2100), event("stop", 2, 7, 1, 6000),
              event("start", 3, 8, 0, 10000), event("stop", 3, 8, 0, 14000)]
    result = summarize_trace(json.dumps({"events": events}), 2)
    assert result["launch_max_core_us"] == [3., 4.]
    assert result["mean_us"] == 3.5
    assert result["core_events_per_launch"] == [2, 1]


def test_missing_stop_or_missing_launch_rejected():
    with pytest.raises(ValueError, match="Incomplete"):
        summarize_trace(json.dumps({"events": [event("start", 1, 7, 0, 100)]}), 1)
    with pytest.raises(ValueError, match="Incomplete"):
        summarize_trace(json.dumps({"events": []}), 1)
