"""Tests for day-code assignment in image_metadata_gui."""
import pytest

# build_day_code_map is a pure-Python function with no GUI dependencies;
# tkinter is only imported at the top of the module, so we skip if unavailable.
try:
    from metrics_petri.pipelinesam.image_metadata_gui import build_day_code_map
    _HAS_TK = True
except ImportError:
    _HAS_TK = False

pytestmark = pytest.mark.skipif(not _HAS_TK, reason="tkinter not available in this environment")


def test_day_code_calendar_delta():
    result = build_day_code_map({"img1.jpg": "2026-02-10"}, "2026-02-06")
    assert result["2026-02-10"] == "d04"
    assert result["2026-02-06"] == "d00"


def test_day_code_exp_date_is_d00():
    result = build_day_code_map({"img.jpg": "2026-02-06"}, "2026-02-06")
    assert result["2026-02-06"] == "d00"


def test_day_code_before_exp_clamped():
    result = build_day_code_map({"img.jpg": "2026-02-01"}, "2026-02-06")
    assert result["2026-02-01"] == "d00"


def test_day_code_no_exp_sequential():
    result = build_day_code_map({"a.jpg": "2026-02-01", "b.jpg": "2026-02-03"}, "")
    assert result["2026-02-01"] == "d01"
    assert result["2026-02-03"] == "d02"


def test_empty_dates_returns_empty():
    result = build_day_code_map({}, "2026-02-06")
    assert result == {}
