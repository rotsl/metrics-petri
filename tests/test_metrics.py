"""Tests for pure-Python metric constants and utilities."""
import pytest
from datetime import date


def test_container_mm_constant():
    from metrics_petri.pipelinesam.pipeline import CONTAINER_MM
    assert CONTAINER_MM == 90.0


def test_pipelinesam_container_mm_matches_pipeline():
    from metrics_petri.pipelinesam.pipeline import CONTAINER_MM as SAM_MM
    from metrics_petri.pipeline.analysis import CONTAINER_MM as GUI_MM
    assert SAM_MM == GUI_MM


def test_parse_date_prefix_full_date():
    from metrics_petri.pipelinesam.dish_cropper import parse_date_prefix
    assert parse_date_prefix("06/02/2026") == "20260206_"


def test_parse_date_prefix_short_month_abbrev():
    from metrics_petri.pipelinesam.dish_cropper import parse_date_prefix
    # Short form uses current year — pin to a known date via today arg
    result = parse_date_prefix("06/Feb", today=date(2026, 1, 1))
    assert result == "20260206_"


def test_parse_date_prefix_empty_returns_empty():
    from metrics_petri.pipelinesam.dish_cropper import parse_date_prefix
    assert parse_date_prefix("") == ""
    assert parse_date_prefix(None) == ""


def test_parse_date_prefix_invalid_raises():
    from metrics_petri.pipelinesam.dish_cropper import parse_date_prefix
    with pytest.raises(ValueError):
        parse_date_prefix("not-a-date")


def test_smallunet_instantiates():
    torch = pytest.importorskip("torch")
    from metrics_petri._model import SmallUNet
    model = SmallUNet()
    x = torch.zeros(1, 3, 256, 256)
    out = model(x)
    assert out.shape == (1, 1, 256, 256)
    assert 0.0 <= out.min().item() <= out.max().item() <= 1.0


def test_smallunet_reexport_pipeline():
    from metrics_petri.pipeline.model import SmallUNet as A
    from metrics_petri._model import SmallUNet as B
    assert A is B


def test_smallunet_reexport_pipelinesam():
    from metrics_petri.pipelinesam.model_small_unet import SmallUNet as A
    from metrics_petri._model import SmallUNet as B
    assert A is B


def test_version_consistent():
    import metrics_petri
    from importlib.metadata import version
    assert metrics_petri.__version__ == version("metrics-petri")
