"""Tests for pure-Python metric constants and utilities."""
import importlib
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest


def test_container_mm_constant():
    from metrics_petri.pipelinesam.pipeline import CONTAINER_MM
    assert CONTAINER_MM == 90.0


def test_pipelinesam_container_mm_matches_pipeline():
    from metrics_petri.pipeline.analysis import CONTAINER_MM as GUI_MM
    from metrics_petri.pipelinesam.pipeline import CONTAINER_MM as SAM_MM
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


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_device_selection(
    monkeypatch, cuda_available, mps_available, expected
):
    from metrics_petri import _model

    monkeypatch.setattr(_model.torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(_model.torch.backends.mps, "is_available", lambda: mps_available)

    assert _model._select_device() == expected


def test_pipelines_use_shared_device_selection():
    from metrics_petri._model import _select_device
    from metrics_petri.pipeline.analysis import DEVICE as GUI_DEVICE
    from metrics_petri.pipelinesam.pipeline import DEVICE as BATCH_DEVICE

    assert GUI_DEVICE == _select_device()
    assert BATCH_DEVICE == _select_device()


def test_smallunet_reexport_pipeline():
    from metrics_petri._model import SmallUNet as B
    from metrics_petri.pipeline.model import SmallUNet as A
    assert A is B


def test_smallunet_reexport_pipelinesam():
    from metrics_petri._model import SmallUNet as B
    from metrics_petri.pipelinesam.model_small_unet import SmallUNet as A
    assert A is B


@pytest.mark.parametrize(
    ("module_name", "loader_name"),
    [
        ("metrics_petri.pipeline.analysis", "load_model"),
        ("metrics_petri.pipelinesam.pipeline", "get_model"),
    ],
)
def test_model_loaders_use_weights_only(module_name, loader_name, monkeypatch):
    module = importlib.import_module(module_name)
    real_torch_load = module.torch.load
    load_options = {}

    def checked_torch_load(*args, **kwargs):
        load_options.update(kwargs)
        return real_torch_load(*args, **kwargs)

    module._model = None
    monkeypatch.setattr(module.torch, "load", checked_torch_load)
    getattr(module, loader_name)()

    assert load_options["weights_only"] is True
    module._model = None


def test_pipelinesam_import_does_not_resolve_model_path(tmp_path):
    code = """
from metrics_petri import _paths

def fail_if_called():
    raise AssertionError("model path resolved during import")

_paths._find_model_path = fail_if_called
import metrics_petri.pipelinesam.pipeline
"""
    env = {**os.environ, "MPLCONFIGDIR": str(tmp_path)}

    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_pipelinesam_import_does_not_suppress_future_warnings(tmp_path):
    code = """
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", FutureWarning)
    import metrics_petri.pipelinesam.pipeline
    warnings.warn("future warning remains visible", FutureWarning)

assert any(str(item.message) == "future warning remains visible" for item in caught)
"""
    env = {**os.environ, "MPLCONFIGDIR": str(tmp_path)}

    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_legacy_model_path_is_resolved_lazily(monkeypatch, tmp_path):
    module = importlib.import_module("metrics_petri.pipelinesam.pipeline")
    model_path = Path(tmp_path, "legacy-model.pt")
    monkeypatch.setattr(module, "_find_model_path", lambda: model_path)

    with pytest.warns(DeprecationWarning, match="MODEL_PATH is deprecated"):
        resolved = module.MODEL_PATH

    assert resolved == model_path


def test_version_consistent():
    from importlib.metadata import version

    import metrics_petri
    assert metrics_petri.__version__ == version("metrics-petri")
