"""Pipeline-level tests for shipped example images."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from metrics_petri.pipelinesam import cli, pipeline

EXAMPLE_DIR = Path("input_images/06FEB")
EXAMPLE_IMAGES = [
    "20260210_P001_06-FEB_WT_PCBM_SUB_d04_TOP.jpg",
    "20260212_P001_06-FEB_WT_PCBM_SUB_d06_TOP.jpg",
    "20260216_P001_06-FEB_WT_PCBM_SUB_d10_TOP.jpg",
]


class _FullMaskModel:
    """Tiny deterministic model replacement for exercising image-processing code."""

    def __call__(self, x):
        import torch

        return torch.ones((1, 1, 256, 256), dtype=torch.float32)


def test_example_images_are_shipped():
    assert sorted(path.name for path in EXAMPLE_DIR.glob("*.jpg")) == EXAMPLE_IMAGES


@pytest.mark.filterwarnings("ignore:Parameter `min_size` is deprecated:FutureWarning")
@pytest.mark.filterwarnings("ignore:`RegionProperties.equivalent_diameter`:FutureWarning")
def test_real_example_image_runs_full_processing_path(monkeypatch):
    """Run one real example image through detection, metrics, and overlay generation."""
    monkeypatch.setattr(pipeline, "_model", _FullMaskModel())
    image_path = EXAMPLE_DIR / EXAMPLE_IMAGES[0]

    row, overlays = cli._process_image(
        image_path,
        {"image_path": image_path.name, "day_code": "d04"},
        threshold=0.5,
        dish_size_mm=90.0,
    )

    assert row["image_path"] == image_path.name
    assert row["area_mm2"] > 0
    assert row["diameter_mm"] > 0
    assert row["perimeter_mm"] > 0
    assert set(overlays) == {
        f"{image_path.stem}_raw_dish.jpg",
        f"{image_path.stem}_mask.jpg",
        f"{image_path.stem}_colony.jpg",
        f"{image_path.stem}_cracks.jpg",
    }


def test_three_example_images_write_expected_archive(monkeypatch, tmp_path):
    """Exercise run_batch archive writing for all shipped example images."""
    output_zip = tmp_path / "analysis.zip"

    def fake_process_image(img_path, meta, threshold, dish_size_mm):
        return (
            {
                **meta,
                "area_mm2": 10.0,
                "diameter_mm": 5.0,
                "threshold_seen": threshold,
                "dish_size_seen": dish_size_mm,
            },
            {},
        )

    monkeypatch.setattr(cli, "_process_image", fake_process_image)
    monkeypatch.setattr(
        cli,
        "_write_charts",
        lambda df, out_dir: out_dir.mkdir(parents=True, exist_ok=True),
    )

    cli.run_batch(EXAMPLE_DIR, output_zip, threshold=0.5, dish_size_mm=90.0, seed=0)

    with zipfile.ZipFile(output_zip) as zf:
        names = set(zf.namelist())
        assert {"analysis_full.csv", "analysis_full.json", "provenance.json"} <= names
        rows = json.loads(zf.read("analysis_full.json"))
        provenance = json.loads(zf.read("provenance.json"))
        with zf.open("analysis_full.csv") as csv_file:
            df = pd.read_csv(csv_file)

    assert [row["image_path"] for row in rows] == EXAMPLE_IMAGES
    assert provenance["settings"] == {"threshold": 0.5, "dish_size_mm": 90.0, "seed": 0}
    assert len(df) == 3


def test_batch_and_gui_pipeline_constants_match():
    from metrics_petri.pipeline import analysis as gui_analysis
    from metrics_petri.pipelinesam import pipeline as batch_pipeline

    assert batch_pipeline.IMAGE_SIZE == gui_analysis.IMAGE_SIZE == 256
    assert batch_pipeline.CONTAINER_MM == gui_analysis.CONTAINER_MM == 90.0
