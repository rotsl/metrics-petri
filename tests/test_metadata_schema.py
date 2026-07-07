"""Tests for CSV and JSON metadata structural validation."""

import json

import pandas as pd
import pytest

from metrics_petri.pipelinesam.cli import _load_metadata


def test_csv_metadata_preserves_custom_columns_and_fills_optional_columns(tmp_path):
    path = tmp_path / "metadata.csv"
    pd.DataFrame(
        [{"image_path": "day01/image.jpg", "custom_group": "control"}]
    ).to_csv(path, index=False)

    metadata = _load_metadata(path)

    assert metadata.loc[0, "custom_group"] == "control"
    assert metadata.loc[0, "experiment_name"] == ""


def test_json_metadata_accepts_list_of_objects(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps([{"image_path": "day01/image.jpg"}]), encoding="utf-8"
    )

    metadata = _load_metadata(path)

    assert metadata.loc[0, "image_path"] == "day01/image.jpg"


@pytest.mark.parametrize("records", [{"image_path": "image.jpg"}, ["image.jpg"]])
def test_json_metadata_rejects_non_record_structures(tmp_path, records):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError, match="must be a list of objects"):
        _load_metadata(path)


def test_metadata_requires_image_path_column(tmp_path):
    path = tmp_path / "metadata.csv"
    pd.DataFrame([{"image_date": "2026-07-06"}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required column 'image_path'"):
        _load_metadata(path)


@pytest.mark.parametrize("image_path", ["", "   ", None])
def test_metadata_rejects_blank_image_paths(tmp_path, image_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps([{"image_path": image_path}]), encoding="utf-8")

    with pytest.raises(ValueError, match="blank image_path"):
        _load_metadata(path)


def test_empty_json_metadata_remains_structurally_valid(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text("[]", encoding="utf-8")

    metadata = _load_metadata(path)

    assert metadata.empty
    assert "image_path" in metadata.columns
