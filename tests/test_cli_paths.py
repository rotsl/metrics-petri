"""Tests for batch CLI metadata path confinement."""

from pathlib import Path

import pandas as pd
import pytest

from metrics_petri.pipelinesam.cli import (
    _build_metadata_tasks,
    _resolve_metadata_image_path,
    run_batch,
)


def test_metadata_image_path_allows_nested_path(tmp_path):
    input_dir = tmp_path / "input"
    image = input_dir / "experiment" / "day01" / "image.jpg"
    image.parent.mkdir(parents=True)
    image.touch()

    resolved = _resolve_metadata_image_path(
        input_dir, Path("experiment/day01/image.jpg")
    )

    assert resolved == image.resolve()


def test_metadata_image_path_rejects_parent_traversal(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    with pytest.raises(ValueError, match="escapes input directory"):
        _resolve_metadata_image_path(input_dir, "../outside.jpg")


def test_metadata_image_path_rejects_absolute_path(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    outside = tmp_path / "outside.jpg"

    with pytest.raises(ValueError, match="must be relative"):
        _resolve_metadata_image_path(input_dir, outside.resolve())


def test_metadata_image_path_rejects_escaping_symlink(tmp_path):
    input_dir = tmp_path / "input"
    outside_dir = tmp_path / "outside"
    input_dir.mkdir()
    outside_dir.mkdir()
    (input_dir / "linked").symlink_to(outside_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes input directory"):
        _resolve_metadata_image_path(input_dir, "linked/image.jpg")


def test_metadata_tasks_reject_path_traversal(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    metadata = pd.DataFrame([{"image_path": "../outside.jpg"}])

    with pytest.raises(ValueError, match="escapes input directory"):
        _build_metadata_tasks(input_dir, metadata)


def test_metadata_tasks_skip_missing_in_root_files(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    metadata = pd.DataFrame([{"image_path": "missing.jpg"}])

    assert _build_metadata_tasks(input_dir, metadata) == []


def test_run_batch_reports_metadata_paths_that_match_no_files(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "actual.jpg").touch()
    metadata_csv = input_dir / "image_metadata.csv"
    metadata_csv.write_text("image_path\nmissing.jpg\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="No metadata image_path entries"):
        run_batch(input_dir, tmp_path / "analysis.zip", metadata_csv=metadata_csv, seed=None)
