"""Tests for result archive provenance."""

import json
import zipfile

from metrics_petri._provenance import build_provenance
from metrics_petri.pipelinesam import cli


def test_build_provenance_contains_run_context():
    provenance = build_provenance(
        interface="test-interface",
        threshold=0.42,
        dish_size_mm=60.0,
        device="cpu",
    )

    assert provenance["created_at_utc"].endswith("Z")
    assert provenance["interface"] == "test-interface"
    assert provenance["versions"]["python"]
    assert provenance["versions"]["metrics-petri"]
    assert provenance["model"]["filename"] == "best_area_w_0.7.pt"
    assert provenance["model"]["sha256"] == (
        "e868313abe5335d60cb92ed3d968b04b80e1c63c7c63ea031699f109d38a840d"
    )
    assert provenance["settings"] == {"threshold": 0.42, "dish_size_mm": 60.0}
    assert provenance["runtime"]["device"] == "cpu"


def test_cli_archive_includes_provenance_json(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    image_path = input_dir / "plate.jpg"
    image_path.write_bytes(b"not decoded by this test")
    output_zip = tmp_path / "analysis.zip"

    monkeypatch.setattr(cli, "_find_images", lambda root: [image_path])
    monkeypatch.setattr(cli, "_detect_image_date", lambda path: "2026-01-01")
    monkeypatch.setattr(
        cli,
        "_process_image",
        lambda img_path, meta, threshold, dish_size_mm: (
            {
                **meta,
                "area_mm2": 1.0,
                "diameter_mm": 1.0,
            },
            {},
        ),
    )
    monkeypatch.setattr(
        cli,
        "_write_charts",
        lambda df, out_dir: out_dir.mkdir(parents=True, exist_ok=True),
    )

    cli.run_batch(input_dir, output_zip, threshold=0.42, dish_size_mm=60.0)

    with zipfile.ZipFile(output_zip) as zf:
        assert "analysis_full.csv" in zf.namelist()
        assert "analysis_full.json" in zf.namelist()
        assert "provenance.json" in zf.namelist()
        provenance = json.loads(zf.read("provenance.json"))

    assert provenance["interface"] == "metrics-petri-cli"
    assert provenance["settings"] == {"threshold": 0.42, "dish_size_mm": 60.0}
    assert provenance["model"]["sha256"]
