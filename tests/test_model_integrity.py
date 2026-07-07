"""Tests for official model checkpoint integrity verification."""

import hashlib

import pytest

from metrics_petri import _paths


def _set_checksum_file(monkeypatch, tmp_path, content: bytes):
    checksum = hashlib.sha256(content).hexdigest()
    checksum_file = tmp_path / "model.pt.sha256"
    checksum_file.write_text(f"{checksum}  model.pt\n", encoding="utf-8")
    monkeypatch.setattr(_paths, "_MODEL_CHECKSUM_PATH", checksum_file)


def test_model_checksum_accepts_matching_file(monkeypatch, tmp_path):
    content = b"known model bytes"
    model = tmp_path / "model.pt"
    model.write_bytes(content)
    _set_checksum_file(monkeypatch, tmp_path, content)

    assert _paths._verify_model_checksum(model) == model.resolve()


def test_model_checksum_rejects_modified_file(monkeypatch, tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"modified model bytes")
    _set_checksum_file(monkeypatch, tmp_path, b"expected model bytes")

    with pytest.raises(ValueError, match="Model checksum mismatch"):
        _paths._verify_model_checksum(model)


def test_explicit_custom_model_is_not_compared_to_official_checksum(
    monkeypatch, tmp_path
):
    model = tmp_path / "custom.pt"
    model.write_bytes(b"custom model bytes")
    monkeypatch.setenv("UNET_MODEL", str(model))
    _set_checksum_file(monkeypatch, tmp_path, b"official model bytes")

    assert _paths._verify_model_if_managed(model) == model.resolve()
