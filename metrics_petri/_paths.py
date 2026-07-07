# SPDX-License-Identifier: MIT
"""Shared model-path resolution for pipeline and pipelinesam."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

_HF_REPO = "rotsl/grayleafspot-segmentation"
_HF_FILE = "best_area_w_0.7.pt"
_MODEL_CHECKSUM_PATH = Path(__file__).resolve().parent / "models" / f"{_HF_FILE}.sha256"

_DEFAULT_MODEL_CANDIDATES = [
    Path("metrics_petri/models/best_area_w_0.7.pt"),
    Path(__file__).resolve().parent / "models" / _HF_FILE,
]


def _expected_model_sha256() -> str:
    """Return the published SHA-256 digest for the official checkpoint."""
    try:
        digest = _MODEL_CHECKSUM_PATH.read_text(encoding="utf-8").split()[0].lower()
    except (OSError, IndexError) as exc:
        raise RuntimeError(f"Model checksum file is unavailable: {_MODEL_CHECKSUM_PATH}") from exc
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise RuntimeError(f"Model checksum file is invalid: {_MODEL_CHECKSUM_PATH}")
    return digest


def _verify_model_checksum(path: Path) -> Path:
    """Verify an official model file and return its resolved path."""
    resolved = path.expanduser().resolve()
    hasher = hashlib.sha256()
    with resolved.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            hasher.update(chunk)
    actual = hasher.hexdigest()
    expected = _expected_model_sha256()
    if actual != expected:
        raise ValueError(
            f"Model checksum mismatch for {resolved}: expected {expected}, got {actual}"
        )
    return resolved


def _verify_model_if_managed(path: Path) -> Path:
    """Verify official models while treating an explicit UNET_MODEL path as trusted."""
    resolved = path.expanduser().resolve()
    custom = os.getenv("UNET_MODEL")
    if custom and Path(custom).expanduser().resolve() == resolved:
        return resolved
    return _verify_model_checksum(resolved)


def _find_model_path() -> Path | None:
    """Return model path if found locally. Returns None without downloading."""
    env = os.getenv("UNET_MODEL")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p
    for c in _DEFAULT_MODEL_CANDIDATES:
        if c.exists():
            return c.resolve()
    try:
        from importlib.resources import files
        p = Path(str(files("metrics_petri.models").joinpath(_HF_FILE)))
        if p.exists():
            return p
    except Exception:
        pass
    return None
