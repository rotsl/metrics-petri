# SPDX-License-Identifier: MIT
"""Shared model-path resolution for pipeline and pipelinesam."""
from __future__ import annotations

import os
from pathlib import Path

_HF_REPO = "rotsl/grayleafspot-segmentation"
_HF_FILE = "best_area_w_0.7.pt"

_DEFAULT_MODEL_CANDIDATES = [
    Path("models/best_area_w_0.7.pt"),
    Path(__file__).resolve().parent / "models" / _HF_FILE,
]


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
