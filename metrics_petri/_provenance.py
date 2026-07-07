# SPDX-License-Identifier: MIT
"""Build machine-readable run provenance for metrics-petri result archives."""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

from . import __version__
from ._paths import _HF_FILE, _expected_model_sha256


def _distribution_version(name: str) -> str | None:
    """Return an installed distribution version, or None when unavailable."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def build_provenance(
    *,
    interface: str,
    threshold: float,
    dish_size_mm: float,
    device: str,
) -> dict:
    """Return a JSON-serialisable provenance record for an analysis run."""
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "interface": interface,
        "versions": {
            "python": sys.version.split()[0],
            "metrics-petri": __version__,
            "torch": _distribution_version("torch"),
            "numpy": _distribution_version("numpy"),
            "scikit-image": _distribution_version("scikit-image"),
            "scipy": _distribution_version("scipy"),
            "pandas": _distribution_version("pandas"),
        },
        "model": {
            "filename": _HF_FILE,
            "sha256": _expected_model_sha256(),
        },
        "settings": {
            "threshold": float(threshold),
            "dish_size_mm": float(dish_size_mm),
        },
        "runtime": {
            "device": str(device),
            "platform": platform.platform(),
        },
    }
