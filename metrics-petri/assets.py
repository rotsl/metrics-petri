from __future__ import annotations

from pathlib import Path


def get_assets_dir() -> Path:
    return Path(__file__).resolve().parent / "dist"


def assets_exist() -> bool:
    assets_dir = get_assets_dir()
    return assets_dir.exists() and (assets_dir / "index.html").exists()
