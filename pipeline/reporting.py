"""Pure-Python report generation (matplotlib only, no R/Quarto)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from textwrap import fill
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

MPL_DIR = Path.cwd() / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))


# ── data helpers ───────────────────────────────────────────────────────────

def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(result: dict, *keys: str, default: float = 0.0) -> float:
    cur: Any = result
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    if cur is None:
        return default
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def _build_dataframe(results: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for r in results:
        rows.append(
            {
                "filename": r.get("filename") or r.get("image_path", ""),
                "day": int(r.get("day") or r.get("days_since_start") or 0),
                "area": _metric(r, "area_mm2") or _metric(r, "morphology", "areaMm2"),
                "diameter": _metric(r, "diameter_mm") or _metric(r, "morphology", "diameterMm"),
                "perimeter": _metric(r, "perimeter_mm") or _metric(r, "morphology", "perimeterMm"),
                "circularity": _metric(r, "morphology", "circularity"),
                "eccentricity": _metric(r, "eccentricity") or _metric(r, "morphology", "eccentricity"),
                "edge_roughness": _metric(r, "edge_roughness") or _metric(r, "morphology", "edgeRoughness"),
                "entropy": _metric(r, "entropy") or _metric(r, "texture", "entropy"),
                "center_edge_delta": _metric(r, "texture", "centerToEdgeDelta"),
                "density_index": _metric(r, "texture", "densityIndex"),
                "contrast": _metric(r, "texture", "contrast"),
                "crack_count": _metric(r, "crack_count") or _metric(r, "cracks", "count"),
                "crack_coverage": _metric(r, "crack_coverage_pct") or _metric(r, "cracks", "coveragePct"),
                "relative_growth_rate": _metric(r, "rgr_per_day") or _metric(r, "kinematics", "relativeGrowthRate"),
                "area_growth_rate": _metric(r, "relative_growth_per_day") or _metric(r, "kinematics", "areaGrowthRate"),
                "radial_velocity": _metric(r, "kinematics", "radialVelocity"),
                "qc_status": str(r.get("qcStatus", r.get("error", "ok"))),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["day", "filename"]).reset_index(drop=True)
    return df


# ── figure helpers ─────────────────────────────────────────────────────────

def _save_figure(fig: Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_lines(
    ax: Any,
    df: pd.DataFrame,
    columns: list[tuple[str, str, str]],
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    else:
        for key, label, color in columns:
            ax.plot(df["day"], df[key], marker="o", linewidth=2, label=label, color=color)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _plot_scatter(ax: Any, df: pd.DataFrame) -> None:
    if df.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    else:
        sizes = np.clip(df["diameter"].to_numpy(dtype=float), 1.0, None) * 4.0
        sc = ax.scatter(df["area"], df["circularity"], s=sizes, c=df["day"], cmap="viridis", alpha=0.8)
        plt.colorbar(sc, ax=ax, label="Day")
    ax.set_title("Colony area vs circularity")
    ax.set_xlabel("Colony area (mm²)")
    ax.set_ylabel("Circularity")
    ax.grid(alpha=0.2)


def _plot_heatmap(ax: Any, df: pd.DataFrame) -> None:
    cols = ["area", "circularity", "entropy", "edge_roughness", "crack_coverage", "relative_growth_rate"]
    labels = ["Area", "Circularity", "Entropy", "Roughness", "Crack cov.", "Rel. growth"]
    if df.empty or len(df) < 2:
        ax.text(0.5, 0.5, "Need at least two images", ha="center", va="center")
    else:
        corr = df[cols].corr().to_numpy(dtype=float)
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title("Feature correlation heatmap")


def _save_table_figure(
    path: Path, title: str, df: pd.DataFrame, columns: list[str]
) -> None:
    sel = df.loc[:, columns].copy()
    if "filename" in sel.columns:
        sel["filename"] = sel["filename"].astype(str).map(lambda v: fill(v, width=28))
    for col in sel.columns:
        if col == "filename":
            continue
        sel[col] = sel[col].map(
            lambda v: f"{float(v):.3f}"
            if isinstance(v, (int, float, np.integer, np.floating))
            else str(v)
        )
    row_count = max(len(sel), 1)
    fig, ax = plt.subplots(figsize=(13.5, max(2.2, 0.55 * row_count + 1.6)))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=18)
    tbl = ax.table(
        cellText=sel.values.tolist(),
        colLabels=list(sel.columns),
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.7)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if row == 0:
            cell.set_facecolor("#eef2ff")
            cell.set_text_props(weight="bold", color="#1f2937")
        elif col == 0:
            cell.set_text_props(ha="left")
        else:
            cell.set_text_props(ha="right")
    _save_figure(fig, path)


# ── public API ─────────────────────────────────────────────────────────────

def write_graphs(
    results: list[dict],
    df: pd.DataFrame,
    assets_dir: Path,
) -> list[dict[str, str]]:
    """Write PNG graphs to assets_dir. Returns list of {title, image} dicts."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    sections: list[dict[str, str]] = []

    graph_specs = [
        (
            "01_colony_expansion.png",
            "Colony expansion",
            lambda ax: _plot_lines(
                ax, df,
                [("area", "Area (mm²)", "#4f46e5"), ("diameter", "Diameter (mm)", "#059669")],
                "Colony expansion over time", "Time (days)", "Size (mm² / mm)",
            ),
        ),
        (
            "02_growth_roughness.png",
            "Relative growth and edge roughness",
            lambda ax: _plot_lines(
                ax, df,
                [
                    ("relative_growth_rate", "Relative growth rate (1/day)", "#ea580c"),
                    ("edge_roughness", "Edge roughness", "#b45309"),
                ],
                "Relative growth and edge roughness", "Time (days)", "Growth / roughness",
            ),
        ),
        (
            "03_crack_burden.png",
            "Crack burden",
            lambda ax: _plot_lines(
                ax, df,
                [
                    ("crack_coverage", "Crack coverage (%)", "#f59e0b"),
                    ("crack_count", "Crack count", "#7c2d12"),
                ],
                "Crack burden over time", "Time (days)", "Coverage / count",
            ),
        ),
        (
            "04_texture.png",
            "Texture organisation",
            lambda ax: _plot_lines(
                ax, df,
                [
                    ("entropy", "Texture entropy (bits)", "#0f766e"),
                    ("center_edge_delta", "Center-to-edge intensity", "#14b8a6"),
                ],
                "Texture organisation", "Time (days)", "Texture signal",
            ),
        ),
        ("05_area_circularity.png", "Area vs circularity", lambda ax: _plot_scatter(ax, df)),
        ("06_feature_correlation.png", "Feature correlation heatmap", lambda ax: _plot_heatmap(ax, df)),
    ]

    for filename, title, renderer in graph_specs:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        renderer(ax)
        path = assets_dir / filename
        _save_figure(fig, path)
        sections.append({"title": title, "image": path.name})

    # Area distribution
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    if not df.empty:
        bins = min(8, max(3, len(df)))
        ax.hist(df["area"], bins=bins, color="#6366f1", alpha=0.85, edgecolor="white")
        ax.grid(alpha=0.2)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    ax.set_title("Area distribution")
    ax.set_xlabel("Area (mm²)")
    ax.set_ylabel("Image count")
    path = assets_dir / "07_area_distribution.png"
    _save_figure(fig, path)
    sections.append({"title": "Area distribution", "image": path.name})

    # Summary tables
    _save_table_figure(
        assets_dir / "08_table_morphology.png",
        "Per-image morphology and QC",
        df,
        ["filename", "day", "area", "diameter", "perimeter", "eccentricity", "qc_status"],
    )
    _save_table_figure(
        assets_dir / "09_table_texture.png",
        "Per-image texture metrics",
        df,
        ["filename", "entropy", "center_edge_delta", "density_index"],
    )
    _save_table_figure(
        assets_dir / "10_table_growth.png",
        "Per-image crack and growth metrics",
        df,
        ["filename", "crack_coverage", "relative_growth_rate", "area_growth_rate"],
    )

    return sections


def generate_report(
    run_dir: Path,
    experiment_override: str = "",
) -> dict[str, Any]:
    """Generate matplotlib report assets from a saved run directory.

    The run_dir must contain analysis.json and manifest.json.
    Returns a dict with paths to generated assets.
    """
    results = _read_json(run_dir / "analysis.json")
    if not isinstance(results, list):
        raise ValueError("analysis.json must contain a list of results")

    df = _build_dataframe(results)
    assets_dir = run_dir / "report_assets"
    sections = write_graphs(results, df, assets_dir)

    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    experiment_name = experiment_override.strip() or str(
        manifest.get("metadata", {}).get("experiment_name", "") or ""
    ).strip()

    record = {
        "runId": run_dir.name,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "assetsDir": str(assets_dir),
        "graphCount": len(sections),
        "experimentName": experiment_name,
        "sections": sections,
    }
    (run_dir / "report_bundle.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return record
