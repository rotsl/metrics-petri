from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MPL_DIR = Path.cwd() / ".mplconfig"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as PdfImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a markdown and PDF report for a saved run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--tags", default="")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _slugify(value: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in slug.split("-") if part)[:60]


def _metric(result: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    current: Any = result
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    if current is None:
        return default
    try:
        return float(current)
    except (TypeError, ValueError):
        return default


def _build_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        radial = result.get("rawAnalysis", {}).get("radial_profile", {})
        diagnostics = result.get("rawAnalysis", {}).get("segmentation_diagnostics", {})
        texture = result.get("texture", {})
        zonation = texture.get("radialZonation", {})
        cracks = result.get("cracks", {})
        rows.append(
            {
                "filename": result.get("filename", ""),
                "day": int(result.get("day", 0)),
                "area": _metric(result, "morphology", "areaMm2"),
                "radius": _metric(result, "morphology", "equivalentRadiusMm"),
                "diameter": _metric(result, "morphology", "diameterMm"),
                "perimeter": _metric(result, "morphology", "perimeterMm"),
                "circularity": _metric(result, "morphology", "circularity"),
                "eccentricity": _metric(result, "morphology", "eccentricity"),
                "edge_roughness": _metric(result, "morphology", "edgeRoughness"),
                "entropy": _metric(result, "texture", "entropy"),
                "center_edge_delta": _metric(result, "texture", "centerToEdgeDelta"),
                "density_index": _metric(result, "texture", "densityIndex"),
                "contrast": _metric(result, "texture", "contrast"),
                "core": float(zonation.get("core", 0.0) or 0.0),
                "middle": float(zonation.get("middle", 0.0) or 0.0),
                "outer": float(zonation.get("outer", 0.0) or 0.0),
                "crack_count": float(cracks.get("count", 0.0) or 0.0),
                "crack_coverage": float(cracks.get("coveragePct", 0.0) or 0.0),
                "relative_growth_rate": _metric(result, "kinematics", "relativeGrowthRate"),
                "area_growth_rate": _metric(result, "kinematics", "areaGrowthRate"),
                "radial_velocity": _metric(result, "kinematics", "radialVelocity"),
                "ring_spacing_mm": float(radial.get("ringSpacingMm", 0.0) or 0.0),
                "qc_status": str(result.get("qcStatus", "unknown")),
                "qc_notes": str(result.get("qcNotes", "")),
                "hybrid_strategy": str(diagnostics.get("hybrid_strategy", "unknown")),
                "gemma_blend_decision": str(diagnostics.get("gemma_blend_decision", "not recorded")),
                "sam_decision": str(diagnostics.get("sam_decision", "not recorded")),
            }
        )
    dataframe = pd.DataFrame(rows)
    if not dataframe.empty:
        dataframe = dataframe.sort_values(["day", "filename"]).reset_index(drop=True)
    return dataframe


def _save_figure(figure: Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_lines(ax: Any, dataframe: pd.DataFrame, columns: list[tuple[str, str, str]], title: str, xlabel: str, ylabel: str) -> None:
    if dataframe.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    else:
        for key, label, color in columns:
            ax.plot(dataframe["day"], dataframe[key], marker="o", linewidth=2, label=label, color=color)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def _plot_scatter(ax: Any, dataframe: pd.DataFrame) -> None:
    if dataframe.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    else:
        sizes = np.clip(dataframe["diameter"].to_numpy(dtype=float), 1.0, None) * 4.0
        scatter = ax.scatter(dataframe["area"], dataframe["circularity"], s=sizes, c=dataframe["day"], cmap="viridis", alpha=0.8)
        plt.colorbar(scatter, ax=ax, label="Day")
    ax.set_title("Area versus circularity")
    ax.set_xlabel("Colony area (mm²)")
    ax.set_ylabel("Circularity")
    ax.grid(alpha=0.2)


def _plot_heatmap(ax: Any, dataframe: pd.DataFrame) -> None:
    columns = ["area", "circularity", "entropy", "edge_roughness", "crack_coverage", "relative_growth_rate"]
    labels = ["Area", "Circularity", "Entropy", "Roughness", "Crack cov.", "Rel. growth"]
    if dataframe.empty or len(dataframe) < 2:
        ax.text(0.5, 0.5, "Need at least two images", ha="center", va="center")
    else:
        corr = dataframe[columns].corr().to_numpy(dtype=float)
        image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title("Feature correlation heatmap")


def _mean_radial_profile(results: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray] | None:
    grid = np.linspace(0.0, 1.0, 60)
    profiles: list[np.ndarray] = []
    for result in results:
        radial = result.get("rawAnalysis", {}).get("radial_profile", {})
        fractions = radial.get("radiusFraction") or []
        intensities = radial.get("meanIntensity") or []
        if len(fractions) < 2 or len(fractions) != len(intensities):
            continue
        profiles.append(np.interp(grid, np.asarray(fractions, dtype=float), np.asarray(intensities, dtype=float)))
    if not profiles:
        return None
    return grid, np.mean(np.vstack(profiles), axis=0)


def _write_graphs(results: list[dict[str, Any]], dataframe: pd.DataFrame, assets_dir: Path) -> list[dict[str, str]]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    sections: list[dict[str, str]] = []

    graph_specs = [
        (
            "01_colony_expansion.png",
            "Colony expansion",
            lambda ax: _plot_lines(
                ax,
                dataframe,
                [("area", "Area (mm²)", "#4f46e5"), ("diameter", "Diameter (mm)", "#059669")],
                "Colony expansion over time",
                "Time (days)",
                "Size (mm² / mm)",
            ),
        ),
        (
            "02_growth_roughness.png",
            "Relative growth and edge roughness",
            lambda ax: _plot_lines(
                ax,
                dataframe,
                [("relative_growth_rate", "Relative growth rate (1/day)", "#ea580c"), ("edge_roughness", "Edge roughness", "#b45309")],
                "Relative growth and edge roughness",
                "Time (days)",
                "Growth / roughness",
            ),
        ),
        (
            "03_stress_remodeling.png",
            "Stress remodeling",
            lambda ax: _plot_lines(
                ax,
                dataframe,
                [("crack_coverage", "Crack coverage (%)", "#f59e0b"), ("crack_count", "Crack count", "#7c2d12")],
                "Stress remodeling through crack burden",
                "Time (days)",
                "Coverage / count",
            ),
        ),
        (
            "04_texture_intensity.png",
            "Texture organization",
            lambda ax: _plot_lines(
                ax,
                dataframe,
                [("entropy", "Texture entropy (bits)", "#0f766e"), ("center_edge_delta", "Center-to-edge intensity", "#14b8a6")],
                "Texture organization and radial intensity shift",
                "Time (days)",
                "Texture signal",
            ),
        ),
        ("05_area_circularity.png", "Area versus circularity", lambda ax: _plot_scatter(ax, dataframe)),
        ("06_feature_correlation.png", "Feature correlation heatmap", lambda ax: _plot_heatmap(ax, dataframe)),
    ]

    for filename, title, renderer in graph_specs:
        figure, ax = plt.subplots(figsize=(8.5, 4.5))
        renderer(ax)
        path = assets_dir / filename
        _save_figure(figure, path)
        sections.append({"title": title, "image": path.name})

    figure, ax = plt.subplots(figsize=(8.5, 4.5))
    radial = _mean_radial_profile(results)
    if radial is None:
        ax.text(0.5, 0.5, "No radial profile available", ha="center", va="center")
    else:
        grid, values = radial
        ax.plot(grid, values, color="#2563eb", linewidth=2)
        ax.grid(alpha=0.25)
    ax.set_title("Mean radial intensity profile")
    ax.set_xlabel("Normalized radius (center to edge)")
    ax.set_ylabel("Mean grayscale intensity")
    path = assets_dir / "07_radial_profile.png"
    _save_figure(figure, path)
    sections.append({"title": "Mean radial intensity profile", "image": path.name})

    figure, ax = plt.subplots(figsize=(8.5, 4.5))
    if dataframe.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    else:
        bins = min(8, max(3, len(dataframe)))
        ax.hist(dataframe["area"], bins=bins, color="#6366f1", alpha=0.85, edgecolor="white")
        ax.grid(alpha=0.2)
    ax.set_title("Area distribution")
    ax.set_xlabel("Area (mm²)")
    ax.set_ylabel("Image count")
    path = assets_dir / "08_area_distribution.png"
    _save_figure(figure, path)
    sections.append({"title": "Area distribution", "image": path.name})

    return sections


def _model_summary(manifest: dict[str, Any], dataframe: pd.DataFrame) -> str:
    hybrid = Counter(str(value) for value in dataframe.get("hybrid_strategy", pd.Series(dtype=str)).tolist() if value)
    gemma = Counter(str(value) for value in dataframe.get("gemma_blend_decision", pd.Series(dtype=str)).tolist() if value)
    sam = Counter(str(value) for value in dataframe.get("sam_decision", pd.Series(dtype=str)).tolist() if value)
    hybrid_text = ", ".join(f"{key}: {count}" for key, count in hybrid.most_common(3)) or "none recorded"
    gemma_text = ", ".join(f"{key}: {count}" for key, count in gemma.most_common(3)) or "none recorded"
    sam_text = ", ".join(f"{key}: {count}" for key, count in sam.most_common(3)) or "none recorded"
    return (
        f"Run engine: {manifest.get('engine', 'unknown')}. Engine model: {manifest.get('engine_model', 'unknown')}. "
        f"Hybrid segmentation decisions: {hybrid_text}. Gemma blend decisions: {gemma_text}. "
        f"SAM decisions: {sam_text}."
    )


def _qc_summary(dataframe: pd.DataFrame) -> str:
    counts = Counter(dataframe["qc_status"].tolist()) if not dataframe.empty else Counter()
    return ", ".join(f"{status}: {count}" for status, count in counts.items()) or "none recorded"


def _growth_text(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "No colony growth values were available."
    start_area = dataframe["area"].iloc[0]
    end_area = dataframe["area"].iloc[-1]
    peak = dataframe.loc[dataframe["relative_growth_rate"].idxmax()]
    return (
        f"Colony area changed from {start_area:.2f} mm^2 to {end_area:.2f} mm^2. "
        f"The highest relative growth rate was {peak['relative_growth_rate']:.3f} 1/day on day {int(peak['day'])}."
    )


def _stress_text(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "No crack measurements were available."
    peak = dataframe.loc[dataframe["crack_coverage"].idxmax()]
    return (
        f"Mean crack coverage was {dataframe['crack_coverage'].mean():.2f}%, peaking at "
        f"{peak['crack_coverage']:.2f}% on day {int(peak['day'])}."
    )


def _texture_text(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "No texture measurements were available."
    ring = dataframe["ring_spacing_mm"][dataframe["ring_spacing_mm"] > 0]
    ring_text = f"Mean ring spacing was {ring.mean():.2f} mm." if not ring.empty else "Ring spacing was not confidently resolved."
    return (
        f"Texture entropy averaged {dataframe['entropy'].mean():.3f} bits, and the mean center-to-edge intensity shift was "
        f"{dataframe['center_edge_delta'].mean():.3f}. {ring_text}"
    )


def _shape_text(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "No morphology measurements were available."
    return (
        f"Mean circularity was {dataframe['circularity'].mean():.3f}, and mean edge roughness was "
        f"{dataframe['edge_roughness'].mean():.3f}."
    )


def _build_markdown(
    run_id: str,
    manifest: dict[str, Any],
    dataframe: pd.DataFrame,
    sections: list[dict[str, str]],
    experiment_name: str,
    tags: list[str],
) -> str:
    section_notes = {
        "Colony expansion": _growth_text(dataframe),
        "Relative growth and edge roughness": _shape_text(dataframe),
        "Stress remodeling": _stress_text(dataframe),
        "Texture organization": _texture_text(dataframe),
        "Area versus circularity": _shape_text(dataframe),
        "Feature correlation heatmap": "This heatmap shows how morphology, stress, and texture features co-vary in the current run.",
        "Mean radial intensity profile": "The radial profile summarizes grayscale density changes from colony center to edge.",
        "Area distribution": "This distribution shows whether images cluster around one size state or several growth states.",
    }
    lines = [
        f"# Magnaporthe Growth Report: {run_id}",
        "",
        "## Run Metadata",
        "",
        f"- Report generated at: {datetime.now(timezone.utc).isoformat()}",
        f"- Engine: {manifest.get('engine', 'unknown')}",
        f"- Engine model: {manifest.get('engine_model', 'unknown')}",
        f"- Image count: {len(dataframe)}",
        f"- Experiment: {experiment_name or 'not provided'}",
        f"- Tags: {', '.join(str(tag) for tag in tags) if tags else 'none'}",
        "",
        "## Automated Summary",
        "",
        _model_summary(manifest, dataframe),
        "",
        f"QC outcomes: {_qc_summary(dataframe)}.",
        "",
        _growth_text(dataframe),
        "",
        _stress_text(dataframe),
        "",
        _texture_text(dataframe),
        "",
        _shape_text(dataframe),
        "",
        "## Graphs",
        "",
    ]
    for section in sections:
        lines.extend(
            [
                f"### {section['title']}",
                "",
                section_notes.get(section["title"], "Biology-focused summary for this graph."),
                "",
                f"![{section['title']}](report_assets/{section['image']})",
                "",
            ]
        )
    if not dataframe.empty:
        lines.extend(
            [
                "## Per-image Metrics",
                "",
                "| Filename | Day | Area (mm^2) | Diameter (mm) | Circularity | Entropy | Crack coverage (%) | QC |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for _, row in dataframe.iterrows():
            lines.append(
                f"| {row['filename']} | {int(row['day'])} | {row['area']:.2f} | {row['diameter']:.2f} | "
                f"{row['circularity']:.3f} | {row['entropy']:.3f} | {row['crack_coverage']:.2f} | {row['qc_status']} |"
            )
        lines.append("")
    return "\n".join(lines)


def _build_pdf(
    run_id: str,
    manifest: dict[str, Any],
    dataframe: pd.DataFrame,
    sections: list[dict[str, str]],
    assets_dir: Path,
    pdf_path: Path,
    experiment_name: str,
    tags: list[str],
) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodySmall", fontSize=9, leading=12, spaceAfter=6))
    story: list[Any] = []
    story.append(Paragraph(f"Magnaporthe Growth Report: {run_id}", styles["Title"]))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph(_model_summary(manifest, dataframe), styles["BodyText"]))
    story.append(Paragraph(f"QC outcomes: {_qc_summary(dataframe)}.", styles["BodyText"]))
    story.append(Spacer(1, 0.15 * inch))

    metadata_table = Table(
        [
            ["Engine", str(manifest.get("engine", "unknown"))],
            ["Engine model", str(manifest.get("engine_model", "unknown"))],
            ["Image count", str(len(dataframe))],
            ["Experiment", experiment_name or "not provided"],
            ["Tags", ", ".join(str(tag) for tag in tags) if tags else "none"],
            ["Generated", datetime.now(timezone.utc).isoformat()],
        ],
        colWidths=[1.5 * inch, 4.8 * inch],
    )
    metadata_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(metadata_table)
    story.append(Spacer(1, 0.18 * inch))
    for heading, text in [
        ("Growth summary", _growth_text(dataframe)),
        ("Stress summary", _stress_text(dataframe)),
        ("Texture summary", _texture_text(dataframe)),
        ("Shape summary", _shape_text(dataframe)),
    ]:
        story.append(Paragraph(heading, styles["Heading2"]))
        story.append(Paragraph(text, styles["BodyText"]))
    story.append(PageBreak())

    notes = {
        "Colony expansion": _growth_text(dataframe),
        "Relative growth and edge roughness": _shape_text(dataframe),
        "Stress remodeling": _stress_text(dataframe),
        "Texture organization": _texture_text(dataframe),
        "Area versus circularity": _shape_text(dataframe),
        "Feature correlation heatmap": "This panel summarizes pairwise feature covariation within the current run.",
        "Mean radial intensity profile": "The radial profile captures density changes from center to edge.",
        "Area distribution": "This histogram shows how colony sizes are distributed across the run.",
    }
    for section in sections:
        story.append(Paragraph(section["title"], styles["Heading2"]))
        story.append(Paragraph(notes.get(section["title"], ""), styles["BodySmall"]))
        story.append(PdfImage(str(assets_dir / section["image"]), width=6.8 * inch, height=3.6 * inch))
        story.append(Spacer(1, 0.15 * inch))

    if not dataframe.empty:
        story.append(PageBreak())
        story.append(Paragraph("Per-image metrics", styles["Heading2"]))
        rows = [["Filename", "Day", "Area", "Diameter", "Circularity", "Entropy", "Crack cov.", "QC"]]
        for _, row in dataframe.iterrows():
            rows.append(
                [
                    str(row["filename"]),
                    str(int(row["day"])),
                    f"{row['area']:.2f}",
                    f"{row['diameter']:.2f}",
                    f"{row['circularity']:.3f}",
                    f"{row['entropy']:.3f}",
                    f"{row['crack_coverage']:.2f}",
                    str(row["qc_status"]),
                ]
            )
        table = Table(rows, repeatRows=1, colWidths=[2.35 * inch, 0.42 * inch, 0.72 * inch, 0.72 * inch, 0.72 * inch, 0.62 * inch, 0.8 * inch, 0.45 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2ff")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("LEADING", (0, 0), (-1, -1), 8),
                    ("PADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(table)

    SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36).build(story)


def generate_report(run_dir: Path, experiment_override: str = "", tags_override: list[str] | None = None) -> dict[str, Any]:
    manifest = _read_json(run_dir / "manifest.json")
    results = _read_json(run_dir / "analysis.json")
    if not isinstance(results, list):
        raise ValueError("analysis.json must contain a list of results")

    dataframe = _build_dataframe(results)
    assets_dir = run_dir / "report_assets"
    sections = _write_graphs(results, dataframe, assets_dir)
    metadata = manifest.get("metadata", {}) if isinstance(manifest.get("metadata", {}), dict) else {}
    experiment_name = experiment_override.strip() or str(metadata.get("experiment_name", "") or "").strip()
    base_tags = metadata.get("tags", []) if isinstance(metadata.get("tags", []), list) else []
    tags = [str(tag) for tag in dict.fromkeys([*base_tags, *(tags_override or [])])]
    markdown = _build_markdown(run_dir.name, manifest, dataframe, sections, experiment_name, tags)
    suffix_parts = [part for part in [_slugify(experiment_name), _slugify("-".join(tags))] if part]
    stem = f"report_{'_'.join(suffix_parts)}" if suffix_parts else "report"

    markdown_path = run_dir / f"{stem}.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    pdf_path = run_dir / f"{stem}.pdf"
    _build_pdf(run_dir.name, manifest, dataframe, sections, assets_dir, pdf_path, experiment_name, tags)

    return {
        "runId": run_dir.name,
        "template": "biology_report_v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "markdownPath": f"/outputs/{run_dir.name}/{markdown_path.name}",
        "pdfPath": f"/outputs/{run_dir.name}/{pdf_path.name}",
        "assetsDir": f"/outputs/{run_dir.name}/report_assets",
        "markdownContent": markdown,
        "graphCount": len(sections),
        "experimentName": experiment_name,
        "tags": tags,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    tag_values = [part.strip() for part in str(args.tags).split(",") if part.strip()]
    payload = generate_report(Path(args.run_dir), experiment_override=args.experiment_name, tags_override=tag_values)
    if args.json_output:
        print(json.dumps(payload))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
