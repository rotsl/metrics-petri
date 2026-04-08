"""Utility helpers for mask/annotation loading, plotting, and report rendering."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from string import Template

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
EVAL_ROOT = Path(__file__).resolve().parents[1]
RAW_IMAGE_ROOT = EVAL_ROOT / "data" / "ground_truth" / "raw_images"
PREDICTION_MASK_ROOT = EVAL_ROOT / "data" / "predictions" / "masks"


def canonical_stem(path: Path) -> str:
    """Return a normalized stem for matching messy annotation filenames."""

    stem = path.stem
    while stem.endswith("."):
        stem = stem[:-1]
    return stem


def list_annotation_files(directory: Path) -> list[Path]:
    """List JSON annotation files recursively."""

    return sorted(path for path in directory.rglob("*.json") if path.is_file())


def list_mask_files(directory: Path) -> list[Path]:
    """List binary mask image files recursively."""

    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def pair_prediction_and_ground_truth(pred_dir: Path, gt_dir: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    """Match prediction and GT files by normalized stem."""

    if pred_dir.suffix.lower() == ".json" or gt_dir.suffix.lower() == ".json":
        pred_files = list_annotation_files(pred_dir)
        gt_files = list_annotation_files(gt_dir)
    else:
        pred_json = list_annotation_files(pred_dir)
        gt_json = list_annotation_files(gt_dir)
        if pred_json and gt_json:
            pred_files = pred_json
            gt_files = gt_json
        else:
            pred_files = list_mask_files(pred_dir)
            gt_files = list_mask_files(gt_dir)

    gt_by_stem = {canonical_stem(path): path for path in gt_files}

    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for pred_path in pred_files:
        gt_path = gt_by_stem.get(canonical_stem(pred_path))
        if gt_path is None:
            missing.append(pred_path.name)
            continue
        pairs.append((pred_path, gt_path))

    return pairs, missing


def load_annotation(annotation_path: Path) -> dict[str, object]:
    """Load a JSON annotation file."""

    return json.loads(annotation_path.read_text())


def load_binary_mask(mask_path: Path) -> tuple[np.ndarray, dict[str, object]]:
    """Load a binary mask image and convert it to a foreground boolean array."""

    image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read mask image: {mask_path}")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mask = image > 0
    metadata = {
        "annotation_type": "mask",
        "image_shape": tuple(int(value) for value in mask.shape),
        "polygon_count": 0,
        "vertex_count": 0,
        "image_path": mask_path.name,
    }
    return mask.astype(bool), metadata


def _polygon_points_to_array(points: list[object]) -> np.ndarray:
    """Convert polygon point payloads into an ``Nx2`` float array."""

    if not points:
        return np.empty((0, 2), dtype=np.float32)

    if isinstance(points[0], dict):
        coords = [[float(point["x"]), float(point["y"])] for point in points]
    else:
        coords = [[float(point[0]), float(point[1])] for point in points]
    return np.asarray(coords, dtype=np.float32)


def _mask_from_polygon_points(points: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Rasterize polygon points into a binary mask."""

    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) < 3:
        return mask.astype(bool)
    polygon = np.round(points).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [polygon.reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


def annotation_to_mask(annotation: dict[str, object], annotation_type: str) -> tuple[np.ndarray, dict[str, object]]:
    """Convert a GT or prediction annotation payload into a binary mask."""

    if annotation_type == "ground_truth":
        height = int(annotation["imageHeight"])
        width = int(annotation["imageWidth"])
        shapes = annotation.get("shapes", [])
        mask = np.zeros((height, width), dtype=bool)
        total_vertices = 0
        for shape in shapes:
            points = _polygon_points_to_array(shape.get("points", []))
            total_vertices += int(len(points))
            mask |= _mask_from_polygon_points(points, (height, width))
        metadata = {
            "annotation_type": annotation_type,
            "image_shape": (height, width),
            "polygon_count": len(shapes),
            "vertex_count": total_vertices,
            "image_path": annotation.get("imagePath", ""),
        }
        return mask, metadata

    if annotation_type == "prediction":
        source_image = annotation.get("source_image") or annotation.get("imagePath")
        image_shape = None
        if source_image:
            source_name = Path(str(source_image)).name
            raw_candidates = [
                RAW_IMAGE_ROOT / source_name,
                RAW_IMAGE_ROOT / source_name.replace("0260220_", "20260220_"),
            ]
            for raw_image_path in raw_candidates:
                if raw_image_path.exists():
                    image = cv2.imread(str(raw_image_path), cv2.IMREAD_UNCHANGED)
                    if image is not None:
                        image_shape = image.shape[:2]
                        break

        if image_shape is None and source_image:
            stem = canonical_stem(Path(str(source_image)))
            mask_candidates = [
                PREDICTION_MASK_ROOT / f"{stem}.png",
                PREDICTION_MASK_ROOT / f"{stem}.jpg",
                PREDICTION_MASK_ROOT / f"{stem}.jpeg",
            ]
            for mask_path in mask_candidates:
                if mask_path.exists():
                    image = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                    if image is not None:
                        image_shape = image.shape[:2]
                        break

        if image_shape is None:
            raise FileNotFoundError(
                f"Could not infer image shape for prediction annotation {source_image!r}. "
                "A matching raw image is required in evaluation_report/data/ground_truth/raw_images."
            )

        points = _polygon_points_to_array(annotation.get("colony_polygon", []))
        mask = _mask_from_polygon_points(points, image_shape)
        metadata = {
            "annotation_type": annotation_type,
            "image_shape": image_shape,
            "polygon_count": 1 if len(points) >= 3 else 0,
            "vertex_count": int(len(points)),
            "image_path": str(source_image or ""),
        }
        return mask, metadata

    raise ValueError(f"Unsupported annotation_type: {annotation_type}")


def align_mask_shapes(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Resize the ground-truth mask to the prediction shape if needed."""

    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)
    diagnostics = {
        "shape_aligned": False,
        "prediction_shape": tuple(int(value) for value in pred.shape),
        "ground_truth_shape": tuple(int(value) for value in gt.shape),
    }

    if pred.shape == gt.shape:
        return pred, gt, diagnostics

    resized = cv2.resize(
        gt.astype(np.uint8),
        (pred.shape[1], pred.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    diagnostics["shape_aligned"] = True
    diagnostics["aligned_ground_truth_shape"] = tuple(int(value) for value in resized.shape)
    return pred, resized, diagnostics


def validate_masks(pred_mask: np.ndarray, gt_mask: np.ndarray, pred_path: Path, gt_path: Path) -> None:
    """Validate annotation-derived mask compatibility before evaluation."""

    if pred_mask.size == 0 or gt_mask.size == 0:
        raise ValueError(f"Empty mask encountered for pair {pred_path.name} / {gt_path.name}")


def ensure_directories(*directories: Path) -> None:
    """Create directories safely if they do not exist."""

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def render_metric_histograms(dataframe: pd.DataFrame, metric_columns: list[str], output_dir: Path) -> None:
    """Render simple histograms for metric distributions."""

    ensure_directories(output_dir)
    for metric in metric_columns:
        plt.figure(figsize=(6, 4))
        plt.hist(
            dataframe[metric].astype(float),
            bins=min(10, max(3, len(dataframe))),
            color="#3b82f6",
            edgecolor="black",
        )
        plt.xlabel(metric.replace("_", " ").title())
        plt.ylabel("Image count")
        plt.title(f"{metric.replace('_', ' ').title()} distribution")
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric}_histogram.png", dpi=150)
        plt.close()


def render_confusion_barplot(dataframe: pd.DataFrame, output_dir: Path) -> None:
    """Render aggregate confusion counts as a simple bar chart."""

    ensure_directories(output_dir)
    totals = {
        "TP": int(dataframe["tp"].sum()),
        "FP": int(dataframe["fp"].sum()),
        "FN": int(dataframe["fn"].sum()),
        "TN": int(dataframe["tn"].sum()),
    }

    plt.figure(figsize=(6, 4))
    plt.bar(list(totals.keys()), list(totals.values()), color=["#10b981", "#f59e0b", "#ef4444", "#64748b"])
    plt.ylabel("Pixel count")
    plt.title("Aggregate confusion counts")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_counts.png", dpi=150)
    plt.close()


def render_area_error_plot(dataframe: pd.DataFrame, output_dir: Path) -> None:
    """Render predicted-vs-GT area scatter and relative error distribution."""

    ensure_directories(output_dir)

    plt.figure(figsize=(6, 5))
    plt.scatter(
        dataframe["ground_truth_pixels"].astype(float),
        dataframe["prediction_pixels"].astype(float),
        color="#7c3aed",
        alpha=0.8,
    )
    limit = float(max(dataframe["ground_truth_pixels"].max(), dataframe["prediction_pixels"].max()))
    plt.plot([0, limit], [0, limit], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Ground-truth foreground pixels")
    plt.ylabel("Predicted foreground pixels")
    plt.title("Predicted vs ground-truth area")
    plt.tight_layout()
    plt.savefig(output_dir / "area_agreement.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(
        dataframe["relative_area_error_pct"].astype(float),
        bins=min(10, max(3, len(dataframe))),
        color="#f97316",
        edgecolor="black",
    )
    plt.xlabel("Relative area error (%)")
    plt.ylabel("Image count")
    plt.title("Relative area error distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "relative_area_error_histogram.png", dpi=150)
    plt.close()


def latex_escape(value: object) -> str:
    """Escape text for LaTeX insertion."""

    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    pattern = re.compile("|".join(re.escape(key) for key in replacements))
    return pattern.sub(lambda match: replacements[match.group(0)], text)


def build_summary_rows(summary_json: dict[str, object]) -> str:
    """Build LaTeX table rows for the aggregated metric table."""

    rows: list[str] = []
    metrics = summary_json.get("metrics", {})
    for metric_name, payload in metrics.items():
        row = (
            f"{latex_escape(metric_name)} & "
            f"{payload['mean']:.4f} & "
            f"{payload['std']:.4f} & "
            f"{payload['ci_lower']:.4f} & "
            f"{payload['ci_upper']:.4f} \\\\"
        )
        rows.append(row)
    return "\n".join(rows)


def build_area_rows(summary_json: dict[str, object]) -> str:
    """Build LaTeX table rows for annotation-area agreement statistics."""

    area_stats = summary_json.get("area_statistics", {})
    row_specs = [
        ("mean_predicted_area_px", "Mean predicted area (px)"),
        ("mean_ground_truth_area_px", "Mean GT area (px)"),
        ("mean_relative_area_error_pct", "Mean relative area error (\\%)"),
        ("median_relative_area_error_pct", "Median relative area error (\\%)"),
        ("shape_alignment_rate", "Shape alignment rate"),
    ]

    rows: list[str] = []
    for key, label in row_specs:
        value = area_stats.get(key, 0.0)
        if key == "shape_alignment_rate":
            rows.append(f"{label} & {float(value):.4f} \\\\")
        else:
            rows.append(f"{label} & {float(value):.4f} \\\\")
    return "\n".join(rows)


def build_conclusion_text(summary_json: dict[str, object]) -> str:
    """Generate a short statistically grounded narrative for the PDF."""

    metrics = summary_json["metrics"]
    comparison_mode = str(summary_json.get("comparison_mode", "annotations"))
    comparison_label = "mask pairs" if comparison_mode == "masks" else "annotation pairs"
    matched = int(summary_json.get("num_images", 0))
    requested = int(summary_json.get("requested_pairs", matched))
    precision_mean = metrics["precision"]["mean"]
    recall_mean = metrics["recall"]["mean"]
    f1_mean = metrics["f1_score"]["mean"]
    iou_mean = metrics["iou"]["mean"]
    iou_lower = metrics["iou"]["ci_lower"]
    iou_upper = metrics["iou"]["ci_upper"]
    area_error = summary_json.get("area_statistics", {}).get("median_relative_area_error_pct", 0.0)
    shape_alignment_rate = summary_json.get("area_statistics", {}).get("shape_alignment_rate", 0.0)

    if iou_mean >= 0.8:
        performance_label = "strong"
    elif iou_mean >= 0.6:
        performance_label = "moderate"
    else:
        performance_label = "limited"

    text = (
        f"Across {matched} matched {comparison_label} out of {requested} available predictions, the hybrid segmentation "
        f"pipeline delivered {performance_label} overlap with the manual labels. The mean IoU was {iou_mean:.4f} "
        f"with a bootstrap 95\\% confidence interval of [{iou_lower:.4f}, {iou_upper:.4f}], while the mean F1-score "
        f"was {f1_mean:.4f}. Precision ({precision_mean:.4f}) and recall ({recall_mean:.4f}) show whether the system "
        f"leans more toward under-segmentation or over-segmentation. The median relative foreground-area error was "
        f"{area_error:.2f}\\%, and {shape_alignment_rate:.4f} of pairs required canvas-size alignment before pixel-wise "
        f"comparison. These statistics suggest that the final performance estimate is driven by the full annotation set "
        f"rather than a single optimistic example, but the unmatched files and the alignment rate should still be reviewed "
        f"before treating the numbers as a final benchmark."
    )
    return latex_escape(text)


def render_report_template(
    template_path: Path,
    output_path: Path,
    summary_json: dict[str, object],
) -> None:
    """Fill the LaTeX template using summary statistics."""

    template = Template(template_path.read_text())
    comparison_mode = str(summary_json.get("comparison_mode", "annotations"))
    comparison_sentence = (
        "Ground-truth and prediction masks are compared directly as binary foreground maps."
        if comparison_mode == "masks"
        else "Ground-truth and prediction annotations are both rasterized into binary masks before pixel-wise comparison."
    )
    payload = {
        "NUM_IMAGES": summary_json.get("num_images", 0),
        "REQUESTED_PAIRS": summary_json.get("requested_pairs", 0),
        "MATCHED_PAIRS": summary_json.get("num_images", 0),
        "MISSING_PAIRS": summary_json.get("missing_pairs", 0),
        "SUMMARY_ROWS": build_summary_rows(summary_json),
        "AREA_ROWS": build_area_rows(summary_json),
        "PIPELINE_DESCRIPTION": latex_escape(
            "The evaluated output is the final hybrid segmentation contour produced after classical segmentation, "
            "U-Net support, Gemma-guided decision logic when available, and SAM refinement. "
            + comparison_sentence
        ),
        "PRECISION_MEAN": f"{summary_json['metrics']['precision']['mean']:.4f}",
        "RECALL_MEAN": f"{summary_json['metrics']['recall']['mean']:.4f}",
        "F1_MEAN": f"{summary_json['metrics']['f1_score']['mean']:.4f}",
        "IOU_MEAN": f"{summary_json['metrics']['iou']['mean']:.4f}",
        "CONCLUSION_TEXT": build_conclusion_text(summary_json),
    }
    output_path.write_text(template.safe_substitute(payload))


def write_json(data: dict[str, object], output_path: Path) -> None:
    """Write a JSON object with indentation."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
