"""End-to-end evaluation entrypoint for mask or annotation-derived segmentation metrics and reporting."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from metrics import metric_bundle
from stats import save_summary, summarise_metrics
from utils import (
    align_mask_shapes,
    annotation_to_mask,
    ensure_directories,
    load_binary_mask,
    load_annotation,
    pair_prediction_and_ground_truth,
    render_area_error_plot,
    render_confusion_barplot,
    render_metric_histograms,
    render_report_template,
    validate_masks,
)


LOGGER = logging.getLogger("evaluation_report")
METRIC_COLUMNS = ["precision", "recall", "f1_score", "iou"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evaluation entrypoint."""

    parser = argparse.ArgumentParser(
        description="Evaluate Magnaporthe segmentation predictions by comparing prediction and GT masks or annotations."
    )
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory containing predicted masks or annotation JSON files.")
    parser.add_argument("--gt-dir", type=Path, required=True, help="Directory containing ground-truth masks or annotation JSON files.")
    parser.add_argument("--metrics-out", type=Path, required=True, help="Path to the per-image CSV output.")
    parser.add_argument("--summary-out", type=Path, required=True, help="Path to the JSON summary output.")
    parser.add_argument("--report-template", type=Path, required=True, help="Path to the LaTeX template.")
    parser.add_argument("--report-out", type=Path, required=True, help="Path to the filled LaTeX output.")
    parser.add_argument("--figures-dir", type=Path, required=True, help="Directory for rendered figures.")
    parser.add_argument(
        "--filename-contains",
        default="",
        help="Optional case-insensitive filename substring filter applied to prediction files before matching.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args()


def main() -> None:
    """Run mask or annotation evaluation, statistics, figure generation, and LaTeX rendering."""

    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(levelname)s] %(message)s")

    ensure_directories(args.metrics_out.parent, args.summary_out.parent, args.figures_dir, args.report_out.parent)

    if not args.pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {args.pred_dir}")
    if not args.gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory does not exist: {args.gt_dir}")

    pairs, missing = pair_prediction_and_ground_truth(args.pred_dir, args.gt_dir)
    if args.filename_contains:
        needle = args.filename_contains.lower()
        pairs = [(pred_path, gt_path) for pred_path, gt_path in pairs if needle in pred_path.name.lower()]
        missing = [name for name in missing if needle in name.lower()]
    if missing:
        LOGGER.warning("Missing GT files for %d prediction(s): %s", len(missing), ", ".join(missing))
    if not pairs:
        raise RuntimeError("No matching prediction / ground-truth pairs were found.")

    LOGGER.info("Evaluating %d image pairs", len(pairs))

    rows: list[dict[str, object]] = []
    comparison_mode = "annotations" if pairs[0][0].suffix.lower() == ".json" else "masks"
    for index, (pred_path, gt_path) in enumerate(pairs, start=1):
        LOGGER.info("Processing %d/%d: %s", index, len(pairs), pred_path.name)

        if pred_path.suffix.lower() == ".json" and gt_path.suffix.lower() == ".json":
            pred_annotation = load_annotation(pred_path)
            gt_annotation = load_annotation(gt_path)
            pred_mask, pred_meta = annotation_to_mask(pred_annotation, "prediction")
            gt_mask, gt_meta = annotation_to_mask(gt_annotation, "ground_truth")
        else:
            pred_mask, pred_meta = load_binary_mask(pred_path)
            gt_mask, gt_meta = load_binary_mask(gt_path)
        pred_mask, gt_mask, alignment = align_mask_shapes(pred_mask, gt_mask)
        validate_masks(pred_mask, gt_mask, pred_path, gt_path)

        metrics = metric_bundle(pred_mask, gt_mask)
        prediction_pixels = int(pred_mask.sum())
        ground_truth_pixels = int(gt_mask.sum())
        relative_area_error_pct = (
            abs(prediction_pixels - ground_truth_pixels) / max(ground_truth_pixels, 1) * 100.0
        )

        rows.append(
            {
                "image_id": pred_path.stem,
                "prediction_file": pred_path.name,
                "ground_truth_file": gt_path.name,
                "prediction_source_image": pred_meta.get("image_path", ""),
                "ground_truth_source_image": gt_meta.get("image_path", ""),
                "prediction_polygon_count": int(pred_meta.get("polygon_count", 0)),
                "ground_truth_polygon_count": int(gt_meta.get("polygon_count", 0)),
                "prediction_vertex_count": int(pred_meta.get("vertex_count", 0)),
                "ground_truth_vertex_count": int(gt_meta.get("vertex_count", 0)),
                "prediction_pixels": prediction_pixels,
                "ground_truth_pixels": ground_truth_pixels,
                "relative_area_error_pct": float(relative_area_error_pct),
                "prediction_empty": bool(prediction_pixels == 0),
                "ground_truth_empty": bool(ground_truth_pixels == 0),
                "shape_aligned": bool(alignment["shape_aligned"]),
                "prediction_shape": "x".join(str(value) for value in alignment["prediction_shape"]),
                "ground_truth_shape": "x".join(str(value) for value in alignment["ground_truth_shape"]),
                **metrics,
            }
        )

    dataframe = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    dataframe.to_csv(args.metrics_out, index=False)
    LOGGER.info("Saved per-image metrics to %s", args.metrics_out)

    summary = summarise_metrics(dataframe, METRIC_COLUMNS)
    summary["comparison_mode"] = comparison_mode
    summary["requested_pairs"] = int(len(pairs) + len(missing))
    summary["missing_pairs"] = int(len(missing))
    summary["missing_prediction_to_gt_matches"] = missing
    summary["area_statistics"] = {
        "mean_predicted_area_px": float(dataframe["prediction_pixels"].mean()),
        "mean_ground_truth_area_px": float(dataframe["ground_truth_pixels"].mean()),
        "mean_relative_area_error_pct": float(dataframe["relative_area_error_pct"].mean()),
        "median_relative_area_error_pct": float(dataframe["relative_area_error_pct"].median()),
        "shape_alignment_rate": float(dataframe["shape_aligned"].mean()),
    }
    save_summary(summary, args.summary_out)
    LOGGER.info("Saved summary statistics to %s", args.summary_out)

    render_metric_histograms(dataframe, METRIC_COLUMNS, args.figures_dir)
    render_confusion_barplot(dataframe, args.figures_dir)
    render_area_error_plot(dataframe, args.figures_dir)
    LOGGER.info("Rendered figures to %s", args.figures_dir)

    render_report_template(args.report_template, args.report_out, summary)
    LOGGER.info("Rendered LaTeX report to %s", args.report_out)


if __name__ == "__main__":
    main()
