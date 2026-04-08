"""Metric definitions for binary segmentation evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfusionCounts:
    """Pixel-wise confusion counts for a binary mask pair."""

    tp: int
    fp: int
    fn: int
    tn: int


def confusion_counts(pred_mask: np.ndarray, gt_mask: np.ndarray) -> ConfusionCounts:
    """Return TP, FP, FN, and TN for two boolean masks."""

    pred = np.asarray(pred_mask, dtype=bool)
    gt = np.asarray(gt_mask, dtype=bool)

    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    tn = int(np.count_nonzero(~pred & ~gt))
    return ConfusionCounts(tp=tp, fp=fp, fn=fn, tn=tn)


def safe_divide(numerator: float, denominator: float) -> float:
    """Return a safe division result, falling back to 0.0 for zero denominators."""

    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def precision(counts: ConfusionCounts) -> float:
    """Compute precision = TP / (TP + FP)."""

    return safe_divide(counts.tp, counts.tp + counts.fp)


def recall(counts: ConfusionCounts) -> float:
    """Compute recall = TP / (TP + FN)."""

    return safe_divide(counts.tp, counts.tp + counts.fn)


def f1_score(counts: ConfusionCounts) -> float:
    """Compute F1 = 2 * (precision * recall) / (precision + recall)."""

    p = precision(counts)
    r = recall(counts)
    return safe_divide(2.0 * p * r, p + r)


def iou(counts: ConfusionCounts) -> float:
    """Compute IoU = TP / (TP + FP + FN)."""

    return safe_divide(counts.tp, counts.tp + counts.fp + counts.fn)


def metric_bundle(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float | int]:
    """Return confusion counts and metric values for one mask pair."""

    counts = confusion_counts(pred_mask, gt_mask)
    return {
        "tp": counts.tp,
        "fp": counts.fp,
        "fn": counts.fn,
        "tn": counts.tn,
        "precision": precision(counts),
        "recall": recall(counts),
        "f1_score": f1_score(counts),
        "iou": iou(counts),
    }
