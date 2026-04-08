"""Statistical summaries for segmentation metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_confidence_interval(
    values: np.ndarray,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> tuple[float, float]:
    """Estimate a bootstrap confidence interval for the mean."""

    series = np.asarray(values, dtype=float)
    if series.size == 0:
        return (0.0, 0.0)
    if series.size == 1:
        value = float(series[0])
        return (value, value)

    rng = np.random.default_rng(random_seed)
    boot_means = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = rng.choice(series, size=series.size, replace=True)
        boot_means[index] = float(sample.mean())

    alpha = 1.0 - confidence
    lower = float(np.quantile(boot_means, alpha / 2.0))
    upper = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return lower, upper


def summarise_metrics(
    dataframe: pd.DataFrame,
    metric_columns: list[str],
    confidence: float = 0.95,
) -> dict[str, object]:
    """Compute mean, standard deviation, and confidence intervals for each metric."""

    summary: dict[str, object] = {
        "num_images": int(len(dataframe)),
        "confidence_level": confidence,
        "metrics": {},
    }

    for metric in metric_columns:
        values = dataframe[metric].to_numpy(dtype=float)
        lower, upper = bootstrap_confidence_interval(values, confidence=confidence)
        summary["metrics"][metric] = {
            "mean": float(np.mean(values)) if values.size else 0.0,
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "min": float(np.min(values)) if values.size else 0.0,
            "max": float(np.max(values)) if values.size else 0.0,
            "ci_lower": lower,
            "ci_upper": upper,
        }

    return summary


def save_summary(summary: dict[str, object], output_path: Path) -> None:
    """Write summary statistics to JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
