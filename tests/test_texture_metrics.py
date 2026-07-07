"""Tests for colony-scoped image texture metrics."""

import numpy as np

from metrics_petri.pipelinesam.pipeline import _compute_texture_metrics


def test_texture_metrics_ignore_background_pixels():
    mask = np.array([[True, False], [True, False]])
    first = np.array([[10, 0], [20, 0]], dtype=np.uint8)
    changed_background = np.array([[10, 100], [20, 255]], dtype=np.uint8)

    first_metrics = _compute_texture_metrics(first, mask)
    changed_metrics = _compute_texture_metrics(changed_background, mask)

    assert first_metrics == changed_metrics
    assert first_metrics[1] == 5.0


def test_texture_metrics_return_zero_for_empty_mask():
    gray = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    mask = np.zeros_like(gray, dtype=bool)

    assert _compute_texture_metrics(gray, mask) == (0.0, 0.0)
