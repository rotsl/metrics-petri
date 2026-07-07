"""Tests for batch CLI calibration options."""

import pytest

from metrics_petri.pipelinesam.cli import _pixel_scale, _set_seed, build_parser


def test_dish_size_defaults_to_90_mm():
    args = build_parser().parse_args([])

    assert args.dish_size_mm == 90.0


def test_dish_size_accepts_positive_value():
    args = build_parser().parse_args(["--dish-size-mm", "60"])

    assert args.dish_size_mm == 60.0


@pytest.mark.parametrize("value", ["0", "-10", "nan", "inf"])
def test_dish_size_rejects_invalid_value(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--dish-size-mm", value])


def test_pixel_scale_uses_dish_diameter():
    assert _pixel_scale(90.0, 450.0) == pytest.approx(0.1)
    assert _pixel_scale(60.0, 300.0) == pytest.approx(0.1)


def test_seed_defaults_to_zero():
    args = build_parser().parse_args([])

    assert args.seed == 0


def test_seed_accepts_non_negative_integer():
    args = build_parser().parse_args(["--seed", "123"])

    assert args.seed == 123


def test_seed_rejects_negative_integer():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--seed", "-1"])


def test_set_seed_calls_torch_manual_seed(monkeypatch):
    seen: list[int] = []

    import torch

    monkeypatch.setattr(torch, "manual_seed", lambda seed: seen.append(seed))

    _set_seed(123)

    assert seen == [123]
