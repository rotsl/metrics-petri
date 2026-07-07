"""Tests for safe Gradio GUI command-line defaults."""

import pytest

from metrics_petri.pipeline.cli import _validate_exposure, build_parser


def test_gui_defaults_to_loopback_without_authentication():
    args = build_parser().parse_args([])

    assert args.host == "127.0.0.1"
    assert args.port == 7860
    assert args.auth is None


def test_gui_auth_parses_username_and_password():
    args = build_parser().parse_args(["--auth", "researcher:secret"])

    assert args.auth == ("researcher", "secret")


@pytest.mark.parametrize("value", ["missing-colon", ":password", "username:"])
def test_gui_auth_rejects_incomplete_credentials(value):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--auth", value])


def test_gui_rejects_exposure_without_authentication():
    with pytest.raises(ValueError, match="without authentication"):
        _validate_exposure("0.0.0.0", None)


def test_gui_allows_exposure_with_authentication():
    _validate_exposure("0.0.0.0", ("researcher", "secret"))


def test_gui_allows_loopback_without_authentication():
    _validate_exposure("127.0.0.1", None)
