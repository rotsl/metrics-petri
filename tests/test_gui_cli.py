"""Tests for safe Gradio GUI command-line defaults."""

import warnings

import pytest

from metrics_petri.pipeline.cli import _warn_if_exposed, build_parser


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


def test_gui_warns_when_exposed_without_authentication():
    with pytest.warns(RuntimeWarning, match="without authentication"):
        _warn_if_exposed("0.0.0.0", None)


def test_gui_does_not_warn_when_exposed_with_authentication():
    with warnings.catch_warnings(record=True) as warnings_seen:
        warnings.simplefilter("always")
        _warn_if_exposed("0.0.0.0", ("researcher", "secret"))

    assert not warnings_seen
