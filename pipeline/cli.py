"""CLI entry point for the metrics-petri Gradio GUI."""

from __future__ import annotations

import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metrics-petri-gui",
        description="Launch the metrics-petri Gradio web interface.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Bind port (default: 7860)")
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open browser automatically"
    )
    parser.add_argument("--model", default=None, help="Path to SmallUNet checkpoint (.pt)")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.model:
        os.environ["UNET_MODEL"] = args.model

    from .app import demo, CSS

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        inbrowser=not args.no_browser,
        css=CSS,
    )


if __name__ == "__main__":
    main()
