from __future__ import annotations

import argparse

from .launcher import run_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Metrics Petri as a packaged app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--skip-model-bootstrap", action="store_true")
    parser.add_argument("--skip-node-api", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_app(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        skip_model_bootstrap=args.skip_model_bootstrap or args.skip_node_api,
    )


if __name__ == "__main__":
    main()
