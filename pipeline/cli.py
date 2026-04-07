from __future__ import annotations

import argparse
import json
from pathlib import Path
from dotenv import load_dotenv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Magnaporthe growth analysis.")
    parser.add_argument("--engine", choices=["local", "gemini"], required=True)
    parser.add_argument("--input-dir", default="input_images")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-dir", default="models/gemma-4-e2b-it-MLX-4bit")
    parser.add_argument("--gemini-model", default="gemma-3-27b-it")
    parser.add_argument("--filename", action="append", dest="filenames")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main() -> None:
    load_dotenv(".env")
    parser = build_parser()
    args = parser.parse_args()
    from pipeline.analysis import run_analysis_batch

    payload = run_analysis_batch(
        engine=args.engine,
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        filenames=args.filenames,
        model_dir=Path(args.model_dir),
        gemini_model=args.gemini_model,
    )
    if args.json_output:
        print(json.dumps(payload))
        return
    print(json.dumps(payload["run"], indent=2))


if __name__ == "__main__":
    main()
