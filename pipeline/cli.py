"""CLI entry point for the metrics-petri Gradio GUI."""

from __future__ import annotations

import argparse
import os


def _importable(mod: str) -> bool:
    import importlib
    try:
        importlib.import_module(mod)
        return True
    except ImportError:
        return False


def _run_doctor() -> None:
    import sys
    rows: list[tuple[str, str]] = []
    ok = True

    v = sys.version_info
    rows.append(("Python", f"{v.major}.{v.minor}.{v.micro}"))

    try:
        import numpy as np
        nv = np.__version__
        if int(nv.split(".")[0]) >= 2:
            rows.append(("NumPy", f"{nv}  ⚠️  NumPy 2.x conflicts with PyTorch — run: pip install 'numpy<2'"))
            ok = False
        else:
            rows.append(("NumPy", nv))
    except ImportError:
        rows.append(("NumPy", "MISSING"))
        ok = False

    try:
        import torch
        rows.append(("Torch", f"{torch.__version__} — OK"))
        if torch.backends.mps.is_available():
            rows.append(("MPS", "Available"))
        elif torch.cuda.is_available():
            rows.append(("CUDA", "Available"))
        else:
            rows.append(("Accelerator", "None (CPU)"))
    except ImportError:
        rows.append(("Torch", "MISSING"))
        ok = False

    try:
        import gradio as gr
        rows.append(("Gradio", f"{gr.__version__} — OK"))
    except ImportError:
        rows.append(("Gradio", "MISSING"))
        ok = False

    try:
        from pipeline.analysis import _find_model_path
        p = _find_model_path()
        if p and p.exists():
            mb = p.stat().st_size / 1_048_576
            rows.append(("Model", f"{p} ({mb:.1f} MB) — OK"))
        else:
            rows.append(("Model", "Not found locally — will auto-download from HuggingFace on first run"))
    except Exception as exc:
        rows.append(("Model", f"Error: {exc}"))

    dep_map = {"pandas": "pandas", "cv2": "opencv-python-headless",
               "skimage": "scikit-image", "scipy": "scipy",
               "matplotlib": "matplotlib", "PIL": "Pillow",
               "rawpy": "rawpy", "pillow_heif": "pillow-heif"}
    missing = [pkg for mod, pkg in dep_map.items() if not _importable(mod)]
    rows.append(("Dependencies", f"⚠️  Missing: {', '.join(missing)}" if missing else "Healthy"))
    if missing:
        ok = False

    w = max(len(k) for k, _ in rows)
    for label, value in rows:
        print(f"{label:<{w}}  {value}")
    print()
    if not ok:
        print("⚠️  Issues found — see above")
        sys.exit(1)
    print("✓  All checks passed")


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
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "doctor":
        _run_doctor()
        return

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
