#!/usr/bin/env python3
"""metrics-petri CLI — batch pipeline without GUI.

Usage:
    metrics-petri [INPUT_DIR] [--output OUT.zip] [--threshold 0.5]
                  [--metadata image_metadata.csv] [--model path/to/model.pt]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage import filters, measure, morphology
from skimage.filters import threshold_local

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".gif",
    ".heic", ".heif", ".dng", ".cr2", ".nef", ".arw", ".raf", ".orf", ".rw2",
}


def _detect_image_date(path: Path) -> str:
    """Return ISO date string from EXIF DateTimeOriginal or filename YYYYMMDD pattern.

    Returns empty string when no reliable date can be found. File mtime is not
    used — it reflects copy/download time, not capture time.
    """
    import re as _re
    from PIL import ExifTags as _ExifTags
    _DATE_RE = _re.compile(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])")
    # 1. EXIF DateTimeOriginal
    try:
        import datetime as _dt
        im = Image.open(path)
        ex = im.getexif()
        if ex:
            for tid, tn in _ExifTags.TAGS.items():
                if tn == "DateTimeOriginal":
                    v = ex.get(tid)
                    if v:
                        return _dt.datetime.strptime(v, "%Y:%m:%d %H:%M:%S").date().isoformat()
    except Exception:
        pass
    # 2. YYYYMMDD in filename
    m = _DATE_RE.search(path.stem)
    if m:
        try:
            import datetime as _dt
            return _dt.date(int(m[1]), int(m[2]), int(m[3])).isoformat()
        except Exception:
            pass
    return ""


def _find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _load_as_pil(path: Path) -> Image.Image:
    """Open any supported image format and return an RGB PIL image."""
    suffix = path.suffix.lower()
    if suffix in {".heic", ".heif"}:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            pass
    if suffix in {".dng", ".cr2", ".nef", ".arw", ".raf", ".orf", ".rw2", ".raw"}:
        try:
            import rawpy
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess()
            return Image.fromarray(rgb)
        except Exception:
            pass
    return Image.open(path).convert("RGB")


def _generate_overlays(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    crack: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    stem: str,
) -> dict[str, Image.Image]:
    """Return {filename: PIL.Image} for each overlay panel."""
    # Raw + dish circle
    p_raw = img_rgb.copy()
    cv2.circle(p_raw, (int(cx), int(cy)), int(r), (0, 255, 0), 3)

    # Binary mask (white on black)
    p_bin = np.zeros_like(img_rgb)
    p_bin[mask] = [255, 255, 255]

    # Colony overlay (red tint)
    p_col = img_rgb.copy()
    if mask.any():
        p_col[mask] = (
            p_col[mask].astype(np.float32) * 0.5
            + np.array([255, 0, 0], dtype=np.float32) * 0.5
        ).astype(np.uint8)
    cv2.circle(p_col, (int(cx), int(cy)), int(r), (0, 255, 0), 2)

    # Crack overlay (yellow tint)
    p_crk = img_rgb.copy()
    if crack.any():
        ck = (
            cv2.dilate(
                crack.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            > 0
        )
        p_crk[ck] = (
            p_crk[ck].astype(np.float32) * 0.3
            + np.array([255, 255, 0], dtype=np.float32) * 0.7
        ).astype(np.uint8)

    return {
        f"{stem}_raw_dish.jpg": Image.fromarray(p_raw),
        f"{stem}_mask.jpg": Image.fromarray(p_bin),
        f"{stem}_colony.jpg": Image.fromarray(p_col),
        f"{stem}_cracks.jpg": Image.fromarray(p_crk),
    }


def _process_image(
    img_path: Path,
    meta: dict,
    threshold: float,
) -> tuple[dict, dict[str, Image.Image]]:
    """Run full pipeline on one image; return (metrics_row, overlay_images)."""
    from .pipeline import (
        CONTAINER_MM,
        build_mask,
        circle_mask,
        detect_container,
        get_model,
        get_radius,
        infer_mask,
        largest_component,
    )
    from skimage.filters import frangi, meijering
    from skimage.measure import shannon_entropy
    from skimage import exposure

    model = get_model()
    img_pil = _load_as_pil(img_path)
    img = np.array(img_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img.shape[:2]

    cx, cy, r_cont = detect_container(gray)
    px_to_mm = (CONTAINER_MM / 2) / max(r_cont, 1)
    cont_mask = circle_mask(w, h, cx, cy, r_cont)

    unet = infer_mask(model, img, threshold=threshold) & cont_mask
    if not unet.any():
        unet = largest_component(build_mask(gray, cx, cy, r_cont) & cont_mask)

    mask = largest_component(unet)

    # Crack detection
    corr = cv2.divide(gray, cv2.GaussianBlur(gray, (0, 0), 50), scale=255)
    filled = ndimage.binary_fill_holes(mask)
    local_t = threshold_local(corr, 51, offset=5)
    crack = morphology.remove_small_objects((corr < local_t) & filled, 15)

    # Morphometrics
    props = measure.regionprops(mask.astype(int))[0]
    area_px = props.area
    area_mm2 = area_px * px_to_mm**2
    radius_px = get_radius(props)
    diameter_mm = 2 * radius_px * px_to_mm
    perimeter_mm = props.perimeter * px_to_mm
    eccentricity = props.eccentricity
    cy_c, cx_c = props.centroid
    centre_delta_mm = float(np.hypot(cx_c - cx, cy_c - cy) * px_to_mm)
    edge_roughness = props.perimeter / (2 * np.pi * radius_px) if radius_px > 0 else float("nan")
    entropy = float(shannon_entropy(gray))
    texture_std = float(gray.std())

    crack_px = int(crack.sum())
    crack_area_mm2 = crack_px * px_to_mm**2
    crack_cov_pct = 100.0 * crack_px / area_px if area_px else 0.0
    crack_count = int(measure.label(crack).max())

    # Hyphae (frangi / meijering)
    eq = exposure.equalize_adapthist(corr / 255.0)
    fra = frangi(eq, sigmas=range(1, 4), black_ridges=False)
    mej = meijering(eq, sigmas=range(1, 4), black_ridges=False)
    hyb = mej * (fra > 0.5 * filters.threshold_otsu(fra) if fra.any() else 0)

    def _skel(a: np.ndarray) -> np.ndarray:
        if not a.any():
            return np.zeros_like(a, dtype=bool)
        t = filters.threshold_otsu(a[a > 0]) if (a > 0).any() else 0.1
        return morphology.skeletonize(morphology.remove_small_objects(a > 0.8 * t, 20))

    sk_f, sk_m, sk_h = _skel(fra), _skel(mej), _skel(hyb)

    metrics = {
        **meta,
        "px_to_mm": px_to_mm,
        "area_mm2": round(area_mm2, 4),
        "diameter_mm": round(diameter_mm, 4),
        "perimeter_mm": round(perimeter_mm, 4),
        "eccentricity": round(eccentricity, 6),
        "edge_roughness": round(edge_roughness, 6) if not math.isnan(edge_roughness) else "",
        "centre_delta_mm": round(centre_delta_mm, 4),
        "entropy": round(entropy, 6),
        "texture_std": round(float(gray.std()), 6),
        "crack_px": crack_px,
        "crack_area_mm2": round(crack_area_mm2, 6),
        "crack_coverage_pct": round(crack_cov_pct, 4),
        "crack_count": crack_count,
        "hyph_frangi_px": int(sk_f.sum()),
        "hyph_meijering_px": int(sk_m.sum()),
        "hyph_hybrid_px": int(sk_h.sum()),
        "hyph_frangi_mm": round(int(sk_f.sum()) * px_to_mm, 4),
        "hyph_meijering_mm": round(int(sk_m.sum()) * px_to_mm, 4),
        "hyph_hybrid_mm": round(int(sk_h.sum()) * px_to_mm, 4),
    }

    overlays = _generate_overlays(img, mask, crack, cx, cy, r_cont, img_path.stem)
    return metrics, overlays


def _x_label_for(dc_val: object, day_num: int) -> str:
    dc = str(dc_val) if dc_val else ""
    if dc.startswith("d") and dc[1:].isdigit():
        return dc
    return f"d{day_num:02d}" if day_num > 0 else "d?"


def _write_charts(df: pd.DataFrame, out_dir: Path) -> None:
    """Write growth-curve PNGs to out_dir using day_code on the x-axis."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for col in ("area_mm2", "diameter_mm", "crack_coverage_pct", "crack_count",
                "edge_roughness", "entropy", "rgr_per_day", "relative_growth_per_day"):
        if col not in df.columns:
            continue
        sub = df[["days_since_start", "day_code", col]].copy() if "day_code" in df.columns \
            else df[["days_since_start", col]].copy()
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna(subset=[col])
        if len(sub) < 2:
            continue

        sort_col = "days_since_start" if "days_since_start" in sub.columns else sub.columns[0]
        sub = sub.sort_values(sort_col).reset_index(drop=True)

        x = sub[sort_col].tolist()
        dc_col = sub["day_code"] if "day_code" in sub.columns else pd.Series([""] * len(sub))
        x_labels = [_x_label_for(dc, int(d)) for dc, d in zip(dc_col, x)]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, sub[col].tolist(), "o-", linewidth=2, markersize=7, color="#4f46e5")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30 if len(x_labels) > 6 else 0, fontsize=8)
        ax.set_xlabel("Day")
        ax.set_ylabel(col.replace("_", " "))
        ax.set_title(col.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{col}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


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
        from pipelinesam.pipeline import _find_model_path
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
               "matplotlib": "matplotlib", "PIL": "Pillow", "rawpy": "rawpy"}
    missing = [pkg for mod, pkg in dep_map.items() if not _importable(mod)]
    rows.append(("Dependencies", f"⚠️  Missing: {', '.join(missing)}" if missing else "Healthy"))
    if missing:
        ok = False

    w = max(len(k) for k, _ in rows)
    for label, value in rows:
        print(f"{label:<{w}}  {value}")
    print()
    if ok:
        print("✓  All checks passed")
    else:
        print("⚠️  Issues found — see above")
        sys.exit(1)


def _importable(mod: str) -> bool:
    import importlib
    try:
        importlib.import_module(mod)
        return True
    except ImportError:
        return False


def run_batch(
    input_dir: Path,
    output_zip: Path,
    threshold: float = 0.5,
    metadata_csv: Path | None = None,
) -> None:
    if metadata_csv and metadata_csv.exists():
        if metadata_csv.suffix.lower() == ".json":
            import json as _json
            meta_df = pd.DataFrame(_json.loads(metadata_csv.read_text(encoding="utf-8")))
        else:
            meta_df = pd.read_csv(metadata_csv)
        for col in ("experiment_name", "experiment_date", "image_date", "day_code", "user_name", "plates_count"):
            if col not in meta_df.columns:
                meta_df[col] = ""
        tasks = [
            (input_dir / str(r["image_path"]), r.to_dict())
            for _, r in meta_df.iterrows()
            if (input_dir / str(r["image_path"])).exists()
        ]
    else:
        paths = _find_images(input_dir)
        # Auto-detect image dates; anchor experiment_date to earliest found date
        raw_dates = {p: _detect_image_date(p) for p in paths}
        known = [d for d in raw_dates.values() if d]
        auto_exp_date = min(known) if known else ""
        tasks = [
            (
                p,
                {
                    "image_path": str(p.relative_to(input_dir)),
                    "image_date": raw_dates[p],
                    "experiment_date": auto_exp_date,
                },
            )
            for p in paths
        ]

    if not tasks:
        raise FileNotFoundError(f"No images found in {input_dir}")

    print(f"Processing {len(tasks)} image(s) …", flush=True)

    results: list[dict] = []
    all_overlays: dict[str, Image.Image] = {}

    for i, (img_path, meta) in enumerate(tasks, 1):
        print(f"  [{i}/{len(tasks)}] {img_path.name}", flush=True)
        try:
            row, overlays = _process_image(img_path, meta, threshold)
            results.append(row)
            all_overlays.update(overlays)
        except Exception as exc:
            print(f"    WARNING: {exc}", file=sys.stderr, flush=True)
            results.append({**meta, "error": str(exc)})

    def _dc_to_num(s: object, fallback: int = 0) -> int:
        if isinstance(s, str) and s.startswith("d") and s[1:].isdigit():
            return int(s[1:])
        return fallback

    df = pd.DataFrame(results)

    # days_since_start: prefer numeric value from day_code column (set by the user
    # in the GUI or metadata CSV); fall back to (image_date - experiment_date) arithmetic.
    if "day_code" in df.columns:
        dc_nums = df["day_code"].apply(_dc_to_num)
        df["days_since_start"] = dc_nums
    elif "image_date" in df.columns and "experiment_date" in df.columns:
        _id = pd.to_datetime(df["image_date"], errors="coerce")
        _ed = pd.to_datetime(df["experiment_date"], errors="coerce")
        df["days_since_start"] = (_id - _ed).dt.days.fillna(0).astype(int) + 1

    # Growth-rate calculations using actual calendar-day intervals between images
    if "image_date" in df.columns and "experiment_date" in df.columns:
        df["image_date"] = pd.to_datetime(df["image_date"], errors="coerce")
        df["experiment_date"] = pd.to_datetime(df["experiment_date"], errors="coerce")
        # Sort by day_code numeric value so rows are in chronological order
        if "days_since_start" in df.columns:
            sort_cols = (
                ["experiment_name", "days_since_start"]
                if "experiment_name" in df.columns
                else ["days_since_start"]
            )
        else:
            sort_cols = (
                ["experiment_name", "image_date"]
                if "experiment_name" in df.columns
                else ["image_date"]
            )
        df = df.sort_values(sort_cols)
        df["rgr_per_day"] = float("nan")
        df["relative_growth_per_day"] = float("nan")
        grp_col = "experiment_name" if "experiment_name" in df.columns else None
        groups = df.groupby(grp_col) if grp_col else [(None, df)]
        for _, g in groups:
            g = g.sort_values("days_since_start" if "days_since_start" in g.columns else "image_date")
            for j in range(1, len(g)):
                # Use actual calendar days for rate math, not just day-code difference
                try:
                    dd = int((g.iloc[j]["image_date"] - g.iloc[j - 1]["image_date"]).days)
                except Exception:
                    dd = int(g.iloc[j].get("days_since_start", j) - g.iloc[j - 1].get("days_since_start", j - 1))
                a1 = g.iloc[j - 1].get("area_mm2") or 0
                a2 = g.iloc[j].get("area_mm2") or 0
                if dd > 0 and a1 > 0 and a2 > 0:
                    df.loc[g.index[j], "rgr_per_day"] = (
                        math.log(float(a2)) - math.log(float(a1))
                    ) / dd
                    df.loc[g.index[j], "relative_growth_per_day"] = (
                        float(a2) - float(a1)
                    ) / dd

    # Write zip
    tmp = Path(tempfile.mkdtemp())
    csv_p = tmp / "analysis_full.csv"
    json_p = tmp / "analysis_full.json"
    df.to_csv(csv_p, index=False)
    df.to_json(json_p, orient="records", indent=2, date_format="iso")

    overlays_dir = tmp / "overlays"
    overlays_dir.mkdir()
    for name, img_pil in all_overlays.items():
        img_pil.save(overlays_dir / name, quality=92)

    charts_dir = tmp / "charts"
    _write_charts(df, charts_dir)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_p, "analysis_full.csv")
        zf.write(json_p, "analysis_full.json")
        for f in sorted(overlays_dir.glob("*.jpg")):
            zf.write(f, f"overlays/{f.name}")
        for f in sorted(charts_dir.glob("*.png")):
            zf.write(f, f"charts/{f.name}")

    ok = sum(1 for r in results if "error" not in r)
    print(f"\n✓  {ok}/{len(results)} images analysed")
    print(f"✓  Output: {output_zip}")


def main() -> None:
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "doctor":
        _run_doctor()
        return
    parser = argparse.ArgumentParser(
        prog="metrics-petri",
        description="metrics-petri — petri dish colony analysis (no GUI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Folder containing images (default: current directory)",
    )
    parser.add_argument("--output", "-o", default=None, help="Output zip file path")
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.5,
        help="Mask confidence threshold (0–1)",
    )
    parser.add_argument(
        "--metadata",
        "-m",
        default=None,
        help="Path to image_metadata.csv or image_metadata.json",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to SmallUNet checkpoint (.pt)",
    )
    args = parser.parse_args()

    if args.model:
        os.environ["UNET_MODEL"] = args.model

    input_dir = Path(args.input).expanduser().resolve() if args.input else Path.cwd()
    if not input_dir.is_dir():
        print(f"error: '{input_dir}' is not a directory", file=sys.stderr)
        sys.exit(1)

    output_zip = (
        Path(args.output).expanduser().resolve() if args.output else input_dir / "analysis.zip"
    )
    if output_zip.suffix.lower() != ".zip":
        output_zip = output_zip.with_suffix(".zip")

    if args.metadata:
        metadata_csv = Path(args.metadata).expanduser().resolve()
    elif (input_dir / "image_metadata.json").exists():
        metadata_csv = input_dir / "image_metadata.json"
    else:
        metadata_csv = input_dir / "image_metadata.csv"

    try:
        run_batch(input_dir, output_zip, threshold=args.threshold, metadata_csv=metadata_csv)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
