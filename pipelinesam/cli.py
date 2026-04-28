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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


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
    img_pil = Image.open(img_path).convert("RGB")
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


def run_batch(
    input_dir: Path,
    output_zip: Path,
    threshold: float = 0.5,
    metadata_csv: Path | None = None,
) -> None:
    if metadata_csv and metadata_csv.exists():
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
        tasks = [(p, {"image_path": str(p.relative_to(input_dir))}) for p in paths]

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

    df = pd.DataFrame(results)

    # Growth-rate calculations when date metadata is present
    if "image_date" in df.columns and "experiment_date" in df.columns:
        df["image_date"] = pd.to_datetime(df["image_date"], errors="coerce")
        df["experiment_date"] = pd.to_datetime(df["experiment_date"], errors="coerce")
        df["days_since_start"] = (df["image_date"] - df["experiment_date"]).dt.days
        sort_cols = (
            ["experiment_name", "image_date"]
            if "experiment_name" in df.columns
            else ["image_date"]
        )
        df = df.sort_values(sort_cols)
        df["rgr_per_day"] = float("nan")
        df["relative_growth_per_day"] = float("nan")
        if "experiment_name" in df.columns:
            for _, g in df.groupby("experiment_name"):
                g = g.sort_values("image_date")
                for j in range(1, len(g)):
                    dt = (g.iloc[j]["image_date"] - g.iloc[j - 1]["image_date"]).days
                    a1 = g.iloc[j - 1].get("area_mm2") or 0
                    a2 = g.iloc[j].get("area_mm2") or 0
                    if dt > 0 and a1 > 0 and a2 > 0:
                        df.loc[g.index[j], "rgr_per_day"] = (
                            math.log(float(a2)) - math.log(float(a1))
                        ) / dt
                        df.loc[g.index[j], "relative_growth_per_day"] = (
                            float(a2) - float(a1)
                        ) / dt

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

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_p, "analysis_full.csv")
        zf.write(json_p, "analysis_full.json")
        for f in sorted(overlays_dir.glob("*.jpg")):
            zf.write(f, f"overlays/{f.name}")

    ok = sum(1 for r in results if "error" not in r)
    print(f"\n✓  {ok}/{len(results)} images analysed")
    print(f"✓  Output: {output_zip}")


def main() -> None:
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
        help="Path to image_metadata.csv",
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

    metadata_csv = (
        Path(args.metadata).expanduser().resolve()
        if args.metadata
        else input_dir / "image_metadata.csv"
    )

    try:
        run_batch(input_dir, output_zip, threshold=args.threshold, metadata_csv=metadata_csv)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
