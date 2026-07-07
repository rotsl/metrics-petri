#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# pipeline.py — SmallUNet sample analysis using models/best_area_w_0.7.pt

import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from skimage import exposure, filters, measure, morphology
from skimage.filters import frangi, meijering, threshold_local
from skimage.measure import shannon_entropy

from metrics_petri._model import _select_device
from metrics_petri._paths import (
    _DEFAULT_MODEL_CANDIDATES,
    _HF_FILE,
    _HF_REPO,
    _find_model_path,
    _verify_model_checksum,
    _verify_model_if_managed,
)
from metrics_petri.pipelinesam.model_small_unet import SmallUNet

# ---------------- CONFIG ----------------
IMG_DIR = Path(os.getenv('IMG_DIR', '.')).resolve()
OUT_DIR = IMG_DIR
METADATA_CSV = IMG_DIR / 'image_metadata.csv'
CONTAINER_MM = 90.0

IMAGE_SIZE = 256
MASK_THRESHOLD = float(os.getenv('MASK_THRESHOLD', '0.5'))
DEVICE = _select_device()


def _resolve_model_path() -> Path:
    """Return model path, auto-downloading from HuggingFace if not found locally."""
    p = _find_model_path()
    if p:
        return _verify_model_if_managed(p)
    try:
        from huggingface_hub import hf_hub_download
        print(f'[UNet] downloading checkpoint from HuggingFace ({_HF_REPO})…', flush=True)
        cached = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE)
        return _verify_model_checksum(Path(cached))
    except ValueError:
        raise
    except Exception as exc:
        raise FileNotFoundError(
            'UNet checkpoint not found. Set UNET_MODEL=/path/to/best_area_w_0.7.pt '
            'or run: make download-model\n'
            f'HuggingFace download also failed: {exc}'
        ) from exc


def __getattr__(name: str):
    """Resolve deprecated module attributes lazily for notebook compatibility."""
    if name == "MODEL_PATH":
        warnings.warn(
            "MODEL_PATH is deprecated; use _resolve_model_path() when the model is needed",
            DeprecationWarning,
            stacklevel=2,
        )
        return _find_model_path() or _DEFAULT_MODEL_CANDIDATES[-1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ---------------- MODEL (lazy singleton) ----------------
_model: SmallUNet | None = None


def get_model() -> SmallUNet:
    global _model
    if _model is None:
        model_path = _resolve_model_path()
        print(f"[UNet] loading {model_path} on {DEVICE}")
        m = SmallUNet(in_channels=3, out_channels=1, base_channels=16)
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        m.load_state_dict(state_dict, strict=True)
        m.eval()
        m.to(DEVICE)
        _model = m
    return _model


def infer_mask(model: SmallUNet, img_rgb: np.ndarray, threshold: float = MASK_THRESHOLD) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    x = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    x = x.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        raw = model(x)[0, 0].detach().cpu().numpy()
    binary = (raw > threshold).astype(np.uint8)
    return cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


# ---------------- HELPERS ----------------
def circle_mask(w, h, cx, cy, r):
    yy, xx = np.ogrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def detect_container(gray):
    h, w = gray.shape
    blur = cv2.medianBlur(gray, 9)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2, min(h, w) // 2,
                               param1=100, param2=30,
                               minRadius=int(min(h, w) * 0.35),
                               maxRadius=int(min(h, w) * 0.55))
    if circles is not None:
        x, y, r = circles[0][0]
        return float(x), float(y), float(r)
    return w / 2, h / 2, min(h, w) * 0.45


def build_mask(gray, cx, cy, r):
    container = circle_mask(gray.shape[1], gray.shape[0], cx, cy, r * 0.95)
    # Otsu on container pixels only so the dark background outside does not skew the threshold
    container_vals = gray[container]
    t = filters.threshold_otsu(container_vals)
    m = (gray > t) & container
    m = morphology.remove_small_objects(m, 500)
    return morphology.opening(m, morphology.disk(3))


def largest_component(m):
    L = measure.label(m)
    if L.max() == 0:
        return m
    props = measure.regionprops(L)
    return L == max(props, key=lambda r: r.area).label


def get_radius(props):
    return getattr(props, 'equivalent_diameter_area', props.equivalent_diameter) / 2


def _compute_texture_metrics(
    gray: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    """Return entropy and intensity standard deviation within the colony mask."""
    pixels = np.asarray(gray)[np.asarray(mask, dtype=bool)]
    if pixels.size == 0:
        return 0.0, 0.0
    return float(shannon_entropy(pixels)), float(pixels.std())


# ---------------- CORE ANALYSIS ----------------
def analyze_image(img_path: Path, meta: dict):
    model = get_model()
    img_pil = Image.open(img_path).convert('RGB')
    img = np.array(img_pil)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img.shape[:2]

    cx, cy, r_container = detect_container(gray)
    px_to_mm = (CONTAINER_MM / 2) / max(r_container, 1)
    container_mask = circle_mask(w, h, cx, cy, r_container)

    unet_mask = infer_mask(model, img) & container_mask
    if not np.any(unet_mask):
        classical = build_mask(gray, cx, cy, r_container) & container_mask
        unet_mask = largest_component(classical)

    mask = largest_component(unet_mask)

    props = measure.regionprops(mask.astype(int))[0]
    area_px = props.area
    area_mm2 = area_px * px_to_mm ** 2
    radius_px = get_radius(props)
    diameter_mm = 2 * radius_px * px_to_mm
    perimeter_mm = props.perimeter * px_to_mm
    eccentricity = props.eccentricity
    cy_c, cx_c = props.centroid
    centre_delta_mm = np.hypot(cx_c - cx, cy_c - cy) * px_to_mm
    edge_roughness = props.perimeter / (2 * np.pi * radius_px) if radius_px > 0 else np.nan

    entropy, texture_std = _compute_texture_metrics(gray, mask)

    corr = cv2.divide(gray, cv2.GaussianBlur(gray, (0, 0), 50), scale=255)
    filled = ndimage.binary_fill_holes(mask)
    local_t = threshold_local(corr, 51, offset=5)
    crack = (corr < local_t) & filled
    crack = morphology.remove_small_objects(crack, 15)
    crack_px = int(crack.sum())
    crack_area_mm2 = crack_px * px_to_mm ** 2
    crack_coverage_pct = 100 * crack_px / area_px if area_px else 0
    crack_count = int(measure.label(crack).max())

    eq = exposure.equalize_adapthist(corr / 255.)
    mej = meijering(eq, sigmas=range(1, 4), black_ridges=False)
    fra = frangi(eq, sigmas=range(1, 4), black_ridges=False)
    hyb = mej * (fra > 0.5 * filters.threshold_otsu(fra))

    def skel(a):
        if not np.any(a):
            return np.zeros_like(a, bool)
        t = filters.threshold_otsu(a[a > 0]) if np.any(a > 0) else 0.1
        return morphology.skeletonize(morphology.remove_small_objects(a > 0.8 * t, 20))

    sk_f, sk_m, sk_h = skel(fra), skel(mej), skel(hyb)
    hyph_frangi_px, hyph_meijering_px, hyph_hybrid_px = map(int, [sk_f.sum(), sk_m.sum(), sk_h.sum()])
    hyph_frangi_mm = hyph_frangi_px * px_to_mm
    hyph_meijering_mm = hyph_meijering_px * px_to_mm
    hyph_hybrid_mm = hyph_hybrid_px * px_to_mm

    return {
        **meta,
        'px_to_mm': px_to_mm,
        'area_mm2': area_mm2,
        'diameter_mm': diameter_mm,
        'perimeter_mm': perimeter_mm,
        'eccentricity': eccentricity,
        'edge_roughness': edge_roughness,
        'centre_delta_mm': centre_delta_mm,
        'entropy': entropy,
        'texture_std': texture_std,
        'crack_px': crack_px,
        'crack_area_mm2': crack_area_mm2,
        'crack_coverage_pct': crack_coverage_pct,
        'crack_count': crack_count,
        'hyph_frangi_px': hyph_frangi_px,
        'hyph_meijering_px': hyph_meijering_px,
        'hyph_hybrid_px': hyph_hybrid_px,
        'hyph_frangi_mm': hyph_frangi_mm,
        'hyph_meijering_mm': hyph_meijering_mm,
        'hyph_hybrid_mm': hyph_hybrid_mm,
    }


# ---------------- MAIN ----------------
def main():
    from metrics_petri.pipelinesam.cli import _build_metadata_tasks, _load_metadata

    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Missing {METADATA_CSV} — run input.py first")

    meta_df = _load_metadata(METADATA_CSV)
    tasks = _build_metadata_tasks(IMG_DIR, meta_df)

    print(f"Processing {len(tasks)} images on {DEVICE}...")

    results = []
    for i, (p, m) in enumerate(tasks, 1):
        results.append(analyze_image(p, m))
        print(f"[{i}/{len(tasks)}] {results[-1]['image_path']}")

    df = pd.DataFrame(results)
    df['image_date'] = pd.to_datetime(df['image_date'], errors='coerce')
    df['experiment_date'] = pd.to_datetime(df['experiment_date'], errors='coerce')
    df['days_since_start'] = (df['image_date'] - df['experiment_date']).dt.days

    df = df.sort_values(['experiment_name', 'image_date'])
    df['rgr_per_day'] = np.nan
    df['relative_growth_per_day'] = np.nan

    for exp, g in df.groupby('experiment_name'):
        g = g.sort_values('image_date')
        for i in range(1, len(g)):
            dt = (g.iloc[i]['image_date'] - g.iloc[i - 1]['image_date']).days
            if dt > 0 and g.iloc[i - 1]['area_mm2'] > 0:
                rgr = (np.log(g.iloc[i]['area_mm2']) - np.log(g.iloc[i - 1]['area_mm2'])) / dt
                rel = (g.iloc[i]['area_mm2'] - g.iloc[i - 1]['area_mm2']) / dt
                df.loc[g.index[i], 'rgr_per_day'] = rgr
                df.loc[g.index[i], 'relative_growth_per_day'] = rel

    csv_out = OUT_DIR / 'analysis_full.csv'
    json_out = OUT_DIR / 'analysis_full.json'
    df.to_csv(csv_out, index=False)
    df.to_json(json_out, orient='records', indent=2, date_format='iso')
    print(f"\n✓ Saved {csv_out}\n✓ Saved {json_out}")


if __name__ == '__main__':
    main()
