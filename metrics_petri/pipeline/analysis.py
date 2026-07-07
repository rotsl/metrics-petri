# SPDX-License-Identifier: MIT
"""Core analysis functions for the GUI package.

Uses models/best_area_w_0.7.pt (SmallUNet) for segmentation.
Override path with env var UNET_MODEL.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage import filters, measure, morphology
from skimage.filters import frangi, meijering

from metrics_petri._model import _select_device
from metrics_petri._paths import (
    _HF_FILE,
    _HF_REPO,
    _find_model_path,
    _verify_model_checksum,
    _verify_model_if_managed,
)

from .model import SmallUNet

# ── constants ──────────────────────────────────────────────────────────────
CONTAINER_MM = 90.0
IMAGE_SIZE = 256
DEVICE = _select_device()

# ── model ──────────────────────────────────────────────────────────────────
_model: SmallUNet | None = None


def _resolve_model_path() -> Path:
    """Return model path, auto-downloading from HuggingFace if not found locally."""
    p = _find_model_path()
    if p:
        return _verify_model_if_managed(p)
    try:
        from huggingface_hub import hf_hub_download
        print(f"[UNet] downloading checkpoint from HuggingFace ({_HF_REPO})…", flush=True)
        cached = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILE)
        return _verify_model_checksum(Path(cached))
    except ValueError:
        raise
    except Exception as exc:
        raise FileNotFoundError(
            "UNet checkpoint not found. Set UNET_MODEL=/path/to/best_area_w_0.7.pt "
            "or run: make download-model\n"
            f"HuggingFace download also failed: {exc}"
        ) from exc


def load_model() -> SmallUNet:
    global _model
    if _model is None:
        p = _resolve_model_path()
        m = SmallUNet(in_channels=3, out_channels=1, base_channels=16)
        ckpt = torch.load(p, map_location=DEVICE, weights_only=True)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        m.load_state_dict(sd, strict=True)
        m.eval()
        m.to(DEVICE)
        _model = m
    return _model


def infer_mask(img_pil: Image.Image, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Return (overlay_rgb, binary_mask_uint8) for a PIL image."""
    model = load_model()
    img_arr = np.array(img_pil.convert("RGB"))
    resized = cv2.resize(img_arr, (IMAGE_SIZE, IMAGE_SIZE))
    x = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    x = x.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prob = model(x)[0, 0].detach().cpu().numpy()
    mask = (prob > threshold).astype(np.uint8) * 255
    mask = cv2.resize(mask, (img_pil.width, img_pil.height), interpolation=cv2.INTER_NEAREST)
    overlay = img_arr.copy()
    overlay[mask > 0] = (
        overlay[mask > 0].astype(np.float32) * 0.5
        + np.array([255, 0, 0], dtype=np.float32) * 0.5
    ).astype(np.uint8)
    return Image.fromarray(overlay), Image.fromarray(mask)


# ── dish detection ─────────────────────────────────────────────────────────

def detect_container(img_bgr: np.ndarray) -> tuple[int, int, int, float] | None:
    """Detect petri dish via Hough circles.

    Returns (cx, cy, radius_px, px_to_mm) or None on failure.
    """
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        h, w = gray.shape
        mn = int(min(h, w) * 0.25)
        mx = int(min(h, w) * 0.52)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min(h, w) // 2,
            param1=100,
            param2=40,
            minRadius=mn,
            maxRadius=mx,
        )
        if circles is None:
            return None
        circles = np.round(circles[0]).astype(int)
        ic, jc = w / 2, h / 2
        best_idx, best_score = 0, -1.0
        for i, (cx, cy, r) in enumerate(circles):
            score = r / (1 + math.hypot(cx - ic, cy - jc) / 100)
            if score > best_score:
                best_score, best_idx = score, i
        cx, cy, r = int(circles[best_idx][0]), int(circles[best_idx][1]), int(circles[best_idx][2])
        return cx, cy, r, CONTAINER_MM / (2 * r)
    except Exception:
        return None


# ── crack detection ────────────────────────────────────────────────────────

def detect_cracks(gray: np.ndarray, sample_mask: np.ndarray) -> np.ndarray:
    """Return boolean crack mask inside the colony."""
    if sample_mask.sum() < 100:
        return np.zeros_like(sample_mask, dtype=bool)
    interior = gray.copy()
    interior[~sample_mask] = 0
    er = morphology.erosion(sample_mask, morphology.disk(5))
    iu = (interior * 255 if interior.max() <= 1 else interior).astype(np.uint8)
    lt = filters.threshold_local(iu, block_size=51, method="gaussian")
    dk = (iu < (lt - 15)) & er
    dk = morphology.opening(dk, morphology.disk(1))
    lb = measure.label(dk)
    cm = np.zeros_like(dk, dtype=bool)
    for rp in measure.regionprops(lb):
        if rp.area < 10:
            continue
        if rp.major_axis_length > 0 and rp.minor_axis_length > 0:
            if (
                rp.major_axis_length / rp.minor_axis_length > 2.5
                or rp.eccentricity > 0.85
            ):
                cm[lb == rp.label] = True
    return cm


# ── hyphae detection ───────────────────────────────────────────────────────

def detect_hyphae(
    gray: np.ndarray, sample_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (frangi_skeleton, meijering_skeleton, hybrid_skeleton)."""
    if sample_mask.sum() < 100:
        z = np.zeros_like(sample_mask, dtype=bool)
        return z, z.copy(), z.copy()
    ex = morphology.dilation(sample_mask, morphology.disk(20))
    fr = frangi(gray, sigmas=range(1, 5), black_ridges=False)
    fr[~ex] = 0
    th_f = fr[ex].mean() + 2 * fr[ex].std() if ex.sum() > 0 else 0.01
    fs = morphology.skeletonize(fr > th_f)
    mr = meijering(gray, sigmas=range(1, 5), black_ridges=False)
    mr[~ex] = 0
    th_m = mr[ex].mean() + 2 * mr[ex].std() if ex.sum() > 0 else 0.01
    ms = morphology.skeletonize(mr > th_m)
    return fs, ms, fs | ms


# ── morphometrics ──────────────────────────────────────────────────────────

def compute_metrics(
    mask_bool: np.ndarray,
    gray: np.ndarray,
    px2mm: float,
    dcx: float,
    dcy: float,
    crack_mask: np.ndarray,
    feat_f: np.ndarray,
    feat_m: np.ndarray,
    feat_h: np.ndarray,
) -> dict:
    mm2 = px2mm**2
    if mask_bool.sum() < 50:
        return {
            k: 0
            for k in (
                "area_mm2", "diameter_mm", "perimeter_mm", "eccentricity",
                "edge_roughness", "centre_delta_mm", "entropy", "texture_std",
                "crack_px", "crack_area_mm2", "crack_coverage_pct", "crack_count",
                "hyph_frangi_mm", "hyph_meijering_mm", "hyph_hybrid_mm",
            )
        }

    pr = measure.regionprops(mask_bool.astype(np.uint8))[0]
    R: dict = {}
    R["area_mm2"] = round(pr.area * mm2, 4)
    pm = measure.perimeter(mask_bool)
    R["perimeter_mm"] = round(pm * px2mm, 4)
    R["diameter_mm"] = round(pr.equivalent_diameter_area * px2mm, 4)
    R["eccentricity"] = round(pr.eccentricity, 6)
    eq = math.pi * pr.equivalent_diameter_area
    R["edge_roughness"] = round(pm / eq, 6) if eq > 0 else 0
    cy_c, cx_c = pr.centroid
    R["centre_delta_mm"] = round(math.hypot(cx_c - dcx, cy_c - dcy) * px2mm, 4)

    gu8 = (gray * 255).astype(np.uint8) if gray.max() <= 1 else gray.astype(np.uint8)
    R["entropy"] = (
        round(
            float(
                filters.rank.entropy(gu8, morphology.disk(5), mask=mask_bool)[
                    mask_bool
                ].mean()
            ),
            6,
        )
        if pr.area > 100
        else 0
    )
    R["texture_std"] = round(float(gray[mask_bool].std()), 6)

    R["crack_px"] = int(crack_mask.sum())
    R["crack_area_mm2"] = round(crack_mask.sum() * mm2, 6)
    R["crack_coverage_pct"] = round(100 * crack_mask.sum() / pr.area, 4) if pr.area > 0 else 0
    R["crack_count"] = int(measure.label(crack_mask).max())

    R["hyph_frangi_mm"] = round(int(feat_f.sum()) * px2mm, 4)
    R["hyph_meijering_mm"] = round(int(feat_m.sum()) * px2mm, 4)
    R["hyph_hybrid_mm"] = round(int(feat_h.sum()) * px2mm, 4)
    return R


# ── overlay panels ─────────────────────────────────────────────────────────

def create_full_overlays(
    img_bgr: np.ndarray,
    sample_mask: np.ndarray,
    crack_mask: np.ndarray,
    feat_hybrid: np.ndarray,
    container_info: tuple | None,
    fname: str,
) -> list[tuple[Image.Image, str]]:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    if container_info:
        dcx, dcy, dr = int(container_info[0]), int(container_info[1]), int(container_info[2])
    else:
        dcx, dcy, dr = w // 2, h // 2, min(h, w) // 2

    # Panel 1: raw + dish circle + colony contour
    p1 = img_rgb.copy()
    if container_info:
        cv2.circle(p1, (dcx, dcy), dr, (0, 255, 0), 3)
    cts, _ = cv2.findContours(sample_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p1, cts, -1, (255, 0, 0), 2)

    # Panel 2: binary mask
    p2 = np.zeros_like(img_rgb)
    p2[sample_mask] = [255, 255, 255]

    # Panel 3: colony overlay (red)
    p3 = img_rgb.copy()
    if sample_mask.sum() > 0:
        p3[sample_mask] = (
            p3[sample_mask].astype(np.float32) * 0.5
            + np.array([255, 0, 0], dtype=np.float32) * 0.5
        ).astype(np.uint8)
    if container_info:
        cv2.circle(p3, (dcx, dcy), dr, (0, 255, 0), 2)

    # Panel 4: crack overlay (yellow)
    p4 = img_rgb.copy()
    if crack_mask.sum() > 0:
        ck = (
            cv2.dilate(
                crack_mask.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            > 0
        )
        p4[ck] = (
            p4[ck].astype(np.float32) * 0.3
            + np.array([255, 255, 0], dtype=np.float32) * 0.7
        ).astype(np.uint8)
    if container_info:
        cv2.circle(p4, (dcx, dcy), dr, (0, 255, 0), 2)
        cv2.drawContours(p4, cts, -1, (255, 0, 0), 1)

    # Panel 5: hyphae overlay (cyan)
    p5 = img_rgb.copy()
    if feat_hybrid.sum() > 0:
        hy = (
            cv2.dilate(
                feat_hybrid.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            > 0
        )
        p5[hy] = (
            p5[hy].astype(np.float32) * 0.3
            + np.array([0, 255, 255], dtype=np.float32) * 0.7
        ).astype(np.uint8)
    if container_info:
        cv2.circle(p5, (dcx, dcy), dr, (0, 255, 0), 2)
        cv2.drawContours(p5, cts, -1, (255, 0, 0), 1)

    # Panel 6: combined (colony + cracks + hyphae)
    p6 = img_rgb.copy()
    if sample_mask.sum() > 0:
        p6[sample_mask] = (
            p6[sample_mask].astype(np.float32) * 0.6
            + np.array([255, 0, 0], dtype=np.float32) * 0.4
        ).astype(np.uint8)
    if crack_mask.sum() > 0:
        ck2 = (
            cv2.dilate(
                crack_mask.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            > 0
        )
        p6[ck2] = [255, 255, 0]
    if feat_hybrid.sum() > 0:
        hy2 = (
            cv2.dilate(
                feat_hybrid.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            )
            > 0
        )
        p6[hy2] = [0, 255, 255]
    if container_info:
        cv2.circle(p6, (dcx, dcy), dr, (0, 255, 0), 2)

    return [
        (Image.fromarray(p1), f"{fname} — Raw+Dish"),
        (Image.fromarray(p2), f"{fname} — Mask"),
        (Image.fromarray(p3), f"{fname} — Colony"),
        (Image.fromarray(p4), f"{fname} — Cracks"),
        (Image.fromarray(p5), f"{fname} — Hyphae"),
        (Image.fromarray(p6), f"{fname} — Combined"),
    ]


# ── growth charts ──────────────────────────────────────────────────────────

def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img


def _dc_to_num(day_code_str: object, fallback: int = 0) -> int:
    """Convert 'd07' → 7.  Returns fallback for anything unparseable."""
    if isinstance(day_code_str, str) and day_code_str.startswith("d") and day_code_str[1:].isdigit():
        return int(day_code_str[1:])
    return fallback


def make_growth_charts(results: list[dict]) -> list[tuple[Image.Image, str]]:
    if len(results) < 2:
        return []

    df = pd.DataFrame(results)
    if "error" in df.columns:
        df = df[df["error"].fillna("").astype(str).str.strip() == ""].copy()
    if len(df) < 2:
        return []

    numeric_cols = [
        "days_since_start", "area_mm2", "diameter_mm", "perimeter_mm",
        "eccentricity", "edge_roughness", "centre_delta_mm",
        "entropy", "texture_std",
        "crack_area_mm2", "crack_coverage_pct", "crack_count",
        "hyph_frangi_mm", "hyph_meijering_mm", "hyph_hybrid_mm",
        "rgr_per_day", "relative_growth_per_day",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build a numeric sort key from day_code ("d07" → 7); fall back to days_since_start
    if "day_code" in df.columns:
        df["_day_num"] = df["day_code"].apply(_dc_to_num)
        if "days_since_start" in df.columns:
            mask = df["_day_num"] == 0
            df.loc[mask, "_day_num"] = df.loc[mask, "days_since_start"].fillna(0).astype(int)
    elif "days_since_start" in df.columns:
        df["_day_num"] = df["days_since_start"].fillna(0).astype(int)
    else:
        df["_day_num"] = range(len(df))

    df = df.sort_values("_day_num").reset_index(drop=True)

    # Build an x-axis label for every row: use day_code when valid, else construct dXX from number
    def _make_x_label(dc_val, day_num: int) -> str:
        dc = str(dc_val) if dc_val else ""
        if dc.startswith("d") and dc[1:].isdigit():
            return dc
        return f"d{day_num:02d}" if day_num > 0 else "d?"

    df["_x_label"] = [
        _make_x_label(dc, int(n))
        for dc, n in zip(
            df.get("day_code", pd.Series([""] * len(df))),
            df["_day_num"],
        )
    ]

    charts: list[tuple[Image.Image, str]] = []

    chart_defs = [
        ("area_mm2",                "Area (mm²)",            "Colony Area",              "#e74c3c", "o", True),
        ("diameter_mm",             "Diameter (mm)",         "Colony Diameter",           "#2980b9", "s", False),
        ("perimeter_mm",            "Perimeter (mm)",        "Colony Perimeter",          "#8e44ad", "^", False),
        ("eccentricity",            "Eccentricity",          "Colony Eccentricity",       "#e67e22", "D", False),
        ("edge_roughness",          "Edge Roughness",        "Edge Roughness (P/πd)",     "#16a085", "v", False),
        ("centre_delta_mm",         "Centre Offset (mm)",    "Colony Centre Offset",      "#d35400", "p", False),
        ("entropy",                 "Entropy",               "Colony Texture Entropy",    "#7f8c8d", "h", False),
        ("texture_std",             "Texture Std Dev",       "Colony Texture Std Dev",    "#2c3e50", "*", False),
        ("crack_area_mm2",          "Crack Area (mm²)",      "Crack Area",                "#f1c40f", "o", True),
        ("crack_coverage_pct",      "Crack Coverage (%)",    "Crack Coverage",            "#d4ac0d", "s", False),
        ("crack_count",             "Crack Count",           "Number of Cracks",          "#b7950b", "^", False),
        ("hyph_frangi_mm",          "Length (mm)",           "Hyphae Length — Frangi",    "#1abc9c", "o", False),
        ("hyph_meijering_mm",       "Length (mm)",           "Hyphae Length — Meijering", "#3498db", "s", False),
        ("hyph_hybrid_mm",          "Length (mm)",           "Hyphae Length — Hybrid",    "#2ecc71", "D", False),
        ("rgr_per_day",             "RGR (ln mm²/day)",      "Relative Growth Rate",      "#c0392b", "o", False),
        ("relative_growth_per_day", "Growth (mm²/day)",      "Absolute Growth Rate",      "#e74c3c", "s", False),
    ]

    for col, ylabel, title, color, marker, fill in chart_defs:
        if col not in df.columns:
            continue
        valid = df[col].notna() & (df[col].astype(str).str.strip() != "")
        if valid.sum() < 2:
            continue
        sub = df.loc[valid].sort_values("_day_num").reset_index(drop=True).copy()
        x = sub["_day_num"].tolist()
        x_labels = sub["_x_label"].tolist()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, sub[col].tolist(), f"{marker}-", color=color, lw=2, ms=8)
        if fill:
            ax.fill_between(x, 0, sub[col].tolist(), alpha=0.15, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30 if len(x_labels) > 6 else 0, fontsize=8)
        ax.set(xlabel="Day", ylabel=ylabel, title=title)
        ax.grid(True, alpha=0.3)
        charts.append((_fig_to_pil(fig), title))

    return charts
