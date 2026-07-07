#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Petri Dish Cropper
==================
Automatically detects and crops individual petri dishes from batch images.
Supports 2-8+ dishes per image. Only extracts fully visible dishes.

Usage:
    metrics-petri-crop -i /path/to/images
    metrics-petri-crop -i photo.jpg --date 06/Feb --debug
"""

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Core Detection Engine
# ---------------------------------------------------------------------------

def detect_petri_dishes(image_path: str | Path, debug: bool = False):
    """
    Detect petri dishes in an image using Otsu thresholding + contour analysis.
    Falls back to Hough Circle Transform if Otsu finds too few candidates.

    Returns:
        List of (cx, cy, radius) tuples for each detected full dish.
        Sorted top-to-bottom, left-to-right (reading order).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    min_dim = min(h, w)

    # ---- Method 1: Otsu thresholding ----
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((7, 7), np.uint8)
    otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel)
    otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours_otsu, _ = cv2.findContours(otsu_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    min_area = (min_dim * 0.05) ** 2
    max_area = (min_dim * 0.55) ** 2

    for cnt in contours_otsu:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > 1.7:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)

        (ex, ey), er = cv2.minEnclosingCircle(cnt)
        cx, cy = int(ex), int(ey)
        r = int(er)

        # Brightness contrast validation
        inner = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(inner, (cx, cy), int(r * 0.7), 255, -1)
        inner_mean = cv2.mean(gray, mask=inner)[0]

        ring = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(ring, (cx, cy), int(r * 1.15), 255, -1)
        ring_inner = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(ring_inner, (cx, cy), int(r * 0.85), 255, -1)
        ring_mask = cv2.subtract(ring, ring_inner)
        ring_mean = cv2.mean(gray, mask=ring_mask)[0]
        contrast = inner_mean - ring_mean

        candidates.append({
            'center': (cx, cy), 'radius': r,
            'circularity': circularity, 'area': area,
            'contrast': contrast, 'method': 'otsu'
        })

    # ---- Method 2: Hough fallback ----
    if len(candidates) < 2:
        min_r = int(min_dim * 0.08)
        max_r = int(min_dim * 0.45)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=min_r * 1.5, param1=100, param2=25,
            minRadius=min_r, maxRadius=max_r)

        if circles is not None:
            for c in np.uint16(np.around(circles[0])):
                cx, cy, r = int(c[0]), int(c[1]), int(c[2])

                covered = False
                for cand in candidates:
                    dist = np.sqrt((cx - cand['center'][0])**2 + (cy - cand['center'][1])**2)
                    if dist < (r + cand['radius']) * 0.5:
                        covered = True
                        break
                if covered:
                    continue

                margin = int(r * 0.15)
                if not (cx - r + margin > 0 and cx + r - margin < w and
                        cy - r + margin > 0 and cy + r - margin < h):
                    continue

                inner = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(inner, (cx, cy), int(r * 0.7), 255, -1)
                inner_mean = cv2.mean(gray, mask=inner)[0]
                ring = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(ring, (cx, cy), int(r * 1.15), 255, -1)
                ring_inner = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(ring_inner, (cx, cy), int(r * 0.85), 255, -1)
                ring_mask = cv2.subtract(ring, ring_inner)
                ring_mean = cv2.mean(gray, mask=ring_mask)[0]
                contrast = inner_mean - ring_mean
                if contrast < 5:
                    continue

                candidates.append({
                    'center': (cx, cy), 'radius': r,
                    'circularity': 0.7, 'area': np.pi * r * r,
                    'contrast': contrast, 'method': 'hough'
                })

    # ---- Filter & deduplicate ----
    if candidates:
        max_contrast = max(c['contrast'] for c in candidates)
        for c in candidates:
            c['score'] = c['circularity'] * max(0, c['contrast']) / max(max_contrast, 1)
        candidates.sort(key=lambda x: x['score'], reverse=True)

    final = []
    for cand in candidates:
        cx, cy = cand['center']
        r = cand['radius']

        too_close = False
        for fx, fy, fr in final:
            dist = np.sqrt((cx - fx)**2 + (cy - fy)**2)
            if dist < (r + fr) * 0.55:
                too_close = True
                break
        if too_close:
            continue

        if cand['circularity'] < 0.4:
            continue
        if cand['contrast'] < 3:
            continue

        margin = int(r * 0.12)
        if not (cx - r + margin > 0 and cx + r - margin < w and
                cy - r + margin > 0 and cy + r - margin < h):
            continue

        final.append((cx, cy, r))

    final.sort(key=lambda c: (c[1], c[0]))

    if debug:
        debug_img = img.copy()
        for (x, y, r) in final:
            cv2.circle(debug_img, (x, y), r, (0, 255, 0), 3)
            cv2.circle(debug_img, (x, y), 4, (0, 0, 255), -1)
        return final, debug_img

    return final


def crop_dish(img: np.ndarray, cx: int, cy: int, r: int, padding: float = 0.05) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Extract a square crop around a detected dish.

    Args:
        img: Source BGR image
        cx, cy: Dish center
        r: Dish radius
        padding: Extra space around dish as fraction of radius (default 5%)

    Returns:
        (cropped_image, (x1, y1, x2, y2))
    """
    h, w = img.shape[:2]
    size = int(r * 2 * (1 + padding))

    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)

    if x2 - x1 < size:
        x1 = max(0, x2 - size)
    if y2 - y1 < size:
        y1 = max(0, y2 - size)

    crop = img[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def process_image(image_path: str | Path, output_dir: str | Path,
                  padding: float = 0.05, save_debug: bool = False,
                  date_prefix: str = "") -> dict:
    """
    Process a single image: detect dishes and save cropped outputs.

    Returns:
        dict with 'input', 'count', 'output_paths', 'debug_path' (optional)
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        return {'input': str(image_path), 'count': 0, 'output_paths': [], 'error': 'Could not load image'}

    detected, debug_img = detect_petri_dishes(image_path, debug=True)

    stem = image_path.stem
    output_paths = []

    for i, (cx, cy, r) in enumerate(detected):
        crop, _ = crop_dish(img, cx, cy, r, padding=padding)
        out_path = output_dir / f"{date_prefix}{stem}_dish_{i+1:02d}.png"
        cv2.imwrite(str(out_path), crop)
        output_paths.append(str(out_path))

    result = {
        'input': str(image_path),
        'count': len(detected),
        'output_paths': output_paths
    }

    if save_debug:
        debug_path = output_dir / f"{date_prefix}{stem}_debug.jpg"
        cv2.imwrite(str(debug_path), debug_img)
        result['debug_path'] = str(debug_path)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MONTH_NUMBERS = {
    name: number
    for number, names in enumerate((
        ("jan", "january"), ("feb", "february"), ("mar", "march"),
        ("apr", "april"), ("may",), ("jun", "june"),
        ("jul", "july"), ("aug", "august"),
        ("sep", "sept", "september"), ("oct", "october"),
        ("nov", "november"), ("dec", "december"),
    ), start=1)
    for name in names
}


def parse_date_prefix(value: str | None, today: date | None = None) -> str:
    """Convert a supported user date to a YYYYMMDD_ filename prefix."""
    if not value or not value.strip():
        return ""

    raw = " ".join(value.strip().split())
    try:
        if raw.count("/") == 2:
            parsed = datetime.strptime(raw, "%d/%m/%Y").date()
        else:
            parts = raw.replace("/", " ").split()
            if len(parts) != 2:
                raise ValueError
            day_text, month_text = parts
            month = MONTH_NUMBERS.get(month_text.lower())
            if month is None:
                raise ValueError
            parsed = date((today or date.today()).year, month, int(day_text))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Use DD/MM/YYYY, DD/Mon, or DD Mon (for example "
            "06/02/2026, 06/Feb, or 06 Feb)."
        ) from exc

    return parsed.strftime("%Y%m%d_")


def resolve_output_dir(input_path: Path, output: str | None) -> Path:
    """Resolve the CLI output directory, defaulting beside the input."""
    if output:
        return Path(output)
    input_dir = input_path if input_path.is_dir() else input_path.parent
    return input_dir / "cropped"


def main():
    parser = argparse.ArgumentParser(
        description="Auto-detect and crop petri dishes from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i ./photos
  %(prog)s -i photo.jpg --date 06/Feb --padding 0.1 --debug
  %(prog)s -i photo.jpg -o ./custom-output
        """
    )
    parser.add_argument('-i', '--input', required=True, help='Input image file or directory')
    parser.add_argument('-o', '--output', help='Output directory (default: cropped/ beside the input)')
    parser.add_argument('-p', '--padding', type=float, default=0.05, help='Padding around dish as fraction of radius (default: 0.05)')
    parser.add_argument('-d', '--debug', action='store_true', help='Save debug overlay images')
    parser.add_argument('-D', '--date', help='Optional crop filename date: DD/MM/YYYY, DD/Mon, or DD Mon')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = resolve_output_dir(input_path, args.output)
    try:
        date_prefix = parse_date_prefix(args.date)
    except ValueError as exc:
        parser.error(str(exc))

    if input_path.is_dir():
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp'):
            image_files.extend(input_path.glob(ext))
            image_files.extend(input_path.glob(ext.upper()))
        image_files = sorted(set(image_files))
        if not image_files:
            print(f"No images found in {input_path}")
            sys.exit(1)
    else:
        image_files = [input_path]

    print(f"Processing {len(image_files)} image(s) -> {output_dir}")
    print("-" * 50)

    total_dishes = 0
    for img_path in image_files:
        result = process_image(
            img_path, output_dir, padding=args.padding,
            save_debug=args.debug, date_prefix=date_prefix
        )
        if 'error' in result:
            print(f"  [SKIP] {img_path.name}: {result['error']}")
        else:
            print(f"  [OK]   {img_path.name}: {result['count']} dish(es) cropped")
            total_dishes += result['count']

    print("-" * 50)
    print(f"Done. Total dishes extracted: {total_dishes}")
    print(f"Output saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
