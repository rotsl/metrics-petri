from __future__ import annotations

import base64
import csv
import json
import math
import os
import platform
import re
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw
from scipy import ndimage, signal
from skimage import filters, measure, morphology
from skimage.feature import canny
from skimage.filters import gaussian, sobel, threshold_local
from skimage.morphology import convex_hull_image, dilation, remove_small_objects, skeletonize

warnings.filterwarnings(
    "ignore",
    message=r"At least one mel filter has all zero values\..*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Parameter `min_size` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Parameter `area_threshold` is deprecated.*",
    category=FutureWarning,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PETRI_DISH_DIAMETER_MM = 90.0
LOCAL_MODEL_DIR = Path(os.getenv("LOCAL_MODEL_DIR", "models/gemma-4-e2b-it-MLX-4bit"))
LOCAL_MODEL_ID = os.getenv("LOCAL_MODEL_ID", "models/gemma-4-e2b-it-MLX-4bit")
LOCAL_UNET_PATH = Path(os.getenv("LOCAL_UNET_PATH", "models/best_unet.pt"))
LOCAL_SAM_CHECKPOINT = Path(os.getenv("LOCAL_SAM_CHECKPOINT", "models/sam_vit_b_01ec64.pth"))
LOCAL_SAM_MODEL_TYPE = os.getenv("LOCAL_SAM_MODEL_TYPE", "vit_b")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemma-3-27b-it")
GENERATION_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

_LOCAL_BUNDLE: tuple[Any, Any] | None = None
_MLX_BUNDLE: tuple[Any, Any, Any] | None = None
_UNET_BUNDLE: Any | None = None
_SAM_PREDICTOR: Any | None = None
_MLX_RUNTIME_OK: bool | None = None
_MLX_RUNTIME_REASON: str | None = None

SEGMENTATION_PROMPT_TEMPLATE = """
Analyze this petri dish image of Magnaporthe growth.

The petri dish has already been calibrated by classical image processing.
Use this dish geometry as fixed context instead of trying to infer the dish yourself:
- dish_center: {dish_center}
- dish_radius: {dish_radius}

Tasks:
1. Segment the fungal colony as a polygon inside the calibrated dish.
2. Identify visible cracks or internal bands within the colony.
3. Describe visible internal-band morphology conservatively.

Return JSON only, with this shape:
{{
  "colony_polygon": [{{"x": 0-1000, "y": 0-1000}}],
  "cracks": [[{{"x": 0-1000, "y": 0-1000}}]],
  "internal_band_description": string
}}

Notes:
- Coordinates are normalized from 0 to 1000 relative to the image.
- The dish boundary is already known; do not estimate or return dish geometry.
- Prefer one clean polygon covering the main colony within the dish.
- If cracks are not visible, return an empty array.
- Be conservative and avoid hallucinating structures.
""".strip()


@dataclass
class AnalysisRun:
    engine: str
    engine_model: str
    created_at: str
    output_dir: str
    analysis_json: str
    analysis_csv: str


def list_input_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(
        [path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    )


def extract_day(filename: str) -> int | None:
    match = re.search(r"day(\d+)", filename, re.IGNORECASE)
    if not match:
        match = re.search(r"d(\d+)", filename, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def parse_day(filename: str) -> int:
    explicit_day = extract_day(filename)
    if explicit_day is not None:
        return explicit_day
    generic_number = re.search(r"(\d+)", filename)
    return int(generic_number.group(1)) if generic_number else 0


def infer_days_for_images(images: list[Path]) -> dict[Path, int]:
    explicit: list[tuple[int, Path]] = []
    inferred: list[Path] = []
    for image in images:
        day = extract_day(image.name)
        if day is None:
            inferred.append(image)
        else:
            explicit.append((day, image))

    taken_days = {day for day, _ in explicit}
    next_day = 1
    day_map = {image: day for day, image in explicit}
    for image in inferred:
        while next_day in taken_days:
            next_day += 1
        day_map[image] = next_day
        taken_days.add(next_day)
        next_day += 1
    return day_map


def _coerce_model_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    text = getattr(payload, "text", None)
    if isinstance(text, str):
        return text
    raise TypeError(f"Unsupported model response type: {type(payload).__name__}")


def _extract_json(text: Any) -> dict[str, Any]:
    text = _coerce_model_text(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not contain JSON")
    return json.loads(text[start : end + 1])


def _build_segmentation_prompt(dish_center: dict[str, float], dish_radius: float) -> str:
    return SEGMENTATION_PROMPT_TEMPLATE.format(
        dish_center=json.dumps(
            {
                "x": round(float(dish_center["x"]), 2),
                "y": round(float(dish_center["y"]), 2),
            }
        ),
        dish_radius=round(float(dish_radius), 2),
    )


def _to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.dtype in (np.float32, np.float64):
        img = img - np.min(img)
        maxv = float(np.max(img))
        if maxv > 0:
            img = img / maxv
        img = (img * 255).clip(0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _load_local_bundle(model_dir: Path) -> tuple[Any, Any]:
    global _LOCAL_BUNDLE
    if _LOCAL_BUNDLE is None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForMultimodalLM.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype="auto",
            device_map="auto",
        )
        model.eval()
        _LOCAL_BUNDLE = (processor, model)
    return _LOCAL_BUNDLE


def _load_unet_segmenter(model_path: Path = LOCAL_UNET_PATH) -> Any:
    global _UNET_BUNDLE
    if _UNET_BUNDLE is None:
        try:
            import albumentations as A
            import segmentation_models_pytorch as smp
            import torch
            from albumentations.pytorch import ToTensorV2
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Local U-Net dependencies are missing. Install albumentations and segmentation-models-pytorch in the project venv."
            ) from exc

        if not model_path.exists():
            raise FileNotFoundError(f"U-Net checkpoint not found: {model_path}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        transform = A.Compose(
            [
                A.Resize(512, 512),
                A.Normalize(),
                ToTensorV2(),
            ]
        )
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        _UNET_BUNDLE = (transform, model, device)
    return _UNET_BUNDLE


def _predict_unet_mask(image: np.ndarray, dish_mask: np.ndarray, model_path: Path = LOCAL_UNET_PATH) -> np.ndarray:
    import torch

    transform, model, device = _load_unet_segmenter(model_path)
    rgb = _to_rgb_uint8(image)
    height, width = rgb.shape[:2]
    aug = transform(image=rgb)
    tensor = aug["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        scores = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
    mask = (scores > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0
    mask = np.logical_and(mask, dish_mask)
    mask = morphology.closing(mask, morphology.disk(5))
    mask = morphology.remove_small_holes(mask, 256)
    return mask


def _load_mlx_bundle(model_ref: str) -> tuple[Any, Any, Any]:
    global _MLX_BUNDLE
    if _MLX_BUNDLE is None:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        model, processor = load(model_ref)
        config = load_config(model_ref)
        _MLX_BUNDLE = (model, processor, config)
    return _MLX_BUNDLE


def _mlx_runtime_available() -> bool:
    global _MLX_RUNTIME_OK, _MLX_RUNTIME_REASON
    if _MLX_RUNTIME_OK is not None:
        return _MLX_RUNTIME_OK
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        _MLX_RUNTIME_OK = False
        _MLX_RUNTIME_REASON = "MLX is only supported for local Gemma prior on Apple Silicon Macs."
        return _MLX_RUNTIME_OK
    if os.getenv("LOCAL_ENABLE_MLX", "1") != "1":
        _MLX_RUNTIME_OK = False
        _MLX_RUNTIME_REASON = "LOCAL_ENABLE_MLX is disabled."
        return _MLX_RUNTIME_OK
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mlx, mlx_vlm; print('ok')",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    _MLX_RUNTIME_OK = probe.returncode == 0
    if not _MLX_RUNTIME_OK:
        stderr = (probe.stderr or "").strip()
        stdout = (probe.stdout or "").strip()
        detail = stderr or stdout or "Unknown MLX initialization error."
        _MLX_RUNTIME_REASON = detail.splitlines()[-1][:300]
    else:
        _MLX_RUNTIME_REASON = None
    return _MLX_RUNTIME_OK


def _mlx_runtime_reason() -> str:
    if _MLX_RUNTIME_OK is None:
        _mlx_runtime_available()
    return _MLX_RUNTIME_REASON or "Unknown MLX initialization error."


def _analyze_with_mlx_model(image_path: Path, model_ref: str, prompt: str) -> dict[str, Any]:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor, config = _load_mlx_bundle(model_ref)
    formatted_prompt = apply_chat_template(processor, config, prompt, num_images=1)
    output = generate(
        model,
        processor,
        formatted_prompt,
        [str(image_path)],
        verbose=False,
        max_tokens=1024,
    )
    return _extract_json(output)


def _empty_gemma_prior() -> dict[str, Any]:
    return {
        "colony_polygon": [],
        "cracks": [],
        "internal_band_description": "",
        "engine_warning": "Gemma prior unavailable for this run.",
    }


def _analyze_with_local_model(image_path: Path, model_dir: Path, prompt: str) -> dict[str, Any]:
    import torch

    processor, model = _load_local_bundle(model_dir)
    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    input_len = int(inputs["input_ids"].shape[-1])

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=1024)

    decoded = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    return _extract_json(decoded)


def _analyze_with_gemini(image_path: Path, model_name: str, prompt: str) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for Gemini analysis")

    mime_type = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    image_bytes = image_path.read_bytes()
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64.b64encode(image_bytes).decode("utf-8"),
                        }
                    },
                ]
            }
        ]
    }

    response = requests.post(
        GENERATION_ENDPOINT.format(model=model_name, api_key=api_key),
        json=payload,
        timeout=300,
    )
    if not response.ok:
        detail = response.text
        try:
            detail = json.dumps(response.json(), indent=2)
        except ValueError:
            detail = response.text
        raise RuntimeError(
            f"Gemini API request failed with status {response.status_code} for model {model_name}: {detail}"
        )
    body = response.json()
    text = body["candidates"][0]["content"]["parts"][0]["text"]
    return _extract_json(text)


def _mask_bbox(mask: np.ndarray, pad: int = 8) -> np.ndarray | None:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if ys.size == 0 or xs.size == 0:
        return None
    y0 = max(int(ys.min()) - pad, 0)
    y1 = int(ys.max()) + pad + 1
    x0 = max(int(xs.min()) - pad, 0)
    x1 = int(xs.max()) + pad + 1
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _sample_points(mask: np.ndarray, max_points: int, label: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int32)
    if ys.size <= max_points:
        idx = np.arange(ys.size)
    else:
        idx = np.linspace(0, ys.size - 1, max_points, dtype=int)
    points = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    labels = np.full((len(idx),), label, dtype=np.int32)
    return points, labels


def _build_sam_prompts(
    base_mask: np.ndarray,
    classical_mask: np.ndarray,
    dish_mask: np.ndarray,
    max_positive: int = 12,
    max_negative: int = 12,
) -> dict[str, np.ndarray | None]:
    base = np.asarray(base_mask, dtype=bool)
    classical = np.asarray(classical_mask, dtype=bool)
    dish = np.asarray(dish_mask, dtype=bool)

    positive_seed = np.logical_or(base, classical) & dish
    strong_positive = np.logical_and(base, classical) & dish
    if strong_positive.sum() > 0:
        positive_seed = strong_positive

    negative_seed = dish & ~np.logical_or(base, classical)
    pos_pts, pos_labels = _sample_points(positive_seed, max_positive, 1)
    neg_pts, neg_labels = _sample_points(negative_seed, max_negative, 0)

    if pos_pts.size == 0:
        box = _mask_bbox(np.logical_or(base, classical) & dish) or _mask_bbox(dish)
        if box is None:
            return {"point_coords": np.empty((0, 2), dtype=np.float32), "point_labels": np.empty((0,), dtype=np.int32), "box": None}
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        pos_pts = np.array([[cx, cy]], dtype=np.float32)
        pos_labels = np.array([1], dtype=np.int32)

    point_coords = np.concatenate([pos_pts, neg_pts], axis=0)
    point_labels = np.concatenate([pos_labels, neg_labels], axis=0)
    box = _mask_bbox(np.logical_or(base, classical) & dish)
    if box is None:
        box = _mask_bbox(dish)
    return {
        "point_coords": point_coords,
        "point_labels": point_labels,
        "box": box,
    }


def _load_sam_predictor(
    checkpoint_path: Path = LOCAL_SAM_CHECKPOINT,
    model_type: str = LOCAL_SAM_MODEL_TYPE,
) -> Any:
    global _SAM_PREDICTOR
    if _SAM_PREDICTOR is None:
        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry
        except ModuleNotFoundError as exc:
            raise RuntimeError("segment-anything is required for SAM refinement in local mode.") from exc

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        model = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        model.to(device=device)
        _SAM_PREDICTOR = SamPredictor(model)
    return _SAM_PREDICTOR


def _normalized_to_pixels(points: list[dict[str, float]], width: int, height: int) -> np.ndarray:
    if not points:
        return np.empty((0, 2), dtype=float)
    coords = np.array([[float(point["x"]) * width / 1000.0, float(point["y"]) * height / 1000.0] for point in points])
    return coords


def _mask_from_polygon(width: int, height: int, polygon_xy: np.ndarray) -> np.ndarray:
    mask_image = Image.new("L", (width, height), 0)
    if len(polygon_xy) >= 3:
        draw = ImageDraw.Draw(mask_image)
        draw.polygon([tuple(point) for point in polygon_xy.tolist()], fill=1, outline=1)
    return np.array(mask_image, dtype=bool)


def _largest_component(mask: np.ndarray, center_x: float, center_y: float, reference_mask: np.ndarray | None = None) -> np.ndarray:
    labeled = measure.label(mask.astype(np.uint8))
    regions = measure.regionprops(labeled)
    if not regions:
        return np.zeros_like(mask, dtype=bool)

    best_score = -float("inf")
    best_mask = np.zeros_like(mask, dtype=bool)
    for region in regions:
        region_mask = labeled == region.label
        centroid_y, centroid_x = region.centroid
        distance_penalty = math.hypot(centroid_x - center_x, centroid_y - center_y)
        overlap_bonus = 0.0
        if reference_mask is not None and np.any(reference_mask):
            overlap_bonus = float(np.count_nonzero(region_mask & reference_mask)) * 2.0
        score = float(region.area) + overlap_bonus - distance_penalty * 25.0
        if score > best_score:
            best_score = score
            best_mask = region_mask
    return best_mask


def _circle_mask(width: int, height: int, center_x: float, center_y: float, radius_px: float) -> np.ndarray:
    yy, xx = np.indices((height, width))
    return (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius_px**2


def _score_circle_candidate(gray: np.ndarray, center_x: float, center_y: float, radius_px: float) -> float:
    angles = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
    inner_radius = max(radius_px - 6.0, radius_px * 0.985)
    outer_radius = min(radius_px + 6.0, radius_px * 1.015)

    inside_x = np.clip(np.round(center_x + inner_radius * np.cos(angles)).astype(int), 0, gray.shape[1] - 1)
    inside_y = np.clip(np.round(center_y + inner_radius * np.sin(angles)).astype(int), 0, gray.shape[0] - 1)
    outside_x = np.clip(np.round(center_x + outer_radius * np.cos(angles)).astype(int), 0, gray.shape[1] - 1)
    outside_y = np.clip(np.round(center_y + outer_radius * np.sin(angles)).astype(int), 0, gray.shape[0] - 1)

    ring_contrast = np.abs(
        gray[outside_y, outside_x].astype(np.float32) - gray[inside_y, inside_x].astype(np.float32)
    ).mean()
    centeredness = 1.0 - (
        math.hypot(center_x - (gray.shape[1] / 2.0), center_y - (gray.shape[0] / 2.0))
        / max(min(gray.shape) / 2.0, 1.0)
    )
    return float(ring_contrast + max(centeredness, 0.0) * 8.0)


def _detect_dish_geometry(gray: np.ndarray) -> tuple[float, float, float]:
    height, width = gray.shape
    scale = 1.0
    working = gray
    max_dim = max(height, width)
    if max_dim > 1200:
        scale = 1200.0 / max_dim
        working = cv2.resize(gray, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    blurred = cv2.GaussianBlur(working, (9, 9), 2.0)
    min_dim = min(working.shape)
    min_radius = max(int(min_dim * 0.35), 40)
    max_radius = max(int(min_dim * 0.52), min_radius + 5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dim * 0.5,
        param1=120,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is not None and len(circles[0]) > 0:
        scored = []
        for candidate in circles[0]:
            cx, cy, radius = map(float, candidate)
            score = _score_circle_candidate(working, cx, cy, radius)
            scored.append((score, cx, cy, radius))
        _, cx, cy, radius = max(scored, key=lambda item: item[0])
        return cx / scale, cy / scale, radius / scale

    fallback_radius = min(height, width) * 0.47
    return width / 2.0, height / 2.0, fallback_radius


def _pixels_to_normalized(center_x: float, center_y: float, radius_px: float, width: int, height: int) -> tuple[dict[str, float], float]:
    return (
        {
            "x": float(np.clip((center_x / max(width, 1)) * 1000.0, 0.0, 1000.0)),
            "y": float(np.clip((center_y / max(height, 1)) * 1000.0, 0.0, 1000.0)),
        },
        float(np.clip((radius_px / max(width, 1)) * 1000.0, 0.0, 1000.0)),
    )


def _clip_points_to_dish(points_xy: np.ndarray, center_x: float, center_y: float, radius_px: float) -> np.ndarray:
    if len(points_xy) == 0:
        return points_xy
    clipped = points_xy.astype(float).copy()
    deltas = clipped - np.array([center_x, center_y], dtype=float)
    distances = np.linalg.norm(deltas, axis=1)
    outside = distances > radius_px
    if np.any(outside):
        safe_distances = np.maximum(distances[outside], 1e-6)
        scale = (radius_px - 1.0) / safe_distances
        clipped[outside] = np.column_stack(
            [
                center_x + deltas[outside, 0] * scale,
                center_y + deltas[outside, 1] * scale,
            ]
        )
    return clipped


def _pixels_to_normalized_points(points_xy: np.ndarray, width: int, height: int) -> list[dict[str, float]]:
    if len(points_xy) == 0:
        return []
    return [
        {
            "x": float(np.clip((point[0] / max(width, 1)) * 1000.0, 0.0, 1000.0)),
            "y": float(np.clip((point[1] / max(height, 1)) * 1000.0, 0.0, 1000.0)),
        }
        for point in points_xy
    ]


def _line_length(points_xy: np.ndarray) -> float:
    if len(points_xy) < 2:
        return 0.0
    segments = np.diff(points_xy, axis=0)
    return float(np.linalg.norm(segments, axis=1).sum())


def _build_mask_if_missing(gray: np.ndarray, center_x: float, center_y: float, radius_px: float) -> np.ndarray:
    dish_mask = _circle_mask(gray.shape[1], gray.shape[0], center_x, center_y, radius_px)
    threshold = np.percentile(gray[dish_mask], 45) if np.any(dish_mask) else np.percentile(gray, 45)
    return np.logical_and(gray < threshold, dish_mask)


def _correct_background(gray: np.ndarray, dish_mask: np.ndarray) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    background = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=45, sigmaY=45)
    corrected = gray_f / np.maximum(background, 1.0)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    corrected = corrected.astype(np.uint8)
    corrected[~dish_mask] = 0
    return corrected


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0.0, 0.0
    return float(xs.mean()), float(ys.mean())


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = np.count_nonzero(mask_a | mask_b)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(mask_a & mask_b) / union)


def _apply_hybrid_strategy(ml_mask: np.ndarray, classical_mask: np.ndarray, strategy: str) -> np.ndarray:
    ml = np.asarray(ml_mask, dtype=bool)
    classical = np.asarray(classical_mask, dtype=bool)
    normalized = strategy.lower()
    if normalized == "ml":
        return ml
    if normalized == "classical":
        return classical
    if normalized == "union":
        return np.logical_or(ml, classical)
    if normalized == "intersection":
        return np.logical_and(ml, classical)
    return ml


def _mask_fraction(mask: np.ndarray, reference_mask: np.ndarray) -> float:
    reference_area = max(int(np.count_nonzero(reference_mask)), 1)
    return float(np.count_nonzero(mask) / reference_area)


def _outer_ring_occupancy(mask: np.ndarray, center_x: float, center_y: float, radius_px: float) -> float:
    yy, xx = np.indices(mask.shape)
    distances = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    outer_ring = (distances >= radius_px * 0.88) & (distances <= radius_px)
    ring_area = max(int(np.count_nonzero(outer_ring)), 1)
    return float(np.count_nonzero(np.asarray(mask, dtype=bool) & outer_ring) / ring_area)


def _select_hybrid_strategy(
    iou: float,
    size_ratio: float,
    classical_fraction: float,
    ml_fraction: float,
) -> tuple[str, str]:
    if classical_fraction >= 0.28 and ml_fraction <= 0.18 and size_ratio >= 3.0 and iou < 0.30:
        return "Intersection", "Classical mask remains substantially larger than ML in a low-overlap case, selecting Intersection"
    if classical_fraction >= 0.70 and ml_fraction <= 0.20 and size_ratio >= 4.0 and iou < 0.18:
        return "Intersection", "Classical mask is much larger than ML for a low-overlap case, selecting Intersection"
    if classical_fraction >= 0.60 and ml_fraction <= 0.18 and size_ratio >= 6.0 and iou < 0.15:
        return "ML", "Classical mask strongly over-expanded relative to ML in a low-overlap case, selecting ML"
    if classical_fraction >= 0.92 and ml_fraction <= 0.75 and iou < 0.2:
        return "ML", "Classical mask is near full-dish while ML is much smaller and IoU < 0.2, selecting ML"
    if classical_fraction >= 0.85 and ml_fraction <= 0.55 and size_ratio > 2.5:
        return "Intersection", "Classical mask is implausibly large relative to ML, selecting Intersection"
    if iou >= 0.7:
        return "ML", "IoU >= 0.7, selecting ML"
    if 0.3 <= iou < 0.7:
        if size_ratio > 1.5:
            return "Intersection", "0.3 <= IoU < 0.7 and size_ratio > 1.5, selecting Intersection"
        if size_ratio < 0.7:
            return "Union", "0.3 <= IoU < 0.7 and size_ratio < 0.7, selecting Union"
        return "ML", "0.3 <= IoU < 0.7 and 0.7 <= size_ratio <= 1.5, selecting ML"
    return "Classical", "IoU < 0.3, selecting Classical"


def _stabilize_mask_against_dish(
    final_mask: np.ndarray,
    classical_mask: np.ndarray,
    ml_mask: np.ndarray,
    pre_sam_mask: np.ndarray,
    dish_mask: np.ndarray,
    center_x: float,
    center_y: float,
    radius_px: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    final = np.asarray(final_mask, dtype=bool) & np.asarray(dish_mask, dtype=bool)
    classical = np.asarray(classical_mask, dtype=bool) & np.asarray(dish_mask, dtype=bool)
    ml = np.asarray(ml_mask, dtype=bool) & np.asarray(dish_mask, dtype=bool)
    base = np.asarray(pre_sam_mask, dtype=bool) & np.asarray(dish_mask, dtype=bool)
    dish = np.asarray(dish_mask, dtype=bool)

    final_fraction = _mask_fraction(final, dish)
    classical_fraction = _mask_fraction(classical, dish)
    ml_fraction = _mask_fraction(ml, dish)
    base_fraction = _mask_fraction(base, dish)
    final_ring = _outer_ring_occupancy(final, center_x, center_y, radius_px)
    ml_ring = _outer_ring_occupancy(ml, center_x, center_y, radius_px)
    classical_ring = _outer_ring_occupancy(classical, center_x, center_y, radius_px)
    classical_ml_iou = _mask_iou(classical, ml) if np.any(classical | ml) else 0.0

    diagnostics = {
        "dish_fraction_final": float(final_fraction),
        "dish_fraction_classical": float(classical_fraction),
        "dish_fraction_ml": float(ml_fraction),
        "dish_fraction_pre_sam": float(base_fraction),
        "outer_ring_occupancy_final": float(final_ring),
        "outer_ring_occupancy_classical": float(classical_ring),
        "outer_ring_occupancy_ml": float(ml_ring),
        "stability_decision": "accepted_final",
    }

    if (
        final_fraction >= 0.92
        and classical_fraction >= 0.85
        and ml_fraction <= 0.75
        and classical_ml_iou < 0.25
    ):
        fallback = np.logical_and(ml, morphology.dilation(base, morphology.disk(8)))
        if np.count_nonzero(fallback) == 0:
            fallback = ml
        fallback = _largest_component(fallback, center_x, center_y, ml if np.any(ml) else base)
        fallback = np.logical_and(fallback, dish)
        diagnostics["stability_decision"] = "fallback_to_ml_constrained"
        diagnostics["stability_reason"] = (
            "Final mask was near full-dish while classical and ML strongly disagreed; "
            "falling back to a constrained ML-based mask."
        )
        diagnostics["dish_fraction_fallback"] = float(_mask_fraction(fallback, dish))
        diagnostics["outer_ring_occupancy_fallback"] = float(_outer_ring_occupancy(fallback, center_x, center_y, radius_px))
        return fallback, diagnostics

    if final_fraction >= 0.97 and final_ring >= 0.97 and ml_fraction < 0.90:
        fallback = np.logical_and(base, morphology.dilation(ml, morphology.disk(12)))
        if np.count_nonzero(fallback) == 0:
            fallback = ml
        fallback = _largest_component(fallback, center_x, center_y, ml if np.any(ml) else base)
        fallback = np.logical_and(fallback, dish)
        diagnostics["stability_decision"] = "fallback_from_full_dish"
        diagnostics["stability_reason"] = "Final mask saturated the dish boundary and was replaced by a more conservative consensus mask."
        diagnostics["dish_fraction_fallback"] = float(_mask_fraction(fallback, dish))
        diagnostics["outer_ring_occupancy_fallback"] = float(_outer_ring_occupancy(fallback, center_x, center_y, radius_px))
        return fallback, diagnostics

    return final, diagnostics


def _blend_gemma_prior(base_mask: np.ndarray, gemma_prior: np.ndarray, dish_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    base = np.asarray(base_mask, dtype=bool) & dish_mask
    prior = np.asarray(gemma_prior, dtype=bool) & dish_mask
    if not np.any(prior):
        return base, {
            "gemma_prior_iou_with_hybrid": 0.0,
            "gemma_prior_area_px": 0,
            "base_hybrid_area_px": int(base.sum()),
            "gemma_blend_decision": "no_prior",
        }
    iou = _mask_iou(base, prior)
    prior_area = max(int(prior.sum()), 1)
    ratio = float(base.sum() / prior_area)
    if iou >= 0.6:
        final = np.logical_or(base, np.logical_and(prior, morphology.dilation(base, morphology.disk(8))))
        decision = "prior_supported_union"
    elif iou >= 0.35:
        final = np.logical_or(np.logical_and(base, morphology.dilation(prior, morphology.disk(10))), np.logical_and(prior, morphology.dilation(base, morphology.disk(10))))
        decision = "prior_consensus_blend"
    else:
        final = base
        decision = "prior_rejected"
    final = np.logical_and(final, dish_mask)
    return final, {
        "gemma_prior_iou_with_hybrid": float(iou),
        "gemma_prior_area_px": int(prior.sum()),
        "base_hybrid_area_px": int(base.sum()),
        "gemma_prior_area_ratio_vs_hybrid": ratio,
        "gemma_blend_decision": decision,
    }


def _predict_sam_candidate(
    predictor: Any,
    image: np.ndarray,
    base_mask: np.ndarray,
    classical_mask: np.ndarray,
    dish_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    rgb = _to_rgb_uint8(image)
    predictor.set_image(rgb)
    prompts = _build_sam_prompts(base_mask, classical_mask, dish_mask)
    masks, scores, _ = predictor.predict(
        point_coords=prompts["point_coords"],
        point_labels=prompts["point_labels"],
        box=prompts["box"],
        multimask_output=True,
    )
    dish = np.asarray(dish_mask, dtype=bool)
    base = np.asarray(base_mask, dtype=bool)
    classical = np.asarray(classical_mask, dtype=bool)
    best_mask: np.ndarray | None = None
    best_score = -1e9
    best_diag: dict[str, Any] = {}

    for idx, candidate in enumerate(masks):
        cand = np.asarray(candidate, dtype=bool) & dish
        area = int(cand.sum())
        base_area = max(int(base.sum()), 1)
        area_ratio = area / base_area
        iou_base = _mask_iou(cand, base)
        iou_classical = _mask_iou(cand, classical)
        pred_score = float(scores[idx])
        score = pred_score + 0.45 * iou_base + 0.25 * iou_classical - 0.10 * abs(np.log(max(area_ratio, 1e-6)))
        if score > best_score:
            best_score = score
            best_mask = cand
            best_diag = {
                "sam_candidate_index": idx,
                "sam_predicted_score": pred_score,
                "sam_area": area,
                "sam_area_ratio_vs_base": float(area_ratio),
                "sam_iou_with_base": float(iou_base),
                "sam_iou_with_classical": float(iou_classical),
                "sam_candidate_score": float(score),
            }
    if best_mask is None:
        return base, {"sam_decision": "predict_failed"}
    best_diag["prompt_box"] = prompts["box"].tolist() if prompts["box"] is not None else None
    best_diag["num_positive_points"] = int((prompts["point_labels"] == 1).sum())
    best_diag["num_negative_points"] = int((prompts["point_labels"] == 0).sum())
    return best_mask.astype(bool), best_diag


def _refine_hybrid_mask_with_sam(
    image: np.ndarray,
    base_mask: np.ndarray,
    classical_mask: np.ndarray,
    dish_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    predictor = _load_sam_predictor()
    sam_mask, sam_diag = _predict_sam_candidate(
        predictor=predictor,
        image=image,
        base_mask=base_mask,
        classical_mask=classical_mask,
        dish_mask=dish_mask,
    )
    base_iou_classical = _mask_iou(base_mask, classical_mask)
    sam_iou_base = _mask_iou(sam_mask, base_mask)
    sam_iou_classical = _mask_iou(sam_mask, classical_mask)
    base_area = max(int(base_mask.sum()), 1)
    sam_area_ratio = float(sam_mask.sum() / base_area)

    if sam_iou_base >= 0.60:
        if sam_area_ratio > 1.35:
            final = (sam_mask & base_mask) | (sam_mask & classical_mask)
            decision = "sam_constrained_by_base"
        elif sam_area_ratio < 0.75:
            final = sam_mask | (base_mask & classical_mask)
            decision = "sam_recovered_inside_base"
        else:
            final = sam_mask
            decision = "sam_direct_refinement"
    elif sam_iou_classical >= max(0.55, base_iou_classical + 0.10):
        final = sam_mask | (base_mask & classical_mask)
        decision = "sam_supported_by_classical"
    else:
        final = base_mask
        decision = "base_mask_retained"

    final = np.asarray(final, dtype=bool) & np.asarray(dish_mask, dtype=bool)
    diagnostics = {
        "sam_decision": decision,
        "base_iou_with_classical": float(base_iou_classical),
        "sam_iou_with_base": float(sam_iou_base),
        "sam_iou_with_classical": float(sam_iou_classical),
        "sam_area_ratio_vs_base": float(sam_area_ratio),
        "sam_area_px": int(sam_mask.sum()),
        "sam_final_area_px": int(final.sum()),
    }
    diagnostics.update(sam_diag)
    return final, diagnostics


def _refine_colony_mask(
    rgb: np.ndarray,
    gray: np.ndarray,
    dish_mask: np.ndarray,
    center_x: float,
    center_y: float,
    initial_mask: np.ndarray,
) -> np.ndarray:
    corrected_gray = _correct_background(gray, dish_mask)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0].astype(np.float32)
    blurred = cv2.GaussianBlur(corrected_gray, (0, 0), sigmaX=9)
    texture = np.abs(corrected_gray.astype(np.float32) - blurred.astype(np.float32))
    corrected_f = corrected_gray.astype(np.float32)
    yy, xx = np.indices(gray.shape)
    radial_distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    radial_fraction = radial_distance / max(radius_px := max(float(np.sqrt(np.count_nonzero(dish_mask) / math.pi)), 1.0), 1.0)

    dish_values_l = lightness[dish_mask]
    dish_values_t = texture[dish_mask]
    dish_values_c = corrected_f[dish_mask]
    if dish_values_l.size == 0 or dish_values_c.size == 0:
        return initial_mask

    l_low, l_high = np.percentile(dish_values_l, [5, 95])
    t_low, t_high = np.percentile(dish_values_t, [5, 95])
    c_low, c_mid, c_high = np.percentile(dish_values_c, [10, 50, 90])
    scaled_l = np.clip((lightness - l_low) / max(l_high - l_low, 1.0), 0.0, 1.0)
    scaled_t = np.clip((texture - t_low) / max(t_high - t_low, 1.0), 0.0, 1.0)
    contrast_scale = max(c_high - c_low, 1.0)
    scaled_c = np.clip(np.abs(corrected_f - c_mid) / contrast_scale, 0.0, 1.0)
    gradient = sobel(corrected_f / 255.0).astype(np.float32)
    gradient_hi = float(np.percentile(gradient[dish_mask], 95)) if np.any(dish_mask) else 1.0
    scaled_g = np.clip(gradient / max(gradient_hi, 1e-3), 0.0, 1.0)
    radial_prior = 1.0 - np.clip((radial_fraction - 0.76) / 0.24, 0.0, 1.0)
    radial_prior = np.clip(radial_prior, 0.2, 1.0)
    score = (
        (0.38 * scaled_c)
        + (0.28 * scaled_t)
        + (0.20 * scaled_g)
        + (0.14 * np.maximum(scaled_l, 1.0 - scaled_l))
    ) * radial_prior

    score_uint8 = np.clip(score * 255.0, 0, 255).astype(np.uint8)
    threshold = filters.threshold_otsu(score_uint8[dish_mask]) if np.count_nonzero(dish_mask) > 32 else 128
    low_threshold = max(0.0, float(threshold) * 0.82)
    high_candidate = (score_uint8 >= float(threshold)) & dish_mask
    candidate = (score_uint8 >= low_threshold) & dish_mask
    support_region = radial_fraction <= 0.88

    if np.any(initial_mask):
        grown_initial = morphology.dilation(initial_mask, morphology.disk(25))
        candidate = np.logical_or(candidate, np.logical_and(initial_mask, dish_mask))
        candidate = np.logical_or(candidate, np.logical_and(grown_initial, candidate))
        high_candidate = np.logical_or(high_candidate, morphology.erosion(grown_initial, morphology.disk(5)))
    else:
        high_candidate = np.logical_and(high_candidate, support_region)

    seed_mask = np.logical_and(high_candidate, support_region)
    if np.any(seed_mask):
        labeled = measure.label(candidate.astype(np.uint8), connectivity=2)
        regions = measure.regionprops(labeled)
        if regions:
            best_score = -float("inf")
            selected = np.zeros_like(candidate, dtype=bool)
            for region in regions:
                component = labeled == region.label
                overlap_seed = int(np.count_nonzero(component & seed_mask))
                if overlap_seed == 0 and region.area < 0.01 * np.count_nonzero(dish_mask):
                    continue
                centroid_y, centroid_x = region.centroid
                distance_penalty = math.hypot(centroid_x - center_x, centroid_y - center_y)
                outer_penalty = _outer_ring_occupancy(component, center_x, center_y, radius_px)
                overlap_initial = (
                    float(np.count_nonzero(component & initial_mask)) / max(np.count_nonzero(initial_mask), 1)
                    if np.any(initial_mask)
                    else 0.0
                )
                region_score = (
                    float(region.area)
                    + (overlap_seed * 60.0)
                    + (overlap_initial * 15000.0)
                    - (outer_penalty * region.area * 1.25)
                    - (distance_penalty * 18.0)
                )
                if region_score > best_score:
                    best_score = region_score
                    selected = component
            if np.any(selected):
                candidate = selected

    candidate = np.logical_and(candidate, dish_mask)

    try:
        grabcut_mask = np.full(gray.shape, cv2.GC_BGD, dtype=np.uint8)
        grabcut_mask[dish_mask] = cv2.GC_PR_BGD

        fg_seed = np.zeros_like(dish_mask, dtype=bool)
        if np.any(initial_mask):
            fg_seed = morphology.erosion(initial_mask, morphology.disk(15))
        if not np.any(fg_seed):
            high_score_threshold = np.percentile(score_uint8[dish_mask], 82) if np.any(dish_mask) else 200
            fg_seed = (score_uint8 >= high_score_threshold) & dish_mask
            fg_seed = morphology.erosion(fg_seed, morphology.disk(7))
        fg_seed = np.logical_and(fg_seed, support_region)

        bg_seed = (~dish_mask) | ((score_uint8 <= np.percentile(score_uint8[dish_mask], 22)) & dish_mask)
        bg_seed = np.logical_or(bg_seed, radial_fraction >= 0.97)

        grabcut_mask[bg_seed] = cv2.GC_BGD
        grabcut_mask[np.logical_and(candidate, dish_mask)] = cv2.GC_PR_FGD
        grabcut_mask[fg_seed] = cv2.GC_FGD

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.grabCut(bgr, grabcut_mask, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)
        grabcut_candidate = np.isin(grabcut_mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
        candidate = np.logical_or(candidate, grabcut_candidate)
    except cv2.error:
        pass

    candidate = morphology.closing(candidate, morphology.disk(11))
    candidate = morphology.remove_small_objects(candidate, max(int(np.count_nonzero(dish_mask) * 0.005), 1000))
    candidate = morphology.remove_small_holes(candidate, max(int(np.count_nonzero(dish_mask) * 0.01), 2000))
    candidate = _largest_component(candidate, center_x, center_y, initial_mask if np.any(initial_mask) else None)
    candidate = morphology.closing(candidate, morphology.disk(9))
    candidate = morphology.remove_small_holes(candidate, max(int(np.count_nonzero(dish_mask) * 0.02), 3000))
    candidate = np.logical_and(candidate, dish_mask)

    if np.any(candidate):
        candidate_fraction = _mask_fraction(candidate, dish_mask)
        candidate_outer = _outer_ring_occupancy(candidate, center_x, center_y, radius_px)
        initial_fraction = _mask_fraction(initial_mask, dish_mask) if np.any(initial_mask) else 0.0
        if (
            np.any(initial_mask)
            and candidate_fraction >= max(0.65, initial_fraction * 2.3)
            and candidate_outer >= 0.30
            and _mask_iou(candidate, morphology.dilation(initial_mask, morphology.disk(35))) < 0.45
        ):
            constrained = candidate & morphology.dilation(initial_mask, morphology.disk(95))
            constrained = morphology.closing(constrained, morphology.disk(7))
            constrained = morphology.remove_small_holes(
                constrained,
                max(int(np.count_nonzero(dish_mask) * 0.01), 2000),
            )
            constrained = _largest_component(constrained, center_x, center_y, initial_mask)
            if np.any(constrained):
                candidate = constrained

    if not np.any(candidate):
        return np.logical_and(initial_mask, dish_mask) if np.any(initial_mask) else _build_mask_if_missing(gray, center_x, center_y, min(gray.shape) * 0.45)
    return candidate


def _restrict_to_internal_band(colony_mask: np.ndarray, pixel_size_mm: float, band_mm: float = 3.0) -> np.ndarray:
    colony = np.asarray(colony_mask, dtype=bool)
    if colony.sum() == 0:
        return colony
    radius_px = max(1, int(round(float(band_mm) / max(float(pixel_size_mm), 1e-6))))
    inner = morphology.erosion(colony, morphology.disk(radius_px))
    if inner.sum() == 0:
        return colony
    return inner.astype(bool)


def _detect_cracks(image: np.ndarray, mask: np.ndarray, std_threshold: float = 0.3, min_size: int = 15) -> tuple[np.ndarray, np.ndarray, float]:
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool), np.zeros_like(mask, dtype=bool), 0.0

    filled_mask = ndimage.binary_fill_holes(mask)
    try:
        analysis_mask = convex_hull_image(filled_mask)
    except Exception:
        analysis_mask = filled_mask

    values = image[analysis_mask]
    mean_val = float(values.mean()) if values.size else 0.0
    std_val = float(values.std()) if values.size else 0.0
    threshold = mean_val - std_threshold * std_val

    smoothed = gaussian(image, sigma=1.5)
    edges = sobel(smoothed)
    edges[~analysis_mask] = 0
    canny_edges = canny(smoothed, sigma=1.5, low_threshold=0.02, high_threshold=0.08) & analysis_mask

    masked_image = np.asarray(image, dtype=float).copy()
    masked_image[~analysis_mask] = mean_val
    local_thresh = threshold_local(masked_image, block_size=51, offset=0.02)
    local_dark = (image < local_thresh) & analysis_mask
    global_dark = (image < threshold) & analysis_mask

    edge_threshold = np.percentile(edges[analysis_mask], 70) if np.any(analysis_mask) else 0.0
    strong_edges = edges > edge_threshold
    crack_mask = (local_dark | global_dark) & (strong_edges | canny_edges)

    very_dark = (image < mean_val - 2.0 * std_val) & analysis_mask
    crack_mask = crack_mask | very_dark

    props = measure.regionprops(analysis_mask.astype(int))
    if props:
        centroid = props[0].centroid
        y_coords, x_coords = np.ogrid[:image.shape[0], :image.shape[1]]
        dist_from_center = np.sqrt((y_coords - centroid[0]) ** 2 + (x_coords - centroid[1]) ** 2)
        equiv_radius = np.sqrt(analysis_mask.sum() / np.pi)
        central_mask = (dist_from_center < equiv_radius * 0.7) & analysis_mask
    else:
        central_mask = analysis_mask

    local_bright = (image > local_thresh + 0.01) & analysis_mask
    global_bright = (image > mean_val + std_threshold * 0.5 * std_val) & analysis_mask
    very_bright = (image > mean_val + 0.8 * std_val) & analysis_mask
    central_bright = (image > mean_val + 0.5 * std_val) & central_mask
    bright_cracks = (local_bright | global_bright) & (strong_edges | canny_edges)
    crack_mask = crack_mask | bright_cracks | very_bright | central_bright

    crack_mask = dilation(crack_mask, morphology.disk(2)) & analysis_mask
    if crack_mask.sum() > 0:
        crack_mask = remove_small_objects(crack_mask.astype(bool), min_size=min_size)
    crack_skeleton = skeletonize(crack_mask) if crack_mask.sum() > 0 else np.zeros_like(crack_mask, dtype=bool)
    return crack_mask.astype(bool), crack_skeleton.astype(bool), float(threshold)


def _measure_crack_properties(crack_skeleton: np.ndarray) -> dict[str, float | int]:
    if crack_skeleton.sum() == 0:
        return {
            "total_length_px": 0,
            "num_segments": 0,
            "mean_segment_length_px": 0.0,
        }
    labeled, num_segments = ndimage.label(crack_skeleton)
    total_length = int(crack_skeleton.sum())
    mean_length = total_length / num_segments if num_segments > 0 else 0.0
    return {
        "total_length_px": total_length,
        "num_segments": int(num_segments),
        "mean_segment_length_px": float(mean_length),
    }


def _crack_polylines_from_skeleton(crack_skeleton: np.ndarray, width: int, height: int) -> list[list[dict[str, float]]]:
    contours, _ = cv2.findContours(crack_skeleton.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    polylines: list[list[dict[str, float]]] = []
    for contour in contours:
        points = contour.reshape(-1, 2).astype(float)
        if len(points) < 2:
            continue
        epsilon = max(1.0, 0.01 * cv2.arcLength(contour, False))
        approx = cv2.approxPolyDP(contour, epsilon, False).reshape(-1, 2).astype(float)
        if len(approx) < 2:
            continue
        polylines.append(_pixels_to_normalized_points(approx, width, height))
    return polylines


def _polygon_from_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 2), dtype=float)
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(2.0, 0.006 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(float)
    if len(approx) < 3:
        return np.empty((0, 2), dtype=float)
    approx[:, 0] = np.clip(approx[:, 0], 0, width - 1)
    approx[:, 1] = np.clip(approx[:, 1], 0, height - 1)
    return approx


def _texture_from_mask(gray: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    masked = gray[mask]
    if masked.size == 0:
        masked = gray.reshape(-1)

    energy = float(np.mean((masked / 255.0) ** 2))
    contrast = float(np.std(masked))
    horizontal_diff = np.abs(np.diff(gray.astype(float), axis=1))
    homogeneity = float(1.0 / (1.0 + np.mean(horizontal_diff)))
    correlation = float(np.clip(np.corrcoef(masked[:-1], masked[1:])[0, 1] if masked.size > 2 else 0.0, -1.0, 1.0))

    coords = np.column_stack(np.nonzero(mask))
    if coords.size == 0:
        coords = np.column_stack(np.nonzero(np.ones_like(mask, dtype=bool)))
    centroid = coords.mean(axis=0)
    distances = np.sqrt((coords[:, 0] - centroid[0]) ** 2 + (coords[:, 1] - centroid[1]) ** 2)
    max_distance = float(distances.max()) if distances.size else 1.0
    zonation = {}
    for label, lower, upper in (
        ("core", 0.0, 1.0 / 3.0),
        ("middle", 1.0 / 3.0, 2.0 / 3.0),
        ("outer", 2.0 / 3.0, 1.0),
    ):
        selector = np.logical_and(distances >= lower * max_distance, distances < upper * max_distance)
        zone_pixels = gray[coords[selector][:, 0], coords[selector][:, 1]] if np.any(selector) else masked
        zonation[label] = float(np.std(zone_pixels))

    return {
        "contrast": contrast,
        "correlation": correlation,
        "energy": energy,
        "homogeneity": homogeneity,
        "radialZonation": zonation,
    }


def _shannon_entropy(masked_values: np.ndarray, bins: int = 32) -> float:
    if masked_values.size == 0:
        return 0.0
    hist, _ = np.histogram(masked_values, bins=bins, range=(0, 255), density=True)
    hist = hist[hist > 0]
    if hist.size == 0:
        return 0.0
    return float(-(hist * np.log2(hist)).sum())


def _edge_roughness(area_px: float, perimeter_px: float) -> float:
    if area_px <= 0 or perimeter_px <= 0:
        return 0.0
    equivalent_circle_perimeter = 2.0 * math.sqrt(math.pi * area_px)
    return float(max(perimeter_px / max(equivalent_circle_perimeter, 1e-6) - 1.0, 0.0))


def _radial_intensity_profile(
    gray: np.ndarray,
    mask: np.ndarray,
    pixel_to_mm: float,
    bins: int = 24,
) -> dict[str, Any]:
    coords = np.column_stack(np.nonzero(mask))
    if coords.size == 0:
        return {
            "radiusFraction": [],
            "radiusMm": [],
            "meanIntensity": [],
            "ringSpacingMm": 0.0,
            "centerToEdgeDelta": 0.0,
            "densityIndex": 0.0,
        }

    centroid = coords.mean(axis=0)
    distances = np.sqrt((coords[:, 0] - centroid[0]) ** 2 + (coords[:, 1] - centroid[1]) ** 2)
    max_distance = float(distances.max()) if distances.size else 1.0
    bin_edges = np.linspace(0.0, max_distance, bins + 1)
    profile_means: list[float] = []
    radius_fraction: list[float] = []
    radius_mm: list[float] = []

    for start, end in zip(bin_edges[:-1], bin_edges[1:]):
        selector = (distances >= start) & (distances < end)
        if not np.any(selector):
            continue
        band_pixels = gray[coords[selector][:, 0], coords[selector][:, 1]]
        profile_means.append(float(np.mean(band_pixels)))
        midpoint = (start + end) / 2.0
        radius_fraction.append(float(midpoint / max(max_distance, 1e-6)))
        radius_mm.append(float(midpoint * pixel_to_mm))

    if not profile_means:
        return {
            "radiusFraction": [],
            "radiusMm": [],
            "meanIntensity": [],
            "ringSpacingMm": 0.0,
            "centerToEdgeDelta": 0.0,
            "densityIndex": 0.0,
        }

    smoothed = ndimage.gaussian_filter1d(np.asarray(profile_means, dtype=float), sigma=1.0, mode="nearest")
    peak_indices, _ = signal.find_peaks(smoothed, distance=max(1, bins // 8))
    if len(peak_indices) >= 2:
        peak_positions = np.asarray(radius_mm, dtype=float)[peak_indices]
        ring_spacing_mm = float(np.mean(np.diff(peak_positions)))
    else:
        ring_spacing_mm = 0.0

    center_slice = max(1, len(profile_means) // 3)
    edge_slice = max(1, len(profile_means) // 4)
    center_mean = float(np.mean(profile_means[:center_slice]))
    edge_mean = float(np.mean(profile_means[-edge_slice:]))
    colony_pixels = gray[mask]
    density_index = float(np.clip(1.0 - (np.mean(colony_pixels) / 255.0), 0.0, 1.0))

    return {
        "radiusFraction": radius_fraction,
        "radiusMm": radius_mm,
        "meanIntensity": profile_means,
        "ringSpacingMm": ring_spacing_mm,
        "centerToEdgeDelta": edge_mean - center_mean,
        "densityIndex": density_index,
    }


def _shape_metrics(mask: np.ndarray) -> tuple[float, float, float]:
    labeled = measure.label(mask.astype(np.uint8))
    regions = measure.regionprops(labeled)
    if not regions:
        return 0.0, 0.0, 0.0
    region = max(regions, key=lambda item: item.area)
    perimeter = float(region.perimeter if region.perimeter > 0 else 0.0)
    circularity = float(4 * math.pi * region.area / perimeter**2) if perimeter > 0 else 0.0
    return float(region.area), perimeter, float(region.eccentricity)


def analyze_image(
    image_path: Path,
    engine: str,
    model_dir: Path = LOCAL_MODEL_DIR,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
    day_override: int | None = None,
) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    rgb = np.array(image)
    gray = np.array(image.convert("L"))
    detected_center_x, detected_center_y, detected_radius_px = _detect_dish_geometry(gray)
    detected_dish_center, detected_dish_radius = _pixels_to_normalized(
        detected_center_x, detected_center_y, detected_radius_px, width, height
    )
    segmentation_prompt = _build_segmentation_prompt(detected_dish_center, detected_dish_radius)
    gemma_prior_used = True
    gemma_skip_reason: str | None = None

    if engine == "local":
        if platform.system() == "Darwin" and platform.machine() == "arm64" and _mlx_runtime_available():
            try:
                raw = _analyze_with_mlx_model(image_path, LOCAL_MODEL_ID, segmentation_prompt)
                engine_model = LOCAL_MODEL_ID
            except Exception as exc:
                raw = _empty_gemma_prior()
                engine_model = f"{LOCAL_MODEL_ID} (Gemma prior skipped)"
                gemma_prior_used = False
                gemma_skip_reason = f"MLX Gemma inference failed: {exc}"
        elif model_dir.exists() and model_dir != LOCAL_MODEL_DIR:
            raw = _analyze_with_local_model(image_path, model_dir, segmentation_prompt)
            engine_model = str(model_dir)
        else:
            raw = _empty_gemma_prior()
            engine_model = f"{LOCAL_MODEL_ID} (Gemma prior skipped)"
            gemma_prior_used = False
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                gemma_skip_reason = _mlx_runtime_reason()
            else:
                gemma_skip_reason = "This machine is not Apple Silicon, so the MLX Gemma prior is unavailable."
    elif engine == "gemini":
        raw = _analyze_with_gemini(image_path, gemini_model, segmentation_prompt)
        engine_model = gemini_model
    else:
        raise ValueError(f"Unsupported engine: {engine}")

    center_x = detected_center_x
    center_y = detected_center_y
    radius_px = detected_radius_px
    pixel_to_mm = (PETRI_DISH_DIAMETER_MM / 2.0) / max(radius_px, 1.0)
    dish_mask = _circle_mask(width, height, center_x, center_y, radius_px)

    raw["dish_center"] = detected_dish_center
    raw["dish_radius"] = detected_dish_radius

    polygon_xy = _normalized_to_pixels(raw.get("colony_polygon", []), width, height)
    polygon_xy = _clip_points_to_dish(polygon_xy, center_x, center_y, radius_px)
    gemma_prior_mask = _mask_from_polygon(width, height, polygon_xy)
    if np.any(gemma_prior_mask):
        gemma_prior_mask = np.logical_and(gemma_prior_mask, dish_mask)

    classical_mask = _refine_colony_mask(rgb, gray, dish_mask, center_x, center_y, gemma_prior_mask)
    unet_mask = _predict_unet_mask(rgb, dish_mask)
    unet_mask = _largest_component(unet_mask, center_x, center_y, gemma_prior_mask if np.any(gemma_prior_mask) else classical_mask)

    classical_area_px = int(np.count_nonzero(classical_mask))
    unet_area_px = int(np.count_nonzero(unet_mask))
    classical_vs_unet_iou = _mask_iou(classical_mask, unet_mask) if np.any(classical_mask | unet_mask) else 0.0
    size_ratio = float(classical_area_px / max(unet_area_px, 1))
    classical_fraction = _mask_fraction(classical_mask, dish_mask)
    unet_fraction = _mask_fraction(unet_mask, dish_mask)
    strategy, strategy_reason = _select_hybrid_strategy(
        classical_vs_unet_iou,
        size_ratio,
        classical_fraction,
        unet_fraction,
    )
    hybrid_mask = _apply_hybrid_strategy(unet_mask, classical_mask, strategy)

    pre_sam_mask, gemma_diag = _blend_gemma_prior(hybrid_mask, gemma_prior_mask, dish_mask)
    sam_diag: dict[str, Any] = {"sam_decision": "not_run"}
    try:
        mask, sam_diag = _refine_hybrid_mask_with_sam(
            image=rgb,
            base_mask=pre_sam_mask,
            classical_mask=classical_mask,
            dish_mask=dish_mask,
        )
    except Exception as exc:
        mask = pre_sam_mask
        sam_diag = {"sam_decision": "skipped", "sam_error": str(exc)}

    mask = _largest_component(mask, center_x, center_y, pre_sam_mask if np.any(pre_sam_mask) else classical_mask)
    mask = np.logical_and(mask, dish_mask)
    mask, stability_diag = _stabilize_mask_against_dish(
        final_mask=mask,
        classical_mask=classical_mask,
        ml_mask=unet_mask,
        pre_sam_mask=pre_sam_mask,
        dish_mask=dish_mask,
        center_x=center_x,
        center_y=center_y,
        radius_px=radius_px,
    )

    initial_area_px = int(np.count_nonzero(gemma_prior_mask))
    refined_area_px = int(np.count_nonzero(mask))
    prior_iou = _mask_iou(gemma_prior_mask, mask) if np.any(gemma_prior_mask) else 0.0
    initial_centroid_x, initial_centroid_y = _mask_centroid(gemma_prior_mask)
    refined_centroid_x, refined_centroid_y = _mask_centroid(mask)
    centroid_shift_px = math.hypot(refined_centroid_x - initial_centroid_x, refined_centroid_y - initial_centroid_y)
    refined_polygon_xy = _polygon_from_mask(mask, width, height)
    if len(refined_polygon_xy) >= 3:
        polygon_xy = _clip_points_to_dish(refined_polygon_xy, center_x, center_y, radius_px)
    raw["colony_polygon"] = _pixels_to_normalized_points(polygon_xy, width, height)

    area_px, perimeter_px, eccentricity = _shape_metrics(mask)
    area_mm2 = float(area_px * (pixel_to_mm**2))
    perimeter_mm = float(perimeter_px * pixel_to_mm)
    equivalent_radius_mm = float(math.sqrt(area_mm2 / math.pi)) if area_mm2 > 0 else 0.0
    diameter_mm = equivalent_radius_mm * 2.0
    circularity = float(4 * math.pi * area_px / perimeter_px**2) if perimeter_px > 0 else 0.0

    raw["morphology_estimates"] = {
        "area_mm2": area_mm2,
        "perimeter_mm": perimeter_mm,
        "diameter_mm": diameter_mm,
    }
    raw["segmentation_diagnostics"] = {
        "petri_dish_diameter_mm": PETRI_DISH_DIAMETER_MM,
        "initial_mask_area_px": initial_area_px,
        "classical_mask_area_px": classical_area_px,
        "unet_mask_area_px": unet_area_px,
        "hybrid_pre_sam_area_px": int(np.count_nonzero(pre_sam_mask)),
        "refined_mask_area_px": refined_area_px,
        "refinement_area_ratio": float(refined_area_px / max(initial_area_px, 1)) if initial_area_px > 0 else 0.0,
        "refinement_iou_with_model_prior": prior_iou,
        "refinement_centroid_shift_px": centroid_shift_px,
        "classical_unet_iou": float(classical_vs_unet_iou),
        "classical_unet_size_ratio": float(size_ratio),
        "classical_dish_fraction": float(classical_fraction),
        "unet_dish_fraction": float(unet_fraction),
        "hybrid_strategy": strategy,
        "hybrid_strategy_reason": strategy_reason,
        **gemma_diag,
        **sam_diag,
        **stability_diag,
    }

    texture = _texture_from_mask(gray, mask)
    texture["entropy"] = _shannon_entropy(gray[mask])
    radial_profile = _radial_intensity_profile(gray, mask, pixel_to_mm)
    texture["centerToEdgeDelta"] = radial_profile["centerToEdgeDelta"]
    texture["densityIndex"] = radial_profile["densityIndex"]
    corrected_gray = _correct_background(gray, dish_mask).astype(np.float32) / 255.0
    proportional_band_mm = max(0.25, equivalent_radius_mm * 0.10)
    internal_mask = _restrict_to_internal_band(mask, pixel_to_mm, band_mm=proportional_band_mm)
    crack_mask, crack_skeleton, crack_threshold = _detect_cracks(corrected_gray, internal_mask, std_threshold=0.3, min_size=15)
    crack_props = _measure_crack_properties(crack_skeleton)
    raw["cracks"] = _crack_polylines_from_skeleton(crack_skeleton, width, height)
    raw["internal_band_description"] = (
        f"Internal-band crack analysis used a {proportional_band_mm:.2f} mm inward band with "
        f"{crack_props['num_segments']} detected crack segments."
    )
    raw["crack_analysis"] = {
        "analysis_band_mm": proportional_band_mm,
        "analysis_threshold": crack_threshold,
        "crack_area_px": int(crack_mask.sum()),
        **crack_props,
    }
    raw["radial_profile"] = radial_profile

    total_crack_length_px = float(crack_props["total_length_px"])
    total_crack_length_mm = total_crack_length_px * pixel_to_mm
    coverage_pct = min(100.0, (total_crack_length_px / max(perimeter_px, 1.0)) * 100.0)
    proportional_coverage_pct = min(100.0, (total_crack_length_px / max(math.sqrt(max(area_px, 1.0)), 1.0)) * 10.0)
    edge_roughness = _edge_roughness(area_px, perimeter_px)

    qc_notes = []
    if engine == "local":
        qc_notes.append(f"Local inference using {engine_model}.")
    else:
        qc_notes.append(f"Cloud inference via Gemini API model {gemini_model}.")
    if not gemma_prior_used:
        qc_notes.append(
            f"Gemma prior was skipped. Reason: {gemma_skip_reason or 'Unknown MLX runtime error.'} "
            "The final mask still used classical segmentation, U-Net, and SAM."
        )
    qc_notes.append(f"Dish geometry calibrated from grayscale image edges using an assumed {PETRI_DISH_DIAMETER_MM:.0f} mm petri dish.")
    qc_notes.append("Saved colony geometry is clipped to remain inside the detected dish.")
    qc_notes.append(f"Final colony mask combines Gemma prior, classical segmentation, and local U-Net using step-8 hybrid strategy '{strategy}', then refines with SAM.")
    qc_notes.append("Classical refinement now uses background-deviation, texture, and edge cues so lighter or brighter colonies do not over-expand as easily.")
    if stability_diag.get("stability_decision") != "accepted_final":
        qc_notes.append(
            "Mask sanity checks detected dish-saturating behaviour and replaced the final mask with a more conservative fallback."
        )
    qc_notes.append("Crack analysis is computed classically from the final internal-band mask instead of relying on model crack guesses alone.")
    if not raw.get("colony_polygon"):
        qc_notes.append("Fallback threshold mask used because the model did not return a colony polygon.")

    return {
        "id": image_path.stem,
        "filename": image_path.name,
        "day": day_override if day_override is not None else parse_day(image_path.name),
        "imageUrl": f"/input_images/{quote(image_path.name)}",
        "pixelToMm": pixel_to_mm,
        "morphology": {
            "areaMm2": area_mm2,
            "equivalentRadiusMm": equivalent_radius_mm,
            "diameterMm": diameter_mm,
            "perimeterMm": perimeter_mm,
            "circularity": circularity,
            "eccentricity": eccentricity,
            "edgeRoughness": edge_roughness,
        },
        "texture": texture,
        "cracks": {
            "count": len(raw.get("cracks", [])),
            "totalLengthMm": total_crack_length_mm,
            "coveragePct": coverage_pct,
            "proportionalCoveragePct": proportional_coverage_pct,
            "internalBandSummary": raw.get("internal_band_description") or "No internal bands reported.",
        },
        "kinematics": {
            "radialVelocity": 0.0,
            "areaGrowthRate": 0.0,
            "relativeGrowthRate": 0.0,
            "radialAcceleration": 0.0,
        },
        "qcStatus": "pass",
        "qcNotes": " ".join(qc_notes),
        "rawAnalysis": raw,
        "_engineModel": engine_model,
    }


def _attach_kinematics(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_results = sorted(results, key=lambda item: (item["day"], item["filename"]))
    for index, result in enumerate(sorted_results):
        if index == 0:
            continue
        previous = sorted_results[index - 1]
        delta_t = result["day"] - previous["day"]
        if delta_t <= 0:
            continue
        velocity = (
            result["morphology"]["equivalentRadiusMm"] - previous["morphology"]["equivalentRadiusMm"]
        ) / delta_t
        area_rate = (result["morphology"]["areaMm2"] - previous["morphology"]["areaMm2"]) / delta_t
        relative_growth_rate = area_rate / max(previous["morphology"]["areaMm2"], 1e-6)
        acceleration = 0.0
        if index > 1:
            prior = sorted_results[index - 2]
            prior_dt = previous["day"] - prior["day"]
            if prior_dt > 0:
                prior_velocity = (
                    previous["morphology"]["equivalentRadiusMm"] - prior["morphology"]["equivalentRadiusMm"]
                ) / prior_dt
                acceleration = (velocity - prior_velocity) / delta_t
        result["kinematics"] = {
            "radialVelocity": velocity,
            "areaGrowthRate": area_rate,
            "relativeGrowthRate": relative_growth_rate,
            "radialAcceleration": acceleration,
        }
    return sorted_results


def write_outputs(results: list[dict[str, Any]], engine: str, output_dir: Path) -> AnalysisRun:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / f"{timestamp}_{engine}"
    run_dir.mkdir(parents=True, exist_ok=True)

    sanitized_results = []
    engine_model = ""
    for result in results:
        item = dict(result)
        engine_model = str(item.pop("_engineModel", ""))
        sanitized_results.append(item)

    analysis_json = run_dir / "analysis.json"
    analysis_csv = run_dir / "analysis.csv"
    manifest_json = run_dir / "manifest.json"

    analysis_json.write_text(json.dumps(sanitized_results, indent=2), encoding="utf-8")

    with analysis_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "filename",
                "day",
                "area_mm2",
                "radius_mm",
                "diameter_mm",
                "perimeter_mm",
                "circularity",
                "eccentricity",
                "edge_roughness",
                "contrast",
                "correlation",
                "energy",
                "homogeneity",
                "texture_entropy",
                "center_to_edge_intensity_delta",
                "density_index",
                "ring_spacing_mm",
                "radial_velocity_mm_per_day",
                "area_growth_rate_mm2_per_day",
                "relative_growth_rate_per_day",
                "crack_count",
                "crack_length_mm",
                "crack_coverage_pct",
                "qc_status",
            ]
        )
        for result in sanitized_results:
            writer.writerow(
                [
                    result["filename"],
                    result["day"],
                    result["morphology"]["areaMm2"],
                    result["morphology"]["equivalentRadiusMm"],
                    result["morphology"]["diameterMm"],
                    result["morphology"]["perimeterMm"],
                    result["morphology"]["circularity"],
                    result["morphology"]["eccentricity"],
                    result["morphology"]["edgeRoughness"],
                    result["texture"]["contrast"],
                    result["texture"]["correlation"],
                    result["texture"]["energy"],
                    result["texture"]["homogeneity"],
                    result["texture"]["entropy"],
                    result["texture"]["centerToEdgeDelta"],
                    result["texture"]["densityIndex"],
                    result.get("rawAnalysis", {}).get("radial_profile", {}).get("ringSpacingMm", 0.0),
                    result["kinematics"]["radialVelocity"],
                    result["kinematics"]["areaGrowthRate"],
                    result["kinematics"]["relativeGrowthRate"],
                    result["cracks"]["count"],
                    result["cracks"]["totalLengthMm"],
                    result["cracks"]["coveragePct"],
                    result["qcStatus"],
                ]
            )

    run = AnalysisRun(
        engine=engine,
        engine_model=engine_model,
        created_at=datetime.now(timezone.utc).isoformat(),
        output_dir=f"/outputs/{run_dir.name}",
        analysis_json=f"/outputs/{run_dir.name}/analysis.json",
        analysis_csv=f"/outputs/{run_dir.name}/analysis.csv",
    )
    manifest_json.write_text(json.dumps(asdict(run), indent=2), encoding="utf-8")
    return run


def _emit_progress(current: int, total: int, stage: str) -> None:
    print(f"[progress] {current}/{total} {stage}", file=sys.stderr, flush=True)


def run_analysis_batch(
    engine: str,
    input_dir: Path,
    output_dir: Path,
    filenames: list[str] | None = None,
    model_dir: Path = LOCAL_MODEL_DIR,
    gemini_model: str = DEFAULT_GEMINI_MODEL,
) -> dict[str, Any]:
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = [input_dir / filename for filename in filenames] if filenames else list_input_images(input_dir)
    images = [path for path in requested if path.exists()]
    if not images:
        raise FileNotFoundError(f"No input images found in {input_dir}")
    inferred_days = infer_days_for_images(images)

    total = len(images)
    results: list[dict[str, Any]] = []
    _emit_progress(0, total, f"Preparing {total} image(s)")
    for index, image_path in enumerate(images, start=1):
        _emit_progress(index - 1, total, f"Analyzing {image_path.name}")
        results.append(
            analyze_image(
                image_path,
                engine=engine,
                model_dir=model_dir,
                gemini_model=gemini_model,
                day_override=inferred_days.get(image_path),
            )
        )
        _emit_progress(index, total, f"Finished {image_path.name}")
    results = _attach_kinematics(results)
    _emit_progress(total, total, "Writing outputs")
    run = write_outputs(results, engine=engine, output_dir=output_dir)
    _emit_progress(total, total, f"Saved run to {run.output_dir}")
    return {
        "results": results,
        "run": asdict(run),
    }
