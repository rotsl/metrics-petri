from __future__ import annotations

import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display
from PIL import Image

from pipeline.analysis import (
    DEFAULT_GEMINI_MODEL,
    LOCAL_MODEL_DIR,
    LOCAL_MODEL_ID,
    LOCAL_SAM_MODEL_TYPE,
    PETRI_DISH_DIAMETER_MM,
    _analyze_with_gemini,
    _analyze_with_local_model,
    _analyze_with_mlx_model,
    _apply_hybrid_strategy,
    _blend_gemma_prior,
    _build_segmentation_prompt,
    _circle_mask,
    _clip_points_to_dish,
    _correct_background,
    _crack_polylines_from_skeleton,
    _detect_cracks,
    _detect_dish_geometry,
    _edge_roughness,
    _empty_gemma_prior,
    _largest_component,
    _mask_fraction,
    _mask_from_polygon,
    _mask_iou,
    _measure_crack_properties,
    _mlx_runtime_available,
    _mlx_runtime_reason,
    _normalized_to_pixels,
    _pixels_to_normalized,
    _pixels_to_normalized_points,
    _predict_unet_mask,
    _radial_intensity_profile,
    _refine_colony_mask,
    _refine_hybrid_mask_with_sam,
    _restrict_to_internal_band,
    _select_hybrid_strategy,
    _shape_metrics,
    _stabilize_mask_against_dish,
    _texture_from_mask,
    list_input_images,
)


@dataclass(frozen=True)
class NotebookRuntime:
    root: Path
    input_dir: Path
    output_dir: Path
    archives_dir: Path
    local_gemma_model_dir: Path
    notebook_local_model_id: str
    local_unet_checkpoint: Path
    local_sam_checkpoint: Path


_RUNTIME: NotebookRuntime | None = None


def resolve_notebook_runtime(cwd: Path) -> NotebookRuntime:
    candidates = [cwd, cwd.parent]
    root = next(
        (
            candidate
            for candidate in candidates
            if (candidate / "pipeline").exists() and (candidate / "input_images").exists()
        ),
        cwd,
    )
    local_generic_model_dir = LOCAL_MODEL_DIR if LOCAL_MODEL_DIR.is_absolute() else root / LOCAL_MODEL_DIR
    local_unet_checkpoint = root / "models" / "best_unet.pt"
    local_sam_checkpoint = root / "models" / "sam_vit_b_01ec64.pth"
    return NotebookRuntime(
        root=root,
        input_dir=root / "input_images",
        output_dir=root / "outputs",
        archives_dir=root / "archives",
        local_generic_model_dir=local_generic_model_dir,
        notebook_local_model_id=str(local_generic_model_dir),
        local_unet_checkpoint=local_unet_checkpoint,
        local_sam_checkpoint=local_sam_checkpoint,
    )


def set_runtime(runtime: NotebookRuntime) -> None:
    global _RUNTIME
    _RUNTIME = runtime


def get_runtime() -> NotebookRuntime:
    if _RUNTIME is None:
        raise RuntimeError("Notebook runtime has not been initialised.")
    return _RUNTIME


def configure_pipeline_defaults(analysis_module: Any, runtime: NotebookRuntime) -> None:
    analysis_module.LOCAL_MODEL_DIR = runtime.local_generic_model_dir
    analysis_module.LOCAL_MODEL_ID = runtime.notebook_local_model_id
    analysis_module.LOCAL_UNET_PATH = runtime.local_unet_checkpoint
    analysis_module.LOCAL_SAM_CHECKPOINT = runtime.local_sam_checkpoint
    analysis_module._load_unet_segmenter.__defaults__ = (runtime.local_unet_checkpoint,)
    analysis_module._predict_unet_mask.__defaults__ = (runtime.local_unet_checkpoint,)
    analysis_module._load_sam_predictor.__defaults__ = (runtime.local_sam_checkpoint, LOCAL_SAM_MODEL_TYPE)
    analysis_module._SAM_PREDICTOR = None
    analysis_module._MLX_BUNDLE = None
    analysis_module._MLX_RUNTIME_OK = None
    analysis_module._MLX_RUNTIME_REASON = None


def _display_image_name(path: Path) -> str:
    runtime = get_runtime()
    try:
        return path.relative_to(runtime.input_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _visible_input_images(directory: Path) -> list[Path]:
    return [
        path
        for path in list_input_images(directory)
        if path.is_file() and path.name not in {".gitkeep", ".DS_Store"}
    ]


def _read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(22, 163, 74), alpha: float = 0.35) -> np.ndarray:
    base = np.asarray(rgb, dtype=np.uint8).copy()
    overlay = base.copy()
    overlay[np.asarray(mask, dtype=bool)] = np.array(color, dtype=np.uint8)
    blended = cv2.addWeighted(base, 1.0 - alpha, overlay, alpha, 0)
    contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (255, 255, 255), 2)
    return blended


def _draw_dish_circle(rgb: np.ndarray, center_x: float, center_y: float, radius_px: float) -> np.ndarray:
    preview = np.asarray(rgb, dtype=np.uint8).copy()
    cv2.circle(preview, (int(round(center_x)), int(round(center_y))), int(round(radius_px)), (59, 130, 246), 5)
    cv2.circle(preview, (int(round(center_x)), int(round(center_y))), 8, (244, 63, 94), -1)
    return preview


def _show_grid(paths: list[Path], title: str, max_items: int = 12, cols: int = 3) -> None:
    subset = paths[:max_items]
    if not subset:
        display(Markdown(f"**{title}**\n\nNo images to show."))
        return
    rows = math.ceil(len(subset) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.0))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")
    for ax, path in zip(axes.flat, subset):
        ax.imshow(_read_rgb(path))
        ax.set_title(_display_image_name(path), fontsize=9, pad=10)
        ax.axis("off")
    fig.suptitle(f"{title}\n(relative paths include subfolders)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def _show_panels(images: list[np.ndarray], titles: list[str], cols: int = 3) -> None:
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.6, rows * 4.0))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image, title in zip(axes.flat, images, titles):
        ax.imshow(image, cmap="gray" if getattr(image, "ndim", 3) == 2 else None)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


from .trace_sections import *  # noqa: F401,F403


__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "NotebookRuntime",
    "LOCAL_MODEL_DIR",
    "LOCAL_MODEL_ID",
    "configure_pipeline_defaults",
    "resolve_notebook_runtime",
    "set_runtime",
    "get_runtime",
    "_display_image_name",
    "_draw_dish_circle",
    "_overlay_mask",
    "_read_rgb",
    "_show_grid",
    "_show_panels",
    "_visible_input_images",
    "_mlx_runtime_available",
    "_mlx_runtime_reason",
]
