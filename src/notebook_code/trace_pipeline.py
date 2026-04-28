from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pipeline.analysis import (
    LOCAL_SAM_MODEL_TYPE,
    CONTAINER_DIAMETER_MM,
    _analyze_with_gemini,
    _analyze_with_local_model,
    _analyze_with_mlx_model,
    _apply_hybrid_strategy,
    _blend_generic_prior,
    _build_segmentation_prompt,
    _circle_mask,
    _correct_background,
    _crack_polylines_from_skeleton,
    _detect_cracks,
    _detect_container_geometry,
    _edge_roughness,
    _largest_component,
    _mask_fraction,
    _mask_from_polygon,
    _mask_iou,
    _measure_crack_properties,
    _mlx_runtime_available,
    _mlx_runtime_reason,
    _predict_unet_mask,
    _pixels_to_normalized,
    _normalized_to_pixels,
    _radial_intensity_profile,
    _shannon_entropy,
    _refine_sample_mask,
    _refine_hybrid_mask_with_sam,
    _restrict_to_internal_band,
    _select_hybrid_strategy,
    _shape_metrics,
    _stabilize_mask_against_container,
    _texture_from_mask,
)

from . import get_runtime, _display_image_name


def trace_pipeline_image(image_path: Path, engine: str, generic_model: str) -> dict[str, Any]:
    runtime = get_runtime()
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    rgb = np.array(image)
    gray = np.array(image.convert("L"))

    center_x, center_y, radius_px = _detect_container_geometry(gray)
    detected_container_center, detected_container_radius = _pixels_to_normalized(center_x, center_y, radius_px, width, height)
    segmentation_prompt = _build_segmentation_prompt(detected_container_center, detected_container_radius)
    pixel_to_mm = (CONTAINER_DIAMETER_MM / 2.0) / max(radius_px, 1.0)
    container_mask = _circle_mask(width, height, center_x, center_y, radius_px)

    generic_prior_used = True
    generic_skip_reason = None
    if engine == "local":
        if _mlx_runtime_available():
            try:
                raw = _analyze_with_mlx_model(image_path, runtime.notebook_local_model_id, segmentation_prompt)
                engine_model = runtime.notebook_local_model_id
            except Exception as exc:
                raw = _analyze_with_local_model(image_path, gemini_model, segmentation_prompt)
                gemma_prior_used = False
                gemma_skip_reason = f"MLX prior failed: {exc}"
                engine_model = gemini_model
        else:
            raw = _analyze_with_local_model(image_path, gemini_model, segmentation_prompt)
            gemma_prior_used = False
            gemma_skip_reason = _mlx_runtime_reason()
            engine_model = gemini_model
    else:
        raw = _analyze_with_gemini(image_path, gemini_model, segmentation_prompt)
        engine_model = gemini_model

    sample_polygon = raw.get("sample_polygon", [])
    polygon_xy = _normalized_to_pixels(sample_polygon, width, height)
    generic_prior_mask = _mask_from_polygon(width, height, polygon_xy)
    if not np.any(generic_prior_mask):
        generic_prior_mask = np.zeros((height, width), dtype=bool)

    classical_mask = _refine_sample_mask(
        rgb,
        gray,
        container_mask,
        center_x,
        center_y,
        generic_prior_mask,
    )
    classical_diag: dict[str, Any] = {}
    unet_mask = _predict_unet_mask(rgb, container_mask, model_path=runtime.local_unet_checkpoint)
    unet_mask = _largest_component(unet_mask, center_x, center_y, classical_mask)

    classical_area_px = int(np.count_nonzero(classical_mask))
    unet_area_px = int(np.count_nonzero(unet_mask))
    classical_fraction = _mask_fraction(classical_mask, container_mask)
    unet_fraction = _mask_fraction(unet_mask, container_mask)
    iou = _mask_iou(classical_mask, unet_mask) if np.any(classical_mask | unet_mask) else 0.0
    size_ratio = classical_area_px / max(unet_area_px, 1)
    strategy, strategy_reason = _select_hybrid_strategy(iou, size_ratio, classical_fraction, unet_fraction)
    hybrid_mask = _apply_hybrid_strategy(unet_mask, classical_mask, strategy)
    pre_generic_mask, generic_diag = _blend_generic_prior(hybrid_mask, generic_prior_mask, container_mask)

    generic_mask, generic_diag2 = _refine_hybrid_mask_with_generic(rgb, pre_generic_mask, classical_mask, container_mask)
    final_mask, stability_diag = _stabilize_mask_against_container(
        generic_mask,
        classical_mask,
        unet_mask,
        pre_generic_mask,
        container_mask,
        center_x,
        center_y,
        radius_px,
    )
    if not np.any(final_mask):
        final_mask = generic_mask if np.any(generic_mask) else pre_generic_mask

    corrected_gray = _correct_background(gray, container_mask)
    proportional_band_mm = 3.0
    final_mask = _restrict_to_internal_band(final_mask, pixel_to_mm, band_mm=proportional_band_mm)

    crack_mask, crack_skeleton, crack_threshold = _detect_cracks(corrected_gray, final_mask, std_threshold=0.3, min_size=15)
    crack_props = _measure_crack_properties(crack_skeleton)
    area_px, perimeter_px, eccentricity = _shape_metrics(final_mask)
    area_mm2 = float(area_px * (pixel_to_mm**2))
    equivalent_radius_mm = float(np.sqrt(area_mm2 / np.pi)) if area_mm2 > 0 else 0.0
    diameter_mm = float(2.0 * equivalent_radius_mm)
    perimeter_mm = float(perimeter_px * pixel_to_mm)
    circularity = float((4.0 * np.pi * area_px) / max(perimeter_px**2, 1.0)) if area_px > 0 else 0.0
    morphology = {
        "areaMm2": area_mm2,
        "equivalentRadiusMm": equivalent_radius_mm,
        "diameterMm": diameter_mm,
        "perimeterMm": perimeter_mm,
        "circularity": circularity,
        "eccentricity": float(eccentricity),
    }
    radial_profile = _radial_intensity_profile(gray, final_mask, pixel_to_mm)
    texture = _texture_from_mask(gray, final_mask)
    texture["entropy"] = float(_shannon_entropy(gray[final_mask]))
    texture["centerToEdgeDelta"] = float(radial_profile["centerToEdgeDelta"])
    texture["densityIndex"] = float(radial_profile["densityIndex"])
    morphology["edgeRoughness"] = _edge_roughness(area_px, perimeter_px)

    result = {
        "id": image_path.stem,
        "filename": _display_image_name(image_path),
        "imageUrl": "/input_images/" + "/".join(Path(_display_image_name(image_path)).parts),
        "day": 0,
        "pixelToMm": pixel_to_mm,
        "morphology": morphology,
        "texture": texture,
        "cracks": {
            "count": int(crack_props.get("num_segments", 0)),
            "totalLengthMm": float(crack_props.get("total_length_px", 0.0) * pixel_to_mm),
            "coveragePct": float(min(100.0, (float(crack_props.get("total_length_px", 0.0)) / max(perimeter_px, 1.0)) * 100.0)),
            "proportionalCoveragePct": float(min(100.0, (float(crack_props.get("total_length_px", 0.0)) / max(np.sqrt(max(area_px, 1.0)), 1.0)) * 10.0)),
            "internalBandSummary": f"Internal-band crack analysis used a {proportional_band_mm:.2f} mm inward band with {int(crack_props.get('num_segments', 0))} detected crack segments.",
        },
        "kinematics": {
            "radialVelocity": 0.0,
            "areaGrowthRate": 0.0,
            "relativeGrowthRate": 0.0,
            "radialAcceleration": 0.0,
        },
        "qcStatus": "pass",
        "qcNotes": " ".join([
            f"Local trace for {_display_image_name(image_path)}.",
            f"Final sample mask uses the GUI-aligned hybrid/generic path.",
        ]),
        "rawAnalysis": {
            "radial_profile": radial_profile,
            "cracks": raw.get("cracks", []),
            "morphology_estimates": raw.get("morphology_estimates", {}),
        },
    }
    return {
        "image_path": image_path,
        "rgb": rgb,
        "gray": gray,
        "center_x": center_x,
        "center_y": center_y,
        "radius_px": radius_px,
        "detected_container_center": detected_container_center,
        "detected_container_radius": detected_container_radius,
        "segmentation_prompt": segmentation_prompt,
        "pixel_to_mm": pixel_to_mm,
        "container_mask": container_mask,
        "generic_prior_mask": generic_prior_mask,
        "generic_prior_used": generic_prior_used,
        "generic_skip_reason": generic_skip_reason,
        "engine_model": engine_model,
        "classical_mask": classical_mask,
        "classical_diag": classical_diag,
        "unet_mask": unet_mask,
        "hybrid_mask": hybrid_mask,
        "pre_generic_mask": pre_generic_mask,
        "generic_diag": generic_diag,
        "generic_mask": generic_mask,
        "final_mask": final_mask,
        "generic_diag2": generic_diag2,
        "stability_diag": stability_diag,
        "internal_mask": final_mask,
        "crack_mask": crack_mask,
        "crack_skeleton": crack_skeleton,
        "crack_threshold": crack_threshold,
        "crack_props": crack_props,
        "proportional_band_mm": proportional_band_mm,
        "radial_profile": radial_profile,
        "payload": result,
        "segmentation_diagnostics": {
            "hybrid_strategy": strategy,
            "hybrid_strategy_reason": strategy_reason,
            "classical_unet_iou": iou,
            "classical_unet_size_ratio": size_ratio,
            "classical_container_fraction": classical_fraction,
            "unet_container_fraction": unet_fraction,
        },
    }
