from __future__ import annotations

from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from . import _draw_container_circle, _overlay_mask, _show_panels


def _heading(trace: dict[str, Any]) -> None:
    display(Markdown(f"### {trace['image_path'].name}"))


def render_trace_overview(traces: list[dict[str, Any]]) -> None:
    print("Built traces for", len(traces), "image(s).")
    display(
        pd.DataFrame(
            [
                {
                    "filename": trace["image_path"].name,
                    "engine_model": trace["engine_model"],
                    "gemma_prior_used": trace["gemma_prior_used"],
                    "final_area_mm2": trace["payload"]["morphology"]["areaMm2"],
                }
                for trace in traces
            ]
        )
    )


def render_container_sections(traces: list[dict[str, Any]]) -> None:
    for trace in traces:
        _heading(trace)
        _show_panels(
            [
                trace["rgb"],
                _draw_container_circle(trace["rgb"], trace["center_x"], trace["center_y"], trace["radius_px"]),
                trace["container_mask"].astype(float),
            ],
            ["RGB input", "Detected container overlay", "Container mask"],
            cols=3,
        )
        print("pixel_to_mm =", trace["pixel_to_mm"])
        print(
            "container center (px) =",
            (round(trace["center_x"], 2), round(trace["center_y"], 2)),
        )
        print("container radius (px) =", round(trace["radius_px"], 2))
        print("generic_prior_used:", trace["generic_prior_used"])


def render_segmentation_sections(traces: list[dict[str, Any]]) -> None:
    for trace in traces:
        _heading(trace)
        _show_panels(
            [
                _overlay_mask(trace["rgb"], trace["classical_mask"], color=(22, 163, 74), alpha=0.35),
                _overlay_mask(trace["rgb"], trace["unet_mask"], color=(14, 165, 233), alpha=0.35),
                _overlay_mask(trace["rgb"], trace["pre_sam_mask"], color=(249, 115, 22), alpha=0.35),
                np.logical_xor(trace["classical_mask"], trace["unet_mask"]).astype(float),
            ],
            [
                "Classical overlay",
                "U-Net overlay",
                "Hybrid pre-SAM overlay",
                "Classical/U-Net disagreement",
            ],
            cols=2,
        )
        diag = trace["segmentation_diagnostics"]
        print("hybrid strategy =", diag["hybrid_strategy"])
        print("strategy reason =", diag["hybrid_strategy_reason"])
        print("classical_unet_iou =", diag["classical_unet_iou"])
        print("classical_unet_size_ratio =", diag["classical_unet_size_ratio"])
        print("classical_dish_fraction =", diag["classical_dish_fraction"])
        print("unet_dish_fraction =", diag["unet_dish_fraction"])


def render_prior_sections(traces: list[dict[str, Any]]) -> None:
    for trace in traces:
        _heading(trace)
        prior_title = f"Prior source: {trace['engine_model']}" if trace["gemma_prior_used"] else "Prior skipped"
        prior_overlay = trace["rgb"].copy()
        if np.any(trace["gemma_prior_mask"]):
            prior_overlay = _overlay_mask(trace["rgb"], trace["gemma_prior_mask"], color=(168, 85, 247), alpha=0.22)
            prior_title = f"Prior source: {trace['engine_model']}" if trace["generic_prior_used"] else "Prior skipped"
            prior_overlay = trace["rgb"].copy()
            if np.any(trace["generic_prior_mask"]):
                prior_overlay = _overlay_mask(trace["rgb"], trace["generic_prior_mask"], color=(168, 85, 247), alpha=0.22)
        _show_panels(
            [
                prior_overlay,
                _overlay_mask(trace["rgb"], trace["hybrid_mask"], color=(249, 115, 22), alpha=0.35),
                _overlay_mask(trace["rgb"], trace["pre_sam_mask"], color=(236, 72, 153), alpha=0.35),
            ],
            [prior_title, "Hybrid mask before prior blend", "Blended pre-SAM mask"],
            cols=3,
        )
        print("segmentation prompt =")
        print(trace["segmentation_prompt"])
        if not trace["gemma_prior_used"]:
        if not trace["generic_prior_used"]:
            print("prior note =", trace["generic_skip_reason"])
        print("prior area (px) =", int(trace["generic_prior_mask"].sum()))
        print("hybrid area before prior blend (px) =", int(trace["hybrid_mask"].sum()))
        print("area after prior blend (pre-generic, px) =", int(trace["pre_generic_mask"].sum()))
        print("generic diagnostics =", trace["generic_diag"])


def render_sam_sections(traces: list[dict[str, Any]]) -> None:
    for trace in traces:
        _heading(trace)
        _show_panels(
            [
                _overlay_mask(trace["rgb"], trace["pre_sam_mask"], color=(249, 115, 22), alpha=0.35),
                _overlay_mask(trace["rgb"], trace["sam_mask"], color=(168, 85, 247), alpha=0.35),
                _overlay_mask(trace["rgb"], trace["final_mask"], color=(239, 68, 68), alpha=0.35),
            ],
            ["Pre-SAM hybrid", "SAM-refined mask", "Stable final mask"],
            cols=3,
        )
        print("SAM diagnostics =", trace["sam_diag"])
        print("stability diagnostics =", trace["stability_diag"])


def render_crack_sections(traces: list[dict[str, Any]]) -> None:
    for trace in traces:
        _heading(trace)
        _show_panels(
            [
                _overlay_mask(trace["rgb"], trace["final_mask"], color=(239, 68, 68), alpha=0.35),
                trace["internal_mask"].astype(float),
                trace["crack_mask"].astype(float),
                trace["crack_skeleton"].astype(float),
            ],
            ["Stable final mask", "Internal band mask", "Crack candidate mask", "Crack skeleton"],
            cols=2,
        )
        profile_df = pd.DataFrame(
            {
                "radius_mm": trace["radial_profile"]["radiusMm"],
                "mean_intensity": trace["radial_profile"]["meanIntensity"],
            }
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(profile_df["radius_mm"], profile_df["mean_intensity"], marker="o", linewidth=2)
        ax.set_title(f"Radial intensity profile: {trace['image_path'].name}")
        ax.set_xlabel("Radius from colony center (mm)")
            ax.set_xlabel("Radius from sample center (mm)")
        ax.set_ylabel("Mean grayscale intensity")
        ax.grid(alpha=0.25)
        plt.show()
        print("internal analysis band (mm) =", trace["proportional_band_mm"])
        print("crack threshold =", trace["crack_threshold"])
        print("crack properties =", trace.get("crack_props", {}))


def render_summary_sections(traces: list[dict[str, Any]]) -> None:
    for trace in traces:
        _heading(trace)
        payload = trace["payload"]
        summary = pd.Series(
            {
                "engine_model": trace["engine_model"],
                "area_mm2": payload["morphology"]["areaMm2"],
                "equivalent_radius_mm": payload["morphology"]["equivalentRadiusMm"],
                "diameter_mm": payload["morphology"]["diameterMm"],
                "perimeter_mm": payload["morphology"]["perimeterMm"],
                "circularity": payload["morphology"]["circularity"],
                "eccentricity": payload["morphology"]["eccentricity"],
                "edge_roughness": payload["morphology"]["edgeRoughness"],
                "texture_entropy": payload["texture"]["entropy"],
                "center_to_edge_delta": payload["texture"]["centerToEdgeDelta"],
                "density_index": payload["texture"]["densityIndex"],
                "ring_spacing_mm": trace["radial_profile"]["ringSpacingMm"],
                "crack_count": payload["cracks"]["count"],
                "crack_length_mm": payload["cracks"]["totalLengthMm"],
                "crack_coverage_pct": payload["cracks"]["coveragePct"],
            }
        )
        display(summary.to_frame("value"))
        _show_panels(
            [
                _overlay_mask(trace["rgb"], trace["final_mask"], color=(239, 68, 68), alpha=0.35),
                trace["final_mask"].astype(float),
            ],
            ["Final overlay", "Final mask"],
            cols=2,
        )


__all__ = [
    "render_trace_overview",
    "render_dish_sections",
    "render_segmentation_sections",
    "render_prior_sections",
    "render_sam_sections",
    "render_crack_sections",
    "render_summary_sections",
]
