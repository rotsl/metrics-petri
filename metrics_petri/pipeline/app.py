# SPDX-License-Identifier: MIT
"""Gradio GUI for metrics-petri colony segmentation and analysis.

Run via:
    metrics-petri-gui
    python -m pipeline
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import signal
import tempfile
import threading
import time as _time
import zipfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch
from PIL import Image, ExifTags

from .analysis import (
    CONTAINER_MM,
    compute_metrics,
    create_full_overlays,
    detect_container,
    detect_cracks,
    detect_hyphae,
    infer_mask,
    load_model,
    make_growth_charts,
)

# ── constants ──────────────────────────────────────────────────────────────
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".gif",
    ".heic", ".heif", ".dng", ".cr2", ".nef", ".arw", ".raf", ".orf", ".rw2",
}
THUMB_SIZE = (160, 160)
DATE_RE = re.compile(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])")
MAX_IMAGES = 50
CSS = (
    ".gallery-wrap{max-height:65vh;overflow-y:auto} "
    ".footer-text{text-align:center;margin-top:8px}"
)
_META_COLS = ["filename", "image_date", "experiment_date", "day_code"]


# ── image helpers ──────────────────────────────────────────────────────────

def load_as_pil(path: str) -> Image.Image:
    """Open any supported image format and return an RGB PIL image."""
    suffix = Path(path).suffix.lower()
    if suffix in {".heic", ".heif"}:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            pass
    if suffix in {".dng", ".cr2", ".nef", ".arw", ".raf", ".orf", ".rw2", ".raw"}:
        try:
            import rawpy
            with rawpy.imread(path) as raw:
                rgb = raw.postprocess()
            return Image.fromarray(rgb)
        except Exception:
            pass
    return Image.open(path).convert("RGB")


def make_thumbnail(p: str) -> Image.Image:
    try:
        im = load_as_pil(p)
        im.thumbnail(THUMB_SIZE, Image.LANCZOS)
        return im
    except Exception:
        return Image.new("RGB", THUMB_SIZE, (200, 200, 200))


def detect_image_date(p: str) -> str:
    """Extract image date from EXIF DateTimeOriginal or filename pattern.

    Returns ISO date string, or empty string when no reliable date is found.
    File mtime is intentionally NOT used — Gradio temp copies have current mtime.
    """
    # 1. Try EXIF DateTimeOriginal (most reliable for camera images)
    try:
        im = Image.open(p)
        ex = im.getexif()
        if ex:
            for tid, tn in ExifTags.TAGS.items():
                if tn == "DateTimeOriginal":
                    v = ex.get(tid)
                    if v:
                        return dt.datetime.strptime(v, "%Y:%m:%d %H:%M:%S").date().isoformat()
    except Exception:
        pass
    # 2. Try YYYYMMDD or YYYY-MM-DD pattern in the filename stem
    m = DATE_RE.search(Path(p).stem)
    if m:
        try:
            return dt.date(int(m[1]), int(m[2]), int(m[3])).isoformat()
        except Exception:
            pass
    # 3. No reliable date found — return empty so the user can fill it in Step 3
    return ""


def day_code(img_d: str, exp_d: str) -> str:
    try:
        d = (dt.date.fromisoformat(img_d) - dt.date.fromisoformat(exp_d)).days + 1
        return f"d{max(d, 1):02d}"
    except Exception:
        return ""


def _build_meta_table(paths: list, dates: dict, orig_names: dict, exp_date_str: str) -> pd.DataFrame:
    rows = []
    for p in paths:
        orig = orig_names.get(p, Path(p).name)
        imd = dates.get(p, "")
        dc = day_code(imd, exp_date_str) if exp_date_str and imd else ""
        rows.append({
            "filename": orig,
            "image_date": imd,
            "experiment_date": exp_date_str or "",
            "day_code": dc,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=_META_COLS)


def _zip_stem(user_name: str, exp_name: str) -> str:
    """Return a filesystem-safe stem for the output zip (user_name preferred, then exp_name)."""
    for raw in (user_name, exp_name):
        s = re.sub(r"[^\w-]", "", (raw or "").strip().replace(" ", "_")).lower()
        if s:
            return s
    return "analysis"


def write_ics(rems: list[dict], path: str) -> None:
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//MetricsPetri//EN"]
    for r in rems:
        uid = r["image_path"].replace("/", "_")
        ds = r["remind_me"].replace("-", "").replace(" ", "T").replace(":", "") + "00"
        lines += [
            "BEGIN:VEVENT",
            f"UID:{uid}@mp",
            f"DTSTART:{ds}",
            f"SUMMARY:Reminder — {r['experiment_name']}: {Path(r['image_path']).name}",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    with open(path, "w") as f:
        f.write("\r\n".join(lines))


# ── Gradio app ─────────────────────────────────────────────────────────────

with gr.Blocks(title="metrics-petri Colony Segmentation") as demo:
    paths_st = gr.State([])
    dates_st = gr.State({})
    rems_st = gr.State({})
    orig_names_st = gr.State({})
    cur_idx = gr.State(-1)
    results_st = gr.State([])

    gr.Markdown(
        "# 🔬 metrics-petri — Colony Segmentation\n"
        "Upload → **Run Inference** → instant results  |  Toggle *Full Pipeline* for morphometrics\n\n"
        "Model: `models/best_area_w_0.7.pt` · SmallUNet (area-consistency w=0.7)"
    )

    with gr.Accordion("📂 Step 1 — Upload Images", open=True):
        upload = gr.File(
            label="Drag & drop petri dish images (JPEG, PNG, TIFF, BMP, WebP, HEIF, RAW)",
            file_count="multiple",
            file_types=["image", ".heic", ".heif", ".dng", ".cr2", ".nef", ".arw", ".raf", ".orf", ".rw2"],
        )
        up_st = gr.Markdown("")

    with gr.Accordion("⚙️ Step 2 — Settings", open=True):
        with gr.Row():
            threshold_slider = gr.Slider(
                label="Mask confidence threshold",
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.01,
            )
            full_pipeline_cb = gr.Checkbox(
                label="Full Pipeline (slower: dish detection, cracks, hyphae, morphometrics)",
                value=False,
            )
        with gr.Row():
            exp_name = gr.Textbox(label="Experiment Name", placeholder="MagExp01")
            exp_date = gr.Textbox(label="Experiment Date (YYYY-MM-DD)", placeholder="2025-04-01")
            user_name = gr.Textbox(label="User Name", placeholder="Your name")
            plates_count = gr.Number(
                label="Plates", value=1, minimum=1, maximum=200, precision=0
            )

    with gr.Accordion("🖼️ Step 3 — Review & Edit Dates", open=False):
        gr.Markdown(
            "Image dates are auto-extracted from EXIF metadata, filename pattern (YYYYMMDD), "
            "or file modification time. Day codes are computed from *(image date − experiment date)*. "
            "Click a thumbnail to edit an individual image's date."
        )
        meta_table = gr.Dataframe(
            label="Extracted metadata",
            headers=_META_COLS,
            interactive=False,
            wrap=True,
        )
        with gr.Row():
            with gr.Column(scale=2):
                gallery = gr.Gallery(
                    label="Images",
                    columns=4,
                    height=400,
                    object_fit="contain",
                    allow_preview=False,
                    interactive=False,
                )
            with gr.Column(scale=1):
                sel_img = gr.Image(label="Selected", height=200, interactive=False)
                sel_fn = gr.Textbox(label="Filename", interactive=False)
                sel_dt = gr.Textbox(label="Image Date (YYYY-MM-DD)", interactive=True)
                sel_dc = gr.Textbox(label="Day Code", interactive=False)
                sel_rm = gr.Textbox(
                    label="Remind Me",
                    placeholder="YYYY-MM-DD HH:MM",
                    interactive=True,
                )
                sv_btn = gr.Button("💾 Save Date", variant="primary")
                sv_st = gr.Markdown("")

    with gr.Accordion("📥 Step 4 — Export Metadata", open=False):
        exp_btn = gr.Button("📥 Export CSV / JSON / ICS", variant="primary")
        exp_st = gr.Markdown("")
        meta_preview = gr.Dataframe(
            label="image_metadata.csv", interactive=False, wrap=True
        )
        meta_dl = gr.File(label="⬇️ Download metadata zip", interactive=False)

    with gr.Accordion("▶️ Step 5 — Run Inference", open=True):
        run_btn = gr.Button("▶ Run Inference", variant="primary", size="lg")
        run_st = gr.Markdown("")
        gr.Markdown("### Results")
        overlay_gallery = gr.Gallery(
            label="Segmentation results",
            columns=3,
            height=500,
            object_fit="contain",
            allow_preview=True,
        )
        gr.Markdown("### Growth Charts (full pipeline, ≥2 images)")
        chart_gallery = gr.Gallery(
            label="Growth curves",
            columns=3,
            height=400,
            object_fit="contain",
            allow_preview=True,
        )
        gr.Markdown("### Results Table (full pipeline)")
        results_df = gr.Dataframe(
            label="analysis_full.csv", interactive=False, wrap=True
        )
        results_dl = gr.File(label="⬇️ Download analysis zip", interactive=False)

    with gr.Group(visible=False, elem_id="close-confirm-panel") as close_panel:
        gr.HTML(
            "<div style='border:2px solid #ef4444;border-radius:8px;padding:16px;"
            "background:#fff5f5;margin:8px 0'>"
            "<h3 style='color:#dc2626;margin:0 0 8px'>Stop metrics-petri server?</h3>"
            "<p style='margin:0 0 10px'>This will kill the process on <strong>port 7860</strong>.</p>"
            "<p style='margin:0'><strong>To restart:</strong></p>"
            "<pre style='background:#f1f5f9;padding:8px;border-radius:4px;"
            "font-size:0.9em;margin:6px 0'>metrics-petri-gui</pre>"
            "<p style='margin:4px 0'>or from the repository:</p>"
            "<pre style='background:#f1f5f9;padding:8px;border-radius:4px;"
            "font-size:0.9em;margin:6px 0'>make run-gui</pre>"
            "</div>"
        )
        close_status = gr.Markdown("")
        with gr.Row():
            cancel_close_btn = gr.Button("Cancel", variant="secondary")
            confirm_close_btn = gr.Button("⏹ Shut down now", variant="stop")

    with gr.Row():
        gr.HTML(
            "<div style='padding-top:10px;border-top:1px solid #e5e7eb;"
            "color:#6b7280;font-size:0.85em'>"
            "Developed by "
            "<a href='https://www.tsl.ac.uk/about/people/rohan-rebello' target='_blank' "
            "style='color:#4f46e5;text-decoration:none'>Rohan R</a>"
            " &nbsp;·&nbsp; MIT"
            "</div>"
        )
        close_btn = gr.Button("⏹ Close", variant="stop", size="sm", scale=0)

    # ── event handlers ─────────────────────────────────────────────────────

    def on_upload(files, ed):
        if not files:
            return [], {}, {}, {}, [], "", -1, pd.DataFrame(columns=_META_COLS)

        paths: list[str] = []
        orig_names: dict[str, str] = {}

        for f in files[:MAX_IMAGES]:
            orig_path = str(f)
            orig_name = Path(f).name
            suffix = Path(f).suffix.lower()
            if suffix not in IMAGE_EXTS:
                continue
            # Convert HEIF/RAW to a temp PNG the rest of the pipeline can read
            if suffix not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".gif"}:
                try:
                    img_pil = load_as_pil(orig_path)
                    tmp_f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    img_pil.save(tmp_f.name)
                    converted = tmp_f.name
                except Exception:
                    continue
            else:
                converted = orig_path
            paths.append(converted)
            orig_names[converted] = orig_name

        if not paths:
            return [], {}, {}, {}, [], "⚠️ No supported images found.", -1, pd.DataFrame(columns=_META_COLS)

        dates = {p: detect_image_date(p) for p in paths}
        rems = {p: "" for p in paths}
        gallery_items = [(p, orig_names.get(p, Path(p).name)) for p in paths]

        # Auto-populate experiment date with earliest detected image date so d01 anchors correctly.
        # Only use non-empty dates; if no dates found, leave exp_date for the user to fill.
        effective_ed = ed
        if not ed:
            known = [d for d in dates.values() if d]
            if known:
                effective_ed = min(known)

        table = _build_meta_table(paths, dates, orig_names, effective_ed)

        n_dated = sum(1 for d in dates.values() if d)
        date_note = (
            f" — dates extracted for {n_dated}/{len(paths)} images"
            if n_dated < len(paths)
            else ""
        )

        return (
            paths, dates, rems, orig_names, gallery_items,
            f"✅ **{len(paths)}** images loaded{date_note}.",
            -1, table, effective_ed,
        )

    upload.upload(
        on_upload,
        [upload, exp_date],
        [paths_st, dates_st, rems_st, orig_names_st, gallery, up_st, cur_idx, meta_table, exp_date],
    )

    def on_exp_date_change(paths, dates, orig_names, ed):
        return _build_meta_table(paths, dates, orig_names, ed)

    exp_date.change(
        on_exp_date_change,
        [paths_st, dates_st, orig_names_st, exp_date],
        [meta_table],
    )

    def on_sel(paths, dates, rems, orig_names, ed, evt: gr.SelectData):
        i = evt.index
        if i < 0 or i >= len(paths):
            return -1, None, "", "", "", ""
        p = paths[i]
        orig = orig_names.get(p, Path(p).name)
        imd = dates.get(p, "")
        return (
            i,
            make_thumbnail(p),
            orig,
            imd,
            day_code(imd, ed) if ed and imd else "",
            rems.get(p, ""),
        )

    gallery.select(
        on_sel,
        [paths_st, dates_st, rems_st, orig_names_st, exp_date],
        [cur_idx, sel_img, sel_fn, sel_dt, sel_dc, sel_rm],
    )

    def on_save(paths, dates, rems, orig_names, i, nd, nr, ed):
        if i < 0 or i >= len(paths):
            return dates, rems, "", "⚠️ Select an image first.", _build_meta_table(paths, dates, orig_names, ed)
        p = paths[i]
        dates = dict(dates)
        rems = dict(rems)
        dates[p] = nd
        rems[p] = nr
        orig = orig_names.get(p, Path(p).name)
        return (
            dates, rems,
            day_code(nd, ed) if ed and nd else "",
            f"✅ **{orig}** → {nd}",
            _build_meta_table(paths, dates, orig_names, ed),
        )

    sv_btn.click(
        on_save,
        [paths_st, dates_st, rems_st, orig_names_st, cur_idx, sel_dt, sel_rm, exp_date],
        [dates_st, rems_st, sel_dc, sv_st, meta_table],
    )

    def on_export(paths, dates, rems, orig_names, en, ed, un, pc):
        if not paths:
            return "⚠️ Upload images first.", None, None
        tmp = tempfile.mkdtemp()
        rows: list[dict] = []
        rl: list[dict] = []
        for p in paths:
            imd = dates.get(p, detect_image_date(p))
            rm = rems.get(p, "")
            orig = orig_names.get(p, Path(p).name)
            row = dict(
                image_path=orig,
                experiment_name=en or "",
                experiment_date=ed or "",
                image_date=imd,
                day_code=day_code(imd, ed) if ed and imd else "",
                user_name=un or "",
                plates_count=int(pc) if pc else 1,
                remind_me=rm,
            )
            rows.append(row)
            if rm.strip():
                rl.append(dict(row))
        cp = Path(tmp) / "image_metadata.csv"
        with open(cp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        jp = Path(tmp) / "image_metadata.json"
        with open(jp, "w") as f:
            json.dump(rows, f, indent=2)
        zf_list = [cp, jp]
        if rl:
            ip = Path(tmp) / "reminders.ics"
            write_ics(rl, str(ip))
            zf_list.append(ip)
        zp = Path(tmp) / "image_metadata.zip"
        with zipfile.ZipFile(zp, "w") as z:
            for f2 in zf_list:
                z.write(f2, f2.name)
        return f"✅ Exported **{len(rows)}** images.", pd.DataFrame(rows), str(zp)

    exp_btn.click(
        on_export,
        [paths_st, dates_st, rems_st, orig_names_st, exp_name, exp_date, user_name, plates_count],
        [exp_st, meta_preview, meta_dl],
    )

    def on_run(paths, dates, rems, orig_names, en, ed, un, pc, thresh, full_pipeline, progress=gr.Progress()):
        if not paths:
            return "⚠️ Upload images first.", [], [], None, None, []
        try:
            load_model()
        except Exception as e:
            return f"❌ Model failed to load: {e}", [], [], None, None, []

        results: list[dict] = []
        vis: list[tuple] = []
        errors: list[str] = []

        # ── fast mode (segmentation only) ───────────────────────────────
        if not full_pipeline:
            for p in progress.tqdm(paths, desc="Segmenting"):
                orig = orig_names.get(p, Path(p).name)
                try:
                    img = load_as_pil(p)
                    overlay, mask = infer_mask(img, thresh)
                    vis.append((img, f"{orig} — Raw"))
                    vis.append((mask, f"{orig} — Mask"))
                    vis.append((overlay, f"{orig} — Overlay"))
                except Exception as e:
                    errors.append(f"{orig}: {e}")
            em = f"\n\n⚠️ Errors: {'; '.join(errors)}" if errors else ""
            ok = len(paths) - len(errors)
            return (
                f"✅ **{ok}/{len(paths)}** segmented (fast mode, threshold={thresh:.2f}).{em}",
                vis, [], None, None, [],
            )

        # ── full pipeline ────────────────────────────────────────────────
        for p in progress.tqdm(paths, desc="Full pipeline"):
            orig = orig_names.get(p, Path(p).name)
            imd = dates.get(p, detect_image_date(p))
            try:
                img_bgr = cv2.imread(str(p))
                if img_bgr is None:
                    raise RuntimeError(f"Cannot read: {p}")

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                from .analysis import IMAGE_SIZE, DEVICE
                img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
                model = load_model()
                x = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
                x = x.unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    prob = model(x)[0, 0].detach().cpu().numpy()
                h, w = img_bgr.shape[:2]
                mask_small = (prob > thresh).astype(np.uint8) * 255
                colony_mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST) > 0

                dish_info = detect_container(img_bgr)
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

                if dish_info:
                    dcx, dcy, dr, px2mm = dish_info
                else:
                    dcx, dcy = w // 2, h // 2
                    dr = min(h, w) // 2
                    px2mm = 1.0

                crack_mask = detect_cracks(gray, colony_mask)
                hyph_f, hyph_m, hyph_h = detect_hyphae(gray, colony_mask)

                metrics = compute_metrics(
                    colony_mask, gray, px2mm, dcx, dcy, crack_mask, hyph_f, hyph_m, hyph_h
                )
                metrics.update(
                    colony_pixels=int(colony_mask.sum()),
                    dish_detected=dish_info is not None,
                    dish_radius_px=dr,
                    px_to_mm=round(px2mm, 6),
                    calibration_diameter_mm=round(2 * dr * px2mm, 4),
                    calibration_error_pct=(
                        round(abs(2 * dr * px2mm - CONTAINER_MM) / CONTAINER_MM * 100, 4)
                        if dish_info else 0
                    ),
                    image_path=orig,
                    experiment_name=en or "",
                    experiment_date=ed or "",
                    image_date=imd,
                    day_code=day_code(imd, ed) if ed and imd else "",
                    user_name=un or "",
                    plates_count=int(pc) if pc else 1,
                )
                results.append(metrics)

                panels = create_full_overlays(
                    img_bgr, colony_mask, crack_mask, hyph_h, dish_info, orig
                )
                vis.extend(panels)

            except Exception as e:
                errors.append(f"{orig}: {e}")
                results.append({"image_path": orig, "error": str(e)})

        from .analysis import _dc_to_num

        ok_results = [r for r in results if not r.get("error")]
        if len(ok_results) > 1:
            # Sort by day_code numeric value first; fall back to image_date string sort
            ok_results.sort(key=lambda r: (
                _dc_to_num(r.get("day_code", ""), fallback=9999),
                r.get("image_date", ""),
            ))
            # days_since_start: prefer numeric value from day_code (set in Steps 3/4);
            # fall back to date arithmetic so growth rate math always has a value.
            try:
                base = dt.date.fromisoformat(ok_results[0].get("image_date", ""))
            except Exception:
                base = None
            for i, r in enumerate(ok_results):
                dc_num = _dc_to_num(r.get("day_code", ""), fallback=0)
                if dc_num:
                    r["days_since_start"] = dc_num
                else:
                    try:
                        r["days_since_start"] = (
                            (dt.date.fromisoformat(r.get("image_date", "")) - base).days + 1
                            if base else i
                        )
                    except Exception:
                        r["days_since_start"] = i
                if i == 0:
                    r["rgr_per_day"] = ""
                    r["relative_growth_per_day"] = ""
                    continue
                prev = ok_results[i - 1]
                try:
                    import math
                    dd = (
                        dt.date.fromisoformat(r["image_date"])
                        - dt.date.fromisoformat(prev["image_date"])
                    ).days
                    a2, a1 = float(r.get("area_mm2", 0)), float(prev.get("area_mm2", 0))
                    if dd > 0 and a1 > 0 and a2 > 0:
                        r["rgr_per_day"] = round((math.log(a2) - math.log(a1)) / dd, 6)
                        r["relative_growth_per_day"] = round((a2 - a1) / dd, 4)
                    else:
                        r["rgr_per_day"] = ""
                        r["relative_growth_per_day"] = ""
                except Exception:
                    r["rgr_per_day"] = ""
                    r["relative_growth_per_day"] = ""

        chart_items = make_growth_charts(ok_results) if len(ok_results) >= 2 else []

        tmp = tempfile.mkdtemp()
        all_results = ok_results + [r for r in results if r.get("error")]
        cp = Path(tmp) / "analysis_full.csv"
        if all_results:
            ks = list(all_results[0].keys())
            with open(cp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=ks, extrasaction="ignore")
                w.writeheader()
                w.writerows(all_results)
        jp = Path(tmp) / "analysis_full.json"
        with open(jp, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        for i, (cimg, cap) in enumerate(chart_items):
            cimg.save(str(Path(tmp) / f"chart_{i}.png"))

        # Build image_metadata and include it in the same zip
        meta_rows: list[dict] = []
        for p in paths:
            imd = dates.get(p, detect_image_date(p))
            orig = orig_names.get(p, Path(p).name)
            rm = (rems or {}).get(p, "")
            meta_rows.append(dict(
                image_path=orig,
                experiment_name=en or "",
                experiment_date=ed or "",
                image_date=imd,
                day_code=day_code(imd, ed) if ed and imd else "",
                user_name=un or "",
                plates_count=int(pc) if pc else 1,
                remind_me=rm,
            ))
        if meta_rows:
            mcp = Path(tmp) / "image_metadata.csv"
            with open(mcp, "w", newline="") as f:
                w2 = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
                w2.writeheader()
                w2.writerows(meta_rows)
            mjp = Path(tmp) / "image_metadata.json"
            with open(mjp, "w") as f:
                json.dump(meta_rows, f, indent=2)
            rl2 = [r for r in meta_rows if r.get("remind_me", "").strip()]
            if rl2:
                icp = Path(tmp) / "reminders.ics"
                write_ics(rl2, str(icp))

        stem = _zip_stem(un or "", en or "")
        zp = Path(tmp) / f"{stem}.zip"
        with zipfile.ZipFile(zp, "w") as z:
            for fp in Path(tmp).glob("*"):
                if fp.name != f"{stem}.zip":
                    z.write(fp, fp.name)

        em = f"\n\n⚠️ Errors: {'; '.join(errors)}" if errors else ""
        cm = f"\n\n📊 **{len(chart_items)} charts**" if chart_items else ""
        return (
            f"✅ **{len(ok_results)}/{len(results)}** analysed.{cm}{em}",
            vis, chart_items, pd.DataFrame(all_results), str(zp), all_results,
        )

    run_btn.click(
        on_run,
        [
            paths_st, dates_st, rems_st, orig_names_st,
            exp_name, exp_date, user_name, plates_count,
            threshold_slider, full_pipeline_cb,
        ],
        [run_st, overlay_gallery, chart_gallery, results_df, results_dl, results_st],
    )

    # ── close-server handlers ───────────────────────────────────────────────

    close_btn.click(
        lambda: gr.update(visible=True),
        outputs=[close_panel],
    )

    cancel_close_btn.click(
        lambda: (gr.update(visible=False), gr.update(value="")),
        outputs=[close_panel, close_status],
    )

    def on_confirm_close():
        def _kill():
            _time.sleep(1.2)          # allow response to reach the browser
            os.kill(os.getpid(), signal.SIGTERM)
        threading.Thread(target=_kill, daemon=True).start()
        return gr.update(value="🔴 Shutting down…")

    confirm_close_btn.click(
        on_confirm_close,
        outputs=[close_status],
    )


def main() -> None:
    demo.launch(server_name="0.0.0.0", server_port=7860, css=CSS)


if __name__ == "__main__":
    main()
