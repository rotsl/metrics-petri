"""Gradio GUI for metrics-petri colony segmentation and analysis.

Run via:
    metrics-petri-gui
    python -m pipeline
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import os
import re
import tempfile
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
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
THUMB_SIZE = (160, 160)
DATE_RE = re.compile(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])")
MAX_IMAGES = 50
CSS = ".gallery-wrap{max-height:65vh;overflow-y:auto} .footer-text{text-align:center;margin-top:8px}"


# ── helpers ────────────────────────────────────────────────────────────────

def make_thumbnail(p: str) -> Image.Image:
    try:
        im = Image.open(p)
        im.thumbnail(THUMB_SIZE, Image.LANCZOS)
        return im
    except Exception:
        return Image.new("RGB", THUMB_SIZE, (200, 200, 200))


def detect_image_date(p: str) -> str:
    m = DATE_RE.search(Path(p).stem)
    if m:
        try:
            return dt.date(int(m[1]), int(m[2]), int(m[3])).isoformat()
        except Exception:
            pass
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
    try:
        return dt.date.fromtimestamp(os.path.getmtime(p)).isoformat()
    except Exception:
        return dt.date.today().isoformat()


def day_code(img_d: str, exp_d: str) -> str:
    try:
        d = (dt.date.fromisoformat(img_d) - dt.date.fromisoformat(exp_d)).days + 1
        return f"d{max(d, 1):02d}"
    except Exception:
        return "d??"


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

with gr.Blocks(title="metrics-petri Colony Segmentation", css=CSS) as demo:
    paths_st = gr.State([])
    dates_st = gr.State({})
    rems_st = gr.State({})
    cur_idx = gr.State(-1)
    results_st = gr.State([])

    gr.Markdown(
        "# 🔬 metrics-petri — Colony Segmentation\n"
        "Upload → **Run Inference** → instant results  |  Toggle *Full Pipeline* for morphometrics\n\n"
        "Model: `models/best_area_w_0.7.pt` · SmallUNet (area-consistency w=0.7)"
    )

    with gr.Accordion("📂 Step 1 — Upload Images", open=True):
        upload = gr.File(
            label="Drag & drop petri dish images",
            file_count="multiple",
            file_types=["image"],
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
            exp_date = gr.Textbox(label="Experiment Date", placeholder="2025-04-01")
            user_name = gr.Textbox(label="User Name", placeholder="Your name")
            plates_count = gr.Number(
                label="Plates", value=1, minimum=1, maximum=200, precision=0
            )

    with gr.Accordion("🖼️ Step 3 — Review & Edit Dates", open=False):
        gr.Markdown("*Click thumbnail → edit date → Save*")
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
                sel_dt = gr.Textbox(label="Image Date", interactive=True)
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

    gr.Markdown(
        "<div class='footer-text'>\n\n---\n"
        "Developed at TSL, Norwich · Apache 2.0\n</div>"
    )

    # ── event handlers ─────────────────────────────────────────────────────

    def on_upload(files):
        if not files:
            return [], {}, {}, [], "", -1
        paths = [
            str(f) for f in files if Path(str(f)).suffix.lower() in IMAGE_EXTS
        ][:MAX_IMAGES]
        if not paths:
            return [], {}, {}, [], "", -1
        dates = {p: detect_image_date(p) for p in paths}
        rems = {p: "" for p in paths}
        return (
            paths,
            dates,
            rems,
            [(p, Path(p).name) for p in paths],
            f"✅ **{len(paths)}** images loaded.",
            -1,
        )

    upload.upload(
        on_upload,
        [upload],
        [paths_st, dates_st, rems_st, gallery, up_st, cur_idx],
    )

    def on_sel(paths, dates, rems, ed, evt: gr.SelectData):
        i = evt.index
        if i < 0 or i >= len(paths):
            return -1, None, "", "", "", ""
        p = paths[i]
        return (
            i,
            make_thumbnail(p),
            Path(p).name,
            dates.get(p, ""),
            day_code(dates.get(p, ""), ed) if ed else "",
            rems.get(p, ""),
        )

    gallery.select(
        on_sel,
        [paths_st, dates_st, rems_st, exp_date],
        [cur_idx, sel_img, sel_fn, sel_dt, sel_dc, sel_rm],
    )

    def on_save(paths, dates, rems, i, nd, nr, ed):
        if i < 0 or i >= len(paths):
            return dates, rems, "", "⚠️ Select an image first."
        p = paths[i]
        dates = dict(dates)
        rems = dict(rems)
        dates[p] = nd
        rems[p] = nr
        return dates, rems, day_code(nd, ed) if ed else "", f"✅ **{Path(p).name}** → {nd}"

    sv_btn.click(
        on_save,
        [paths_st, dates_st, rems_st, cur_idx, sel_dt, sel_rm, exp_date],
        [dates_st, rems_st, sel_dc, sv_st],
    )

    def on_export(paths, dates, rems, en, ed, un, pc):
        if not paths:
            return "⚠️ Upload images first.", None, None
        tmp = tempfile.mkdtemp()
        rows: list[dict] = []
        rl: list[dict] = []
        for p in paths:
            imd = dates.get(p, detect_image_date(p))
            rm = rems.get(p, "")
            row = dict(
                image_path=Path(p).name,
                experiment_name=en or "",
                experiment_date=ed or "",
                image_date=imd,
                day_code=day_code(imd, ed) if ed else "",
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
        [paths_st, dates_st, rems_st, exp_name, exp_date, user_name, plates_count],
        [exp_st, meta_preview, meta_dl],
    )

    def on_run(paths, dates, en, ed, un, pc, thresh, full_pipeline, progress=gr.Progress()):
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
                try:
                    img = Image.open(p).convert("RGB")
                    overlay, mask = infer_mask(img, thresh)
                    mask_px = np.sum(np.array(mask) > 0)
                    vis.append((img, f"{Path(p).name} — Raw"))
                    vis.append((mask, f"{Path(p).name} — Mask"))
                    vis.append((overlay, f"{Path(p).name} — Overlay"))
                except Exception as e:
                    errors.append(f"{Path(p).name}: {e}")
            em = f"\n\n⚠️ Errors: {'; '.join(errors)}" if errors else ""
            ok = len(paths) - len(errors)
            return (
                f"✅ **{ok}/{len(paths)}** segmented (fast mode, threshold={thresh:.2f}).{em}",
                vis,
                [],
                None,
                None,
                [],
            )

        # ── full pipeline ────────────────────────────────────────────────
        for p in progress.tqdm(paths, desc="Full pipeline"):
            imd = dates.get(p, detect_image_date(p))
            try:
                img_bgr = cv2.imread(str(p))
                if img_bgr is None:
                    raise RuntimeError(f"Cannot read: {p}")

                # Segment
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

                # Dish detection
                dish_info = detect_container(img_bgr)
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0

                if dish_info:
                    dcx, dcy, dr, px2mm = dish_info
                else:
                    dcx, dcy = w // 2, h // 2
                    dr = min(h, w) // 2
                    px2mm = 1.0

                # Crack & hyphae detection
                crack_mask = detect_cracks(gray, colony_mask)
                hyph_f, hyph_m, hyph_h = detect_hyphae(gray, colony_mask)

                # Compute all metrics
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
                        if dish_info
                        else 0
                    ),
                    image_path=Path(p).name,
                    experiment_name=en or "",
                    experiment_date=ed or "",
                    image_date=imd,
                    day_code=day_code(imd, ed) if ed else "",
                    user_name=un or "",
                    plates_count=int(pc) if pc else 1,
                )
                results.append(metrics)

                panels = create_full_overlays(
                    img_bgr, colony_mask, crack_mask, hyph_h, dish_info, Path(p).name
                )
                vis.extend(panels)

            except Exception as e:
                errors.append(f"{Path(p).name}: {e}")
                results.append({"image_path": Path(p).name, "error": str(e)})

        # Growth rates
        ok_results = [r for r in results if not r.get("error")]
        if len(ok_results) > 1:
            ok_results.sort(key=lambda r: r.get("image_date", ""))
            try:
                base = dt.date.fromisoformat(ok_results[0].get("image_date", ""))
            except Exception:
                base = None
            for i, r in enumerate(ok_results):
                try:
                    r["days_since_start"] = (
                        (dt.date.fromisoformat(r.get("image_date", "")) - base).days if base else 0
                    )
                except Exception:
                    r["days_since_start"] = 0
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

        # Write zip
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
        zp = Path(tmp) / "analysis_full.zip"
        with zipfile.ZipFile(zp, "w") as z:
            for fp in Path(tmp).glob("*"):
                if fp.name != "analysis_full.zip":
                    z.write(fp, fp.name)

        em = f"\n\n⚠️ Errors: {'; '.join(errors)}" if errors else ""
        cm = f"\n\n📊 **{len(chart_items)} charts**" if chart_items else ""
        return (
            f"✅ **{len(ok_results)}/{len(results)}** analysed.{cm}{em}",
            vis,
            chart_items,
            pd.DataFrame(all_results),
            str(zp),
            all_results,
        )

    run_btn.click(
        on_run,
        [
            paths_st,
            dates_st,
            exp_name,
            exp_date,
            user_name,
            plates_count,
            threshold_slider,
            full_pipeline_cb,
        ],
        [run_st, overlay_gallery, chart_gallery, results_df, results_dl, results_st],
    )


def main() -> None:
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
