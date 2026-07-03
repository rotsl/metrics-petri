#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
image_metadata_gui.py — Desktop GUI for metrics-petri metadata management.

Replicates Steps 1–4 of the Gradio Space 
but runs entirely offline as a native tkinter application.

Step 1: Choose a folder containing petri-dish images.
Step 2: Enter experiment metadata (name, date, user, plates).
Step 3: Review thumbnails, auto-detected dates, edit dates & reminders.
Step 4: Export image_metadata.csv / .json / .ics into the SAME folder.


Usage:
    pip install Pillow
    python image_metadata_gui.py
"""

import csv
import datetime as dt
import json
import os
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk, ExifTags

# ── Constants ──────────────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
THUMB_SIZE = (120, 120)
DATE_RE = re.compile(r"(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])")
MAX_IMAGES = 200  # generous limit for desktop

# ── Helpers (same logic as the Gradio Space) ───────────────────────────
def detect_image_date(p: str) -> str:
    """Try filename regex → EXIF DateTimeOriginal → file mtime."""
    m = DATE_RE.search(Path(p).stem)
    if m:
        try:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
        except ValueError:
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


def build_day_code_map(dates_dict: dict[str, str], exp_date: str) -> dict[str, str]:
    """Build a mapping from date-string → day-code.

    Day code = calendar days elapsed since the experiment date:
      - exp_date itself → d00  (day 0 = inoculation / experiment setup)
      - exp_date + 1 day → d01, + 4 days → d04, etc.
      - dates before exp_date are clamped to d00

    When no experiment date is provided, codes are assigned sequentially
    (d01, d02, …) from the earliest observed image date.

    Parameters
    ----------
    dates_dict : {image_path: date_iso_str, ...}
    exp_date   : experiment start date (YYYY-MM-DD) from Step 2

    Returns
    -------
    {date_iso_str: "dNN", ...}
    """
    unique_dates: set[dt.date] = set()
    for d_str in dates_dict.values():
        try:
            unique_dates.add(dt.date.fromisoformat(d_str))
        except (ValueError, TypeError):
            pass

    try:
        exp_d = dt.date.fromisoformat(exp_date)
    except (ValueError, TypeError):
        exp_d = None

    if not unique_dates:
        return {}

    if exp_d is not None:
        code_map: dict[str, str] = {}
        for d in unique_dates:
            delta = max(0, (d - exp_d).days)
            code_map[d.isoformat()] = f"d{delta:02d}"
        # Ensure exp_date is always present, even if no image falls on that date
        if exp_d.isoformat() not in code_map:
            code_map[exp_d.isoformat()] = "d00"
        return code_map
    else:
        # No experiment date — sequential from 1
        sorted_dates = sorted(unique_dates)
        return {d.isoformat(): f"d{i:02d}" for i, d in enumerate(sorted_dates, start=1)}


def day_code_lookup(img_date: str, code_map: dict[str, str]) -> str:
    """Look up a single image date in a pre-built code map."""
    return code_map.get(img_date, "d??")


def write_ics(reminders: list, path: str) -> None:
    """Write a minimal iCalendar file for reminders."""
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//FungalPipeline//EN"]
    for r in reminders:
        uid = r["image_path"].replace("/", "_").replace("\\", "_")
        raw = r["remind_me"].strip().replace("-", "").replace(" ", "T").replace(":", "")
        if len(raw) < 15:
            raw += "00"  # pad seconds if absent
        dtstart = raw[:15]
        lines += [
            "BEGIN:VEVENT",
            f"UID:{uid}@fp",
            f"DTSTART:{dtstart}",
            f"SUMMARY:Reminder - {r['experiment_name']}: {Path(r['image_path']).name}",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    with open(path, "w") as f:
        f.write("\r\n".join(lines))


# ══════════════════════════════════════════════════════════════════════
# Main Application
# ══════════════════════════════════════════════════════════════════════
class MetadataApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🔬 Gray Leaf Spot — Image Metadata Manager")
        self.geometry("1100x780")
        self.minsize(900, 650)

        # ── State ──
        self.image_folder: str = ""
        self.image_paths: list[str] = []
        self.dates: dict[str, str] = {}      # path → date
        self.reminders: dict[str, str] = {}   # path → reminder datetime
        self.thumb_cache: dict[str, ImageTk.PhotoImage] = {}
        self.selected_index: int = -1

        # ── Style ──
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Sub.TLabel", font=("Segoe UI", 10))

        self._build_ui()

    # ──────────────────────────────────────────────────────────────────
    # Day-code map (rebuilt whenever dates or experiment date change)
    # ──────────────────────────────────────────────────────────────────
    def _code_map(self) -> dict[str, str]:
        """Return the current date → day-code mapping."""
        exp_d = self.exp_date_var.get().strip()
        return build_day_code_map(self.dates, exp_d)

    # ──────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Main notebook (tabs for each step)
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self._build_step1()
        self._build_step2()
        self._build_step3()
        self._build_step4()

    # ── Step 1: Upload / Select Folder ────────────────────────────────
    def _build_step1(self):
        frame = ttk.Frame(self.nb, padding=16)
        self.nb.add(frame, text="  📂 Step 1 — Select Folder  ")

        ttk.Label(frame, text="Step 1 — Select Image Folder", style="Header.TLabel").pack(anchor="w")
        ttk.Label(frame, text="Choose the folder that contains your petri-dish images.",
                  style="Sub.TLabel").pack(anchor="w", pady=(2, 12))

        row = ttk.Frame(frame)
        row.pack(fill=tk.X)
        self.folder_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.folder_var, state="readonly", width=80).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse …", command=self._browse_folder).pack(side=tk.LEFT, padx=(8, 0))

        self.step1_status = ttk.Label(frame, text="", style="Sub.TLabel")
        self.step1_status.pack(anchor="w", pady=(12, 0))

        # Thumbnail preview
        self.preview_frame = ttk.LabelFrame(frame, text="Preview (first 20 images)", padding=8)
        self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        self.preview_canvas = tk.Canvas(self.preview_frame, bg="#f5f5f5", highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="Select image folder")
        if not folder:
            return
        self.image_folder = folder
        self.folder_var.set(folder)
        self._scan_folder()

    def _scan_folder(self):
        folder = Path(self.image_folder)
        paths = sorted(
            [str(p) for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
            key=lambda x: Path(x).name.lower(),
        )[:MAX_IMAGES]

        self.image_paths = paths
        self.dates = {p: detect_image_date(p) for p in paths}
        self.reminders = {p: "" for p in paths}
        self.thumb_cache.clear()
        self.selected_index = -1

        if not paths:
            self.step1_status.config(text="⚠️  No images found in the selected folder.")
            return

        self.step1_status.config(text=f"✅  {len(paths)} images loaded from {self.image_folder}")
        self._render_preview()
        self._populate_step3_tree()

    def _make_thumbnail(self, path: str) -> ImageTk.PhotoImage:
        if path in self.thumb_cache:
            return self.thumb_cache[path]
        try:
            im = Image.open(path)
            im.thumbnail(THUMB_SIZE, Image.LANCZOS)
        except Exception:
            im = Image.new("RGB", THUMB_SIZE, (200, 200, 200))
        photo = ImageTk.PhotoImage(im)
        self.thumb_cache[path] = photo
        return photo

    def _render_preview(self):
        self.preview_canvas.delete("all")
        x, y = 10, 10
        max_w = self.preview_canvas.winfo_width() or 800
        for i, p in enumerate(self.image_paths[:20]):
            thumb = self._make_thumbnail(p)
            self.preview_canvas.create_image(x, y, anchor="nw", image=thumb)
            self.preview_canvas.create_text(
                x + THUMB_SIZE[0] // 2, y + THUMB_SIZE[1] + 4,
                text=Path(p).name[:18], font=("Segoe UI", 7), anchor="n",
            )
            x += THUMB_SIZE[0] + 16
            if x + THUMB_SIZE[0] > max_w:
                x = 10
                y += THUMB_SIZE[1] + 28

    # ── Step 2: Settings ──────────────────────────────────────────────
    def _build_step2(self):
        frame = ttk.Frame(self.nb, padding=16)
        self.nb.add(frame, text="  ⚙️ Step 2 — Settings  ")

        ttk.Label(frame, text="Step 2 — Experiment Settings", style="Header.TLabel").pack(anchor="w")
        ttk.Label(frame, text="Enter the metadata that applies to the whole experiment.",
                  style="Sub.TLabel").pack(anchor="w", pady=(2, 16))

        form = ttk.Frame(frame)
        form.pack(fill=tk.X)

        self.exp_name_var = tk.StringVar()
        self.exp_date_var = tk.StringVar()
        self.user_name_var = tk.StringVar()
        self.plates_var = tk.IntVar(value=1)

        fields = [
            ("Experiment Name", self.exp_name_var, "MagExp01"),
            ("Experiment Date (YYYY-MM-DD)", self.exp_date_var, "2025-04-01"),
            ("User Name", self.user_name_var, "Your name"),
        ]

        for i, (label, var, placeholder) in enumerate(fields):
            ttk.Label(form, text=label, font=("Segoe UI", 10)).grid(
                row=i, column=0, sticky="w", padx=(0, 12), pady=6,
            )
            entry = ttk.Entry(form, textvariable=var, width=40)
            entry.grid(row=i, column=1, sticky="w", pady=6)
            entry.insert(0, "")
            _add_placeholder(entry, placeholder)

        ttk.Label(form, text="Plates", font=("Segoe UI", 10)).grid(
            row=len(fields), column=0, sticky="w", padx=(0, 12), pady=6,
        )
        plates_spin = ttk.Spinbox(form, from_=1, to=200, textvariable=self.plates_var, width=8)
        plates_spin.grid(row=len(fields), column=1, sticky="w", pady=6)

        self.step2_hint = ttk.Label(
            frame,
            text="💡 Tip: The Experiment Date is used to compute day-codes (d01, d02 …) for each image.",
            style="Sub.TLabel",
        )
        self.step2_hint.pack(anchor="w", pady=(20, 0))

    # ── Step 3: Review & Edit Dates ───────────────────────────────────
    def _build_step3(self):
        frame = ttk.Frame(self.nb, padding=16)
        self.nb.add(frame, text="  🖼️ Step 3 — Review & Edit Dates  ")

        ttk.Label(frame, text="Step 3 — Review & Edit Dates", style="Header.TLabel").pack(anchor="w")
        ttk.Label(frame, text="Click a row to select it, then edit the date / reminder below.",
                  style="Sub.TLabel").pack(anchor="w", pady=(2, 12))

        # ── Treeview (image list) ──
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        cols = ("filename", "image_date", "day_code", "remind_me")
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=14)
        self.tree.heading("filename", text="Filename")
        self.tree.heading("image_date", text="Image Date")
        self.tree.heading("day_code", text="Day Code")
        self.tree.heading("remind_me", text="Remind Me")
        self.tree.column("filename", width=300)
        self.tree.column("image_date", width=120)
        self.tree.column("day_code", width=80)
        self.tree.column("remind_me", width=180)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # ── Edit panel ──
        edit_frame = ttk.LabelFrame(frame, text="Edit Selected Image", padding=10)
        edit_frame.pack(fill=tk.X, pady=(12, 0))

        row1 = ttk.Frame(edit_frame)
        row1.pack(fill=tk.X, pady=2)

        # Thumbnail on the left
        self.sel_thumb_label = ttk.Label(row1)
        self.sel_thumb_label.pack(side=tk.LEFT, padx=(0, 16))

        fields_frame = ttk.Frame(row1)
        fields_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(fields_frame, text="Filename:").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=2)
        self.sel_fn_var = tk.StringVar()
        ttk.Entry(fields_frame, textvariable=self.sel_fn_var, state="readonly", width=40).grid(
            row=0, column=1, sticky="w", pady=2,
        )

        ttk.Label(fields_frame, text="Image Date:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=2)
        self.sel_date_var = tk.StringVar()
        ttk.Entry(fields_frame, textvariable=self.sel_date_var, width=20).grid(
            row=1, column=1, sticky="w", pady=2,
        )

        ttk.Label(fields_frame, text="Day Code:").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=2)
        self.sel_dc_var = tk.StringVar()
        ttk.Entry(fields_frame, textvariable=self.sel_dc_var, state="readonly", width=10).grid(
            row=2, column=1, sticky="w", pady=2,
        )

        ttk.Label(fields_frame, text="Remind Me:").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=2)
        self.sel_rm_var = tk.StringVar()
        ttk.Entry(fields_frame, textvariable=self.sel_rm_var, width=24).grid(
            row=3, column=1, sticky="w", pady=2,
        )
        ttk.Label(fields_frame, text="(YYYY-MM-DD HH:MM)", font=("Segoe UI", 8)).grid(
            row=3, column=2, sticky="w", padx=(4, 0), pady=2,
        )

        btn_row = ttk.Frame(edit_frame)
        btn_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(btn_row, text="💾 Save Changes", command=self._save_date, style="Accent.TButton").pack(side=tk.LEFT)
        self.step3_status = ttk.Label(btn_row, text="", style="Sub.TLabel")
        self.step3_status.pack(side=tk.LEFT, padx=(16, 0))

    def _populate_step3_tree(self):
        self.tree.delete(*self.tree.get_children())
        cm = self._code_map()
        for p in self.image_paths:
            imd = self.dates.get(p, "")
            dc = day_code_lookup(imd, cm)
            rm = self.reminders.get(p, "")
            self.tree.insert("", "end", iid=p, values=(Path(p).name, imd, dc, rm))

    def _on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        p = sel[0]
        try:
            idx = self.image_paths.index(p)
        except ValueError:
            return
        self.selected_index = idx

        self.sel_fn_var.set(Path(p).name)
        self.sel_date_var.set(self.dates.get(p, ""))
        cm = self._code_map()
        self.sel_dc_var.set(day_code_lookup(self.dates.get(p, ""), cm))
        self.sel_rm_var.set(self.reminders.get(p, ""))

        # Show thumbnail
        thumb = self._make_thumbnail(p)
        self.sel_thumb_label.config(image=thumb)
        self.sel_thumb_label.image = thumb  # prevent GC

        self.step3_status.config(text="")

    def _save_date(self):
        if self.selected_index < 0 or self.selected_index >= len(self.image_paths):
            self.step3_status.config(text="⚠️ Select an image first.")
            return
        p = self.image_paths[self.selected_index]
        new_date = self.sel_date_var.get().strip()
        new_rm = self.sel_rm_var.get().strip()

        self.dates[p] = new_date
        self.reminders[p] = new_rm

        # Rebuild the full code map (ordinals may shift after a date edit)
        cm = self._code_map()
        dc = day_code_lookup(new_date, cm)
        self.sel_dc_var.set(dc)

        # Update tree row
        self.tree.item(p, values=(Path(p).name, new_date, dc, new_rm))

        # Refresh ALL rows because ordinals may have shifted
        for other_p in self.image_paths:
            if other_p == p:
                continue
            other_imd = self.dates.get(other_p, "")
            other_dc = day_code_lookup(other_imd, cm)
            other_rm = self.reminders.get(other_p, "")
            self.tree.item(other_p, values=(Path(other_p).name, other_imd, other_dc, other_rm))

        self.step3_status.config(text=f"✅ Saved: {Path(p).name} → {new_date}")

    # ── Step 4: Export ────────────────────────────────────────────────
    def _build_step4(self):
        frame = ttk.Frame(self.nb, padding=16)
        self.nb.add(frame, text="  📥 Step 4 — Export  ")

        ttk.Label(frame, text="Step 4 — Export Metadata", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            frame,
            text="Export image_metadata.csv, .json, and .ics into the SAME folder as the images.",
            style="Sub.TLabel",
        ).pack(anchor="w", pady=(2, 16))

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="📥 Export CSV", command=lambda: self._export("csv"),
                   style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="📥 Export JSON", command=lambda: self._export("json"),
                   style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="📥 Export ICS", command=lambda: self._export("ics"),
                   style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="📥 Export All (CSV + JSON + ICS)", command=lambda: self._export("all"),
                   style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 8))

        self.step4_status = ttk.Label(frame, text="", style="Sub.TLabel")
        self.step4_status.pack(anchor="w", pady=(12, 0))

        # Preview table
        preview_lf = ttk.LabelFrame(frame, text="Preview", padding=8)
        preview_lf.pack(fill=tk.BOTH, expand=True, pady=(12, 0))

        cols = ("image_path", "experiment_name", "experiment_date", "image_date",
                "day_code", "user_name", "plates_count", "remind_me")
        self.export_tree = ttk.Treeview(preview_lf, columns=cols, show="headings", height=12)
        for c in cols:
            self.export_tree.heading(c, text=c)
            self.export_tree.column(c, width=120)
        self.export_tree.column("image_path", width=200)

        vsb2 = ttk.Scrollbar(preview_lf, orient="vertical", command=self.export_tree.yview)
        hsb2 = ttk.Scrollbar(preview_lf, orient="horizontal", command=self.export_tree.xview)
        self.export_tree.configure(yscrollcommand=vsb2.set, xscrollcommand=hsb2.set)

        self.export_tree.grid(row=0, column=0, sticky="nsew")
        vsb2.grid(row=0, column=1, sticky="ns")
        hsb2.grid(row=1, column=0, sticky="ew")
        preview_lf.rowconfigure(0, weight=1)
        preview_lf.columnconfigure(0, weight=1)

        # Refresh preview button
        ttk.Button(frame, text="🔄 Refresh Preview", command=self._refresh_export_preview).pack(
            anchor="w", pady=(8, 0),
        )

    def _build_rows(self) -> list[dict]:
        """Build the metadata rows from current state."""
        en = self.exp_name_var.get().strip()
        ed = self.exp_date_var.get().strip()
        un = self.user_name_var.get().strip()
        pc = self.plates_var.get()
        cm = self._code_map()
        rows = []
        for p in self.image_paths:
            imd = self.dates.get(p, detect_image_date(p))
            rm = self.reminders.get(p, "")
            rows.append({
                "image_path": Path(p).name,
                "experiment_name": en,
                "experiment_date": ed,
                "image_date": imd,
                "day_code": day_code_lookup(imd, cm),
                "user_name": un,
                "plates_count": int(pc) if pc else 1,
                "remind_me": rm,
            })
        return rows

    def _refresh_export_preview(self):
        self.export_tree.delete(*self.export_tree.get_children())
        if not self.image_paths:
            self.step4_status.config(text="⚠️ No images loaded. Go to Step 1 first.")
            return
        rows = self._build_rows()
        for r in rows:
            self.export_tree.insert("", "end", values=tuple(r.values()))
        self.step4_status.config(text=f"Preview: {len(rows)} rows")

    def _export(self, fmt: str):
        if not self.image_paths:
            messagebox.showwarning("No images", "Load images in Step 1 first.")
            return
        if not self.image_folder:
            messagebox.showwarning("No folder", "No image folder selected.")
            return

        rows = self._build_rows()
        out_dir = Path(self.image_folder)
        exported = []

        if fmt in ("csv", "all"):
            csv_path = out_dir / "image_metadata.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            exported.append(str(csv_path))

        if fmt in ("json", "all"):
            json_path = out_dir / "image_metadata.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, ensure_ascii=False)
            exported.append(str(json_path))

        if fmt in ("ics", "all"):
            en = self.exp_name_var.get().strip()
            reminders_list = [
                {**r, "experiment_name": en}
                for r in rows if r["remind_me"].strip()
            ]
            if reminders_list:
                ics_path = out_dir / "reminders.ics"
                write_ics(reminders_list, str(ics_path))
                exported.append(str(ics_path))
            elif fmt == "ics":
                messagebox.showinfo("No reminders",
                                    "No reminders set. Set 'Remind Me' dates in Step 3 first.")
                return

        if exported:
            msg = f"✅ Exported {len(rows)} images:\n\n" + "\n".join(exported)
            self.step4_status.config(text=msg.replace("\n", " | "))
            messagebox.showinfo("Export complete", msg)
        else:
            self.step4_status.config(text="Nothing exported.")


# ── Placeholder helper for ttk.Entry ──────────────────────────────────
def _add_placeholder(entry: ttk.Entry, placeholder: str):
    """Add grey placeholder text that disappears on focus."""
    entry._placeholder = placeholder
    entry._is_placeholder = True

    def _on_focus_in(e):
        if entry._is_placeholder:
            entry.delete(0, tk.END)
            entry.configure(foreground="black")
            entry._is_placeholder = False

    def _on_focus_out(e):
        if not entry.get():
            entry.insert(0, placeholder)
            entry.configure(foreground="grey")
            entry._is_placeholder = True

    entry.insert(0, placeholder)
    entry.configure(foreground="grey")
    entry.bind("<FocusIn>", _on_focus_in)
    entry.bind("<FocusOut>", _on_focus_out)

    # Override .get() to return "" when placeholder is showing
    original_get = entry.get

    def patched_get():
        val = original_get()
        if entry._is_placeholder:
            return ""
        return val

    entry.get = patched_get


# ══════════════════════════════════════════════════════════════════════
def main():
    app = MetadataApp()
    app.mainloop()


if __name__ == "__main__":
    main()