#!/usr/bin/env python3
"""
input.py - Sample image metadata UI

Streamlit replacement for the old Tk app.

What it does:
- Loads a folder path and recursively finds all images, including nested folders
- Shows one image at a time with preview
- Lets user enter metadata per image:
    - experiment_name
    - experiment_date (YYYY-MM-DD)
    - image_date (auto-detected from filename / EXIF / mtime)
    - day_code (auto-calculated as d01, d02...)
    - user_name
    - plates_count
    - optional remind_me (YYYY-MM-DD or YYYY-MM-DD HH:MM)
- Features:
    - progress bar + counter
    - save per image or discard
    - back button to revisit previous image
    - optional autofill from previous values
    - bulk apply metadata to all images in the current folder only
- Outputs written into the selected folder:
    - image_metadata.csv
    - image_metadata.json
    - reminders.ics (if remind_me values are present)

Run with:
    streamlit run pipelinesam/input.py
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image, ImageOps

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover - user-facing runtime guard
    raise SystemExit(
        "Streamlit is not installed. Activate the venv and run:\n"
        "  pip install streamlit\n"
        "then launch with:\n"
        "  streamlit run pipelinesam/input.py"
    ) from exc


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
DEFAULT_INPUT_IMAGES_DIR = (Path(__file__).resolve().parent.parent / "input_images").resolve()

FIELDNAMES = [
    "image_path",
    "experiment_name",
    "experiment_date",
    "image_date",
    "day_code",
    "user_name",
    "plates_count",
    "remind_me",
    "image",
    "experiment",
    "date",
    "day",
    "user",
    "plates",
    "remind",
]


# ---------- utilities ----------

def get_image_date(path: Path) -> datetime:
    m = re.search(r"(\d{8})", path.stem)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d")
        except ValueError:
            pass
    try:
        exif = Image.open(path)._getexif()
        if exif and 36867 in exif:
            return datetime.strptime(exif[36867], "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def calc_day_code(exp: str, img: datetime) -> str:
    try:
        e = datetime.strptime(exp, "%Y-%m-%d")
        return f"d{max(1, (img.date() - e.date()).days + 1):02d}"
    except Exception:
        return ""


def parse_remind(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except Exception:
            pass
    return None


def discover_images(root_dir: Path) -> list[Path]:
    return sorted(p for p in root_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def to_iso_date(value: datetime | None) -> str:
    return value.strftime("%Y-%m-%d") if value else ""


def preview_image(path: Path, size: tuple[int, int] = (420, 420)) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    image.thumbnail(size)
    return image


def build_record(
    path: Path,
    root_dir: Path,
    experiment_name: str,
    experiment_date: str,
    image_date: datetime,
    day_code: str,
    user_name: str,
    plates_count: str,
    remind_me: str,
) -> dict[str, str]:
    rel = path.relative_to(root_dir).as_posix()
    image_date_str = to_iso_date(image_date)
    return {
        "image_path": rel,
        "experiment_name": experiment_name.strip(),
        "experiment_date": experiment_date.strip(),
        "image_date": image_date_str,
        "day_code": day_code.strip(),
        "user_name": user_name.strip(),
        "plates_count": plates_count.strip(),
        "remind_me": remind_me.strip(),
        "image": rel,
        "experiment": experiment_name.strip(),
        "date": experiment_date.strip(),
        "day": day_code.strip(),
        "user": user_name.strip(),
        "plates": plates_count.strip(),
        "remind": remind_me.strip(),
    }


def write_outputs(root_dir: Path, rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    csv_path = root_dir / "image_metadata.csv"
    json_path = root_dir / "image_metadata.json"
    ics_path = root_dir / "reminders.ics"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    lines = ["BEGIN:VCALENDAR", "VERSION:2.0"]
    for row in rows:
        dt = parse_remind(row.get("remind_me") or row.get("remind"))
        if not dt:
            continue
        start = dt.strftime("%Y%m%dT%H%M%S")
        end = (dt + timedelta(minutes=30)).strftime("%Y%m%dT%H%M%S")
        lines += [
            "BEGIN:VEVENT",
            f"SUMMARY:Take image {row.get('experiment_name', '').strip() or 'metadata reminder'}",
            f"DTSTART:{start}",
            f"DTEND:{end}",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    ics_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, json_path, ics_path


def default_folder() -> Path:
    return DEFAULT_INPUT_IMAGES_DIR if DEFAULT_INPUT_IMAGES_DIR.exists() else Path.cwd()


def reset_workflow(root_dir: Path) -> None:
    images = discover_images(root_dir)
    st.session_state.root_dir = root_dir
    st.session_state.images = images
    st.session_state.idx = 0
    st.session_state.store = {}
    st.session_state.prev_values = {}
    st.session_state.finished = False
    st.session_state.form_image_rel = None
    st.session_state.loaded = True
    st.session_state.load_message = ""


def current_image() -> Path | None:
    images: list[Path] = st.session_state.get("images", [])
    idx = int(st.session_state.get("idx", 0))
    if not images:
        return None
    if idx < 0:
        idx = 0
        st.session_state.idx = 0
    if idx >= len(images):
        return None
    return images[idx]


def sync_form_defaults(path: Path, img_date: datetime) -> None:
    root_dir: Path = st.session_state.root_dir
    rel = path.relative_to(root_dir).as_posix()
    store: dict[str, dict[str, str]] = st.session_state.get("store", {})
    prev_values: dict[str, str] = st.session_state.get("prev_values", {})
    autofill = bool(st.session_state.get("autofill", True))

    if st.session_state.get("form_image_rel") == rel:
        return

    row = store.get(rel)
    defaults = {
        "experiment_name": row.get("experiment_name", "") if row else "",
        "experiment_date": row.get("experiment_date", "") if row else "",
        "user_name": row.get("user_name", "") if row else "",
        "plates_count": row.get("plates_count", "") if row else "",
        "remind_me": row.get("remind_me", "") if row else "",
    }

    if not row and autofill:
        defaults.update({
            "experiment_name": prev_values.get("experiment_name", ""),
            "experiment_date": prev_values.get("experiment_date", ""),
            "user_name": prev_values.get("user_name", ""),
            "plates_count": prev_values.get("plates_count", ""),
            "remind_me": prev_values.get("remind_me", ""),
        })

    for key, value in defaults.items():
        st.session_state[key] = value

    st.session_state.form_image_rel = rel
    st.session_state.image_date_value = to_iso_date(img_date)


def collect_current_values(path: Path, img_date: datetime) -> dict[str, str]:
    experiment_name = st.session_state.get("experiment_name", "").strip()
    experiment_date = st.session_state.get("experiment_date", "").strip()
    day_code = calc_day_code(experiment_date, img_date)
    user_name = st.session_state.get("user_name", "").strip()
    plates_count = st.session_state.get("plates_count", "").strip()
    remind_me = st.session_state.get("remind_me", "").strip()
    root_dir: Path = st.session_state.root_dir
    return build_record(
        path=path,
        root_dir=root_dir,
        experiment_name=experiment_name,
        experiment_date=experiment_date,
        image_date=img_date,
        day_code=day_code,
        user_name=user_name,
        plates_count=plates_count,
        remind_me=remind_me,
    )


def save_current_image(path: Path, img_date: datetime) -> None:
    row = collect_current_values(path, img_date)
    rel = row["image_path"]
    st.session_state.store[rel] = row
    st.session_state.prev_values = {
        "experiment_name": row["experiment_name"],
        "experiment_date": row["experiment_date"],
        "user_name": row["user_name"],
        "plates_count": row["plates_count"],
        "remind_me": row["remind_me"],
    }


def bulk_apply_current_folder(path: Path, img_date: datetime) -> int:
    root_dir: Path = st.session_state.root_dir
    current_folder = path.parent
    row_template = collect_current_values(path, img_date)
    applied = 0

    for img in st.session_state.images:
        if img.parent != current_folder:
            continue
        target_date = get_image_date(img)
        row = build_record(
            path=img,
            root_dir=root_dir,
            experiment_name=row_template["experiment_name"],
            experiment_date=row_template["experiment_date"],
            image_date=target_date,
            day_code=calc_day_code(row_template["experiment_date"], target_date),
            user_name=row_template["user_name"],
            plates_count=row_template["plates_count"],
            remind_me=row_template["remind_me"],
        )
        st.session_state.store[row["image_path"]] = row
        applied += 1

    return applied


def complete_workflow() -> None:
    root_dir: Path = st.session_state.root_dir
    rows = [st.session_state.store[k] for k in sorted(st.session_state.store)]
    if not rows:
        st.session_state.finished = True
        st.session_state.saved_paths = None
        return
    st.session_state.saved_paths = write_outputs(root_dir, rows)
    st.session_state.finished = True


def start_over() -> None:
    st.session_state.loaded = False
    st.session_state.finished = False
    st.session_state.images = []
    st.session_state.store = {}
    st.session_state.prev_values = {}
    st.session_state.idx = 0
    st.session_state.form_image_rel = None
    st.session_state.saved_paths = None
    st.session_state.load_message = ""


# ---------- app ----------

def main() -> None:
    st.set_page_config(page_title="Metadata Input", layout="wide")

    for key, value in {
        "loaded": False,
        "finished": False,
        "root_dir": default_folder(),
        "images": [],
        "idx": 0,
        "store": {},
        "prev_values": {},
        "form_image_rel": None,
        "saved_paths": None,
        "load_message": "",
        "autofill": True,
        "folder_input": str(default_folder()),
    }.items():
        st.session_state.setdefault(key, value)

    st.title("Metadata Input")
    st.caption("Streamlit replacement for the old Tk GUI. Recursive folders are supported.")

    with st.sidebar:
        st.header("Folder")
        folder_input = st.text_input("Image folder", value=st.session_state.get("folder_input", str(default_folder())))
        st.session_state.folder_input = folder_input
        st.session_state.autofill = st.checkbox("Autofill previous values", value=st.session_state.get("autofill", True))

        load_clicked = st.button("Load folder", type="primary")
        if load_clicked:
            root_dir = Path(folder_input).expanduser().resolve()
            if not root_dir.exists() or not root_dir.is_dir():
                st.error(f"Folder does not exist: {root_dir}")
            else:
                reset_workflow(root_dir)
                if not st.session_state.images:
                    st.session_state.load_message = f"No images found under {root_dir}"
                else:
                    st.session_state.load_message = f"Loaded {len(st.session_state.images)} images from {root_dir}"
                st.rerun()

        if st.session_state.loaded:
            st.write(f"Root: `{st.session_state.root_dir}`")
            st.write(f"Images: **{len(st.session_state.images)}**")
            st.write(f"Saved: **{len(st.session_state.store)}**")
            if st.button("Start over"):
                start_over()
                st.rerun()

    if st.session_state.load_message:
        if st.session_state.images:
            st.success(st.session_state.load_message)
        else:
            st.warning(st.session_state.load_message)

    if not st.session_state.loaded:
        st.info("Choose a folder in the sidebar and click **Load folder**.")
        st.stop()

    if not st.session_state.images:
        st.warning("No images are loaded yet.")
        st.stop()

    if st.session_state.finished:
        st.success("Done. Metadata files have been written.")
        if st.session_state.saved_paths:
            csv_path, json_path, ics_path = st.session_state.saved_paths
            st.code(
                f"{csv_path}\n{json_path}\n{ics_path}",
                language="text",
            )
        if st.button("Edit entries again"):
            st.session_state.finished = False
            st.session_state.idx = 0
            st.session_state.form_image_rel = None
            st.rerun()
        st.stop()

    images: list[Path] = st.session_state.images
    idx = int(st.session_state.idx)
    if idx >= len(images):
        complete_workflow()
        st.rerun()

    path = current_image()
    if path is None:
        complete_workflow()
        st.rerun()

    img_date = get_image_date(path)
    sync_form_defaults(path, img_date)

    total = len(images)
    progress = min((idx + 1) / total, 1.0)
    st.progress(progress)
    st.write(f"**{idx + 1} / {total}**  |  `{path.relative_to(st.session_state.root_dir).as_posix()}`")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.image(preview_image(path), caption=path.name, use_container_width=True)
    with col_right:
        st.subheader("Metadata")
        st.text_input("Experiment name", key="experiment_name")
        st.text_input("Experiment date (YYYY-MM-DD)", key="experiment_date")
        st.text_input("Image date", value=to_iso_date(img_date), disabled=True)
        day_code = calc_day_code(st.session_state.get("experiment_date", ""), img_date)
        st.text_input("Day code", value=day_code, disabled=True)
        st.text_input("User name", key="user_name")
        st.text_input("Plates count", key="plates_count")
        st.text_input("Remind me", key="remind_me", placeholder="YYYY-MM-DD or YYYY-MM-DD HH:MM")

        btn_cols = st.columns(4)
        with btn_cols[0]:
            save_clicked = st.button("Save", type="primary")
        with btn_cols[1]:
            discard_clicked = st.button("Discard")
        with btn_cols[2]:
            back_clicked = st.button("Back")
        with btn_cols[3]:
            bulk_clicked = st.button("Bulk apply (this folder only)")

        if save_clicked:
            save_current_image(path, img_date)
            st.session_state.idx += 1
            st.rerun()

        if discard_clicked:
            st.session_state.idx += 1
            st.rerun()

        if back_clicked:
            st.session_state.idx = max(0, st.session_state.idx - 1)
            st.rerun()

        if bulk_clicked:
            applied = bulk_apply_current_folder(path, img_date)
            st.success(f"Applied to {applied} image(s) in the current folder.")

    st.divider()
    st.subheader("Saved entries")
    rows = [st.session_state.store[k] for k in sorted(st.session_state.store)]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No entries saved yet.")

    if st.button("Finish and write files"):
        # Write the current form state before finishing so the last image is not lost.
        save_current_image(path, img_date)
        complete_workflow()
        st.rerun()


if __name__ == "__main__":
    main()
