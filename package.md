# metrics-petri

Petri dish colony segmentation and morphometric analysis.

`metrics-petri` measures how a biological sample grows on a petri dish: area, diameter, edge roughness, crack burden, texture entropy, and time-series growth rates — all in physical units calibrated from the dish geometry.

---

## Three entry points, one package

| Entry point | Install | Use |
| ----------- | ------- | --- |
| `metrics-petri` | `pip install metrics-petri` | CLI batch pipeline |
| `metrics-petri-gui` | `pip install "metrics-petri[gui]"` | Gradio browser GUI |
| `metrics-petri-metadata` | `pip install metrics-petri` | Desktop GUI for building `image_metadata.csv` |

---

## Model checkpoint

The package bundles the SmallUNet checkpoint (`best_area_w_0.7.pt`, ~23 MB) inside the wheel. No separate download is needed after `pip install`.

At run time the checkpoint is located in this order:

1. `UNET_MODEL` environment variable or `--model` flag
2. `models/best_area_w_0.7.pt` in the current working directory
3. The installed package location (bundled in the wheel)
4. HuggingFace Hub auto-download (`rotsl/grayleafspot-segmentation`) as a last resort

---

## CLI batch pipeline

```bash
pip install metrics-petri
metrics-petri input_images/
```

Processes every image in the folder and writes a ZIP containing results, overlays, and charts.

```bash
# With metadata for growth rate calculations and day-code charts
metrics-petri input_images/ --metadata input_images/image_metadata.csv

# JSON metadata is also accepted
metrics-petri input_images/ --metadata input_images/image_metadata.json

# Custom output path
metrics-petri input_images/ --output results/run01.zip

# Adjust segmentation threshold (default 0.5)
metrics-petri input_images/ --threshold 0.45

# Custom model checkpoint
metrics-petri input_images/ --model /path/to/checkpoint.pt
```

### Output ZIP

```text
<user_or_experiment_name>.zip
├── analysis_full.csv         one row per image, all metrics
├── analysis_full.json        same data as a JSON array
├── image_metadata.csv        copy of the input metadata (if supplied)
├── image_metadata.json       same metadata as JSON
├── overlays/                 per-image colony mask composites
└── charts/                   growth-rate charts (requires dates in metadata)
```

---

## Metadata desktop GUI

`metrics-petri-metadata` is a native tkinter application for creating the `image_metadata.csv` that drives growth rate calculations and chart labelling.

**Requirements:** tkinter is part of the Python standard library.

- macOS / Windows: bundled with the official Python installer.
- Linux: `sudo apt install python3-tk` (or `sudo dnf install python3-tkinter`)

```bash
metrics-petri-metadata
```

Four-step flow:

1. **Select Folder** — browse to the image folder; dates are auto-detected from filenames, EXIF data, or file modification time.
2. **Settings** — enter experiment name, experiment start date, user name, and plate count.
3. **Review & Edit Dates** — a table lists every image with its date and day code (`d01`, `d02`, …); click any row to correct a date or set a reminder.
4. **Export** — writes `image_metadata.csv`, `image_metadata.json`, and optionally `reminders.ics` into the image folder.

Pass the exported file to the CLI with `--metadata` or upload it in the Gradio GUI's Step 4.

---

## Gradio browser GUI

```bash
pip install "metrics-petri[gui]"
metrics-petri-gui
```

Opens at `http://localhost:7860`. Five-step tab flow:

1. **Upload images** — JPEG, PNG, TIFF, BMP, WebP, HEIF, RAW
2. **Settings** — threshold, fast mode, experiment name, user name, plate count
3. **Review & edit dates** — annotate `experiment_date` / `image_date` per image; day codes are computed automatically
4. **Export metadata** — download `image_metadata.csv` for future runs or for the CLI
5. **Run inference** — segmentation, dish detection, cracks, hyphal skeletons, growth charts, ZIP download

Options:

```bash
metrics-petri-gui --port 8080
metrics-petri-gui --host 127.0.0.1
metrics-petri-gui --no-browser
metrics-petri-gui --model /path/to/checkpoint.pt
```

---

## Measured metrics

| Metric | Unit | Description |
| ------ | ---- | ----------- |
| `area_mm2` | mm² | Colony area |
| `diameter_mm` | mm | Equivalent circle diameter |
| `perimeter_mm` | mm | Colony perimeter |
| `eccentricity` | — | Shape elongation (0 = circle, 1 = line) |
| `edge_roughness` | — | Perimeter / ideal-circle perimeter |
| `centre_delta_mm` | mm | Colony centroid offset from dish centre |
| `entropy` | bits | Shannon entropy of pixel intensities |
| `texture_std` | — | Standard deviation of pixel intensities |
| `crack_area_mm2` | mm² | Crack area |
| `crack_coverage_pct` | % | Crack area as percentage of colony area |
| `crack_count` | — | Number of discrete crack regions |
| `hyph_frangi_mm` | mm | Hyphal length (Frangi filter) |
| `hyph_meijering_mm` | mm | Hyphal length (Meijering filter) |
| `hyph_hybrid_mm` | mm | Hyphal length (hybrid filter) |
| `rgr_per_day` | day⁻¹ | Relative growth rate (requires dates) |
| `relative_growth_per_day` | mm² day⁻¹ | Absolute area growth rate (requires dates) |

Scale is derived from the detected dish circumference (default 90 mm). No calibration target required.

---

## Diagnostics

```bash
metrics-petri doctor
metrics-petri-gui doctor
```

Checks Python, NumPy (warns if 2.x), Torch, accelerator (MPS/CUDA/CPU), model path, and all dependencies. Exits with code 1 on any issue.

---

## Notebook walkthrough

```python
import importlib.resources, shutil, pathlib

src = importlib.resources.files("pipelinesam") / "notebooks" / "example_metrics-petri.ipynb"
shutil.copy(src, pathlib.Path.cwd() / "example_metrics-petri.ipynb")
```

Open `example_metrics-petri.ipynb` in JupyterLab or VS Code to trace each stage of the pipeline interactively.

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1 (CPU works; MPS used automatically on Apple Silicon)
- Gradio ≥ 6.0 (GUI only, via `[gui]` extra)
- tkinter — stdlib, bundled on macOS/Windows; `sudo apt install python3-tk` on Linux

---

## License

Apache 2.0 — [full text](https://github.com/rotsl/metrics-petri/blob/main/LICENSE)

## Citation

```text
Metrics Petri: petri dish colony segmentation and morphometric analysis. Rohan R, 2026.
https://github.com/rotsl/metrics-petri
```
