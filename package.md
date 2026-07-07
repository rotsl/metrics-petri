# metrics-petri

[![PyPI](https://img.shields.io/pypi/v/metrics-petri?style=flat-square&logo=pypi&label=PyPI)](https://pypi.org/project/metrics-petri/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/metrics-petri?style=flat-square&logo=rocket&label=Python)
![PyPI - License](https://img.shields.io/pypi/l/metrics-petri?style=flat-square)
[![CI](https://img.shields.io/github/actions/workflow/status/rotsl/metrics-petri/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/rotsl/metrics-petri/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/rotsl/metrics-petri/mkdocs-gh-pages.yml?branch=main&style=flat-square&label=Docs)](https://github.com/rotsl/metrics-petri/actions/workflows/mkdocs-gh-pages.yml)

Petri dish colony segmentation and morphometric analysis.

`metrics-petri` measures how a biological sample grows on a petri dish: area, diameter, edge roughness, crack burden, texture entropy, and time-series growth rates — all in physical units calibrated from the dish geometry.

---

## Workflow

```text
  📁 Image folder
       │
       ▼
  metrics-petri-metadata          ← optional but recommended
       Step 1 · select folder (dates auto-detected)
       Step 2 · experiment name · start date · plates
       Step 3 · review and assign day codes per image
       Step 4 · export
       │
       │ writes
       ▼
  image_metadata.csv
       │ --metadata
       └───────────────────────────────────────┐
                                               │
  📁 Image folder ─────────────────────────► metrics-petri
                                               │
                                               ▼
                                          results.zip
                                          ├── analysis_full.csv
                                          ├── analysis_full.json
                                          ├── provenance.json
                                          ├── overlays/
                                          └── charts/  ← growth curves with day codes
```

`metrics-petri-metadata` is optional — `metrics-petri` can run on images alone, but supplying metadata enables growth-rate calculations and day-coded charts.

---

## Four entry points, one package

| Entry point | Install | Use |
| ----------- | ------- | --- |
| `metrics-petri` | `pip install metrics-petri` | CLI batch pipeline |
| `metrics-petri-gui` | `pip install "metrics-petri[gui]"` | Browser-based analysis GUI; public network binding requires `--auth` |
| `metrics-petri-metadata` | `pip install metrics-petri` | Desktop GUI for building `image_metadata.csv` |
| `metrics-petri-crop` | `pip install metrics-petri` | CLI crop multi-dish images into per-dish PNGs |

---

## Model checkpoint

The package bundles the SmallUNet checkpoint (`best_area_w_0.7.pt`, ~23 MB) inside the wheel. No separate download is needed after `pip install`. The bundled and downloaded checkpoint is verified against its packaged SHA-256 before use. The model was trained and validated using [**petrimodel**](https://github.com/rotsl/petrimodel) — a companion repository covering training, annotation, sweep evaluation, and manual diameter validation against model-generated masks.

At run time the checkpoint is located in this order:

1. `UNET_MODEL` environment variable or `--model` flag
2. `metrics_petri/models/best_area_w_0.7.pt` in the current working directory
3. The installed package location (bundled in the wheel)
4. HuggingFace Hub auto-download (`rotsl/grayleafspot-segmentation`) as a last resort

---

## CLI batch pipeline

```bash
python3.12 -m venv petrienv
source petrienv/bin/activate
pip install --upgrade pip
pip install metrics-petri
metrics-petri input_images/
```

Processes every image in the folder and writes a ZIP containing result tables,
provenance, overlays, and charts.

Images are resized internally to `256 × 256` RGB for SmallUNet inference, then the mask
is mapped back to the source image. The default mask-confidence threshold is `0.5`;
lower values can include weaker colony signal but may add false positives, while higher
values are stricter and may under-segment faint edges.

```bash
# With metadata for growth rate calculations and day-code charts
metrics-petri input_images/ --metadata input_images/image_metadata.csv

# JSON metadata is also accepted
metrics-petri input_images/ --metadata input_images/image_metadata.json

# Custom output path
metrics-petri input_images/ --output results/run01.zip

# Adjust segmentation threshold (default: 0.5)
metrics-petri input_images/ --threshold 0.45

# Use the actual outside diameter for non-standard dishes
metrics-petri input_images/ --dish-size-mm 60

# Set the PyTorch seed recorded in provenance (default: 0)
metrics-petri input_images/ --seed 123

# Custom model checkpoint
metrics-petri input_images/ --model /path/to/checkpoint.pt
```

### Output ZIP

```text
<user_or_experiment_name>.zip
├── analysis_full.csv         one row per image, all metrics
├── analysis_full.json        same data as a JSON array
├── provenance.json           run settings, versions, device, and model checksum
├── image_metadata.csv        copy of the input metadata (if supplied)
├── image_metadata.json       same metadata as JSON
├── overlays/                 per-image colony mask composites
└── charts/                   growth-rate charts (requires dates in metadata)
```

### Example dataset

The source repository includes three example images under `input_images/06FEB` for
smoke-testing and tutorials:

```bash
metrics-petri input_images/06FEB --output /tmp/metrics-petri-06feb.zip --seed 0
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

Pass the exported file to the CLI with `--metadata`.

---

## Dish cropper

`metrics-petri-crop` automatically detects and crops individual petri dishes from photos where several dishes were captured together in a single image (2–8+ per photo). It is a standalone utility — independent of the analysis pipeline.

```bash
metrics-petri-crop -i input_images/
```

| Flag | Description |
| ---- | ----------- |
| `-i, --input` | Image file or directory |
| `-o, --output` | Output directory (default: `<input>/cropped/`) |
| `-p, --padding` | Extra space around each dish (default: `0.05` = 5% of radius) |
| `-d, --debug` | Save debug overlays showing detection circles |
| `-D, --date` | Prefix output filenames: `DD/MM/YYYY`, `DD/Mon`, or `DD Mon` |

Only fully visible dishes are extracted. Output filenames follow the pattern `{stem}_dish_{NN}.png` (or `{YYYYMMDD}_{stem}_dish_{NN}.png` with `--date`).

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

Scale is derived from the detected dish circumference and the configured outside dish
diameter. The default is 90 mm; pass `--dish-size-mm` when using another size. No
separate calibration target is required.

---

## Diagnostics

```bash
metrics-petri doctor
```

Checks Python, NumPy, Torch, accelerator (MPS/CUDA/CPU), model path, and all dependencies. Exits with code 1 on any issue.

At run time the pipeline selects CUDA first when available, then Apple MPS, then CPU.

---

## Notebook walkthrough

An interactive notebook is available in the [GitHub repository](https://github.com/rotsl/metrics-petri) at `notebooks/example_metrics-petri.ipynb`. It traces the full pipeline — mask inference, dish detection, crack analysis, and growth metrics — with inline plots at each step.

The notebook is not distributed with the pip package. Clone the repository to use it.

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1 (CPU works; MPS used automatically on Apple Silicon)
- tkinter — stdlib, bundled on macOS/Windows; `sudo apt install python3-tk` on Linux

---

## License

MIT — [full text](https://github.com/rotsl/metrics-petri/blob/main/LICENSE)

## Citation

```bibtex
@software{Rohan_R_Metrics_Petri,
author = {{Rohan R}},
title = {{Metrics Petri: petri dish colony segmentation and morphometric analysis}},
url = {https://github.com/rotsl/metrics-petri},
version = {2.1.2},
year = {2026}
}
```
