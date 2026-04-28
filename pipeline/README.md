# metrics-petri-gui — Gradio web interface

`metrics-petri-gui` launches a browser-based interface for interactive petri dish colony analysis. Upload images, annotate experiment dates, run segmentation, and download a ZIP of results — all from a single Gradio tab flow.

No coding required. The full pipeline (UNet segmentation, dish detection, crack analysis, hyphal feature extraction, growth rate calculation) runs in the browser session.

---

## Installation

```bash
pip install "metrics-petri[gui]"
```

The `[gui]` extra adds Gradio and HEIF/RAW image support. The core package without the GUI is `pip install metrics-petri`.

---

## Launch

```bash
metrics-petri-gui
```

The interface opens in your default browser at `http://localhost:7860`.

Options:

```bash
metrics-petri-gui --port 8080              # change port
metrics-petri-gui --host 127.0.0.1         # restrict to localhost
metrics-petri-gui --no-browser             # don't open browser automatically
metrics-petri-gui --model /path/to/model.pt  # custom UNet checkpoint
```

---

## Interface walkthrough

The GUI is organised as a five-step tab flow:

### 1. Upload Images

Drag and drop or click to upload. Accepted formats: JPEG, PNG, TIFF, BMP, WebP, HEIF/HEIC, RAW (DNG, CR2, NEF, ARW).

### 2. Settings

- **Threshold** — segmentation confidence cutoff (default 0.5). Lower values include more of the colony; raise it to exclude dim edges.
- **Fast mode** — runs segmentation only (no dish detection, cracks, or growth charts). Useful for a quick mask preview.
- **Experiment name** — label for the run; appears in the report header.

### 3. Review & Edit Dates

A table shows each uploaded filename alongside editable `experiment_date` and `image_date` fields (YYYY-MM-DD). Filling in dates enables growth rate calculations. Leave blank to analyse morphology only.

### 4. Export Metadata

Download a `image_metadata.csv` that records the experiment name and dates. Re-upload this CSV in future sessions to skip manual date entry.

### 5. Run Inference

Click **Run pipeline**. The interface shows:

- **Fast mode**: raw image, predicted mask, and colour overlay for each image.
- **Full mode**: all of the above plus dish-circle detection, crack overlay, hyphal skeleton panels, and growth rate charts. A **Download results ZIP** button appears when the run completes.

---

## Output ZIP (full mode)

```
results.zip
├── analysis_full.csv
├── analysis_full.json
└── overlays/
    ├── image01_raw.jpg
    ├── image01_mask.jpg
    ├── image01_colony.jpg
    ├── image01_cracks.jpg
    └── ...
```

---

## Custom model checkpoint

Pass `--model` at launch or set the `UNET_MODEL` environment variable:

```bash
UNET_MODEL=/path/to/checkpoint.pt metrics-petri-gui
```

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1
- Gradio ≥ 5.0

All dependencies are installed with `pip install "metrics-petri[gui]"`.

---

## License

Apache 2.0. See [LICENSE](https://github.com/rotsl/metrics-petri/blob/main/LICENSE) for the full text.
