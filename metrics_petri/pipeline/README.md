# metrics-petri-gui — Gradio web interface

`metrics-petri-gui` launches a browser-based interface for interactive petri dish colony analysis. Upload images, annotate experiment dates, run segmentation, and download a ZIP of results — all from a single Gradio tab flow.

No coding required. The full pipeline (UNet segmentation, dish detection, crack analysis, hyphal feature extraction, growth rate calculation) runs in the browser session.

---

## Installation

```bash
pip install "metrics-petri[gui]"
```

The `[gui]` extra adds Gradio and HEIF/RAW image support. If you only need the CLI, use `pip install metrics-petri`.

---

## Launch

```bash
metrics-petri-gui
```

The interface opens in your default browser at `http://localhost:7860` and listens only
on the local machine by default.

Options:

```bash
metrics-petri-gui --port 8080               # change port
metrics-petri-gui --no-browser              # don't open browser automatically
metrics-petri-gui --model /path/to/model.pt # custom UNet checkpoint
metrics-petri-gui --auth user:password       # require login
```

To serve the GUI to other machines, bind it to a network interface and require
authentication:

```bash
metrics-petri-gui --host 0.0.0.0 --auth user:password
```

!!! warning
    Binding to a non-loopback address exposes uploaded images, results, and GUI actions
    to the network. The CLI refuses to start on a non-loopback host unless `--auth`
    is supplied.

---

## Model file lookup order

At start-up the pipeline searches for the checkpoint in this order:

1. `--model` flag or `UNET_MODEL` environment variable — if set, that path is used directly.
2. `metrics_petri/models/best_area_w_0.7.pt` relative to the current working
   directory (useful when running from the cloned repo).
3. The installed package location — `importlib.resources` resolves the bundled file inside the wheel.
4. Hugging Face Hub fallback download — if none of the above exist, the pinned checkpoint revision is downloaded, cached locally, and verified against the packaged SHA-256.

The bundled model is `best_area_w_0.7.pt` (SmallUNet, `base_channels=16`, ~23 MB). A normal `pip install "metrics-petri[gui]"` includes it; no separate download is needed.

To override with a custom checkpoint:

```bash
UNET_MODEL=/path/to/checkpoint.pt metrics-petri-gui
```

---

## Interface walkthrough

The GUI is organised as a five-step tab flow.

### Step 1 — Upload Images

Drag and drop or click to upload. Accepted formats: JPEG, PNG, TIFF, BMP, WebP, HEIF/HEIC, RAW (DNG, CR2, NEF, ARW).

### Step 2 — Settings

| Setting | Default | Description |
| ------- | ------- | ----------- |
| Threshold | 0.5 | Segmentation confidence cut-off. Lower values include more of the colony; raise it to exclude dim edges. |
| Fast mode | off | Runs segmentation only (no dish detection, cracks, or growth charts). Useful for a quick mask preview. |
| Experiment name | — | Label for the run; appears in the report header and ZIP filename. |
| User name | — | Used to name the output ZIP (e.g. `rex.zip`). |
| Plates | 1 | Number of plates in the experiment. |

### Step 3 — Review & Edit Dates

A table shows each uploaded filename alongside auto-detected `experiment_date` and `image_date` fields (`YYYY-MM-DD`).

- Dates are auto-detected from the filename (`YYYYMMDD` pattern) or EXIF `DateTimeOriginal`. File modification time is not used because uploaded files are temporary Gradio copies.
- Edit any date by clicking the cell. Day codes (`d01`, `d02`, …) update automatically.
- Filling in dates enables growth rate calculations and correctly labelled chart axes.
- Leave blank to analyse morphology only.

### Step 4 — Export Metadata

Download `image_metadata.csv` (and optionally `image_metadata.json` / `reminders.ics`). Re-upload this file in a future session to skip manual date entry. The same file can also be passed to the CLI via `--metadata`.

### Step 5 — Run Inference

Click **Run pipeline**. Progress is shown per image.

**Fast mode** outputs for each image:

- Raw image
- Predicted mask
- Colour overlay

**Full mode** additionally outputs:

- Dish-circle detection overlay
- Crack detection overlay
- Hyphal skeleton panels (Frangi, Meijering, hybrid)
- Growth rate charts (one per metric, day-code labels on x-axis)
- **Download results ZIP** button

---

## Output ZIP (full mode)

```text
<user_or_experiment_name>.zip
├── analysis_full.csv         one row per image, all metrics
├── analysis_full.json        same data as a JSON array
├── provenance.json           run settings, versions, device, and model checksum
├── image_metadata.csv        copy of the metadata used
├── image_metadata.json       same metadata as JSON
├── chart_*.png               growth-rate charts, when enough dated images exist
└── overlays/
    ├── image01_raw.jpg
    ├── image01_mask.jpg
    ├── image01_colony.jpg
    ├── image01_cracks.jpg
    └── ...
```

The ZIP is named from the `user_name` field, falling back to `experiment_name`, then `analysis.zip`.

---

## Notebook walkthrough

An interactive notebook (`notebooks/example_metrics-petri.ipynb`) is available in the [GitHub repository](https://github.com/rotsl/metrics-petri). It traces the full pipeline — mask inference, dish detection, crack analysis, and growth metrics — with inline plots at each step.

The notebook is not distributed with the pip package. Clone the repo and run:

```bash
make run-notebook
```

---

## Diagnostics

```bash
metrics-petri-gui doctor
```

Prints Python version, NumPy version, Torch version and accelerator, Gradio version, model path, and dependency health. Exits with code 1 if any issue is found.

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1 (CPU is fine; MPS is used automatically on Apple Silicon)
- Gradio ≥ 6.0

All dependencies are installed with `pip install "metrics-petri[gui]"`.

---

## License

MIT. See [LICENSE](https://github.com/rotsl/metrics-petri/blob/main/LICENSE) for the full text.
