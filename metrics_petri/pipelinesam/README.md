# metrics-petri — CLI batch pipeline

`metrics-petri` segments and measures fungal colonies (or any biological sample) growing on petri dishes. It processes a folder of images and writes a ZIP archive containing CSV and JSON result tables, `provenance.json`, per-image overlay composites, and growth-rate charts.

No browser or GUI required at run time.

---

## Installation

```bash
pip install metrics-petri
```

The package bundles the SmallUNet checkpoint (`best_area_w_0.7.pt`) inside the wheel. No separate model download is needed after a normal `pip install`.

---

## Model file lookup order

At start-up the pipeline searches for the checkpoint in this order:

1. `UNET_MODEL` environment variable — if set, that path is used directly.
2. `metrics_petri/models/best_area_w_0.7.pt` relative to the current working
   directory (useful when running from the cloned repo).
3. The installed package location — `importlib.resources` resolves the bundled file inside the wheel.
4. Hugging Face Hub fallback download — if none of the above exist, the pinned checkpoint revision is downloaded, cached locally, and verified against the packaged SHA-256.

To use a custom checkpoint:

```bash
UNET_MODEL=/path/to/checkpoint.pt metrics-petri images/
# or
metrics-petri images/ --model /path/to/checkpoint.pt
```

---

## Step 0 — Prepare metadata (optional but recommended)

A metadata file tells the pipeline which experiment each image belongs to, the image date, and the experiment start date. It enables:

- Grouping images into time-series experiments
- Computing relative growth rate (RGR, day⁻¹) and area growth rate (mm² day⁻¹)
- Labelling chart axes with day codes (`d01`, `d02`, …)

### Option A — Desktop GUI (`metrics-petri-metadata`)

The package ships a native tkinter desktop app for building the metadata file without writing any code.

**Requirements:** tkinter is part of the Python standard library and is bundled on macOS and Windows. On Linux install it first:

```bash
sudo apt install python3-tk   # Debian / Ubuntu
sudo dnf install python3-tkinter  # Fedora / RHEL
```

**Launch:**

```bash
metrics-petri-metadata
```

| Step | What you do |
| ---- | ----------- |
| **1 — Select Folder** | Browse to the folder containing your petri-dish images. Thumbnails are shown for the first 20 images. Dates are auto-detected from filenames (`YYYYMMDD`), EXIF `DateTimeOriginal`, or file modification time. |
| **2 — Settings** | Enter the experiment name, experiment start date (`YYYY-MM-DD`), your user name, and the number of plates. |
| **3 — Review & Edit Dates** | A table lists every image with its auto-detected date and computed day code. Click any row to edit its date or set an optional reminder (`YYYY-MM-DD HH:MM`). Day codes update automatically when dates change. |
| **4 — Export** | Click **Export All** to write `image_metadata.csv`, `image_metadata.json`, and (if any reminders were set) `reminders.ics` into the same folder as the images. |

After export you will have `image_metadata.csv` ready to pass to the CLI.

### Option B — Manual CSV

Create `image_metadata.csv` with these columns (extras are ignored):

| Column | Format | Required |
| ------ | ------ | -------- |
| `image_path` | filename only, e.g. `plate01.jpg` | yes |
| `experiment_name` | string | no |
| `experiment_date` | `YYYY-MM-DD` | no |
| `image_date` | `YYYY-MM-DD` | no |
| `day_code` | `dNN`, e.g. `d03` | no |
| `user_name` | string | no |
| `plates_count` | integer | no |

A JSON array with the same keys is also accepted (`image_metadata.json`).

---

## Running the pipeline

```bash
# Analyse all images in a folder (no metadata)
metrics-petri input_images/

# With metadata CSV — enables growth rate calculations and day-code charts
metrics-petri input_images/ --metadata input_images/image_metadata.csv

# JSON metadata is also accepted
metrics-petri input_images/ --metadata input_images/image_metadata.json

# Write the ZIP to a specific path
metrics-petri input_images/ --output results/run01.zip

# Adjust segmentation confidence threshold (default 0.5)
metrics-petri input_images/ --threshold 0.45

# Calibrate measurements for a 60 mm dish (default: 90 mm)
metrics-petri input_images/ --dish-size-mm 60

# Set the PyTorch seed recorded in provenance (default: 0)
metrics-petri input_images/ --seed 123

# Custom model checkpoint
metrics-petri input_images/ --model /path/to/checkpoint.pt
```

The command prints per-image progress and writes a single ZIP when complete.

## Example dataset

The source repository includes three example images in `input_images/06FEB` for
smoke-testing the full CLI:

```bash
metrics-petri input_images/06FEB --output /tmp/metrics-petri-06feb.zip --seed 0
```

### Auto-discovery of metadata

If `--metadata` is not given, the CLI looks for `image_metadata.json` and then `image_metadata.csv` inside the image folder automatically.

---

## Output ZIP structure

```text
<user_or_experiment_name>.zip
├── analysis_full.csv         one row per image, all metrics
├── analysis_full.json        same data as a JSON array
├── provenance.json           run settings, versions, device, and model checksum
├── image_metadata.csv        copy of the input metadata (if supplied)
├── image_metadata.json       same metadata as JSON
├── overlays/
│   ├── image01_overlay.jpg   raw image + dish circle + colony mask
│   └── ...
└── charts/
    ├── area_mm2.png          colony area over time
    ├── diameter_mm.png
    ├── edge_roughness.png
    ├── crack_coverage_pct.png
    └── ...                   one chart per key metric (requires dates)
```

The ZIP is named from the `user_name` field in the metadata (e.g. `rex.zip`), falling back to `experiment_name`, then `analysis.zip`.

---

## Metrics in analysis_full.csv

| Column | Unit | Description |
| ------ | ---- | ----------- |
| `area_mm2` | mm² | Colony area |
| `diameter_mm` | mm | Equivalent circle diameter |
| `perimeter_mm` | mm | Colony perimeter |
| `eccentricity` | — | Shape elongation (0 = circle, 1 = line) |
| `edge_roughness` | — | Perimeter / ideal-circle perimeter |
| `centre_delta_mm` | mm | Offset of colony centroid from dish centre |
| `texture_std` | — | Standard deviation of pixel intensities |
| `entropy` | bits | Shannon entropy of pixel intensities |
| `crack_px` | px | Crack area in pixels |
| `crack_area_mm2` | mm² | Crack area in physical units |
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

Prints Python version, NumPy version, Torch version and accelerator, model path, and dependency health. Exits with code 1 if any issue is found.

---

## Notebook walkthrough

An interactive notebook (`notebooks/example_metrics-petri.ipynb`) is available in the [GitHub repository](https://github.com/rotsl/metrics-petri). It traces the full pipeline — mask inference, dish detection, crack analysis, and growth metrics — with inline plots at each step.

The notebook is not distributed with the pip package. Clone the repo and run:

```bash
make run-notebook
```

---

## Supported image formats

JPEG, PNG, TIFF, BMP, WebP. The pipeline resizes each image to 256 × 256 for the neural network and then maps measurements back to physical units using the detected dish circumference.

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1 (CPU is fine; MPS is used automatically on Apple Silicon)
- tkinter — stdlib, bundled on macOS/Windows; `sudo apt install python3-tk` on Linux

All pip-installable dependencies are installed automatically with `pip install metrics-petri`.

---

## License

MIT. See [LICENSE](https://github.com/rotsl/metrics-petri/blob/main/LICENSE) for the full text.
