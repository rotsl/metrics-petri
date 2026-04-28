# metrics-petri — CLI batch pipeline

`metrics-petri` segments and measures fungal colonies (or any biological sample) growing on petri dishes. It processes a folder of images and writes a ZIP archive containing a CSV results table, a JSON record, and per-image overlay composites.

No browser, no GUI, no internet connection required at run time.

---

## Installation

```bash
pip install metrics-petri
```

The package ships with the SmallUNet checkpoint. The first run loads the model from the installed package location automatically.

To use a custom checkpoint:

```bash
UNET_MODEL=/path/to/checkpoint.pt metrics-petri images/
```

or via the `--model` flag:

```bash
metrics-petri images/ --model /path/to/checkpoint.pt
```

---

## Quick start

```bash
# Analyse all images in a folder
metrics-petri input_images/

# Write output to a specific path
metrics-petri input_images/ --output results/run01.zip

# Adjust segmentation threshold (default 0.5)
metrics-petri input_images/ --threshold 0.45

# Supply a metadata CSV for date-aware growth rate calculations
metrics-petri input_images/ --metadata input_images/image_metadata.csv
```

The command prints per-image progress and writes a single ZIP when complete.

---

## Metadata CSV

If you pass `--metadata`, the CSV must contain at least an `image_path` column with paths relative to the image directory. Optional columns that enable growth rate calculations:

| Column | Format | Purpose |
|--------|--------|---------|
| `experiment_date` | `YYYY-MM-DD` | Start date of the experiment |
| `image_date` | `YYYY-MM-DD` | Date each image was captured |
| `experiment_name` | string | Groups images into series |

When `image_date` and `experiment_date` are present, the pipeline computes relative growth rate (RGR, day⁻¹) and area growth rate (mm² day⁻¹).

---

## Output ZIP structure

```
results.zip
├── analysis_full.csv        # one row per image, all metrics
├── analysis_full.json       # same data as JSON array
└── overlays/
    ├── image01_overlay.jpg  # raw image + dish circle + colony mask
    ├── image02_overlay.jpg
    └── ...
```

### Metrics in analysis_full.csv

| Column | Unit | Description |
|--------|------|-------------|
| `area_mm2` | mm² | Colony area |
| `diameter_mm` | mm | Equivalent circle diameter |
| `perimeter_mm` | mm | Colony perimeter |
| `circularity` | — | 4π·area / perimeter² |
| `eccentricity` | — | Shape elongation (0 = circle, 1 = line) |
| `edge_roughness` | — | Perimeter / ideal-circle perimeter |
| `texture_entropy` | bits | Shannon entropy of pixel intensities inside colony |
| `crack_px` | px | Crack area in pixels |
| `crack_area_mm2` | mm² | Crack area in physical units |
| `crack_coverage_pct` | % | Crack area as percentage of colony area |
| `crack_count` | — | Number of discrete crack regions |
| `rgr_per_day` | day⁻¹ | Relative growth rate (log-linear, requires dates) |
| `relative_growth_per_day` | mm² day⁻¹ | Absolute area growth rate (requires dates) |

---

## Notebook walkthrough

After installation, a step-by-step Jupyter notebook is included in the package:

```python
import importlib.resources
import shutil, pathlib

src = importlib.resources.files("pipelinesam") / "notebooks" / "example_metrics-petri.ipynb"
shutil.copy(src, pathlib.Path.cwd() / "example_metrics-petri.ipynb")
```

Then open `example_metrics-petri.ipynb` in JupyterLab or VS Code.

The notebook mirrors the CLI pipeline and lets you inspect each stage interactively.

---

## Supported image formats

JPEG, PNG, TIFF, BMP, WebP. The pipeline resizes each image to 256 × 256 for the neural network and then maps measurements back to physical units using the detected dish circumference (90 mm default).

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1 (CPU is fine; MPS is used automatically on Apple Silicon)

All other dependencies are installed automatically with `pip install metrics-petri`.

---

## License

Apache 2.0. See [LICENSE](https://github.com/rotsl/metrics-petri/blob/main/LICENSE) for the full text.
