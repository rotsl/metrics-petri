# metrics-petri

Macroscopic image analysis for fungal colony growth on petri dishes.

`metrics-petri` measures how a sample expands, whether its edge stays smooth or roughens, when cracks appear, and how centre-to-edge texture evolves over time. It turns a folder of time-series images into physical measurements (mm², mm, day⁻¹) with overlay visualisations.

The repository ships two entry points:

| Entry point | Install | Use |
| --- | --- | --- |
| `metrics-petri` | `pip install metrics-petri` | CLI batch pipeline |
| `metrics-petri-gui` | `pip install "metrics-petri[gui]"` | Gradio browser GUI |

---

## Repository setup

### Prerequisites

- Python 3.10 or later
- `make` (standard on macOS/Linux)
- The UNet checkpoint `models/best_area_w_0.7.pt` (see below)

### Clone and install

```bash
git clone https://github.com/rotsl/metrics-petri.git
cd metrics-petri
make setup          # create venv + install all Python deps
make install        # install package in editable mode
```

To include the Gradio GUI:

```bash
make install-gui
```

### Model checkpoint

The checkpoint is not tracked in the repository. Place it at:

```text
models/best_area_w_0.7.pt
```

If you have it elsewhere:

```bash
UNET_MODEL=/path/to/best_area_w_0.7.pt make run-cli INPUT=input_images/
```

---

## CLI usage (batch pipeline)

```bash
# Analyse a folder of images
metrics-petri input_images/

# Specify output path
metrics-petri input_images/ --output results/run01.zip

# Supply experiment metadata for growth rate calculations
metrics-petri input_images/ --metadata input_images/image_metadata.csv

# Adjust segmentation threshold
metrics-petri input_images/ --threshold 0.45
```

Output is a single ZIP containing `analysis_full.csv`, `analysis_full.json`, and per-image overlays.

Full CLI documentation: [`pipelinesam/README.md`](pipelinesam/README.md)

---

## GUI usage

```bash
make run-gui
# or directly:
metrics-petri-gui
```

Opens at `http://localhost:7860`. Five-step tab flow: upload → settings → edit dates → export metadata → run pipeline.

Full GUI documentation: [`pipeline/README.md`](pipeline/README.md)

---

## Notebook walkthrough

```bash
make run-notebook
```

Opens `notebooks/example_metrics-petri.ipynb`, which traces the full pipeline interactively — mask inference, dish detection, crack analysis, and growth metrics — with inline plots at each step.

---

## Makefile targets

| Target | Description |
| --- | --- |
| `make setup` | Create virtual environment and install core dependencies |
| `make install` | Install in editable mode (core, no GUI) |
| `make install-gui` | Install with Gradio GUI extras |
| `make run-gui` | Launch Gradio interface |
| `make run-cli INPUT=path/` | Run batch CLI on a folder |
| `make run-notebook` | Open the example notebook in JupyterLab |
| `make build-package` | Build wheel and sdist for PyPI |
| `make publish-pypi` | Upload to PyPI with twine |
| `make model-status` | Check whether the checkpoint is present |
| `make clean` | Remove venv, caches, build artefacts |

---

## Repository layout

```text
metrics-petri/
├── pipeline/           # Gradio GUI package (metrics-petri-gui)
│   ├── app.py          # Gradio application
│   ├── analysis.py     # Core analysis functions
│   ├── model.py        # SmallUNet definition
│   ├── reporting.py    # matplotlib report generation
│   └── cli.py          # metrics-petri-gui entry point
├── pipelinesam/        # CLI batch pipeline (metrics-petri)
│   ├── pipeline.py     # Full pipeline logic
│   ├── cli.py          # metrics-petri entry point
│   ├── model_small_unet.py
│   └── notebooks/      # Notebook shipped with the package
├── notebooks/          # Development notebooks
│   └── example_metrics-petri.ipynb
├── models/             # Local model checkpoints (gitignored)
│   └── best_area_w_0.7.pt
├── input_images/       # Source images (gitignored)
├── outputs/            # Analysis outputs (gitignored)
└── pyproject.toml
```

---

## Measured features

| Feature | Unit |
| --- | --- |
| Colony area | mm² |
| Equivalent diameter | mm |
| Perimeter | mm |
| Circularity | — |
| Eccentricity | — |
| Edge roughness | — |
| Texture entropy | bits |
| Crack area / coverage | mm² / % |
| Crack count | — |
| Relative growth rate | day⁻¹ |
| Area growth rate | mm² day⁻¹ |

Scale is derived from the detected dish circumference (default 90 mm). No calibration target required.

---

## License and citation

Apache 2.0 — see [`LICENSE`](LICENSE).

```text
Metrics Petri: petri dish colony segmentation and morphometric analysis. 2026.
https://github.com/rotsl/metrics-petri
```

Machine-readable citation: [`CITATION.cff`](CITATION.cff)
