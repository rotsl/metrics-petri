# metrics-petri

# metrics-petri <img src="metrics-petri-logo.png" align="right" height="139" alt="metrics-petri logo" />

![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Frotsl%2Fmetrics-petri%2Frefs%2Fheads%2Fmain%2Fpyproject.toml%3Ftoken%3DGHSAT0AAAAAADYTJ5BCWDW5TQEC7SBP25UM2PTAXHQ)
![PyPI - License](https://img.shields.io/pypi/l/metrics-petri?style=flat-square&logo=docsify&logoColor=hsl&link=https%3A%2F%2Fgithub.com%2Frotsl%2Fmetrics-petri%2Fblob%2Fmain%2FLICENSE)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/metrics-petri?period=total&units=INTERNATIONAL_SYSTEM&left_color=YELLOW&right_color=RED&left_text=downloads)](https://pepy.tech/projects/metrics-petri)
![Website](https://img.shields.io/website?url=https%3A%2F%2Frotsl.github.io%2Fmetrics-petri%2F&up_message=Documention&up_color=red&logoColor=violet&labelColor=blue&color=green&link=https%3A%2F%2Frotsl.github.io%2Fmetrics-petri%2F)




Macroscopic image analysis for fungal colony growth on petri dishes.

`metrics-petri` measures how a sample expands, whether its edge stays smooth or roughens, when cracks appear, and how centre-to-edge texture evolves over time. It turns a folder of time-series images into physical measurements (mm², mm, day⁻¹) with overlay visualisations.

The repository ships three entry points:

| Entry point | Install | Use |
| --- | --- | --- |
| `metrics-petri` | `pip install metrics-petri` | CLI batch pipeline |
| `metrics-petri-gui` | `pip install "metrics-petri[gui]"` | Gradio browser GUI |
| `metrics-petri-metadata` | `pip install metrics-petri` | Desktop GUI for building `image_metadata.csv` |

---

## Repository setup

### Prerequisites

- Python 3.10 or later
- `make` (standard on macOS/Linux)

### Clone and install

```bash
git clone https://github.com/rotsl/metrics-petri.git
cd metrics-petri
make install        # create venv, install deps, verify model checkpoint
```

To include the Gradio GUI:

```bash
make install-gui
```

`make install` / `make install-gui` each: create a virtual environment, install Python dependencies, and download the UNet checkpoint to `models/best_area_w_0.7.pt` if it is not already present.

### Model checkpoint

The checkpoint `models/best_area_w_0.7.pt` is tracked in this repository and downloaded automatically by `make install`. To fetch it independently:

```bash
make download-model
```

To use a custom checkpoint:

```bash
UNET_MODEL=/path/to/checkpoint.pt make run-cli INPUT=input_images/
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

Output is a single ZIP containing `analysis_full.csv`, `analysis_full.json`, per-image overlays, and growth-curve charts with day codes on the x-axis.

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

This uses the venv created by `make install` and opens `notebooks/example_metrics-petri.ipynb` in JupyterLab. The notebook traces the full pipeline — mask inference, dish detection, crack analysis, and growth metrics — with inline plots at each step.

The notebook is not distributed with the pip package. Clone the repository to use it.

---

## Makefile targets

| Target | Description |
| --- | --- |
| `make install` | Create venv, install deps, download model (no GUI) |
| `make install-gui` | Same, plus Gradio GUI extras |
| `make download-model` | Download UNet checkpoint to `models/` if missing |
| `make model-status` | Check whether the checkpoint is present |
| `make run-gui` | Launch Gradio interface |
| `make run-cli INPUT=path/` | Run batch CLI on a folder |
| `make run-notebook` | Open the example notebook in JupyterLab |
| `make build-package` | Build wheel and sdist for PyPI |
| `make publish-pypi` | Upload to PyPI with twine |
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
├── models/             # UNet checkpoint (tracked in repo)
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

```bibtex
@software{Rohan_R_Metrics_Petri_petri_2026,
author = {{Rohan R}},
title = {{Metrics Petri: petri dish colony segmentation and morphometric analysis}},
url = {https://github.com/rotsl/metrics-petri},
version = {0.0.1},
year = {2026}
}
```

Machine-readable citation: [`CITATION.cff`](CITATION.cff)
