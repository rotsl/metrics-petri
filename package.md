# metrics-petri

Petri dish colony segmentation and morphometric analysis.

`metrics-petri` measures how a biological sample grows on a petri dish: area, diameter, edge roughness, crack burden, texture entropy, and time-series growth rates — all in physical units calibrated from the dish geometry.

---

## Two entry points, one package

| Entry point | Install | Use |
| --- | --- | --- |
| `metrics-petri` | `pip install metrics-petri` | CLI batch pipeline |
| `metrics-petri-gui` | `pip install "metrics-petri[gui]"` | Gradio browser GUI |

---

## CLI batch pipeline

```bash
pip install metrics-petri
metrics-petri input_images/
```

Processes every image in the folder and writes a ZIP containing:

- `analysis_full.csv` — one row per image, all metrics
- `analysis_full.json` — same data as a JSON array
- `overlays/` — per-image colony mask composites

```bash
# Custom output path
metrics-petri input_images/ --output results/run01.zip

# Supply metadata CSV for growth rate calculations
metrics-petri input_images/ --metadata input_images/image_metadata.csv

# Adjust segmentation threshold (default 0.5)
metrics-petri input_images/ --threshold 0.45

# Use a custom checkpoint
metrics-petri input_images/ --model /path/to/checkpoint.pt
```

---

## Gradio GUI

```bash
pip install "metrics-petri[gui]"
metrics-petri-gui
```

Opens in your browser at `http://localhost:7860`. Five-step tab flow:

1. **Upload images** — JPEG, PNG, TIFF, BMP, WebP, HEIF, RAW
2. **Settings** — threshold, fast mode, experiment name
3. **Review & edit dates** — annotate `experiment_date` / `image_date` per image
4. **Export metadata** — download a `image_metadata.csv` for future runs
5. **Run inference** — segmentation, dish detection, cracks, growth charts, ZIP download

---

## Measured metrics

| Metric | Unit |
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

## Notebook walkthrough

A step-by-step notebook is shipped with the package:

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
- Gradio ≥ 5.0 (GUI only, via `[gui]` extra)

---

## License

Apache 2.0 — [full text](https://github.com/rotsl/metrics-petri/blob/main/LICENSE)

## Citation

```text
Metrics Petri: petri dish colony segmentation and morphometric analysis. 2026.
https://github.com/rotsl/metrics-petri
```
