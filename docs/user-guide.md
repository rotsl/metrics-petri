# User guide

## Installation

Metrics Petri requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install metrics-petri
```

For a development checkout, install the package in editable mode:

```bash
pip install -e .
```

## Analyse a folder

```bash
metrics-petri input_images/
```

The command writes a ZIP archive containing tabular results, provenance, overlays,
and growth charts. Common options include:

```bash
metrics-petri input_images/ --output results/run01.zip
metrics-petri input_images/ --metadata input_images/image_metadata.csv
metrics-petri input_images/ --threshold 0.45
metrics-petri input_images/ --dish-size-mm 60
metrics-petri input_images/ --seed 123
```

Supplying metadata enables day-coded charts and growth-rate calculations. The
[batch pipeline guide](batch-pipeline.md) describes the metadata workflow and output
columns.

Physical measurements default to a 90 mm outside dish diameter. Use `--dish-size-mm`
with the actual diameter when analysing another dish size.
The PyTorch seed defaults to `0` and is recorded in `provenance.json`; pass `--seed`
to document a different reproducibility setting.

## Example dataset

The source repository includes three example images in `input_images/06FEB` for
smoke-testing and tutorials:

```bash
metrics-petri input_images/06FEB --output /tmp/metrics-petri-06feb.zip --seed 0
```

Open the ZIP to inspect `analysis_full.csv`, `analysis_full.json`, `provenance.json`,
per-image overlays, and growth charts.

## Crop multi-dish photographs

```bash
metrics-petri-crop -i input_images/
metrics-petri-crop -i photo.jpg --date 06/Feb
metrics-petri-crop -i input_images/ --debug
```

The cropper saves fully visible dishes as individual images in a `cropped/` directory
unless another output location is supplied.

## Model use

The packaged checkpoint segments fungal colony area before morphometric analysis. Read
the [model card](model-card.md) for its training provenance, validation results,
limitations, and out-of-scope uses.

## Further information

The [repository README](https://github.com/rotsl/metrics-petri#readme) includes the full
workflow, notebook instructions, Make targets, measured features, and citation details.
