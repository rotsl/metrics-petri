# Gray Leaf Spot

This repo is a local-first workspace for analyzing *Magnaporthe oryzae* petri-dish images. The same codebase drives the GUI, the CLI, the notebook, the report builder, and the R export.

There are two analysis modes:

- `local`: classical CV + U-Net + SAM, with the optional Gemma MLX prior when that path is available
- `gemini`: cloud inference through a Gemini API key, while keeping the same input and output structure

The repo uses a simple folder layout:

- put source images in [`input_images/`](input_images/)
- active run outputs go to [`outputs/`](outputs/)
- archived runs go to [`archives/`](archives/)

## Layout

```text
.
├── input_images/                         # Source images
├── outputs/                              # Active run outputs
├── archives/                             # Archived run outputs
├── models/                               # Gemma, U-Net, and SAM assets
├── notebooks/
│   └── magnaporthe_pipeline_step_by_step.ipynb
├── pipeline/                             # Shared Python analysis code
├── server/                               # Local API used by the GUI
├── src/                                  # React/Vite frontend
├── Makefile
├── requirements.txt
├── package.json
├── LICENSE
└── README.md
```

## What it does

- Detects the petri dish first and calibrates measurements against an assumed 90 mm dish
- Builds a final colony mask from classical segmentation, U-Net, hybrid mask selection, and SAM refinement
- Keeps masks inside the detected dish
- Writes morphology, texture, crack, radial-profile, and growth metrics in millimetres
- Lets you inspect saved runs in the GUI, archive them, clear them, and generate reports
- Exports a run-specific R script for RStudio
- Ships with a notebook that traces the same pipeline stage by stage

## Setup

### 1. Node and Python

Install the Node side:

```bash
npm install
```

Create the project venv, install the Python dependencies, and register the notebook kernel:

```bash
make install
```

If you want one command that does both:

```bash
make install-all
```

`make install` reuses `venv/` if it already exists.

### 2. Environment

Copy [`.env.example`](.env.example) to `.env` and fill in what you need.

The main keys are:

- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `LOCAL_MODEL_DIR`
- `LOCAL_MODEL_ID`
- `LOCAL_ENABLE_MLX`
- `LOCAL_UNET_PATH`
- `LOCAL_SAM_CHECKPOINT`
- `LOCAL_SAM_MODEL_TYPE`
- `API_PORT`

The GUI can also take a Gemini API key directly in the browser. If the user enters one there, it overrides `.env` for that request.

## Model files

The local pipeline expects three model assets.

### Gemma MLX prior

Expected path:

- [`models/gemma-4-e2b-it-MLX-4bit`](models/gemma-4-e2b-it-MLX-4bit)

Default repo ID:

- `FakeRockert543/gemma-4-e2b-it-MLX-4bit`

Download it with:

```bash
make download-gemma-model
```

### SAM checkpoint

Expected path:

- [`models/sam_vit_b_01ec64.pth`](models/sam_vit_b_01ec64.pth)

Download it with:

```bash
make download-sam-model
```

### U-Net checkpoint

Expected path:

- [`models/best_unet.pt`](models/best_unet.pt)

This file is project-specific. The repo does not download it for you. Put your trained checkpoint at that exact path.

### Helpful model commands

Download the public assets:

```bash
make download-models
```

Check what is present:

```bash
make model-status
```

## Run the app

To start the backend and GUI together:

```bash
make run-app
```

That target runs `npm install` first, then starts:

- the API on `http://127.0.0.1:8000`
- the frontend on `http://127.0.0.1:3000`

You can still start them separately if you want:

```bash
make run-api
make run-frontend
```

Open the GUI at:

```text
http://127.0.0.1:3000
```

## CLI

Run the local path:

```bash
make analyze-local
```

Run the Gemini path:

```bash
make analyze-gemini
```

Generate a report for an existing run:

```bash
make report-run RUN_ID=YYYYMMDDTHHMMSSZ_local
```

Generate a report for an archived run:

```bash
make report-run REPORT_BASE_DIR=archives RUN_ID=YYYYMMDDTHHMMSSZ_local
```

## Notebook

Open the notebook in the classic notebook UI:

```bash
make run-notebook
```

Or open it in JupyterLab:

```bash
make run-lab
```

The notebook lives at [`notebooks/magnaporthe_pipeline_step_by_step.ipynb`](notebooks/magnaporthe_pipeline_step_by_step.ipynb).

It now mirrors the GUI pipeline more closely:

- upload images into `input_images/` from the notebook
- choose multiple images from `input_images/`
- switch between `local` and `gemini`
- trace every selected image through dish detection, prior generation, classical segmentation, U-Net, hybrid selection, SAM, crack analysis, and final outputs
- finish by calling the real shared `run_analysis_batch(...)` entrypoint so the saved files still come from the same pipeline as the GUI

The notebook also places the formulas next to the stage where they are used, instead of collecting them all in one block at the top.

## Local pipeline summary

For each image, the local path does this:

1. Detect the dish from grayscale structure
2. Build the prompt and optional Gemma prior
3. Generate the classical mask
4. Generate the U-Net mask
5. Compare classical and U-Net support with the step-8-style rules
6. Blend in the prior when it helps
7. Refine with SAM
8. Run the mask stability guard
9. Measure cracks from the internal band
10. Export the final mm-based results

## GUI notes

The GUI can:

- load images from `input_images/`
- upload new images into `input_images/`
- run `local` or `gemini`
- show progress and backend logs
- reload past runs from `outputs/`
- archive or clear runs
- generate markdown/PDF reports
- export an R script for RStudio

If the GUI and API ever get out of sync, restart `make run-app` and refresh the browser.

## RStudio export

The `R-STUDIO` button in the GUI downloads a self-contained `.R` script with the run data embedded in it.

The generated script writes:

- `mgp_analysis_results.csv`
- `plots/*.png`
- `plots/plate_feature_heatmaps/*.png`
- `plots/radial_profiles/*.png`
- `README_RStudio.txt`

### Run the exported R script

Open the downloaded `.R` file in RStudio and install the packages once if needed:

```r
install.packages(c(
  "ggplot2",
  "dplyr",
  "tidyr",
  "readr",
  "jsonlite",
  "purrr",
  "stringr",
  "tibble"
))
```

Then run the script with:

```r
source("path/to/downloaded_run_script.R")
```

The export is `ggplot2`-based. It does not rely on `matplotlib`.

## Repo hygiene

Generated content and local caches are ignored on purpose, including:

- `venv/`
- `node_modules/`
- `dist/`
- `outputs/`
- `archives/`
- `rstudio_export_*`
- notebook checkpoints and Python caches

The ignore rules are split between [`.gitignore`](.gitignore) and [`.ignore`](.ignore).

## License

Apache License 2.0. See [`LICENSE`](LICENSE).

Copyright 2026 Rohan R.
