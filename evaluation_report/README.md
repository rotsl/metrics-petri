# Evaluation Report

This folder is a self-contained evaluation workspace for validating the final segmentation output of the Magnaporthe growth pipeline.

It assumes you already have:
- predicted binary masks in `data/predictions/`
- ground-truth binary masks in `data/ground_truth/`

The framework computes per-image segmentation metrics, summarizes the dataset statistically, renders figures, fills a LaTeX report, and compiles a PDF.

## Folder layout

```text
evaluation_report/
├── data/
│   ├── predictions/
│   └── ground_truth/
├── report/
│   ├── figures/
│   └── report.tex
├── results/
├── src/
│   ├── evaluate.py
│   ├── metrics.py
│   ├── stats.py
│   └── utils.py
├── requirements.txt
├── run_evaluation.sh
└── setup_env.sh
```

## What gets measured

The evaluation is pixel-wise and binary:
- Precision
- Recall
- F1-score
- IoU

The per-image results are written to `results/metrics.csv`.

The summary file contains:
- mean
- standard deviation
- bootstrap confidence intervals

Those values are written to `results/summary.json`.

## Data expectations

Place one predicted mask and one ground-truth mask per image pair.

Recommended naming:
- `plate_001.png` in `data/predictions/`
- `plate_001.png` in `data/ground_truth/`

The evaluator matches masks by file stem first, then by full filename.

Masks may be:
- grayscale
- RGB
- `0/255`
- boolean-like

They are binarized automatically.

## Setup

```bash
bash setup_env.sh
```

This script:
- creates `.venv/`
- installs Python dependencies
- checks for `pdflatex`
- attempts a platform-specific LaTeX install when possible

On macOS with Homebrew, it tries `basictex`.

On Debian or Ubuntu, it tries TeX Live packages with `apt-get`.

## Run

```bash
bash run_evaluation.sh
```

This will:
1. evaluate all prediction and ground-truth pairs
2. save `results/metrics.csv`
3. save `results/summary.json`
4. render figures into `report/figures/`
5. fill `report/report_filled.tex`
6. compile `report/report.pdf`

## Notes

- The evaluation is aimed at the final segmentation mask, not intermediate masks.
- Empty-mask edge cases are handled explicitly to avoid divide-by-zero failures.
- The generated report includes confusion-matrix and metric-distribution figures when data is available.
