#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[run] Virtual environment not found. Run: bash setup_env.sh"
  exit 1
fi

source "${VENV_DIR}/bin/activate"

mkdir -p "${ROOT_DIR}/results" "${ROOT_DIR}/report/figures"
mkdir -p "${ROOT_DIR}/.mplconfig"
mkdir -p "${ROOT_DIR}/.texlive/texmf-var" "${ROOT_DIR}/.texlive/texmf-config" "${ROOT_DIR}/.texlive/texmf-home"
export MPLCONFIGDIR="${ROOT_DIR}/.mplconfig"
export MPLBACKEND="Agg"
export TEXMFVAR="${ROOT_DIR}/.texlive/texmf-var"
export TEXMFCONFIG="${ROOT_DIR}/.texlive/texmf-config"
export TEXMFHOME="${ROOT_DIR}/.texlive/texmf-home"

python "${ROOT_DIR}/src/evaluate.py" \
  --pred-dir "${ROOT_DIR}/data/predictions/annotations" \
  --gt-dir "${ROOT_DIR}/data/ground_truth/annotations" \
  --metrics-out "${ROOT_DIR}/results/metrics.csv" \
  --summary-out "${ROOT_DIR}/results/summary.json" \
  --report-template "${ROOT_DIR}/report/report.tex" \
  --report-out "${ROOT_DIR}/report/report_filled.tex" \
  --figures-dir "${ROOT_DIR}/report/figures" \
  --filename-contains "TOP"

PDF_COMPILER=""
if [[ -x "${VENV_DIR}/bin/pdflatex" ]]; then
  PDF_COMPILER="${VENV_DIR}/bin/pdflatex"
elif [[ -x "${VENV_DIR}/bin/tectonic" ]]; then
  PDF_COMPILER="${VENV_DIR}/bin/tectonic"
fi

if [[ -z "${PDF_COMPILER}" ]]; then
  echo "[run] No PDF compiler is available inside ${VENV_DIR}. Run bash setup_env.sh first."
  exit 1
fi

cd "${ROOT_DIR}/report"
if [[ "${PDF_COMPILER}" == *tectonic ]]; then
  "${PDF_COMPILER}" report_filled.tex
else
  "${PDF_COMPILER}" -interaction=nonstopmode -halt-on-error report_filled.tex
  "${PDF_COMPILER}" -interaction=nonstopmode -halt-on-error report_filled.tex
fi

cp -f report_filled.pdf report.pdf
echo "[run] Evaluation complete."
echo "[run] Metrics: ${ROOT_DIR}/results/metrics.csv"
echo "[run] Summary: ${ROOT_DIR}/results/summary.json"
echo "[run] PDF: ${ROOT_DIR}/report/report.pdf"
