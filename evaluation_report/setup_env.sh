#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

echo "[setup] Creating virtual environment at ${VENV_DIR}"
python3 -m venv "${VENV_DIR}"

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${ROOT_DIR}/requirements.txt"

if command -v tectonic >/dev/null 2>&1 && [[ ! -e "${VENV_DIR}/bin/tectonic" ]]; then
  ln -s "$(command -v tectonic)" "${VENV_DIR}/bin/tectonic"
  echo "[setup] Linked tectonic into the venv: ${VENV_DIR}/bin/tectonic"
fi

if command -v pdflatex >/dev/null 2>&1 && [[ ! -e "${VENV_DIR}/bin/pdflatex" ]]; then
  ln -s "$(command -v pdflatex)" "${VENV_DIR}/bin/pdflatex"
  echo "[setup] Linked pdflatex into the venv: ${VENV_DIR}/bin/pdflatex"
fi

if [[ -x "${VENV_DIR}/bin/tectonic" || -x "${VENV_DIR}/bin/pdflatex" ]]; then
  echo "[setup] A PDF compiler is now available inside the venv."
  exit 0
fi

echo "[setup] pdflatex not found. Attempting a platform-specific LaTeX install."
if command -v brew >/dev/null 2>&1; then
  echo "[setup] Installing BasicTeX with Homebrew."
  brew install --cask basictex || true
  export PATH="/Library/TeX/texbin:${PATH}"
  if command -v tlmgr >/dev/null 2>&1; then
    sudo tlmgr update --self || true
    sudo tlmgr install collection-latexrecommended collection-fontsrecommended booktabs geometry hyperref longtable array xcolor || true
  fi
elif command -v apt-get >/dev/null 2>&1; then
  echo "[setup] Installing TeX Live with apt-get."
  sudo apt-get update
  sudo apt-get install -y texlive-latex-base texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra
else
  echo "[setup] No supported package manager found for LaTeX installation."
fi

if command -v pdflatex >/dev/null 2>&1; then
  echo "[setup] pdflatex installed successfully: $(command -v pdflatex)"
else
  echo "[setup] Warning: no PDF compiler is available in the venv. Install LaTeX manually before running run_evaluation.sh."
fi
