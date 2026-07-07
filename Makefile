# Makefile — metrics-petri

PYTHON       ?= python3
VENV_DIR     := venv
VENV_PYTHON  := $(VENV_DIR)/bin/python
VENV_PIP     := $(VENV_PYTHON) -m pip
VENV_JUPYTER := $(VENV_DIR)/bin/jupyter

NOTEBOOK_PATH := notebooks/example_metrics-petri.ipynb
UNET_MODEL    := metrics_petri/models/best_area_w_0.7.pt
MODEL_URL     := https://huggingface.co/rotsl/grayleafspot-segmentation/resolve/main/best_area_w_0.7.pt
KERNEL_NAME   := metrics-petri
INPUT         ?= input_images/

.PHONY: all setup download-model install install-gui install-release-tools register-kernel \
        model-status run-gui run-cli run-notebook run-lab \
        build-package publish-testpypi publish-pypi clean

all: install

# ── environment ────────────────────────────────────────────────────────────────

setup:
	mkdir -p input_images outputs archives metrics_petri/models .mplconfig
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	$(VENV_PIP) install --upgrade pip setuptools wheel
	@echo "Virtual environment ready at $(VENV_DIR)"

# ── model checkpoint ───────────────────────────────────────────────────────────

download-model:
	@mkdir -p metrics_petri/models
	@if [ -f "$(UNET_MODEL)" ]; then \
		echo "  ✓  Model present: $(UNET_MODEL)"; \
	else \
		echo "  ↓  Downloading UNet checkpoint from HuggingFace…"; \
		curl -fL --progress-bar -o "$(UNET_MODEL)" "$(MODEL_URL)"; \
		echo "  ✓  Saved to $(UNET_MODEL)"; \
	fi
	@$(PYTHON) -c "from pathlib import Path; from metrics_petri._paths import _verify_model_checksum; _verify_model_checksum(Path('$(UNET_MODEL)'))"
	@echo "  ✓  SHA-256 verified"

model-status:
	@echo "UNet checkpoint: $(UNET_MODEL)"
	@if [ -f "$(UNET_MODEL)" ]; then echo "  ✓  present"; else echo "  ✗  MISSING — run: make download-model"; fi

# ── install ────────────────────────────────────────────────────────────────────

install: setup download-model
	$(VENV_PIP) install -e .
	$(MAKE) register-kernel

install-gui: setup download-model
	$(VENV_PIP) install -e ".[gui]"
	$(MAKE) register-kernel

install-release-tools: setup
	$(VENV_PIP) install build twine

register-kernel: setup
	$(VENV_PYTHON) -m ipykernel install --user \
		--name "$(KERNEL_NAME)" \
		--display-name "Python ($(KERNEL_NAME))"

# ── run ────────────────────────────────────────────────────────────────────────

run-gui: install-gui
	UNET_MODEL=$(UNET_MODEL) $(VENV_DIR)/bin/metrics-petri-gui

run-cli: install
	@test -d "$(INPUT)" || (echo "Usage: make run-cli INPUT=path/to/images/" && exit 1)
	UNET_MODEL=$(UNET_MODEL) $(VENV_DIR)/bin/metrics-petri "$(INPUT)"

run-notebook: install
	UNET_MODEL=$(UNET_MODEL) $(VENV_JUPYTER) lab $(NOTEBOOK_PATH)

run-lab: run-notebook

# ── build & publish ────────────────────────────────────────────────────────────

build-package: install-release-tools
	$(VENV_PYTHON) -m build

publish-testpypi: install-release-tools build-package
	$(VENV_PYTHON) -m twine upload --repository testpypi dist/*.whl dist/*.tar.gz

publish-pypi: install-release-tools build-package
	$(VENV_PYTHON) -m twine upload dist/*.whl dist/*.tar.gz

# ── clean ──────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mplconfig dist build *.egg-info
	rm -rf $(VENV_DIR)
