# Makefile for Magnaporthe Growth Analyzer

PYTHON ?= python3.13
VENV_DIR := venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_PYTHON) -m pip
VENV_JUPYTER := $(VENV_DIR)/bin/jupyter
VENV_HF := $(VENV_DIR)/bin/hf
NOTEBOOK_PATH := notebooks/magnaporthe_pipeline_step_by_step.ipynb
HF_CACHE_DIR := models/gemma-4-e2b-it-MLX-4bit
HF_MODEL_ID := FakeRockert543/gemma-4-e2b-it-MLX-4bit
UNET_MODEL_PATH := models/best_unet.pt
SAM_CHECKPOINT_PATH := models/sam_vit_b_01ec64.pth
SAM_CHECKPOINT_URL := https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
KERNEL_NAME := magnaporthe-growth-analyzer
REPORT_BASE_DIR ?= outputs

.PHONY: all setup install install-all install-hf-cli register-kernel download-models download-gemma-model download-sam-model check-local-models model-status npm-deps clean run-notebook run-lab run-api run-frontend run-app analyze-local analyze-gemini report-run

all: install

setup:
	mkdir -p input_images outputs archives models .mplconfig
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	$(VENV_PIP) install --upgrade pip setuptools wheel
	@echo "Virtual environment ready at $(VENV_DIR)"

install: setup
	$(VENV_PIP) install -r requirements.txt
	$(MAKE) register-kernel

install-all: install npm-deps

install-hf-cli: setup
	$(VENV_PIP) install "huggingface_hub[cli]"

register-kernel: setup
	$(VENV_PYTHON) -m ipykernel install --user --name "$(KERNEL_NAME)" --display-name "Python ($(KERNEL_NAME))"

download-models: download-gemma-model download-sam-model

download-gemma-model: install-hf-cli
	mkdir -p $(HF_CACHE_DIR)
	$(VENV_HF) download $(HF_MODEL_ID) --local-dir $(HF_CACHE_DIR)

download-sam-model: setup
	mkdir -p $(dir $(SAM_CHECKPOINT_PATH))
	curl -L $(SAM_CHECKPOINT_URL) -o $(SAM_CHECKPOINT_PATH)

check-local-models:
	@test -d $(HF_CACHE_DIR) || (echo "Missing Gemma MLX directory at $(HF_CACHE_DIR)" && exit 1)
	@test -f $(UNET_MODEL_PATH) || (echo "Missing U-Net checkpoint at $(UNET_MODEL_PATH)" && exit 1)
	@test -f $(SAM_CHECKPOINT_PATH) || (echo "Missing SAM checkpoint at $(SAM_CHECKPOINT_PATH)" && exit 1)

model-status:
	@echo "Gemma MLX dir: $(HF_CACHE_DIR)"
	@if [ -d "$(HF_CACHE_DIR)" ]; then echo "  present"; else echo "  missing"; fi
	@echo "U-Net checkpoint: $(UNET_MODEL_PATH)"
	@if [ -f "$(UNET_MODEL_PATH)" ]; then echo "  present"; else echo "  missing"; fi
	@echo "SAM checkpoint: $(SAM_CHECKPOINT_PATH)"
	@if [ -f "$(SAM_CHECKPOINT_PATH)" ]; then echo "  present"; else echo "  missing"; fi
	@echo "Notebook: $(NOTEBOOK_PATH)"

npm-deps:
	npm install

run-api: npm-deps
	npm run api

run-frontend: npm-deps
	npm run dev -- --host 127.0.0.1 --port 3000

run-app: npm-deps
	@trap 'kill 0' INT TERM EXIT; \
	npm run api & \
	npm run dev -- --host 127.0.0.1 --port 3000 & \
	wait

analyze-local: install check-local-models
	$(VENV_PYTHON) -m pipeline.cli --engine local --input-dir input_images --output-dir outputs

analyze-gemini: install
	$(VENV_PYTHON) -m pipeline.cli --engine gemini --input-dir input_images --output-dir outputs

report-run: install
	@test -n "$(RUN_ID)" || (echo "Usage: make report-run RUN_ID=YYYYMMDDTHHMMSSZ_local" && exit 1)
	@test -d "$(REPORT_BASE_DIR)/$(RUN_ID)" || (echo "Missing run directory at $(REPORT_BASE_DIR)/$(RUN_ID)" && exit 1)
	MPLCONFIGDIR=$(PWD)/.mplconfig $(VENV_PYTHON) -m pipeline.reporting --run-dir $(REPORT_BASE_DIR)/$(RUN_ID)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	rm -rf .mplconfig
	rm -rf $(VENV_DIR)

run-notebook: install
	$(VENV_JUPYTER) notebook $(NOTEBOOK_PATH)

run-lab: install
	$(VENV_JUPYTER) lab $(NOTEBOOK_PATH)

build-frontend: npm-deps
	npm run build

bundle-frontend: build-frontend
	rm -rf grayleafspot/dist
	mkdir -p grayleafspot/dist
	cp -R dist/* grayleafspot/dist/

build-package: bundle-frontend
	$(VENV_PYTHON) -m build