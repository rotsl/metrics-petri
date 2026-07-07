# Changelog

Notable changes to Metrics Petri are documented here.

## Unreleased

### Added

- Add CodeQL, Bandit, mypy, Pyright, coverage, check-manifest, and
  `metrics-petri doctor` checks to the local and CI validation path.
- Add a `--seed` CLI flag and record the configured PyTorch seed in provenance.
- Add three example images plus pipeline-level smoke tests for real-image CLI
  behaviour and output archive consistency.

### Security

- Refuse unauthenticated Gradio GUI launches on non-loopback hosts.
- Pin the fallback model-download revision and keep checkpoint SHA-256 verification
  in the runtime path.

### Changed

- Add major-version upper bounds for Pillow, pandas, SciPy, OpenCV, Matplotlib,
  rawpy, and Hugging Face Hub to reduce untested dependency drift.
- Remove the obsolete Streamlit metadata UI module and update the legacy pipeline
  metadata error message to point users at `metrics-petri-metadata`.
- Route model-resolution diagnostics through logging while preserving user-facing CLI
  progress output.
- Document the manual changelog policy and refresh README/package entry-point
  instructions.

## 2.1.2 - 2026-07-07

### Security

- Load PyTorch checkpoints with `weights_only=True`.
- Verify the bundled and downloaded model checkpoint against its published SHA-256.
- Bind the Gradio GUI to loopback by default, add optional authentication, and warn
  about unauthenticated network exposure.
- Remove the remotely accessible in-browser process shutdown control.
- Confine metadata-referenced images to the selected input directory.
- Pin all GitHub Actions to immutable commit SHAs.

### Added

- Security policy, contribution guide, and Contributor Covenant code of conduct.
- SmallUNet model card with training provenance, validation metrics, intended use, and
  limitations.
- MkDocs Material documentation with a docstring-generated API reference.
- Structural validation for CSV and JSON experiment metadata.
- Regression tests for model loading, integrity checks, GUI defaults, metadata paths,
  metadata schemas, texture metrics, and device selection.
- Automatic CUDA selection ahead of Apple MPS and CPU fallbacks.

### Changed

- Calculate entropy and texture standard deviation from colony pixels rather than the
  whole photograph. Values in these two columns are not directly comparable with
  results from version 2.1.0 and earlier.
- Resolve the deprecated `MODEL_PATH` compatibility attribute lazily instead of
  inspecting model paths during module import.
- Use `pyproject.toml` as the single source of dependency declarations.
- Build releases from the checked-in package version and reject mismatched release tags.
- Validate built wheel and source distributions with strict Twine checks before upload.
- Include the checkpoint, checksum, model card, governance files, and documentation
  explicitly in source distributions.
- Add upper bounds for Torch, NumPy, and scikit-image to avoid untested major-version
  upgrades.

### Removed

- Unused root `metadata.json` file.
- Process-wide suppression of dependency `FutureWarning` messages.

## 2.1.0 - 2026-07-03

### Added

- Converted the project into the importable `metrics_petri` Python package with console
  scripts for the batch CLI, Gradio GUI, metadata helper, and dish cropper.
- Added package data for the bundled SmallUNet checkpoint so installed wheels can run
  without a separate model file in most environments.
- Added initial pytest coverage for metadata day-code handling, model architecture
  wiring, device constants, and compatibility re-exports.
- Added CI, PyPI publishing, and GitHub Pages workflow files.

### Changed

- Moved pipeline modules under the `metrics_petri` namespace while keeping CLI entry
  points for end users.
- Reworked package metadata and dependency declarations in `pyproject.toml`.
- Simplified local installation through `requirements.txt` and Makefile targets.
- Updated documentation for packaged CLI, GUI, notebook, and metadata workflows.

## 2.0.0 - 2026-07-03

### Added

- Added the project logo and expanded project/package documentation.
- Added the dish-cropper utility for extracting individual plates from multi-dish
  images.
- Added MkDocs/GitHub Pages deployment groundwork for hosted documentation.

### Changed

- Updated package metadata, README, and PyPI long description for the broader
  `metrics-petri` workflow.
- Refined metadata GUI documentation and example workflow descriptions.
- Prepared the package for the later namespace-package layout introduced in 2.1.0.
