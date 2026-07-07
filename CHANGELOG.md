# Changelog

Notable changes to Metrics Petri are documented here.

## Unreleased

No unreleased changes yet.

## 3.0.0 - 2026-07-07

### Added

- Convert the project into the importable `metrics_petri` Python package.
- Add console scripts for the batch CLI, Gradio GUI, metadata helper, and dish
  cropper.
- Add the dish-cropper utility for extracting individual plates from multi-dish
  images.
- Add package data for the bundled SmallUNet checkpoint so installed wheels can run
  without a separate model file in most environments.
- Add a `--seed` CLI flag and record the configured PyTorch seed in provenance.
- Add automatic CUDA selection ahead of Apple MPS and CPU fallbacks.
- Add structural validation for CSV and JSON experiment metadata.
- Add three example images and pipeline-level smoke tests for real-image CLI
  behaviour and output archive consistency.
- Add regression tests for model loading, checkpoint integrity, GUI defaults,
  metadata paths, metadata schemas, texture metrics, device selection, metadata
  day-code handling, model architecture wiring, device constants, and compatibility
  re-exports.
- Add CodeQL, Bandit, mypy, Pyright, coverage, check-manifest, and
  `metrics-petri doctor` checks to the local and CI validation path.
- Add CI, PyPI publishing, and GitHub Pages workflows.
- Add MkDocs Material documentation with a docstring-generated API reference.
- Add the project logo and expanded project and package documentation.
- Add a SmallUNet model card covering training provenance, validation metrics,
  intended use, and limitations.
- Add a security policy, contribution guide, and Contributor Covenant code of conduct.
- Add GitHub build-provenance attestations for release wheel and source distribution
  artifacts.

### Changed

- Move pipeline modules under the `metrics_petri` namespace while preserving
  end-user CLI entry points.
- Rework package metadata and dependency declarations in `pyproject.toml`.
- Use `pyproject.toml` as the single source of dependency declarations.
- Simplify local installation through `requirements.txt` and Makefile targets.
- Update documentation for packaged CLI, GUI, notebook, metadata, and dish-cropper
  workflows.
- Refresh README and package entry-point instructions.
- Document the manual changelog policy.
- Update package metadata, README content, and the PyPI long description for the
  broader metrics-petri workflow.
- Calculate entropy and texture standard deviation from colony pixels rather than
  the entire photograph. Values in these columns are not directly comparable with
  results from version 2.1.0 and earlier.
- Resolve the deprecated `MODEL_PATH` compatibility attribute lazily instead of
  inspecting model paths during module import.
- Route model-resolution diagnostics through logging while preserving user-facing CLI
  progress output.
- Remove the obsolete Streamlit metadata UI module and update the legacy pipeline
  metadata error message to direct users to `metrics-petri-metadata`.
- Run mypy in CI using the active workflow Python version so dependency stubs are
  parsed with matching syntax support.
- Install GUI extras during CI and publish validation so Pyright can resolve optional
  GUI and HEIF imports.
- Build releases from the checked-in package version and reject mismatched release
  tags.
- Validate built wheel and source distributions with strict Twine checks before
  upload.
- Run the complete validation gate before publishing.
- Retain built distributions as workflow artifacts.
- Switch the PyPI release workflow to Trusted Publishing.
- Include the checkpoint, checksum, model card, governance files, and documentation
  explicitly in source distributions.
- Add major-version upper bounds for Pillow, pandas, SciPy, OpenCV, Matplotlib,
  rawpy, Hugging Face Hub, Torch, NumPy, and scikit-image to reduce untested
  dependency drift.
- Tighten NumPy, SciPy, and OpenCV dependency bounds to avoid incompatible NumPy 2.x
  resolver combinations with current PyTorch wheels.
- Restrict supported Python versions to releases below Python 3.14 so installers do
  not select environments where compatible PyTorch wheels are unavailable.
- Report stale or mismatched metadata image paths clearly instead of falling through
  to a generic "No images found" error.

### Security

- Load PyTorch checkpoints with `weights_only=True`.
- Verify both bundled and downloaded model checkpoints against their published
  SHA-256 checksums.
- Pin the fallback model-download revision while retaining runtime checkpoint
  SHA-256 verification.
- Bind the Gradio GUI to loopback by default.
- Add optional Gradio authentication.
- Refuse unauthenticated Gradio GUI launches on non-loopback hosts.
- Warn users about unauthenticated network exposure.
- Remove the remotely accessible in-browser process shutdown control.
- Restrict metadata-referenced images to the selected input directory.
- Pin all GitHub Actions to immutable commit SHAs.

### Removed

- Remove the obsolete Streamlit metadata UI module.
- Remove the unused root `metadata.json` file.
- Remove process-wide suppression of dependency `FutureWarning` messages.
- Delete PyPI package releases older than version 2.0.0.
- Retain version 2.0.0 on PyPI as a yanked release.
- Delete repository tags and GitHub releases for versions older than 2.0.0.
