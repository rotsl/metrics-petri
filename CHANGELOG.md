# Changelog

Notable changes to Metrics Petri are documented here.

## Unreleased

No unreleased changes yet.

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

Historical change details were not recorded for this release.

## 2.0.0 - 2026-07-03

Historical change details were not recorded for this release.
