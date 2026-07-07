# Contributing

Thank you for helping improve Metrics Petri. Bug reports, documentation fixes, tests,
and focused code changes are all welcome.

By participating, you agree to follow the [Code of Conduct](code-of-conduct.md).
Potential vulnerabilities must be reported privately as described in the
[Security Policy](security.md), not through a public issue.

## Before starting

- Search the existing [issues](https://github.com/rotsl/metrics-petri/issues) before
  opening a new one.
- Open an issue before a large feature, architectural change, new dependency, or model
  replacement so the scope can be agreed first.
- Keep pull requests focused. Unrelated refactors make scientific changes harder to
  review and reproduce.
- Do not commit private experiment data, credentials, generated outputs, virtual
  environments, or editor files.

## Development setup

Metrics Petri supports Python 3.10 through 3.13.

```bash
git clone https://github.com/rotsl/metrics-petri.git
cd metrics-petri
python3 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -e ".[test,docs]"
```

The model checkpoint is tracked in the repository. If it is missing, restore it with:

```bash
make download-model
```

## Making changes

- Follow the existing module structure and naming conventions.
- Prefer small changes that preserve public behaviour and command-line compatibility.
- Add `SPDX-License-Identifier: MIT` near the top of new Python files.
- Add or update tests for behaviour changes and bug fixes.
- Update user documentation, API docstrings, and the changelog when they are affected.
- Keep model and dataset provenance explicit. Model changes must update the model card,
  validation metrics, licensing information, and packaged artifacts as appropriate.

Avoid committing large datasets or new checkpoints without prior discussion. Training
data contributions must include their source, licence, annotation method, and any usage
restrictions.

## Changelog policy

User-visible changes should update `CHANGELOG.md` in the same pull request. This
project currently keeps a manual changelog instead of using towncrier or reno because
the release volume is small and direct review keeps the release notes readable. If
changelog drift becomes common, revisit fragment-per-PR tooling before the next minor
release.

## Validation

Run the test suite from the repository root:

```bash
MPLCONFIGDIR=.mplconfig venv/bin/python -m pytest --cov=metrics_petri --cov-report=term-missing
```

Run static checks:

```bash
venv/bin/python -m ruff check .
venv/bin/python -m mypy --python-version 3.13 metrics_petri
venv/bin/pyright
venv/bin/bandit -c pyproject.toml -r metrics_petri
venv/bin/check-manifest --no-build-isolation
venv/bin/metrics-petri doctor
```

Use the Python minor version matching your active virtual environment for local mypy
runs; CI checks mypy across Python 3.10 through 3.13.

For documentation changes, build the site in strict mode:

```bash
venv/bin/mkdocs build --strict
```

For packaging changes, verify the source distribution and wheel:

```bash
venv/bin/python -m build
```

All relevant checks should pass before a pull request is submitted. If a check cannot
run in your environment, explain why in the pull request.

## Release checklist

Release metadata must stay in sync across the package, citation files, and user-facing
documentation. Use the configured version-bump helper instead of editing one file by
hand:

```bash
venv/bin/python -m pip install -e ".[release]"
venv/bin/bump-my-version bump patch
```

Before publishing a GitHub Release:

- verify `metrics_petri/__init__.py`, `CITATION.cff`, `README.md`, and `package.md`
  all contain the new version;
- run tests, strict docs build, and package build checks;
- include the official model checkpoint SHA-256 from
  `metrics_petri/models/best_area_w_0.7.pt.sha256` in the GitHub Release notes;
- confirm PyPI Trusted Publishing is configured for `rotsl/metrics-petri` and
  `.github/workflows/publish.yml`;
- confirm the release workflow uploads the built wheel and source distribution
  artifacts;
- confirm GitHub build-provenance attestations are generated for `dist/*`; and
- confirm the release tag matches `metrics_petri.__version__`.

## Pull requests

Include:

- a concise description of the problem and solution;
- links to related issues;
- the validation commands run and their results;
- screenshots or example outputs for visible changes; and
- notes about compatibility, model, data, or licensing implications.

Maintainers may request changes to keep the package reliable, reproducible, and within
scope. Reviews should be respectful, specific, and technically grounded.
