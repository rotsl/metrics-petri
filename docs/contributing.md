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

## Validation

Run the test suite from the repository root:

```bash
MPLCONFIGDIR=.mplconfig venv/bin/python -m pytest
```

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

## Pull requests

Include:

- a concise description of the problem and solution;
- links to related issues;
- the validation commands run and their results;
- screenshots or example outputs for visible changes; and
- notes about compatibility, model, data, or licensing implications.

Maintainers may request changes to keep the package reliable, reproducible, and within
scope. Reviews should be respectful, specific, and technically grounded.
