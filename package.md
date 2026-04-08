# Packaging and Development Guide

This document describes how the `grayleafspot` repository is structured for both
development and packaged distribution, and what each maintainer action does.

---

## Repository layout

```
grayleafspot/              ← root Python package (the runnable launcher)
    __init__.py            ← package marker, exposes __version__
    __main__.py            ← enables `python -m grayleafspot`
    cli.py                 ← parses CLI arguments, calls launcher
    launcher.py            ← starts the static file server + Node API subprocess
    assets.py              ← locates bundled frontend assets at runtime
    dist/                  ← built frontend assets (gitignored; populated at release time)
pipeline/                  ← Python analysis pipeline
    __init__.py
    analysis.py
    cli.py
    reporting.py
server/
    index.ts               ← Node/Express API (development entrypoint)
src/                       ← React/TypeScript frontend source (Vite project)
pyproject.toml             ← Python package definition (pip-installable)
MANIFEST.in                ← tells sdist to include non-Python assets
Makefile                   ← all developer and release tasks
package.json               ← Node project definition (frontend + API dev deps)
vite.config.ts             ← Vite build config; output directory is dist/
```

---

## 1. Root Python package as the runnable launcher

`grayleafspot/` (at the repo root) is the Python package that end-users install.
After `pip install .`, the following command starts the app:

```bash
grayleafspot
```

The entry point is declared in `pyproject.toml`:

```toml
[project.scripts]
grayleafspot = "grayleafspot.cli:main"
```

The launcher can also be invoked as a module:

```bash
python -m grayleafspot
```

### What the launcher does

1. Checks that bundled frontend assets exist in `grayleafspot/dist/` (see §4).
2. Starts the Node/Express API as a subprocess (by default on port 8000).
3. Serves the static frontend from `grayleafspot/dist/` (by default on port 3000).
4. Opens the browser automatically.
5. Shuts everything down cleanly on `Ctrl-C`.

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address for the static file server |
| `--port` | `3000` | Port for the static file server |
| `--api-port` | `8000` | Port passed to the Node API as `API_PORT` |
| `--no-browser` | — | Do not open the browser automatically |
| `--skip-node-api` | — | Skip starting the Node API (static frontend only) |

---

## 2. Node server: dev vs. packaged mode

### Development (normal workflow)

The Node/Express API lives in `server/index.ts` and is always the API backend.
The Vite dev server proxies `/api`, `/input_images`, and `/outputs` to it.

Start everything for development:

```bash
make run-app
```

This runs the Node API and the Vite dev server in parallel. The Node API is started
with `npm run api` (`node --import tsx server/index.ts`), listening on port 8000.
The Vite dev server listens on port 3000 with hot-module replacement enabled.

You can also start them separately:

```bash
make run-api        # Node API only (port 8000)
make run-frontend   # Vite dev server only (port 3000)
```

### Packaged mode

When the app is installed via `pip install .` and run with `grayleafspot`, the
Python launcher spawns the Node API as a subprocess using `npm run api`, passing
the chosen port via the `API_PORT` environment variable.

This requires `npm` and Node dependencies to be available in the environment
where the installed app is run. The launcher passes the current working directory
to the Node process, so `input_images/`, `outputs/`, and `archives/` are resolved
relative to the directory from which `grayleafspot` is launched.

If you want to run the packaged app without the Node API (e.g. for
frontend-only testing), use:

```bash
grayleafspot --skip-node-api
```

### Files involved

| File | Purpose |
|---|---|
| `server/index.ts` | Node/Express API — do not rename or move |
| `grayleafspot/launcher.py` | starts Node via `subprocess.Popen(["npm", "run", "api"])` |
| `package.json` `scripts.api` | `node --import tsx server/index.ts` |

---

## 3. Frontend source stays in `src/`

All React/TypeScript source code lives in `src/`. Do not move it.

The Vite project is configured via `vite.config.ts`. The production build output
directory is `dist/` (the Vite default). This is the directory that gets bundled
into the Python package during a release build (see §4).

The `src/` directory is included in source distributions via `MANIFEST.in`:

```
recursive-include src *
```

This lets maintainers inspect the original source when working from an sdist.
It does **not** affect what is served at runtime — only the compiled `dist/` output
inside the Python package is served.

---

## 4. Copying built frontend into the Python package at release time

The Python package (`grayleafspot/`) must contain compiled frontend assets to be
runnable after `pip install .`. These assets are **not committed to Git**
(`.gitignore` excludes `dist`). They are produced at release time and copied in.

### How it works

The `Makefile` has three chained targets:

```makefile
build-frontend: npm-deps
    npm run build                # runs vite build → produces dist/

bundle-frontend: build-frontend
    rm -rf grayleafspot/dist
    mkdir -p grayleafspot/dist
    cp -R dist/* grayleafspot/dist/   # copies dist/ into the Python package

build-package: bundle-frontend
    $(VENV_PYTHON) -m build      # creates wheel + sdist under dist-build/
```

Run the full release build:

```bash
make build-package
```

This produces a wheel (`.whl`) and source distribution (`.tar.gz`) that include
the compiled frontend assets.

### How Python packaging picks up the assets

`pyproject.toml` declares:

```toml
[tool.setuptools]
include-package-data = true

[tool.setuptools.package-data]
grayleafspot = ["dist/**/*"]
```

`MANIFEST.in` declares:

```
recursive-include grayleafspot/dist *
```

Together these ensure that the contents of `grayleafspot/dist/` are included in
both the wheel and the sdist.

### How the launcher finds the assets at runtime

`grayleafspot/assets.py` locates the assets relative to the installed package:

```python
def get_assets_dir() -> Path:
    return Path(__file__).resolve().parent / "dist"
```

This works whether the package was installed into a virtualenv, with `pip install .`,
or run in-place from the repo root (after `make bundle-frontend`).

---

## 5. Developer workflow summary

### First-time setup

```bash
make install-all    # creates venv, installs Python deps, installs Node deps
```

### Day-to-day development

```bash
make run-app        # starts Node API + Vite dev server with HMR
```

### Run analysis pipeline directly

```bash
make analyze-local  # local engine (requires models)
make analyze-gemini # Gemini engine (requires GEMINI_API_KEY in .env)
```

### Jupyter notebooks

```bash
make run-lab        # starts JupyterLab with the correct kernel
```

---

## 6. Release / packaging workflow

```bash
# 1. Ensure all Python and Node dependencies are installed.
make install-all

# 2. Remove any previous build artifacts to avoid stale files.
rm -rf dist/ grayleafspot/dist/ *.egg-info

# 3. Build the frontend, copy assets into the Python package, then build the wheel.
make build-package

# 4. Install and test the wheel locally (replace VERSION with the actual version).
pip install "dist/grayleafspot-0.1.0-py3-none-any.whl"
grayleafspot

# 5. Upload to PyPI (maintainers only — upload only the files for this release).
python -m twine upload dist/grayleafspot-0.1.0-py3-none-any.whl dist/grayleafspot-0.1.0.tar.gz
```

> **Note:** `make build-package` assumes `python -m build` is available.
> Install it once with `pip install build` if it is not already present.
>
> Always clean `dist/` before a new release build so that stale wheels or sdists
> from previous versions are not accidentally installed or uploaded.

---

## 7. Key invariants to preserve

| Rule | Why |
|---|---|
| Do not move `server/index.ts` | `launcher.py` runs `npm run api`; `package.json` points to `server/index.ts` |
| Do not rename `grayleafspot/` | `pyproject.toml` entry point and `packages.find` both reference it by name |
| Keep Vite output in `dist/` | `Makefile` copies from `dist/*`; changing the output dir breaks bundling |
| Do not commit `grayleafspot/dist/` or `dist/` | They are build artifacts; the Makefile regenerates them |
| Keep `grayleafspot/dist/**/*` in `package-data` | Required for the wheel to include assets |
