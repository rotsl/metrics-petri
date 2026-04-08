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

This section covers building a combined package (bundled React frontend + Python
backend in a single wheel), publishing to TestPyPI for validation, and then
publishing the final release to PyPI.

### 6.1 Prerequisites for maintainers

Install the Python build and upload tools once into your release environment:

```bash
pip install build twine
```

You will also need a PyPI account (https://pypi.org/account/register/) and a
TestPyPI account (https://test.pypi.org/account/register/).  Create API tokens
for both under **Account Settings -> API tokens** and store them in
`~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-<your-pypi-api-token>

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-<your-testpypi-api-token>
```

Alternatively you can pass the token on the command line with
`--username __token__ --password pypi-<token>` instead of using `~/.pypirc`.

### 6.2 Bump the version

Edit the version string in **`pyproject.toml`** before every release:

```toml
[project]
version = "0.1.1"   # <- change this
```

### 6.3 Build the combined package

The wheel must contain the compiled React frontend.  The `Makefile` chains the
three necessary steps automatically:

```
build-frontend  ->  builds React/Vite assets into dist/
bundle-frontend ->  copies dist/* into grayleafspot/dist/
build-package   ->  runs `python -m build` -> produces dist-build/
```

Run everything with:

```bash
# 1. Ensure all Python and Node dependencies are installed.
make install-all

# 2. Remove any previous build artifacts to avoid stale files.
rm -rf dist/ dist-build/ grayleafspot/dist/ *.egg-info

# 3. Build the frontend, copy assets, and produce the wheel + sdist.
make build-package
```

After `make build-package` completes you will find two files under `dist-build/`
(the output directory used by `python -m build`):

```
dist-build/
    grayleafspot-0.1.1-py3-none-any.whl   <- installable wheel
    grayleafspot-0.1.1.tar.gz             <- source distribution
```

> **Note:** `make build-package` assumes `python -m build` is available.
> Install it once with `pip install build` if it is not already present.
>
> Always clean `dist/`, `dist-build/`, and `grayleafspot/dist/` before a new
> release build so that stale wheels or sdists from previous versions are not
> accidentally installed or uploaded.

### 6.4 Smoke-test the wheel locally

Before publishing, install the wheel into a clean temporary environment and
confirm the launcher starts:

```bash
python -m venv /tmp/test-grayleafspot
source /tmp/test-grayleafspot/bin/activate
pip install dist-build/grayleafspot-0.1.1-py3-none-any.whl
grayleafspot --help           # should print CLI options
grayleafspot --skip-node-api  # starts the static file server without Node
deactivate
```

### 6.5 Publish to TestPyPI

Upload both the wheel and the sdist to TestPyPI first:

```bash
python -m twine upload --repository testpypi \
    dist-build/grayleafspot-0.1.1-py3-none-any.whl \
    dist-build/grayleafspot-0.1.1.tar.gz
```

Verify the upload at `https://test.pypi.org/project/grayleafspot/`.

### 6.6 Test the TestPyPI install end-to-end

Install the package from TestPyPI into a clean environment.  Because TestPyPI
does not mirror all of PyPI, pass `--extra-index-url` so that regular
dependencies are still resolved from PyPI:

```bash
python -m venv /tmp/testpypi-grayleafspot
source /tmp/testpypi-grayleafspot/bin/activate

pip install \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    grayleafspot

grayleafspot --help
deactivate
```

If the install or the launcher fails, fix the issue, bump the version patch
number (e.g. `0.1.1` -> `0.1.2`), rebuild, and re-upload.  PyPI and TestPyPI
do not allow re-uploading the same version.

### 6.7 Publish to PyPI (production release)

Once you are satisfied with the TestPyPI result:

```bash
python -m twine upload \
    dist-build/grayleafspot-0.1.1-py3-none-any.whl \
    dist-build/grayleafspot-0.1.1.tar.gz
```

The release will be live at `https://pypi.org/project/grayleafspot/` within a
few minutes.

> **Release checklist**
> - [ ] Version bumped in `pyproject.toml`
> - [ ] `dist/`, `dist-build/`, `grayleafspot/dist/`, `*.egg-info` removed before build
> - [ ] `make build-package` completed without errors
> - [ ] Wheel smoke-tested locally
> - [ ] Wheel smoke-tested from TestPyPI
> - [ ] Only the files for the new version passed to `twine upload`

---

## 7. End-user installation and quick-start guide

This section is for **users** who want to install `grayleafspot` from PyPI and
start analysing images right away.

### 7.1 System requirements

| Requirement | Minimum version | Notes |
|---|---|---|
| Python | 3.10 | 3.11 or 3.13 recommended |
| Node.js | 18 LTS | Required to run the backend API |
| npm | 9+ | Bundled with Node.js |

Install Node.js from https://nodejs.org/ (choose the LTS release).  Verify
your installation:

```bash
node --version   # e.g. v20.x.x
npm --version    # e.g. 10.x.x
python3 --version
```

### 7.2 Step 1 — Install the Python package

```bash
pip install grayleafspot
```

This installs the `grayleafspot` launcher, the `pipeline` analysis package, and
all Python dependencies (PyTorch, OpenCV, SAM, Transformers, etc.).

The bundled React frontend is included in the wheel, so no separate frontend
build step is needed.

### 7.3 Step 2 — Install Node dependencies

The app needs Node/Express for its backend API.  After `pip install`, run:

```bash
cd /path/to/your/working/directory
npm install grayleafspot   # installs Node deps from the embedded package.json
```

Alternatively, if you have a fresh working directory, initialise it:

```bash
mkdir my-analysis && cd my-analysis
npm init -y
npm install express tsx dotenv @google/genai
```

> **Tip:** The launcher expects `npm run api` to be available in the directory
> from which it is launched.  The simplest approach is to copy or symlink the
> `package.json` from the installed package into your working directory and run
> `npm install` there.

### 7.4 Step 3 — Prepare your working directory

Create the folders the app reads from and writes to:

```bash
mkdir -p input_images outputs archives
```

Copy `.env.example` from the installed package (or create a `.env` file) and
fill in the keys you need:

```bash
# Minimum for Gemini mode:
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-2.0-flash

# Minimum for local mode (paths to downloaded model files):
LOCAL_UNET_PATH=models/best_unet.pt
LOCAL_SAM_CHECKPOINT=models/sam_vit_b_01ec64.pth
LOCAL_SAM_MODEL_TYPE=vit_b
LOCAL_ENABLE_MLX=false

API_PORT=8000
```

### 7.5 Step 4 — Download model files (local mode only)

Skip this step if you only plan to use the Gemini engine.

Download the SAM checkpoint:

```bash
curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
     -o models/sam_vit_b_01ec64.pth
```

Place your trained U-Net checkpoint at `models/best_unet.pt`.

Download the optional Gemma MLX prior (requires `huggingface_hub`):

```bash
pip install "huggingface_hub[cli]"
hf download FakeRockert543/gemma-4-e2b-it-MLX-4bit \
    --local-dir models/gemma-4-e2b-it-MLX-4bit
```

### 7.6 Step 5 — Launch the app

From your working directory (the one that contains `input_images/`, `outputs/`,
`archives/`, and `.env`):

```bash
grayleafspot
```

The launcher will:

1. Verify that bundled frontend assets are present.
2. Start the Node/Express API on port 8000 (or `API_PORT` from `.env`).
3. Serve the React frontend on port 3000.
4. Open your browser automatically at `http://127.0.0.1:3000`.

Press `Ctrl-C` to stop everything cleanly.

### 7.7 CLI flags

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address for the static file server |
| `--port` | `3000` | Port for the static file server |
| `--api-port` | `8000` | Port for the Node API |
| `--no-browser` | — | Do not open the browser automatically |
| `--skip-node-api` | — | Skip the Node API (static frontend only) |

Example — run on a different port without auto-opening the browser:

```bash
grayleafspot --port 4000 --api-port 9000 --no-browser
```

### 7.8 Using the analysis pipeline directly (without the GUI)

Run the Gemini engine on all images in `input_images/`:

```bash
python -m pipeline.cli --engine gemini --input-dir input_images --output-dir outputs
```

Run the local engine (requires model files from §7.5):

```bash
python -m pipeline.cli --engine local --input-dir input_images --output-dir outputs
```

### 7.9 Troubleshooting

| Symptom | Fix |
|---|---|
| `command not found: grayleafspot` | Ensure the Python `bin/` directory is on your `PATH` (e.g. activate your venv or use `pipx install grayleafspot`) |
| `npm: command not found` | Install Node.js from https://nodejs.org/ |
| Port already in use | Use `--port` / `--api-port` to choose different ports |
| `No module named pipeline` | Install from PyPI again; ensure you are using the right Python/venv |
| Gemini API errors | Check `GEMINI_API_KEY` in `.env`; the key can also be entered in the browser GUI |
| Missing model files | Follow §7.5; check paths in `.env` match the actual file locations |

---

## 8. Key invariants to preserve

| Rule | Why |
|---|---|
| Do not move `server/index.ts` | `launcher.py` runs `npm run api`; `package.json` points to `server/index.ts` |
| Do not rename `grayleafspot/` | `pyproject.toml` entry point and `packages.find` both reference it by name |
| Keep Vite output in `dist/` | `Makefile` copies from `dist/*`; changing the output dir breaks bundling |
| Do not commit `grayleafspot/dist/` or `dist/` | They are build artifacts; the Makefile regenerates them |
| Keep `grayleafspot/dist/**/*` in `package-data` | Required for the wheel to include assets |
