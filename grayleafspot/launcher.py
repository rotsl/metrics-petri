from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from .assets import assets_exist, get_assets_dir


def _start_node_api(api_port: int) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env["API_PORT"] = str(api_port)

    # Uses the existing dev API entrypoint.
    # This preserves your current server/ setup.
    return subprocess.Popen(
        ["npm", "run", "api"],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
    )


def _wait_for_assets_or_fail() -> Path:
    assets_dir = get_assets_dir()
    if not assets_exist():
        raise FileNotFoundError(
            f"Frontend assets not found at {assets_dir}. "
            "Build the frontend and copy the output into grayleafspot/dist before packaging."
        )
    return assets_dir


def _serve_assets(assets_dir: Path, host: str, port: int) -> ThreadingHTTPServer:
    class AppHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(assets_dir), **kwargs)

        def log_message(self, format: str, *args) -> None:
            return

    server = ThreadingHTTPServer((host, port), AppHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def run_app(
    host: str = "127.0.0.1",
    port: int = 3000,
    api_port: int = 8000,
    open_browser: bool = True,
    skip_node_api: bool = False,
) -> None:
    assets_dir = _wait_for_assets_or_fail()

    node_proc = None
    if not skip_node_api:
        node_proc = _start_node_api(api_port)
        time.sleep(1)

    server = _serve_assets(assets_dir, host, port)

    if open_browser:
        webbrowser.open(f"http://{host}:{port}")

    try:
        while True:
            time.sleep(1)
            if node_proc is not None and node_proc.poll() is not None:
                raise RuntimeError("Node API process exited unexpectedly.")
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        if node_proc is not None and node_proc.poll() is None:
            node_proc.terminate()
            try:
                node_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                node_proc.kill()