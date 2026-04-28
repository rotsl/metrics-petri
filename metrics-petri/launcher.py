from __future__ import annotations

import time
import webbrowser
from threading import Thread

from pathlib import Path

from .api import create_server
from .bootstrap import ensure_models


def run_app(
    host: str = "127.0.0.1",
    port: int = 3000,
    open_browser: bool = True,
    skip_model_bootstrap: bool = False,
    skip_node_api: bool = False,
) -> None:
    root_dir = Path.cwd()

    if not skip_model_bootstrap and not skip_node_api:
        for message in ensure_models(root_dir):
            print(message, flush=True)

    server = create_server(host, port)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    if open_browser:
        webbrowser.open(f"http://{host}:{port}")

    try:
        while thread.is_alive():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
