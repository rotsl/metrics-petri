from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from tqdm import TqdmWarning

from . import NotebookRuntime, _mlx_runtime_available, _mlx_runtime_reason, configure_pipeline_defaults, resolve_notebook_runtime, set_runtime


def bootstrap_notebook_environment(cwd: Path, analysis_module: Any) -> dict[str, Any]:
    root_candidates = [cwd, cwd.parent]
    root = next(
        (
            candidate
            for candidate in root_candidates
            if (candidate / "pipeline").exists() and (candidate / "input_images").exists()
        ),
        cwd,
    )
    for candidate in (root / "src", root):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    runtime: NotebookRuntime = resolve_notebook_runtime(root)
    set_runtime(runtime)
    configure_pipeline_defaults(analysis_module, runtime)

    os.environ["LOCAL_ENABLE_MLX"] = os.getenv("LOCAL_ENABLE_MLX", "1")
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.rcParams["axes.grid"] = False

    print("Notebook root =", root)
    print("Input directory =", runtime.input_dir)
    print("Output directory =", runtime.output_dir)
    print("Local U-Net checkpoint =", runtime.local_unet_checkpoint, "| exists =", runtime.local_unet_checkpoint.exists())
    print("Local SAM checkpoint =", runtime.local_sam_checkpoint, "| exists =", runtime.local_sam_checkpoint.exists())
    print("Local Generic model dir =", runtime.local_generic_model_dir, "| exists =", runtime.local_generic_model_dir.exists())
    print("Notebook local model ID =", runtime.notebook_local_model_id)
    print("LOCAL_ENABLE_MLX =", os.environ["LOCAL_ENABLE_MLX"])
    print("MLX runtime available =", _mlx_runtime_available())
    if not _mlx_runtime_available():
        print("MLX runtime reason =", _mlx_runtime_reason())

    return {
        "ROOT": root,
        "NOTEBOOK_RUNTIME": runtime,
        "INPUT_DIR": runtime.input_dir,
        "OUTPUT_DIR": runtime.output_dir,
        "ARCHIVES_DIR": runtime.archives_dir,
        "LOCAL_GENERIC_MODEL_DIR": runtime.local_generic_model_dir,
        "NOTEBOOK_LOCAL_MODEL_ID": runtime.notebook_local_model_id,
        "LOCAL_UNET_CHECKPOINT": runtime.local_unet_checkpoint,
        "LOCAL_SAM_CHECKPOINT": runtime.local_sam_checkpoint,
        "LOCAL_ENABLE_MLX": os.environ["LOCAL_ENABLE_MLX"],
    }


__all__ = ["bootstrap_notebook_environment"]
