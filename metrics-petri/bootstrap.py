from __future__ import annotations

import shutil
from pathlib import Path

import requests
from huggingface_hub import snapshot_download

SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
GEMMA_REPO_ID = "FakeRockert543/gemma-4-e2b-it-MLX-4bit"
KAGGLE_MODEL_ID = "rbrtsl/metrics-petri-unet/pyTorch/v2026"


def _copy_best_unet_model(source_dir: Path, target_path: Path) -> None:
    candidates = [source_dir / "best_unet.pt", *source_dir.rglob("*.pt")]
    for candidate in candidates:
        if candidate.exists():
            shutil.copy2(candidate, target_path)
            return
    raise FileNotFoundError(f"Could not find a .pt checkpoint in {source_dir}")


def ensure_models(root_dir: Path) -> list[str]:
    """Download required models into the caller's working directory if needed."""

    messages: list[str] = []
    models_dir = root_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    unet_target = Path(root_dir / "models" / "best_unet.pt")
    sam_target = Path(root_dir / "models" / "sam_vit_b_01ec64.pth")
    gemma_target = Path(root_dir / "models" / "gemma-4-e2b-it-MLX-4bit")

    if not unet_target.exists():
        import kagglehub

        source = Path(kagglehub.model_download(KAGGLE_MODEL_ID))
        _copy_best_unet_model(source, unet_target)
        messages.append(f"Downloaded U-Net checkpoint to {unet_target}")

    if not sam_target.exists():
        response = requests.get(SAM_CHECKPOINT_URL, stream=True, timeout=120)
        response.raise_for_status()
        with sam_target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        messages.append(f"Downloaded SAM checkpoint to {sam_target}")

    if not gemma_target.exists() or not gemma_target.is_dir() or not any(gemma_target.iterdir()):
        snapshot_download(
            repo_id=GEMMA_REPO_ID,
            local_dir=str(gemma_target),
            local_dir_use_symlinks=False,
        )
        messages.append(f"Downloaded Gemma MLX model to {gemma_target}")

    return messages
