"""Training orchestration helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .constants import LOGGER

def run_training(
    dataset_dir: Path,
    output_dir: Path,
    epochs: int,
    imgsz: int,
    weights: str,
    batch: int,
    execute: bool,
    hyp_yaml: Path | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Create or execute a YOLO segmentation training job."""
    dataset_yaml = (dataset_dir / "dataset.yaml").resolve()
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_yaml}")
    training_plan = {
        "dataset": str(dataset_yaml),
        "weights": weights,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": str(output_dir),
        "task": "segment",
    }
    hyp_path = hyp_yaml.resolve() if hyp_yaml is not None else None
    if hyp_path is not None:
        if not hyp_path.exists():
            raise FileNotFoundError(f"Hyperparameter config not found: {hyp_path}")
        training_plan["cfg"] = str(hyp_path)
    if workers is not None:
        training_plan["workers"] = workers
    if execute:
        try:
            import torch as _torch
        except ImportError:  # pragma: no cover - depends on optional dependency setup
            import re as _re
            cuda_path = os.environ.get("CUDA_PATH", "")
            rocm_path = os.environ.get("ROCM_PATH", "")
            if rocm_path:
                index_url = "https://download.pytorch.org/whl/rocm6.2"
            else:
                _m = _re.search(r"cuda[_-]?(\d+)[._-]\d+", cuda_path, _re.IGNORECASE)
                cuda_major = int(_m.group(1)) if _m else 0
                if cuda_major == 11:
                    index_url = "https://download.pytorch.org/whl/cu114"
                elif cuda_major >= 12:
                    index_url = "https://download.pytorch.org/whl/cu124"
                else:
                    index_url = "https://download.pytorch.org/whl/cpu"
            raise ImportError(
                "Training requires torch and torchvision, which are not included in the Nix "
                "shell by default. Install them for your accelerator with:\n"
                f"  pip install torch torchvision --index-url {index_url}\n"
                "Then rerun the command. See the README for all accelerator options."
            )
        if workers is not None:
            _torch.set_num_threads(workers)
            try:
                _torch.set_num_interop_threads(workers)
            except RuntimeError:
                LOGGER.debug(
                    "Unable to set torch inter-op thread count to %s; continuing with existing setting.",
                    workers,
                )
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on optional dependency setup
            raise ImportError(
                "Training requires ultralytics. Install digitizer with the 'ai' extra: "
                "`uv pip install -e \".[ai]\"`"
            ) from exc

        model = YOLO(weights)
        train_kwargs: dict[str, Any] = {
            "data": str(dataset_yaml),
            "task": "segment",
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "project": str(output_dir),
            "name": "synthetic_plot_digitizer",
        }
        if workers is not None:
            train_kwargs["workers"] = workers
        if hyp_path is not None:
            train_kwargs["cfg"] = str(hyp_path)
        training_plan["result"] = model.train(**train_kwargs).save_dir.as_posix()
    return training_plan

