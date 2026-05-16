"""Training orchestration helpers with MLflow tracking."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .constants import LOGGER

_KEYS_TO_STRIP = {"model", "nc", "names", "scales", "backbone", "head"}
_KEYS_ALREADY_PASSED = {"data", "task", "epochs", "imgsz", "batch", "project", "name", "amp", "workers"}

TRAIN_RUN_NAME = "seg"
MODEL_REGISTRY_NAME = "YOLO_Plot_Digitizer_Curriculum"


def _find_latest_run_dir(project_dir: Path, base_name: str = TRAIN_RUN_NAME) -> Path | None:
    """Find the latest Ultralytics training run directory under *project_dir*."""
    if not project_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for entry in project_dir.iterdir():
        if not entry.is_dir():
            continue
        m = re.match(rf"^{re.escape(base_name)}(\d*)$", entry.name)
        if m:
            suffix = int(m.group(1)) if m.group(1) else 1
            weights_dir = entry / "weights"
            if weights_dir.is_dir() and any(weights_dir.glob("*.pt")):
                candidates.append((suffix, entry))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _load_hyp_overrides(hyp_yaml: Path | None) -> dict[str, Any]:
    """Load hyperparameter overrides from a YAML, stripping architecture keys."""
    if hyp_yaml is None:
        return {}
    import yaml
    with open(hyp_yaml) as f:
        raw = yaml.safe_load(f) or {}
    return {k: v for k, v in raw.items() if k not in _KEYS_TO_STRIP}


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
    amp: bool = False,
    mlflow_run_id: str | None = None,
    mlflow_stage_name: str | None = None,
) -> dict[str, Any]:
    """Create or execute a YOLO segmentation training job."""
    dataset_yaml = (dataset_dir / "dataset.yaml").resolve()
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_yaml}")
    hyp_overrides = _load_hyp_overrides(hyp_yaml)
    hyp_path = hyp_yaml.resolve() if hyp_yaml is not None else None
    training_plan = {
        "dataset": str(dataset_yaml),
        "weights": weights,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": str(output_dir),
        "task": "segment",
        "amp": amp,
    }
    if hyp_path is not None:
        training_plan["cfg"] = str(hyp_path)
    if workers is not None:
        training_plan["workers"] = workers
    if execute:
        try:
            import torch as _torch
        except ImportError:
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
                "Training requires torch. Install with:\n"
                f"  pip install torch torchvision --index-url {index_url}"
            )
        if workers is not None:
            _torch.set_num_threads(workers)
            try:
                _torch.set_num_interop_threads(workers)
            except RuntimeError:
                pass
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                'Training requires ultralytics. Install with: uv pip install -e ".[ai]"'
            ) from exc

        model = YOLO(weights)
        train_kwargs: dict[str, Any] = {
            "data": str(dataset_yaml),
            "task": "segment",
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "project": str(output_dir),
            "name": TRAIN_RUN_NAME,
            "amp": amp,
        }
        if workers is not None:
            train_kwargs["workers"] = workers
        for key, value in hyp_overrides.items():
            if key not in _KEYS_ALREADY_PASSED:
                train_kwargs[key] = value
        result = model.train(**train_kwargs)
        training_plan["result"] = result.save_dir.as_posix()

        # MLflow: log weights and register model
        if mlflow_run_id and mlflow_stage_name:
            _log_to_mlflow(mlflow_run_id, train_kwargs["project"], mlflow_stage_name, weights)

    return training_plan


def _log_to_mlflow(run_id: str, project_dir: str, stage_name: str, input_weights: str) -> None:
    """Log training artifacts to MLflow and register the model."""
    try:
        import mlflow
    except ImportError:
        LOGGER.debug("MLflow not installed — skipping artifact logging")
        return

    run_dir = _find_latest_run_dir(Path(project_dir), TRAIN_RUN_NAME)
    if run_dir is None:
        run_dir = _find_latest_run_dir(Path(project_dir), "synthetic_plot_digitizer")
    if run_dir is None:
        return

    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        return

    artifact_path = f"pesos_etapa_{stage_name}"
    client = mlflow.tracking.MlflowClient()

    try:
        client.log_artifact(run_id, str(best_pt), artifact_path=artifact_path)
        LOGGER.info("  [MLflow] Logged %s/best.pt", artifact_path)

        # Log full weights directory
        client.log_artifacts(run_id, str(run_dir / "weights"), artifact_path=artifact_path)

        # Register model
        try:
            client.create_registered_model(MODEL_REGISTRY_NAME)
        except mlflow.exceptions.MlflowException:
            pass

        model_uri = f"runs:/{run_id}/{artifact_path}"
        mv = client.create_model_version(
            name=MODEL_REGISTRY_NAME,
            source=model_uri,
            run_id=run_id,
            description=f"Curriculum stage: {stage_name}",
        )
        LOGGER.info("  [MLflow] Registered %s version %s", MODEL_REGISTRY_NAME, mv.version)
    except Exception as exc:
        LOGGER.warning("  [MLflow] Could not log artifacts: %s", exc)
