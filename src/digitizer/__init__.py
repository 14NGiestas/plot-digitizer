"""Automatic plot digitizer CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from . import synth
from .ai_segmentation import run_ai_segmentation
from .axis_parsing import parse_range, parse_reference_pair
from .calibration import calibrate_axes
from .cli_support import _set_matplotlib_backend, configure_logging
from .constants import LOGGER
from .digitize_workflow import digitize_image
from .image_ops import discover_images
from .interactive_axis import _format_reference_pair_cli_value, interactive_reference_selection
from .models import AxisCalibration, AxisReferencePair, DigitizeResult, PlotBox, SegmentationResult
from .parser import _json_default, _parse_positive_int, build_parser
from .plotting import build_replot_frame, create_overlay, create_replot
from .points import convert_points, extract_curve_points
from .synth import (
    _apply_degradation_filters,
    _render_arrow_mask,
    _render_curve_mask,
    _render_error_bar_mask,
    _render_hbar_mask,
    _render_vbar_mask,
    _write_synthetic_example,
    generate_synthetic_dataset,
)
from .training import run_training, _find_latest_run_dir, TRAIN_RUN_NAME, MODEL_REGISTRY_NAME


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    if args.command == "generate":
        generate_synthetic_dataset(
            args.output_dir, args.count, args.seed, args.image_format, args.plot_type,
            workers=args.workers, degradations=args.degradations,
            difficulty=args.difficulty, curriculum=args.curriculum,
        )
        total_images = args.count * args.degradations
        LOGGER.info(
            "Generated %s synthetic plot(s) × %s degradation(s) = %s training image(s) (%s) in %s",
            args.count, args.degradations, total_images, args.plot_type, args.output_dir,
        )
        return 0

    if args.command == "train":
        if args.status:
            _show_curriculum_status(args.output_dir)
            return 0
        if args.chain_info:
            _show_chain_info(
                output_dir=args.output_dir,
                from_stage=args.from_stage,
                resume=args.resume,
            )
            return 0
        if args.sync:
            _sync_curriculum_progress(args.output_dir)
            return 0
        _run_curriculum(
            output_dir=args.output_dir,
            samples_per_stage=args.samples_per_stage,
            seed=args.seed,
            epochs=args.epochs,
            batch=args.batch,
            workers=args.workers,
            from_stage=args.from_stage,
            resume=args.resume,
        )
        return 0

    if args.command == "digitize":
        images = discover_images(args.inputs)
        if not images:
            parser.error("No input images were found.")

        _set_matplotlib_backend("Agg")

        x_range = parse_range(args.x_range)
        y_range = parse_range(args.y_range)
        x_reference = parse_reference_pair(args.x_reference, "x")
        y_reference = parse_reference_pair(args.y_reference, "y")

        import concurrent.futures
        import os

        def _process_one(image_path: Path) -> dict[str, Any]:
            result = digitize_image(
                image_path=image_path,
                output_dir=args.output_dir,
                x_range=x_range,
                y_range=y_range,
                x_reference=x_reference,
                y_reference=y_reference,
                x_scale=args.x_scale,
                y_scale=args.y_scale,
                invert_y=args.invert_y,
                weights=args.weights,
                conf_threshold=args.conf_threshold,
                create_overlay_image=args.overlay,
                workers=1,
                imgsz=args.imgsz,
                auto_axis_anchors=not args.disable_auto_axis_anchors,
            )
            return {
                "image": str(image_path),
                "csv_path": str(result.csv_path),
                "replot_csv_path": str(result.replot_csv_path),
                "metadata_path": str(result.metadata_path),
                "replot_path": str(result.replot_path),
                "label_path": str(result.label_path) if result.label_path else None,
                "overlay_path": str(result.overlay_path) if result.overlay_path else None,
                "point_count": result.point_count,
                "dataset_count": result.dataset_count,
            }

        results = []
        n_workers = args.workers if args.workers is not None else min(os.cpu_count() or 1, len(images), 8)
        if n_workers <= 1 or len(images) <= 1:
            for image_path in images:
                results.append(_process_one(image_path))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(_process_one, images))

        print(json.dumps(results, indent=2))
        return 0

    if args.command == "annotate":
        _set_matplotlib_backend("TkAgg")
        from .interactive_annotator import interactive_annotation_session
        resize_to: tuple[int, int] | None = None
        if (args.resize_width is None) ^ (args.resize_height is None):
            parser.error("--resize-width and --resize-height must be used together.")
        if args.resize_width is not None and args.resize_height is not None:
            resize_to = (int(args.resize_width), int(args.resize_height))
        result = interactive_annotation_session(
            image_path=args.input,
            output_dir=args.output_dir,
            line_width=args.line_width,
            resize_to=resize_to,
            update_existing=args.update,
        )
        if result:
            print(json.dumps(result, indent=2))
        else:
            LOGGER.info("No annotations saved.")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


# ── Curriculum helpers ────────────────────────────────────────────────────

def _find_stage_weights(train_dir: Path) -> str | None:
    """Find the best.pt in the latest Ultralytics run under *train_dir*."""
    run_dir = _find_latest_run_dir(train_dir, TRAIN_RUN_NAME)
    if run_dir is None:
        run_dir = _find_latest_run_dir(train_dir, "synthetic_plot_digitizer")
    if run_dir is None:
        return None
    best_pt = run_dir / "weights" / "best.pt"
    if best_pt.exists():
        return str(best_pt)
    return None


def _run_curriculum(
    output_dir: Path,
    samples_per_stage: int,
    seed: int,
    epochs: int,
    batch: int,
    workers: int | None,
    from_stage: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the full curriculum pipeline with MLflow tracking."""
    from .training import _load_hyp_overrides

    stages = [
        {"difficulty": 1, "hyp": Path("hyps/curriculum_stage1.yml"), "name": "stage1"},
        {"difficulty": 2, "hyp": Path("hyps/curriculum_stage2.yml"), "name": "stage2"},
        {"difficulty": 3, "hyp": Path("hyps/curriculum_stage3.yml"), "name": "stage3"},
        {"difficulty": 4, "hyp": Path("hyps/curriculum_stage4.yml"), "name": "stage4"},
    ]

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    # MLflow setup
    mlflow_run_id = _setup_mlflow(root)

    progress_file = root / "progress.json"
    progress: dict[str, Any] = {}
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)

    if resume:
        for stage in reversed(stages):
            key = stage["name"]
            train_dir = root / key / "train"
            found_pt = _find_stage_weights(train_dir)
            if found_pt or (key in progress and progress[key].get("status") == "done"):
                from_stage = stages.index(stage) + 2
                LOGGER.info("Auto-resume: %s has checkpoint, starting from stage %d", key, from_stage)
                break
        if from_stage is None:
            from_stage = 1

    start_idx = (from_stage - 1) if from_stage is not None else 0
    start_idx = max(0, min(start_idx, len(stages)))

    plan: dict[str, Any] = {
        "root": str(root),
        "samples_per_stage": samples_per_stage,
        "start_stage": stages[start_idx]["name"] if start_idx < len(stages) else "done",
        "stages": [],
    }

    if start_idx == 0:
        weights = "yolo11s-seg.pt"
    else:
        prev_stage = stages[start_idx - 1]
        prev_pt = _find_stage_weights(root / prev_stage["name"] / "train")
        if prev_pt:
            weights = prev_pt
            LOGGER.info("✓ Chained weights from %s: %s", prev_stage["name"], weights)
        else:
            weights = "yolo11s-seg.pt"
            LOGGER.warning(" No checkpoint for %s, falling back to base weights", prev_stage["name"])

    for i, stage in enumerate(stages):
        stage_key = stage["name"]
        stage_dir = root / stage_key
        data_dir = stage_dir / "data"
        train_dir = stage_dir / "train"

        stage_plan: dict[str, Any] = {
            "name": stage_key,
            "difficulty": stage["difficulty"],
            "data_dir": str(data_dir),
            "train_dir": str(train_dir),
            "weights": weights,
        }

        if i < start_idx:
            if stage_key in progress:
                stage_plan["status"] = "skipped (already done)"
                stage_plan["weights"] = progress[stage_key].get("weights", weights)
            else:
                stage_plan["status"] = "skipped (before start)"
            plan["stages"].append(stage_plan)
            continue

        has_checkpoint = _find_stage_weights(train_dir) is not None
        is_done = stage_key in progress and progress[stage_key].get("status") == "done"

        if is_done and has_checkpoint and not from_stage:
            LOGGER.info("✓ %s already completed, skipping", stage_key)
            stage_plan["status"] = "skipped (already done)"
            stage_plan["weights"] = progress[stage_key].get("weights", weights)
            plan["stages"].append(stage_plan)
            continue

        if has_checkpoint and not is_done:
            LOGGER.info("↻ %s has partial checkpoint, resuming training", stage_key)

        LOGGER.info("━━━ %s (difficulty %d) ━━━", stage_key.upper(), stage["difficulty"])
        LOGGER.info("  Input weights: %s", weights)

        if data_dir.exists() and (data_dir / "dataset.yaml").exists():
            LOGGER.info("  ✓ Data exists, skipping generation")
        else:
            LOGGER.info("  → Generating %d samples...", samples_per_stage)
            generate_synthetic_dataset(
                output_dir=data_dir,
                count=samples_per_stage,
                seed=seed + stage["difficulty"],
                image_format="png",
                plot_type="mixed",
                workers=workers,
                difficulty=stage["difficulty"],
            )
            LOGGER.info("  ✓ Generation complete")

        hyp_yaml = stage["hyp"]
        hyp_overrides = _load_hyp_overrides(hyp_yaml if hyp_yaml.exists() else None)

        stage_epochs = hyp_overrides.pop("epochs", epochs)
        stage_batch = hyp_overrides.pop("batch", batch)
        stage_imgsz = hyp_overrides.pop("imgsz", 640)

        LOGGER.info("  → Training: %d epochs, imgsz=%d, batch=%d", stage_epochs, stage_imgsz, stage_batch)
        run_training(
            dataset_dir=data_dir,
            output_dir=train_dir,
            epochs=stage_epochs,
            imgsz=stage_imgsz,
            weights=weights,
            batch=stage_batch,
            execute=True,
            hyp_yaml=hyp_yaml,
            workers=workers,
            mlflow_run_id=mlflow_run_id,
            mlflow_stage_name=stage_key,
        )

        found_pt = _find_stage_weights(train_dir)
        if found_pt:
            weights = found_pt
            LOGGER.info("  ✓ Output weights: %s", weights)
        else:
            LOGGER.warning("  ✗ No weights found after training %s", stage_key)

        stage_plan["result_weights"] = weights
        stage_plan["status"] = "done"

        progress[stage_key] = {
            "status": "done",
            "weights": weights,
            "difficulty": stage["difficulty"],
        }
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2, default=_json_default)

        plan["stages"].append(stage_plan)

    print(json.dumps(plan, indent=2, default=_json_default))
    return plan


def _setup_mlflow(root: Path) -> str | None:
    """Initialize MLflow tracking. Returns run_id or None."""
    try:
        import mlflow
    except ImportError:
        LOGGER.debug("MLflow not installed — skipping tracking")
        return None

    mlruns = root / "mlruns"
    mlruns.mkdir(exist_ok=True)
    tracking_uri = f"file:{mlruns}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("plot-digitizer-curriculum")

    import time
    with mlflow.start_run(run_name=f"curriculum-{time.strftime('%Y%m%d-%H%M%S')}") as run:
        run_id = run.info.run_id
        LOGGER.info("MLflow run ID: %s (tracking: %s)", run_id, tracking_uri)
        return run_id


def _show_curriculum_status(output_dir: Path) -> None:
    """Print a human-readable progress report."""
    root = Path(output_dir)
    progress_file = root / "progress.json"

    stages = [
        {"difficulty": 1, "name": "stage1"},
        {"difficulty": 2, "name": "stage2"},
        {"difficulty": 3, "name": "stage3"},
        {"difficulty": 4, "name": "stage4"},
    ]

    progress: dict[str, Any] = {}
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)

    print(f"\n Curriculum Progress: {output_dir}")
    print("=" * 60)

    for stage in stages:
        key = stage["name"]
        best_pt = _find_stage_weights(root / key / "train")
        data_exists = (root / key / "data" / "dataset.yaml").exists()

        if key in progress and progress[key].get("status") == "done":
            status = "DONE"
            marker = "✓"
        elif best_pt:
            status = "DONE (checkpoint found)"
            marker = "✓"
        elif data_exists:
            status = "data generated, not trained"
            marker = "·"
        else:
            status = "not started"
            marker = " "

        wsize = ""
        if best_pt:
            wsize = f" ({Path(best_pt).stat().st_size / 1e6:.1f} MB)"

        print(f"  {marker} {key.upper()} (diff {stage['difficulty']})  [{status}]{wsize}")

    print("=" * 60)
    if progress_file.exists():
        print(f"  Progress file: {progress_file}")
    mlruns = root / "mlruns"
    if mlruns.exists():
        print(f"  MLflow tracking: file:{mlruns}")
        print(f"  View with: mlflow ui --backend-store-uri file:{mlruns}")
    print()


def _show_chain_info(output_dir: Path, from_stage: int | None = None, resume: bool = False) -> None:
    """Print the weight chain that will be used."""
    root = Path(output_dir)
    progress_file = root / "progress.json"

    stages = [
        {"difficulty": 1, "name": "stage1"},
        {"difficulty": 2, "name": "stage2"},
        {"difficulty": 3, "name": "stage3"},
        {"difficulty": 4, "name": "stage4"},
    ]

    progress: dict[str, Any] = {}
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)

    if resume:
        for stage in reversed(stages):
            key = stage["name"]
            found_pt = _find_stage_weights(root / key / "train")
            if found_pt or (key in progress and progress[key].get("status") == "done"):
                from_stage = stages.index(stage) + 2
                break
        if from_stage is None:
            from_stage = 1

    start_idx = (from_stage - 1) if from_stage is not None else 0
    start_idx = max(0, min(start_idx, len(stages)))

    print(f"\n Weight Chain: {output_dir}")
    print("=" * 70)

    if start_idx == 0:
        weights = "yolo11s-seg.pt (base)"
    else:
        prev = stages[start_idx - 1]
        prev_pt = _find_stage_weights(root / prev["name"] / "train")
        if prev_pt:
            weights = prev_pt
        else:
            weights = "yolo11s-seg.pt (base, fallback)"

    for i, stage in enumerate(stages):
        marker = "→" if i == start_idx else " "
        if i < start_idx:
            marker = "✓"
        print(f"  {marker} {stage['name'].upper():10s}  input: {weights}")

        found_pt = _find_stage_weights(root / stage["name"] / "train")
        if found_pt:
            weights = found_pt
            print(f"             output: {weights}")
        else:
            print(f"             output: (will train → {stage['name']}/train/seg*/weights/best.pt)")
            weights = f"({stage['name']}/train/seg*/weights/best.pt)"

    print("=" * 70)
    print()


def _sync_curriculum_progress(output_dir: Path) -> None:
    """Scan checkpoints and create/update progress.json."""
    root = Path(output_dir)
    progress_file = root / "progress.json"

    stages = [
        {"difficulty": 1, "name": "stage1"},
        {"difficulty": 2, "name": "stage2"},
        {"difficulty": 3, "name": "stage3"},
        {"difficulty": 4, "name": "stage4"},
    ]

    progress: dict[str, Any] = {}
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)

    changed = False
    for stage in stages:
        key = stage["name"]
        found_pt = _find_stage_weights(root / key / "train")
        if found_pt and key not in progress:
            progress[key] = {
                "status": "done",
                "weights": found_pt,
                "difficulty": stage["difficulty"],
                "synced": True,
            }
            changed = True
            LOGGER.info("  ✓ Synced %s → %s", key, found_pt)

    if changed:
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2, default=_json_default)
        LOGGER.info("Progress file updated: %s", progress_file)
    else:
        LOGGER.info("No new checkpoints found. Progress file unchanged.")

    _show_curriculum_status(output_dir)
