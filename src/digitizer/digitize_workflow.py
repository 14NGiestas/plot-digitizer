"""Top-level digitization workflow."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .ai_segmentation import run_ai_segmentation, _select_digitization_segmentations
from .annotation_io import CLASS_MAPPING
from .image_ops import load_image, preprocess_image, resolve_plot_box
from .calibration import calibrate_axes
from .constants import LOGGER
from .models import AxisReferencePair, DigitizeResult, SegmentationResult
from .plotting import build_replot_frame, create_overlay, create_replot
from .points import convert_points, extract_curve_points
from .synth.render import _mask_to_yolo_polygon


def _segmentations_to_yolo_label(
    segmentations: list[SegmentationResult],
) -> str:
    """Convert segmentation masks to a YOLO segmentation label string."""
    lines: list[str] = []
    for seg in segmentations:
        polygon = _mask_to_yolo_polygon(seg.mask)
        if not polygon:
            continue
        class_id = seg.class_id if seg.class_id is not None else CLASS_MAPPING.get("curve", 0)
        lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in polygon))
    return "\n".join(lines)


def _run_segmentation(
    image,
    plot_box,
    calibration,
    image_path: Path,
    weights: str | None,
    conf_threshold: float,
    workers: int,
    imgsz: int | None,
) -> tuple[pd.DataFrame, list[SegmentationResult]]:
    """Run segmentation (AI or CV fallback) and extract digitized points."""
    if weights:
        segmentations = run_ai_segmentation(
            image, plot_box, weights, conf_threshold,
            workers=workers, imgsz=imgsz,
        )
        if segmentations:
            segmentations = _select_digitization_segmentations(segmentations)
    else:
        from .cv_segmentation import run_cv_segmentation
        segmentations = run_cv_segmentation(image, plot_box, conf_threshold)

    if not segmentations:
        raise RuntimeError(
            f"Unable to isolate curves in {image_path}. "
            "No curve-class masks were detected."
        )

    point_frames = []
    for seg in segmentations:
        frame = extract_curve_points(seg, plot_box)
        if not frame.empty:
            point_frames.append(frame)

    if not point_frames:
        raise RuntimeError(f"No digitized points were extracted from {image_path}.")

    combined = pd.concat(point_frames, ignore_index=True)
    combined = combined.dropna().sort_values(["dataset_id", "x_px"]).reset_index(drop=True)
    return convert_points(combined, calibration, plot_box), segmentations


def digitize_image(
    image_path: Path,
    output_dir: Path,
    x_range: tuple[float, float] | None,
    y_range: tuple[float, float] | None,
    x_reference: AxisReferencePair | None,
    y_reference: AxisReferencePair | None,
    x_scale: str,
    y_scale: str,
    invert_y: bool,
    weights: str | None,
    conf_threshold: float,
    create_overlay_image: bool,
    workers: int | None = None,
    imgsz: int | None = None,
    auto_axis_anchors: bool = True,
) -> DigitizeResult:
    """Digitize a single image and write artifacts under *output_dir*."""
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    csv_dir = output_dir / "csv"
    for d in (images_dir, labels_dir, csv_dir):
        d.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)
    processed_gray, preprocess_stats = preprocess_image(image)
    plot_box = resolve_plot_box(image_path, image, processed_gray)
    calibration, axis_metadata = calibrate_axes(
        image_path=image_path,
        plot_box=plot_box,
        processed_gray=processed_gray,
        x_range=x_range,
        y_range=y_range,
        x_reference=x_reference,
        y_reference=y_reference,
        x_scale=x_scale,
        y_scale=y_scale,
        invert_y=invert_y,
        auto_axis_anchors=auto_axis_anchors,
    )

    converted, segmentations = _run_segmentation(
        image, plot_box, calibration, image_path,
        weights, conf_threshold, workers or 1, imgsz,
    )

    dest_image = images_dir / image_path.name
    csv_path = csv_dir / f"{image_path.stem}.csv"
    replot_csv_path = csv_dir / f"{image_path.stem}.replot.csv"
    metadata_path = images_dir / f"{image_path.stem}.metadata.json"
    replot_path = images_dir / f"{image_path.stem}.replot.png"
    label_path = labels_dir / f"{image_path.stem}.txt"
    overlay_path = images_dir / f"{image_path.stem}.overlay.png" if create_overlay_image else None

    shutil.copy2(image_path, dest_image)
    converted[["dataset_id", "x_real", "y_real", "confidence"]].to_csv(csv_path, index=False)

    replot_frame = build_replot_frame(converted, x_scale=calibration.x_scale)
    replot_frame.to_csv(replot_csv_path, index=False)
    create_replot(replot_frame, calibration, image_path.name, replot_path)

    label_content = _segmentations_to_yolo_label(segmentations)
    label_path.write_text(label_content)

    metadata = {
        "input_image": str(image_path),
        "image": str(dest_image),
        "plot_box": asdict(plot_box),
        "axis": asdict(calibration),
        "exports": {
            "csv": str(csv_path),
            "replot_csv": str(replot_csv_path),
            "replot_image": str(replot_path),
            "labels": str(label_path),
            "overlay_image": str(overlay_path) if overlay_path else None,
        },
        "preprocessing": preprocess_stats,
        "segmentation": {
            "dataset_count": int(converted["dataset_id"].nunique()),
            "points": int(len(converted)),
            "method_counts": {
                str(method): int(count)
                for method, count in pd.Series([s.method for s in segmentations]).value_counts().items()
            },
            "confidence_stats": {
                "min": float(converted["confidence"].min()),
                "max": float(converted["confidence"].max()),
                "mean": float(converted["confidence"].mean()),
            },
        },
        **axis_metadata,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    if overlay_path is not None:
        create_overlay(image, converted, segmentations, overlay_path)

    return DigitizeResult(
        csv_path=csv_path,
        replot_csv_path=replot_csv_path,
        metadata_path=metadata_path,
        replot_path=replot_path,
        overlay_path=overlay_path,
        point_count=int(len(converted)),
        dataset_count=int(converted["dataset_id"].nunique()),
        label_path=label_path,
    )
