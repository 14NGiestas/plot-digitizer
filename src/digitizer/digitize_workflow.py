"""Top-level digitization workflow."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .ai_segmentation import run_ai_segmentation, _select_digitization_segmentations
from .annotation_io import CLASS_MAPPING
from .model_wrapper import DigitizerModel
from .image_ops import load_image, preprocess_image, resolve_plot_box
from .calibration import calibrate_axes
from .constants import LOGGER
from .image_ops import load_image, preprocess_image, resolve_plot_box
from .models import AxisReferencePair, DigitizeResult, SegmentationResult
from .plotting import build_replot_frame, create_overlay, create_replot
from .points import convert_points, extract_curve_points
from .synth.render import _mask_to_yolo_polygon


def _segmentations_to_yolo_label(
    segmentations: list[SegmentationResult],
) -> str:
    """Convert segmentation masks to a YOLO segmentation label string.

    Each mask is converted to a contour polygon.  The class ID is taken from
    ``segmentation.class_id`` when available, otherwise defaults to 0 (curve).
    """
    lines: list[str] = []
    for seg in segmentations:
        polygon = _mask_to_yolo_polygon(seg.mask)
        if not polygon:
            continue
        # Use the class_id from the AI prediction when available; fall back to
        # the "curve" entry in CLASS_MAPPING. CLASS_MAPPING is validated to be
        # contiguous (0..nc-1) at import time in synth_dataset.py, so "curve"
        # must be present — the .get() default is purely a defensive guard.
        class_id = seg.class_id if seg.class_id is not None else CLASS_MAPPING.get("curve", 0)
        lines.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in polygon))
    return "\n".join(lines)


def _select_digitization_segmentations(
    segmentations: list[SegmentationResult],
) -> list[SegmentationResult]:
    """Keep only curve masks when AI predictions include class IDs."""
    if not segmentations:
        return []
    curve_class_id = CLASS_MAPPING.get("curve")
    if curve_class_id is None or all(seg.class_id is None for seg in segmentations):
        return segmentations
    curve_segmentations = [
        segmentation
        for segmentation in segmentations
        if segmentation.class_id in (None, curve_class_id)
    ]
    return curve_segmentations


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
    """Digitize a single image and write artifacts under *output_dir*.

    Output layout::

        output_dir/
            images/<stem>.<ext>          source image copy
            images/<stem>.metadata.json  processing metadata sidecar
            images/<stem>.replot.png     replot visualisation
            images/<stem>.overlay.png    segmentation overlay (optional)
            labels/<stem>.txt            YOLO segmentation labels from AI
            csv/<stem>.csv               primary digitized (x, y) data
            csv/<stem>.replot.csv        smoothed/interpolated replot data
    """
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

    # Use the wrapper to perform AI segmentation and conversion
    model = DigitizerModel(weights, conf_threshold)
    converted, segmentations = model.digitize(
        image, plot_box, calibration, image_path, workers=workers or 1, imgsz=imgsz
    )

    # Build output paths under the unified subdirectory structure.
    dest_image = images_dir / image_path.name
    csv_path = csv_dir / f"{image_path.stem}.csv"
    replot_csv_path = csv_dir / f"{image_path.stem}.replot.csv"
    metadata_path = images_dir / f"{image_path.stem}.metadata.json"
    replot_path = images_dir / f"{image_path.stem}.replot.png"
    label_path = labels_dir / f"{image_path.stem}.txt"
    overlay_path = images_dir / f"{image_path.stem}.overlay.png" if create_overlay_image else None

    # Copy source image into images/.
    shutil.copy2(image_path, dest_image)

    # Primary CSV output: digitized (x, y) data.
    converted[["dataset_id", "x_real", "y_real", "confidence"]].to_csv(csv_path, index=False)

    # Replot CSV and PNG.
    replot_frame = build_replot_frame(converted, x_scale=calibration.x_scale)
    replot_frame.to_csv(replot_csv_path, index=False)
    create_replot(replot_frame, calibration, image_path.name, replot_path)

    # YOLO labels from AI segmentation (metadata / annotations).
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
                for method, count in pd.Series([segmentation.method for segmentation in segmentations]).value_counts().items()
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
