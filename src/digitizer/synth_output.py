"""Synthetic example export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .annotation_io import CLASS_MAPPING, annotation_to_yolo_line


def _build_frame_annotations(plot_box: dict[str, int]) -> list[dict[str, Any]]:
    """Build synthetic frame annotations from plot bounds."""
    left = float(plot_box["left"])
    top = float(plot_box["top"])
    right = float(plot_box["right"])
    bottom = float(plot_box["bottom"])
    x_mid = (left + right) / 2.0
    y_mid = (top + bottom) / 2.0
    return [
        {"type": "plot_area", "points": [(left, top), (right, bottom)]},
        {"type": "x_axis", "points": [(left, bottom), (right, bottom)]},
        {"type": "y_axis", "points": [(left, top), (left, bottom)]},
        {"type": "x_anchor", "points": [(left, bottom)]},
        {"type": "x_anchor", "points": [(right, bottom)]},
        {"type": "y_anchor", "points": [(left, bottom)]},
        {"type": "y_anchor", "points": [(left, top)]},
        {"type": "x_anchor", "points": [(x_mid, bottom)]},
        {"type": "y_anchor", "points": [(left, y_mid)]},
    ]


def _descriptors_to_pixel_annotations(
    annotation_descriptors: list[dict[str, Any]],
    plot_box: dict[str, int],
    y_range: tuple[float, float],
) -> list[dict[str, Any]]:
    """Convert synth annotation descriptors to pixel-space annotator format.

    The returned annotations use the same schema as those written by the
    interactive annotator (``save_training_sample``), so they can be loaded
    and edited in a subsequent ``digitizer annotate`` session.

    Descriptor coordinate conventions (all 0-1 normalised within the plot area
    unless noted):

    * ``vbar``:     ``x_pos``  — normalised x within axes
    * ``hbar``:     ``y_pos``  — data-space y value (converted via y_range)
    * ``arrow``:    ``start``, ``end`` — (norm_x, norm_y) pairs within axes
    * ``error_bar``: ``x_pos``, ``y_pos``, ``y_err`` — all normalised within axes

    Curve descriptors are omitted because they carry no pixel-space coordinates.
    """
    result: list[dict[str, Any]] = []
    pb_left = float(plot_box["left"])
    pb_right = float(plot_box["right"])
    pb_top = float(plot_box["top"])
    pb_bottom = float(plot_box["bottom"])
    pb_w = pb_right - pb_left
    pb_h = pb_bottom - pb_top  # positive: bottom > top in pixel space

    for desc in annotation_descriptors:
        t = desc.get("type")
        if t == "vbar":
            px = pb_left + float(desc["x_pos"]) * pb_w
            result.append({"type": "vbar", "points": [(px, pb_top)]})
        elif t == "hbar":
            y_span = float(y_range[1]) - float(y_range[0])
            y_norm = (float(desc["y_pos"]) - float(y_range[0])) / y_span if y_span != 0 else 0.5
            py = pb_bottom - y_norm * pb_h
            result.append({"type": "hbar", "points": [(pb_left, py)]})
        elif t == "arrow":
            sx, sy = float(desc["start"][0]), float(desc["start"][1])
            ex, ey = float(desc["end"][0]), float(desc["end"][1])
            start_px = (pb_left + sx * pb_w, pb_bottom - sy * pb_h)
            end_px = (pb_left + ex * pb_w, pb_bottom - ey * pb_h)
            result.append({"type": "arrow", "points": [start_px, end_px]})
        elif t == "error_bar":
            px = pb_left + float(desc["x_pos"]) * pb_w
            py = pb_bottom - float(desc["y_pos"]) * pb_h
            py_err = float(desc["y_err"]) * pb_h
            result.append({"type": "error_bar", "points": [(px, py - py_err), (px, py + py_err)]})
        # curve descriptors carry no pixel-space coordinates; skip.

    return result


def _extract_tick_label_annotations(
    ax: Any,
    renderer: Any,
    width_px: int,
    height_px: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract axis tick labels as bounding-box annotations and YOLO label lines.

    Each visible tick label (x or y) becomes a ``x_tick_label`` / ``y_tick_label``
    annotation with:
    * ``points``: ``[(x0, y0), (x1, y1)]`` bounding box in pixel space (image coords).
    * ``text``: the tick label text (e.g. ``"0.5"``), used for future range inference.

    Returns a pair ``(pixel_annotations, label_lines)`` to be merged into the
    main annotation / label collections.
    """
    x_class = CLASS_MAPPING["x_tick_label"]
    y_class = CLASS_MAPPING["y_tick_label"]
    pixel_annotations: list[dict[str, Any]] = []
    label_lines: list[str] = []

    for tick_label, ann_type, class_id in (
        *[(t, "x_tick_label", x_class) for t in ax.get_xticklabels()],
        *[(t, "y_tick_label", y_class) for t in ax.get_yticklabels()],
    ):
        text = tick_label.get_text().strip()
        if not text:
            continue
        try:
            bb = tick_label.get_window_extent(renderer=renderer)
        except Exception:
            continue
        # Convert display coords (origin bottom-left) to image coords (origin top-left).
        x0 = float(np.clip(bb.x0, 0, width_px))
        x1 = float(np.clip(bb.x1, 0, width_px))
        y0 = float(np.clip(height_px - bb.y1, 0, height_px))
        y1 = float(np.clip(height_px - bb.y0, 0, height_px))
        if x1 <= x0 or y1 <= y0:
            continue
        pixel_annotations.append({
            "type": ann_type,
            "points": [(x0, y0), (x1, y1)],
            "text": text,
        })
        # YOLO bounding-box polygon: 4 corners (x0y0, x1y0, x1y1, x0y1).
        nx0, nx1 = x0 / width_px, x1 / width_px
        ny0, ny1 = y0 / height_px, y1 / height_px
        label_lines.append(
            f"{class_id} {nx0:.6f} {ny0:.6f} {nx1:.6f} {ny0:.6f} "
            f"{nx1:.6f} {ny1:.6f} {nx0:.6f} {ny1:.6f}"
        )

    return pixel_annotations, label_lines


def _save_synthetic_outputs(
    fig: Any,
    ax: Any,
    image_path: Path,
    image_format: str,
    label_path: Path,
    metadata_path: Path,
    annotations_path: Path,
    csv_path: Path,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    use_log_x: bool,
    plot_type: str,
    ground_truth_frames: list[pd.DataFrame],
    label_lines: list[str],
    curve_descriptors: list[dict[str, Any]],
    annotation_descriptors: list[dict[str, Any]],
) -> None:
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axis_bbox = ax.get_window_extent(renderer=renderer)
    width_px, height_px = fig.canvas.get_width_height()
    plot_box = {
        "left": int(axis_bbox.x0),
        "top": int(height_px - axis_bbox.y1),
        "right": int(axis_bbox.x1),
        "bottom": int(height_px - axis_bbox.y0),
    }
    frame_annotations = _build_frame_annotations(plot_box)
    frame_label_lines: list[str] = []
    for ann in frame_annotations:
        line = annotation_to_yolo_line(ann, width_px, height_px)
        if line:
            frame_label_lines.append(line)

    tick_annotations, tick_label_lines = _extract_tick_label_annotations(ax, renderer, width_px, height_px)

    fig.savefig(image_path, dpi=fig.dpi, format=image_format)
    ground_truth = pd.concat(ground_truth_frames, ignore_index=True)
    ground_truth.to_csv(csv_path, index=False)
    all_label_lines = label_lines + frame_label_lines + tick_label_lines
    label_path.write_text("\n".join(all_label_lines))

    # Build all pixel-space annotations: frame elements + vbar/hbar/arrow/error_bar
    # reconstructed from their descriptors using the plot_box, plus tick labels.
    # This uses the same schema as save_training_sample so that `digitizer annotate`
    # can load and edit them in a subsequent session.
    pixel_annotations = _descriptors_to_pixel_annotations(annotation_descriptors, plot_box, y_range)
    pixel_annotations.extend(frame_annotations)
    pixel_annotations.extend(tick_annotations)
    annotations_path.write_text(json.dumps({
        "image": str(image_path),
        "image_width": width_px,
        "image_height": height_px,
        "annotations": pixel_annotations,
    }, indent=2))

    # Metadata carries image-level and axis properties; annotation descriptors
    # (vbar, hbar, etc.) remain here as informational records but do NOT include
    # pixel-space points — those live in annotations_path.
    metadata_path.write_text(json.dumps({
        "image": str(image_path),
        "image_width": width_px,
        "image_height": height_px,
        "x_range": list(x_range),
        "y_range": list(y_range),
        "x_scale": "log" if use_log_x else "linear",
        "y_scale": "linear",
        "invert_y": False,
        "plot_box": plot_box,
        "plot_type": plot_type,
        "curves": curve_descriptors,
        "annotations": annotation_descriptors,
        "annotations_path": str(annotations_path),
        "csv_path": str(csv_path),
    }, indent=2))
