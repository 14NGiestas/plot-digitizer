"""Synthetic example export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .annotation_io import annotation_to_yolo_line


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


def _save_synthetic_outputs(
    fig: Any,
    ax: Any,
    image_path: Path,
    image_format: str,
    label_path: Path,
    metadata_path: Path,
    ground_truth_path: Path,
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
    axis_bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
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

    fig.savefig(image_path, dpi=fig.dpi, format=image_format)
    ground_truth = pd.concat(ground_truth_frames, ignore_index=True)
    ground_truth.to_csv(ground_truth_path, index=False)
    all_label_lines = label_lines + frame_label_lines
    label_path.write_text("\n".join(all_label_lines))
    metadata_path.write_text(json.dumps({
        "image": str(image_path),
        "x_range": list(x_range),
        "y_range": list(y_range),
        "x_scale": "log" if use_log_x else "linear",
        "y_scale": "linear",
        "invert_y": False,
        "plot_box": plot_box,
        "plot_type": plot_type,
        "curves": curve_descriptors,
        "annotations": annotation_descriptors,
        "frame_annotations": frame_annotations,
        "ground_truth_csv": str(ground_truth_path),
    }, indent=2))
