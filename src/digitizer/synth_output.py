"""Synthetic example export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


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
    _, height_px = fig.canvas.get_width_height()
    plot_box = {
        "left": int(axis_bbox.x0),
        "top": int(height_px - axis_bbox.y1),
        "right": int(axis_bbox.x1),
        "bottom": int(height_px - axis_bbox.y0),
    }
    fig.savefig(image_path, dpi=fig.dpi, format=image_format)
    ground_truth = pd.concat(ground_truth_frames, ignore_index=True)
    ground_truth.to_csv(ground_truth_path, index=False)
    label_path.write_text("\n".join(label_lines))
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
        "ground_truth_csv": str(ground_truth_path),
    }, indent=2))
