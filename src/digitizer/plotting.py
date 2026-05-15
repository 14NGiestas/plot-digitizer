"""Overlay and replot export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import DEFAULT_DPI, MAX_REPLOT_LEGEND_DATASETS, MAX_REPLOT_POINTS
from .curve_utils import _interp_curve, _prepare_curve_points
from .models import AxisCalibration, SegmentationResult

def create_overlay(image: np.ndarray, points: pd.DataFrame, segmentations: Sequence[SegmentationResult], output_path: Path) -> None:
    """Write an overlay visualization with masks and extracted points."""
    overlay = image.copy()
    palette = [
        (255, 99, 71),
        (65, 105, 225),
        (50, 205, 50),
        (186, 85, 211),
        (255, 165, 0),
    ]
    for index, segmentation in enumerate(segmentations):
        color = palette[index % len(palette)]
        overlay[segmentation.mask] = (0.35 * overlay[segmentation.mask] + 0.65 * np.array(color)).astype(np.uint8)
    dataset_colors = {
        dataset_id: palette[index % len(palette)]
        for index, dataset_id in enumerate(points["dataset_id"].drop_duplicates())
    }
    for dataset_id, x_px, y_px in points[["dataset_id", "x_px", "y_px"]].itertuples(index=False, name=None):
        cv2.circle(overlay, (int(x_px), int(y_px)), 1, dataset_colors[dataset_id], -1)
    cv2.imwrite(str(output_path), overlay)


def build_replot_frame(points: pd.DataFrame, x_scale: str = "linear", max_points: int = MAX_REPLOT_POINTS) -> pd.DataFrame:
    """Convert tidy digitized points into a wide `x_real + dataset columns` frame.

    `max_points` caps the shared interpolation grid density used for the export.
    Values outside each dataset's observed x-range are left as `NaN`.
    """
    if points.empty:
        return pd.DataFrame(columns=["x_real"])
    dataset_frames: list[tuple[str, pd.DataFrame]] = []
    longest_dataset_length = 2
    for dataset_id, dataset_points in points.groupby("dataset_id", sort=True):
        unique = _prepare_curve_points(dataset_points)
        if len(unique) < 2:
            continue
        dataset_frames.append((str(dataset_id), unique))
        longest_dataset_length = max(longest_dataset_length, len(unique))
    if not dataset_frames:
        return pd.DataFrame(columns=["x_real"])
    point_count = min(max_points, longest_dataset_length)
    x_min = min(float(frame["x_real"].min()) for _, frame in dataset_frames)
    x_max = max(float(frame["x_real"].max()) for _, frame in dataset_frames)
    if x_scale == "log":
        reference_x = np.geomspace(x_min, x_max, point_count)
    else:
        reference_x = np.linspace(x_min, x_max, point_count)
    replot_frame = pd.DataFrame({"x_real": reference_x})
    for dataset_id, dataset_points in dataset_frames:
        y_values = _interp_curve(dataset_points, reference_x)
        valid = (reference_x >= float(dataset_points["x_real"].min())) & (reference_x <= float(dataset_points["x_real"].max()))
        replot_frame[dataset_id] = np.where(valid, y_values, np.nan)
    return replot_frame


def create_replot(replot_frame: pd.DataFrame, calibration: AxisCalibration, image_name: str, output_path: Path) -> Path:
    """Write a clean PNG replot for visual evaluation and return `output_path`."""
    figure, axis = plt.subplots(figsize=(6.0, 4.2), dpi=DEFAULT_DPI)
    plotted_columns = 0
    for column in replot_frame.columns:
        if column == "x_real":
            continue
        series = replot_frame[["x_real", column]].dropna()
        if series.empty:
            continue
        axis.plot(series["x_real"], series[column], linewidth=2.0, label=column)
        plotted_columns += 1
    axis.set_title(f"Digitized replot: {image_name}")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_xscale(calibration.x_scale)
    axis.set_yscale(calibration.y_scale)
    axis.set_xlim(calibration.x_min, calibration.x_max)
    if calibration.invert_y:
        axis.set_ylim(calibration.y_max, calibration.y_min)
    else:
        axis.set_ylim(calibration.y_min, calibration.y_max)
    axis.grid(True, linestyle=":", alpha=0.35)
    if 0 < plotted_columns <= MAX_REPLOT_LEGEND_DATASETS:
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=DEFAULT_DPI)
    plt.close(figure)
    return output_path

