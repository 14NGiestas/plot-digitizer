"""Point extraction and coordinate conversion helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from skimage import measure

from .constants import MIN_COMPONENT_PIXELS
from .models import AxisCalibration, PlotBox, SegmentationResult

def _split_large_components(mask: np.ndarray) -> list[np.ndarray]:
    labeled = measure.label(mask)
    separated: list[np.ndarray] = []
    for region in measure.regionprops(labeled):
        if region.area < MIN_COMPONENT_PIXELS:
            continue
        component = labeled == region.label
        separated.append(component)
    return separated or [mask]


def extract_curve_points(segmentation: SegmentationResult, plot_box: PlotBox, smoothing_window: int = 9) -> pd.DataFrame:
    """Sample one y-value per x-position from a segmentation mask."""
    parts = _split_large_components(segmentation.mask) if segmentation.split_components else [segmentation.mask]
    frames: list[pd.DataFrame] = []
    for part_index, mask in enumerate(parts):
        local = mask[plot_box.top : plot_box.bottom, plot_box.left : plot_box.right]
        xs: list[int] = []
        ys: list[float] = []
        for x_pos in range(local.shape[1]):
            y_positions = np.flatnonzero(local[:, x_pos])
            if y_positions.size == 0:
                continue
            xs.append(plot_box.left + x_pos)
            ys.append(plot_box.top + float(np.median(y_positions)))
        if len(xs) < 8:
            continue
        y_array = np.asarray(ys)
        if len(y_array) >= smoothing_window and smoothing_window >= 5:
            window_size = smoothing_window if smoothing_window % 2 == 1 else smoothing_window + 1
            y_array = savgol_filter(y_array, window_size, 2)
        frame = pd.DataFrame(
            {
                "dataset_id": f"{segmentation.dataset_id}_{part_index}" if len(parts) > 1 else segmentation.dataset_id,
                "x_px": np.asarray(xs, dtype=float),
                "y_px": y_array.astype(float),
                "confidence": segmentation.confidence,
                "segmentation_method": segmentation.method,
            }
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["dataset_id", "x_px", "y_px", "confidence", "segmentation_method"])
    return pd.concat(frames, ignore_index=True)


def convert_points(points: pd.DataFrame, calibration: AxisCalibration, plot_box: PlotBox) -> pd.DataFrame:
    """Convert pixel-space curve points into real coordinates."""
    if points.empty:
        return points
    x_real, y_real = calibration.pixel_to_real(points["x_px"].to_numpy(), points["y_px"].to_numpy(), plot_box)
    converted = points.copy()
    converted["x_real"] = x_real
    converted["y_real"] = y_real
    return converted[["dataset_id", "x_real", "y_real", "confidence", "segmentation_method", "x_px", "y_px"]]

