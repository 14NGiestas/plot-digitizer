"""Shared data models for digitizer workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .math_utils import _norm_to_scale

AxisReferencePair = tuple[tuple[float, float], tuple[float, float]]


@dataclass(slots=True)
class PlotBox:
    """Pixel bounds of the plot area."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(1, self.right - self.left)

    @property
    def height(self) -> int:
        return max(1, self.bottom - self.top)

    @property
    def origin(self) -> tuple[int, int]:
        return (self.left, self.bottom)


@dataclass(slots=True)
class AxisCalibration:
    """Mapping from pixel coordinates to real coordinates."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    x_scale: str = "linear"
    y_scale: str = "linear"
    invert_y: bool = False
    x_pixel_min: float | None = None
    x_pixel_max: float | None = None
    y_pixel_bottom: float | None = None
    y_pixel_top: float | None = None

    def pixel_to_real(self, x_px: np.ndarray, y_px: np.ndarray, plot_box: PlotBox) -> tuple[np.ndarray, np.ndarray]:
        """Convert pixel arrays to real-world coordinates."""
        x_left = float(self.x_pixel_min if self.x_pixel_min is not None else plot_box.left)
        x_right = float(self.x_pixel_max if self.x_pixel_max is not None else plot_box.right)
        y_bottom = float(self.y_pixel_bottom if self.y_pixel_bottom is not None else plot_box.bottom)
        y_top = float(self.y_pixel_top if self.y_pixel_top is not None else plot_box.top)
        x_span = x_right - x_left
        y_span = y_bottom - y_top
        if x_span <= 0:
            raise ValueError("X-axis calibration pixel bounds are invalid (right must be greater than left).")
        if y_span <= 0:
            raise ValueError("Y-axis calibration pixel bounds are invalid (bottom must be greater than top).")
        x_norm = np.clip((x_px - x_left) / x_span, 0.0, 1.0)
        y_norm = np.clip((y_bottom - y_px) / y_span, 0.0, 1.0)
        if self.invert_y:
            y_norm = 1.0 - y_norm
        x_real = _norm_to_scale(x_norm, self.x_min, self.x_max, self.x_scale)
        y_real = _norm_to_scale(y_norm, self.y_min, self.y_max, self.y_scale)
        return x_real, y_real


@dataclass(slots=True)
class SegmentationResult:
    """Mask and metadata for one detected curve."""

    dataset_id: str
    mask: np.ndarray
    confidence: float
    method: str
    split_components: bool = False


@dataclass(slots=True)
class DigitizeResult:
    """Final digitization payload for one image."""

    csv_path: Path
    replot_csv_path: Path
    metadata_path: Path
    replot_path: Path
    overlay_path: Path | None
    point_count: int
    dataset_count: int

