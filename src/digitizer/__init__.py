#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.26",
#   "matplotlib>=3.8",
#   "opencv-python>=4.10",
#   "pandas>=2.2",
#   "scikit-image>=0.24",
#   "scipy>=1.13",
#   "scikit-learn>=1.5",
#   "ultralytics>=8.3",
# ]
# ///
"""Automatic plot digitizer CLI."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import math
import multiprocessing
import os
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import matplotlib

# Defer backend selection until argument parsing to support interactive mode
_MATPLOTLIB_BACKEND_SET = False


def _set_matplotlib_backend(backend: str) -> None:
    """Set matplotlib backend before importing pyplot."""
    global _MATPLOTLIB_BACKEND_SET
    if not _MATPLOTLIB_BACKEND_SET:
        matplotlib.use(backend)
        _MATPLOTLIB_BACKEND_SET = True


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment
from scipy.signal import savgol_filter
from skimage import measure, morphology
from sklearn.cluster import DBSCAN, MiniBatchKMeans

LOGGER = logging.getLogger("plot_digitizer")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEFAULT_X_RANGE = (0.0, 1.0)
DEFAULT_Y_RANGE = (0.0, 1.0)
MIN_COMPONENT_PIXELS = 24
PLOT_MARGIN_FRACTION = 0.02
DEFAULT_DPI = 140
DBSCAN_NOISE_LABEL = -1
MAX_DARK_THRESHOLD = 220
DARK_PIXEL_PERCENTILE = 82
BASE_CV_CONFIDENCE = 0.45
MAX_CV_CONFIDENCE = 0.99
MAX_POLYGON_POINTS = 200
MAX_COLOR_CLUSTERS = 4
MAX_CLUSTER_SAMPLE_SIZE = 2000
MINIBATCH_KMEANS_BATCH_SIZE = 1024
MIN_CURVES_PER_PLOT = 1
MAX_CURVES_PER_PLOT = 3
VALIDATION_THRESHOLD = 0.05
MAX_REPLOT_POINTS = 600
MAX_REPLOT_LEGEND_DATASETS = 10
DEFAULT_GENERATE_WORKERS_CAP = 8
INTERACTIVE_CLICK_RADIUS_SCALE = 0.02
INTERACTIVE_ZOOM_HALF_SIZE_SCALE = 0.06
INTERACTIVE_ZOOM_HALF_SIZE_MIN = 24
INTERACTIVE_SELECTION_HELP_TEXT = (
    "Left click to add/select and drag points. Right click removes nearest point. "
    "Press Enter when 4 points are set (X1, X2, Y1, Y2). Press Esc to cancel."
)
INTERACTIVE_SELECTION_LIMIT_REACHED_TEXT = (
    "Already selected 4 points. Drag existing points or right-click to remove one."
)
INTERACTIVE_SELECTION_REMOVED_TEXT = (
    "Point removed. Left click to add/select and drag. Right click removes nearest point."
)
SINE_AMPLITUDE_RANGE = (0.5, 1.8)
SINE_FREQUENCY_RANGE = (0.6, 2.4)
SINE_OFFSET_RANGE = (-0.75, 0.75)
EXP_SCALE_RANGE = (0.2, 1.1)
EXP_GROWTH_RANGE = (0.15, 0.55)
EXP_OFFSET_RANGE = (-0.8, 0.3)
DAMPED_AMPLITUDE_RANGE = (0.8, 1.8)
DAMPED_DECAY_RANGE = (0.05, 0.2)
DAMPED_FREQUENCY_RANGE = (1.0, 2.6)
POLY_A_RANGE = (-0.05, 0.05)
POLY_B_RANGE = (-0.4, 0.4)
POLY_C_RANGE = (-0.8, 0.8)
NOISE_STD_RANGE = (0.01, 0.05)
DENSE_CURVE_PROBABILITY = 0.4
DENSE_CURVE_COUNT_RANGE = (4, 6)
BASE_CURVE_COUNT_RANGE = (2, 4)
VBAR_COUNT_RANGE = (1, 3)
HBAR_COUNT_RANGE = (1, 2)
ARROW_COUNT_RANGE = (0, 2)
ERROR_BAR_COUNT_RANGE = (2, 5)
CURVE_LINEWIDTHS = [0.6, 0.8, 1.0, 1.2, 1.6, 2.0]
CURVE_LINEWIDTH_PROBABILITIES = [0.28, 0.24, 0.2, 0.14, 0.09, 0.05]
GRID_ENABLED_PROBABILITY = 0.6
GRID_ALPHA = 0.4
LOG_X_PROBABILITY = 0.3
LOG_X_MIN = 0.1
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


def _norm_to_scale(values: np.ndarray, minimum: float, maximum: float, scale: str) -> np.ndarray:
    if scale == "log":
        if minimum <= 0 or maximum <= 0:
            raise ValueError("Logarithmic axes require positive bounds.")
        return np.exp(np.log(minimum) + values * (np.log(maximum) - np.log(minimum)))
    return minimum + values * (maximum - minimum)


def _remove_small_regions(mask: np.ndarray, min_area: int) -> np.ndarray:
    cleaned = morphology.area_opening(mask.astype(np.uint8), area_threshold=min_area)
    return cleaned.astype(bool)


def _rectangle(height: int, width: int) -> np.ndarray:
    if hasattr(morphology, "footprint_rectangle"):
        return morphology.footprint_rectangle((height, width))
    return morphology.rectangle(height, width)


def configure_logging(verbose: bool) -> None:
    """Configure structured logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def parse_range(value: str | None, default: tuple[float, float] | None = None) -> tuple[float, float] | None:
    """Parse a min:max range."""
    if value is None:
        return default
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected min:max, got {value!r}")
    start, end = (float(part) for part in parts)
    if start == end:
        raise argparse.ArgumentTypeError("Range start and end must differ.")
    return (start, end)


def parse_reference_pair(value: str | None, axis_name: str) -> AxisReferencePair | None:
    """Parse axis reference points in `px0:real0,px1:real1` format."""
    if value is None:
        return None
    points = value.split(",")
    if len(points) != 2:
        raise argparse.ArgumentTypeError(f"Expected two {axis_name}-axis points in px0:real0,px1:real1 format.")
    parsed: list[tuple[float, float]] = []
    for point in points:
        parts = point.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"Invalid {axis_name}-axis reference point: {point!r}")
        pixel_value, real_value = (float(part.strip()) for part in parts)
        parsed.append((pixel_value, real_value))
    if parsed[0][0] == parsed[1][0]:
        raise argparse.ArgumentTypeError(f"{axis_name.upper()}-axis reference pixel positions must differ.")
    return parsed[0], parsed[1]


def _resolve_bounds_from_references(
    plot_box: PlotBox,
    first: tuple[float, float],
    second: tuple[float, float],
    axis_name: str,
    scale: str,
    invert_y: bool,
) -> tuple[float, float]:
    """Infer full axis bounds from two known pixel-to-real reference points."""
    pixel_first, real_first = first
    pixel_second, real_second = second
    if axis_name == "x":
        norm_first = np.clip((pixel_first - plot_box.left) / plot_box.width, 0.0, 1.0)
        norm_second = np.clip((pixel_second - plot_box.left) / plot_box.width, 0.0, 1.0)
    else:
        norm_first = np.clip((plot_box.bottom - pixel_first) / plot_box.height, 0.0, 1.0)
        norm_second = np.clip((plot_box.bottom - pixel_second) / plot_box.height, 0.0, 1.0)
        if invert_y:
            norm_first = 1.0 - norm_first
            norm_second = 1.0 - norm_second
    if np.isclose(norm_first, norm_second):
        raise ValueError(f"{axis_name.upper()}-axis reference points map to the same normalized position.")

    if scale == "log":
        if real_first <= 0 or real_second <= 0:
            raise ValueError(f"{axis_name.upper()}-axis logarithmic references require positive real values.")
        transformed_first = float(np.log(real_first))
        transformed_second = float(np.log(real_second))
        slope = (transformed_second - transformed_first) / (norm_second - norm_first)
        minimum = transformed_first - norm_first * slope
        maximum = minimum + slope
        return float(np.exp(minimum)), float(np.exp(maximum))

    slope = (real_second - real_first) / (norm_second - norm_first)
    minimum = real_first - norm_first * slope
    maximum = minimum + slope
    return float(minimum), float(maximum)


def interactive_reference_selection(image_path: Path) -> tuple[AxisReferencePair, AxisReferencePair]:
    """Collect two X-axis and two Y-axis calibration points interactively."""
    image = cv2.cvtColor(load_image(image_path), cv2.COLOR_BGR2RGB)
    image_height, image_width = image.shape[:2]
    figure, (axis, zoom_axis) = plt.subplots(
        ncols=2,
        figsize=(13, 7),
        gridspec_kw={"width_ratios": [3.2, 1.4]},
    )
    axis.imshow(image)
    axis.axis("off")
    zoom_axis.axis("off")
    figure.suptitle(
        "Interactive axis calibration",
        fontsize=13,
    )

    info_text = figure.text(
        0.02,
        0.02,
        INTERACTIVE_SELECTION_HELP_TEXT,
        fontsize=9,
    )

    point_labels = ("X1", "X2", "Y1", "Y2")
    point_colors = ("tab:blue", "tab:blue", "tab:orange", "tab:orange")
    points: list[tuple[float, float]] = []
    point_artists: list[Any] = []
    label_artists: list[Any] = []
    dragging_index: int | None = None
    cancelled = False
    click_radius = max(10.0, float(max(image_width, image_height)) * INTERACTIVE_CLICK_RADIUS_SCALE)
    zoom_half_size = max(
        INTERACTIVE_ZOOM_HALF_SIZE_MIN,
        max(image_width, image_height) * INTERACTIVE_ZOOM_HALF_SIZE_SCALE,
    )

    def _distance(x0: float, y0: float, x1: float, y1: float) -> float:
        return float(np.hypot(x1 - x0, y1 - y0))

    def _nearest_point_index(x_coord: float, y_coord: float) -> tuple[int | None, float]:
        if not points:
            return None, float("inf")
        best_index = 0
        best_distance = _distance(x_coord, y_coord, points[0][0], points[0][1])
        for index in range(1, len(points)):
            candidate_distance = _distance(x_coord, y_coord, points[index][0], points[index][1])
            if candidate_distance < best_distance:
                best_index = index
                best_distance = candidate_distance
        return best_index, best_distance

    def _refresh_zoom(active_index: int | None) -> None:
        zoom_axis.clear()
        zoom_axis.imshow(image)
        zoom_axis.axis("off")
        if active_index is None or active_index >= len(points):
            zoom_axis.set_title("Zoom (select/move a point)", fontsize=10)
            return
        x_coord, y_coord = points[active_index]
        x_min = max(0.0, x_coord - zoom_half_size)
        x_max = min(image_width - 1, x_coord + zoom_half_size)
        y_min = max(0.0, y_coord - zoom_half_size)
        y_max = min(image_height - 1, y_coord + zoom_half_size)
        zoom_axis.set_xlim(x_min, x_max)
        zoom_axis.set_ylim(y_max, y_min)
        zoom_axis.set_title(f"Zoom: {point_labels[active_index]} ({x_coord:.1f}, {y_coord:.1f})", fontsize=10)
        zoom_axis.axvline(x_coord, color="yellow", linestyle="--", linewidth=1.2)
        zoom_axis.axhline(y_coord, color="yellow", linestyle="--", linewidth=1.2)
        for index, (point_x, point_y) in enumerate(points):
            marker = "o" if index == active_index else "x"
            zoom_axis.plot(point_x, point_y, marker=marker, color=point_colors[index], markersize=7)

    def _redraw(active_index: int | None = None) -> None:
        nonlocal point_artists, label_artists
        for artist in point_artists + label_artists:
            artist.remove()
        point_artists = []
        label_artists = []
        for index, (x_coord, y_coord) in enumerate(points):
            point_artist = axis.plot(
                x_coord,
                y_coord,
                marker="o",
                markersize=8,
                color=point_colors[index],
                markeredgecolor="white",
                markeredgewidth=0.8,
                linestyle="None",
            )[0]
            label_artist = axis.text(
                x_coord + 6.0,
                y_coord - 6.0,
                point_labels[index],
                color=point_colors[index],
                fontsize=9,
                weight="bold",
                bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 1.5},
            )
            point_artists.append(point_artist)
            label_artists.append(label_artist)
        axis.set_title(
            "Pick points in order: X1, X2, Y1, Y2 "
            f"({len(points)}/4 selected)",
            fontsize=10,
        )
        if active_index is None and points:
            active_index = len(points) - 1
        _refresh_zoom(active_index)
        figure.canvas.draw_idle()

    def _on_click(event: Any) -> None:
        nonlocal dragging_index
        if event.inaxes is not axis or event.xdata is None or event.ydata is None:
            return
        nearest_index, nearest_distance = _nearest_point_index(float(event.xdata), float(event.ydata))
        if event.button == 1:
            if nearest_index is not None and nearest_distance <= click_radius:
                dragging_index = nearest_index
            elif len(points) < 4:
                points.append((float(event.xdata), float(event.ydata)))
                dragging_index = len(points) - 1
            else:
                info_text.set_text(INTERACTIVE_SELECTION_LIMIT_REACHED_TEXT)
                dragging_index = nearest_index if nearest_index is not None else None
            _redraw(dragging_index)
        elif event.button == 3 and nearest_index is not None and nearest_distance <= click_radius:
            points.pop(nearest_index)
            dragging_index = None
            info_text.set_text(INTERACTIVE_SELECTION_REMOVED_TEXT)
            _redraw(None)

    def _on_motion(event: Any) -> None:
        if dragging_index is None or event.inaxes is not axis or event.xdata is None or event.ydata is None:
            return
        x_coord = float(np.clip(event.xdata, 0.0, float(image_width - 1)))
        y_coord = float(np.clip(event.ydata, 0.0, float(image_height - 1)))
        points[dragging_index] = (x_coord, y_coord)
        _redraw(dragging_index)

    def _on_release(_event: Any) -> None:
        nonlocal dragging_index
        dragging_index = None

    def _on_key(event: Any) -> None:
        nonlocal cancelled
        if event.key in ("enter", "return"):
            if len(points) == 4:
                plt.close(figure)
            else:
                info_text.set_text(f"Need exactly 4 points before continuing (currently {len(points)}).")
                figure.canvas.draw_idle()
        elif event.key in ("escape", "q"):
            cancelled = True
            plt.close(figure)

    figure.canvas.mpl_connect("button_press_event", _on_click)
    figure.canvas.mpl_connect("motion_notify_event", _on_motion)
    figure.canvas.mpl_connect("button_release_event", _on_release)
    figure.canvas.mpl_connect("key_press_event", _on_key)
    _redraw(None)
    plt.show()

    if cancelled:
        raise RuntimeError("Interactive calibration cancelled.")
    if len(points) != 4:
        raise RuntimeError("Interactive calibration requires selecting exactly 4 points.")

    selected_points = points
    x_axis_points: list[tuple[float, float]] = []
    y_axis_points: list[tuple[float, float]] = []
    for index, (x_coord, y_coord) in enumerate(selected_points):
        if index < 2:
            x_point_index = index + 1
            real_value = float(input(f"Enter real X value for X point {x_point_index} at pixel x={x_coord:.1f}: ").strip())
            x_axis_points.append((float(x_coord), real_value))
        else:
            y_point_index = index - 2 + 1
            real_value = float(input(f"Enter real Y value for Y point {y_point_index} at pixel y={y_coord:.1f}: ").strip())
            y_axis_points.append((float(y_coord), real_value))
    if np.isclose(x_axis_points[0][0], x_axis_points[1][0]) or np.isclose(y_axis_points[0][0], y_axis_points[1][0]):
        raise RuntimeError("Interactive calibration points must use different pixel positions on each axis.")
    return (x_axis_points[0], x_axis_points[1]), (y_axis_points[0], y_axis_points[1])


def _format_reference_pair_cli_value(reference_pair: AxisReferencePair) -> str:
    """Format reference pairs for `--x-reference/--y-reference` CLI reuse.

    Pixel coordinates use fixed 6 decimals for stable, readable replay strings.
    Real values use 15 significant digits to preserve round-trip precision.
    """
    first, second = reference_pair
    return (
        f"{first[0]:.6f}:{first[1]:.15g},"
        f"{second[0]:.6f}:{second[1]:.15g}"
    )


def discover_images(inputs: Sequence[str]) -> list[Path]:
    """Expand input file and directory arguments into image paths."""
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(candidate for candidate in path.iterdir() if candidate.suffix.lower() in IMAGE_EXTENSIONS))
        elif path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
    return paths


def load_image(path: Path) -> np.ndarray:
    """Load an image using OpenCV."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {path}")
    return image


def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply denoising and light grid suppression."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    horizontal = cv2.morphologyEx(clahe, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(clahe, cv2.MORPH_OPEN, vertical_kernel)
    suppressed = cv2.subtract(clahe, cv2.addWeighted(horizontal, 0.5, vertical, 0.5, 0))
    stats = {
        "preprocessing_steps": [
            "gaussian_blur",
            "clahe_contrast_normalization",
            "horizontal_grid_suppression",
            "vertical_grid_suppression",
        ],
        "mean_intensity": float(np.mean(suppressed)),
        "std_intensity": float(np.std(suppressed)),
    }
    return suppressed, stats


def detect_plot_box(image: np.ndarray, processed_gray: np.ndarray) -> PlotBox:
    """Detect the plot box using dark-pixel projections and axis heuristics."""
    height, width = processed_gray.shape
    threshold = min(MAX_DARK_THRESHOLD, int(np.percentile(processed_gray, DARK_PIXEL_PERCENTILE)))
    dark_mask = processed_gray < threshold
    row_counts = dark_mask.sum(axis=1)
    col_counts = dark_mask.sum(axis=0)

    x_candidates = np.flatnonzero(col_counts > max(15, int(height * 0.08)))
    y_candidates = np.flatnonzero(row_counts > max(15, int(width * 0.08)))

    if x_candidates.size == 0 or y_candidates.size == 0:
        LOGGER.warning("Axis projection failed; using conservative full-image crop.")
        margin_x = int(width * 0.08)
        margin_y = int(height * 0.08)
        return PlotBox(margin_x, margin_y, width - margin_x, height - margin_y)

    y_axis = int(np.median(x_candidates[x_candidates < int(width * 0.5)])) if np.any(x_candidates < int(width * 0.5)) else int(x_candidates[0])
    x_axis = int(np.median(y_candidates[y_candidates > int(height * 0.45)])) if np.any(y_candidates > int(height * 0.45)) else int(y_candidates[-1])

    right_edge = int(np.max(np.flatnonzero(col_counts > max(5, int(height * 0.02)))))
    top_edge = int(np.min(np.flatnonzero(row_counts > max(5, int(width * 0.02)))))
    margin_x = max(3, int(width * PLOT_MARGIN_FRACTION))
    margin_y = max(3, int(height * PLOT_MARGIN_FRACTION))

    plot_box = PlotBox(
        left=max(0, y_axis + margin_x),
        top=max(0, top_edge + margin_y),
        right=min(width - 1, right_edge - margin_x),
        bottom=min(height - 1, x_axis - margin_y),
    )
    if plot_box.left >= plot_box.right or plot_box.top >= plot_box.bottom:
        LOGGER.warning("Detected invalid plot box; falling back to image bounds.")
        return PlotBox(margin_x, margin_y, width - margin_x, height - margin_y)
    return plot_box


def _parse_sidecar_metadata(image_path: Path) -> dict[str, Any] | None:
    candidates = [
        image_path.with_suffix(".metadata.json"),
        image_path.parent.parent / f"{image_path.stem}.metadata.json",
    ]
    for metadata_path in candidates:
        if metadata_path.exists():
            return json.loads(metadata_path.read_text())
    return None


def resolve_plot_box(image_path: Path, image: np.ndarray, processed_gray: np.ndarray) -> PlotBox:
    """Use sidecar plot bounds when available, otherwise detect them from the image."""
    sidecar = _parse_sidecar_metadata(image_path) or {}
    sidecar_plot_box = sidecar.get("plot_box")
    if isinstance(sidecar_plot_box, dict):
        return PlotBox(
            left=int(sidecar_plot_box["left"]),
            top=int(sidecar_plot_box["top"]),
            right=int(sidecar_plot_box["right"]),
            bottom=int(sidecar_plot_box["bottom"]),
        )
    return detect_plot_box(image, processed_gray)


def detect_axis_anchor_pixels(processed_gray: np.ndarray, plot_box: PlotBox) -> dict[str, tuple[float, float]] | None:
    """Estimate axis anchor pixel positions from strong dark-line projections."""
    crop = processed_gray[plot_box.top : plot_box.bottom, plot_box.left : plot_box.right]
    if crop.size == 0:
        return None
    threshold = min(MAX_DARK_THRESHOLD, int(np.percentile(crop, DARK_PIXEL_PERCENTILE)))
    dark_mask = crop < threshold
    row_counts = dark_mask.sum(axis=1)
    col_counts = dark_mask.sum(axis=0)
    if row_counts.max(initial=0) <= 0 or col_counts.max(initial=0) <= 0:
        return None

    height, width = crop.shape
    bottom_band_start = int(height * 0.6)
    left_band_end = max(1, int(width * 0.4))
    x_axis_idx_local = int(bottom_band_start + np.argmax(row_counts[bottom_band_start:]))
    y_axis_idx_local = int(np.argmax(col_counts[:left_band_end]))

    x_axis_dark = np.flatnonzero(dark_mask[x_axis_idx_local, :])
    y_axis_dark = np.flatnonzero(dark_mask[:, y_axis_idx_local])
    if x_axis_dark.size < 2 or y_axis_dark.size < 2:
        return None

    x_left = float(plot_box.left + x_axis_dark.min())
    x_right = float(plot_box.left + x_axis_dark.max())
    y_top = float(plot_box.top + y_axis_dark.min())
    y_bottom = float(plot_box.top + y_axis_dark.max())
    if np.isclose(x_left, x_right) or np.isclose(y_top, y_bottom):
        return None
    return {"x": (x_left, x_right), "y": (y_bottom, y_top)}


def calibrate_axes(
    image_path: Path,
    plot_box: PlotBox,
    processed_gray: np.ndarray | None,
    x_range: tuple[float, float] | None,
    y_range: tuple[float, float] | None,
    x_reference: AxisReferencePair | None,
    y_reference: AxisReferencePair | None,
    x_scale: str,
    y_scale: str,
    invert_y: bool,
    auto_axis_anchors: bool = True,
) -> tuple[AxisCalibration, dict[str, Any]]:
    """Resolve axis ranges from CLI hints, sidecars, OCR, or defaults."""
    sidecar = _parse_sidecar_metadata(image_path) or {}
    x_bounds = x_range or tuple(sidecar.get("x_range", DEFAULT_X_RANGE))
    y_bounds = y_range or tuple(sidecar.get("y_range", DEFAULT_Y_RANGE))
    used_auto_x = False
    used_auto_y = False
    auto_anchor_pixels: dict[str, tuple[float, float]] | None = None
    if auto_axis_anchors and processed_gray is not None and (x_reference is None or y_reference is None):
        auto_anchor_pixels = detect_axis_anchor_pixels(processed_gray, plot_box)
        if auto_anchor_pixels is not None and x_reference is None:
            used_auto_x = True
        if auto_anchor_pixels is not None and y_reference is None:
            used_auto_y = True

    if x_reference is not None:
        x_bounds = _resolve_bounds_from_references(
            plot_box=plot_box,
            first=x_reference[0],
            second=x_reference[1],
            axis_name="x",
            scale=x_scale,
            invert_y=invert_y,
        )
    if y_reference is not None:
        y_bounds = _resolve_bounds_from_references(
            plot_box=plot_box,
            first=y_reference[0],
            second=y_reference[1],
            axis_name="y",
            scale=y_scale,
            invert_y=invert_y,
        )
    calibration = AxisCalibration(
        x_min=float(x_bounds[0]),
        x_max=float(x_bounds[1]),
        y_min=float(y_bounds[0]),
        y_max=float(y_bounds[1]),
        x_scale=str(sidecar.get("x_scale", x_scale)),
        y_scale=str(sidecar.get("y_scale", y_scale)),
        invert_y=bool(sidecar.get("invert_y", invert_y)),
        x_pixel_min=(auto_anchor_pixels["x"][0] if used_auto_x and auto_anchor_pixels is not None else None),
        x_pixel_max=(auto_anchor_pixels["x"][1] if used_auto_x and auto_anchor_pixels is not None else None),
        y_pixel_bottom=(auto_anchor_pixels["y"][0] if used_auto_y and auto_anchor_pixels is not None else None),
        y_pixel_top=(auto_anchor_pixels["y"][1] if used_auto_y and auto_anchor_pixels is not None else None),
    )
    metadata = {
        "origin_pixel": {"x": plot_box.left, "y": plot_box.bottom},
        "axis_anchor_pixels": auto_anchor_pixels,
        "axis_detection": {
            "x_range_source": (
                "auto-anchor"
                if used_auto_x
                else ("reference" if x_reference else ("cli" if x_range else ("sidecar" if "x_range" in sidecar else "default")))
            ),
            "y_range_source": (
                "auto-anchor"
                if used_auto_y
                else ("reference" if y_reference else ("cli" if y_range else ("sidecar" if "y_range" in sidecar else "default")))
            ),
            "x_scale": calibration.x_scale,
            "y_scale": calibration.y_scale,
            "invert_y": calibration.invert_y,
        },
        "warnings": [],
    }
    if not x_range and "x_range" not in sidecar and not x_reference and not used_auto_x:
        metadata["warnings"].append("X-axis bounds were not auto-detected; defaulting to 0:1. Pass --x-range for better accuracy.")
    if not y_range and "y_range" not in sidecar and not y_reference and not used_auto_y:
        metadata["warnings"].append("Y-axis bounds were not auto-detected; defaulting to 0:1. Pass --y-range for better accuracy.")
    return calibration, metadata


def run_ai_segmentation(
    image: np.ndarray,
    plot_box: PlotBox,
    weights: str | None,
    conf_threshold: float,
    workers: int | None = None,
) -> list[SegmentationResult]:
    """Run YOLO segmentation if weights are available."""
    if not weights:
        return []
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - fallback path when ultralytics is unavailable
        LOGGER.warning("Ultralytics import failed, falling back to CV segmentation: %s", exc)
        return []

    model = YOLO(weights)
    predict_kwargs: dict[str, Any] = {"conf": conf_threshold, "verbose": False}
    if workers is not None:
        predict_kwargs["workers"] = workers
    predictions = model.predict(image, **predict_kwargs)
    results: list[SegmentationResult] = []
    if not predictions:
        return results
    masks = getattr(predictions[0], "masks", None)
    if masks is None or masks.data is None:
        return results
    for index, mask_tensor in enumerate(masks.data):
        mask = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8)
        cropped = np.zeros_like(mask)
        cropped[plot_box.top : plot_box.bottom, plot_box.left : plot_box.right] = mask[plot_box.top : plot_box.bottom, plot_box.left : plot_box.right]
        if cropped.sum() < MIN_COMPONENT_PIXELS:
            continue
        confidence = float(predictions[0].boxes.conf[index].cpu().item()) if predictions[0].boxes is not None else conf_threshold
        results.append(
            SegmentationResult(
                dataset_id=f"dataset_{index}",
                mask=cropped.astype(bool),
                confidence=confidence,
                method="ai",
            )
        )
    return results


def _foreground_mask(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    foreground = gray < np.percentile(gray, 96)
    foreground[:3, :] = False
    foreground[-3:, :] = False
    foreground[:, :3] = False
    foreground[:, -3:] = False
    return _remove_small_regions(foreground, MIN_COMPONENT_PIXELS)


def _saturated_mask(crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    saturated = (saturation > 45) & (value < 245)
    saturated = _remove_small_regions(saturated, MIN_COMPONENT_PIXELS)
    return morphology.closing(saturated, _rectangle(3, 7))


def _cluster_by_color(crop: np.ndarray, foreground: np.ndarray) -> list[np.ndarray]:
    pixels = crop[foreground]
    if len(pixels) < 100:
        return []
    filtered = pixels.astype(np.float32)
    sample_count = min(len(filtered), MAX_CLUSTER_SAMPLE_SIZE)
    sample_indices = np.linspace(0, len(filtered) - 1, sample_count).astype(int)
    sample = filtered[sample_indices]
    cluster_count = int(min(MAX_COLOR_CLUSTERS, max(1, len(np.unique(sample, axis=0)))))
    if cluster_count <= 1:
        return []
    model = MiniBatchKMeans(
        n_clusters=cluster_count,
        n_init=5,
        random_state=42,
        batch_size=MINIBATCH_KMEANS_BATCH_SIZE,
    )
    model.fit(sample)
    labels = model.predict(filtered)
    masks: list[np.ndarray] = []
    for cluster_id in range(cluster_count):
        cluster_mask = np.zeros(foreground.shape, dtype=bool)
        cluster_mask[foreground] = labels == cluster_id
        cluster_mask = morphology.closing(cluster_mask, _rectangle(3, 9))
        cluster_mask = _remove_small_regions(cluster_mask, 80)
        if cluster_mask.sum() >= MIN_COMPONENT_PIXELS:
            masks.append(cluster_mask)
    return masks


def _cluster_by_geometry(foreground: np.ndarray) -> list[np.ndarray]:
    ys, xs = np.nonzero(foreground)
    if len(xs) < MIN_COMPONENT_PIXELS:
        return []
    sample_size = min(len(xs), 1500)
    indices = np.linspace(0, len(xs) - 1, sample_size).astype(int)
    points = np.column_stack((xs[indices] / max(1, foreground.shape[1]), ys[indices] / max(1, foreground.shape[0])))
    clustering = DBSCAN(eps=0.04, min_samples=15).fit(points)
    masks: list[np.ndarray] = []
    for cluster_id in sorted(set(clustering.labels_) - {DBSCAN_NOISE_LABEL}):
        sample_mask = np.zeros(foreground.shape, dtype=bool)
        sample_mask[ys[indices][clustering.labels_ == cluster_id], xs[indices][clustering.labels_ == cluster_id]] = True
        sample_mask = morphology.binary_dilation(sample_mask, morphology.disk(2))
        sample_mask = morphology.binary_closing(sample_mask, morphology.disk(2))
        sample_mask &= foreground
        if sample_mask.sum() >= MIN_COMPONENT_PIXELS:
            masks.append(sample_mask)
    return masks


def run_cv_segmentation(image: np.ndarray, plot_box: PlotBox) -> list[SegmentationResult]:
    """Segment curves using color and geometric clustering."""
    crop = image[plot_box.top : plot_box.bottom, plot_box.left : plot_box.right]
    foreground = _saturated_mask(crop)
    if foreground.sum() < MIN_COMPONENT_PIXELS:
        foreground = _foreground_mask(crop)
    if foreground.sum() < MIN_COMPONENT_PIXELS:
        return []

    candidate_masks = _cluster_by_color(crop, foreground)
    method = "cv_color"
    split_components = False
    if not candidate_masks:
        candidate_masks = _cluster_by_geometry(foreground)
        method = "cv_geometry"
        split_components = True
    if not candidate_masks:
        candidate_masks = [foreground]
        method = "cv_binary"
        split_components = True

    results: list[SegmentationResult] = []
    for index, local_mask in enumerate(candidate_masks):
        local_mask = morphology.closing(local_mask, _rectangle(3, 9))
        local_mask = _remove_small_regions(local_mask, 120)
        horizontal_coverage = np.mean(np.any(local_mask, axis=0))
        if horizontal_coverage < 0.15:
            continue
        global_mask = np.zeros(image.shape[:2], dtype=bool)
        global_mask[plot_box.top : plot_box.bottom, plot_box.left : plot_box.right] = local_mask
        if global_mask.sum() < MIN_COMPONENT_PIXELS:
            continue
        confidence = float(
            min(
                MAX_CV_CONFIDENCE,
                BASE_CV_CONFIDENCE + global_mask.sum() / max(1, plot_box.width * plot_box.height),
            )
        )
        results.append(
            SegmentationResult(
                dataset_id=f"dataset_{index}",
                mask=global_mask,
                confidence=confidence,
                method=method,
                split_components=split_components,
            )
        )
    return results


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
    auto_axis_anchors: bool = True,
) -> DigitizeResult:
    """Digitize a single image and write CSV/JSON artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
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

    segmentations = run_ai_segmentation(
        image,
        plot_box,
        weights,
        conf_threshold,
        workers=workers,
    )
    if not segmentations:
        LOGGER.info("AI segmentation unavailable or empty for %s; using CV fallback.", image_path.name)
        segmentations = run_cv_segmentation(image, plot_box)
    if not segmentations:
        raise RuntimeError(f"Unable to isolate curves in {image_path}. Try passing --x-range/--y-range or a segmentation model.")

    point_frames = [extract_curve_points(segmentation, plot_box) for segmentation in segmentations]
    combined = pd.concat(point_frames, ignore_index=True) if point_frames else pd.DataFrame()
    combined = combined.dropna().sort_values(["dataset_id", "x_px"]).reset_index(drop=True)
    converted = convert_points(combined, calibration, plot_box)
    if converted.empty:
        raise RuntimeError(f"No digitized points were extracted from {image_path}.")

    csv_path = output_dir / f"{image_path.stem}.digitized.csv"
    replot_csv_path = output_dir / f"{image_path.stem}.replot.csv"
    metadata_path = output_dir / f"{image_path.stem}.metadata.json"
    replot_path = output_dir / f"{image_path.stem}.replot.png"
    overlay_path = output_dir / f"{image_path.stem}.overlay.png" if create_overlay_image else None

    converted[["dataset_id", "x_real", "y_real", "confidence"]].to_csv(csv_path, index=False)
    replot_frame = build_replot_frame(converted, x_scale=calibration.x_scale)
    replot_frame.to_csv(replot_csv_path, index=False)
    create_replot(replot_frame, calibration, image_path.name, replot_path)
    metadata = {
        "input_image": str(image_path),
        "plot_box": asdict(plot_box),
        "axis": asdict(calibration),
        "exports": {
            "digitized_csv": str(csv_path),
            "replot_csv": str(replot_csv_path),
            "replot_image": str(replot_path),
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
    )


from . import synthetic as _synthetic
from .synthetic import (
    _apply_degradation_filters,
    _render_arrow_mask,
    _render_curve_mask,
    _render_error_bar_mask,
    _render_hbar_mask,
    _render_vbar_mask,
    generate_synthetic_dataset,
    run_training,
)

_SYNTHETIC_PATCH_LOCK = threading.Lock()


def _write_synthetic_example(index: int, output_dir: Path, rng: np.random.Generator, image_format: str, plot_type: str = "general") -> None:
    """Compatibility wrapper for tests that patch private synthetic helpers on this module."""
    with _SYNTHETIC_PATCH_LOCK:
        original_apply = _synthetic._apply_degradation_filters
        original_curve = _synthetic._render_curve_mask
        original_vbar = _synthetic._render_vbar_mask
        original_hbar = _synthetic._render_hbar_mask
        original_arrow = _synthetic._render_arrow_mask
        original_error_bar = _synthetic._render_error_bar_mask
        try:
            _synthetic._apply_degradation_filters = _apply_degradation_filters
            _synthetic._render_curve_mask = _render_curve_mask
            _synthetic._render_vbar_mask = _render_vbar_mask
            _synthetic._render_hbar_mask = _render_hbar_mask
            _synthetic._render_arrow_mask = _render_arrow_mask
            _synthetic._render_error_bar_mask = _render_error_bar_mask
            _synthetic._write_synthetic_example(index, output_dir, rng, image_format, plot_type)
        finally:
            _synthetic._apply_degradation_filters = original_apply
            _synthetic._render_curve_mask = original_curve
            _synthetic._render_vbar_mask = original_vbar
            _synthetic._render_hbar_mask = original_hbar
            _synthetic._render_arrow_mask = original_arrow
            _synthetic._render_error_bar_mask = original_error_bar


def _prepare_curve_points(points: pd.DataFrame) -> pd.DataFrame:
    """Return one curve sorted by x with duplicate x-values removed."""
    return points.drop_duplicates(subset="x_real").sort_values("x_real")


def _interp_curve(points: pd.DataFrame, reference_x: np.ndarray) -> np.ndarray:
    """Linearly interpolate one curve onto a shared x-grid for validation/export."""
    if len(points) < 2:
        raise ValueError("At least two points are required for interpolation.")
    unique = _prepare_curve_points(points)
    interpolator = interp1d(unique["x_real"], unique["y_real"], fill_value="extrapolate")
    return interpolator(reference_x)


def validate_digitization(prediction_csv: Path, truth_csv: Path, output_json: Path | None = None) -> dict[str, Any]:
    """Compare digitized results against ground truth curves."""
    predicted = pd.read_csv(prediction_csv)
    truth = pd.read_csv(truth_csv)
    predicted_groups = list(predicted.groupby("dataset_id"))
    truth_groups = list(truth.groupby("dataset_id"))

    if not predicted_groups or not truth_groups:
        raise ValueError("Validation requires at least one predicted and one truth dataset.")

    truth_ranges = {
        dataset_id: max(1e-6, float(np.ptp(group["y_real"])))
        for dataset_id, group in truth_groups
    }

    truth_ids = [dataset_id for dataset_id, _ in truth_groups]
    predicted_ids = [dataset_id for dataset_id, _ in predicted_groups]
    assignment_matrix_size = max(len(truth_groups), len(predicted_groups))
    cost_matrix = np.full((len(truth_groups), assignment_matrix_size), np.inf, dtype=float)

    for truth_index, (truth_id, truth_frame) in enumerate(truth_groups):
        reference_x = truth_frame["x_real"].to_numpy()
        truth_y = truth_frame["y_real"].to_numpy()
        for predicted_index, (_, predicted_frame) in enumerate(predicted_groups):
            aligned = _interp_curve(predicted_frame, reference_x)
            cost_matrix[truth_index, predicted_index] = float(np.mean(np.abs(aligned - truth_y)))
        # Only dummy prediction columns are needed: every truth curve must be assigned,
        # while extra predicted curves can remain unused in the rectangular cost matrix.
        for dummy_index in range(len(predicted_groups), assignment_matrix_size):
            cost_matrix[truth_index, dummy_index] = truth_ranges[truth_id]

    truth_assignment, predicted_assignment = linear_sum_assignment(cost_matrix)
    metrics: list[dict[str, Any]] = []
    total_error: list[float] = []
    for truth_index, assigned_index in zip(truth_assignment.tolist(), predicted_assignment.tolist(), strict=True):
        truth_id = truth_ids[truth_index]
        predicted_id = predicted_ids[assigned_index] if assigned_index < len(predicted_ids) else None
        mae = float(cost_matrix[truth_index, assigned_index])
        metrics.append(
            {
                "truth_dataset_id": truth_id,
                "predicted_dataset_id": predicted_id,
                "mae": mae,
            }
        )
        total_error.append(mae)

    summary = {
        "mean_absolute_error": float(np.mean(total_error)),
        "mean_absolute_percentage_error_proxy": float(
            np.mean([row["mae"] / truth_ranges[row["truth_dataset_id"]] for row in metrics])
            * 100.0
        ),
        "per_curve": metrics,
        "passed_under_5_percent": bool(
            np.mean([row["mae"] / truth_ranges[row["truth_dataset_id"]] for row in metrics])
            < VALIDATION_THRESHOLD
        ),
    }
    if output_json is not None:
        output_json.write_text(json.dumps(summary, indent=2))
    return summary


def _parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Automatic AI-assisted plot digitizer.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate synthetic plots and YOLO segmentation labels.")
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--count", type=int, default=16)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--image-format", default="png", choices=["png", "jpg"])
    generate_parser.add_argument("--plot-type", default="mixed", choices=["general", "bandstructure", "mixed"],
                                  help="Type of plots: general (standard curves), bandstructure (physics band diagrams), or mixed")
    generate_parser.add_argument(
        "--workers",
        type=_parse_positive_int,
        default=None,
        metavar="N",
        help="Number of worker processes for parallel generation (default: min(os.cpu_count(), count, 8)). Use 1 for sequential.",
    )

    train_parser = subparsers.add_parser("train", help="Train or plan a YOLO segmentation model.")
    train_parser.add_argument("--dataset-dir", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, default=Path("training-runs"))
    train_parser.add_argument("--weights", default="yolov8n-seg.pt")
    train_parser.add_argument("--epochs", type=int, default=25)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--batch", type=int, default=8)
    train_parser.add_argument("--hyp-yaml", type=Path, default=None, help="Optional Ultralytics training override YAML (cfg).")
    train_parser.add_argument("--execute", action="store_true", help="Run training immediately. Otherwise, only print the plan.")
    train_parser.add_argument(
        "--workers",
        type=_parse_positive_int,
        default=None,
        metavar="N",
        help="Number of training workers. Used for Ultralytics DataLoader workers and torch CPU thread pools when training executes. Set to available CPU cores, e.g. --workers 16.",
    )

    digitize_parser = subparsers.add_parser("digitize", help="Digitize one or more plot images.")
    digitize_parser.add_argument("inputs", nargs="+", help="Input image files or directories.")
    digitize_parser.add_argument("--output-dir", type=Path, default=Path("digitized-output"))
    digitize_parser.add_argument("--x-range", type=str, default=None)
    digitize_parser.add_argument("--y-range", type=str, default=None)
    digitize_parser.add_argument("--x-reference", type=str, default=None, help="Known X-axis points in px0:real0,px1:real1 format.")
    digitize_parser.add_argument("--y-reference", type=str, default=None, help="Known Y-axis points in px0:real0,px1:real1 format.")
    digitize_parser.add_argument(
        "--interactive-axis-selection",
        action="store_true",
        help="Interactively click two X-axis points and two Y-axis points, then enter their real values.",
    )
    digitize_parser.add_argument("--x-scale", choices=["linear", "log"], default="linear")
    digitize_parser.add_argument("--y-scale", choices=["linear", "log"], default="linear")
    digitize_parser.add_argument("--invert-y", action="store_true")
    digitize_parser.add_argument(
        "--disable-auto-axis-anchors",
        action="store_true",
        help="Disable automatic axis-anchor point detection for calibration fallback.",
    )
    digitize_parser.add_argument("--weights", default=None, help="YOLO .pt or .onnx segmentation weights.")
    digitize_parser.add_argument("--conf-threshold", type=float, default=0.25)
    digitize_parser.add_argument(
        "--workers",
        type=_parse_positive_int,
        default=None,
        metavar="N",
        help="Number of DataLoader worker processes for AI digitizing inference (default: Ultralytics default). Set to CPU core count, e.g. --workers 16 for a 16-core system.",
    )
    digitize_parser.add_argument("--overlay", action="store_true", help="Write overlay images.")

    validate_parser = subparsers.add_parser("validate", help="Validate a digitized CSV against ground truth.")
    validate_parser.add_argument("--prediction-csv", type=Path, required=True)
    validate_parser.add_argument("--truth-csv", type=Path, required=True)
    validate_parser.add_argument("--output-json", type=Path, default=None)

    return parser


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    if args.command == "generate":
        generate_synthetic_dataset(
            args.output_dir, args.count, args.seed, args.image_format, args.plot_type,
            workers=args.workers,
        )
        LOGGER.info("Generated %s synthetic plots (%s) in %s", args.count, args.plot_type, args.output_dir)
        return 0

    if args.command == "train":
        plan = run_training(
            args.dataset_dir,
            args.output_dir,
            args.epochs,
            args.imgsz,
            args.weights,
            args.batch,
            args.execute,
            args.hyp_yaml,
            workers=args.workers,
        )
        print(json.dumps(plan, indent=2, default=_json_default))
        return 0

    if args.command == "digitize":
        images = discover_images(args.inputs)
        if not images:
            parser.error("No input images were found.")
        
        # Set matplotlib backend based on interactive mode
        if args.interactive_axis_selection:
            _set_matplotlib_backend("TkAgg")  # Interactive backend for GUI
        else:
            _set_matplotlib_backend("Agg")  # Non-interactive backend
        
        x_range = parse_range(args.x_range)
        y_range = parse_range(args.y_range)
        x_reference = parse_reference_pair(args.x_reference, "x")
        y_reference = parse_reference_pair(args.y_reference, "y")
        if args.interactive_axis_selection and (x_reference is not None or y_reference is not None):
            parser.error("Cannot combine --interactive-axis-selection with --x-reference or --y-reference.")
        if args.interactive_axis_selection and images:
            x_reference, y_reference = interactive_reference_selection(images[0])
            LOGGER.info(
                "Interactive axis selection complete. Reuse with: "
                "--x-reference \"%s\" --y-reference \"%s\"",
                _format_reference_pair_cli_value(x_reference),
                _format_reference_pair_cli_value(y_reference),
            )
        results = []
        for image_path in images:
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
                workers=args.workers,
                auto_axis_anchors=not args.disable_auto_axis_anchors,
            )
            results.append(
                {
                    "image": str(image_path),
                    "csv_path": str(result.csv_path),
                    "replot_csv_path": str(result.replot_csv_path),
                    "metadata_path": str(result.metadata_path),
                    "replot_path": str(result.replot_path),
                    "overlay_path": str(result.overlay_path) if result.overlay_path else None,
                    "point_count": result.point_count,
                    "dataset_count": result.dataset_count,
                }
            )
        print(json.dumps(results, indent=2))
        return 0

    if args.command == "validate":
        summary = validate_digitization(args.prediction_csv, args.truth_csv, args.output_json)
        print(json.dumps(summary, indent=2))
        return 0 if summary["passed_under_5_percent"] else 1

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
