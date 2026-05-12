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
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from skimage import measure, morphology
from sklearn.cluster import DBSCAN, KMeans

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
MIN_CURVES_PER_PLOT = 1
MAX_CURVES_PER_PLOT = 3
VALIDATION_THRESHOLD = 0.05
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

    def pixel_to_real(self, x_px: np.ndarray, y_px: np.ndarray, plot_box: PlotBox) -> tuple[np.ndarray, np.ndarray]:
        """Convert pixel arrays to real-world coordinates."""
        x_norm = np.clip((x_px - plot_box.left) / plot_box.width, 0.0, 1.0)
        y_norm = np.clip((plot_box.bottom - y_px) / plot_box.height, 0.0, 1.0)
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
    metadata_path: Path
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
    return morphology.footprint_rectangle((height, width))


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


def calibrate_axes(
    image_path: Path,
    plot_box: PlotBox,
    x_range: tuple[float, float] | None,
    y_range: tuple[float, float] | None,
    x_scale: str,
    y_scale: str,
    invert_y: bool,
) -> tuple[AxisCalibration, dict[str, Any]]:
    """Resolve axis ranges from CLI hints, sidecars, OCR, or defaults."""
    sidecar = _parse_sidecar_metadata(image_path) or {}
    x_bounds = x_range or tuple(sidecar.get("x_range", DEFAULT_X_RANGE))
    y_bounds = y_range or tuple(sidecar.get("y_range", DEFAULT_Y_RANGE))
    calibration = AxisCalibration(
        x_min=float(x_bounds[0]),
        x_max=float(x_bounds[1]),
        y_min=float(y_bounds[0]),
        y_max=float(y_bounds[1]),
        x_scale=str(sidecar.get("x_scale", x_scale)),
        y_scale=str(sidecar.get("y_scale", y_scale)),
        invert_y=bool(sidecar.get("invert_y", invert_y)),
    )
    metadata = {
        "origin_pixel": {"x": plot_box.left, "y": plot_box.bottom},
        "axis_detection": {
            "x_range_source": "cli" if x_range else ("sidecar" if "x_range" in sidecar else "default"),
            "y_range_source": "cli" if y_range else ("sidecar" if "y_range" in sidecar else "default"),
            "x_scale": calibration.x_scale,
            "y_scale": calibration.y_scale,
            "invert_y": calibration.invert_y,
        },
        "warnings": [],
    }
    if not x_range and "x_range" not in sidecar:
        metadata["warnings"].append("X-axis bounds were not auto-detected; defaulting to 0:1. Pass --x-range for better accuracy.")
    if not y_range and "y_range" not in sidecar:
        metadata["warnings"].append("Y-axis bounds were not auto-detected; defaulting to 0:1. Pass --y-range for better accuracy.")
    return calibration, metadata


def run_ai_segmentation(image: np.ndarray, plot_box: PlotBox, weights: str | None, conf_threshold: float) -> list[SegmentationResult]:
    """Run YOLO segmentation if weights are available."""
    if not weights:
        return []
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - fallback path when ultralytics is unavailable
        LOGGER.warning("Ultralytics import failed, falling back to CV segmentation: %s", exc)
        return []

    model = YOLO(weights)
    predictions = model.predict(image, conf=conf_threshold, verbose=False)
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
    sample_count = min(len(filtered), 500)
    sample_indices = np.linspace(0, len(filtered) - 1, sample_count).astype(int)
    sample = filtered[sample_indices]
    cluster_count = int(min(4, max(1, len(np.unique(sample, axis=0)))))
    if cluster_count <= 1:
        return []
    model = KMeans(n_clusters=cluster_count, n_init=5, random_state=42)
    labels = model.fit_predict(filtered)
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


def digitize_image(
    image_path: Path,
    output_dir: Path,
    x_range: tuple[float, float] | None,
    y_range: tuple[float, float] | None,
    x_scale: str,
    y_scale: str,
    invert_y: bool,
    weights: str | None,
    conf_threshold: float,
    create_overlay_image: bool,
) -> DigitizeResult:
    """Digitize a single image and write CSV/JSON artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image = load_image(image_path)
    processed_gray, preprocess_stats = preprocess_image(image)
    plot_box = resolve_plot_box(image_path, image, processed_gray)
    calibration, axis_metadata = calibrate_axes(image_path, plot_box, x_range, y_range, x_scale, y_scale, invert_y)

    segmentations = run_ai_segmentation(image, plot_box, weights, conf_threshold)
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
    metadata_path = output_dir / f"{image_path.stem}.metadata.json"
    overlay_path = output_dir / f"{image_path.stem}.overlay.png" if create_overlay_image else None

    converted[["dataset_id", "x_real", "y_real", "confidence"]].to_csv(csv_path, index=False)
    metadata = {
        "input_image": str(image_path),
        "plot_box": asdict(plot_box),
        "axis": asdict(calibration),
        "preprocessing": preprocess_stats,
        "segmentation": {
            "dataset_count": int(converted["dataset_id"].nunique()),
            "points": int(len(converted)),
            "method_counts": pd.Series([segmentation.method for segmentation in segmentations]).value_counts().to_dict(),
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
        metadata_path=metadata_path,
        overlay_path=overlay_path,
        point_count=int(len(converted)),
        dataset_count=int(converted["dataset_id"].nunique()),
    )


def _random_curve(x_values: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    curve_type = rng.choice(["sin", "poly", "exp", "damped"])
    if curve_type == "sin":
        amplitude = rng.uniform(*SINE_AMPLITUDE_RANGE)
        frequency = rng.uniform(*SINE_FREQUENCY_RANGE)
        phase = rng.uniform(0.0, math.pi)
        offset = rng.uniform(*SINE_OFFSET_RANGE)
        y_values = offset + amplitude * np.sin(frequency * x_values + phase)
    elif curve_type == "exp":
        scale = rng.uniform(*EXP_SCALE_RANGE)
        growth = rng.uniform(*EXP_GROWTH_RANGE)
        offset = rng.uniform(*EXP_OFFSET_RANGE)
        y_values = offset + scale * np.exp(growth * (x_values - x_values.min()))
    elif curve_type == "damped":
        amplitude = rng.uniform(*DAMPED_AMPLITUDE_RANGE)
        decay = rng.uniform(*DAMPED_DECAY_RANGE)
        frequency = rng.uniform(*DAMPED_FREQUENCY_RANGE)
        y_values = amplitude * np.exp(-decay * x_values) * np.cos(frequency * x_values)
    else:
        a, b, c = rng.uniform(*POLY_A_RANGE), rng.uniform(*POLY_B_RANGE), rng.uniform(*POLY_C_RANGE)
        y_values = a * (x_values * x_values) + b * x_values + c
    noise = rng.normal(0.0, rng.uniform(*NOISE_STD_RANGE), size=x_values.shape)
    return y_values + noise, str(curve_type)


def _render_curve_mask(fig_size: tuple[float, float], dpi: int, x_values: np.ndarray, y_values: np.ndarray, x_range: tuple[float, float], y_range: tuple[float, float], style: dict[str, Any]) -> np.ndarray:
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.plot(x_values, y_values, color="white", linewidth=style["linewidth"], linestyle=style["linestyle"])
    ax.axis("off")
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return buffer[:, :, 0] > 150


def _mask_to_yolo_polygon(mask: np.ndarray) -> list[float]:
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return []
    contour = max(contours, key=len)
    polygon: list[float] = []
    height, width = mask.shape
    step = max(1, len(contour) // MAX_POLYGON_POINTS)
    for y_coord, x_coord in contour[::step]:
        polygon.extend([float(np.clip(x_coord / width, 0.0, 1.0)), float(np.clip(y_coord / height, 0.0, 1.0))])
    return polygon if len(polygon) >= 6 else []


def _write_synthetic_example(index: int, output_dir: Path, rng: np.random.Generator, image_format: str) -> None:
    fig_size = (6.0, 4.2)
    dpi = DEFAULT_DPI
    image_name = f"plot_{index:04d}.{image_format}"
    image_path = output_dir / "images" / image_name
    label_path = output_dir / "labels" / f"plot_{index:04d}.txt"
    metadata_path = output_dir / "images" / f"plot_{index:04d}.metadata.json"
    ground_truth_path = output_dir / "ground_truth" / f"plot_{index:04d}.csv"

    x_range = (0.0, float(rng.uniform(6.0, 12.0)))
    x_values = np.linspace(*x_range, 480)
    curve_count = int(rng.integers(MIN_CURVES_PER_PLOT, MAX_CURVES_PER_PLOT + 1))
    raw_curves = [_random_curve(x_values, rng) for _ in range(curve_count)]
    all_y = np.concatenate([curve for curve, _ in raw_curves])
    y_margin = max(0.5, float(np.ptp(all_y) * 0.1))
    y_range = (float(all_y.min() - y_margin), float(all_y.max() + y_margin))

    colors = ["tab:red", "tab:blue", "tab:green", "tab:purple"]
    linestyles = ["-", "--", "-.", ":"]
    linewidths = [2.0, 2.4, 2.2, 2.6]

    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    if rng.random() > 0.3:
        ax.grid(True, linestyle="--" if rng.random() > 0.5 else ":", alpha=0.4)
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Synthetic Plot")

    ground_truth_frames: list[pd.DataFrame] = []
    label_lines: list[str] = []
    curve_descriptors: list[dict[str, Any]] = []

    for curve_index, (y_values, curve_type) in enumerate(raw_curves):
        style = {
            "color": colors[curve_index % len(colors)],
            "linestyle": linestyles[curve_index % len(linestyles)],
            "linewidth": linewidths[curve_index % len(linewidths)],
        }
        ax.plot(x_values, y_values, **style)
        dataset_id = f"dataset_{curve_index}"
        curve_descriptors.append({"dataset_id": dataset_id, "curve_type": curve_type, **style})
        ground_truth_frames.append(pd.DataFrame({"dataset_id": dataset_id, "x_real": x_values, "y_real": y_values}))
        mask = _render_curve_mask(fig_size, dpi, x_values, y_values, x_range, y_range, style)
        polygon = _mask_to_yolo_polygon(mask)
        if polygon:
            label_lines.append("0 " + " ".join(f"{value:.6f}" for value in polygon))

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
    fig.savefig(image_path, dpi=dpi)
    plt.close(fig)

    ground_truth = pd.concat(ground_truth_frames, ignore_index=True)
    ground_truth.to_csv(ground_truth_path, index=False)
    label_path.write_text("\n".join(label_lines))
    metadata = {
        "image": str(image_path),
        "x_range": list(x_range),
        "y_range": list(y_range),
        "x_scale": "linear",
        "y_scale": "linear",
        "invert_y": False,
        "plot_box": plot_box,
        "curves": curve_descriptors,
        "ground_truth_csv": str(ground_truth_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))


def generate_synthetic_dataset(output_dir: Path, count: int, seed: int, image_format: str) -> None:
    """Generate a synthetic plot dataset with YOLO segmentation labels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("images", "labels", "ground_truth"):
        (output_dir / subdir).mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    for index in range(count):
        _write_synthetic_example(index, output_dir, rng, image_format)
    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir}",
                "train: images",
                "val: images",
                "test: images",
                "names:",
                "  0: curve",
            ]
        )
    )


def run_training(dataset_dir: Path, output_dir: Path, epochs: int, imgsz: int, weights: str, batch: int, execute: bool) -> dict[str, Any]:
    """Create or execute a YOLO segmentation training job."""
    dataset_yaml = dataset_dir / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_yaml}")
    training_plan = {
        "dataset": str(dataset_yaml),
        "weights": weights,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": str(output_dir),
        "task": "segment",
    }
    if execute:
        from ultralytics import YOLO

        model = YOLO(weights)
        training_plan["result"] = model.train(
            data=str(dataset_yaml),
            task="segment",
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project=str(output_dir),
            name="synthetic_plot_digitizer",
        ).save_dir.as_posix()
    return training_plan


def _interp_curve(points: pd.DataFrame, reference_x: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        raise ValueError("At least two points are required for interpolation.")
    unique = points.drop_duplicates(subset="x_real").sort_values("x_real")
    interpolator = interp1d(unique["x_real"], unique["y_real"], fill_value="extrapolate")
    return interpolator(reference_x)


def validate_digitization(prediction_csv: Path, truth_csv: Path, output_json: Path | None = None) -> dict[str, Any]:
    """Compare digitized results against ground truth curves."""
    predicted = pd.read_csv(prediction_csv)
    truth = pd.read_csv(truth_csv)
    metrics: list[dict[str, Any]] = []
    total_error = []
    predicted_groups = list(predicted.groupby("dataset_id"))
    truth_groups = list(truth.groupby("dataset_id"))

    if not predicted_groups or not truth_groups:
        raise ValueError("Validation requires at least one predicted and one truth dataset.")

    truth_ranges = {
        dataset_id: max(1e-6, float(np.ptp(group["y_real"])))
        for dataset_id, group in truth_groups
    }

    for truth_id, truth_frame in truth_groups:
        reference_x = truth_frame["x_real"].to_numpy()
        truth_y = truth_frame["y_real"].to_numpy()
        best: dict[str, Any] | None = None
        for predicted_id, predicted_frame in predicted_groups:
            aligned = _interp_curve(predicted_frame, reference_x)
            mae = float(np.mean(np.abs(aligned - truth_y)))
            score = {
                "truth_dataset_id": truth_id,
                "predicted_dataset_id": predicted_id,
                "mae": mae,
            }
            if best is None or score["mae"] < best["mae"]:
                best = score
        if best is None:
            raise ValueError(f"Unable to align predicted curves to truth dataset {truth_id!r}.")
        metrics.append(best)
        total_error.append(best["mae"])

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

    train_parser = subparsers.add_parser("train", help="Train or plan a YOLO segmentation model.")
    train_parser.add_argument("--dataset-dir", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, default=Path("training-runs"))
    train_parser.add_argument("--weights", default="yolov8n-seg.pt")
    train_parser.add_argument("--epochs", type=int, default=25)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--batch", type=int, default=8)
    train_parser.add_argument("--execute", action="store_true", help="Run training immediately. Otherwise, only print the plan.")

    digitize_parser = subparsers.add_parser("digitize", help="Digitize one or more plot images.")
    digitize_parser.add_argument("inputs", nargs="+", help="Input image files or directories.")
    digitize_parser.add_argument("--output-dir", type=Path, default=Path("digitized-output"))
    digitize_parser.add_argument("--x-range", type=str, default=None)
    digitize_parser.add_argument("--y-range", type=str, default=None)
    digitize_parser.add_argument("--x-scale", choices=["linear", "log"], default="linear")
    digitize_parser.add_argument("--y-scale", choices=["linear", "log"], default="linear")
    digitize_parser.add_argument("--invert-y", action="store_true")
    digitize_parser.add_argument("--weights", default=None, help="YOLO .pt or .onnx segmentation weights.")
    digitize_parser.add_argument("--conf-threshold", type=float, default=0.25)
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
        generate_synthetic_dataset(args.output_dir, args.count, args.seed, args.image_format)
        LOGGER.info("Generated %s synthetic plots in %s", args.count, args.output_dir)
        return 0

    if args.command == "train":
        plan = run_training(args.dataset_dir, args.output_dir, args.epochs, args.imgsz, args.weights, args.batch, args.execute)
        print(json.dumps(plan, indent=2, default=_json_default))
        return 0

    if args.command == "digitize":
        images = discover_images(args.inputs)
        if not images:
            parser.error("No input images were found.")
        x_range = parse_range(args.x_range)
        y_range = parse_range(args.y_range)
        results = []
        for image_path in images:
            result = digitize_image(
                image_path=image_path,
                output_dir=args.output_dir,
                x_range=x_range,
                y_range=y_range,
                x_scale=args.x_scale,
                y_scale=args.y_scale,
                invert_y=args.invert_y,
                weights=args.weights,
                conf_threshold=args.conf_threshold,
                create_overlay_image=args.overlay,
            )
            results.append(
                {
                    "image": str(image_path),
                    "csv_path": str(result.csv_path),
                    "metadata_path": str(result.metadata_path),
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
