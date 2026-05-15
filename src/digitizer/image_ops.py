"""Image discovery, loading, preprocessing, and plot-box resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .constants import (
    DARK_PIXEL_PERCENTILE,
    IMAGE_EXTENSIONS,
    LOGGER,
    MAX_DARK_THRESHOLD,
    PLOT_MARGIN_FRACTION,
)
from .models import PlotBox

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

