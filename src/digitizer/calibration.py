"""Axis calibration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .axis_parsing import _resolve_bounds_from_references
from .constants import DARK_PIXEL_PERCENTILE, MAX_DARK_THRESHOLD
from .image_ops import _parse_sidecar_metadata
from .models import AxisCalibration, AxisReferencePair, PlotBox

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
    sidecar_x_range = sidecar.get("x_range")
    sidecar_y_range = sidecar.get("y_range")
    x_bounds = x_range or (tuple(sidecar_x_range) if sidecar_x_range is not None else None)
    y_bounds = y_range or (tuple(sidecar_y_range) if sidecar_y_range is not None else None)
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
    if x_bounds is None:
        raise RuntimeError(
            "Unable to calibrate X-axis bounds. Provide --x-range/--x-reference, "
            "sidecar metadata, or --interactive-axis-selection."
        )
    if y_bounds is None:
        raise RuntimeError(
            "Unable to calibrate Y-axis bounds. Provide --y-range/--y-reference, "
            "sidecar metadata, or --interactive-axis-selection."
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
                else ("reference" if x_reference else ("cli" if x_range else "sidecar"))
            ),
            "y_range_source": (
                "auto-anchor"
                if used_auto_y
                else ("reference" if y_reference else ("cli" if y_range else "sidecar"))
            ),
            "x_scale": calibration.x_scale,
            "y_scale": calibration.y_scale,
            "invert_y": calibration.invert_y,
        },
        "warnings": [],
    }
    return calibration, metadata
