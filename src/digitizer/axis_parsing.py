"""Axis range and reference parsing helpers."""

from __future__ import annotations

import argparse

import numpy as np

from .models import AxisReferencePair, PlotBox

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

