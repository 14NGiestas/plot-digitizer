"""Interactive axis point selection helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .constants import (
    INTERACTIVE_CLICK_RADIUS_SCALE,
    INTERACTIVE_SELECTION_HELP_TEXT,
    INTERACTIVE_SELECTION_LIMIT_REACHED_TEXT,
    INTERACTIVE_SELECTION_REMOVED_TEXT,
    INTERACTIVE_ZOOM_HALF_SIZE_MIN,
    INTERACTIVE_ZOOM_HALF_SIZE_SCALE,
)
from .image_ops import load_image
from .models import AxisReferencePair

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

