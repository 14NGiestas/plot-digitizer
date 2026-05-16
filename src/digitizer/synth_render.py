"""Synthetic mask rendering helpers."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from skimage import measure, morphology

from .constants import CURVE_MASK_PADDING_PIXELS, MAX_POLYGON_POINTS

def _render_vbar_mask(fig_size: tuple[float, float], dpi: int, x_pos: float, y_range: tuple[float, float], 
                      width: float, style: dict[str, Any]) -> np.ndarray:
    """Render a vertical bar mask."""
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi, facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(0, 1)
    ax.set_ylim(*y_range)
    ax.axvline(x=x_pos, ymin=0, ymax=1, color="white", linewidth=width, linestyle=style.get("linestyle", "-"))
    ax.axis("off")
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return np.max(buffer[:, :, :3], axis=2) > 200


def _render_hbar_mask(
    fig_size: tuple[float, float],
    dpi: int,
    y_pos: float,
    x_range: tuple[float, float],
    height: float,
    style: dict[str, Any],
    x_scale: str = "linear",
) -> np.ndarray:
    """Render a horizontal bar mask."""
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi, facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(*x_range)
    ax.set_xscale(x_scale)
    ax.set_ylim(0, 1)
    ax.axhline(y=y_pos, xmin=0, xmax=1, color="white", linewidth=height, linestyle=style.get("linestyle", "-"))
    ax.axis("off")
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return np.max(buffer[:, :, :3], axis=2) > 200


def _render_arrow_mask(fig_size: tuple[float, float], dpi: int, start: tuple[float, float], 
                       end: tuple[float, float], style: dict[str, Any]) -> np.ndarray:
    """Render an arrow annotation mask."""
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi, facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="white", 
                                                           lw=style.get("linewidth", 2.0)))
    ax.axis("off")
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return np.max(buffer[:, :, :3], axis=2) > 200


def _render_error_bar_mask(fig_size: tuple[float, float], dpi: int, x_pos: float, y_pos: float,
                           y_err: float, style: dict[str, Any]) -> np.ndarray:
    """Render an error bar mask."""
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi, facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    cap_width = style.get("cap_width", 0.03)
    ax.errorbar(x_pos, y_pos, yerr=y_err, fmt='none', ecolor="white", 
                elinewidth=style.get("linewidth", 1.5), capsize=cap_width * fig_size[0] * dpi)
    ax.axis("off")
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return np.max(buffer[:, :, :3], axis=2) > 200


def _render_curve_mask(
    fig_size: tuple[float, float],
    dpi: int,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    style: dict[str, Any],
    x_scale: str = "linear",
    curve_mask_padding_pixels: int = CURVE_MASK_PADDING_PIXELS,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi, facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(*x_range)
    ax.set_xscale(x_scale)
    ax.set_ylim(*y_range)
    ax.plot(x_values, y_values, color="white", linewidth=style["linewidth"], linestyle=style["linestyle"])
    ax.axis("off")
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    mask = np.max(buffer[:, :, :3], axis=2) > 200
    if curve_mask_padding_pixels > 0:
        mask = morphology.dilation(mask, morphology.disk(curve_mask_padding_pixels))
    return mask


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
