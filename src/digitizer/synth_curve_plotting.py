"""Synthetic plot setup and curve-layer helpers."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import (
    BASE_CURVE_COUNT_RANGE,
    CURVE_LINEWIDTH_PROBABILITIES,
    CURVE_LINEWIDTHS,
    DENSE_CURVE_COUNT_RANGE,
    DENSE_CURVE_PROBABILITY,
    GRID_ALPHA,
    GRID_ENABLED_PROBABILITY,
)
from .synth_curves import _generate_bandstructure_curves, _random_curve
from .synth_render import _mask_to_yolo_polygon


def _style_axis_text(ax: Any, rng: np.random.Generator, title: str, xlabel: str, ylabel: str) -> None:
    """Apply varied typography so synthetic plots cover more visual styles."""
    font_family = rng.choice(["DejaVu Sans", "DejaVu Serif", "STIXGeneral", "monospace"])
    label_font_family = rng.choice([font_family, "sans-serif", "serif"])
    title_size = float(rng.uniform(11.0, 16.5))
    label_size = float(rng.uniform(9.0, 13.5))
    tick_size = float(rng.uniform(7.0, 11.0))
    title_weight = rng.choice(["normal", "medium", "semibold", "bold"])
    label_weight = rng.choice(["normal", "normal", "medium", "semibold"])
    title_style = rng.choice(["normal", "normal", "italic"])
    x_rotation = int(rng.choice([0, 0, 0, 15, -15, 30, 45]))
    y_rotation = int(rng.choice([0, 0, 0, 10, -10]))

    ax.set_title(title, fontfamily=font_family, fontsize=title_size, fontweight=title_weight, fontstyle=title_style)
    ax.set_xlabel(xlabel, fontfamily=label_font_family, fontsize=label_size, fontweight=label_weight)
    ax.set_ylabel(ylabel, fontfamily=label_font_family, fontsize=label_size, fontweight=label_weight)
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=tick_size,
        direction=rng.choice(["in", "out", "inout"]),
        length=float(rng.uniform(3.0, 6.5)),
        width=float(rng.uniform(0.8, 1.6)),
    )
    for tick_label in ax.get_xticklabels():
        tick_label.set_fontfamily(font_family)
        tick_label.set_rotation(x_rotation)
    for tick_label in ax.get_yticklabels():
        tick_label.set_fontfamily(font_family)
        tick_label.set_rotation(y_rotation)


def _create_plot_axes(fig_size: tuple[float, float], dpi: int, x_range: tuple[float, float], use_log_x: bool, rng: np.random.Generator):
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    if use_log_x:
        ax.set_xscale("log")
    if rng.random() < GRID_ENABLED_PROBABILITY:
        ax.grid(True, linestyle="--" if rng.random() > 0.5 else ":", alpha=GRID_ALPHA)
    else:
        ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linestyle(str(rng.choice(["-", "--", ":"])))
        spine_width = float(rng.uniform(0.8, 1.8))
        spine.set_linewidth(spine_width)
    ax.set_xlim(*x_range)
    return fig, ax


def _configure_plot_curves(
    plot_type: str,
    ax: Any,
    x_values: np.ndarray,
    rng: np.random.Generator,
    curve_count_range: tuple[int, int] | None = None,
) -> tuple[list[tuple[np.ndarray, str]], tuple[float, float], list[dict[str, Any]]]:
    annotation_descriptors: list[dict[str, Any]] = []
    if plot_type == "bandstructure":
        raw_curves = _generate_bandstructure_curves(x_values, rng, int(rng.integers(4, 10)))
        y_range = (-2.5, 2.5)
        ax.set_ylim(*y_range)
        _style_axis_text(ax, rng, "Band Structure", "k-path", "Energy (eV)")
        if rng.random() > 0.5:
            fermi_y = rng.uniform(-0.5, 0.5)
            ax.axhline(y=fermi_y, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
            annotation_descriptors.append({"type": "hbar", "class_id": 2, "y_pos": fermi_y, "description": "fermi_level"})
        return raw_curves, y_range, annotation_descriptors

    if curve_count_range is not None:
        count = int(rng.integers(curve_count_range[0], curve_count_range[1] + 1))
    elif rng.random() < DENSE_CURVE_PROBABILITY:
        count = int(rng.integers(DENSE_CURVE_COUNT_RANGE[0], DENSE_CURVE_COUNT_RANGE[1] + 1))
    else:
        count = int(rng.integers(BASE_CURVE_COUNT_RANGE[0], BASE_CURVE_COUNT_RANGE[1] + 1))
    raw_curves = [_random_curve(x_values, rng) for _ in range(count)]
    all_y = np.concatenate([curve for curve, _ in raw_curves])
    y_margin = max(0.5, float(np.ptp(all_y) * 0.1))
    y_range = (float(all_y.min() - y_margin), float(all_y.max() + y_margin))
    ax.set_ylim(*y_range)
    _style_axis_text(ax, rng, "Synthetic Plot", "X", "Y")
    return raw_curves, y_range, annotation_descriptors


def _add_curve_layers(
    ax: Any,
    x_values: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    use_log_x: bool,
    raw_curves: list[tuple[np.ndarray, str]],
    rng: np.random.Generator,
    show_legend: bool = False,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], list[str]]:
    colors = ["tab:red", "tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:cyan"]
    linestyles = ["-", "--", "-.", ":"]
    ground_truth_frames: list[pd.DataFrame] = []
    curve_descriptors: list[dict[str, Any]] = []
    label_lines: list[str] = []
    for curve_index, (y_values, curve_type) in enumerate(raw_curves):
        style = {
            "color": colors[curve_index % len(colors)],
            "linestyle": linestyles[curve_index % len(linestyles)],
            "linewidth": float(rng.choice(CURVE_LINEWIDTHS, p=CURVE_LINEWIDTH_PROBABILITIES)),
        }
        ax.plot(x_values, y_values, label=f"Series {curve_index + 1}", **style)
        dataset_id = f"dataset_{curve_index}"
        curve_descriptors.append({"dataset_id": dataset_id, "curve_type": curve_type, **style})
        ground_truth_frames.append(pd.DataFrame({"dataset_id": dataset_id, "x_real": x_values, "y_real": y_values}))
    if show_legend:
        ax.legend()
    return ground_truth_frames, curve_descriptors, label_lines
