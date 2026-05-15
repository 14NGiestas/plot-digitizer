"""Synthetic annotation-layer helpers."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .constants import (
    ARROW_COUNT_RANGE,
    ARROW_LABEL_PROBABILITY,
    ARROW_LABEL_TEXTS,
    ERROR_BAR_COUNT_RANGE,
    HBAR_COUNT_RANGE,
    VBAR_COUNT_RANGE,
)
from .synth_render import _mask_to_yolo_polygon


def _add_annotation_layers(
    ax: Any,
    rng: np.random.Generator,
    fig_size: tuple[float, float],
    dpi: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    use_log_x: bool,
    x_norm_to_data: Callable[[float], float],
    y_norm_to_data: Callable[[float], float],
    render_vbar_mask_fn: Any,
    render_hbar_mask_fn: Any,
    render_arrow_mask_fn: Any,
    render_error_bar_mask_fn: Any,
    vbar_count_range: tuple[int, int] | None = None,
    hbar_count_range: tuple[int, int] | None = None,
    arrow_count_range: tuple[int, int] | None = None,
    error_bar_count_range: tuple[int, int] | None = None,
    add_arrow_labels: bool = True,
) -> tuple[list[str], list[dict[str, Any]]]:
    label_lines: list[str] = []
    annotation_descriptors: list[dict[str, Any]] = []

    eff_vbar = vbar_count_range if vbar_count_range is not None else VBAR_COUNT_RANGE
    eff_hbar = hbar_count_range if hbar_count_range is not None else HBAR_COUNT_RANGE
    eff_arrow = arrow_count_range if arrow_count_range is not None else ARROW_COUNT_RANGE
    eff_error = error_bar_count_range if error_bar_count_range is not None else ERROR_BAR_COUNT_RANGE

    for vbar_idx in range(int(rng.integers(eff_vbar[0], eff_vbar[1] + 1))):
        x_pos = rng.uniform(0.1, 0.9)
        style = {"linewidth": rng.uniform(1.0, 3.0), "linestyle": "-"}
        ax.axvline(x=x_norm_to_data(x_pos), ymin=0, ymax=1, color="black", linewidth=style["linewidth"], linestyle=style["linestyle"])
        polygon = _mask_to_yolo_polygon(render_vbar_mask_fn(fig_size, dpi, x_pos, y_range, style["linewidth"], style))
        if polygon:
            label_lines.append("1 " + " ".join(f"{value:.6f}" for value in polygon))
            annotation_descriptors.append({"type": "vbar", "class_id": 1, "x_pos": x_pos, "description": f"high_symmetry_point_{vbar_idx}"})

    for hbar_idx in range(int(rng.integers(eff_hbar[0], eff_hbar[1] + 1))):
        y_pos_norm = rng.uniform(0.1, 0.9)
        y_pos = y_range[0] + y_pos_norm * (y_range[1] - y_range[0])
        style = {"linewidth": rng.uniform(1.0, 2.5), "linestyle": "--"}
        ax.axhline(y=y_pos, xmin=0, xmax=1, color="black", linewidth=style["linewidth"], linestyle=style["linestyle"])
        polygon = _mask_to_yolo_polygon(render_hbar_mask_fn(fig_size, dpi, y_pos_norm, x_range, style["linewidth"], style, x_scale="log" if use_log_x else "linear"))
        if polygon:
            label_lines.append("2 " + " ".join(f"{value:.6f}" for value in polygon))
            annotation_descriptors.append({"type": "hbar", "class_id": 2, "y_pos": y_pos, "description": f"reference_line_{hbar_idx}"})

    for arrow_idx in range(int(rng.integers(eff_arrow[0], eff_arrow[1] + 1))):
        start = (rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8))
        end = (rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8))
        style = {"linewidth": rng.uniform(1.5, 3.0)}
        ax.annotate("", xy=(x_norm_to_data(end[0]), y_norm_to_data(end[1])), xytext=(x_norm_to_data(start[0]), y_norm_to_data(start[1])), arrowprops={"arrowstyle": "->", "color": "black", "lw": style["linewidth"]})
        if add_arrow_labels and rng.random() < ARROW_LABEL_PROBABILITY:
            label_text = str(rng.choice(ARROW_LABEL_TEXTS))
            mid_x = x_norm_to_data((start[0] + end[0]) / 2.0)
            mid_y = y_norm_to_data((start[1] + end[1]) / 2.0)
            ax.text(mid_x, mid_y, label_text, fontsize=float(rng.uniform(7.0, 11.0)), color="black", ha="left", va="bottom")
        polygon = _mask_to_yolo_polygon(render_arrow_mask_fn(fig_size, dpi, start, end, style))
        if polygon:
            label_lines.append("3 " + " ".join(f"{value:.6f}" for value in polygon))
            annotation_descriptors.append({"type": "arrow", "class_id": 3, "start": start, "end": end, "description": f"annotation_arrow_{arrow_idx}"})

    for eb_idx in range(int(rng.integers(eff_error[0], eff_error[1] + 1))):
        x_pos = rng.uniform(0.1, 0.9)
        y_pos = rng.uniform(0.2, 0.8)
        y_err = rng.uniform(0.05, 0.2)
        style = {"linewidth": rng.uniform(1.0, 2.0), "cap_width": 0.02}
        ax.errorbar(x_norm_to_data(x_pos), y_norm_to_data(y_pos), yerr=float(y_err * (y_range[1] - y_range[0])), fmt="none", ecolor="black", elinewidth=style["linewidth"], capsize=style["cap_width"] * fig_size[0] * dpi)
        polygon = _mask_to_yolo_polygon(render_error_bar_mask_fn(fig_size, dpi, x_pos, y_pos, y_err, style))
        if polygon:
            label_lines.append("4 " + " ".join(f"{value:.6f}" for value in polygon))
            annotation_descriptors.append({"type": "error_bar", "class_id": 4, "x_pos": x_pos, "y_pos": y_pos, "y_err": y_err, "description": f"error_bar_{eb_idx}"})

    return label_lines, annotation_descriptors
