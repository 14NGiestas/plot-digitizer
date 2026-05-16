"""Synthetic single-example rendering and writing helpers."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .constants import DEFAULT_DPI, DENSE_CURVE_COUNT_RANGE, LEGEND_PROBABILITY, LOG_X_MIN, LOG_X_PROBABILITY
from .math_utils import _norm_to_scale
from .synth_annotations import _add_annotation_layers
from .synth_curve_plotting import _add_curve_layers, _configure_plot_curves, _create_plot_axes
from .synth_degrade import _apply_degradation_filters
from .synth_output import _save_synthetic_outputs
from .synth_render import _mask_to_yolo_polygon, _render_arrow_mask, _render_curve_mask, _render_error_bar_mask, _render_hbar_mask, _render_vbar_mask


def _resolve_difficulty_settings(difficulty: int) -> dict[str, Any] | None:
    """Return per-difficulty overrides for count ranges and feature flags.

    Returns ``None`` for *difficulty* == 0, meaning "use all existing defaults"
    (backward-compatible behaviour for the compatibility facade and tests).

    Difficulty levels:
    * 1 – easy: 1–2 curves, no annotations, mild degradation, no legend.
    * 2 – medium-easy: 2–3 curves, vbar/hbar only, no arrows/error-bars, no legend.
    * 3 – medium-hard: normal curve count, all annotation types, legend possible.
    * 4 – hard: dense curves, max annotations, legend always on, heavy degradation.
    """
    if difficulty == 0:
        return None
    if difficulty == 1:
        return {
            "curve_count_range": (1, 2),
            "vbar_count_range": (0, 0),
            "hbar_count_range": (0, 0),
            "arrow_count_range": (0, 0),
            "error_bar_count_range": (0, 0),
            "show_legend": False,
            "add_arrow_labels": False,
            "degradation_intensity": "mild",
        }
    if difficulty == 2:
        return {
            "curve_count_range": (2, 3),
            "vbar_count_range": (0, 1),
            "hbar_count_range": (0, 1),
            "arrow_count_range": (0, 0),
            "error_bar_count_range": (0, 2),
            "show_legend": False,
            "add_arrow_labels": False,
            "degradation_intensity": "normal",
        }
    if difficulty == 3:
        return {
            "curve_count_range": None,
            "vbar_count_range": None,
            "hbar_count_range": None,
            "arrow_count_range": (0, 1),
            "error_bar_count_range": None,
            "show_legend": None,  # resolved by caller via LEGEND_PROBABILITY
            "add_arrow_labels": True,
            "degradation_intensity": "normal",
        }
    # difficulty == 4
    return {
        "curve_count_range": DENSE_CURVE_COUNT_RANGE,
        "vbar_count_range": None,
        "hbar_count_range": None,
        "arrow_count_range": (1, 3),
        "error_bar_count_range": None,
        "show_legend": True,
        "add_arrow_labels": True,
        "degradation_intensity": "heavy",
    }


def _write_synthetic_example(
    index: int,
    output_dir: Path,
    rng: np.random.Generator,
    image_format: str,
    plot_type: str = "general",
    degradations: int = 1,
    difficulty: int = 0,
    *,
    apply_degradation_filters_fn: Any = None,
    render_curve_mask_fn: Any = None,
    render_vbar_mask_fn: Any = None,
    render_hbar_mask_fn: Any = None,
    render_arrow_mask_fn: Any = None,
    render_error_bar_mask_fn: Any = None,
) -> None:
    """Generate a synthetic plot with support for bandstructures and complex annotations.

    *difficulty* controls the curriculum level (0 = legacy/full-feature defaults,
    1–4 = progressively harder; see :func:`_resolve_difficulty_settings`).

    When *degradations* > 1, the same base plot is saved as *degradations* separate
    images under different random degradation conditions.  The YOLO label file,
    annotations file, and ground-truth CSV are written once per base plot and shared
    across all degraded variants (since geometry is unchanged by degradation).
    """
    apply_degradation_filters_fn = apply_degradation_filters_fn or _apply_degradation_filters
    render_curve_mask_fn = render_curve_mask_fn or _render_curve_mask
    render_vbar_mask_fn = render_vbar_mask_fn or _render_vbar_mask
    render_hbar_mask_fn = render_hbar_mask_fn or _render_hbar_mask
    render_arrow_mask_fn = render_arrow_mask_fn or _render_arrow_mask
    render_error_bar_mask_fn = render_error_bar_mask_fn or _render_error_bar_mask

    settings = _resolve_difficulty_settings(difficulty)

    fig_size = (6.0, 4.2)
    dpi = DEFAULT_DPI
    base_stem = f"plot_{index:04d}"
    label_path = output_dir / "labels" / f"{base_stem}.txt"
    metadata_path = output_dir / "images" / f"{base_stem}.metadata.json"
    annotations_path = output_dir / "annotations" / f"{base_stem}.json"
    csv_path = output_dir / "csv" / f"{base_stem}.csv"
    single_mode = degradations == 1
    if single_mode:
        image_path = output_dir / "images" / f"{base_stem}.{image_format}"
    else:
        image_path = output_dir / "images" / f"{base_stem}_clean.{image_format}"

    use_log_x = bool(rng.random() < LOG_X_PROBABILITY)
    x_range = ((LOG_X_MIN if use_log_x else 0.0), float(rng.uniform(6.0, 12.0)))
    x_values = np.geomspace(*x_range, 480) if use_log_x else np.linspace(*x_range, 480)

    fig, ax = _create_plot_axes(fig_size, dpi, x_range, use_log_x, rng)
    raw_curves, y_range, annotation_descriptors = _configure_plot_curves(
        plot_type, ax, x_values, rng,
        curve_count_range=settings["curve_count_range"] if settings else None,
    )

    # Resolve show_legend for this sample.
    if settings is None:
        show_legend = False
        add_arrow_labels = False  # preserves rng sequence for difficulty=0 (backward compat)
        degradation_intensity = "normal"
    elif settings["show_legend"] is None:
        show_legend = bool(rng.random() < LEGEND_PROBABILITY)
        add_arrow_labels = bool(settings["add_arrow_labels"])
        degradation_intensity = str(settings["degradation_intensity"])
    else:
        show_legend = bool(settings["show_legend"])
        add_arrow_labels = bool(settings["add_arrow_labels"])
        degradation_intensity = str(settings["degradation_intensity"])

    x_axis_scale = "log" if use_log_x else "linear"
    x_norm_to_data = lambda norm_x: float(_norm_to_scale(np.array([float(np.clip(norm_x, 0.0, 1.0))], dtype=float), x_range[0], x_range[1], x_axis_scale)[0])
    y_norm_to_data = lambda norm_y: float(y_range[0] + float(np.clip(norm_y, 0.0, 1.0)) * (y_range[1] - y_range[0]))

    ground_truth_frames, curve_descriptors, label_lines = _add_curve_layers(
        ax, x_values, x_range, y_range, use_log_x, raw_curves, rng, show_legend=show_legend,
    )
    new_label_lines, new_annotation_descriptors = _add_annotation_layers(
        ax, rng, fig_size, dpi, x_range, y_range, use_log_x,
        x_norm_to_data, y_norm_to_data,
        vbar_count_range=settings["vbar_count_range"] if settings else None,
        hbar_count_range=settings["hbar_count_range"] if settings else None,
        arrow_count_range=settings["arrow_count_range"] if settings else None,
        error_bar_count_range=settings["error_bar_count_range"] if settings else None,
        add_arrow_labels=add_arrow_labels,
    )
    label_lines.extend(new_label_lines)
    annotation_descriptors.extend(new_annotation_descriptors)

    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axis_bbox = ax.get_window_extent(renderer=renderer)
    ax_position = axis_bbox.transformed(fig.transFigure.inverted()).bounds

    for i, desc in enumerate(curve_descriptors):
        df = ground_truth_frames[i]
        mask = render_curve_mask_fn(
            fig_size, dpi, df["x_real"].values, df["y_real"].values, x_range, y_range, desc,
            x_scale="log" if use_log_x else "linear", ax_position=ax_position,
        )
        polygon = _mask_to_yolo_polygon(mask)
        if polygon:
            label_lines.append("0 " + " ".join(f"{value:.6f}" for value in polygon))

    for desc in annotation_descriptors:
        t = desc["type"]
        style = desc.get("style", {"linewidth": 1.5, "linestyle": "--"})
        if t == "vbar":
            mask = render_vbar_mask_fn(fig_size, dpi, desc["x_pos"], y_range, style["linewidth"], style, ax_position=ax_position)
        elif t == "hbar":
            y_pos_norm = desc.get("y_pos_norm", (desc["y_pos"] - y_range[0]) / (y_range[1] - y_range[0]))
            mask = render_hbar_mask_fn(fig_size, dpi, y_pos_norm, x_range, style["linewidth"], style, x_scale="log" if use_log_x else "linear", ax_position=ax_position)
        elif t == "arrow":
            mask = render_arrow_mask_fn(fig_size, dpi, desc["start"], desc["end"], style, ax_position=ax_position)
        elif t == "error_bar":
            mask = render_error_bar_mask_fn(fig_size, dpi, desc["x_pos"], desc["y_pos"], desc["y_err"], style, ax_position=ax_position)
        else:
            continue
        polygon = _mask_to_yolo_polygon(mask)
        if polygon:
            label_lines.append(f"{desc['class_id']} " + " ".join(f"{value:.6f}" for value in polygon))

    _save_synthetic_outputs(
        fig, ax, image_path, image_format, label_path, metadata_path, annotations_path, csv_path,
        x_range, y_range, use_log_x, plot_type,
        ground_truth_frames, label_lines, curve_descriptors, annotation_descriptors,
    )
    plt.close(fig)

    if single_mode:
        apply_degradation_filters_fn(image_path, rng, degradation_intensity)
    else:
        deg_seeds = rng.integers(0, 2**31, size=degradations)
        variant_images: list[Path] = []
        variant_labels: list[Path] = []
        for j in range(degradations):
            variant_stem = f"{base_stem}_deg{j:02d}"
            variant_image = output_dir / "images" / f"{variant_stem}.{image_format}"
            variant_label = output_dir / "labels" / f"{variant_stem}.txt"
            shutil.copy2(image_path, variant_image)
            shutil.copy2(label_path, variant_label)
            deg_rng = np.random.default_rng(int(deg_seeds[j]))
            apply_degradation_filters_fn(variant_image, deg_rng, degradation_intensity)
            variant_images.append(variant_image)
            variant_labels.append(variant_label)
        image_path.unlink(missing_ok=True)
        label_path.unlink(missing_ok=True)
