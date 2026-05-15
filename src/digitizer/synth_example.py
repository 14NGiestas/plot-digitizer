"""Synthetic single-example rendering and writing helpers."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .constants import DEFAULT_DPI, LOG_X_MIN, LOG_X_PROBABILITY
from .math_utils import _norm_to_scale
from .synth_annotations import _add_annotation_layers
from .synth_curve_plotting import _add_curve_layers, _configure_plot_curves, _create_plot_axes
from .synth_degrade import _apply_degradation_filters
from .synth_output import _save_synthetic_outputs
from .synth_render import _render_arrow_mask, _render_curve_mask, _render_error_bar_mask, _render_hbar_mask, _render_vbar_mask


def _write_synthetic_example(
    index: int,
    output_dir: Path,
    rng: np.random.Generator,
    image_format: str,
    plot_type: str = "general",
    degradations: int = 1,
    *,
    apply_degradation_filters_fn: Any = None,
    render_curve_mask_fn: Any = None,
    render_vbar_mask_fn: Any = None,
    render_hbar_mask_fn: Any = None,
    render_arrow_mask_fn: Any = None,
    render_error_bar_mask_fn: Any = None,
) -> None:
    """Generate a synthetic plot with support for bandstructures and complex annotations.

    When *degradations* > 1, the same base plot is saved as *degradations* separate
    images under different random degradation conditions.  The YOLO label file,
    annotations file, and ground-truth CSV are written once per base plot and shared
    across all degraded variants (since geometry is unchanged by degradation).

    Degraded variants are named ``plot_{index:04d}_deg{j:02d}.{ext}`` for
    ``j`` in ``range(degradations)``.  When *degradations* == 1 (the default) the
    variant suffix is omitted and the file is named ``plot_{index:04d}.{ext}``,
    preserving backward compatibility.
    """
    apply_degradation_filters_fn = apply_degradation_filters_fn or _apply_degradation_filters
    render_curve_mask_fn = render_curve_mask_fn or _render_curve_mask
    render_vbar_mask_fn = render_vbar_mask_fn or _render_vbar_mask
    render_hbar_mask_fn = render_hbar_mask_fn or _render_hbar_mask
    render_arrow_mask_fn = render_arrow_mask_fn or _render_arrow_mask
    render_error_bar_mask_fn = render_error_bar_mask_fn or _render_error_bar_mask

    fig_size = (6.0, 4.2)
    dpi = DEFAULT_DPI
    # Base paths (used for labels, annotations, csv, and metadata).
    base_stem = f"plot_{index:04d}"
    label_path = output_dir / "labels" / f"{base_stem}.txt"
    metadata_path = output_dir / "images" / f"{base_stem}.metadata.json"
    annotations_path = output_dir / "annotations" / f"{base_stem}.json"
    csv_path = output_dir / "csv" / f"{base_stem}.csv"
    # When degradations == 1 keep old single-file naming; otherwise write a
    # clean base image temporarily and copy+degrade into variant files.
    single_mode = degradations == 1
    if single_mode:
        image_path = output_dir / "images" / f"{base_stem}.{image_format}"
    else:
        # Write the clean matplotlib figure to a temp file first.
        image_path = output_dir / "images" / f"{base_stem}_clean.{image_format}"

    use_log_x = bool(rng.random() < LOG_X_PROBABILITY)
    x_range = ((LOG_X_MIN if use_log_x else 0.0), float(rng.uniform(6.0, 12.0)))
    x_values = np.geomspace(*x_range, 480) if use_log_x else np.linspace(*x_range, 480)

    fig, ax = _create_plot_axes(fig_size, dpi, x_range, use_log_x, rng)
    raw_curves, y_range, annotation_descriptors = _configure_plot_curves(plot_type, ax, x_values, rng)

    x_axis_scale = "log" if use_log_x else "linear"
    x_norm_to_data = lambda norm_x: float(_norm_to_scale(np.array([float(np.clip(norm_x, 0.0, 1.0))], dtype=float), x_range[0], x_range[1], x_axis_scale)[0])
    y_norm_to_data = lambda norm_y: float(y_range[0] + float(np.clip(norm_y, 0.0, 1.0)) * (y_range[1] - y_range[0]))

    ground_truth_frames, curve_descriptors, label_lines = _add_curve_layers(
        ax, x_values, x_range, y_range, use_log_x, raw_curves, rng, fig_size, dpi, render_curve_mask_fn,
    )
    new_label_lines, new_annotation_descriptors = _add_annotation_layers(
        ax,
        rng,
        fig_size,
        dpi,
        x_range,
        y_range,
        use_log_x,
        x_norm_to_data,
        y_norm_to_data,
        render_vbar_mask_fn,
        render_hbar_mask_fn,
        render_arrow_mask_fn,
        render_error_bar_mask_fn,
    )
    label_lines.extend(new_label_lines)
    annotation_descriptors.extend(new_annotation_descriptors)
    _save_synthetic_outputs(
        fig,
        ax,
        image_path,
        image_format,
        label_path,
        metadata_path,
        annotations_path,
        csv_path,
        x_range,
        y_range,
        use_log_x,
        plot_type,
        ground_truth_frames,
        label_lines,
        curve_descriptors,
        annotation_descriptors,
    )
    plt.close(fig)

    if single_mode:
        # Classic single-image mode: degrade in-place.
        apply_degradation_filters_fn(image_path, rng)
    else:
        # Multi-degradation mode: produce N independently degraded copies of the
        # clean base image.  Each copy gets its own degradation rng derived from
        # the parent so the sequence is fully deterministic.
        deg_seeds = rng.integers(0, 2**31, size=degradations)
        for j in range(degradations):
            variant_stem = f"{base_stem}_deg{j:02d}"
            variant_image = output_dir / "images" / f"{variant_stem}.{image_format}"
            variant_label = output_dir / "labels" / f"{variant_stem}.txt"
            shutil.copy2(image_path, variant_image)
            # Copy label file so every image→label pair is self-contained.
            shutil.copy2(label_path, variant_label)
            deg_rng = np.random.default_rng(int(deg_seeds[j]))
            apply_degradation_filters_fn(variant_image, deg_rng)
        # Remove the clean base image and base label — they were only needed
        # as copy sources; the variant files are the canonical training samples.
        image_path.unlink(missing_ok=True)
        label_path.unlink(missing_ok=True)
