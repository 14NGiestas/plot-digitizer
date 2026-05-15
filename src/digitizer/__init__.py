"""Automatic plot digitizer CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import synthetic as _synthetic
from .ai_segmentation import run_ai_segmentation
from .axis_parsing import _resolve_bounds_from_references, parse_range, parse_reference_pair
from .calibration import calibrate_axes, detect_axis_anchor_pixels
from .cli_support import _set_matplotlib_backend, configure_logging
from .constants import LOGGER
from .curve_utils import _interp_curve, _prepare_curve_points
from .cv_segmentation import (
    _cluster_by_color,
    _cluster_by_geometry,
    _foreground_mask,
    _saturated_mask,
    run_cv_segmentation,
)
from .digitize_workflow import digitize_image
from .image_ops import (
    _parse_sidecar_metadata,
    detect_plot_box,
    discover_images,
    load_image,
    preprocess_image,
    resolve_plot_box,
)
from .interactive_axis import _format_reference_pair_cli_value, interactive_reference_selection
from .math_utils import _norm_to_scale, _rectangle, _remove_small_regions
from .models import AxisCalibration, AxisReferencePair, DigitizeResult, PlotBox, SegmentationResult
from .parser import _json_default, _parse_positive_int, build_parser
from .plotting import build_replot_frame, create_overlay, create_replot
from .points import _split_large_components, convert_points, extract_curve_points
from .synthetic import (
    _apply_degradation_filters,
    _render_arrow_mask,
    _render_curve_mask,
    _render_error_bar_mask,
    _render_hbar_mask,
    _render_vbar_mask,
    generate_synthetic_dataset,
    run_training,
)
from .validation import validate_digitization


def _write_synthetic_example(
    index: int,
    output_dir: Path,
    rng: np.random.Generator,
    image_format: str,
    plot_type: str = "general",
) -> None:
    """Compatibility wrapper for tests patching legacy `digitizer.*` private helpers.

    New internal code should call ``digitizer.synthetic._write_synthetic_example``
    directly instead of relying on this legacy patch-routing wrapper.
    """
    _synthetic._write_synthetic_example(
        index,
        output_dir,
        rng,
        image_format,
        plot_type,
        apply_degradation_filters_fn=_apply_degradation_filters,
        render_curve_mask_fn=_render_curve_mask,
        render_vbar_mask_fn=_render_vbar_mask,
        render_hbar_mask_fn=_render_hbar_mask,
        render_arrow_mask_fn=_render_arrow_mask,
        render_error_bar_mask_fn=_render_error_bar_mask,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    if args.command == "generate":
        generate_synthetic_dataset(
            args.output_dir, args.count, args.seed, args.image_format, args.plot_type,
            workers=args.workers,
        )
        LOGGER.info("Generated %s synthetic plots (%s) in %s", args.count, args.plot_type, args.output_dir)
        return 0

    if args.command == "train":
        plan = run_training(
            args.dataset_dir,
            args.output_dir,
            args.epochs,
            args.imgsz,
            args.weights,
            args.batch,
            args.execute,
            args.hyp_yaml,
            workers=args.workers,
        )
        print(json.dumps(plan, indent=2, default=_json_default))
        return 0

    if args.command == "digitize":
        images = discover_images(args.inputs)
        if not images:
            parser.error("No input images were found.")
        
        # Set matplotlib backend based on interactive mode
        if args.interactive_axis_selection:
            _set_matplotlib_backend("TkAgg")  # Interactive backend for GUI
        else:
            _set_matplotlib_backend("Agg")  # Non-interactive backend
        
        x_range = parse_range(args.x_range)
        y_range = parse_range(args.y_range)
        x_reference = parse_reference_pair(args.x_reference, "x")
        y_reference = parse_reference_pair(args.y_reference, "y")
        if args.interactive_axis_selection and (x_reference is not None or y_reference is not None):
            parser.error("Cannot combine --interactive-axis-selection with --x-reference or --y-reference.")
        if args.interactive_axis_selection and images:
            x_reference, y_reference = interactive_reference_selection(images[0])
            LOGGER.info(
                "Interactive axis selection complete. Reuse with: "
                "--x-reference \"%s\" --y-reference \"%s\"",
                _format_reference_pair_cli_value(x_reference),
                _format_reference_pair_cli_value(y_reference),
            )
        results = []
        for image_path in images:
            result = digitize_image(
                image_path=image_path,
                output_dir=args.output_dir,
                x_range=x_range,
                y_range=y_range,
                x_reference=x_reference,
                y_reference=y_reference,
                x_scale=args.x_scale,
                y_scale=args.y_scale,
                invert_y=args.invert_y,
                weights=args.weights,
                conf_threshold=args.conf_threshold,
                create_overlay_image=args.overlay,
                workers=args.workers,
                auto_axis_anchors=not args.disable_auto_axis_anchors,
            )
            results.append(
                {
                    "image": str(image_path),
                    "csv_path": str(result.csv_path),
                    "replot_csv_path": str(result.replot_csv_path),
                    "metadata_path": str(result.metadata_path),
                    "replot_path": str(result.replot_path),
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

