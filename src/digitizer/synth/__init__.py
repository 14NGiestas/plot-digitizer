"""Synthetic data generation module for plot digitizer."""

from .dataset import generate_synthetic_dataset
from .example import _write_synthetic_example
from .curves import _generate_bandstructure_curves, _random_curve
from .degrade import _apply_degradation_filters
from .render import (
    _mask_to_yolo_polygon,
    _render_arrow_mask,
    _render_curve_mask,
    _render_error_bar_mask,
    _render_hbar_mask,
    _render_vbar_mask,
)

__all__ = [
    "generate_synthetic_dataset",
    "_write_synthetic_example",
    "_generate_bandstructure_curves",
    "_random_curve",
    "_apply_degradation_filters",
    "_mask_to_yolo_polygon",
    "_render_arrow_mask",
    "_render_curve_mask",
    "_render_error_bar_mask",
    "_render_hbar_mask",
    "_render_vbar_mask",
]
