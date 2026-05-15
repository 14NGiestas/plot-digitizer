"""Compatibility facade for synthetic generation and training helpers."""

from __future__ import annotations

from .synth_dataset import SampleGenerationTask, _generate_one_sample, generate_synthetic_dataset
from .synth_degrade import _apply_degradation_filters
from .synth_example import _write_synthetic_example
from .synth_render import (
    _mask_to_yolo_polygon,
    _render_arrow_mask,
    _render_curve_mask,
    _render_error_bar_mask,
    _render_hbar_mask,
    _render_vbar_mask,
)
from .training import run_training
