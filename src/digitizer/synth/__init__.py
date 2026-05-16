"""Synthetic data generation module for plot digitizer."""

from .dataset import generate_synthetic_dataset
from .example import _write_synthetic_example
from .curves import _generate_bandstructure_curves, _random_curve

__all__ = [
    "generate_synthetic_dataset",
    "_write_synthetic_example",
    "_generate_bandstructure_curves",
    "_random_curve",
]
