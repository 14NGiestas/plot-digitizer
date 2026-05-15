"""Synthetic curve generation helpers."""

from __future__ import annotations

import math

import numpy as np

from .constants import (
    DAMPED_AMPLITUDE_RANGE,
    DAMPED_DECAY_RANGE,
    DAMPED_FREQUENCY_RANGE,
    EXP_GROWTH_RANGE,
    EXP_OFFSET_RANGE,
    EXP_SCALE_RANGE,
    NOISE_STD_RANGE,
    POLY_A_RANGE,
    POLY_B_RANGE,
    POLY_C_RANGE,
    SINE_AMPLITUDE_RANGE,
    SINE_FREQUENCY_RANGE,
    SINE_OFFSET_RANGE,
)

def _random_curve(x_values: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    curve_type = rng.choice(["sin", "poly", "exp", "damped"])
    if curve_type == "sin":
        amplitude = rng.uniform(*SINE_AMPLITUDE_RANGE)
        frequency = rng.uniform(*SINE_FREQUENCY_RANGE)
        phase = rng.uniform(0.0, math.pi)
        offset = rng.uniform(*SINE_OFFSET_RANGE)
        y_values = offset + amplitude * np.sin(frequency * x_values + phase)
    elif curve_type == "exp":
        scale = rng.uniform(*EXP_SCALE_RANGE)
        growth = rng.uniform(*EXP_GROWTH_RANGE)
        offset = rng.uniform(*EXP_OFFSET_RANGE)
        y_values = offset + scale * np.exp(growth * (x_values - x_values.min()))
    elif curve_type == "damped":
        amplitude = rng.uniform(*DAMPED_AMPLITUDE_RANGE)
        decay = rng.uniform(*DAMPED_DECAY_RANGE)
        frequency = rng.uniform(*DAMPED_FREQUENCY_RANGE)
        y_values = amplitude * np.exp(-decay * x_values) * np.cos(frequency * x_values)
    else:
        a, b, c = rng.uniform(*POLY_A_RANGE), rng.uniform(*POLY_B_RANGE), rng.uniform(*POLY_C_RANGE)
        y_values = a * (x_values * x_values) + b * x_values + c
    noise = rng.normal(0.0, rng.uniform(*NOISE_STD_RANGE), size=x_values.shape)
    return y_values + noise, str(curve_type)


def _generate_bandstructure_curves(x_values: np.ndarray, rng: np.random.Generator, n_bands: int) -> list[tuple[np.ndarray, str]]:
    """Generate bandstructure-like curves with multiple bands and avoided crossings."""
    bands = []

    # Generate base parabolic bands
    for band_idx in range(n_bands):
        band_offset = rng.uniform(-1.5, 1.5)
        effective_mass = rng.uniform(0.3, 1.2)
        curvature = rng.choice([-1, 1]) * effective_mass

        # Base parabola
        x_centered = x_values - x_values.mean()
        y_base = band_offset + curvature * (x_centered ** 2) / (x_centered.max() ** 2 + 0.1)

        # Add avoided crossing features
        if rng.random() > 0.5 and band_idx < n_bands - 1:
            crossing_x = rng.uniform(x_values.min(), x_values.max())
            gap_size = rng.uniform(0.1, 0.4)
            avoidance = gap_size * np.exp(-((x_values - crossing_x) ** 2) / 0.5)
            y_base += avoidance * curvature

        # Add small oscillations (umklapp-like features)
        if rng.random() > 0.6:
            osc_freq = rng.uniform(2, 5)
            osc_amp = rng.uniform(0.02, 0.08)
            y_base += osc_amp * np.sin(osc_freq * x_values + rng.uniform(0, 2 * np.pi))

        noise = rng.normal(0.0, rng.uniform(0.005, 0.02), size=x_values.shape)
        bands.append((y_base + noise, f"band_{band_idx}"))

    return bands

