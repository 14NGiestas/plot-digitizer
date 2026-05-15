"""Curve interpolation helpers shared by exports and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def _prepare_curve_points(points: pd.DataFrame) -> pd.DataFrame:
    """Return one curve sorted by x with duplicate x-values removed."""
    return points.drop_duplicates(subset="x_real").sort_values("x_real")


def _interp_curve(points: pd.DataFrame, reference_x: np.ndarray) -> np.ndarray:
    """Linearly interpolate one curve onto a shared x-grid for validation/export."""
    if len(points) < 2:
        raise ValueError("At least two points are required for interpolation.")
    unique = _prepare_curve_points(points)
    interpolator = interp1d(unique["x_real"], unique["y_real"], fill_value="extrapolate")
    return interpolator(reference_x)

