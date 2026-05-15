"""Small math and morphology helpers."""

from __future__ import annotations

import numpy as np
from skimage import morphology

def _norm_to_scale(values: np.ndarray, minimum: float, maximum: float, scale: str) -> np.ndarray:
    if scale == "log":
        if minimum <= 0 or maximum <= 0:
            raise ValueError("Logarithmic axes require positive bounds.")
        return np.exp(np.log(minimum) + values * (np.log(maximum) - np.log(minimum)))
    return minimum + values * (maximum - minimum)


def _remove_small_regions(mask: np.ndarray, min_area: int) -> np.ndarray:
    cleaned = morphology.area_opening(mask.astype(np.uint8), area_threshold=min_area)
    return cleaned.astype(bool)


def _rectangle(height: int, width: int) -> np.ndarray:
    if hasattr(morphology, "footprint_rectangle"):
        return morphology.footprint_rectangle((height, width))
    return morphology.rectangle(height, width)

