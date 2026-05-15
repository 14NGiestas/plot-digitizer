"""AI-based segmentation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .constants import LOGGER, MIN_COMPONENT_PIXELS
from .models import PlotBox, SegmentationResult

def run_ai_segmentation(
    image: np.ndarray,
    plot_box: PlotBox,
    weights: str | None,
    conf_threshold: float,
    workers: int | None = None,
) -> list[SegmentationResult]:
    """Run YOLO segmentation if weights are available."""
    if not weights:
        return []
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - fallback path when ultralytics is unavailable
        LOGGER.warning("Ultralytics import failed, falling back to CV segmentation: %s", exc)
        return []

    model = YOLO(weights)
    predict_kwargs: dict[str, Any] = {"conf": conf_threshold, "verbose": False}
    if workers is not None:
        predict_kwargs["workers"] = workers
    predictions = model.predict(image, **predict_kwargs)
    results: list[SegmentationResult] = []
    if not predictions:
        return results
    masks = getattr(predictions[0], "masks", None)
    if masks is None or masks.data is None:
        return results
    for index, mask_tensor in enumerate(masks.data):
        mask = (mask_tensor.cpu().numpy() > 0.5).astype(np.uint8)
        cropped = np.zeros_like(mask)
        cropped[plot_box.top : plot_box.bottom, plot_box.left : plot_box.right] = mask[plot_box.top : plot_box.bottom, plot_box.left : plot_box.right]
        if cropped.sum() < MIN_COMPONENT_PIXELS:
            continue
        confidence = float(predictions[0].boxes.conf[index].cpu().item()) if predictions[0].boxes is not None else conf_threshold
        class_id = int(predictions[0].boxes.cls[index].cpu().item()) if predictions[0].boxes is not None else None
        results.append(
            SegmentationResult(
                dataset_id=f"dataset_{index}",
                mask=cropped.astype(bool),
                confidence=confidence,
                method="ai",
                class_id=class_id,
            )
        )
    return results

