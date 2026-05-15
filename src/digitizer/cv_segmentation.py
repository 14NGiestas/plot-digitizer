"""Computer-vision curve segmentation helpers."""

from __future__ import annotations

import cv2
import numpy as np
from skimage import morphology
from sklearn.cluster import DBSCAN, MiniBatchKMeans

from .constants import (
    BASE_CV_CONFIDENCE,
    DBSCAN_NOISE_LABEL,
    MAX_CLUSTER_SAMPLE_SIZE,
    MAX_COLOR_CLUSTERS,
    MAX_CV_CONFIDENCE,
    MIN_COMPONENT_PIXELS,
    MINIBATCH_KMEANS_BATCH_SIZE,
)
from .math_utils import _rectangle, _remove_small_regions
from .models import PlotBox, SegmentationResult

def _foreground_mask(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    foreground = gray < np.percentile(gray, 96)
    foreground[:3, :] = False
    foreground[-3:, :] = False
    foreground[:, :3] = False
    foreground[:, -3:] = False
    return _remove_small_regions(foreground, MIN_COMPONENT_PIXELS)


def _saturated_mask(crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    saturated = (saturation > 45) & (value < 245)
    saturated = _remove_small_regions(saturated, MIN_COMPONENT_PIXELS)
    return morphology.closing(saturated, _rectangle(3, 7))


def _cluster_by_color(crop: np.ndarray, foreground: np.ndarray) -> list[np.ndarray]:
    pixels = crop[foreground]
    if len(pixels) < 100:
        return []
    filtered = pixels.astype(np.float32)
    sample_count = min(len(filtered), MAX_CLUSTER_SAMPLE_SIZE)
    sample_indices = np.linspace(0, len(filtered) - 1, sample_count).astype(int)
    sample = filtered[sample_indices]
    cluster_count = int(min(MAX_COLOR_CLUSTERS, max(1, len(np.unique(sample, axis=0)))))
    if cluster_count <= 1:
        return []
    model = MiniBatchKMeans(
        n_clusters=cluster_count,
        n_init=5,
        random_state=42,
        batch_size=MINIBATCH_KMEANS_BATCH_SIZE,
    )
    model.fit(sample)
    labels = model.predict(filtered)
    masks: list[np.ndarray] = []
    for cluster_id in range(cluster_count):
        cluster_mask = np.zeros(foreground.shape, dtype=bool)
        cluster_mask[foreground] = labels == cluster_id
        cluster_mask = morphology.closing(cluster_mask, _rectangle(3, 9))
        cluster_mask = _remove_small_regions(cluster_mask, 80)
        if cluster_mask.sum() >= MIN_COMPONENT_PIXELS:
            masks.append(cluster_mask)
    return masks


def _cluster_by_geometry(foreground: np.ndarray) -> list[np.ndarray]:
    ys, xs = np.nonzero(foreground)
    if len(xs) < MIN_COMPONENT_PIXELS:
        return []
    sample_size = min(len(xs), 1500)
    indices = np.linspace(0, len(xs) - 1, sample_size).astype(int)
    points = np.column_stack((xs[indices] / max(1, foreground.shape[1]), ys[indices] / max(1, foreground.shape[0])))
    clustering = DBSCAN(eps=0.04, min_samples=15).fit(points)
    masks: list[np.ndarray] = []
    for cluster_id in sorted(set(clustering.labels_) - {DBSCAN_NOISE_LABEL}):
        sample_mask = np.zeros(foreground.shape, dtype=bool)
        sample_mask[ys[indices][clustering.labels_ == cluster_id], xs[indices][clustering.labels_ == cluster_id]] = True
        sample_mask = morphology.binary_dilation(sample_mask, morphology.disk(2))
        sample_mask = morphology.binary_closing(sample_mask, morphology.disk(2))
        sample_mask &= foreground
        if sample_mask.sum() >= MIN_COMPONENT_PIXELS:
            masks.append(sample_mask)
    return masks


def run_cv_segmentation(image: np.ndarray, plot_box: PlotBox) -> list[SegmentationResult]:
    """Segment curves using color and geometric clustering."""
    crop = image[plot_box.top : plot_box.bottom, plot_box.left : plot_box.right]
    foreground = _saturated_mask(crop)
    if foreground.sum() < MIN_COMPONENT_PIXELS:
        foreground = _foreground_mask(crop)
    if foreground.sum() < MIN_COMPONENT_PIXELS:
        return []

    candidate_masks = _cluster_by_color(crop, foreground)
    method = "cv_color"
    split_components = False
    if not candidate_masks:
        candidate_masks = _cluster_by_geometry(foreground)
        method = "cv_geometry"
        split_components = True
    if not candidate_masks:
        candidate_masks = [foreground]
        method = "cv_binary"
        split_components = True

    results: list[SegmentationResult] = []
    for index, local_mask in enumerate(candidate_masks):
        local_mask = morphology.closing(local_mask, _rectangle(3, 9))
        local_mask = _remove_small_regions(local_mask, 120)
        horizontal_coverage = np.mean(np.any(local_mask, axis=0))
        if horizontal_coverage < 0.15:
            continue
        global_mask = np.zeros(image.shape[:2], dtype=bool)
        global_mask[plot_box.top : plot_box.bottom, plot_box.left : plot_box.right] = local_mask
        if global_mask.sum() < MIN_COMPONENT_PIXELS:
            continue
        confidence = float(
            min(
                MAX_CV_CONFIDENCE,
                BASE_CV_CONFIDENCE + global_mask.sum() / max(1, plot_box.width * plot_box.height),
            )
        )
        results.append(
            SegmentationResult(
                dataset_id=f"dataset_{index}",
                mask=global_mask,
                confidence=confidence,
                method=method,
                split_components=split_components,
            )
        )
    return results
