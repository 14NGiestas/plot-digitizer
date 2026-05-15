"""Synthetic image degradation helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .constants import LOGGER


def _apply_degradation_filters(image_path: Path, rng: np.random.Generator) -> None:
    """Apply degradation filters to simulate old/scanned article quality."""
    image = cv2.imread(str(image_path))
    if image is None:
        LOGGER.warning("Could not load image for degradation: %s", image_path)
        return

    apply_jpeg = rng.random() > 0.3
    apply_noise = rng.random() > 0.4
    apply_blur = rng.random() > 0.5
    apply_contrast = rng.random() > 0.6
    apply_bw = rng.random() > 0.85
    apply_salt_pepper = rng.random() > 0.7
    degraded = image.copy()

    if apply_jpeg:
        quality = int(rng.uniform(15, 75))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', degraded, encode_param)
        degraded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if apply_noise:
        noise_std = rng.uniform(5, 25)
        noise = rng.normal(0, noise_std, degraded.shape).astype(np.int16)
        degraded = np.clip(degraded.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if apply_blur:
        kernel_size = int(rng.choice([3, 5, 7]))
        degraded = cv2.GaussianBlur(degraded, (kernel_size, kernel_size), 0)
    if apply_contrast:
        alpha = rng.uniform(0.6, 0.9)
        beta = rng.uniform(-10, 10)
        degraded = cv2.convertScaleAbs(degraded, alpha=alpha, beta=beta)
    if apply_salt_pepper:
        salt_pepper_prob = rng.uniform(0.005, 0.02)
        salt_mask = rng.random(degraded.shape[:2]) < salt_pepper_prob
        pepper_mask = rng.random(degraded.shape[:2]) < salt_pepper_prob
        for color_channel in range(3):
            degraded[salt_mask, color_channel] = 255
            degraded[pepper_mask, color_channel] = 0
    if apply_bw:
        gray = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
        if rng.random() > 0.5:
            _, degraded_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            degraded = cv2.cvtColor(degraded_binary, cv2.COLOR_GRAY2BGR)
        else:
            degraded = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(str(image_path), degraded)
    LOGGER.debug(
        'Applied degradations to %s: jpeg=%s, noise=%s, blur=%s, contrast=%s, bw=%s, salt_pepper=%s',
        image_path.name,
        apply_jpeg,
        apply_noise,
        apply_blur,
        apply_contrast,
        apply_bw,
        apply_salt_pepper,
    )
