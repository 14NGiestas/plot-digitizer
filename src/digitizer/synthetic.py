"""Synthetic dataset generation and training helpers for digitizer."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import math
import multiprocessing
import os
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import measure

LOGGER = logging.getLogger("plot_digitizer")
DEFAULT_DPI = 140
MAX_POLYGON_POINTS = 200
DEFAULT_GENERATE_WORKERS_CAP = 8
SINE_AMPLITUDE_RANGE = (0.5, 1.8)
SINE_FREQUENCY_RANGE = (0.6, 2.4)
SINE_OFFSET_RANGE = (-0.75, 0.75)
EXP_SCALE_RANGE = (0.2, 1.1)
EXP_GROWTH_RANGE = (0.15, 0.55)
EXP_OFFSET_RANGE = (-0.8, 0.3)
DAMPED_AMPLITUDE_RANGE = (0.8, 1.8)
DAMPED_DECAY_RANGE = (0.05, 0.2)
DAMPED_FREQUENCY_RANGE = (1.0, 2.6)
POLY_A_RANGE = (-0.05, 0.05)
POLY_B_RANGE = (-0.4, 0.4)
POLY_C_RANGE = (-0.8, 0.8)
NOISE_STD_RANGE = (0.01, 0.05)
DENSE_CURVE_PROBABILITY = 0.4
DENSE_CURVE_COUNT_RANGE = (4, 6)
BASE_CURVE_COUNT_RANGE = (2, 4)
VBAR_COUNT_RANGE = (1, 3)
HBAR_COUNT_RANGE = (1, 2)
ARROW_COUNT_RANGE = (0, 2)
ERROR_BAR_COUNT_RANGE = (2, 5)
CURVE_LINEWIDTHS = [0.6, 0.8, 1.0, 1.2, 1.6, 2.0]
CURVE_LINEWIDTH_PROBABILITIES = [0.28, 0.24, 0.2, 0.14, 0.09, 0.05]
GRID_ENABLED_PROBABILITY = 0.6
GRID_ALPHA = 0.4
LOG_X_PROBABILITY = 0.3
LOG_X_MIN = 0.1


def _norm_to_scale(values: np.ndarray, minimum: float, maximum: float, scale: str) -> np.ndarray:
    """Map normalized values in [0, 1] to real values using a linear or log scale."""
    if scale == "log":
        if minimum <= 0 or maximum <= 0:
            raise ValueError("Logarithmic axes require positive bounds.")
        return np.exp(np.log(minimum) + values * (np.log(maximum) - np.log(minimum)))
    return minimum + values * (maximum - minimum)


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


def _render_vbar_mask(fig_size: tuple[float, float], dpi: int, x_pos: float, y_range: tuple[float, float], 
                      width: float, style: dict[str, Any]) -> np.ndarray:
    """Render a vertical bar mask."""
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi, facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(0, 1)
    ax.set_ylim(*y_range)
    ax.axvline(x=x_pos, ymin=0, ymax=1, color="white", linewidth=width, linestyle=style.get("linestyle", "-"))
    ax.axis("off")
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return np.max(buffer[:, :, :3], axis=2) > 200


def _render_hbar_mask(
    fig_size: tuple[float, float],
    dpi: int,
    y_pos: float,
    x_range: tuple[float, float],
    height: float,
    style: dict[str, Any],
    x_scale: str = "linear",
) -> np.ndarray:
    """Render a horizontal bar mask."""
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi, facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(*x_range)
    ax.set_xscale(x_scale)
    ax.set_ylim(0, 1)
    ax.axhline(y=y_pos, xmin=0, xmax=1, color="white", linewidth=height, linestyle=style.get("linestyle", "-"))
    ax.axis("off")
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return np.max(buffer[:, :, :3], axis=2) > 200


def _render_arrow_mask(fig_size: tuple[float, float], dpi: int, start: tuple[float, float], 
                       end: tuple[float, float], style: dict[str, Any]) -> np.ndarray:
    """Render an arrow annotation mask."""
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi, facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="white", 
                                                           lw=style.get("linewidth", 2.0)))
    ax.axis("off")
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return np.max(buffer[:, :, :3], axis=2) > 200


def _render_error_bar_mask(fig_size: tuple[float, float], dpi: int, x_pos: float, y_pos: float,
                           y_err: float, style: dict[str, Any]) -> np.ndarray:
    """Render an error bar mask."""
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi, facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    cap_width = style.get("cap_width", 0.03)
    ax.errorbar(x_pos, y_pos, yerr=y_err, fmt='none', ecolor="white", 
                elinewidth=style.get("linewidth", 1.5), capsize=cap_width * fig_size[0] * dpi)
    ax.axis("off")
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return np.max(buffer[:, :, :3], axis=2) > 200


def _render_curve_mask(
    fig_size: tuple[float, float],
    dpi: int,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    style: dict[str, Any],
    x_scale: str = "linear",
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi, facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(*x_range)
    ax.set_xscale(x_scale)
    ax.set_ylim(*y_range)
    ax.plot(x_values, y_values, color="white", linewidth=style["linewidth"], linestyle=style["linestyle"])
    ax.axis("off")
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return np.max(buffer[:, :, :3], axis=2) > 200


def _mask_to_yolo_polygon(mask: np.ndarray) -> list[float]:
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return []
    contour = max(contours, key=len)
    polygon: list[float] = []
    height, width = mask.shape
    step = max(1, len(contour) // MAX_POLYGON_POINTS)
    for y_coord, x_coord in contour[::step]:
        polygon.extend([float(np.clip(x_coord / width, 0.0, 1.0)), float(np.clip(y_coord / height, 0.0, 1.0))])
    return polygon if len(polygon) >= 6 else []


def _apply_degradation_filters(image_path: Path, rng: np.random.Generator) -> None:
    """Apply degradation filters to simulate old/scanned article quality.
    
    This function applies various image degradations to improve model resilience
    when processing real-world images from old scientific articles, including:
    - JPEG compression artifacts
    - Gaussian noise (film grain simulation)
    - Blur (low resolution scanning)
    - Contrast reduction (faded ink)
    - Binarization (black and white scans)
    - Salt-and-pepper noise (dust/scratches)
    
    Args:
        image_path: Path to the image file to degrade (modified in-place)
        rng: NumPy random generator for reproducible randomness
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        LOGGER.warning("Could not load image for degradation: %s", image_path)
        return
    
    # Randomly select degradation types (can apply multiple)
    apply_jpeg = rng.random() > 0.3  # 70% chance
    apply_noise = rng.random() > 0.4  # 60% chance
    apply_blur = rng.random() > 0.5  # 50% chance
    apply_contrast = rng.random() > 0.6  # 40% chance
    apply_bw = rng.random() > 0.85  # 15% chance (less common)
    apply_salt_pepper = rng.random() > 0.7  # 30% chance
    
    degraded = image.copy()
    
    # JPEG compression artifacts (re-encode with low quality)
    if apply_jpeg:
        quality = int(rng.uniform(15, 75))  # Low to medium quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", degraded, encode_param)
        degraded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    
    # Gaussian noise (simulates film grain or scanner noise)
    if apply_noise:
        noise_std = rng.uniform(5, 25)
        noise = rng.normal(0, noise_std, degraded.shape).astype(np.int16)
        degraded = np.clip(degraded.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Gaussian blur (simulates low-resolution scanning)
    if apply_blur:
        kernel_size = int(rng.choice([3, 5, 7]))
        degraded = cv2.GaussianBlur(degraded, (kernel_size, kernel_size), 0)
    
    # Contrast reduction (simulates faded ink or poor photocopying)
    if apply_contrast:
        alpha = rng.uniform(0.6, 0.9)  # Reduce contrast
        beta = rng.uniform(-10, 10)  # Slight brightness shift
        degraded = cv2.convertScaleAbs(degraded, alpha=alpha, beta=beta)
    
    # Salt-and-pepper noise (simulates dust, scratches, or printing artifacts)
    if apply_salt_pepper:
        salt_pepper_prob = rng.uniform(0.005, 0.02)
        salt_mask = rng.random(degraded.shape[:2]) < salt_pepper_prob
        pepper_mask = rng.random(degraded.shape[:2]) < salt_pepper_prob
        for c in range(3):
            degraded[salt_mask, c] = 255
            degraded[pepper_mask, c] = 0
    
    # Convert to black and white (binary or grayscale)
    if apply_bw:
        if rng.random() > 0.5:
            # Pure binary (thresholded)
            gray = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
            _, degraded_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            degraded = cv2.cvtColor(degraded_binary, cv2.COLOR_GRAY2BGR)
        else:
            # Grayscale only
            gray = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
            degraded = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Save degraded image
    cv2.imwrite(str(image_path), degraded)
    LOGGER.debug("Applied degradations to %s: jpeg=%s, noise=%s, blur=%s, contrast=%s, bw=%s, salt_pepper=%s",
                 image_path.name, apply_jpeg, apply_noise, apply_blur, apply_contrast, apply_bw, apply_salt_pepper)


def _write_synthetic_example(index: int, output_dir: Path, rng: np.random.Generator, image_format: str, 
                             plot_type: str = "general") -> None:
    """Generate a synthetic plot with support for bandstructures and complex annotations."""
    fig_size = (6.0, 4.2)
    dpi = DEFAULT_DPI
    image_name = f"plot_{index:04d}.{image_format}"
    image_path = output_dir / "images" / image_name
    label_path = output_dir / "labels" / f"plot_{index:04d}.txt"
    metadata_path = output_dir / "images" / f"plot_{index:04d}.metadata.json"
    ground_truth_path = output_dir / "ground_truth" / f"plot_{index:04d}.csv"

    use_log_x = bool(rng.random() < LOG_X_PROBABILITY)
    x_min = LOG_X_MIN if use_log_x else 0.0
    x_range = (x_min, float(rng.uniform(6.0, 12.0)))
    x_values = np.geomspace(*x_range, 480) if use_log_x else np.linspace(*x_range, 480)
    
    colors = ["tab:red", "tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:cyan"]
    linestyles = ["-", "--", "-.", ":"]
    # Weighted sampling favors thinner strokes for better faint-curve recall.
    linewidths = CURVE_LINEWIDTHS

    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    if use_log_x:
        ax.set_xscale("log")
    if rng.random() < GRID_ENABLED_PROBABILITY:
        ax.grid(True, linestyle="--" if rng.random() > 0.5 else ":", alpha=GRID_ALPHA)
    else:
        ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linestyle(str(rng.choice(["-", "--", ":"])))
    ax.set_xlim(*x_range)
    
    ground_truth_frames: list[pd.DataFrame] = []
    label_lines: list[str] = []
    curve_descriptors: list[dict[str, Any]] = []
    annotation_descriptors: list[dict[str, Any]] = []
    
    all_y = []

    if plot_type == "bandstructure":
        # Generate bandstructure-like plots with multiple bands
        n_bands = int(rng.integers(4, 10))
        raw_curves = _generate_bandstructure_curves(x_values, rng, n_bands)
        y_range = (-2.5, 2.5)  # Typical bandstructure energy range
        ax.set_ylim(*y_range)
        ax.set_xlabel("k-path")
        ax.set_ylabel("Energy (eV)")
        ax.set_title("Band Structure")
        
        # Add Fermi level line
        if rng.random() > 0.5:
            fermi_y = rng.uniform(-0.5, 0.5)
            ax.axhline(y=fermi_y, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
            annotation_descriptors.append({
                "type": "hbar",
                "class_id": 2,  # hbar class
                "y_pos": fermi_y,
                "description": "fermi_level"
            })
    else:
        # General plots
        # Oversample dense curve scenes to improve recall under overlap.
        if rng.random() < DENSE_CURVE_PROBABILITY:
            curve_count = int(rng.integers(DENSE_CURVE_COUNT_RANGE[0], DENSE_CURVE_COUNT_RANGE[1] + 1))
        else:
            curve_count = int(rng.integers(BASE_CURVE_COUNT_RANGE[0], BASE_CURVE_COUNT_RANGE[1] + 1))
        raw_curves = [_random_curve(x_values, rng) for _ in range(curve_count)]
        all_y = np.concatenate([curve for curve, _ in raw_curves])
        y_margin = max(0.5, float(np.ptp(all_y) * 0.1))
        y_range = (float(all_y.min() - y_margin), float(all_y.max() + y_margin))
        ax.set_ylim(*y_range)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Synthetic Plot")

    x_axis_scale = "log" if use_log_x else "linear"

    def _x_norm_to_data(norm_x: float) -> float:
        norm = np.array([float(np.clip(norm_x, 0.0, 1.0))], dtype=float)
        return float(_norm_to_scale(norm, x_range[0], x_range[1], x_axis_scale)[0])

    def _y_norm_to_data(norm_y: float) -> float:
        return float(y_range[0] + float(np.clip(norm_y, 0.0, 1.0)) * (y_range[1] - y_range[0]))

    for curve_index, (y_values, curve_type) in enumerate(raw_curves):
        style = {
            "color": colors[curve_index % len(colors)],
            "linestyle": linestyles[curve_index % len(linestyles)],
            "linewidth": float(rng.choice(linewidths, p=CURVE_LINEWIDTH_PROBABILITIES)),
        }
        ax.plot(x_values, y_values, **style)
        dataset_id = f"dataset_{curve_index}"
        curve_descriptors.append({"dataset_id": dataset_id, "curve_type": curve_type, **style})
        ground_truth_frames.append(pd.DataFrame({"dataset_id": dataset_id, "x_real": x_values, "y_real": y_values}))
        mask = _render_curve_mask(
            fig_size,
            dpi,
            x_values,
            y_values,
            x_range,
            y_range,
            style,
            x_scale="log" if use_log_x else "linear",
        )
        polygon = _mask_to_yolo_polygon(mask)
        if polygon:
            label_lines.append("0 " + " ".join(f"{value:.6f}" for value in polygon))

    # Add complex annotations (vbars, hbars, arrows, error bars)
    # Keep vbar/hbar/error_bar present in all samples; arrows remain optional.
    n_vbars = int(rng.integers(VBAR_COUNT_RANGE[0], VBAR_COUNT_RANGE[1] + 1))
    for vbar_idx in range(n_vbars):
        x_pos = rng.uniform(0.1, 0.9)
        vbar_width = rng.uniform(1.0, 3.0)
        style = {"linewidth": vbar_width, "linestyle": "-"}
        ax.axvline(
            x=_x_norm_to_data(x_pos),
            ymin=0,
            ymax=1,
            color="black",
            linewidth=vbar_width,
            linestyle=style["linestyle"],
        )
        mask = _render_vbar_mask(fig_size, dpi, x_pos, y_range, vbar_width, style)
        polygon = _mask_to_yolo_polygon(mask)
        if polygon:
            class_id = 1  # vbar class
            label_lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in polygon))
            annotation_descriptors.append({
                "type": "vbar",
                "class_id": class_id,
                "x_pos": x_pos,
                "description": f"high_symmetry_point_{vbar_idx}"
            })

    n_hbars = int(rng.integers(HBAR_COUNT_RANGE[0], HBAR_COUNT_RANGE[1] + 1))
    for hbar_idx in range(n_hbars):
        y_pos_norm = rng.uniform(0.1, 0.9)
        y_pos = y_range[0] + y_pos_norm * (y_range[1] - y_range[0])
        hbar_height = rng.uniform(1.0, 2.5)
        style = {"linewidth": hbar_height, "linestyle": "--"}
        ax.axhline(y=y_pos, xmin=0, xmax=1, color="black", linewidth=hbar_height, linestyle=style["linestyle"])
        mask = _render_hbar_mask(fig_size, dpi, y_pos_norm, x_range, hbar_height, style, x_scale="log" if use_log_x else "linear")
        polygon = _mask_to_yolo_polygon(mask)
        if polygon:
            class_id = 2  # hbar class
            label_lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in polygon))
            annotation_descriptors.append({
                "type": "hbar",
                "class_id": class_id,
                "y_pos": y_pos,
                "description": f"reference_line_{hbar_idx}"
            })

    # Arrows are optional in real plots, so allow zero while preserving exposure.
    n_arrows = int(rng.integers(ARROW_COUNT_RANGE[0], ARROW_COUNT_RANGE[1] + 1))
    for arrow_idx in range(n_arrows):
        start = (rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8))
        end = (rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8))
        style = {"linewidth": rng.uniform(1.5, 3.0)}
        ax.annotate(
            "",
            xy=(_x_norm_to_data(end[0]), _y_norm_to_data(end[1])),
            xytext=(_x_norm_to_data(start[0]), _y_norm_to_data(start[1])),
            arrowprops={"arrowstyle": "->", "color": "black", "lw": style["linewidth"]},
        )
        mask = _render_arrow_mask(fig_size, dpi, start, end, style)
        polygon = _mask_to_yolo_polygon(mask)
        if polygon:
            class_id = 3  # arrow class
            label_lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in polygon))
            annotation_descriptors.append({
                "type": "arrow",
                "class_id": class_id,
                "start": start,
                "end": end,
                "description": f"annotation_arrow_{arrow_idx}"
            })

    n_error_bars = int(rng.integers(ERROR_BAR_COUNT_RANGE[0], ERROR_BAR_COUNT_RANGE[1] + 1))
    for eb_idx in range(n_error_bars):
        x_pos = rng.uniform(0.1, 0.9)
        y_pos = rng.uniform(0.2, 0.8)
        y_err = rng.uniform(0.05, 0.2)
        style = {"linewidth": rng.uniform(1.0, 2.0), "cap_width": 0.02}
        y_err_data = float(y_err * (y_range[1] - y_range[0]))
        ax.errorbar(
            _x_norm_to_data(x_pos),
            _y_norm_to_data(y_pos),
            yerr=y_err_data,
            fmt="none",
            ecolor="black",
            elinewidth=style["linewidth"],
            capsize=style["cap_width"] * fig_size[0] * dpi,
        )
        mask = _render_error_bar_mask(fig_size, dpi, x_pos, y_pos, y_err, style)
        polygon = _mask_to_yolo_polygon(mask)
        if polygon:
            class_id = 4  # error_bar class
            label_lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in polygon))
            annotation_descriptors.append({
                "type": "error_bar",
                "class_id": class_id,
                "x_pos": x_pos,
                "y_pos": y_pos,
                "y_err": y_err,
                "description": f"error_bar_{eb_idx}"
            })

    fig.tight_layout()
    fig.canvas.draw()
    axis_bbox = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    width_px, height_px = fig.canvas.get_width_height()
    plot_box = {
        "left": int(axis_bbox.x0),
        "top": int(height_px - axis_bbox.y1),
        "right": int(axis_bbox.x1),
        "bottom": int(height_px - axis_bbox.y0),
    }
    
    fig.savefig(image_path, dpi=dpi, format=image_format)
    plt.close(fig)
    
    # Apply degradation filters to simulate old/scanned article quality
    _apply_degradation_filters(image_path, rng)

    ground_truth = pd.concat(ground_truth_frames, ignore_index=True)
    ground_truth.to_csv(ground_truth_path, index=False)
    label_path.write_text("\n".join(label_lines))
    metadata = {
        "image": str(image_path),
        "x_range": list(x_range),
        "y_range": list(y_range),
        "x_scale": "log" if use_log_x else "linear",
        "y_scale": "linear",
        "invert_y": False,
        "plot_box": plot_box,
        "plot_type": plot_type,
        "curves": curve_descriptors,
        "annotations": annotation_descriptors,
        "ground_truth_csv": str(ground_truth_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))


SampleGenerationTask = tuple[int, Path, np.random.SeedSequence, str, str]


def _generate_one_sample(args: SampleGenerationTask) -> None:
    """Worker function for parallel synthetic sample generation.

    Accepts a tuple so it can be passed through :func:`ProcessPoolExecutor.map`
    without requiring Python 3.12+ keyword-argument pickling.
    """
    index, output_dir, child_seed, image_format, plot_type = args
    rng = np.random.default_rng(child_seed)
    _write_synthetic_example(index, output_dir, rng, image_format, plot_type)


def generate_synthetic_dataset(
    output_dir: Path,
    count: int,
    seed: int,
    image_format: str,
    plot_type: str = "mixed",
    workers: int | None = None,
) -> None:
    """Generate a synthetic plot dataset with YOLO segmentation labels.

    Samples are generated in parallel using :class:`ProcessPoolExecutor` by
    default (one process per CPU core).  Each sample receives an independent
    :class:`numpy.random.SeedSequence` child so results are fully deterministic
    and identical regardless of the number of workers used.

    Args:
        output_dir: Output directory for the dataset.
        count: Number of images to generate.
        seed: Random seed for reproducibility.
        image_format: Image format (``"png"`` or ``"jpg"``).
        plot_type: Type of plots to generate – ``"general"``,
            ``"bandstructure"``, or ``"mixed"``.
        workers: Number of worker processes.  ``None`` (default) uses
            ``min(os.cpu_count(), count, 8)``. Pass ``1`` for strictly
            sequential execution (useful for debugging). The cap keeps
            process and memory overhead reasonable on high-core systems.
    """
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("images", "labels", "ground_truth"):
        (output_dir / subdir).mkdir(exist_ok=True)

    # Derive independent child seeds from the master SeedSequence.
    # children[0] drives plot-type assignment; children[1:] drive per-sample rngs.
    ss = np.random.SeedSequence(seed)
    all_children = ss.spawn(count + 1)
    type_rng = np.random.default_rng(all_children[0])
    sample_seeds = all_children[1:]

    plot_types: list[str] = [
        type_rng.choice(["general", "bandstructure"]) if plot_type == "mixed" else plot_type
        for _ in range(count)
    ]

    tasks: list[SampleGenerationTask] = [
        (i, output_dir, sample_seeds[i], image_format, plot_types[i])
        for i in range(count)
    ]

    cpu_count = os.cpu_count() or 1
    n_workers = workers if workers is not None else min(cpu_count, count, DEFAULT_GENERATE_WORKERS_CAP)
    if n_workers <= 1:
        for task in tasks:
            _generate_one_sample(task)
    else:
        mp_ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as executor:
            # Consume the iterator to propagate any worker exceptions.
            for _ in executor.map(_generate_one_sample, tasks):
                pass
    
    # Multi-class segmentation labels must stay contiguous from 0..nc-1
    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: images",
                "val: images",
                "test: images",
                "nc: 5",
                "names:",
                "  0: curve",
                "  1: vbar",
                "  2: hbar", 
                "  3: arrow",
                "  4: error_bar",
            ]
        )
    )


def run_training(
    dataset_dir: Path,
    output_dir: Path,
    epochs: int,
    imgsz: int,
    weights: str,
    batch: int,
    execute: bool,
    hyp_yaml: Path | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Create or execute a YOLO segmentation training job."""
    dataset_yaml = (dataset_dir / "dataset.yaml").resolve()
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_yaml}")
    training_plan = {
        "dataset": str(dataset_yaml),
        "weights": weights,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": str(output_dir),
        "task": "segment",
    }
    hyp_path = hyp_yaml.resolve() if hyp_yaml is not None else None
    if hyp_path is not None:
        if not hyp_path.exists():
            raise FileNotFoundError(f"Hyperparameter config not found: {hyp_path}")
        training_plan["cfg"] = str(hyp_path)
    if workers is not None:
        training_plan["workers"] = workers
    if execute:
        try:
            import torch as _torch
        except ImportError:  # pragma: no cover - depends on optional dependency setup
            import re as _re
            cuda_path = os.environ.get("CUDA_PATH", "")
            rocm_path = os.environ.get("ROCM_PATH", "")
            if rocm_path:
                index_url = "https://download.pytorch.org/whl/rocm6.2"
            else:
                _m = _re.search(r"cuda[_-]?(\d+)[._-]\d+", cuda_path, _re.IGNORECASE)
                cuda_major = int(_m.group(1)) if _m else 0
                if cuda_major == 11:
                    index_url = "https://download.pytorch.org/whl/cu118"
                elif cuda_major >= 12:
                    index_url = "https://download.pytorch.org/whl/cu124"
                else:
                    index_url = "https://download.pytorch.org/whl/cpu"
            raise ImportError(
                "Training requires torch and torchvision, which are not included in the Nix "
                "shell by default. Install them for your accelerator with:\n"
                f"  pip install torch torchvision --index-url {index_url}\n"
                "Then rerun the command. See the README for all accelerator options."
            )
        if workers is not None:
            _torch.set_num_threads(workers)
            try:
                _torch.set_num_interop_threads(workers)
            except RuntimeError:
                LOGGER.debug(
                    "Unable to set torch inter-op thread count to %s; continuing with existing setting.",
                    workers,
                )
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on optional dependency setup
            raise ImportError(
                "Training requires ultralytics. Install digitizer with the 'ai' extra: "
                "`uv pip install -e \".[ai]\"`"
            ) from exc

        model = YOLO(weights)
        train_kwargs: dict[str, Any] = {
            "data": str(dataset_yaml),
            "task": "segment",
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "project": str(output_dir),
            "name": "synthetic_plot_digitizer",
        }
        if workers is not None:
            train_kwargs["workers"] = workers
        if hyp_path is not None:
            train_kwargs["cfg"] = str(hyp_path)
        training_plan["result"] = model.train(**train_kwargs).save_dir.as_posix()
    return training_plan

