"""Annotation I/O helpers: polygon geometry and YOLO label writing.

Converts human-supplied annotation descriptors (curves, bars, arrows, frame and
axis elements) into normalised YOLO segmentation label lines and writes
training sample side-cars (image copy + label .txt + metadata .json).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

CLASS_MAPPING: dict[str, int] = {
    "curve": 0,
    "vbar": 1,
    "hbar": 2,
    "arrow": 3,
    "error_bar": 4,
    "plot_area": 5,
    "x_axis": 6,
    "y_axis": 7,
    "x_anchor": 8,
    "y_anchor": 9,
}


def _normalize_pixel_coords(
    points: list[tuple[float, float]], image_width: int, image_height: int
) -> list[float]:
    """Convert absolute pixel (x, y) pairs to flat normalized YOLO coords."""
    out: list[float] = []
    for x, y in points:
        out.append(float(np.clip(x / image_width, 0.0, 1.0)))
        out.append(float(np.clip(y / image_height, 0.0, 1.0)))
    return out


def polygon_from_vbar(
    x: float, y_top: float, y_bottom: float, line_width: float,
    image_width: int, image_height: int,
) -> list[float]:
    """Normalized YOLO polygon for a vertical bar annotation."""
    half_w = line_width / 2.0
    points = [
        (x - half_w, y_top), (x + half_w, y_top),
        (x + half_w, y_bottom), (x - half_w, y_bottom),
    ]
    return _normalize_pixel_coords(points, image_width, image_height)


def polygon_from_hbar(
    y: float, x_left: float, x_right: float, line_width: float,
    image_width: int, image_height: int,
) -> list[float]:
    """Normalized YOLO polygon for a horizontal bar annotation."""
    half_h = line_width / 2.0
    points = [
        (x_left, y - half_h), (x_right, y - half_h),
        (x_right, y + half_h), (x_left, y + half_h),
    ]
    return _normalize_pixel_coords(points, image_width, image_height)


def polygon_from_arrow(
    start: tuple[float, float], end: tuple[float, float],
    line_width: float, image_width: int, image_height: int,
) -> list[float]:
    """Normalized YOLO polygon for an arrow annotation (line envelope)."""
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = float(np.hypot(dx, dy))
    if length < 1e-6:
        return []
    nx, ny = -dy / length, dx / length
    half_w = line_width / 2.0
    points = [
        (start[0] + nx * half_w, start[1] + ny * half_w),
        (end[0] + nx * half_w, end[1] + ny * half_w),
        (end[0] - nx * half_w, end[1] - ny * half_w),
        (start[0] - nx * half_w, start[1] - ny * half_w),
    ]
    return _normalize_pixel_coords(points, image_width, image_height)


def polygon_from_curve(
    points: list[tuple[float, float]], line_width: float,
    image_width: int, image_height: int,
) -> list[float]:
    """Normalized YOLO polygon for a polyline curve annotation (stroke envelope)."""
    if len(points) < 2:
        return []
    half_w = line_width / 2.0
    upper: list[tuple[float, float]] = []
    lower: list[tuple[float, float]] = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = float(np.hypot(dx, dy))
        if length < 1e-6:
            continue
        nx, ny = -dy / length, dx / length
        upper.extend([(p1[0] + nx * half_w, p1[1] + ny * half_w),
                       (p2[0] + nx * half_w, p2[1] + ny * half_w)])
        lower.extend([(p1[0] - nx * half_w, p1[1] - ny * half_w),
                       (p2[0] - nx * half_w, p2[1] - ny * half_w)])
    if not upper:
        return []
    return _normalize_pixel_coords(upper + list(reversed(lower)), image_width, image_height)


def polygon_from_error_bar(
    x: float, y_top: float, y_bottom: float,
    cap_width: float, line_width: float,
    image_width: int, image_height: int,
) -> list[float]:
    """Normalized YOLO polygon for an error bar annotation."""
    half_cap = cap_width / 2.0
    half_line = line_width / 2.0
    points = [
        (x - half_cap, y_top), (x + half_cap, y_top),
        (x + half_line, y_top), (x + half_line, y_bottom),
        (x + half_cap, y_bottom), (x - half_cap, y_bottom),
        (x - half_line, y_bottom), (x - half_line, y_top),
    ]
    return _normalize_pixel_coords(points, image_width, image_height)


def polygon_from_rectangle(
    top_left: tuple[float, float],
    bottom_right: tuple[float, float],
    image_width: int,
    image_height: int,
) -> list[float]:
    """Normalized YOLO polygon for a rectangle annotation."""
    x1, y1 = top_left
    x2, y2 = bottom_right
    left, right = sorted((float(x1), float(x2)))
    top, bottom = sorted((float(y1), float(y2)))
    points = [
        (left, top),
        (right, top),
        (right, bottom),
        (left, bottom),
    ]
    return _normalize_pixel_coords(points, image_width, image_height)


def polygon_from_line(
    start: tuple[float, float],
    end: tuple[float, float],
    line_width: float,
    image_width: int,
    image_height: int,
) -> list[float]:
    """Normalized YOLO polygon for a generic line (envelope rectangle)."""
    return polygon_from_arrow(start, end, line_width, image_width, image_height)


def polygon_from_point(
    center: tuple[float, float],
    point_size: float,
    image_width: int,
    image_height: int,
) -> list[float]:
    """Normalized YOLO polygon for a point-like annotation as a square box."""
    cx, cy = float(center[0]), float(center[1])
    half = max(1.0, point_size / 2.0)
    return polygon_from_rectangle((cx - half, cy - half), (cx + half, cy + half), image_width, image_height)


def scale_annotation_points(
    ann: dict[str, Any], sx: float, sy: float,
) -> dict[str, Any]:
    """Return a copy of *ann* with all points scaled by (sx, sy).

    Use this when the image is saved at a different resolution than the one
    the user annotated so that absolute pixel coords stay correct.
    """
    scaled = dict(ann)
    scaled["points"] = [(float(x) * sx, float(y) * sy) for x, y in ann.get("points", [])]
    if "line_width" in ann:
        scaled["line_width"] = float(ann["line_width"]) * ((sx + sy) / 2.0)
    if "cap_width" in ann:
        scaled["cap_width"] = float(ann["cap_width"]) * sx
    if "point_size" in ann:
        scaled["point_size"] = float(ann["point_size"]) * ((sx + sy) / 2.0)
    return scaled


def annotation_to_yolo_line(
    ann: dict[str, Any],
    image_width: int,
    image_height: int,
    default_line_width: float = 3.0,
) -> str | None:
    """Convert one annotation descriptor dict to a YOLO segmentation label line.

    Returns ``None`` when the annotation cannot produce a valid polygon.
    """
    ann_type = ann.get("type", "")
    if ann_type not in CLASS_MAPPING:
        return None
    class_id = CLASS_MAPPING[ann_type]
    pts: list[tuple[float, float]] = [(float(p[0]), float(p[1])) for p in ann.get("points", [])]
    lw = float(ann.get("line_width", default_line_width))
    cap_w = float(ann.get("cap_width", lw * 5.0))
    point_size = float(ann.get("point_size", max(3.0, lw * 1.75)))

    if ann_type == "vbar" and pts:
        x = pts[0][0]
        polygon = polygon_from_vbar(x, 0.0, float(image_height), lw, image_width, image_height)
    elif ann_type == "hbar" and pts:
        y = pts[0][1]
        polygon = polygon_from_hbar(y, 0.0, float(image_width), lw, image_width, image_height)
    elif ann_type == "arrow" and len(pts) >= 2:
        polygon = polygon_from_arrow(pts[0], pts[1], lw, image_width, image_height)
    elif ann_type == "curve" and len(pts) >= 2:
        polygon = polygon_from_curve(pts, lw, image_width, image_height)
    elif ann_type == "error_bar" and len(pts) >= 2:
        x = pts[0][0]
        y_top = min(pts[0][1], pts[1][1])
        y_bottom = max(pts[0][1], pts[1][1])
        polygon = polygon_from_error_bar(x, y_top, y_bottom, cap_w, lw, image_width, image_height)
    elif ann_type == "plot_area" and len(pts) >= 2:
        polygon = polygon_from_rectangle(pts[0], pts[1], image_width, image_height)
    elif ann_type == "x_axis" and len(pts) >= 2:
        polygon = polygon_from_line(pts[0], pts[1], lw, image_width, image_height)
    elif ann_type == "y_axis" and len(pts) >= 2:
        polygon = polygon_from_line(pts[0], pts[1], lw, image_width, image_height)
    elif ann_type == "x_anchor" and pts:
        polygon = polygon_from_point(pts[0], point_size, image_width, image_height)
    elif ann_type == "y_anchor" and pts:
        polygon = polygon_from_point(pts[0], point_size, image_width, image_height)
    else:
        return None

    if len(polygon) < 6:
        return None
    return f"{class_id} " + " ".join(f"{v:.6f}" for v in polygon)


def save_training_sample(
    image_path: Path,
    annotations: list[dict[str, Any]],
    output_dir: Path,
    default_line_width: float = 3.0,
    resize_to: tuple[int, int] | None = None,
) -> dict[str, str]:
    """Copy *image_path* plus YOLO labels and metadata sidecar into *output_dir*.

    The output layout follows the same convention used by ``generate_synthetic_dataset``::

        output_dir/
            images/<stem>.<ext>
            images/<stem>.metadata.json
            labels/<stem>.txt

    Returns a dict with ``image_path``, ``label_path``, and ``metadata_path`` keys.
    """
    import cv2
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    image_height, image_width = image.shape[:2]

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    dest_image = images_dir / image_path.name
    if resize_to is None:
        dest_width, dest_height = image_width, image_height
        scaled_annotations = [dict(ann) for ann in annotations]
        shutil.copy2(image_path, dest_image)
    else:
        dest_width = int(resize_to[0])
        dest_height = int(resize_to[1])
        if dest_width < 1 or dest_height < 1:
            raise ValueError("resize_to dimensions must be >= 1")
        sx = dest_width / float(image_width)
        sy = dest_height / float(image_height)
        scaled_annotations = [scale_annotation_points(ann, sx, sy) for ann in annotations]
        import cv2
        resized = cv2.resize(image, (dest_width, dest_height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dest_image), resized)

    label_lines: list[str] = []
    for ann in scaled_annotations:
        line = annotation_to_yolo_line(ann, dest_width, dest_height, default_line_width)
        if line:
            label_lines.append(line)

    label_path = labels_dir / f"{image_path.stem}.txt"
    label_path.write_text("\n".join(label_lines))

    metadata_path = images_dir / f"{image_path.stem}.metadata.json"
    metadata_path.write_text(json.dumps({
        "image": str(dest_image),
        "image_width": dest_width,
        "image_height": dest_height,
        "source_image": str(image_path),
        "source_image_width": image_width,
        "source_image_height": image_height,
        "annotations": scaled_annotations,
        "class_mapping": CLASS_MAPPING,
        "label_count": len(label_lines),
    }, indent=2))

    return {
        "image_path": str(dest_image),
        "label_path": str(label_path),
        "metadata_path": str(metadata_path),
    }


def load_training_sample_annotations(
    image_path: Path,
    output_dir: Path,
    target_size: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Load previously saved annotations for *image_path* from *output_dir*.

    Returns an empty list when no metadata sidecar exists.
    """
    metadata_path = output_dir / "images" / f"{image_path.stem}.metadata.json"
    if not metadata_path.exists():
        return []

    metadata = json.loads(metadata_path.read_text())
    annotations = [dict(ann) for ann in metadata.get("annotations", []) if isinstance(ann, dict)]
    if not annotations or target_size is None:
        return annotations

    stored_width = int(metadata.get("image_width", 0))
    stored_height = int(metadata.get("image_height", 0))
    target_width, target_height = int(target_size[0]), int(target_size[1])
    if stored_width < 1 or stored_height < 1:
        return annotations
    if stored_width == target_width and stored_height == target_height:
        return annotations

    sx = target_width / float(stored_width)
    sy = target_height / float(stored_height)
    return [scale_annotation_points(ann, sx, sy) for ann in annotations]
