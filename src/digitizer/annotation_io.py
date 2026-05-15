"""Annotation I/O helpers: polygon geometry and YOLO label writing.

Converts human-supplied annotation descriptors (curves, bars, arrows, frame and
axis elements) into normalised YOLO segmentation label lines and writes
training sample side-cars (image copy + label .txt + metadata .json).
"""

from __future__ import annotations

import json
import logging
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

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
    "x_tick_label": 10,
    "y_tick_label": 11,
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
    elif ann_type in ("x_tick_label", "y_tick_label") and len(pts) >= 2:
        # Tick labels are stored as [(x0,y0),(x1,y1)] bounding-box corners in
        # image coordinates (top-left origin, y increases downward).
        polygon = polygon_from_rectangle(pts[0], pts[1], image_width, image_height)
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
    csv_path: Path | None = None,
) -> dict[str, str]:
    """Copy *image_path* plus YOLO labels, annotations, and metadata into *output_dir*.

    The output layout follows the same convention used by ``generate_synthetic_dataset``::

        output_dir/
            images/<stem>.<ext>              image copy
            images/<stem>.metadata.json      image-level properties only
            labels/<stem>.txt                YOLO-format derived labels
            annotations/<stem>.json          raw annotation points (source of truth)
            csv/<stem>.csv                   associated CSV when *csv_path* is given

    Annotation points are stored in ``annotations/`` rather than embedded in
    the metadata JSON so that the two concerns (image properties vs. annotation
    content) remain cleanly separated.  :func:`load_training_sample_annotations`
    reads from the annotations file first, with a metadata-embedded fallback
    for backward compatibility.

    Returns a dict with ``image_path``, ``label_path``, ``metadata_path``,
    ``annotations_path``, and optionally ``csv_path`` keys.
    """
    import cv2
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    image_height, image_width = image.shape[:2]

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    annotations_dir = output_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

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

    # Raw annotation points go to annotations/ (separate from image metadata).
    annotations_path = annotations_dir / f"{image_path.stem}.json"
    annotations_path.write_text(json.dumps({
        "image": str(dest_image),
        "image_width": dest_width,
        "image_height": dest_height,
        "annotations": scaled_annotations,
    }, indent=2))

    # Copy associated CSV (ground truth / digitized data) when provided.
    dest_csv: Path | None = None
    if csv_path is not None and csv_path.exists():
        csv_dir = output_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        dest_csv = csv_dir / csv_path.name
        shutil.copy2(csv_path, dest_csv)

    # Metadata carries image-level properties and a reference to the annotations file.
    metadata_path = images_dir / f"{image_path.stem}.metadata.json"
    metadata: dict[str, Any] = {
        "image": str(dest_image),
        "image_width": dest_width,
        "image_height": dest_height,
        "source_image": str(image_path),
        "source_image_width": image_width,
        "source_image_height": image_height,
        "annotations_path": str(annotations_path),
        "class_mapping": CLASS_MAPPING,
        "label_count": len(label_lines),
    }
    if dest_csv is not None:
        metadata["csv_path"] = str(dest_csv)
    metadata_path.write_text(json.dumps(metadata, indent=2))

    result: dict[str, str] = {
        "image_path": str(dest_image),
        "label_path": str(label_path),
        "metadata_path": str(metadata_path),
        "annotations_path": str(annotations_path),
    }
    if dest_csv is not None:
        result["csv_path"] = str(dest_csv)
    return result


def load_training_sample_annotations(
    image_path: Path,
    output_dir: Path,
    target_size: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Load previously saved annotations for *image_path* from *output_dir*.

    Search order:
    1. ``output_dir/annotations/<stem>.json`` – primary location written by
       :func:`save_training_sample` and ``generate_synthetic_dataset``.
    2. Sibling ``annotations/<stem>.json`` one level above *image_path*
       (covers images that live inside an ``images/`` subdirectory of a
       dataset, e.g. ``synthetic-data/annotations/plot_0000.json`` when
       annotating ``synthetic-data/images/plot_0000.png``).
    3. ``output_dir/images/<stem>.metadata.json`` – backward-compat fallback
       for datasets written before the annotations/ split.  Reads the
       ``annotations`` key if present (manual annotation format with
       ``image_width``/``image_height``), or the ``frame_annotations`` key
       for synthetic metadata.
    4. ``image_path.parent/<stem>.metadata.json`` – last-resort fallback for
       metadata sidecars stored alongside the image.

    Annotations without a ``points`` key are silently skipped.

    Returns an empty list when nothing is found.
    """
    stem = image_path.stem

    # --- Attempt 1 & 2: dedicated annotations/ file --------------------------
    for ann_path in (
        output_dir / "annotations" / f"{stem}.json",
        image_path.parent.parent / "annotations" / f"{stem}.json",
    ):
        if not ann_path.exists():
            continue
        try:
            data = json.loads(ann_path.read_text())
        except Exception:
            LOGGER.warning("Could not parse annotations file %s", ann_path)
            continue
        raw = data.get("annotations", [])
        annotations = _filter_annotations_with_points(raw, ann_path)
        if not annotations:
            continue
        stored_w = int(data.get("image_width", 0))
        stored_h = int(data.get("image_height", 0))
        return _rescale_if_needed(annotations, stored_w, stored_h, target_size)

    # --- Attempts 3 & 4: legacy metadata JSON fallbacks ----------------------
    for metadata_path in (
        output_dir / "images" / f"{stem}.metadata.json",
        image_path.parent / f"{stem}.metadata.json",
    ):
        if not metadata_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
        except Exception:
            LOGGER.warning("Could not parse metadata file %s", metadata_path)
            continue

        # Annotation metadata (save_training_sample) has image_width; synthetic
        # metadata does not but has frame_annotations with pixel-space points.
        if "image_width" in metadata:
            raw = metadata.get("annotations", [])
        else:
            raw = metadata.get("frame_annotations", [])

        annotations = _filter_annotations_with_points(raw, metadata_path)
        if not annotations:
            continue
        stored_w = int(metadata.get("image_width", 0))
        stored_h = int(metadata.get("image_height", 0))
        return _rescale_if_needed(annotations, stored_w, stored_h, target_size)

    return []


def _filter_annotations_with_points(
    raw: list[Any], source: Path
) -> list[dict[str, Any]]:
    """Return copies of annotation dicts that have a non-empty ``points`` key."""
    result: list[dict[str, Any]] = []
    for ann in raw:
        if not isinstance(ann, dict):
            LOGGER.warning("Ignoring malformed annotation in %s: expected dict", source)
            continue
        if "points" not in ann or not ann["points"]:
            LOGGER.debug(
                "Skipping annotation without points in %s: type=%s",
                source, ann.get("type", "unknown"),
            )
            continue
        result.append(deepcopy(ann))
    return result


def _rescale_if_needed(
    annotations: list[dict[str, Any]],
    stored_w: int,
    stored_h: int,
    target_size: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    """Scale annotation points when *target_size* differs from stored dimensions."""
    if not annotations or target_size is None:
        return annotations
    target_w, target_h = int(target_size[0]), int(target_size[1])
    if stored_w < 1 or stored_h < 1:
        return annotations
    if stored_w == target_w and stored_h == target_h:
        return annotations
    sx = target_w / float(stored_w)
    sy = target_h / float(stored_h)
    return [scale_annotation_points(ann, sx, sy) for ann in annotations]


def import_annotations_from_old_format(
    source: Path,
    output_dir: Path,
) -> Path:
    """Import annotations from the old embedded-metadata format into the new layout.

    Reads *source* (either a ``*.metadata.json`` file or an image path whose
    metadata sidecar is discovered automatically) and writes the annotations
    to ``output_dir/annotations/<stem>.json``.

    Old format: annotations are stored in a ``annotations`` list inside the
    metadata JSON file, with each entry having a ``"points"`` key.
    Synthetic format: ``frame_annotations`` list inside metadata JSON.

    Returns the path of the written annotations file.
    """
    # Resolve the metadata JSON path.
    if source.suffix.lower() == ".json":
        metadata_path = source
    else:
        # Treat as an image path; look for the sidecar.
        metadata_path = source.parent / f"{source.stem}.metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"No metadata sidecar found for {source}. "
                f"Expected {metadata_path}"
            )

    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception as exc:
        raise ValueError(f"Could not parse metadata file {metadata_path}: {exc}") from exc

    # Collect annotations: prefer new-style 'annotations' with points;
    # fall back to synthetic 'frame_annotations'.
    raw: list[Any] = []
    if "image_width" in metadata:
        raw = metadata.get("annotations", [])
    if not raw:
        raw = metadata.get("frame_annotations", [])

    annotations = _filter_annotations_with_points(raw, metadata_path)
    if not annotations:
        raise ValueError(
            f"No importable annotations (with 'points' key) found in {metadata_path}"
        )

    # Derive the output stem. Metadata files are typically named
    # ``<stem>.metadata.json``, so strip the ``.metadata`` part when present.
    # If the file doesn't follow that convention the raw stem is used as-is.
    raw_stem = metadata_path.stem  # e.g. "plot_0000.metadata" or "plot_0000"
    stem = raw_stem.removesuffix(".metadata")
    ann_dir = output_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    out_path = ann_dir / f"{stem}.json"

    image_w = int(metadata.get("image_width", 0))
    image_h = int(metadata.get("image_height", 0))
    image_ref = metadata.get("image", str(metadata_path.parent / f"{stem}.png"))

    out_path.write_text(json.dumps({
        "image": image_ref,
        "image_width": image_w,
        "image_height": image_h,
        "annotations": annotations,
        "imported_from": str(metadata_path),
    }, indent=2))
    LOGGER.info(
        "Imported %d annotation(s) from %s → %s", len(annotations), metadata_path, out_path
    )
    return out_path
