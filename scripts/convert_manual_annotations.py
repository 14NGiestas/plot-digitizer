#!/usr/bin/env python3
"""Convert manual annotations (vbars, arrows, etc.) to YOLO segmentation format.

This script helps you integrate manually collected annotation points into the
training data format expected by the digitizer's YOLO segmentation model.

Supported annotation types:
- vbar: Vertical bars (e.g., high-symmetry point markers in bandstructures)
- hbar: Horizontal bars (e.g., Fermi level lines, threshold lines)
- arrow: Arrow annotations pointing to features
- error_bar: Error bar markers
- curve: Curve/line segments

Usage:
    # Convert from CSV format
    python convert_manual_annotations.py --input manual_annotations.csv --image-width 1200 --image-height 800 --output-dir converted_labels/
    
    # Convert from JSON format  
    python convert_manual_annotations.py --input annotations.json --output-dir converted_labels/
    
    # Merge with existing synthetic dataset
    python convert_manual_annotations.py --input manual.csv --merge-with datasets/synthetic_bandstructure/ --output-dir merged_dataset/
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def normalize_coordinates(points: list[tuple[float, float]], image_width: int, image_height: int) -> list[float]:
    """Convert pixel coordinates to normalized YOLO format [x1, y1, x2, y2, ...]."""
    normalized = []
    for x, y in points:
        nx = float(np.clip(x / image_width, 0.0, 1.0))
        ny = float(np.clip(y / image_height, 0.0, 1.0))
        normalized.extend([nx, ny])
    return normalized


def vbar_to_polygon(x_center: float, y_top: float, y_bottom: float, 
                    width: float, image_width: int, image_height: int) -> list[float]:
    """Convert vertical bar parameters to polygon points."""
    half_width = width / 2
    points = [
        (x_center - half_width, y_top),
        (x_center + half_width, y_top),
        (x_center + half_width, y_bottom),
        (x_center - half_width, y_bottom),
    ]
    return normalize_coordinates(points, image_width, image_height)


def hbar_to_polygon(y_center: float, x_left: float, x_right: float,
                    height: float, image_width: int, image_height: int) -> list[float]:
    """Convert horizontal bar parameters to polygon points."""
    half_height = height / 2
    points = [
        (x_left, y_center - half_height),
        (x_right, y_center - half_height),
        (x_right, y_center + half_height),
        (x_left, y_center + half_height),
    ]
    return normalize_coordinates(points, image_width, image_height)


def arrow_to_polygon(start: tuple[float, float], end: tuple[float, float],
                     arrow_width: float = 5.0, image_width: int = 1200, 
                     image_height: int = 800) -> list[float]:
    """Convert arrow annotation to polygon (simplified as line envelope)."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = np.sqrt(dx**2 + dy**2)
    if length == 0:
        return []
    
    nx = -dy / length
    ny = dx / length
    
    half_w = arrow_width / 2
    points = [
        (start[0] + nx * half_w, start[1] + ny * half_w),
        (end[0] + nx * half_w, end[1] + ny * half_w),
        (end[0] - nx * half_w, end[1] - ny * half_w),
        (start[0] - nx * half_w, start[1] - ny * half_w),
    ]
    return normalize_coordinates(points, image_width, image_height)


def error_bar_to_polygon(x_center: float, y_center: float, y_error: float,
                         cap_width: float = 10.0, line_width: float = 2.0,
                         image_width: int = 1200, image_height: int = 800) -> list[float]:
    """Convert error bar to polygon."""
    y_top = y_center - y_error
    y_bottom = y_center + y_error
    half_cap = cap_width / 2
    half_line = line_width / 2
    
    points = [
        (x_center - half_cap, y_top),
        (x_center + half_cap, y_top),
        (x_center + half_line, y_top),
        (x_center + half_line, y_bottom),
        (x_center + half_cap, y_bottom),
        (x_center - half_cap, y_bottom),
        (x_center - half_line, y_bottom),
        (x_center - half_line, y_top),
    ]
    return normalize_coordinates(points, image_width, image_height)


def curve_to_polygon(points: list[tuple[float, float]], image_width: int, 
                     image_height: int, line_width: float = 3.0) -> list[float]:
    """Convert curve points to polygon by creating an envelope."""
    if len(points) < 2:
        return []
    
    all_points = []
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            continue
        
        nx = -dy / length
        ny = dx / length
        half_w = line_width / 2
        
        all_points.extend([
            (p1[0] + nx * half_w, p1[1] + ny * half_w),
            (p2[0] + nx * half_w, p2[1] + ny * half_w),
        ])
    
    for i in range(len(points) - 1, 0, -1):
        p1 = points[i]
        p2 = points[i - 1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.sqrt(dx**2 + dy**2)
        if length == 0:
            continue
        
        nx = -dy / length
        ny = dx / length
        half_w = line_width / 2
        
        all_points.extend([
            (p1[0] - nx * half_w, p1[1] - ny * half_w),
            (p2[0] - nx * half_w, p2[1] - ny * half_w),
        ])
    
    return normalize_coordinates(all_points, image_width, image_height)


CLASS_MAPPING = {
    "curve": 0,
    "vbar": 1,
    "hbar": 2,
    "arrow": 3,
    "error_bar": 4,
}


def convert_csv_annotations(input_path: Path, output_dir: Path, 
                           image_width: int, image_height: int) -> dict[str, list[str]]:
    """Convert CSV annotations to YOLO format."""
    labels_by_image: dict[str, list[str]] = {}
    
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = Path(row['image_name']).stem
            ann_type = row['type'].lower()
            
            if ann_type not in CLASS_MAPPING:
                print(f"Warning: Unknown annotation type '{ann_type}', skipping")
                continue
            
            class_id = CLASS_MAPPING[ann_type]
            
            try:
                if ann_type == "vbar":
                    x_center = float(row['x'])
                    y_top = float(row['y_top'])
                    y_bottom = float(row['y_bottom'])
                    width = float(row.get('width', 3.0))
                    polygon = vbar_to_polygon(x_center, y_top, y_bottom, width, 
                                             image_width, image_height)
                
                elif ann_type == "hbar":
                    y_center = float(row['y'])
                    x_left = float(row['x_left'])
                    x_right = float(row['x_right'])
                    height = float(row.get('height', 3.0))
                    polygon = hbar_to_polygon(y_center, x_left, x_right, height,
                                             image_width, image_height)
                
                elif ann_type == "arrow":
                    start_x = float(row['start_x'])
                    start_y = float(row['start_y'])
                    end_x = float(row['end_x'])
                    end_y = float(row['end_y'])
                    polygon = arrow_to_polygon((start_x, start_y), (end_x, end_y),
                                              arrow_width=float(row.get('width', 5.0)),
                                              image_width=image_width, 
                                              image_height=image_height)
                
                elif ann_type == "error_bar":
                    x_center = float(row['x'])
                    y_center = float(row['y'])
                    y_error = float(row['y_error'])
                    polygon = error_bar_to_polygon(x_center, y_center, y_error,
                                                   image_width=image_width,
                                                   image_height=image_height)
                
                elif ann_type == "curve":
                    if 'points_json' in row:
                        points = json.loads(row['points_json'])
                        polygon = curve_to_polygon(points, image_width, image_height)
                    else:
                        points = []
                        i = 0
                        while f'x{i}' in row and f'y{i}' in row:
                            points.append((float(row[f'x{i}']), float(row[f'y{i}'])))
                            i += 1
                        polygon = curve_to_polygon(points, image_width, image_height,
                                                  line_width=float(row.get('width', 3.0)))
                else:
                    continue
                
                if polygon and len(polygon) >= 6:
                    label_line = f"{class_id} " + " ".join(f"{v:.6f}" for v in polygon)
                    
                    if image_name not in labels_by_image:
                        labels_by_image[image_name] = []
                    labels_by_image[image_name].append(label_line)
                    
            except (KeyError, ValueError) as e:
                print(f"Warning: Error processing row: {e}")
                continue
    
    return labels_by_image


def convert_json_annotations(input_path: Path, output_dir: Path,
                            default_image_width: int = 1200,
                            default_image_height: int = 800) -> dict[str, list[str]]:
    """Convert JSON annotations to YOLO format."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    labels_by_image: dict[str, list[str]] = {}
    images_info = data.get("images", {})
    
    for ann in data.get("annotations", []):
        image_name = Path(ann['image']).stem
        ann_type = ann['type'].lower()
        
        if ann_type not in CLASS_MAPPING:
            print(f"Warning: Unknown annotation type '{ann_type}', skipping")
            continue
        
        class_id = CLASS_MAPPING[ann_type]
        
        img_info = images_info.get(ann['image'], {})
        img_width = img_info.get('width', default_image_width)
        img_height = img_info.get('height', default_image_height)
        
        try:
            if ann_type == "vbar":
                polygon = vbar_to_polygon(
                    ann['x'], ann['y_top'], ann['y_bottom'],
                    ann.get('width', 3.0), img_width, img_height
                )
            elif ann_type == "hbar":
                polygon = hbar_to_polygon(
                    ann['y'], ann['x_left'], ann['x_right'],
                    ann.get('height', 3.0), img_width, img_height
                )
            elif ann_type == "arrow":
                polygon = arrow_to_polygon(
                    (ann['start_x'], ann['start_y']),
                    (ann['end_x'], ann['end_y']),
                    ann.get('width', 5.0), img_width, img_height
                )
            elif ann_type == "error_bar":
                polygon = error_bar_to_polygon(
                    ann['x'], ann['y'], ann['y_error'],
                    image_width=img_width, image_height=img_height
                )
            elif ann_type == "curve":
                points = ann.get('points', [])
                polygon = curve_to_polygon(
                    points, img_width, img_height,
                    line_width=ann.get('width', 3.0)
                )
            else:
                continue
            
            if polygon and len(polygon) >= 6:
                label_line = f"{class_id} " + " ".join(f"{v:.6f}" for v in polygon)
                
                if image_name not in labels_by_image:
                    labels_by_image[image_name] = []
                labels_by_image[image_name].append(label_line)
                
        except (KeyError, ValueError) as e:
            print(f"Warning: Error processing annotation: {e}")
            continue
    
    return labels_by_image


def merge_with_dataset(labels_by_image: dict[str, list[str]], 
                       source_dataset_dir: Path, 
                       output_dir: Path) -> None:
    """Merge converted labels with an existing synthetic dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import shutil
    
    for subdir in ["images", "ground_truth"]:
        src_subdir = source_dataset_dir / subdir
        dst_subdir = output_dir / subdir
        dst_subdir.mkdir(parents=True, exist_ok=True)
        if src_subdir.exists():
            for item in src_subdir.iterdir():
                if item.is_file():
                    shutil.copy2(item, dst_subdir / item.name)
    
    src_labels_dir = source_dataset_dir / "labels"
    dst_labels_dir = output_dir / "labels"
    dst_labels_dir.mkdir(exist_ok=True)
    
    if src_labels_dir.exists():
        for item in src_labels_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, dst_labels_dir / item.name)
    
    for image_name, label_lines in labels_by_image.items():
        label_path = dst_labels_dir / f"{image_name}.txt"
        existing_lines = []
        if label_path.exists():
            existing_lines = label_path.read_text().strip().split('\n')
            existing_lines = [l for l in existing_lines if l.strip()]
        
        all_lines = existing_lines + label_lines
        label_path.write_text('\n'.join(all_lines))
    
    src_yaml = source_dataset_dir / "dataset.yaml"
    if src_yaml.exists():
        shutil.copy2(src_yaml, output_dir / "dataset.yaml")


def main():
    parser = argparse.ArgumentParser(
        description="Convert manual annotations to YOLO segmentation format"
    )
    parser.add_argument("--input", type=Path, required=True,
                       help="Input CSV or JSON file with manual annotations")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for YOLO format labels")
    parser.add_argument("--image-width", type=int, default=1200,
                       help="Default image width (if not specified in annotations)")
    parser.add_argument("--image-height", type=int, default=800,
                       help="Default image height (if not specified in annotations)")
    parser.add_argument("--merge-with", type=Path, default=None,
                       help="Merge with existing synthetic dataset directory")
    
    args = parser.parse_args()
    
    if args.input.suffix.lower() == '.csv':
        labels_by_image = convert_csv_annotations(
            args.input, args.output_dir, 
            args.image_width, args.image_height
        )
    elif args.input.suffix.lower() == '.json':
        labels_by_image = convert_json_annotations(
            args.input, args.output_dir,
            args.image_width, args.image_height
        )
    else:
        print("Error: Input file must be CSV or JSON")
        return 1
    
    if args.merge_with:
        merge_with_dataset(labels_by_image, args.merge_with, args.output_dir)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        labels_dir = args.output_dir / "labels"
        labels_dir.mkdir(exist_ok=True)
        
        for image_name, label_lines in labels_by_image.items():
            label_path = labels_dir / f"{image_name}.txt"
            label_path.write_text('\n'.join(label_lines))
    
    print(f"Converted {sum(len(v) for v in labels_by_image.values())} annotations "
          f"for {len(labels_by_image)} images")
    print(f"Output written to: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
