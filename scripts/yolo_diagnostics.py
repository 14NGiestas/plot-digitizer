#!/usr/bin/env python3
"""YOLOv8 diagnostics helpers for plot-digitizer segmentation models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


def _load_gt_union_mask(label_path: Path, shape: tuple[int, int], classes: set[int] | None = None) -> np.ndarray:
    height, width = shape
    gt = np.zeros((height, width), dtype=np.uint8)
    if not label_path.exists():
        return gt
    for raw in label_path.read_text().splitlines():
        parts = raw.split()
        if len(parts) < 7:
            continue
        class_id = int(parts[0])
        if classes is not None and class_id not in classes:
            continue
        coords = np.asarray([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= width
        coords[:, 1] *= height
        poly = np.round(coords).astype(np.int32)
        if len(poly) >= 3:
            cv2.fillPoly(gt, [poly], 1)
    return gt


def _load_pred_union_mask(result, shape: tuple[int, int], classes: set[int] | None = None) -> np.ndarray:
    height, width = shape
    pred = np.zeros((height, width), dtype=np.uint8)
    if result.masks is None or result.boxes is None:
        return pred
    cls = result.boxes.cls.cpu().numpy().astype(int)
    masks = result.masks.data.cpu().numpy()
    for idx, class_id in enumerate(cls):
        if classes is not None and class_id not in classes:
            continue
        mask = (masks[idx] > 0.5).astype(np.uint8)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        pred = np.maximum(pred, mask)
    return pred


def _make_overlay(image: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    fn_mask = (gt == 1) & (pred == 0)
    fp_mask = (gt == 0) & (pred == 1)
    out = image.copy()
    out[fn_mask] = [0, 0, 255]
    out[fp_mask] = [0, 255, 255]
    return out


def _extract_maps(val_result, name: str) -> list[float]:
    metric = getattr(val_result, name, None)
    if metric is None:
        return []
    return [float(v) for v in metric.maps]


def run_diagnostics(
    model_path: Path,
    data_yaml: Path,
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    conf: float,
    iou: float,
    val_conf: float,
    target_classes: set[int],
    worst_k: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))

    val_result = model.val(data=str(data_yaml), split="val", conf=val_conf, iou=iou, plots=True)
    box_maps = _extract_maps(val_result, "box")
    seg_maps = _extract_maps(val_result, "seg")
    metrics = {
        "box_map_per_class": box_maps,
        "seg_map_per_class": seg_maps,
    }
    (output_dir / "per_class_metrics.json").write_text(json.dumps(metrics, indent=2))

    image_paths = sorted(
        [p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}]
    )
    confs: list[float] = []
    recalls: list[tuple[str, float]] = []

    for image_path in image_paths:
        result = model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            max_det=500,
            verbose=False,
            retina_masks=True,
        )[0]
        if result.boxes is not None and len(result.boxes):
            confs.extend(result.boxes.conf.cpu().numpy().astype(float).tolist())

        image = cv2.imread(str(image_path))
        if image is None:
            continue
        gt = _load_gt_union_mask(labels_dir / f"{image_path.stem}.txt", image.shape[:2], classes=target_classes)
        pred = _load_pred_union_mask(result, image.shape[:2], classes=target_classes)
        denom = int(gt.sum())
        if denom == 0:
            continue
        recall = float((gt & pred).sum() / denom)
        recalls.append((image_path.name, recall))

    fig = plt.figure(figsize=(8, 4))
    plt.hist(confs, bins=30)
    plt.title("Prediction confidence histogram")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "confidence_histogram.png", dpi=140)
    plt.close(fig)

    worst = sorted(recalls, key=lambda row: row[1])[:worst_k]
    (output_dir / "worst_recall_samples.json").write_text(
        json.dumps([{"image": name, "recall": score} for name, score in worst], indent=2)
    )

    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    for name, score in worst:
        image_path = images_dir / name
        result = model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            max_det=500,
            verbose=False,
            retina_masks=True,
        )[0]
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        gt = _load_gt_union_mask(labels_dir / f"{image_path.stem}.txt", image.shape[:2], classes=target_classes)
        pred = _load_pred_union_mask(result, image.shape[:2], classes=target_classes)
        overlay = _make_overlay(image, gt, pred)
        cv2.imwrite(str(overlay_dir / f"{image_path.stem}_recall_{score:.3f}.png"), overlay)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run YOLOv8 diagnostics on plot-digitizer datasets.")
    parser.add_argument("--model", type=Path, required=True, help="Path to YOLO model weights (best.pt).")
    parser.add_argument("--data-yaml", type=Path, required=True, help="Path to dataset.yaml.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory with validation images.")
    parser.add_argument("--labels-dir", type=Path, required=True, help="Directory with YOLO segmentation labels.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for diagnostics artifacts.")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--val-conf", type=float, default=0.001, help="Validation confidence for per-class metric sweep.")
    parser.add_argument("--target-classes", type=int, nargs="*", default=[0], help="Classes used for FN/FP overlays.")
    parser.add_argument("--worst-k", type=int, default=20, help="Number of worst-recall samples to export.")
    args = parser.parse_args()

    run_diagnostics(
        model_path=args.model,
        data_yaml=args.data_yaml,
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        conf=args.conf,
        iou=args.iou,
        val_conf=args.val_conf,
        target_classes=set(args.target_classes),
        worst_k=args.worst_k,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
