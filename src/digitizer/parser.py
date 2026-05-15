"""CLI parser construction helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

def _parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Automatic AI-assisted plot digitizer.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Generate synthetic plots and YOLO segmentation labels.")
    generate_parser.add_argument("--output-dir", type=Path, required=True)
    generate_parser.add_argument("--count", type=int, default=16)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument("--image-format", default="png", choices=["png", "jpg"])
    generate_parser.add_argument("--plot-type", default="mixed", choices=["general", "bandstructure", "mixed"],
                                  help="Type of plots: general (standard curves), bandstructure (physics band diagrams), or mixed")
    generate_parser.add_argument(
        "--workers",
        type=_parse_positive_int,
        default=None,
        metavar="N",
        help="Number of worker processes for parallel generation (default: min(os.cpu_count(), count, 8)). Use 1 for sequential.",
    )

    train_parser = subparsers.add_parser("train", help="Train or plan a YOLO segmentation model.")
    train_parser.add_argument("--dataset-dir", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, default=Path("training-runs"))
    train_parser.add_argument("--weights", default="yolov8n-seg.pt")
    train_parser.add_argument("--epochs", type=int, default=25)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--batch", type=int, default=8)
    train_parser.add_argument("--hyp-yaml", type=Path, default=None, help="Optional Ultralytics training override YAML (cfg).")
    train_parser.add_argument("--execute", action="store_true", help="Run training immediately. Otherwise, only print the plan.")
    train_parser.add_argument(
        "--workers",
        type=_parse_positive_int,
        default=None,
        metavar="N",
        help="Number of training workers. Used for Ultralytics DataLoader workers and torch CPU thread pools when training executes. Set to available CPU cores, e.g. --workers 16.",
    )

    digitize_parser = subparsers.add_parser("digitize", help="Digitize one or more plot images.")
    digitize_parser.add_argument("inputs", nargs="+", help="Input image files or directories.")
    digitize_parser.add_argument("--output-dir", type=Path, default=Path("digitized-output"))
    digitize_parser.add_argument("--x-range", type=str, default=None)
    digitize_parser.add_argument("--y-range", type=str, default=None)
    digitize_parser.add_argument("--x-reference", type=str, default=None, help="Known X-axis points in px0:real0,px1:real1 format.")
    digitize_parser.add_argument("--y-reference", type=str, default=None, help="Known Y-axis points in px0:real0,px1:real1 format.")
    digitize_parser.add_argument(
        "--interactive-axis-selection",
        action="store_true",
        help="Interactively click two X-axis points and two Y-axis points, then enter their real values.",
    )
    digitize_parser.add_argument("--x-scale", choices=["linear", "log"], default="linear")
    digitize_parser.add_argument("--y-scale", choices=["linear", "log"], default="linear")
    digitize_parser.add_argument("--invert-y", action="store_true")
    digitize_parser.add_argument(
        "--disable-auto-axis-anchors",
        action="store_true",
        help="Disable automatic axis-anchor point detection for calibration fallback.",
    )
    digitize_parser.add_argument("--weights", default=None, help="YOLO .pt or .onnx segmentation weights.")
    digitize_parser.add_argument("--conf-threshold", type=float, default=0.25)
    digitize_parser.add_argument(
        "--workers",
        type=_parse_positive_int,
        default=None,
        metavar="N",
        help="Number of DataLoader worker processes for AI digitizing inference (default: Ultralytics default). Set to CPU core count, e.g. --workers 16 for a 16-core system.",
    )
    digitize_parser.add_argument("--overlay", action="store_true", help="Write overlay images.")

    validate_parser = subparsers.add_parser("validate", help="Validate a digitized CSV against ground truth.")
    validate_parser.add_argument("--prediction-csv", type=Path, required=True)
    validate_parser.add_argument("--truth-csv", type=Path, required=True)
    validate_parser.add_argument("--output-json", type=Path, default=None)

    annotate_parser = subparsers.add_parser(
        "annotate",
        help="Interactively annotate a plot image and save a YOLO training sample.",
    )
    annotate_parser.add_argument("input", type=Path, help="Input plot image to annotate.")
    annotate_parser.add_argument(
        "--output-dir", type=Path, default=Path("annotated-output"),
        help="Directory where the training sample (image, label, metadata) is written.",
    )
    annotate_parser.add_argument(
        "--line-width", type=float, default=3.0,
        help="Stroke width (pixels) used to build polygon envelopes for vbar/hbar/etc.",
    )
    annotate_parser.add_argument(
        "--resize-width",
        type=_parse_positive_int,
        default=None,
        help="Optional output image width in pixels. Requires --resize-height.",
    )
    annotate_parser.add_argument(
        "--resize-height",
        type=_parse_positive_int,
        default=None,
        help="Optional output image height in pixels. Requires --resize-width.",
    )
    annotate_parser.add_argument(
        "--update",
        action="store_true",
        help="Load existing saved annotations for this image and continue editing before saving.",
    )

    return parser


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")
