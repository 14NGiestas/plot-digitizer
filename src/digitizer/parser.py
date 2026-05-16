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

    # ── generate ──────────────────────────────────────────────────────────
    gen = subparsers.add_parser("generate", help="Generate synthetic plots and YOLO segmentation labels.")
    gen.add_argument("--output-dir", type=Path, default=Path("train-dataset"))
    gen.add_argument("--count", type=int, default=16)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--image-format", default="png", choices=["png", "jpg"])
    gen.add_argument("--plot-type", default="mixed", choices=["general", "bandstructure", "mixed"])
    gen.add_argument("--degradations", type=_parse_positive_int, default=1, metavar="N",
                     help="Degraded variants per base plot (default: 1).")
    gen.add_argument("--workers", type=_parse_positive_int, default=None, metavar="N",
                     help="Worker processes for parallel generation.")
    gen.add_argument("--difficulty", type=int, choices=[0, 1, 2, 3, 4], default=0, metavar="LEVEL",
                     help="Difficulty level (0=all, 1=easy … 4=hard).")
    gen.add_argument("--curriculum", action="store_true",
                     help="Distribute samples across difficulty levels 1–4.")

    # ── train (= curriculum) ──────────────────────────────────────────────
    trn = subparsers.add_parser("train", help="Run curriculum learning with MLflow tracking.")
    trn.add_argument("--output-dir", type=Path, default=Path("runs"))
    trn.add_argument("--samples-per-stage", type=int, default=500)
    trn.add_argument("--seed", type=int, default=42)
    trn.add_argument("--epochs", type=int, default=25)
    trn.add_argument("--batch", type=int, default=16)
    trn.add_argument("--workers", type=_parse_positive_int, default=None, metavar="N")
    trn.add_argument("--from-stage", type=int, choices=[1, 2, 3, 4], default=None,
                     help="Resume from stage N (1–4).")
    trn.add_argument("--resume", action="store_true",
                     help="Auto-resume from last completed stage.")
    trn.add_argument("--status", action="store_true",
                     help="Show progress without running.")
    trn.add_argument("--chain-info", action="store_true",
                     help="Show weight chain without running.")
    trn.add_argument("--sync", action="store_true",
                     help="Scan checkpoints and update progress.json.")

    # ── digitize ──────────────────────────────────────────────────────────
    dig = subparsers.add_parser("digitize", help="Digitize one or more plot images.")
    dig.add_argument("inputs", nargs="+", help="Input image files or directories.")
    dig.add_argument("--output-dir", type=Path, default=Path("digitized-output"))
    dig.add_argument("--x-range", type=str, default=None)
    dig.add_argument("--y-range", type=str, default=None)
    dig.add_argument("--x-reference", type=str, default=None,
                     help="Known X-axis points: px0:real0,px1:real1")
    dig.add_argument("--y-reference", type=str, default=None,
                     help="Known Y-axis points: px0:real0,px1:real1")
    dig.add_argument("--x-scale", choices=["linear", "log"], default="linear")
    dig.add_argument("--y-scale", choices=["linear", "log"], default="linear")
    dig.add_argument("--invert-y", action="store_true")
    dig.add_argument("--disable-auto-axis-anchors", action="store_true")
    dig.add_argument("--weights", default=None, help="YOLO .pt or .onnx weights.")
    dig.add_argument("--conf-threshold", type=float, default=0.25)
    dig.add_argument("--imgsz", type=_parse_positive_int, default=None)
    dig.add_argument("--workers", type=_parse_positive_int, default=None, metavar="N")
    dig.add_argument("--overlay", action="store_true", help="Write overlay images.")

    # ── annotate ─────────────────────────────────────────────────────────
    ann = subparsers.add_parser(
        "annotate",
        help="Interactively annotate a plot image and save a YOLO training sample.",
    )
    ann.add_argument("input", type=Path, help="Input plot image to annotate.")
    ann.add_argument("--output-dir", type=Path, default=Path("train-dataset"))
    ann.add_argument("--line-width", type=float, default=3.0,
                     help="Stroke width for polygon envelopes.")
    ann.add_argument("--resize-width", type=_parse_positive_int, default=None)
    ann.add_argument("--resize-height", type=_parse_positive_int, default=None)
    ann.add_argument("--update", action="store_true",
                     help="Deprecated — kept for backward compatibility.")

    return parser


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")
