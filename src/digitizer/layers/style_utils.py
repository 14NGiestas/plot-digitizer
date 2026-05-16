"""Style label loading utilities for interpretation layer training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def load_style_labels(style_labels_path: Path) -> dict[str, Any]:
    """Load style labels from JSON file.
    
    Args:
        style_labels_path: Path to style_labels.json file
        
    Returns:
        Dict with 'curves' and 'arrows' lists containing style attributes
    """
    if not style_labels_path.exists():
        return {"curves": [], "arrows": []}
    
    with open(style_labels_path) as f:
        return json.load(f)


def style_labels_to_tensors(
    style_labels: dict[str, Any],
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Convert style labels to PyTorch tensors for training.
    
    Args:
        style_labels: Dict from load_style_labels()
        device: Target device for tensors
        
    Returns:
        Dict with tensor values for each style attribute
    """
    tensors = {}
    
    # Process curves
    curves = style_labels.get("curves", [])
    if curves:
        tensors["linestyle"] = torch.tensor(
            [c["linestyle"] for c in curves], device=device
        )
        tensors["marker"] = torch.tensor(
            [c["marker"] for c in curves], device=device
        )
        tensors["color"] = torch.tensor(
            [c["color"] for c in curves], device=device
        )
    
    # Process arrows
    arrows = style_labels.get("arrows", [])
    if arrows:
        tensors["arrowstyle"] = torch.tensor(
            [a["arrowstyle"] for a in arrows], device=device
        )
        tensors["linewidth"] = torch.tensor(
            [[a["linewidth"]] for a in arrows], device=device
        )
        if all("label_pos" in a for a in arrows):
            tensors["label_pos"] = torch.tensor(
                [[a["label_pos"]["x"], a["label_pos"]["y"]] for a in arrows],
                device=device,
            )
    
    return tensors
