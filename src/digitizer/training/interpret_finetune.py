"""Fine-tuning script for interpretation heads.

This script loads a trained YOLO segmentation model, freezes all layers except
the InterpretationProjection heads, and fine-tunes them using style labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS

from ..layers.interpretation import InterpretationProjection
from ..layers.loss import InterpretationLoss
from ..layers.style_utils import load_style_labels, style_labels_to_tensors


class StyleDataset(Dataset):
    """Dataset for style label fine-tuning."""

    def __init__(self, image_dir: Path, label_dir: Path, transforms=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transforms = transforms
        
        # Find all images
        self.image_paths = sorted([
            p for p in image_dir.iterdir() 
            if p.suffix.lower() in IMG_FORMATS
        ])
        
        if not self.image_paths:
            raise ValueError(f"No images found in {image_dir}")
            
        print(f"Found {len(self.image_paths)} images for style fine-tuning")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        stem = img_path.stem
        
        # Load style labels
        style_path = self.label_dir / f"{stem}_styles.json"
        style_labels = load_style_labels(style_path)
        
        # Convert to tensors
        targets = style_labels_to_tensors(style_labels)
        
        return {
            "image_path": img_path,
            "targets": targets,
        }


def fine_tune_interpretation_heads(
    model_path: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = "cpu",
):
    """Fine-tune interpretation heads on a trained model.
    
    Args:
        model_path: Path to trained YOLO model (.pt)
        dataset_dir: Directory containing images/ and labels/
        output_dir: Directory to save fine-tuned model
        epochs: Number of fine-tuning epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
    """
    model_path = Path(model_path)
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(str(model_path))
    model.to(device)
    
    # Find InterpretationProjection layer
    interp_layer = None
    for m in model.model.modules():
        if isinstance(m, InterpretationProjection):
            interp_layer = m
            break
    
    if interp_layer is None:
        raise ValueError("InterpretationProjection layer not found in model")
    
    print("Found InterpretationProjection layer")
    
    # Freeze all layers except interpretation heads
    for name, param in model.model.named_parameters():
        param.requires_grad = False
    
    # Unfreeze interpretation heads
    for name, param in interp_layer.named_parameters():
        param.requires_grad = True
    
    print("Frozen backbone and segmentation head, unfreezing interpretation heads")
    
    # Setup dataset
    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"
    
    if not image_dir.exists():
        raise ValueError(f"Image directory not found: {image_dir}")
    if not label_dir.exists():
        raise ValueError(f"Label directory not found: {label_dir}")
    
    dataset = StyleDataset(image_dir, label_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        [p for p in interp_layer.parameters() if p.requires_grad],
        lr=lr,
    )
    
    # Setup loss
    loss_fn = InterpretationLoss()
    
    # Training loop
    print(f"Starting fine-tuning for {epochs} epochs...")
    for epoch in range(epochs):
        interp_layer.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            img_path = batch["image_path"][0]
            targets = batch["targets"]
            
            # Skip if no targets
            if not targets:
                continue
            
            # Load image
            from PIL import Image
            import torchvision.transforms.functional as TF
            
            img = Image.open(img_path).convert("RGB")
            img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)
            
            # Move targets to device
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Forward pass through model
            with torch.set_grad_enabled(True):
                # Run model forward
                # The model will execute all layers, including InterpretationProjection
                # InterpretationProjection stores outputs in self.last_outputs
                model.model(img_tensor)
                
                # Get interpretation outputs
                predictions = interp_layer.last_outputs
            
            # Compute loss
            losses = loss_fn(predictions, targets)
            loss = losses["total"]
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save model
    output_path = output_dir / "best.pt"
    model.save(str(output_path))
    print(f"Saved fine-tuned model to {output_path}")
    
    return output_path
