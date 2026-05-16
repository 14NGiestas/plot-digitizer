"""Loss functions for interpretation layer training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpretationLoss(nn.Module):
    """Combined loss function for interpretation layer training.
    
    Computes losses for:
    - Style classification (linestyle, marker, color, arrowstyle)
    - Style regression (linewidth, label position)
    """
    
    def __init__(self, weight_style: float = 1.0, weight_regression: float = 0.5):
        super().__init__()
        self.weight_style = weight_style
        self.weight_regression = weight_regression
        
        # Classification losses
        self.linestyle_loss = nn.CrossEntropyLoss()
        self.marker_loss = nn.CrossEntropyLoss()
        self.color_loss = nn.CrossEntropyLoss()
        self.arrowstyle_loss = nn.CrossEntropyLoss()
        
        # Regression losses
        self.linewidth_loss = nn.MSELoss()
        self.label_pos_loss = nn.MSELoss()
    
    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute all interpretation losses.
        
        Args:
            predictions: Dict with keys:
                - linestyle: [batch, 4] logits
                - marker: [batch, 4] logits
                - color: [batch, 6] logits
                - arrowstyle: [batch, 5] logits (optional)
                - linewidth: [batch, 1] 
                - label_pos: [batch, 2] (optional)
            targets: Dict with same keys containing ground truth values
            
        Returns:
            Dict with individual losses and total loss
        """
        losses = {}
        
        # Style classification losses
        if "linestyle" in predictions and "linestyle" in targets:
            losses["linestyle"] = self.linestyle_loss(
                predictions["linestyle"], targets["linestyle"].long()
            )
        
        if "marker" in predictions and "marker" in targets:
            losses["marker"] = self.marker_loss(
                predictions["marker"], targets["marker"].long()
            )
        
        if "color" in predictions and "color" in targets:
            losses["color"] = self.color_loss(
                predictions["color"], targets["color"].long()
            )
        
        if "arrowstyle" in predictions and "arrowstyle" in targets:
            losses["arrowstyle"] = self.arrowstyle_loss(
                predictions["arrowstyle"], targets["arrowstyle"].long()
            )
        
        # Regression losses
        if "linewidth" in predictions and "linewidth" in targets:
            losses["linewidth"] = self.linewidth_loss(
                predictions["linewidth"], targets["linewidth"]
            )
        
        if "label_pos" in predictions and "label_pos" in targets:
            losses["label_pos"] = self.label_pos_loss(
                predictions["label_pos"], targets["label_pos"]
            )
        
        # Compute total loss
        style_losses = sum(v for k, v in losses.items() if k in ("linestyle", "marker", "color", "arrowstyle"))
        regression_losses = sum(v for k, v in losses.items() if k in ("linewidth", "label_pos"))
        
        losses["total"] = self.weight_style * style_losses + self.weight_regression * regression_losses
        
        return losses
