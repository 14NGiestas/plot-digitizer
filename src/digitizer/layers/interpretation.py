import torch
import torch.nn as nn
from typing import Any

class InterpretationProjection(nn.Module):
    """
    Projects YOLO feature maps to a fixed-dimension embedding
    and predicts plot style attributes.
    """
    def __init__(self, c1: int, c2: int, pool_type: str = 'avg'):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1) if pool_type == 'avg' else nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten()
        
        # Base projection for embedding
        self.embed_proj = nn.Linear(c1, c2)
        
        # Classification heads for style attributes
        # Predicts indices for: linestyle, marker, color palette
        self.linestyle_head = nn.Linear(c1, 4)  # 4 types: '-', '--', ':', '-.'
        self.marker_head = nn.Linear(c1, 4)     # 4 types: 'None', 'o', 'x', 's'
        self.color_head = nn.Linear(c1, 5)      # 5 palettes
        
        # Arrow/Label attribute regression heads
        self.arrow_head_size_head = nn.Linear(c1, 1) # scalar size
        self.label_pos_head = nn.Linear(c1, 2)       # (x, y) relative position

    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        """Forward pass that returns tensor for YOLO graph but computes style outputs.
        
        The interpretation outputs are stored in self.last_outputs for access during training.
        """
        if isinstance(x, list):
            x = x[-1] # Take last scale
        
        # Store original features for segmentation head
        seg_features = x
        
        # Compute interpretation outputs
        features = self.pool(x)
        features = self.flatten(features)
        
        embed = self.embed_proj(features)
        linestyle = self.linestyle_head(features)
        marker = self.marker_head(features)
        color = self.color_head(features)
        arrow_size = self.arrow_head_size_head(features)
        label_pos = self.label_pos_head(features)
        
        # Store outputs for training access
        self.last_outputs = {
            "embedding": embed,
            "linestyle": linestyle,
            "marker": marker,
            "color": color,
            "arrow_size": arrow_size,
            "label_pos": label_pos,
        }
        
        # Return original features to maintain YOLO graph
        return seg_features
