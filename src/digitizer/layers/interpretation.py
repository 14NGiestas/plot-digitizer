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

    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
        if isinstance(x, list):
            x = x[-1] # Take last scale
        
        # Original features for segmentation head
        seg_features = x
        
        features = self.pool(x)
        features = self.flatten(features)
        
        # Features for heads
        embed = self.embed_proj(features)
        
        # Style predictions
        linestyle = self.linestyle_head(features)
        marker = self.marker_head(features)
        color = self.color_head(features)
        
        # Annotation attributes
        arrow_size = self.arrow_head_size_head(features)
        label_pos = self.label_pos_head(features)
        
        # Return features for the next layer in YOLO, and store others in a way 
        # that doesn't break the sequential list-passing of YOLO.
        # This is a hacky but effective way to pass extra data without breaking the graph
        # For training, we can access these via a hook.
        return seg_features
