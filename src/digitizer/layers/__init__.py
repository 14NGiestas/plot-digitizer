import ultralytics.nn.tasks
from .interpretation import InterpretationProjection
from .loss import InterpretationLoss
from .style_utils import load_style_labels, style_labels_to_tensors

# Register the custom module so the YAML parser finds it
ultralytics.nn.tasks.InterpretationProjection = InterpretationProjection

__all__ = [
    "InterpretationProjection",
    "InterpretationLoss",
    "load_style_labels",
    "style_labels_to_tensors",
]
