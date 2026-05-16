import ultralytics.nn.tasks
from .interpretation import InterpretationProjection

# Register the custom module so the YAML parser finds it
ultralytics.nn.tasks.InterpretationProjection = InterpretationProjection
