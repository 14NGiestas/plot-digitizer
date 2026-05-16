import torch
from ultralytics import YOLO
import sys
from pathlib import Path

# Adiciona src ao path para garantir que o package digitizer seja encontrado
sys.path.append(str(Path(__file__).parent.parent / "src"))

from digitizer.layers.interpretation import InterpretationProjection

def test_custom_model_load():
    # Carrega modelo com arquitetura customizada
    model = YOLO("runs/custom_yolo11s-seg-interpret.yaml")
    
    # Verifica se a camada foi injetada corretamente na estrutura
    # Baseado no YAML acima, ela deve ser a camada 17 (após a inserção)
    assert isinstance(model.model.model[17], InterpretationProjection), "InterpretationProjection layer not found at expected index"
    print("✓ Model loaded and InterpretationProjection layer registered.")

    # Dummy input
    dummy = torch.randn(1, 3, 640, 640)
    
    # Forward pass
    output = model(dummy)
    print("✓ Forward pass completed.")
    
    # Verify the layer has the expected attributes
    layer = model.model.model[17]
    assert hasattr(layer, 'embed_proj'), "Layer missing embed_proj"
    assert hasattr(layer, 'linestyle_head'), "Layer missing linestyle_head"
    print("✓ Interpretation layer has expected attributes.")

if __name__ == "__main__":
    test_custom_model_load()
