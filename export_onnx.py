"""
SmartQC — ONNX export script
 
Loads the trained weights from model/smartqc_weights.pt and exports the
model to ONNX format, ready for Qualcomm AI Hub optimization.
 
Run (from the SmartQC project root, with venv active):
    python export_onnx.py
"""
 
import json
from pathlib import Path
 
import torch
import torch.nn as nn
from torchvision import models
 
WEIGHTS_PATH = Path("model/smartqc_weights.pt")
ONNX_PATH = Path("model/smartqc_model.onnx")
CLASS_NAMES_PATH = Path("model/class_names.json")
 
IMG_SIZE = 224
 
 
def main():
    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
    class_names = checkpoint["class_names"]
    print(f"Loaded checkpoint. Classes: {class_names}")
 
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
 
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
 
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamic_axes=None,  # fixed batch size of 1 — simplest for on-device deployment
    )
    print(f"Exported ONNX model to: {ONNX_PATH.resolve()}")
 
    # Consolidate into a single self-contained .onnx file (no external
    # .data sidecar) — AI Hub's uploader expects one regular file.
    import onnx
    onnx_model = onnx.load(str(ONNX_PATH), load_external_data=True)
    onnx.save_model(
        onnx_model,
        str(ONNX_PATH),
        save_as_external_data=False,
    )
    print("Consolidated model into a single self-contained .onnx file.")
 
    # Save class names so the app / AI Hub steps can reference them,
    # in the exact index order the model outputs.
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved class names to: {CLASS_NAMES_PATH.resolve()}")
 
    print("\nNext: verify the model with ONNX Runtime, then run it through Qualcomm AI Hub.")
 
 
if __name__ == "__main__":
    main()
 