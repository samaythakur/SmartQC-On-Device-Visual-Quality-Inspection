"""
SmartQC — verify the exported ONNX model runs correctly with ONNX Runtime.
 
Run:
    python verify_onnx.py
"""
 
import json
from pathlib import Path
 
import numpy as np
import onnxruntime as ort
 
ONNX_PATH = Path("model/smartqc_model.onnx")
CLASS_NAMES_PATH = Path("model/class_names.json")
 
 
def main():
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
 
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Model input: {input_name}  shape={input_shape}")
 
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    outputs = session.run(None, {input_name: dummy})
    logits = outputs[0][0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    idx = int(np.argmax(probs))
 
    print(f"Ran successfully. Sample output -> {class_names[idx]} ({probs[idx]*100:.1f}%)")
    print("ONNX model is valid and ready for Qualcomm AI Hub optimization.")
 
 
if __name__ == "__main__":
    main()
 