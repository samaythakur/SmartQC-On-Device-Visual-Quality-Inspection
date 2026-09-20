# SmartQC — On-Device Visual Quality Inspection (Snapdragon)

A Windows desktop app that classifies product quality/defects from a webcam
or uploaded image, running inference locally on the Snapdragon NPU via
Qualcomm AI Hub-optimized ONNX models. No cloud calls at inference time.

Built for the Snapdragon® AI Lab Build & Present Challenge.

## Status
This repo currently ships a working PyQt UI skeleton (`app/main.py`) with a
mock inference fallback, so the app is demoable end-to-end before the real
model is dropped in. Wire in your trained model to go live.

## Run it
```bash
cd app
pip install -r requirements.txt
python main.py
```

## Wiring in your model
1. Train / export your classifier to ONNX.
2. Run it through Qualcomm AI Hub to compile/quantize for your Snapdragon
   target device.
3. Place the optimized `.onnx` file at `app/model/smartqc_model.onnx`.
4. Update `CLASS_NAMES` and `INPUT_SIZE` in `main.py` to match your model.
5. Install `onnxruntime-qnn` (or the appropriate QNN-enabled onnxruntime
   build) so the `QNNExecutionProvider` is available for the "npu" backend.

## Architecture
```
Webcam / Image Upload
  -> Preprocessing (OpenCV/PIL)
  -> ONNX model (Qualcomm AI Hub optimized)
  -> ONNX Runtime + QNN Execution Provider (Snapdragon NPU)
  -> Result: label + confidence + latency
  -> PyQt UI: live view, CPU-vs-NPU benchmark, history log, trend chart
```

## Features
- Live webcam or image-upload classification with confidence score
- CPU-vs-NPU latency benchmark toggle (the core "why Snapdragon" demo)
- Local SQLite inspection history log
- Rolling defect-rate trend chart

## Project background
Builds on an earlier AI-based bread quality indicator system
(ESP32-CAM + MobileNetV2), retargeted here as a fully software-based
Windows application for Snapdragon-powered laptops.
