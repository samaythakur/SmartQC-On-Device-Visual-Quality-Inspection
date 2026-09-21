# SmartQC: On-Device Visual Quality Inspection 
(Qualcomm | Snapdragon® AI Lab Build & Present Challenge)

SmartQC is a Windows desktop application that inspects product quality from a webcam feed or uploaded image, entirely on-device — no cloud calls, no internet dependency at inference time. The model was trained, evaluated, and compiled/optimized for the **Snapdragon X Elite** using **Qualcomm AI Hub**.

📄 [Project Report](./SmartQC_Project_Report.pdf) · 🎞️ [Pitch Deck (PDF)](./SmartQC_Pitch_Deck.pdf) · 📊 [Pitch Deck (PPTX)](./SmartQC_Pitch_Deck.pptx) · 🎥 [Demo Video](./SmartQC_Demo.mp4)

---

## Results

| Metric | Value |
|---|---|
| Validation accuracy | **99.2%** |
| ROC-AUC | **0.9997** |
| Validation set | 129 images (1 misclassification) |
| Real-world batch test | 352 uncurated images, **97.4%** correct |
| Qualcomm AI Hub compile job | `jpe782z15` → **Results Ready** on Snapdragon X Elite CRD |

## Features

- **Live / upload inspection** — classify a webcam frame or uploaded image, with label, confidence, and latency displayed
- **Confidence-threshold flagging** — adjustable slider (50–99%); predictions below the threshold are flagged **UNCERTAIN** instead of a forced good/defective label
- **Batch inspection** — process an entire folder of images at once, view a results table and summary counts, export to CSV
- **Explainability (Grad-CAM)** — on-demand heatmap overlay showing which region of the image drove the model's decision
- **Inspection history** — every inspection logged locally to SQLite (timestamp, label, confidence, latency)
- **Defect-rate trend chart** — rolling defect-rate visualization from the logged history
- **Fully offline at inference time** — no data ever leaves the device

## Architecture

```
Webcam / Image Upload
  → Preprocessing (OpenCV / PIL)
  → ONNX model, quantized and optimized via Qualcomm AI Hub
  → ONNX Runtime + QNN Execution Provider → Snapdragon NPU
  → Result: classification + confidence + inference latency
  → PyQt5 Desktop UI: live view, history, trend, explainability, batch mode
```

## Tech Stack

- **UI:** Python, PyQt5
- **Image processing:** OpenCV, Pillow
- **Model training:** PyTorch, torchvision (MobileNetV2, ImageNet-pretrained)
- **Model export:** ONNX (opset 13)
- **On-device inference:** ONNX Runtime (QNN Execution Provider path for Snapdragon NPU)
- **Cloud optimization:** Qualcomm AI Hub (`qai-hub`, `qai-hub-models`)
- **Explainability:** Grad-CAM (computed directly on the PyTorch model)
- **Storage:** SQLite
- **Evaluation:** scikit-learn, matplotlib

## Project Origin

SmartQC extends an earlier embedded-systems project — an AI-based bread quality indicator using an ESP32-CAM and a MobileNetV2 classifier — retargeted from a microcontroller pipeline into a fully software-based Windows application, explicitly optimized for Snapdragon-powered laptops.

## Run It

Run these from the **project root** (the top-level folder containing `app/`, `train.py`, etc.) — not from inside `app/`:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r app\requirements.txt
cd app
python smartqc_main.py
```

If you already have a `venv` set up from a previous step, skip straight to:
```bash
venv\Scripts\activate
cd app
python smartqc_main.py
```

## Reproduce From Scratch

All commands below assume you're in the **project root** (not inside `app/`) with the virtual environment activated.

1. **Train:** `python train.py` — fine-tunes MobileNetV2 on the Fresh/Mold bread dataset, saves `model/smartqc_weights.pt`
2. **Export to ONNX:** `python export_onnx.py` then `python verify_onnx.py`
3. **Qualcomm AI Hub optimization:**
   ```bash
   pip install qai-hub qai-hub-models
   qai-hub configure --api_token YOUR_TOKEN
   python aihub_compile.py
   ```
   Copy the resulting optimized model into `app/model/`.
4. **Evaluate:** `python eval_metrics.py` — reproduces the validation split, prints confusion matrix / precision / recall / F1 / ROC-AUC, saves plots
5. **Run the app:** see [Run It](#run-it) above

Full step-by-step instructions and the complete source code of every script are documented in the [Project Report](./SmartQC_Project_Report.pdf) (Appendix).

## Repository Structure

```
├── app/
│   ├── smartqc_main.py       # Desktop application (PyQt5)
│   ├── gradcam_engine.py     # Grad-CAM explainability
│   ├── requirements.txt
│   └── model/                # Trained weights + optimized ONNX model
├── Dataset-Fresh & Mold Bread Images/
│   ├── Fresh Bread(Good Bread) Images/
│   └── Mold Bread(Bad Bread) Images/
├── train.py                  # Model training
├── export_onnx.py            # ONNX export
├── verify_onnx.py            # ONNX verification
├── aihub_compile.py          # Qualcomm AI Hub compilation
├── eval_metrics.py           # Evaluation metrics + plots
├── list_devices.py           # AI Hub device listing (utility)
├── find_windows_device.py    # AI Hub Windows device search (utility)
├── SmartQC_Project_Report.pdf
├── SmartQC_Pitch_Deck.pdf / .pptx
└── SmartQC_Demo.mp4
```

See the [Project Report](./SmartQC_Project_Report.pdf) for full details, evaluation methodology, and future work.

