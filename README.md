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

Explanation-
1. The problem (why this project exists):-
"I built SmartQC, an on-device visual quality inspection tool. The idea started from a problem I'd seen before — small food producers and workshops can't afford industrial inspection hardware, but the alternative, cloud-based AI inspection, has real downsides: latency, ongoing cost per scan, and sending production images off-site. So I set out to build something that ran entirely on-device, on hardware a small business might already own."

2. Building and validating the model:-
"I trained an image classifier — MobileNetV2 — on a labeled dataset of good and defective bread samples, using transfer learning so I didn't need to train from scratch. It reached 99.2% validation accuracy. But I didn't stop at accuracy — I ran a full evaluation: confusion matrix, precision, recall, and ROC-AUC, which came out to 0.9997. That mattered to me because accuracy alone can hide problems, like a model that's great on one class and bad on the other."

3. Making it deployment-ready:-
"Since the challenge was built around Qualcomm's Snapdragon platform, I exported the model to ONNX and ran it through Qualcomm AI Hub, their cloud compilation service, to optimize it specifically for the Snapdragon X Elite chip. That gave me a genuinely deployment-ready model — not just a model that works in a notebook, but one compiled for real target hardware."

4. Turning results into product decisions:-
"Then I built an actual desktop app around it, and I made some deliberate product decisions here, not just engineering ones. For example, I added a confidence-threshold feature — if the model isn't confident, instead of forcing a guess, it flags the result as uncertain for manual review. I added that specifically after testing the model against a larger, messier batch of 352 real-world images and seeing it get a few genuinely ambiguous cases wrong — so I used that evaluation data to decide what feature to build next, rather than guessing."

5. Building for the real user, not just the demo:-
"I also added Grad-CAM explainability — a heatmap that shows which part of the image the model actually looked at — because for something like quality inspection, an operator won't trust a black-box yes/no answer. Being able to see 'here's the mold it detected' builds real trust in the tool. And I added batch processing with CSV export, because a real inspection workflow isn't one image at a time — someone would want to run a whole folder and get a report."

6. Results:-
"On the real-world batch test — 352 images the model hadn't seen — it got 97.4% right, correctly flagged the ambiguous ones instead of guessing wrong, and that gave me confidence the system would hold up outside a clean lab dataset."

7. Conclusion:-
"So end to end: I went from identifying a real gap — affordable, private, on-device inspection — to a trained and rigorously evaluated model, to a deployment-optimized build, to an actual usable application with features driven by what the evaluation data told me operators would need, all documented so someone else could reproduce or extend it."


