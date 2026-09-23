"""
SmartQC — Streamlit web demo

A lightweight browser-based version of SmartQC for public sharing via
Streamlit Community Cloud. Loads the same trained PyTorch weights used
by the desktop app and exposes: image upload -> prediction + confidence
+ Grad-CAM explainability heatmap.

Note: this runs standard CPU inference on Streamlit's servers, not
on-device Snapdragon NPU inference — it's a shareable demo of the
model's behavior, not the on-device deployment story (see the GitHub
repo / project report for that).

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy: push this file + requirements.txt + smartqc_weights.pt to a
GitHub repo, then deploy at share.streamlit.io.
"""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models
import os


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "smartqc_weights.pt"
)

IMG_SIZE = 224

st.set_page_config(
    page_title="SmartQC — Visual Quality Inspection",
    page_icon="🍞",
    layout="wide"
)


# ---------------------------------------------------------------------------
# Load model once, cached across reruns
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")

    class_names = checkpoint["class_names"]

    model = models.mobilenet_v2()

    model.classifier[1] = torch.nn.Linear(
        model.last_channel,
        len(class_names)
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, class_names


model, CLASS_NAMES = load_model()

target_layer = model.features[-1]

_state = {
    "activations": None,
    "gradients": None
}


def _save_activation(module, input, output):
    _state["activations"] = output.detach()


def _save_gradient(module, grad_input, grad_output):
    _state["gradients"] = grad_output[0].detach()


target_layer.register_forward_hook(_save_activation)
target_layer.register_full_backward_hook(_save_gradient)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def clean_label(raw_label: str) -> str:
    low = raw_label.lower()

    return "Good" if "good" in low or "fresh" in low else "Defective"


def preprocess(image_rgb: np.ndarray) -> torch.Tensor:
    img = cv2.resize(
        image_rgb,
        (IMG_SIZE, IMG_SIZE)
    )

    img = img.astype(np.float32) / 255.0

    img = (
        img - np.array([0.485, 0.456, 0.406])
    ) / np.array([0.229, 0.224, 0.225])

    img = np.transpose(
        img,
        (2, 0, 1)
    )[None, ...].astype(np.float32)

    return torch.from_numpy(img)


def predict_with_gradcam(
    image_rgb: np.ndarray,
    threshold_pct: int
):
    x = preprocess(image_rgb)

    logits = model(x)

    probs = F.softmax(
        logits,
        dim=1
    )[0]

    class_idx = int(
        torch.argmax(probs).item()
    )

    confidence = float(
        probs[class_idx].item()
    )

    # ---------------------------------------------------------------
    # Grad-CAM
    # ---------------------------------------------------------------

    model.zero_grad()

    logits[0, class_idx].backward()

    activations = _state["activations"][0]
    gradients = _state["gradients"][0]

    weights = gradients.mean(
        dim=(1, 2)
    )

    cam = torch.zeros(
        activations.shape[1:],
        dtype=torch.float32
    )

    for c, w in enumerate(weights):
        cam += w * activations[c]

    cam = F.relu(cam).numpy()

    if cam.max() > 0:
        cam = cam / cam.max()

    cam_resized = cv2.resize(
        cam,
        (
            image_rgb.shape[1],
            image_rgb.shape[0]
        )
    )

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized),
        cv2.COLORMAP_JET
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        image_rgb,
        0.6,
        heatmap_rgb,
        0.4,
        0
    )

    # ---------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------

    is_uncertain = (
        confidence < threshold_pct / 100.0
    )

    label = clean_label(
        CLASS_NAMES[class_idx]
    )

    per_class = {
        clean_label(name): float(probs[i])
        for i, name in enumerate(CLASS_NAMES)
    }

    return (
        label,
        confidence,
        is_uncertain,
        overlay,
        per_class
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title(
    "🍞 SmartQC — On-Device Visual Quality Inspection"
)

st.markdown(
    """
    Upload a bread image to classify it as **Good** or **Defective**, with a
    Grad-CAM heatmap showing which region influenced the decision.

    This model was trained to **99.2% validation accuracy** (0.9997 ROC-AUC)
    and compiled/optimized for the **Snapdragon X Elite** via Qualcomm AI Hub
    for the original desktop application.

    This web demo runs standard CPU inference for public sharing — see the
    [GitHub repo](https://github.com/samaythakur/SmartQC-On-Device-Visual-Quality-Inspection)
    for the full on-device story, evaluation, and source code.
    """
)


# ---------------------------------------------------------------------------
# Main application layout
# ---------------------------------------------------------------------------

col1, col2 = st.columns(2)


# ---------------------------------------------------------------------------
# Left column
# ---------------------------------------------------------------------------

with col1:

    uploaded_file = st.file_uploader(
        "Upload a bread image",
        type=["png", "jpg", "jpeg"]
    )

    threshold = st.slider(
        "Confidence threshold (%)",
        50,
        99,
        70
    )

    if uploaded_file is not None:

        # UPDATED STREAMLIT API
        st.image(
            uploaded_file,
            caption="Uploaded image",
            width="stretch"
        )


# ---------------------------------------------------------------------------
# Right column
# ---------------------------------------------------------------------------

with col2:

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_rgb = np.array(image)

        with st.spinner("Running inference..."):

            (
                label,
                confidence,
                is_uncertain,
                overlay,
                per_class
            ) = predict_with_gradcam(
                image_rgb,
                threshold
            )

        # ---------------------------------------------------------------
        # Prediction result
        # ---------------------------------------------------------------

        if is_uncertain:

            st.warning(
                f"⚠️ UNCERTAIN — best guess: "
                f"{label} ({confidence * 100:.1f}%)"
            )

        elif label == "Good":

            st.success(
                f"✅ GOOD — "
                f"{confidence * 100:.1f}% confidence"
            )

        else:

            st.error(
                f"❌ DEFECTIVE — "
                f"{confidence * 100:.1f}% confidence"
            )

        # ---------------------------------------------------------------
        # Class probabilities
        # ---------------------------------------------------------------

        st.bar_chart(per_class)

        # ---------------------------------------------------------------
        # Grad-CAM
        # ---------------------------------------------------------------

        st.image(
            overlay,
            caption="Grad-CAM Explainability Heatmap",
            width="stretch"
        )

        st.caption(
            "Warmer colors (red/yellow) show the regions "
            "that most influenced this result."
        )

    else:

        st.info(
            "Upload an image on the left to run an inspection."
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()

st.caption(
    "Built by Samay Thakur — Electronics & Instrumentation Engineering, "
    "M.S. Ramaiah Institute of Technology, Bengaluru."
)
