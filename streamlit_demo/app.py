"""
SmartQC — Streamlit Live Demo
Compact UI version.

Browser-based demonstration of SmartQC.
The public web demo runs CPU inference on Streamlit Cloud.
The original desktop application is designed for Snapdragon NPU
deployment through the Qualcomm AI Hub / QNN path.
"""

import os
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import models


# =============================================================================
# CONFIG
# =============================================================================

WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "smartqc_weights.pt",
)

IMG_SIZE = 224

GITHUB_URL = (
    "https://github.com/samaythakur/"
    "SmartQC-On-Device-Visual-Quality-Inspection"
)


# =============================================================================
# PAGE
# =============================================================================

st.set_page_config(
    page_title="SmartQC — Visual Quality Inspection",
    page_icon="🍞",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =============================================================================
# COMPACT UI
# =============================================================================

st.markdown(
    """
    <style>
    /* Overall spacing */
    .block-container {
        max-width: 1400px;
        padding-top: 0.8rem;
        padding-bottom: 0.4rem;
    }

    /* Reduce Streamlit vertical gaps */
    div[data-testid="stVerticalBlock"] {
        gap: 0.55rem;
    }

    div[data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }

    /* Header */
    .smartqc-header {
        padding: 0.1rem 0 0.35rem 0;
    }

    .smartqc-title {
        font-size: 2.05rem;
        line-height: 1.15;
        font-weight: 750;
        margin: 0;
    }

    .smartqc-subtitle {
        font-size: 0.92rem;
        opacity: 0.68;
        margin-top: 0.25rem;
    }

    /* Small status badge */
    .status-row {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        margin-top: -2.1rem;
        margin-bottom: 0.7rem;
    }

    .status-badge {
        display: inline-block;
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        background: rgba(76, 154, 255, 0.10);
        border: 1px solid rgba(76, 154, 255, 0.30);
        color: #79b8ff;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Compact information strip */
    .demo-strip {
        padding: 0.55rem 0.8rem;
        margin: 0.25rem 0 0.65rem 0;
        border-left: 3px solid #4c9aff;
        background: rgba(76, 154, 255, 0.07);
        border-radius: 5px;
        font-size: 0.82rem;
        opacity: 0.9;
    }

    /* Result cards */
    .result-card {
        padding: 0.8rem 0.9rem;
        border-radius: 9px;
        margin: 0.35rem 0 0.55rem 0;
        border: 1px solid rgba(128,128,128,0.25);
    }

    .result-good {
        background: rgba(40, 167, 69, 0.12);
    }

    .result-defective {
        background: rgba(220, 53, 69, 0.13);
    }

    .result-uncertain {
        background: rgba(255, 193, 7, 0.12);
    }

    .result-label {
        font-size: 1.25rem;
        font-weight: 700;
    }

    .result-confidence {
        font-size: 0.82rem;
        opacity: 0.75;
    }

    /* Compact section headings */
    h2, h3 {
        margin-top: 0.35rem !important;
        margin-bottom: 0.35rem !important;
    }

    /* Compact footer */
    .footer {
        text-align: center;
        opacity: 0.48;
        padding: 0.65rem 0 0.2rem 0;
        font-size: 0.72rem;
    }

    /* Make tab area compact */
    button[data-baseweb="tab"] {
        padding-top: 0.45rem;
        padding-bottom: 0.45rem;
    }

    /* Hide unnecessary Streamlit menu text when possible */
    footer {
        visibility: hidden;
        height: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SESSION STATE
# =============================================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "batch_results" not in st.session_state:
    st.session_state.batch_results = pd.DataFrame()


# =============================================================================
# MODEL
# =============================================================================

@st.cache_resource
def load_model():
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Model weights not found at: {WEIGHTS_PATH}"
        )

    checkpoint = torch.load(
        WEIGHTS_PATH,
        map_location="cpu",
    )

    class_names = checkpoint["class_names"]

    model = models.mobilenet_v2()
    model.classifier[1] = torch.nn.Linear(
        model.last_channel,
        len(class_names),
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, class_names


model, CLASS_NAMES = load_model()


# =============================================================================
# HELPERS
# =============================================================================

def clean_label(raw_label: str) -> str:
    low = raw_label.lower()

    if "good" in low or "fresh" in low:
        return "Good"

    return "Defective"


def preprocess(image_rgb: np.ndarray) -> torch.Tensor:
    img = cv2.resize(
        image_rgb,
        (IMG_SIZE, IMG_SIZE),
    )

    img = img.astype(np.float32) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    img = img[None, ...].astype(np.float32)

    return torch.from_numpy(img)


def predict_image(image_rgb: np.ndarray):
    """CPU prediction used by batch inspection."""

    x = preprocess(image_rgb)

    start = time.perf_counter()

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    latency_ms = (time.perf_counter() - start) * 1000.0

    class_idx = int(torch.argmax(probs).item())
    confidence = float(probs[class_idx].item())
    label = clean_label(CLASS_NAMES[class_idx])

    per_class = {
        clean_label(name): float(probs[i].item())
        for i, name in enumerate(CLASS_NAMES)
    }

    return label, confidence, latency_ms, per_class


def predict_with_gradcam(image_rgb: np.ndarray):
    """Prediction + Grad-CAM for the latest inspection."""

    x = preprocess(image_rgb)
    target_layer = model.features[-1]

    state = {
        "activations": None,
        "gradients": None,
    }

    def save_activation(module, inputs, output):
        state["activations"] = output.detach()

    def save_gradient(module, grad_input, grad_output):
        state["gradients"] = grad_output[0].detach()

    forward_handle = target_layer.register_forward_hook(
        save_activation
    )
    backward_handle = target_layer.register_full_backward_hook(
        save_gradient
    )

    try:
        start = time.perf_counter()

        model.zero_grad()
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

        class_idx = int(torch.argmax(probs).item())
        confidence = float(probs[class_idx].item())

        logits[0, class_idx].backward()

        latency_ms = (time.perf_counter() - start) * 1000.0

        activations = state["activations"][0]
        gradients = state["gradients"][0]

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32,
        )

        for channel, weight in enumerate(weights):
            cam += weight * activations[channel]

        cam = F.relu(cam).numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        cam_resized = cv2.resize(
            cam,
            (image_rgb.shape[1], image_rgb.shape[0]),
        )

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized),
            cv2.COLORMAP_JET,
        )

        heatmap_rgb = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB,
        )

        overlay = cv2.addWeighted(
            image_rgb,
            0.60,
            heatmap_rgb,
            0.40,
            0,
        )

        label = clean_label(CLASS_NAMES[class_idx])

        per_class = {
            clean_label(name): float(probs[i].item())
            for i, name in enumerate(CLASS_NAMES)
        }

        return {
            "label": label,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "overlay": overlay,
            "per_class": per_class,
            "class_idx": class_idx,
        }

    finally:
        forward_handle.remove()
        backward_handle.remove()


def add_history(
    filename,
    label,
    confidence,
    latency_ms,
    uncertain,
):
    result = "Uncertain" if uncertain else label

    st.session_state.history.append(
        {
            "Time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "File": filename,
            "Result": result,
            "Confidence": round(confidence * 100, 1),
            "Latency (ms)": round(latency_ms, 2),
            "Backend": "CPU",
        }
    )


def history_dataframe():
    if not st.session_state.history:
        return pd.DataFrame(
            columns=[
                "Time",
                "File",
                "Result",
                "Confidence",
                "Latency (ms)",
                "Backend",
            ]
        )

    return pd.DataFrame(st.session_state.history)


def calculate_summary():
    df = history_dataframe()

    if df.empty:
        return {
            "total": 0,
            "good": 0,
            "defective": 0,
            "uncertain": 0,
        }

    return {
        "total": len(df),
        "good": int((df["Result"] == "Good").sum()),
        "defective": int((df["Result"] == "Defective").sum()),
        "uncertain": int((df["Result"] == "Uncertain").sum()),
    }


def result_box(label, confidence, uncertain):
    if uncertain:
        css = "result-uncertain"
        text = f"⚠️ UNCERTAIN · Best guess: {label}"
    elif label == "Good":
        css = "result-good"
        text = "✓ GOOD"
    else:
        css = "result-defective"
        text = "✕ DEFECTIVE"

    st.markdown(
        f"""
        <div class="result-card {css}">
            <div class="result-label">{text}</div>
            <div class="result-confidence">
                {confidence * 100:.1f}% confidence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    """
    <div class="smartqc-header">
        <div class="smartqc-title">
            SmartQC: On-Device Visual Quality Inspection
        </div>
        <div class="smartqc-subtitle">
            Computer vision quality inspection with edge AI.
        </div>
    </div>

    <div class="status-row">
        <span class="status-badge">● Web Demo · CPU</span>
    </div>

    <div class="demo-strip">
        <b>Web demo:</b> CPU inference on Streamlit servers.
        The original desktop application is designed for Snapdragon
        NPU deployment through the Qualcomm AI Hub / QNN path.
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("SmartQC")

    st.markdown(
        """
        **Visual Quality Inspection**

        - 🟢 Good
        - 🔴 Defective
        - 🟡 Uncertain
        """
    )

    st.divider()

    st.subheader("Model")
    st.write("MobileNetV2")
    st.write("Input: 224 × 224")
    st.write("Backend: CPU")
    st.write("Grad-CAM: Enabled")

    st.divider()

    st.subheader("Model Results")
    st.write("Validation Accuracy: **99.2%**")
    st.write("ROC-AUC: **0.9997**")
    st.write("Real-world test: **97.4%**")

    st.divider()

    st.markdown(f"[📂 GitHub Repository]({GITHUB_URL})")


# =============================================================================
# TABS
# =============================================================================

(
    inspection_tab,
    history_tab,
    trend_tab,
    explainability_tab,
    batch_tab,
) = st.tabs(
    [
        "🔍 Inspection",
        "📋 History",
        "📈 Trend",
        "🧠 Explainability",
        "📦 Batch",
    ]
)


# =============================================================================
# INSPECTION
# =============================================================================

with inspection_tab:

    input_source = st.radio(
        "Input source",
        ["Upload Image", "Camera"],
        horizontal=True,
    )

    uploaded_file = None

    if input_source == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload a bread image",
            type=["png", "jpg", "jpeg"],
            help="Upload a clear image of the product.",
        )
    else:
        uploaded_file = st.camera_input("Take a picture")

    threshold = st.slider(
        "Confidence threshold",
        min_value=50,
        max_value=99,
        value=70,
        step=1,
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        image_rgb = np.array(image)

        left, right = st.columns(
            [1.15, 0.85],
            gap="medium",
        )

        with left:
            st.markdown("### Input")

            st.image(
                image_rgb,
                caption="Inspection image",
                width="stretch",
            )

        with right:
            st.markdown("### Controls")

            if st.button(
                "🔍 Run Inspection",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Running inspection..."):
                    result = predict_with_gradcam(image_rgb)

                uncertain = (
                    result["confidence"] < threshold / 100.0
                )

                result["uncertain"] = uncertain

                st.session_state.last_result = {
                    **result,
                    "image": image_rgb,
                    "filename": uploaded_file.name,
                }

                add_history(
                    filename=uploaded_file.name,
                    label=result["label"],
                    confidence=result["confidence"],
                    latency_ms=result["latency_ms"],
                    uncertain=uncertain,
                )

            last = st.session_state.last_result

            if last is not None:

                result_box(
                    last["label"],
                    last["confidence"],
                    last["uncertain"],
                )

                m1, m2 = st.columns(2)

                with m1:
                    st.metric(
                        "Confidence",
                        f"{last['confidence'] * 100:.1f}%",
                    )

                with m2:
                    st.metric(
                        "Latency",
                        f"{last['latency_ms']:.1f} ms",
                    )

                st.caption("Backend: CPU (Streamlit server)")

                st.markdown("**Class probabilities**")

                prob_df = pd.DataFrame(
                    {
                        "Probability": last["per_class"]
                    }
                )

                st.bar_chart(
                    prob_df,
                    height=180,
                )

            else:
                st.info("Run an inspection to see the result.")


# =============================================================================
# HISTORY
# =============================================================================

with history_tab:

    df_history = history_dataframe()
    summary = calculate_summary()

    if summary["total"] > 0:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total", summary["total"])
        c2.metric("Good", summary["good"])
        c3.metric("Defective", summary["defective"])
        c4.metric("Uncertain", summary["uncertain"])

        st.dataframe(
            df_history,
            width="stretch",
            hide_index=True,
        )

        csv_history = df_history.to_csv(index=False)

        st.download_button(
            "⬇️ Export History CSV",
            data=csv_history,
            file_name="smartqc_history.csv",
            mime="text/csv",
        )

        if st.button("Clear Session History"):
            st.session_state.history = []
            st.rerun()

    else:
        st.info("No inspections have been performed yet.")


# =============================================================================
# TREND
# =============================================================================

with trend_tab:

    df_history = history_dataframe()

    if not df_history.empty:

        trend_df = df_history.copy()

        trend_df["Defective"] = (
            trend_df["Result"]
            .eq("Defective")
            .astype(int)
        )

        trend_df["Inspection #"] = range(
            1,
            len(trend_df) + 1,
        )

        window = min(10, len(trend_df))

        trend_df["Rolling Defect Rate"] = (
            trend_df["Defective"]
            .rolling(
                window=window,
                min_periods=1,
            )
            .mean()
            * 100
        )

        trend_chart = trend_df[
            ["Inspection #", "Rolling Defect Rate"]
        ].set_index("Inspection #")

        st.line_chart(
            trend_chart,
            height=260,
        )

        total = len(trend_df)
        defective = int(
            (trend_df["Result"] == "Defective").sum()
        )

        defect_rate = (
            defective / total
        ) * 100

        c1, c2, c3 = st.columns(3)

        c1.metric("Inspections", total)
        c2.metric("Defective", defective)
        c3.metric("Defect Rate", f"{defect_rate:.1f}%")

    else:
        st.info("Run a few inspections to generate the trend.")


# =============================================================================
# EXPLAINABILITY
# =============================================================================

with explainability_tab:

    last = st.session_state.last_result

    if last is not None:

        left, right = st.columns(
            2,
            gap="medium",
        )

        with left:
            st.markdown("### Original")
            st.image(
                last["image"],
                width="stretch",
            )

        with right:
            st.markdown("### Grad-CAM")
            st.image(
                last["overlay"],
                width="stretch",
            )

        c1, c2 = st.columns(2)

        c1.metric(
            "Prediction",
            last["label"],
        )

        c2.metric(
            "Confidence",
            f"{last['confidence'] * 100:.1f}%",
        )

        st.caption(
            "Grad-CAM highlights image regions that influenced "
            "the prediction. It is not a pixel-level defect mask."
        )

    else:
        st.info(
            "Run an inspection first. "
            "The latest inspection will appear here."
        )


# =============================================================================
# BATCH
# =============================================================================

with batch_tab:

    st.markdown("### Batch Inspection")

    st.caption(
        "Select a folder. SmartQC will inspect all supported images inside it."
    )

    batch_files = st.file_uploader(
        "Select inspection folder",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files="directory",
        key="batch_uploader",
        help=(
            "Select a folder. All supported PNG/JPG images "
            "inside the folder will be uploaded for inspection."
        ),
    )

    batch_threshold = st.slider(
        "Batch confidence threshold",
        min_value=50,
        max_value=99,
        value=70,
        step=1,
        key="batch_threshold",
    )

    if batch_files:

        st.success(
            f"{len(batch_files)} image(s) found in the selected folder."
        )

        if st.button(
            "📦 Run Batch Inspection",
            type="primary",
            use_container_width=True,
        ):

            batch_results = []
            progress = st.progress(0)
            status = st.empty()

            for index, file in enumerate(batch_files):

                status.write(
                    f"Inspecting {index + 1}/{len(batch_files)}: "
                    f"{file.name}"
                )

                try:
                    image = Image.open(file).convert("RGB")
                    image_rgb = np.array(image)

                    (
                        label,
                        confidence,
                        latency_ms,
                        per_class,
                    ) = predict_image(image_rgb)

                    uncertain = (
                        confidence
                        < batch_threshold / 100.0
                    )

                    result_label = (
                        "Uncertain"
                        if uncertain
                        else label
                    )

                    row = {
                        "File": file.name,
                        "Result": result_label,
                        "Confidence (%)": round(
                            confidence * 100,
                            1,
                        ),
                        "Latency (ms)": round(
                            latency_ms,
                            2,
                        ),
                        "Backend": "CPU",
                    }

                    batch_results.append(row)

                    add_history(
                        filename=file.name,
                        label=label,
                        confidence=confidence,
                        latency_ms=latency_ms,
                        uncertain=uncertain,
                    )

                except Exception:
                    batch_results.append(
                        {
                            "File": file.name,
                            "Result": "Error",
                            "Confidence (%)": 0.0,
                            "Latency (ms)": 0.0,
                            "Backend": "CPU",
                        }
                    )

                progress.progress(
                    (index + 1) / len(batch_files)
                )

            status.empty()

            st.session_state.batch_results = pd.DataFrame(
                batch_results
            )

            st.success(
                f"Completed inspection of {len(batch_results)} image(s)."
            )

    batch_df = st.session_state.batch_results

    if not batch_df.empty:

        st.divider()
        st.markdown("### Results")

        st.dataframe(
            batch_df,
            width="stretch",
            hide_index=True,
        )

        valid_results = batch_df[
            batch_df["Result"].isin(
                ["Good", "Defective", "Uncertain"]
            )
        ]

        total = len(valid_results)

        good = int(
            (valid_results["Result"] == "Good").sum()
        )

        defective = int(
            (valid_results["Result"] == "Defective").sum()
        )

        uncertain = int(
            (valid_results["Result"] == "Uncertain").sum()
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total", total)
        c2.metric("Good", good)
        c3.metric("Defective", defective)
        c4.metric("Uncertain", uncertain)

        csv_batch = batch_df.to_csv(index=False)

        st.download_button(
            "⬇️ Export Batch Results CSV",
            data=csv_batch,
            file_name="smartqc_batch_results.csv",
            mime="text/csv",
        )

    elif not batch_files:
        st.info("Select a folder above to start a batch inspection.")


# =============================================================================
# COMPACT FOOTER
# =============================================================================

st.divider()

st.markdown(
    f"""
    <div class="footer">
        <b>SmartQC</b> · Built by Samay Thakur ·
        Electronics & Instrumentation Engineering ·
        M.S. Ramaiah Institute of Technology ·
        <a href="{GITHUB_URL}" target="_blank">GitHub</a>
        · Public web demo uses CPU inference
    </div>
    """,
    unsafe_allow_html=True,
)
