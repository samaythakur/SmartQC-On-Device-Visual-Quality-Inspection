"""
SmartQC — Streamlit Live Demo

Browser-based demonstration of SmartQC.

Features:
- Single-image inspection
- Browser camera capture
- Confidence thresholding
- Good / Defective / Uncertain prediction
- Confidence scores
- Grad-CAM explainability
- Current-session inspection history
- Defect-rate trend
- Batch inspection
- CSV export

IMPORTANT:
This public web demo runs CPU inference on Streamlit Cloud.
It does NOT execute inference on a Snapdragon NPU.

The original SmartQC desktop application is the on-device deployment
version designed for Snapdragon NPU execution using the Qualcomm AI
Hub / QNN deployment path.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
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


# ============================================================================
# CONFIGURATION
# ============================================================================

WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "smartqc_weights.pt"
)

IMG_SIZE = 224

GITHUB_URL = (
    "https://github.com/samaythakur/"
    "SmartQC-On-Device-Visual-Quality-Inspection"
)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="SmartQC — Visual Quality Inspection",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown(
    """
    <style>

    /* Main page */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
    }

    /* Hide Streamlit sidebar */
    section[data-testid="stSidebar"] {
        display: none;
    }

    [data-testid="collapsedControl"] {
        display: none;
    }

    /* Result cards */
    .result-card {
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.5rem 0 1rem 0;
        border: 1px solid rgba(128,128,128,0.25);
    }

    .result-label {
        font-size: 1.7rem;
        font-weight: 700;
    }

    .result-confidence {
        font-size: 1rem;
        opacity: 0.8;
    }

    /* Metric cards */
    .metric-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
    }

    .metric-label {
        font-size: 0.8rem;
        opacity: 0.7;
    }

    /* Info box */
    .demo-note {
        padding: 0.8rem 1rem;
        border-left: 4px solid #4c9aff;
        background: rgba(76,154,255,0.08);
        border-radius: 5px;
        margin: 1rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        opacity: 0.6;
        padding-top: 2rem;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================================
# SESSION STATE
# ============================================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model():

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Model weights not found at: {WEIGHTS_PATH}"
        )

    checkpoint = torch.load(
        WEIGHTS_PATH,
        map_location="cpu"
    )

    class_names = checkpoint["class_names"]

    model = models.mobilenet_v2()

    model.classifier[1] = torch.nn.Linear(
        model.last_channel,
        len(class_names)
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model.eval()

    return model, class_names


model, CLASS_NAMES = load_model()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_label(raw_label: str) -> str:
    """
    Convert dataset class names into user-facing labels.
    """

    low = raw_label.lower()

    if "good" in low or "fresh" in low:
        return "Good"

    return "Defective"


def preprocess(image_rgb: np.ndarray) -> torch.Tensor:
    """
    Convert RGB image into MobileNetV2 input tensor.
    """

    img = cv2.resize(
        image_rgb,
        (IMG_SIZE, IMG_SIZE)
    )

    img = img.astype(
        np.float32
    ) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    img = (
        img - mean
    ) / std

    img = np.transpose(
        img,
        (2, 0, 1)
    )

    img = img[None, ...].astype(
        np.float32
    )

    return torch.from_numpy(img)


def predict_image(image_rgb: np.ndarray):
    """
    Standard CPU prediction without Grad-CAM.

    Used for batch inspection.
    """

    x = preprocess(image_rgb)

    start = time.perf_counter()

    with torch.no_grad():

        logits = model(x)

        probs = F.softmax(
            logits,
            dim=1
        )[0]

    latency_ms = (
        time.perf_counter() - start
    ) * 1000.0

    class_idx = int(
        torch.argmax(probs).item()
    )

    confidence = float(
        probs[class_idx].item()
    )

    label = clean_label(
        CLASS_NAMES[class_idx]
    )

    per_class = {
        clean_label(name): float(probs[i].item())
        for i, name in enumerate(CLASS_NAMES)
    }

    return (
        label,
        confidence,
        latency_ms,
        per_class
    )


def predict_with_gradcam(
    image_rgb: np.ndarray
):
    """
    Prediction + Grad-CAM.

    Grad-CAM hooks are registered temporarily so that Streamlit reruns
    do not accumulate duplicate hooks.
    """

    x = preprocess(image_rgb)

    target_layer = model.features[-1]

    state = {
        "activations": None,
        "gradients": None
    }

    def save_activation(
        module,
        input,
        output
    ):
        state["activations"] = output.detach()

    def save_gradient(
        module,
        grad_input,
        grad_output
    ):
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

        logits[0, class_idx].backward()

        latency_ms = (
            time.perf_counter() - start
        ) * 1000.0

        activations = state["activations"][0]

        gradients = state["gradients"][0]

        # Average gradients across spatial dimensions
        weights = gradients.mean(
            dim=(1, 2)
        )

        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32
        )

        for channel, weight in enumerate(weights):

            cam += (
                weight *
                activations[channel]
            )

        cam = F.relu(
            cam
        ).numpy()

        if cam.max() > 0:

            cam = (
                cam /
                cam.max()
            )

        cam_resized = cv2.resize(
            cam,
            (
                image_rgb.shape[1],
                image_rgb.shape[0]
            )
        )

        heatmap = cv2.applyColorMap(
            np.uint8(
                255 * cam_resized
            ),
            cv2.COLORMAP_JET
        )

        heatmap_rgb = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        overlay = cv2.addWeighted(
            image_rgb,
            0.60,
            heatmap_rgb,
            0.40,
            0
        )

        label = clean_label(
            CLASS_NAMES[class_idx]
        )

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
            "class_idx": class_idx
        }

    finally:

        forward_handle.remove()
        backward_handle.remove()


def add_history(
    filename,
    label,
    confidence,
    latency_ms,
    uncertain
):
    """
    Add an inspection to the current browser session history.
    """

    if uncertain:
        result = "Uncertain"
    else:
        result = label

    entry = {
        "Time": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "File": filename,
        "Result": result,
        "Confidence": round(
            confidence * 100,
            1
        ),
        "Latency (ms)": round(
            latency_ms,
            2
        ),
        "Backend": "CPU"
    }

    st.session_state.history.append(
        entry
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
                "Backend"
            ]
        )

    return pd.DataFrame(
        st.session_state.history
    )


def calculate_summary():

    df = history_dataframe()

    if df.empty:

        return {
            "total": 0,
            "good": 0,
            "defective": 0,
            "uncertain": 0
        }

    return {
        "total": len(df),
        "good": int(
            (df["Result"] == "Good").sum()
        ),
        "defective": int(
            (df["Result"] == "Defective").sum()
        ),
        "uncertain": int(
            (df["Result"] == "Uncertain").sum()
        )
    }


# ============================================================================
# HEADER
# ============================================================================

st.title("SmartQC: On-Device Visual Quality Inspection")


# ============================================================================
# ============================================================================
# TABS

# ============================================================================

(
    inspection_tab,
    history_tab,
    trend_tab,
    explainability_tab,
    batch_tab
) = st.tabs(
    [
        "Inspection",
        "History",
        "Trend",
        "Explainability",
        "Batch"
    ]
)


# ============================================================================
# INSPECTION TAB
# ============================================================================

with inspection_tab:

    st.subheader("Inspection")

    input_source = st.radio(
        "Input source",
        [
            "Upload Image",
            "Camera"
        ],
        horizontal=True
    )

    uploaded_file = None

    # ------------------------------------------------------------------------
    # Upload image
    # ------------------------------------------------------------------------

    if input_source == "Upload Image":

        uploaded_file = st.file_uploader(
            "Upload a bread image",
            type=[
                "png",
                "jpg",
                "jpeg"
            ],
            help="Upload a clear image of the product."
        )

    # ------------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------------

    else:

        uploaded_file = st.camera_input(
            "Take a picture"
        )

    threshold = st.slider(
        "Confidence threshold (%)",
        min_value=50,
        max_value=99,
        value=70,
        step=1
    )

    st.caption(
        "Predictions below this threshold are flagged as "
        "**UNCERTAIN** instead of being forced into Good/Defective."
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_rgb = np.array(
            image
        )

        st.divider()

        left, right = st.columns(
            [1.1, 1]
        )

        # --------------------------------------------------------------------
        # Image
        # --------------------------------------------------------------------

        with left:

            st.markdown(
                "### Input Image"
            )

            st.image(
                image_rgb,
                caption="Inspection image",
                width="stretch"
            )

        # --------------------------------------------------------------------
        # Run inspection
        # --------------------------------------------------------------------

        with right:

            st.markdown(
                "### Inspection Controls"
            )

            run_inspection = st.button(
                "Run Inspection",
                type="primary",
                use_container_width=True
            )

            if run_inspection:

                with st.spinner(
                    "Running SmartQC inspection..."
                ):

                    result = predict_with_gradcam(
                        image_rgb
                    )

                label = result["label"]
                confidence = result["confidence"]
                latency_ms = result["latency_ms"]

                uncertain = (
                    confidence <
                    threshold / 100.0
                )

                result["uncertain"] = uncertain

                # Save result for other tabs
                st.session_state.last_result = {
                    **result,
                    "image": image_rgb,
                    "filename": uploaded_file.name
                }

                # Add history
                add_history(
                    filename=uploaded_file.name,
                    label=label,
                    confidence=confidence,
                    latency_ms=latency_ms,
                    uncertain=uncertain
                )

            # ---------------------------------------------------------------
            # Display latest result
            # ---------------------------------------------------------------

            last = st.session_state.last_result

            if last is not None:

                label = last["label"]
                confidence = last["confidence"]
                latency_ms = last["latency_ms"]
                uncertain = last["uncertain"]

                if uncertain:

                    st.warning(
                        f"UNCERTAIN — Best guess: "
                        f"{label} "
                        f"({confidence * 100:.1f}%)"
                    )

                elif label == "Good":

                    st.success(
                        f"GOOD — "
                        f"{confidence * 100:.1f}% confidence"
                    )

                else:

                    st.error(
                        f"DEFECTIVE — "
                        f"{confidence * 100:.1f}% confidence"
                    )

                metric1, metric2 = st.columns(2)

                with metric1:

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.1f}%"
                    )

                with metric2:

                    st.metric(
                        "Latency",
                        f"{latency_ms:.1f} ms"
                    )

                st.caption(
                    "Backend: CPU (Streamlit server)"
                )

                st.markdown(
                    "### Class Probabilities"
                )

                probability_df = pd.DataFrame(
                    {
                        "Probability": last[
                            "per_class"
                        ]
                    }
                )

                st.bar_chart(
                    probability_df
                )

            else:

                st.info(
                    "Upload an image and click "
                    "**Run Inspection**."
                )


# ============================================================================
# HISTORY TAB
# ============================================================================

with history_tab:

    st.subheader(
        "Inspection History"
    )

    st.caption(
        "History shown here is for the current browser session."
    )

    df_history = history_dataframe()

    summary = calculate_summary()

    if summary["total"] > 0:

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Total",
                summary["total"]
            )

        with c2:
            st.metric(
                "Good",
                summary["good"]
            )

        with c3:
            st.metric(
                "Defective",
                summary["defective"]
            )

        with c4:
            st.metric(
                "Uncertain",
                summary["uncertain"]
            )

        st.divider()

        st.dataframe(
            df_history,
            use_container_width=True,
            hide_index=True
        )

        csv_history = df_history.to_csv(
            index=False
        )

        st.download_button(
            "Export History CSV",
            data=csv_history,
            file_name="smartqc_history.csv",
            mime="text/csv"
        )

        if st.button(
            "Clear Current Session History"
        ):

            st.session_state.history = []

            st.rerun()

    else:

        st.info(
            "No inspections have been performed yet."
        )


# ============================================================================
# TREND TAB
# ============================================================================

with trend_tab:

    st.subheader(
        "Defect Rate Trend"
    )

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
            len(trend_df) + 1
        )

        # Rolling defect rate
        window = min(
            10,
            len(trend_df)
        )

        trend_df["Rolling Defect Rate"] = (
            trend_df["Defective"]
            .rolling(
                window=window,
                min_periods=1
            )
            .mean()
        )

        trend_chart = trend_df[
            [
                "Inspection #",
                "Rolling Defect Rate"
            ]
        ].set_index(
            "Inspection #"
        )

        st.line_chart(
            trend_chart
        )

        st.caption(
            "Rolling defect rate based on the inspections "
            "performed during the current session."
        )

        st.divider()

        total = len(trend_df)

        defective = int(
            (
                trend_df["Result"]
                == "Defective"
            ).sum()
        )

        defect_rate = (
            defective / total
        ) * 100

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Total Inspections",
                total
            )

        with c2:
            st.metric(
                "Defective",
                defective
            )

        with c3:
            st.metric(
                "Defect Rate",
                f"{defect_rate:.1f}%"
            )

    else:

        st.info(
            "Run a few inspections to generate the trend."
        )


# ============================================================================
# EXPLAINABILITY TAB
# ============================================================================

with explainability_tab:

    st.subheader(
        "Grad-CAM Explainability"
    )

    st.caption(
        "Grad-CAM highlights image regions that influenced "
        "the model's prediction."
    )

    last = st.session_state.last_result

    if last is not None:

        left, right = st.columns(2)

        with left:

            st.markdown(
                "### Original Image"
            )

            st.image(
                last["image"],
                width="stretch"
            )

        with right:

            st.markdown(
                "### Grad-CAM Heatmap"
            )

            st.image(
                last["overlay"],
                width="stretch"
            )

        st.divider()

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Prediction",
                last["label"]
            )

        with c2:

            st.metric(
                "Confidence",
                f"{last['confidence'] * 100:.1f}%"
            )

        st.info(
            "Warmer regions such as red/yellow indicate "
            "areas that contributed more strongly to the "
            "model's decision. Grad-CAM is an explanation "
            "of model attention, not a pixel-level defect mask."
        )

    else:

        st.info(
            "Run an inspection first. "
            "The latest inspection will appear here."
        )


# ============================================================================
# BATCH INSPECTION TAB
# ============================================================================

with batch_tab:

    st.subheader(
        "Batch Inspection"
    )

    st.caption(
        "Select a folder. SmartQC will inspect all supported images inside it."
    )

    batch_files = st.file_uploader(
        "Select inspection folder",
        type=[
            "png",
            "jpg",
            "jpeg"
        ],
        accept_multiple_files="directory",
        key="batch_uploader",
        help="Select a folder. SmartQC will inspect all supported images inside it."
    )

    batch_threshold = st.slider(
        "Batch confidence threshold (%)",
        min_value=50,
        max_value=99,
        value=70,
        step=1,
        key="batch_threshold"
    )

    if batch_files:

        st.write(
            f"**{len(batch_files)} image(s) found in the selected folder.**"
        )

        if st.button(
            "Run Batch Inspection",
            type="primary"
        ):

            batch_results = []

            progress = st.progress(
                0
            )

            status = st.empty()

            for index, file in enumerate(
                batch_files
            ):

                status.write(
                    f"Inspecting "
                    f"{index + 1}/{len(batch_files)}: "
                    f"{file.name}"
                )

                try:

                    image = Image.open(
                        file
                    ).convert("RGB")

                    image_rgb = np.array(
                        image
                    )

                    (
                        label,
                        confidence,
                        latency_ms,
                        per_class
                    ) = predict_image(
                        image_rgb
                    )

                    uncertain = (
                        confidence <
                        batch_threshold / 100.0
                    )

                    if uncertain:
                        result_label = "Uncertain"
                    else:
                        result_label = label

                    row = {
                        "File": file.name,
                        "Result": result_label,
                        "Confidence (%)": round(
                            confidence * 100,
                            1
                        ),
                        "Latency (ms)": round(
                            latency_ms,
                            2
                        ),
                        "Backend": "CPU"
                    }

                    batch_results.append(
                        row
                    )

                    # Add to current history
                    add_history(
                        filename=file.name,
                        label=label,
                        confidence=confidence,
                        latency_ms=latency_ms,
                        uncertain=uncertain
                    )

                except Exception as e:

                    batch_results.append(
                        {
                            "File": file.name,
                            "Result": "Error",
                            "Confidence (%)": 0.0,
                            "Latency (ms)": 0.0,
                            "Backend": "CPU"
                        }
                    )

                progress.progress(
                    (index + 1) /
                    len(batch_files)
                )

            status.empty()

            batch_df = pd.DataFrame(
                batch_results
            )

            st.session_state.batch_results = (
                batch_df
            )

            st.success(
                f"Completed inspection of "
                f"{len(batch_results)} image(s)."
            )

    # ------------------------------------------------------------------------
    # Display batch results
    # ------------------------------------------------------------------------

    if (
        "batch_results"
        in st.session_state
        and not st.session_state.batch_results.empty
    ):

        batch_df = st.session_state.batch_results

        st.divider()

        st.markdown(
            "### Batch Results"
        )

        st.dataframe(
            batch_df,
            use_container_width=True,
            hide_index=True
        )

        # Summary
        valid_results = batch_df[
            batch_df["Result"].isin(
                [
                    "Good",
                    "Defective",
                    "Uncertain"
                ]
            )
        ]

        total = len(
            valid_results
        )

        good = int(
            (
                valid_results["Result"]
                == "Good"
            ).sum()
        )

        defective = int(
            (
                valid_results["Result"]
                == "Defective"
            ).sum()
        )

        uncertain = int(
            (
                valid_results["Result"]
                == "Uncertain"
            ).sum()
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Total",
                total
            )

        with c2:
            st.metric(
                "Good",
                good
            )

        with c3:
            st.metric(
                "Defective",
                defective
            )

        with c4:
            st.metric(
                "Uncertain",
                uncertain
            )

        csv_batch = batch_df.to_csv(
            index=False
        )

        st.download_button(
            "Export Batch Results to CSV",
            data=csv_batch,
            file_name="smartqc_batch_results.csv",
            mime="text/csv"
        )

    else:

        if not batch_files:

            st.info(
                "Select a folder above to start a batch inspection."
            )


# ============================================================================
# FOOTER
# ============================================================================

st.divider()

st.markdown(
    f"""
    <div class="footer">
    <b>SmartQC</b> · Built by Samay Thakur ·
    <a href="{GITHUB_URL}" target="_blank">GitHub</a> ·
    Public web demo uses CPU inference · Snapdragon NPU deployment belongs to the original desktop application.
    </div>
    """,
    unsafe_allow_html=True
)