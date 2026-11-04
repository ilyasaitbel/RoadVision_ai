"""
RoadVision AI (VeyloraX) — Premium SaaS Test Playground
=======================================================
Playground UI for the vehicle detection models: image inference,
full video processing and live camera capture.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
import tempfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ==================================================
# 1. Configuration & Constants
# ==================================================
st.set_page_config(
    page_title="RoadVision AI | Test Playground",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "src" / "models" / "vehicle_detection.pt"

CUSTOM_CLASSES = {
    "ambulance", "army vehicle", "auto rickshaw", "bicycle", "bus",
    "garbagevan", "human hauler", "minibus", "minivan", "motorbike",
    "pickup", "policecar", "rickshaw", "scooter", "suv", "taxi",
    "three wheelers -CNG-", "truck", "van", "wheelbarrow",
}

COCO_VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 6: "train", 7: "truck"}

CLASS_ICONS = {
    "car": "🚗", "motorcycle": "🏍️", "bus": "🚌", "truck": "🚚",
    "bicycle": "🚲", "train": "🚆", "ambulance": "🚑", "policecar": "🚓",
    "taxi": "🚕", "suv": "🚙", "pickup": "🛻", "van": "🚐", "minibus": "🚐",
    "minivan": "🚐", "auto rickshaw": "🛺", "rickshaw": "🛺",
    "three wheelers -CNG-": "🛺", "motorbike": "🏍️", "scooter": "🛵",
    "human hauler": "🚐", "garbagevan": "🚛", "army vehicle": "🪖", "wheelbarrow": "🛒",
}

# ==================================================
# 2. Premium SaaS CSS Theme
# ==================================================
st.markdown("""
<style>
    /* Global Theme */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #152238 0%, #090e17 60%, #05080f 100%);
        color: #e2e8f0;
    }

    /* Hide default Streamlit elements for a cleaner look */
    #MainMenu, footer, header {visibility: hidden;}

    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 2rem 0 3rem 0;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        max-width: 600px;
        margin: 0 auto;
    }

    /* Tabs Styling (SaaS look) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(30, 41, 59, 0.5);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8;
        padding: 0 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(79, 172, 254, 0.15) !important;
        color: #4facfe !important;
        border: 1px solid rgba(79, 172, 254, 0.3) !important;
    }

    /* Cards & Metrics */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 15px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stMetricValue"] { color: #4facfe; font-weight: 700; }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: none;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        color: #fff;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 172, 254, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# 3. Core Logic & Model Loading
# ==================================================
@st.cache_resource(show_spinner="Initializing AI Core...")
def load_model(path: str) -> YOLO:
    return YOLO(path)

@st.cache_resource
def resolve_model_mode(_model: YOLO) -> dict:
    names = _model.names
    name_values = {str(v).lower() for v in names.values()}
    is_custom = len(CUSTOM_CLASSES & name_values) >= 5

    if is_custom:
        selectable = {i: names[i] for i in names}
    else:
        vehicle_names = {v.lower() for v in COCO_VEHICLES.values()}
        selectable = {i: names[i] for i in names if str(names[i]).lower() in vehicle_names}
    return {"is_custom": is_custom, "selectable": selectable}

if not MODEL_PATH.exists():
    st.error(f"**Model not found:** `{MODEL_PATH}`")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
    mode = resolve_model_mode(model)
except Exception as e:
    st.error(f"**Failed to load the model:** `{e}`")
    st.stop()

SELECTABLE = mode["selectable"]

def run_detection(frame, conf: float, iou: float, allowed_ids: list[int]):
    kwargs = dict(conf=conf, iou=iou, verbose=False)
    if allowed_ids:
        kwargs["classes"] = allowed_ids
    return model.predict(source=frame, **kwargs)[0]

def detections_table(result) -> pd.DataFrame:
    rows = []
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        rows.append({
            "Class": f"{CLASS_ICONS.get(str(result.names[cls_id]).lower(), '🚗')} {result.names[cls_id]}",
            "Confidence": round(float(box.conf[0].item()), 3),
            "Box": f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
        })
    if not rows: return pd.DataFrame(columns=["Class", "Confidence", "Box"])
    return pd.DataFrame(rows).sort_values("Confidence", ascending=False, ignore_index=True)

# ==================================================
# 4. Sidebar: Settings Engine
# ==================================================
with st.sidebar:
    st.markdown("### ⚙️ Detection Engine")
    st.caption("Fine-tune model parameters before testing.")

    confidence = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("NMS IoU Threshold", 0.10, 0.90, 0.45, 0.05)

    st.divider()
    allowed_names = st.multiselect(
        "Target Classes",
        options=sorted(SELECTABLE.values()),
        default=sorted(SELECTABLE.values()),
    )
    allowed_ids = [i for i, n in SELECTABLE.items() if n in allowed_names]

    st.divider()
    st.success(f"**Status:** Online\n\n**Mode:** {'Custom' if mode['is_custom'] else 'COCO Baseline'}")

# ==================================================
# 5. Main UI: The Playground
# ==================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">RoadVision Playground</div>
    <div class="hero-subtitle">Test our enterprise-grade vehicle detection models instantly across images, video streams, and live captures.</div>
</div>
""", unsafe_allow_html=True)

tab_img, tab_vid, tab_cam = st.tabs(["🖼️ Image Inference", "🎬 Video Processing", "📸 Live Capture"])

# --- IMAGE TEST ---
with tab_img:
    st.write("") # Spacer
    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded = st.file_uploader("Upload Test Image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            arr = np.ascontiguousarray(np.array(image))
            st.image(arr, caption="Source Image", use_container_width=True)
            run_btn = st.button("🚀 Run Inference", use_container_width=True)

    with col2:
        if uploaded and run_btn:
            with st.spinner("Processing through neural network..."):
                result = run_detection(arr, confidence, iou, allowed_ids)

                # persist results so they survive Streamlit reruns (slider moves, tab switches)
                st.session_state.img_result_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
                st.session_state.img_table = detections_table(result)
                st.session_state.img_speed = result.speed.get("inference", 0.0)

        # render persisted results (also right after the button click)
        if uploaded and "img_result_rgb" in st.session_state:
            result_rgb = st.session_state.img_result_rgb

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Objects", len(st.session_state.img_table))
            m2.metric("Inference Time", f"{st.session_state.img_speed:.1f}ms")
            m3.metric("Classes Detected",
                      st.session_state.img_table["Class"].nunique())

            st.image(result_rgb, use_container_width=True, clamp=True)

            st.markdown("##### Detailed Detections")
            st.dataframe(st.session_state.img_table, use_container_width=True, hide_index=True)
        elif not uploaded:
            st.info("👈 Upload an image to start the test.")

# --- VIDEO TEST ---
with tab_vid:
    st.write("")
    v_col1, v_col2 = st.columns([1, 2])

    with v_col1:
        uploaded_video = st.file_uploader("Upload Video Segment", type=["mp4", "mov", "avi", "mkv"])
        if uploaded_video:
            st.video(uploaded_video)
            run_vid_btn = st.button("🚀 Process Video", use_container_width=True)

    with v_col2:
        if uploaded_video and run_vid_btn:
            input_path = output_path = None
            cap = writer = None
            try:
                suffix = os.path.splitext(uploaded_video.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                    f.write(uploaded_video.getvalue())
                    input_path = f.name

                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    raise ValueError("Could not open the video file.")

                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if fps <= 0 or width <= 0 or height <= 0:
                    raise ValueError("Invalid video metadata.")

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                    output_path = f.name

                writer = cv2.VideoWriter(
                    output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise ValueError("Could not create the output video.")

                progress = st.progress(0.0, text="Starting…")
                preview_slot = st.empty()
                preview_every = max(total // 20, 1)

                frame_idx = 0
                total_det = 0
                per_class: dict[str, int] = {}

                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    result = run_detection(frame, confidence, iou, allowed_ids)
                    total_det += len(result.boxes)
                    for box in result.boxes:
                        name = str(result.names[int(box.cls[0].item())])
                        per_class[name] = per_class.get(name, 0) + 1

                    writer.write(result.plot())  # plot() returns BGR

                    frame_idx += 1
                    if frame_idx % preview_every == 0:
                        preview_slot.image(
                            cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB),
                            caption=f"Frame {frame_idx}/{total}",
                            use_container_width=True,
                        )
                    progress.progress(min(frame_idx / max(total, 1), 1.0),
                                      text=f"Processing frame {frame_idx}/{total}")

                progress.progress(1.0, text="Done ✅")
                st.success(
                    f"**{frame_idx}** frames processed · "
                    f"**{total_det}** total detections · "
                    f"≈{total_det / max(frame_idx, 1):.2f} vehicles/frame"
                )

                st.markdown("##### Processed Video")
                st.video(output_path)

                if per_class:
                    with st.expander("📊 Class breakdown across the video"):
                        st.bar_chart(pd.Series(per_class).sort_values(ascending=False))

                with open(output_path, "rb") as vf:
                    st.download_button(
                        "⬇️ Download processed video (MP4)",
                        data=vf.read(),
                        file_name="roadvision_result.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Processing error: {e}")
            finally:
                if cap is not None:
                    cap.release()
                if writer is not None:
                    writer.release()
                if input_path and os.path.exists(input_path):
                    os.remove(input_path)
        elif not uploaded_video:
            st.info("👈 Upload a video file to test batch processing.")

# --- LIVE CAMERA ---
with tab_cam:
    st.write("")
    c_col1, c_col2 = st.columns([1, 2])

    with c_col1:
        shot = st.camera_input("Initialize Camera")

    with c_col2:
        if shot:
            image = Image.open(shot).convert("RGB")
            arr = np.ascontiguousarray(np.array(image))
            with st.spinner("Detecting..."):
                result = run_detection(arr, confidence, iou, allowed_ids)
                res_img = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

                st.image(res_img, use_container_width=True)
                st.dataframe(detections_table(result), use_container_width=True, hide_index=True)

                buf = BytesIO()
                Image.fromarray(res_img).save(buf, format="PNG")
                st.download_button(
                    "⬇️ Download annotated snapshot",
                    data=buf.getvalue(),
                    file_name="roadvision_webcam.png",
                    mime="image/png",
                )
        else:
            st.info("👈 Take a snapshot to run live detection.")
