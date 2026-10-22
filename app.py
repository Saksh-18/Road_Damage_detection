import sys

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

# YOLOv7 repo must sit next to this file as ./yolov7
sys.path.insert(0, "yolov7")

# Newer PyTorch defaults to weights_only=True, which breaks YOLOv7 checkpoints.
_orig_load = torch.load


def _safe_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)  # type: ignore[call-arg]


torch.load = _safe_load  # type: ignore[assignment]


from models.experimental import attempt_load  # noqa: E402
from utils.datasets import letterbox  # noqa: E402
from utils.general import non_max_suppression, scale_coords  # noqa: E402

WEIGHTS = "best.pt"
# Order MUST match the names list in your rdd2022.yaml
NAMES = ["D00", "D10", "D20", "D40"]
DESCRIPTIONS = {
    "D00": "Longitudinal crack",
    "D10": "Transverse crack",
    "D20": "Alligator crack",
    "D40": "Pothole",
}
COLORS = {  # BGR for OpenCV
    "D00": (0, 200, 0),
    "D10": (255, 150, 0),
    "D20": (0, 0, 255),
    "D40": (0, 200, 255),
}

st.set_page_config(page_title="Road Damage Detector", page_icon="🛣️", layout="wide")


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = attempt_load(WEIGHTS, map_location=device).eval()
    return model, device


def detect(image_rgb, model, device, imgsz, conf, iou):
    im0 = image_rgb[:, :, ::-1].copy()  # RGB -> BGR
    stride = int(model.stride.max())
    im = letterbox(im0, imgsz, stride=stride)[0]
    im = np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))  # BGR->RGB, HWC->CHW
    im = torch.from_numpy(im).to(device).float() / 255.0

    with torch.no_grad():
        pred = model(im[None])[0]
        pred = non_max_suppression(pred, conf, iou)[0]

    rows = []
    thickness = max(2, im0.shape[1] // 400)
    font_scale = max(0.5, im0.shape[1] / 1600)
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(im.shape[1:], pred[:, :4], im0.shape).round()
        for *xyxy, score, cls in pred.cpu().numpy():
            x1, y1, x2, y2 = map(int, xyxy)
            name = NAMES[int(cls)]
            color = COLORS[name]
            cv2.rectangle(im0, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                im0,
                f"{name} {score:.2f}",
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                thickness,
            )
            rows.append(
                {
                    "Class": name,
                    "Type": DESCRIPTIONS[name],
                    "Confidence": round(float(score), 3),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                }
            )
    return im0[:, :, ::-1], pd.DataFrame(rows)


# ---------------- UI ----------------
st.title("🛣️ Road Damage Detector")
st.caption("YOLOv7 trained on RDD2022 (India + China Drone)")

with st.sidebar:
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05)
    iou = st.slider("NMS IoU threshold", 0.10, 0.90, 0.45, 0.05)
    imgsz = st.selectbox(
        "Inference size",
        [640, 960, 1024, 1280],
        index=0,
        help="The model was trained at 640. Larger sizes can help with thin cracks "
        "on high-res drone images, but may behave differently.",
    )
    st.markdown("**Classes**")
    for k, v in DESCRIPTIONS.items():
        st.markdown(f"- `{k}`: {v}")

try:
    model, device = load_model()
except Exception as e:
    st.error(f"Could not load `{WEIGHTS}`: {e}")
    st.stop()

st.sidebar.caption(f"Running on: {device}")

uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = np.array(Image.open(uploaded).convert("RGB"))
    with st.spinner("Detecting damage..."):
        result, df = detect(image, model, device, imgsz, conf, iou)

    left, right = st.columns(2)
    left.subheader("Original")
    left.image(image, width='stretch')
    right.subheader("Detections")
    right.image(result, width='stretch')

    st.divider()
    if df.empty:
        st.info("No damage detected at this confidence threshold. Try lowering it.")
    else:
        st.subheader(f"{len(df)} detection(s)")
        counts = df["Class"].value_counts().reindex(NAMES, fill_value=0)
        cols = st.columns(len(NAMES))
        for col, name in zip(cols, NAMES):
            col.metric(f"{name} ({DESCRIPTIONS[name]})", int(counts[name]))
        st.dataframe(df, width='stretch', hide_index=True)
        st.download_button(
            "Download detections (CSV)",
            df.to_csv(index=False).encode(),
            "detections.csv",
            "text/csv",
        )
else:
    st.info("Upload an image to get started.")
