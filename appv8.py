# -*- coding: utf-8 -*-
"""
Road Damage Intelligence Platform (YOLOv7 / RDD2022)
Streamlit Application with Enhanced Visibility, Balanced Alignment,
Original CSV Sheet Compatibility, and Transparent Cost/Metric Calculations.
"""

import io
import sys
from datetime import datetime

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

from pathlib import Path

# ── YOLOv7 Module Path & PyTorch Safe Load ────────────────────────────────────
YOLOV7_DIR = Path(__file__).resolve().parent / "yolov7"
if str(YOLOV7_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV7_DIR))
sys.path.insert(0, "yolov7")
_orig_load = torch.load


def _safe_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)  # type: ignore[call-arg]


torch.load = _safe_load  # type: ignore[assignment]


from models.experimental import attempt_load  # noqa: E402
from utils.datasets import letterbox  # noqa: E402
from utils.general import non_max_suppression, scale_coords  # noqa: E402

# ── Constants & Engineering Benchmarks ────────────────────────────────────────
WEIGHTS = "best.pt"
NAMES = ["D00", "D10", "D20", "D40"]
DESCRIPTIONS = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
}

# Severity Weight (1 to 4 scale based on structural impact)
SEVERITY_WEIGHT = {
    "D00": 1,  # Superficial linear fatigue
    "D10": 2,  # Joint / thermal contraction
    "D20": 3,  # Severe structural base fatigue
    "D40": 4,  # High hazard depression / hole
}

# Unit repair cost estimates in INR (based on standard municipal PWD highway maintenance schedule)
REPAIR_COST = {
    "D00": 500,   # Crack sealing with rubberized asphalt emulsion (per occurrence)
    "D10": 800,   # Routing, cleaning & hot-pour elastomeric sealant (per occurrence)
    "D20": 2000,  # Full-depth cutout, base re-compaction & hot-mix asphalt patching
    "D40": 3500,  # Edge squaring, tack coat, dense bitumen infill & mechanical compaction
}

PRIORITY_MAP = {
    "D40": ("Immediate Action", "#ff5a5a", "Direct hazard to tires and traffic safety"),
    "D20": ("High Priority", "#ff8a50", "Base failure; will develop into major potholes if untreated"),
    "D10": ("Schedule Soon", "#ffb236", "Water ingress risk through transverse joints"),
    "D00": ("Routine Maintenance", "#48c78e", "Surface-level sealing to prevent moisture seepage"),
}

COLORS_RGB = {
    "D00": (72, 199, 142),   # Bright Green
    "D10": (255, 178, 54),   # Bright Amber
    "D20": (255, 90, 90),    # Vibrant Coral Red
    "D40": (168, 120, 240),  # Bright Purple
}
COLORS_BGR = {k: (v[2], v[1], v[0]) for k, v in COLORS_RGB.items()}


def clean_html(raw_html: str) -> str:
    """Strips leading and trailing spaces from each line to prevent Streamlit Markdown from treating HTML as code."""
    return "\n".join(line.strip() for line in raw_html.strip().splitlines() if line.strip())



# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Road Damage Intelligence Platform",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── High Contrast & High Visibility CSS ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Main app background and font */
.stApp {
    background-color: #0b0f17 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #f0f6fc !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

#MainMenu, footer {
    visibility: hidden;
}

/* Custom scrollbars */
::-webkit-scrollbar {
    width: 7px;
    height: 7px;
}
::-webkit-scrollbar-track {
    background: #0d121c;
}
::-webkit-scrollbar-thumb {
    background: #303a48;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #48566a;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #111622 !important;
    border-right: 1px solid #232d3d !important;
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #f0f6fc !important;
}

/* Hero Section */
.hero-card {
    background: linear-gradient(135deg, #131b29 0%, #1a2538 50%, #101724 100%);
    border: 1px solid #2d3b4e;
    border-radius: 14px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.5rem;
    position: relative;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}
.hero-card h1 {
    font-size: 2.1rem;
    font-weight: 800;
    color: #ffffff !important;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.02em;
}
.hero-card p {
    font-size: 0.95rem;
    color: #c9d1d9 !important;
    margin: 0;
    line-height: 1.5;
}
.badge-pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 16px;
    background: rgba(88, 166, 255, 0.16);
    border: 1px solid rgba(88, 166, 255, 0.35);
    color: #79c0ff !important;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Balanced KPI Grid */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.kpi-box {
    background: #141b27;
    border: 1px solid #283546;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    transition: transform 0.15s ease, border-color 0.15s ease;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}
.kpi-box:hover {
    border-color: #58a6ff;
    transform: translateY(-2px);
}
.kpi-box .accent-stripe {
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    border-radius: 12px 0 0 12px;
}
.kpi-box .label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #a5b4c6 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.35rem;
}
.kpi-box .value {
    font-size: 1.95rem;
    font-weight: 800;
    line-height: 1.1;
    color: #ffffff !important;
    margin-bottom: 0.25rem;
}
.kpi-box .subtext {
    font-size: 0.82rem;
    color: #8b9bb0 !important;
    font-weight: 500;
}

/* Section Dividers & Headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    font-size: 1.15rem;
    font-weight: 700;
    color: #ffffff !important;
    border-bottom: 1px solid #232d3d;
    padding-bottom: 0.65rem;
    margin: 1.8rem 0 1.2rem 0;
}
.section-header .dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #58a6ff;
    box-shadow: 0 0 10px rgba(88, 166, 255, 0.8);
}

/* Card Containers */
.dashboard-card {
    background: #141b27 !important;
    border: 1px solid #283546 !important;
    border-radius: 12px !important;
    padding: 1.25rem 1.4rem !important;
    min-height: 420px !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25) !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #141b27 !important;
    border: 1px solid #283546 !important;
    border-radius: 12px !important;
    padding: 1.15rem 1.25rem !important;
    min-height: 420px !important;
    height: auto !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25) !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
}

/* Ensure Matplotlib / Streamlit pyplot images expand automatically to full card size */
[data-testid="stImage"], [data-testid="stPlot"] {
    width: 100% !important;
    height: auto !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    margin: auto 0 !important;
}

[data-testid="stImage"] > img {
    max-width: 100% !important;
    width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
}
.dashboard-card-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #e6edf3 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.85rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Damage Class Row item */
.class-item-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.65rem 0.9rem;
    background: #101622;
    border: 1px solid #202b3a;
    border-radius: 8px;
    margin-bottom: 0.55rem;
}

/* Calculation & Cost Explainer Box */
.math-explainer {
    background: #121926;
    border: 1px solid #263346;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-top: 1rem;
    color: #c9d1d9;
    font-size: 0.88rem;
    line-height: 1.6;
}
.math-explainer h4 {
    color: #58a6ff !important;
    margin: 0 0 0.5rem 0;
    font-size: 1rem;
}

/* Streamlit Native Buttons */
.stDownloadButton > button {
    background: #162030 !important;
    border: 1px solid #364860 !important;
    color: #f0f6fc !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    border-color: #58a6ff !important;
    background: #1b283d !important;
    color: #ffffff !important;
    transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)


# ── Calculation Helpers ───────────────────────────────────────────────────────
def calculate_severity_score(df: pd.DataFrame) -> float:
    """
    Road Severity Score (0 to 100)
    Normalized score based on damage severity weights and detection confidences.
    Formula:
      Sum(Severity_Weight[class] * Confidence) / (Total_Detections * Max_Weight(4)) * 100
    """
    if df.empty:
        return 0.0
    total_weighted = sum(
        SEVERITY_WEIGHT[r["Class"]] * r["Confidence"] for _, r in df.iterrows()
    )
    max_possible = len(df) * max(SEVERITY_WEIGHT.values())
    return round((total_weighted / max_possible) * 100, 1)


def get_severity_badge(score: float):
    """Returns classification label, background color, and text color for a score."""
    if score == 0:
        return "None", "#232d3d", "#8b949e"
    if score < 25:
        return "Low", "rgba(72, 199, 142, 0.2)", "#48c78e"
    if score < 55:
        return "Moderate", "rgba(255, 178, 54, 0.2)", "#ffb236"
    if score < 80:
        return "High", "rgba(255, 90, 90, 0.25)", "#ff5a5a"
    return "Critical", "rgba(168, 120, 240, 0.25)", "#c084fc"


def calculate_repair_cost(df: pd.DataFrame) -> int:
    """
    Estimated Repair Cost (in INR ₹)
    Total = Sum(Unit_Cost[Class] for each detection)
    """
    if df.empty:
        return 0
    return sum(REPAIR_COST[r["Class"]] for _, r in df.iterrows())


def calculate_area_coverage(df: pd.DataFrame, img_w: int, img_h: int) -> float:
    """Calculates percentage of total image area covered by damage bounding boxes."""
    if df.empty or img_w == 0 or img_h == 0:
        return 0.0
    total_bbox_area = sum((r["x2"] - r["x1"]) * (r["y2"] - r["y1"]) for _, r in df.iterrows())
    pct = (total_bbox_area / (img_w * img_h)) * 100.0
    return round(min(pct, 100.0), 2)


def get_hex_color(cls: str) -> str:
    r, g, b = COLORS_RGB[cls]
    return f"#{r:02x}{g:02x}{b:02x}"


# ── High Contrast Matplotlib Charts ───────────────────────────────────────────
def apply_chart_theme():
    plt.rcParams.update({
        "figure.facecolor": "#141b27",
        "axes.facecolor": "#141b27",
        "axes.edgecolor": "#2c3b4f",
        "axes.labelcolor": "#c9d1d9",
        "xtick.color": "#c9d1d9",
        "ytick.color": "#c9d1d9",
        "text.color": "#f0f6fc",
        "grid.color": "#1f2a3a",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "font.family": "sans-serif",
    })


def chart_donut(counts: dict):
    apply_chart_theme()
    labels = [k for k in NAMES if counts.get(k, 0) > 0]
    sizes = [counts[k] for k in labels]
    colors = [get_hex_color(k) for k in labels]
    total = sum(sizes)

    fig, ax = plt.subplots(figsize=(4.0, 3.8), facecolor="#141b27")
    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.48, edgecolor="#141b27", linewidth=2.5),
    )
    ax.text(0, 0.08, str(total), ha="center", va="center", fontsize=24, fontweight="bold", color="#ffffff")
    ax.text(0, -0.16, "Detections", ha="center", va="center", fontsize=10, color="#a5b4c6", fontweight="500")

    patches = [
        mpatches.Patch(color=colors[i], label=f"{labels[i]} ({sizes[i]})")
        for i in range(len(labels))
    ]
    ax.legend(
        handles=patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor="#e6edf3",
    )
    ax.set_title("Damage Distribution", fontsize=11, fontweight="bold", color="#f0f6fc", pad=12)
    fig.tight_layout()
    return fig


def chart_radar(counts: dict):
    apply_chart_theme()
    n_vars = len(NAMES)
    vals = [counts.get(k, 0) * SEVERITY_WEIGHT[k] for k in NAMES]
    vals += vals[:1]  # Complete the circle
    angles = [n / float(n_vars) * 2 * np.pi for n in range(n_vars)] + [0]

    fig, ax = plt.subplots(figsize=(4.0, 3.8), subplot_kw=dict(polar=True), facecolor="#141b27")
    ax.set_facecolor("#141b27")
    ax.plot(angles, vals, "o-", lw=2.2, color="#58a6ff", markersize=6)
    ax.fill(angles, vals, alpha=0.25, color="#58a6ff")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(NAMES, fontsize=10, fontweight="bold", color="#f0f6fc")
    ax.tick_params(axis="y", colors="#38495f", labelcolor="#a5b4c6", labelsize=8)
    ax.spines["polar"].set_color("#2c3b4f")
    ax.set_title("Severity Impact Profile", fontsize=11, fontweight="bold", color="#f0f6fc", pad=16)
    fig.tight_layout()
    return fig


def chart_confidence(df: pd.DataFrame):
    apply_chart_theme()
    fig, ax = plt.subplots(figsize=(6.0, 3.2), facecolor="#141b27")
    for cls in NAMES:
        sub = df[df["Class"] == cls]
        if sub.empty:
            continue
        ax.scatter(
            sub.index,
            sub["Confidence"],
            label=f"{cls} ({DESCRIPTIONS[cls]})",
            color=get_hex_color(cls),
            s=70,
            alpha=0.9,
            edgecolors="#141b27",
            linewidth=1.2,
            zorder=3,
        )

    mean_conf = df["Confidence"].mean()
    ax.axhline(mean_conf, color="#58a6ff", linestyle="--", lw=1.5, label=f"Mean Conf: {mean_conf:.2f}")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Detection Index (#)", fontsize=9, fontweight="bold", color="#c9d1d9")
    ax.set_ylabel("Confidence Score", fontsize=9, fontweight="bold", color="#c9d1d9")
    ax.set_title("Per-Detection Model Confidence", fontsize=11, fontweight="bold", color="#f0f6fc", pad=10)
    ax.legend(fontsize=8, frameon=False, labelcolor="#e6edf3", loc="lower right")
    ax.grid(True, axis="y")
    fig.tight_layout()
    return fig


def chart_area_distribution(df: pd.DataFrame):
    apply_chart_theme()
    fig, ax = plt.subplots(figsize=(6.0, 3.2), facecolor="#141b27")
    df_sorted = df.copy()
    df_sorted["Area_px"] = (df_sorted["x2"] - df_sorted["x1"]) * (df_sorted["y2"] - df_sorted["y1"])
    df_sorted = df_sorted.sort_values(by="Area_px", ascending=True).reset_index(drop=True)

    colors = [get_hex_color(r["Class"]) for _, r in df_sorted.iterrows()]
    y_pos = range(len(df_sorted))

    ax.barh(y_pos, df_sorted["Area_px"], color=colors, height=0.65, edgecolor="#141b27")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"#{i+1} {r['Class']}" for i, r in df_sorted.iterrows()], fontsize=8, color="#c9d1d9")
    ax.set_xlabel("Bounding Box Area (px²)", fontsize=9, fontweight="bold", color="#c9d1d9")
    ax.set_title("Damage Surface Area (Relative Size)", fontsize=11, fontweight="bold", color="#f0f6fc", pad=10)
    ax.grid(True, axis="x")
    fig.tight_layout()
    return fig


# ── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = attempt_load(WEIGHTS, map_location=device).eval()
    return model, device


# ── Detection Pipeline ────────────────────────────────────────────────────────
def detect_damage(image_rgb, model, device, imgsz, conf_thresh, iou_thresh):
    im0 = image_rgb[:, :, ::-1].copy()  # RGB to BGR for OpenCV
    stride = int(model.stride.max())
    im = letterbox(im0, imgsz, stride=stride)[0]
    im = np.ascontiguousarray(im[:, :, ::-1].transpose(2, 0, 1))
    im = torch.from_numpy(im).to(device).float() / 255.0

    with torch.no_grad():
        pred = model(im[None])[0]
        pred = non_max_suppression(pred, conf_thresh, iou_thresh)[0]

    rows = []
    line_thick = max(2, im0.shape[1] // 380)
    font_scale = max(0.5, im0.shape[1] / 1500)

    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(im.shape[1:], pred[:, :4], im0.shape).round()
        for *xyxy, score, cls in pred.cpu().numpy():
            x1, y1, x2, y2 = map(int, xyxy)
            name = NAMES[int(cls)]
            color = COLORS_BGR[name]

            # Bounding box
            cv2.rectangle(im0, (x1, y1), (x2, y2), color, line_thick)

            # Label banner
            lbl = f"{name} {score:.2f}"
            (tw, th), bl = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thick)
            cv2.rectangle(im0, (x1, y1 - th - bl - 4), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                im0,
                lbl,
                (x1 + 3, y1 - bl - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (15, 15, 15),
                line_thick,
            )

            # PRESERVE ORIGINAL CSV SCHEMA EXACTLY
            rows.append({
                "Class": name,
                "Type": DESCRIPTIONS[name],
                "Confidence": round(float(score), 3),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            })

    # Return RGB image and DataFrame with the exact original columns
    return im0[:, :, ::-1], pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR CONTROLS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(clean_html("""
    <div style="padding: 0.5rem 0 0.8rem 0;">
        <h2 style="font-size: 1.35rem; font-weight: 800; color: #ffffff; margin: 0;">Configuration</h2>
        <p style="font-size: 0.82rem; color: #a5b4c6; margin: 3px 0 0 0;">YOLOv7 Model & Detection Tuning</p>
    </div>
    """), unsafe_allow_html=True)

    conf = st.slider("Confidence Threshold", 0.05, 0.90, 0.25, 0.05,
                     help="Minimum confidence probability to consider a detection valid.")
    iou = st.slider("NMS IoU Threshold", 0.10, 0.90, 0.45, 0.05,
                    help="Non-Maximum Suppression threshold to eliminate overlapping duplicates.")
    imgsz = st.selectbox("Inference Resolution", [640, 960, 1024, 1280], index=0,
                         help="Model default is 640px. Higher values improve small crack detection at higher compute.")

    st.markdown("<hr style='border-color: #232d3d; margin: 1.2rem 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 0.78rem; font-weight: 700; color: #a5b4c6; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.6rem;'>Damage Class Legend</p>", unsafe_allow_html=True)

    for k, v in DESCRIPTIONS.items():
        r, g, b = COLORS_RGB[k]
        st.markdown(clean_html(f"""
        <div style="display: flex; align-items: center; gap: 0.65rem; margin-bottom: 0.5rem;">
            <div style="width: 11px; height: 11px; border-radius: 50%; background: rgb({r},{g},{b}); flex-shrink: 0;"></div>
            <div>
                <span style="font-weight: 700; color: #ffffff; font-size: 0.84rem;">{k}</span>
                <span style="color: #a5b4c6; font-size: 0.80rem;"> &mdash; {v}</span>
            </div>
        </div>
        """), unsafe_allow_html=True)

    st.markdown("<hr style='border-color: #232d3d; margin: 1.2rem 0;'>", unsafe_allow_html=True)
    try:
        model, device = load_model()
        dev_label = "GPU (CUDA)" if "cuda" in str(device) else "CPU"
        st.markdown(clean_html(f"""
        <div style="background: #141b27; border: 1px solid #283546; border-radius: 8px; padding: 0.75rem 1rem;">
            <div style="font-size: 0.72rem; color: #a5b4c6; text-transform: uppercase; letter-spacing: 0.06em;">System Status</div>
            <div style="font-size: 0.92rem; font-weight: 700; color: #48c78e; margin-top: 0.2rem;">Model Ready & Loaded</div>
            <div style="font-size: 0.78rem; color: #c9d1d9; margin-top: 0.2rem;">Hardware: <b>{dev_label}</b></div>
            <div style="font-size: 0.78rem; color: #c9d1d9;">Weights: <b>best.pt</b> (RDD2022)</div>
        </div>
        """), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load weights best.pt: {e}")
        st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-card">
    <span class="badge-pill">YOLOv7 Architecture</span>
    <span class="badge-pill">RDD2022 Benchmark</span>
    <span class="badge-pill">Road Infrastructure AI</span>
    <h1>Road Damage Intelligence Platform</h1>
    <p>Automated pavement defect detection, quantitative severity assessment, itemized repair budgeting, and exportable engineering telemetry.</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  IMAGE UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header"><div class="dot"></div>Input Road Imagery</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Choose road image for damage analysis",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown("""
    <div style="border: 2px dashed #283546; border-radius: 14px; padding: 3rem 2rem; text-align: center; background: #141b27; margin: 1rem 0;">
        <div style="font-size: 3rem; margin-bottom: 0.8rem;">🛣️</div>
        <h3 style="color: #ffffff; font-weight: 700; margin: 0 0 0.4rem 0;">Upload Road Surface Photograph</h3>
        <p style="color: #a5b4c6; font-size: 0.92rem; margin: 0;">Supports High-Resolution Drone, Vehicle Dashcam, or Inspection Mobile Camera (JPG, PNG)</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    info_boxes = [
        (c1, "🔍", "4 Damage Classes", "Longitudinal cracks (D00), Transverse cracks (D10), Fatigue Alligator cracks (D20), and Potholes (D40)."),
        (c2, "📊", "Severity & Cost Analytics", "Multi-parameter damage indexing, unit cost repair estimates, and bounding box surface coverage."),
        (c3, "📥", "Full CSV & Report Export", "Export exact detection coordinates, confidence logs, annotated imagery, and detailed text summaries."),
    ]
    for col, icon, title, desc in info_boxes:
        col.markdown(f"""
        <div class="dashboard-card">
            <div style="font-size: 1.6rem; margin-bottom: 0.5rem;">{icon}</div>
            <div style="font-weight: 700; color: #ffffff; margin-bottom: 0.35rem; font-size: 0.98rem;">{title}</div>
            <div style="font-size: 0.84rem; color: #a5b4c6; line-height: 1.45;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
image_pil = Image.open(uploaded).convert("RGB")
image_np = np.array(image_pil)
img_h, img_w = image_np.shape[:2]

with st.spinner("Analyzing pavement surface with YOLOv7..."):
    annotated_np, df = detect_damage(image_np, model, device, imgsz, conf, iou)

annotated_pil = Image.fromarray(annotated_np)
counts = df["Class"].value_counts().to_dict() if not df.empty else {}
sev_score = calculate_severity_score(df)
sev_label, sev_bg, sev_col = get_severity_badge(sev_score)
total_repair = calculate_repair_cost(df)
coverage_pct = calculate_area_coverage(df, img_w, img_h)
timestamp_str = datetime.now().strftime("%d %b %Y, %H:%M")


# ── Metadata Chips ────────────────────────────────────────────────────────────
st.markdown(clean_html(f"""
<div style="display: flex; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 1.3rem;">
    <span class="badge-pill" style="margin-bottom:0;">📐 Dimensions: {img_w} × {img_h} px</span>
    <span class="badge-pill" style="margin-bottom:0;">⏱️ Analyzed: {timestamp_str}</span>
    <span class="badge-pill" style="margin-bottom:0;">🎯 Conf Threshold: ≥ {conf:.2f}</span>
    <span class="badge-pill" style="margin-bottom:0;">⚙️ NMS IoU: {iou:.2f}</span>
    <span class="badge-pill" style="margin-bottom:0;">🚀 Resolution: {imgsz}px</span>
</div>
"""), unsafe_allow_html=True)


# ── KPI Metric Cards (Balanced 4-Column Row) ──────────────────────────────────
st.markdown(clean_html(f"""
<div class="kpi-row">
    <div class="kpi-box">
        <div class="accent-stripe" style="background: #58a6ff;"></div>
        <div class="label">Total Detections</div>
        <div class="value">{len(df)}</div>
        <div class="subtext">{len(counts)} defect type(s) identified</div>
    </div>
    <div class="kpi-box">
        <div class="accent-stripe" style="background: {sev_col};"></div>
        <div class="label">Road Severity Index</div>
        <div class="value" style="color: {sev_col} !important;">{sev_score} <span style="font-size:1.1rem; color:#a5b4c6;">/100</span></div>
        <div class="subtext"><span style="background:{sev_bg}; color:{sev_col}; padding:2px 8px; border-radius:12px; font-weight:700;">{sev_label.upper()}</span></div>
    </div>
    <div class="kpi-box">
        <div class="accent-stripe" style="background: #48c78e;"></div>
        <div class="label">Estimated Repair Cost</div>
        <div class="value" style="color: #48c78e !important;">₹{total_repair:,}</div>
        <div class="subtext">PWD schedule benchmark rate</div>
    </div>
    <div class="kpi-box">
        <div class="accent-stripe" style="background: #ffb236;"></div>
        <div class="label">Pavement Area Affected</div>
        <div class="value">{coverage_pct}%</div>
        <div class="subtext">of total inspection frame</div>
    </div>
</div>
"""), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDE-BY-SIDE VISUAL INSPECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header"><div class="dot"></div>Visual Inspection & Localization</div>', unsafe_allow_html=True)
col_left, col_right = st.columns(2, gap="medium")

with col_left:
    st.markdown("<p style='font-size: 0.9rem; font-weight: 700; color: #a5b4c6; margin-bottom: 0.4rem;'>ORIGINAL INPUT IMAGE</p>", unsafe_allow_html=True)
    st.image(image_pil, width="stretch")

with col_right:
    st.markdown("<p style='font-size: 0.9rem; font-weight: 700; color: #a5b4c6; margin-bottom: 0.4rem;'>AI DETECTION & LOCALIZATION</p>", unsafe_allow_html=True)
    st.image(annotated_pil, width="stretch")

if df.empty:
    st.markdown("""
    <div style="background: #141b27; border: 1px solid #283546; border-radius: 12px; padding: 2rem; text-align: center; margin-top: 1.5rem;">
        <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">✅</div>
        <h3 style="color: #48c78e; margin: 0 0 0.4rem 0;">No Road Surface Defects Detected</h3>
        <p style="color: #a5b4c6; font-size: 0.9rem; margin: 0;">
            The roadway meets acceptable condition at confidence threshold ≥ {conf}. If subtle cracking exists, try lowering the confidence threshold slider in the sidebar.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  BALANCED ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header"><div class="dot"></div>Analytics Dashboard</div>', unsafe_allow_html=True)

# ROW 1: 3 Equal-Width Columns for Balanced Alignment
row1_c1, row1_c2, row1_c3 = st.columns([1.1, 1.0, 1.0], gap="medium")

with row1_c1:
    class_rows = []
    for cls in NAMES:
        n = counts.get(cls, 0)
        r, g, b = COLORS_RGB[cls]
        cls_df = df[df["Class"] == cls]
        avg_c = cls_df["Confidence"].mean() if not cls_df.empty else 0.0
        opacity = "1" if n > 0 else "0.45"
        avg_text = f" | Avg Conf: {avg_c:.2f}" if n > 0 else ""
        class_rows.append(f"""
<div class="class-item-row" style="opacity: {opacity}; border-left: 3.5px solid rgb({r},{g},{b});">
<div>
<div style="font-weight: 700; color: #ffffff; font-size: 0.88rem;">{cls} &middot; {DESCRIPTIONS[cls]}</div>
<div style="font-size: 0.76rem; color: #a5b4c6;">Severity Rank: {SEVERITY_WEIGHT[cls]}/4{avg_text}</div>
</div>
<div style="font-size: 1.3rem; font-weight: 800; color: rgb({r},{g},{b}); min-width: 32px; text-align: right;">{n}</div>
</div>
""")

    class_rows_html = "".join(class_rows)

    c1_html = f"""
<div class="dashboard-card">
<div class="dashboard-card-title">
<span>Severity &amp; Class Summary</span>
<span style="font-size:0.75rem; background:{sev_bg}; color:{sev_col}; padding:3px 10px; border-radius:12px; font-weight:700;">{sev_label.upper()}</span>
</div>
<div style="text-align: center; padding: 0.6rem 0 1rem 0; border-bottom: 1px solid #232d3d; margin-bottom: 1rem;">
<div style="font-size: 0.76rem; color: #a5b4c6; text-transform: uppercase; letter-spacing: 0.06em;">Road Severity Index</div>
<div style="font-size: 2.8rem; font-weight: 800; color: {sev_col}; line-height: 1;">{sev_score}</div>
<div style="font-size: 0.78rem; color: #8b9bb0; margin-top: 4px;">Normalized 0–100 Scale</div>
</div>
{class_rows_html}
</div>
"""
    st.markdown(clean_html(c1_html), unsafe_allow_html=True)

with row1_c2:
    with st.container(border=True):
        st.markdown(
            '<div class="dashboard-card-title"><span>Damage Distribution</span><span style="font-size:0.75rem; color:#8b9bb0;">DONUT</span></div>',
            unsafe_allow_html=True,
        )
        fig_donut = chart_donut(counts)
        st.pyplot(fig_donut, width="stretch")
        plt.close(fig_donut)

with row1_c3:
    with st.container(border=True):
        st.markdown(
            '<div class="dashboard-card-title"><span>Severity Profile</span><span style="font-size:0.75rem; color:#8b9bb0;">RADAR</span></div>',
            unsafe_allow_html=True,
        )
        fig_radar = chart_radar(counts)
        st.pyplot(fig_radar, width="stretch")
        plt.close(fig_radar)

# ROW 2: 2 Symmetrical Analytical Chart Columns
st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
row2_c1, row2_c2 = st.columns(2, gap="medium")

with row2_c1:
    with st.container(border=True):
        st.markdown(
            '<div class="dashboard-card-title"><span>Per-Detection Model Confidence</span><span style="font-size:0.75rem; color:#8b9bb0;">SCATTER</span></div>',
            unsafe_allow_html=True,
        )
        fig_conf = chart_confidence(df)
        st.pyplot(fig_conf, width="stretch")
        plt.close(fig_conf)

with row2_c2:
    with st.container(border=True):
        st.markdown(
            '<div class="dashboard-card-title"><span>Damage Surface Area (Relative Size)</span><span style="font-size:0.75rem; color:#8b9bb0;">BAR</span></div>',
            unsafe_allow_html=True,
        )
        fig_area = chart_area_distribution(df)
        st.pyplot(fig_area, width="stretch")
        plt.close(fig_area)


# ══════════════════════════════════════════════════════════════════════════════
#  PREVIOUS CSV SHEET & DETECTION TABLE (PRESERVED EXACTLY AS IT WAS)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header"><div class="dot"></div>Detection Records & Data Table</div>', unsafe_allow_html=True)

# 1. Exact detection count subheader
st.subheader(f"{len(df)} detection(s)")

# 2. Exact 4 metrics for Class counts as in previous app
class_counts = df["Class"].value_counts().reindex(NAMES, fill_value=0)
metric_cols = st.columns(len(NAMES))
for col, name in zip(metric_cols, NAMES):
    col.metric(f"{name} ({DESCRIPTIONS[name]})", int(class_counts[name]))

# 3. Exact DataFrame table with original schema: Class, Type, Confidence, x1, y1, x2, y2
st.dataframe(df, width="stretch", hide_index=True)

# 4. Exact CSV download button preserving original CSV structure
csv_bytes = df.to_csv(index=False).encode("utf-8")
col_dl1, col_dl2 = st.columns([1, 2])
with col_dl1:
    st.download_button(
        label="Download detections (CSV)",
        data=csv_bytes,
        file_name="detections.csv",
        mime="text/csv",
        width="stretch",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  REPAIR BUDGET & METHODOLOGY (HOW ESTIMATED COST & METRICS ARE CALCULATED)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header"><div class="dot"></div>Repair Cost Breakdown & Calculation Methodology</div>', unsafe_allow_html=True)

cost_l, cost_r = st.columns([1.6, 1.0], gap="medium")

with cost_l:
    # Itemized Cost Table
    cost_data = []
    for cls in ["D40", "D20", "D10", "D00"]:
        cnt = counts.get(cls, 0)
        rate = REPAIR_COST[cls]
        subtotal = cnt * rate
        prio_label, prio_color, prio_desc = PRIORITY_MAP[cls]
        cost_data.append({
            "Class": cls,
            "Damage Type": DESCRIPTIONS[cls],
            "Priority": prio_label,
            "Occurrences": cnt,
            "Unit Rate": f"₹{rate:,}",
            "Subtotal": f"₹{subtotal:,}",
        })
    cost_df = pd.DataFrame(cost_data)
    st.dataframe(cost_df, width="stretch", hide_index=True)

with cost_r:
    cost_card_html = f"""
    <div class="dashboard-card" style="text-align: center; padding: 1.8rem 1.5rem; min-height: auto !important;">
        <div style="font-size: 0.78rem; color: #a5b4c6; text-transform: uppercase; letter-spacing: 0.07em;">
            Total Estimated Repair Budget
        </div>
        <div style="font-size: 2.5rem; font-weight: 800; color: #48c78e; margin: 0.4rem 0;">
            ₹{total_repair:,}
        </div>
        <div style="font-size: 0.85rem; color: #c9d1d9; margin-bottom: 0.8rem;">
            Covering <b>{len(df)}</b> detected defect(s)
        </div>
        <div style="display:inline-block; padding: 5px 14px; border-radius: 20px; background: {sev_bg}; color: {sev_col}; font-size: 0.82rem; font-weight: 700;">
            {sev_label.upper()} PRIORITY ACTION
        </div>
    </div>
    """
    st.markdown(clean_html(cost_card_html), unsafe_allow_html=True)

# ── Detailed Calculation Methodology Accordion / Explainer ────────────────────
with st.expander("📖 How are the Estimated Cost, Severity Score, and Coverage Calculated? (Click to view formulas)", expanded=True):
    st.markdown("""
    ### 1. Estimated Repair Cost Calculation
    The repair budget is computed using a **standardized unit rate model** derived from municipal Public Works Department (PWD) highway maintenance schedules:
    
    $$\\text{Total Repair Cost (₹)} = \\sum_{i \\in \\{D00, D10, D20, D40\\}} (\\text{Count}_i \\times \\text{Unit Rate}_i)$$
    
    - **D00 (Longitudinal Crack — ₹500/defect):**
      - *Process:* Cleaning with compressed air followed by rubberized asphalt sealant injection to prevent moisture infiltration along traffic seams.
    - **D10 (Transverse Crack — ₹800/defect):**
      - *Process:* Joint routing/grooving, debris clearance, and application of hot-pour elastomeric sealant across the pavement width.
    - **D20 (Alligator Crack — ₹2,000/patch):**
      - *Process:* Sign of structural sub-base fatigue failure. Requires diamond saw cutting around the affected perimeter, full-depth asphalt removal, base compaction, tack coating, and hot-mix asphalt (HMA) infill.
    - **D40 (Pothole — ₹3,500/repair):**
      - *Process:* Immediate vehicular hazard. Involves vertical squaring of perimeter edges, excavation of loose debris, bitumen emulsion tack coat, layered asphalt aggregate placement, and mechanical vibratory roller compaction.

    ---

    ### 2. Road Severity Index (0–100 Score)
    The severity score reflects both the **urgency of the damage type** and the **AI model's detection confidence**:
    
    $$\\text{Road Severity Index} = \\left( \\frac{\\sum_{j=1}^{N} (\\text{Weight}_j \\times \\text{Confidence}_j)}{N \\times \\text{Max Weight (4)}} \\right) \\times 100$$
    
    - **Damage Severity Weights:**
      - `D00` (Longitudinal): **1.0** (Surface-level)
      - `D10` (Transverse): **2.0** (Moderate joint contraction)
      - `D20` (Alligator): **3.0** (Severe structural fatigue)
      - `D40` (Pothole): **4.0** (Critical tire hazard)
    - **Index Scale:**
      - **0 – 24 (Low):** Minor surface fatigue; routine monitoring recommended.
      - **25 – 54 (Moderate):** Scheduled maintenance needed before monsoon / seasonal freeze.
      - **55 – 79 (High):** Structural degradation underway; schedule priority patching.
      - **80 – 100 (Critical):** Severe road failure; immediate emergency repair crew dispatch required.

    ---

    ### 3. Pavement Area Coverage (%)
    Measures the aggregate bounding-box footprint relative to the total camera view:
    
    $$\\text{Area Coverage (\\%)} = \\min\\left(100, \\frac{\\sum (\\text{Width}_j \\times \\text{Height}_j)}{\\text{Image Width} \\times \\text{Image Height}} \\times 100\\right)$$
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT SECTION (ADDITIONAL FORMATS)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header"><div class="dot"></div>Additional Engineering Exports</div>', unsafe_allow_html=True)
exp_col1, exp_col2 = st.columns(2, gap="medium")

with exp_col1:
    buf_img = io.BytesIO()
    annotated_pil.save(buf_img, format="PNG")
    st.download_button(
        "Download Annotated Imagery (PNG)",
        data=buf_img.getvalue(),
        file_name="annotated_road_damage.png",
        mime="image/png",
        width="stretch",
    )

with exp_col2:
    report_text = [
        "ROAD DAMAGE TELEMETRY & AUDIT REPORT",
        f"Generated Timestamp : {timestamp_str}",
        f"Image File Name     : {uploaded.name} ({img_w}x{img_h} px)",
        "=" * 50,
        "EXECUTIVE SUMMARY:",
        f"Total Defects Found  : {len(df)}",
        f"Road Severity Score  : {sev_score} / 100 ({sev_label})",
        f"Pavement Area Covered: {coverage_pct}%",
        f"Total Repair Budget  : INR {total_repair:,}",
        "=" * 50,
        "DEFECT BREAKDOWN BY CLASS:",
        *[f"  - {k} ({DESCRIPTIONS[k]}): {counts.get(k, 0)} detected (Unit Cost: INR {REPAIR_COST[k]})" for k in NAMES],
        "=" * 50,
        "DETAILED LOCALIZATION LOG:",
        *[
            f"  [{i+1:02d}] {r['Class']} ({r['Type']}) | Conf: {r['Confidence']:.3f} | BBox: ({r['x1']}, {r['y1']}) -> ({r['x2']}, {r['y2']})"
            for i, (_, r) in enumerate(df.iterrows())
        ],
    ]
    st.download_button(
        "Download Audit Report (TXT)",
        data="\n".join(report_text).encode("utf-8"),
        file_name="road_damage_report.txt",
        mime="text/plain",
        width="stretch",
    )

st.markdown("""
<div style="text-align: center; font-size: 0.78rem; color: #48566a; padding: 2.5rem 0 1rem 0;">
    Road Damage Intelligence Platform &middot; YOLOv7 Trained on RDD2022 &middot; Pavement Asset Management System
</div>
""", unsafe_allow_html=True)
