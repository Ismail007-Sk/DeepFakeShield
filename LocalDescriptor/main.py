import streamlit as st
import cv2
import numpy as np
import joblib
from pathlib import Path
from LPQ import lpq
from CLBP import clbp_feature
from BSIF import bsif
from WLD import wld
import tempfile

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="DeepFakeShield",
    page_icon="🛡️",
    layout="centered"
)

# ---------------- PARAMETERS ----------------
RESIZE_DIM = (128, 128)
LPQ_WINDOW = 3
FRAME_STEP = 10

IMAGE_EXT = [".jpg", ".jpeg", ".png", ".bmp", ".webp",".heic"]
VIDEO_EXT = [".mp4", ".avi", ".mov", ".mkv"]

# ---------------- LOAD MODEL ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Models" / "LDmodel_SVM_2.pkl"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
scaler = bundle["scaler"]


# ==========================================================
# ATTENTION FUSION
# ==========================================================
def attention_fusion(feature_list):
    energies = np.array([np.linalg.norm(f) for f in feature_list], dtype=np.float32)
    weights = energies / (energies.sum() + 1e-7)

    fused = []
    for w, f in zip(weights, feature_list):
        fused.append(f * w)

    return np.concatenate(fused)


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================
def extract_features(gray):
    gray = cv2.resize(gray, RESIZE_DIM)

    # ----- BSIF -----
    _, bsif_hist = bsif(gray)

    # ----- LPQ -----
    lpq_raw = lpq(gray, win_size=LPQ_WINDOW)
    lpq_hist, _ = np.histogram(lpq_raw, bins=256, range=(0, 256))
    lpq_hist = lpq_hist.astype(np.float32)
    lpq_hist /= (lpq_hist.sum() + 1e-7)

    # ----- CLBP -----
    clbp_feat = clbp_feature(gray)

    # ----- WLD -----
    _, wld_feat = wld(gray)

    # ----- ATTENTION FUSION -----
    fused_feature = attention_fusion([
        bsif_hist,
        lpq_hist,
        clbp_feat,
        wld_feat
    ])

    return fused_feature


# ==========================================================
# VIDEO PROCESSING
# ==========================================================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    feats = []
    fid = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if fid % FRAME_STEP == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            feats.append(extract_features(gray))
        fid += 1

    cap.release()
    return np.mean(feats, axis=0)


# ==========================================================
# UI
# ==========================================================
st.markdown(
    "<h1 style='text-align:center;'>🛡️ DeepFakeShield</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Hybrid AI-based Image & Video Deepfake Detection</p>",
    unsafe_allow_html=True
)

uploaded = st.file_uploader(
    "Upload Image or Video",
    type=IMAGE_EXT + VIDEO_EXT
)

if uploaded:
    suffix = uploaded.name.split(".")[-1].lower()

    with st.spinner("Analyzing content..."):
        if suffix in IMAGE_EXT:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            features = extract_features(img)

        else:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded.read())
            features = process_video(tfile.name)

        features = scaler.transform(features.reshape(1, -1))
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

    if pred == 1:
        st.success(f"✅ REAL content detected\n\nConfidence: {prob[1]*100:.2f}%")
    else:
        st.error(f"❌ DEEPFAKE detected\n\nConfidence: {prob[0]*100:.2f}%")

st.markdown("---")
st.caption("DeepFakeShield • Hybrid Local Descriptor + ML")
