import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
import joblib
from pathlib import Path
import tempfile

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="DeepFakeShield Hybrid",
    page_icon="🛡️",
    layout="centered"
)

# ---------------- PARAMETERS ----------------
IMG_SIZE = 224
RESIZE_DIM = (128, 128)
LPQ_WINDOW = 3
FRAME_STEP = 10

W_LD = 0.7
W_DL = 0.3

IMAGE_EXT = [".jpg", ".jpeg", ".png", ".bmp", ".webp",".heic"]
VIDEO_EXT = [".mp4", ".avi", ".mov", ".mkv"]

BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# LOAD DL MODEL
# ==========================================================
DL_MODEL_PATH = BASE_DIR / "DeepLearning" / "Models" / "model1.pth"

dl_model = models.efficientnet_b0()
dl_model.classifier[1] = nn.Linear(dl_model.classifier[1].in_features, 2)
dl_model.load_state_dict(torch.load(DL_MODEL_PATH, map_location=DEVICE))
dl_model.to(DEVICE)
dl_model.eval()

dl_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==========================================================
# LOAD LD MODEL
# ==========================================================
LD_MODEL_PATH = BASE_DIR / "LocalDescriptor" /"Models" / "LDmodel_SVM_1.pkl"
bundle = joblib.load(LD_MODEL_PATH)
ld_model = bundle["model"]
ld_scaler = bundle["scaler"]

from LocalDescriptor.LPQ import lpq
from LocalDescriptor.CLBP import clbp_feature
from LocalDescriptor.BSIF import bsif
from LocalDescriptor.WLD import wld

# ==========================================================
# LD FEATURE EXTRACTION
# ==========================================================
def attention_fusion(feature_list):
    energies = np.array([np.linalg.norm(f) for f in feature_list], dtype=np.float32)
    weights = energies / (energies.sum() + 1e-7)
    return np.concatenate([w * f for w, f in zip(weights, feature_list)])

def extract_ld_features(gray):
    gray = cv2.resize(gray, RESIZE_DIM)

    _, bsif_hist = bsif(gray)

    lpq_raw = lpq(gray, win_size=LPQ_WINDOW)
    lpq_hist, _ = np.histogram(lpq_raw, bins=256, range=(0, 256))
    lpq_hist = lpq_hist.astype(np.float32)
    lpq_hist /= (lpq_hist.sum() + 1e-7)

    clbp_feat = clbp_feature(gray)
    _, wld_feat = wld(gray)

    return attention_fusion([bsif_hist, lpq_hist, clbp_feat, wld_feat])

# ==========================================================
# DL PREDICTION
# ==========================================================
def predict_dl(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = dl_transform(rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = dl_model(tensor)
        probs = torch.softmax(output, dim=1)

    return probs.cpu().numpy()[0]

# ==========================================================
# HYBRID PREDICTION
# ==========================================================
def hybrid_predict(frame):

    # ----- DL -----
    dl_probs = predict_dl(frame)

    # ----- LD -----
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ld_feat = extract_ld_features(gray)
    ld_feat = ld_scaler.transform(ld_feat.reshape(1, -1))
    ld_probs = ld_model.predict_proba(ld_feat)[0]

    # ----- Weighted Fusion -----
    final_probs = W_LD * ld_probs + W_DL * dl_probs
    final_pred = np.argmax(final_probs)

    return final_pred, final_probs

# ==========================================================
# VIDEO PROCESSING
# ==========================================================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    probs_list = []
    fid = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if fid % FRAME_STEP == 0:
            _, probs = hybrid_predict(frame)
            probs_list.append(probs)

        fid += 1

    cap.release()
    return np.mean(probs_list, axis=0)



# ==========================================================
# UI
# ==========================================================
st.markdown("<h1 style='text-align:center;'>🛡️ DeepFakeShield Hybrid</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Local Descriptor + Deep Learning Fusion</p>", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload Image or Video", type=IMAGE_EXT + VIDEO_EXT)

if uploaded:
    suffix = uploaded.name.split(".")[-1].lower()

    with st.spinner("Analyzing content using Hybrid AI..."):

        if suffix in IMAGE_EXT:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            pred, probs = hybrid_predict(frame)

        else:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded.read())
            probs = process_video(tfile.name)
            pred = np.argmax(probs)

    if pred == 1:
        st.success(f"✅ REAL content detected\n\nConfidence: {probs[1]*100:.2f}%")
    else:
        st.error(f"❌ DEEPFAKE detected\n\nConfidence: {probs[0]*100:.2f}%")



st.markdown("---")
st.caption("DeepFakeShield • Hybrid (LD + EfficientNet) Weighted Fusion")
