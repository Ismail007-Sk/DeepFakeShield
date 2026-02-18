import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from pathlib import Path
import tempfile

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="DeepFakeShield",
    page_icon="🛡️",
    layout="centered"
)

# ---------------- PARAMETERS ----------------
IMG_SIZE = 224
FRAME_STEP = 10
IMAGE_EXT = [".jpg", ".jpeg", ".png", ".bmp", ".webp",".heic"]
VIDEO_EXT = [".mp4", ".avi", ".mov", ".mkv"]

# ---------------- LOAD MODEL ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Models" / "model1.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==========================================================
# PREDICTION FUNCTION
# ==========================================================
def predict_image(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(frame).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return pred, probs.cpu().numpy()[0]


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
            _, probs = predict_image(frame)
            probs_list.append(probs)

        fid += 1

    cap.release()

    return np.mean(probs_list, axis=0)


# ==========================================================
# UI
# ==========================================================
st.markdown(
    "<h1 style='text-align:center;'>🛡️ DeepFakeShield</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>EfficientNet-based Image & Video Deepfake Detection</p>",
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
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            pred, probs = predict_image(frame)

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
st.caption("DeepFakeShield • EfficientNet Deep Learning Model")
