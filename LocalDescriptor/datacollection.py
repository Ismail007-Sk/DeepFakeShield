import cv2
import numpy as np
from pathlib import Path
from LPQ import lpq
from CLBP import clbp_feature
from BSIF import bsif
from WLD import wld

# ---------- PARAMETERS ----------
RESIZE_DIM = (128, 128)

LPQ_WINDOW = 3
FRAME_STEP = 90   # take every 10th frame

IMAGE_EXT = [".jpg", ".jpeg", ".png", ".bmp", ".webp",".heic"]
VIDEO_EXT = [".mp4", ".avi", ".mov", ".mkv"]

# ---------- PATHS ----------
BASE_DIR = Path(__file__).resolve().parent

# Go one level up to DeepFakeShield
ROOT_DIR = BASE_DIR.parent

# Input (Raw Dataset)
RAW_REAL = ROOT_DIR / "RawDataset" / "Real"
RAW_FAKE = ROOT_DIR / "RawDataset" / "Fake"

# Output (Processed Dataset inside LocalDescriptor)
PRE_REAL = BASE_DIR / "Dataset" / "Real"
PRE_FAKE = BASE_DIR / "Dataset" / "Fake"


PRE_REAL.mkdir(parents=True, exist_ok=True)
PRE_FAKE.mkdir(parents=True, exist_ok=True)


# ==========================================================
# ATTENTION FUSION MODULE
# ==========================================================
def attention_fusion(feature_list):
    """
    Simple deterministic attention fusion.
    Assigns adaptive weights based on feature energy.
    """

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

    # ---------- BSIF ----------
    _, bsif_hist = bsif(gray)

    # ---------- LPQ ----------
    lpq_raw = lpq(gray, win_size=LPQ_WINDOW)
    lpq_hist, _ = np.histogram(lpq_raw, bins=256, range=(0, 256))
    lpq_hist = lpq_hist.astype(np.float32)
    lpq_hist /= (lpq_hist.sum() + 1e-7)

    # ---------- CLBP ----------
    clbp_feat = clbp_feature(gray)

    # ---------- WLD ----------
    _, wld_feat = wld(gray)

    # ---------- ATTENTION FUSION ----------
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
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[WARNING] Cannot open video: {video_path}")
        return None

    features_all = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if frame_id % FRAME_STEP == 0:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                feat = extract_features(gray)
                features_all.append(feat)
            except Exception:
                pass  # skip corrupted frames

        frame_id += 1

    cap.release()

    if len(features_all) == 0:
        print(f"[WARNING] No valid frames in video: {video_path}")
        return None

    return np.mean(features_all, axis=0)


# ==========================================================
# PROCESS FOLDER
# ==========================================================
def process_folder(input_dir, output_dir, label):
    count = 0

    for file in input_dir.iterdir():
        ext = file.suffix.lower()

        # IMAGE
        if ext in IMAGE_EXT:
            img = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            features = extract_features(img)

        # VIDEO
        elif ext in VIDEO_EXT:
            features = process_video(file)
            if features is None:
                continue
        else:
            continue

        np.save(output_dir / f"{label}_{count}.npy", features)
        count += 1

    print(f"{label.upper()} saved: {count}")


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    process_folder(RAW_REAL, PRE_REAL, "real")
    process_folder(RAW_FAKE, PRE_FAKE, "fake")
    print("Image + Video data collection completed ✅")
