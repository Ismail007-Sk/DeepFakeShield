import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import joblib
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "Dataset"

DL_MODEL_PATH = BASE_DIR / "DeepLearning" / "Models" / "model1.pth"
LD_MODEL_PATH = BASE_DIR / "LocalDescriptor" / "Models" / "LDmodel_SVM_1.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- PARAMETERS ----------------
IMG_SIZE = 224
RESIZE_DIM = (128,128)
LPQ_WINDOW = 3

W_LD = 0.5
W_DL = 0.5

# ---------------- LOAD DL MODEL ----------------
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

# ---------------- LOAD LD MODEL ----------------
bundle = joblib.load(LD_MODEL_PATH)
ld_model = bundle["model"]
ld_scaler = bundle["scaler"]

# ---------------- IMPORT FEATURES ----------------
from LocalDescriptor.LPQ import lpq
from LocalDescriptor.CLBP import clbp_feature
from LocalDescriptor.BSIF import bsif
from LocalDescriptor.WLD import wld

# ---------------- FEATURE FUNCTIONS ----------------
def attention_fusion(feature_list):
    energies = np.array([np.linalg.norm(f) for f in feature_list], dtype=np.float32)
    weights = energies / (energies.sum() + 1e-7)
    return np.concatenate([w * f for w, f in zip(weights, feature_list)])

def extract_ld_features(gray):
    gray = cv2.resize(gray, RESIZE_DIM)

    _, bsif_hist = bsif(gray)

    lpq_raw = lpq(gray, win_size=LPQ_WINDOW)
    lpq_hist, _ = np.histogram(lpq_raw, bins=256, range=(0,256))
    lpq_hist = lpq_hist.astype(np.float32)
    lpq_hist /= (lpq_hist.sum() + 1e-7)

    clbp_feat = clbp_feature(gray)
    _, wld_feat = wld(gray)

    return attention_fusion([bsif_hist, lpq_hist, clbp_feat, wld_feat])

# ---------------- PREDICTION FUNCTIONS ----------------
def predict_dl(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = dl_transform(rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = dl_model(tensor)
        probs = torch.softmax(output, dim=1)

    return probs.cpu().numpy()[0]

def hybrid_predict(frame):

    # DL
    dl_probs = predict_dl(frame)

    # LD
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ld_feat = extract_ld_features(gray)
    ld_feat = ld_scaler.transform(ld_feat.reshape(1,-1))
    ld_probs = ld_model.predict_proba(ld_feat)[0]

    # Weighted fusion
    final_probs = W_LD * ld_probs + W_DL * dl_probs
    final_pred = np.argmax(final_probs)

    return final_pred, final_probs

# ---------------- STORAGE ----------------
y_true = []
y_pred = []
y_prob = []

# ---------------- DATASET LOOP ----------------
for label_name in ["fake","real"]:

    folder = os.path.join(DATASET_PATH, label_name)
    true_label = 1 if label_name=="real" else 0

    for img_name in os.listdir(folder):

        img_path = os.path.join(folder, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        pred, probs = hybrid_predict(image)

        y_true.append(true_label)
        y_pred.append(pred)
        y_prob.append(probs[1])   # probability of REAL

# ---------------- METRICS ----------------
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

cm = confusion_matrix(y_true, y_pred)

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)

TN, FP, FN, TP = cm.ravel()

APCER = FP / (FP + TN + 1e-8)
BPCER = FN / (FN + TP + 1e-8)
HTER = (APCER + BPCER) / 2

# ---------------- PRINT RESULTS ----------------
print("\n===== Classification Results =====")

print("Confusion Matrix:\n", cm)

print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"AUC       : {auc:.4f}")

print("\n===== Face Anti-Spoofing Metrics =====")

print(f"APCER : {APCER*100:.2f}%")
print(f"BPCER : {BPCER*100:.2f}%")
print(f"HTER  : {HTER*100:.2f}%")