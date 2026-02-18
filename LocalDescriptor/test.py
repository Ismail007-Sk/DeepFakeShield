import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ----------------------------------
# Base directory
# ----------------------------------
BASE_DIR = Path(__file__).resolve().parent

# ----------------------------------
# Load test data (already scaled)
# ----------------------------------
X_test = np.load(BASE_DIR / "SplitData" / "X_test.npy")
y_test = np.load(BASE_DIR / "SplitData" / "y_test.npy")


print(f"Loaded test data: {X_test.shape}, labels: {y_test.shape}")

# ----------------------------------
# Load trained model
# ----------------------------------
data = joblib.load(BASE_DIR / "Models" / "LDmodel_SVM_1.pkl")
model = data["model"]

print("LD model loaded successfully")

# ----------------------------------
# Prediction
# ----------------------------------
y_pred = model.predict(X_test)

# ----------------------------------
# Standard ML Metrics
# ----------------------------------
cm = confusion_matrix(y_test, y_pred)

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n===== Classification Results =====")
print("Confusion Matrix:\n", cm)
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

# ----------------------------------
# Face Anti-Spoofing Metrics
# ----------------------------------
# Label convention:
# 0 → Fake (Attack)
# 1 → Real (Bona fide)

# APCER: Fake classified as Real
APCER = cm[0, 1] / cm[0].sum() if cm[0].sum() > 0 else 0.0

# BPCER: Real classified as Fake
BPCER = cm[1, 0] / cm[1].sum() if cm[1].sum() > 0 else 0.0

# Half Total Error Rate
HTER = (APCER + BPCER) / 2

print("\n===== Face Anti-Spoofing Metrics =====")
print(f"APCER : {APCER * 100:.2f}%")
print(f"BPCER : {BPCER * 100:.2f}%")
print(f"HTER  : {HTER * 100:.2f}%")
