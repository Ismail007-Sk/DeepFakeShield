import os
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime

# ----------------------------------
# Dataset directories
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent
REAL_DIR = BASE_DIR / "Dataset"  / "Real"
FAKE_DIR = BASE_DIR / "Dataset"  / "Fake"


# ----------------------------------
# Load feature vectors
# ----------------------------------

real_files = list(REAL_DIR.glob("*.npy"))
fake_files = list(FAKE_DIR.glob("*.npy"))

X, y = [], []

for f in real_files:
    X.append(np.load(f).flatten())
    y.append(1)   # Real

for f in fake_files:
    X.append(np.load(f).flatten())
    y.append(0)   # Fake

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} samples ({len(real_files)} real, {len(fake_files)} fake)")

# ----------------------------------
# Train / Test split
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,  ####
    random_state=42,
    stratify=y
)


# ----------------------------------
# Feature normalization (CRITICAL)
# ----------------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------
# Save split data
# ----------------------------------

SPLIT_DIR = BASE_DIR / "SplitData"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

np.save(SPLIT_DIR / "X_train.npy", X_train)
np.save(SPLIT_DIR / "y_train.npy", y_train)
np.save(SPLIT_DIR / "X_test.npy", X_test)
np.save(SPLIT_DIR / "y_test.npy", y_test)

print(f"Split data saved in {SPLIT_DIR}")



# ----------------------------------
# Train SVM
# ----------------------------------

clf = SVC(
    kernel="linear",          # ⬅️ better for texture features
    C=0.1,
    gamma="scale",
    class_weight="balanced",
    probability=True
)
# clf =SVC(kernel='rbf', C=10, gamma='scale', probability=True)

clf.fit(X_train, y_train)

# ----------------------------------
# Evaluation
# ----------------------------------

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc * 100:.2f}%\n")

# ----------------------------------
# Save model + scaler
# ----------------------------------

MODEL_DIR = BASE_DIR / "Models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

joblib.dump(
    {"model": clf, "scaler": scaler},
    MODEL_DIR / f"LDmodel_SVM_{timestamp}.pkl"
)

print("Model + scaler saved")
