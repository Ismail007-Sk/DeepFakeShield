import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# ----------------------------------
# Load Data
# ----------------------------------
BASE_DIR = Path(__file__).resolve().parent

X = np.load(BASE_DIR / "SplitData" / "X_test.npy")
y = np.load(BASE_DIR / "SplitData" / "y_test.npy")

print("Loaded:", X.shape)

# ----------------------------------
# Scaling (IMPORTANT)
# ----------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------
# PCA Visualization
# ----------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure()
for label in np.unique(y):
    plt.scatter(
        X_pca[y == label, 0],
        X_pca[y == label, 1],
        label="Real" if label == 1 else "Fake",
        alpha=0.6
    )

plt.title("PCA Visualization of Local Descriptor Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()

# ----------------------------------
# t-SNE Visualization (optional)
# ----------------------------------
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure()
for label in np.unique(y):
    plt.scatter(
        X_tsne[y == label, 0],
        X_tsne[y == label, 1],
        label="Real" if label == 1 else "Fake",
        alpha=0.6
    )

plt.title("t-SNE Visualization of Local Descriptor Features")
plt.legend()
plt.show()
