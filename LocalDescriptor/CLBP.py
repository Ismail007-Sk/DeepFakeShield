# clbp.py
import cv2
import numpy as np

def clbp(image, radius=1, neighbors=8):
    """
    Normal (single-scale) CLBP implementation
    Components:
    - CLBP_S : Sign
    - CLBP_M : Magnitude
    - CLBP_C : Center
    """

    img = image.astype(np.float32)
    h, w = img.shape

    clbp_s = np.zeros((h, w), dtype=np.uint8)
    clbp_m = np.zeros((h, w), dtype=np.uint8)

    angles = 2 * np.pi * np.arange(neighbors) / neighbors
    dx = radius * np.cos(angles)
    dy = -radius * np.sin(angles)

    center = img

    # ---- Mean magnitude threshold (global) ----
    mag_all = []

    for i in range(neighbors):
        shifted = cv2.remap(
            img,
            (np.arange(w) + dx[i]).astype(np.float32),
            (np.arange(h)[:, None] + dy[i]).astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        mag_all.append(np.abs(shifted - center))

    mean_mag = np.mean(mag_all)

    # ---- CLBP_S and CLBP_M ----
    for i in range(neighbors):
        shifted = cv2.remap(
            img,
            (np.arange(w) + dx[i]).astype(np.float32),
            (np.arange(h)[:, None] + dy[i]).astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        diff = shifted - center

        clbp_s |= ((diff >= 0).astype(np.uint8)) << i
        clbp_m |= ((np.abs(diff) >= mean_mag).astype(np.uint8)) << i

    # ---- CLBP_C ----
    clbp_c = (center >= np.mean(center)).astype(np.uint8)

    return clbp_s, clbp_m, clbp_c


def clbp_feature(image):
    """
    Final CLBP feature vector (histogram-based)
    """

    s, m, c = clbp(image)

    hist_s, _ = np.histogram(s.ravel(), bins=256, range=(0, 256))
    hist_m, _ = np.histogram(m.ravel(), bins=256, range=(0, 256))
    hist_c, _ = np.histogram(c.ravel(), bins=2, range=(0, 2))

    feature = np.concatenate([hist_s, hist_m, hist_c]).astype(np.float32)
    feature /= (np.linalg.norm(feature) + 1e-6)

    return feature