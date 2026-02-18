import numpy as np
import cv2

# ==========================================================
# Enhanced BSIF Filter Bank (5x5, 8 filters)
# ==========================================================
BSIF_5x5_8 = np.array([
    [[0, 0, 0, 1, 1],
     [0, 0, 1, 1, 1],
     [0, 1, 1, 1, 0],
     [1, 1, 1, 0, 0],
     [1, 1, 0, 0, 0]],

    [[1, 1, 1, 1, 0],
     [1, 1, 1, 0, 0],
     [1, 1, 0, 0, 0],
     [1, 0, 0, 0, 1],
     [0, 0, 0, 1, 1]],

    [[0, 0, 1, 1, 1],
     [0, 1, 1, 1, 1],
     [1, 1, 1, 1, 0],
     [1, 1, 1, 0, 0],
     [1, 1, 0, 0, 0]],

    [[1, 1, 1, 0, 0],
     [1, 1, 0, 0, 0],
     [1, 0, 0, 0, 1],
     [0, 0, 0, 1, 1],
     [0, 0, 1, 1, 1]],

    [[1, 1, 0, 0, 0],
     [1, 0, 0, 0, 1],
     [0, 0, 0, 1, 1],
     [0, 0, 1, 1, 1],
     [0, 1, 1, 1, 1]],

    [[1, 0, 0, 0, 1],
     [0, 0, 0, 1, 1],
     [0, 0, 1, 1, 1],
     [0, 1, 1, 1, 1],
     [1, 1, 1, 1, 1]],

    [[0, 0, 0, 1, 1],
     [0, 0, 1, 1, 1],
     [0, 1, 1, 1, 0],
     [1, 1, 1, 0, 0],
     [1, 1, 0, 0, 0]],

    [[1, 1, 1, 1, 1],
     [1, 1, 1, 1, 0],
     [1, 1, 1, 0, 0],
     [1, 1, 0, 0, 0],
     [1, 0, 0, 0, 0]]
]).astype(np.float32)


# ==========================================================
# BSIF Feature Extractor (Improved)
# ==========================================================
def bsif(image_gray):
    """
    Enhanced BSIF with:
    - Z-score normalization
    - High-pass enhancement
    - Visible preview output
    """

    img = image_gray.astype(np.float32)

    # Normalize: removes lighting variation
    img = (img - np.mean(img)) / (np.std(img) + 1e-6)

    H, W = img.shape
    response_stack = []

    # Convolution using OpenCV (faster than scipy)
    for filt in BSIF_5x5_8:
        filtered = cv2.filter2D(img, -1, filt[::-1, ::-1])
        response_stack.append(filtered)

    responses = np.array(response_stack)

    # Binary coding
    binary = (responses > 0).astype(np.uint8)

    bsif_code = np.zeros((H, W), dtype=np.uint8)
    for i in range(8):
        bsif_code |= (binary[i] << i)

    # Histogram (feature vector)
    hist, _ = np.histogram(bsif_code, bins=256, range=(0, 255))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-7)

    # ===== NEW: Visible BSIF Preview =====
    # Use magnitude of filter responses to show skin texture clearly
    magnitude = np.mean(np.abs(responses), axis=0)
    preview = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    return preview, hist
