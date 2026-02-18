import numpy as np
import cv2

def wld(image_gray, bins_T=16, bins_M=8, k=0.2):
    """
    Enhanced Weber Local Descriptor (WLD)
    - More robust to lighting
    - Better micro-texture contrast
    - Clear preview for visualization

    Returns:
        preview   : visible WLD visualization (128x128)
        feature   : normalized WLD histogram feature (bins_T + bins_M)
    """

    # ----- 1) Pre-Smoothing (keeps texture, removes noise) -----
    img = cv2.GaussianBlur(image_gray, (3, 3), 0).astype(np.float32)

    # ----- 2) Gradients -----
    # Sobel is the core part that calculates the actual neighbor-center difference used to form the final texture feature.
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    # Magnitude + Orientation
    M = np.sqrt(gx**2 + gy**2)
    theta = np.arctan2(gy, gx + 1e-6) * (180.0 / np.pi)
    theta[theta < 0] += 180.0

    # ----- 3) Weber Contrast (core WLD strength) -----
    I = img + 1e-6
    # gx tells how brightness changes left ↔ right
    # gy tells how brightness changes up ↕ down
    # So to get the total local change, we combine both:
    diff = gx + gy
    T = np.arctan(diff / (I * k + 1e-6)) * (180.0 / np.pi)
    T[T < 0] += 180.0

    # ----- 4) Histogram Feature -----
    hist_T, _ = np.histogram(T, bins=bins_T, range=(0, 180))
    hist_M, _ = np.histogram(M, bins=bins_M, range=(0, np.max(M) + 1e-6))

    hist_T = hist_T.astype(np.float32) / (hist_T.sum() + 1e-6)
    hist_M = hist_M.astype(np.float32) / (hist_M.sum() + 1e-6)

    feature = np.hstack([hist_T, hist_M])

    # ----- 5) Preview (Important!) -----
    # This makes the face visible with texture details
    preview = np.sqrt(gx**2 + gy**2)  # edge + depth texture
    preview = cv2.normalize(preview, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    return preview, feature
