# LPQ.py
import numpy as np
from scipy.signal import convolve2d

def lpq(img, win_size=3):
    """
    Local Phase Quantization (LPQ)

    Parameters:
        img : 2D numpy array (grayscale image)
        win_size : int, window size (typically 3 or 5)

    Returns:
        lpq_img : 2D numpy array of LPQ codes
    """
    # Ensure image is float32
    img = img.astype(np.float32)
    STFTalpha = 1.0 / win_size  # Short-term Fourier transform alpha
    x = np.arange(-(win_size // 2), win_size // 2 + 1)

    # Define 1D filters
    w0 = np.ones_like(x)
    w1 = np.exp(-2j * np.pi * STFTalpha * x)
    w2 = np.conj(w1)

    # 2D convolution with separable filters
    f = np.stack([
        convolve2d(img, np.outer(w0, w1), mode='same'),
        convolve2d(img, np.outer(w1, w0), mode='same'),
        convolve2d(img, np.outer(w1, w1), mode='same'),
        convolve2d(img, np.outer(w1, w2), mode='same')
    ], axis=-1)

    # Quantize phase to 0 or 1
    lpq_code = (f.real > 0).astype(np.uint8)

    # Combine 4-bit codes into single integer per pixel
    lpq_img = (lpq_code[..., 0] << 3) | (lpq_code[..., 1] << 2) | \
              (lpq_code[..., 2] << 1) | lpq_code[..., 3]

    return lpq_img
