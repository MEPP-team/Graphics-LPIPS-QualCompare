import numpy as np
from scipy import signal

def _gaussian_window(window_size=11, sigma=1.5):
    k = np.arange(window_size, dtype=np.float64) - (window_size - 1) / 2.0
    x, y = np.meshgrid(k, k)
    w = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return w / w.sum()


def ssim(img1, img2, K=None, window=None, L=255):
    """Faithful Python transcription of root `ssim.m` (Zhou Wang v1.0)."""
    if K is None:
        K = [0.01, 0.03]

    if img1.shape != img2.shape:
        return -np.inf, -np.inf

    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError("ssim() expects 2D grayscale arrays")

    M, N = img1.shape
    if M < 11 or N < 11:
        return -np.inf, -np.inf

    if window is None:
        window = _gaussian_window(11, 1.5)

    H, W = window.shape
    if (H * W) < 4 or H > M or W > N:
        return -np.inf, -np.inf

    if len(K) != 2 or K[0] < 0 or K[1] < 0:
        return -np.inf, -np.inf

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Automatic downsampling exactly as in ssim.m
    f = max(1, int(round(min(M, N) / 256.0)))
    if f > 1:
        lpf = np.ones((f, f), dtype=np.float64)
        lpf = lpf / np.sum(lpf)
        img1 = signal.convolve2d(img1, lpf, mode='same', boundary='symm')
        img2 = signal.convolve2d(img2, lpf, mode='same', boundary='symm')
        img1 = img1[::f, ::f]
        img2 = img2[::f, ::f]

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2
    window = window / np.sum(window)

    mu1 = signal.convolve2d(img1, window, mode='valid')
    mu2 = signal.convolve2d(img2, window, mode='valid')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = signal.convolve2d(img1 * img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.convolve2d(img2 * img2, window, mode='valid') - mu2_sq
    sigma12 = signal.convolve2d(img1 * img2, window, mode='valid') - mu1_mu2

    if C1 > 0 and C2 > 0:
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
    else:
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2
        ssim_map = np.ones(mu1.shape, dtype=np.float64)
        index = (denominator1 * denominator2) > 0
        ssim_map[index] = (numerator1[index] * numerator2[index]) / (denominator1[index] * denominator2[index])
        index = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[index] = numerator1[index] / denominator1[index]

    mssim = np.mean(ssim_map)
    return mssim, ssim_map


def cal_ssim(img1, img2):
    """Compatibility alias kept for minimal disruption with online snippets."""
    return ssim(img1, img2)