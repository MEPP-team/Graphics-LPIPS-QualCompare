"""SSIM wrapper matching the recommended Wang et al. configuration."""

import numpy as np
from scipy import signal
from skimage.metrics import structural_similarity


_DEFAULT_K = (0.01, 0.03)
_GAUSSIAN_SIGMA = 1.5
_GAUSSIAN_TRUNCATE = 3.5
_SSIM_BORDER = int(_GAUSSIAN_TRUNCATE * _GAUSSIAN_SIGMA + 0.5)


def _matlab_round_positive(value):
    """Match MATLAB round() for the positive values used by ssim.m."""
    return int(np.floor(value + 0.5))


def _automatic_downsample(image):
    """Apply the low-pass filtering and subsampling used by Wang's ssim.m."""
    factor = max(1, _matlab_round_positive(min(image.shape) / 256.0))
    image = image.astype(np.float64)
    if factor == 1:
        return image

    low_pass = np.ones((factor, factor), dtype=np.float64)
    low_pass /= low_pass.sum()
    filtered = signal.convolve2d(image, low_pass, mode="same", boundary="symm")
    return filtered[::factor, ::factor]


def ssim(img1, img2, K=None, window=None, L=255):
    """Compute SSIM using scikit-image configured to match Wang et al.

    Automatic downsampling follows Zhou Wang's ``ssim.m``. Scikit-image then
    computes SSIM with Gaussian weights, sigma 1.5, population covariance, and
    the explicitly supplied image dynamic range.
    """
    if img1.shape != img2.shape:
        return -np.inf, -np.inf
    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError("ssim() expects 2D grayscale arrays")
    if min(img1.shape) < 11:
        return -np.inf, -np.inf
    if window is not None:
        raise ValueError("Custom SSIM windows are not supported by the scikit-image wrapper")

    K = _DEFAULT_K if K is None else K
    if len(K) != 2 or K[0] < 0 or K[1] < 0:
        return -np.inf, -np.inf

    img1 = _automatic_downsample(img1)
    img2 = _automatic_downsample(img2)

    score, full_map = structural_similarity(
        img1,
        img2,
        data_range=L,
        gaussian_weights=True,
        sigma=_GAUSSIAN_SIGMA,
        use_sample_covariance=False,
        K1=K[0],
        K2=K[1],
        full=True,
    )

    # structural_similarity returns a full-size map, while ssim.m returns only
    # positions where the 11x11 Gaussian window is fully contained.
    valid_map = full_map[
        _SSIM_BORDER:-_SSIM_BORDER,
        _SSIM_BORDER:-_SSIM_BORDER,
    ]
    return float(score), valid_map


def cal_ssim(img1, img2):
    """Compatibility alias retained for existing callers."""
    return ssim(img1, img2)
