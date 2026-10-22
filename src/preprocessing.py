"""
CAMMP-SFSE: Preprocessing Module

256-bin histogram computation and smoothing, gentle edge-preserving
denoising (bilateral/median) that preserves dimple rim gradients.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d


def compute_histogram(gray: np.ndarray, bins: int = 256) -> np.ndarray:
    """Compute the grayscale intensity histogram.
    
    Parameters
    ----------
    gray : np.ndarray
        2D uint8 grayscale image.
    bins : int
        Number of histogram bins (default 256).
    
    Returns
    -------
    np.ndarray
        1D array of length `bins` with pixel counts per intensity.
    """
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    return hist.flatten().astype(np.float64)


def smooth_histogram(hist: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Smooth a histogram using a Gaussian kernel.
    
    Parameters
    ----------
    hist : np.ndarray
        1D histogram array.
    sigma : float
        Standard deviation for Gaussian smoothing.
    
    Returns
    -------
    np.ndarray
        Smoothed histogram.
    """
    return gaussian_filter1d(hist.astype(np.float64), sigma=sigma)


def denoise_bilateral(gray: np.ndarray, d: int = 5,
                      sigma_color: float = 25.0,
                      sigma_space: float = 25.0) -> np.ndarray:
    """Apply gentle bilateral filtering for edge-preserving denoising.
    
    The bilateral filter smooths flat regions while preserving sharp
    intensity gradients around dimple rims and tear ridges.
    
    Parameters
    ----------
    gray : np.ndarray
        2D uint8 grayscale image.
    d : int
        Diameter of pixel neighborhood (use -1 for auto from sigma_space).
    sigma_color : float
        Filter sigma in the intensity (color) space.
    sigma_space : float
        Filter sigma in the coordinate space.
    
    Returns
    -------
    np.ndarray
        Denoised uint8 grayscale image.
    """
    return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)


def denoise_median(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Apply median filtering for salt-and-pepper noise removal.
    
    Parameters
    ----------
    gray : np.ndarray
        2D uint8 grayscale image.
    ksize : int
        Kernel size (must be odd, >= 3).
    
    Returns
    -------
    np.ndarray
        Denoised uint8 grayscale image.
    """
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(gray, ksize)


def preprocess(gray: np.ndarray, use_bilateral: bool = True,
               bilateral_d: int = 5, sigma_color: float = 25.0,
               sigma_space: float = 25.0, median_ksize: int = 3,
               hist_sigma: float = 2.0):
    """Run full preprocessing: compute histogram then gentle denoise.
    
    Parameters
    ----------
    gray : np.ndarray
        2D uint8 grayscale image.
    use_bilateral : bool
        If True, use bilateral filter; else use median filter.
    bilateral_d : int
        Bilateral filter diameter.
    sigma_color : float
        Bilateral filter color sigma.
    sigma_space : float
        Bilateral filter spatial sigma.
    median_ksize : int
        Median filter kernel size.
    hist_sigma : float
        Gaussian sigma for histogram smoothing.
    
    Returns
    -------
    dict
        Keys: 'denoised' (uint8 image), 'histogram' (raw 256-bin),
        'histogram_smooth' (smoothed histogram).
    """
    hist_raw = compute_histogram(gray)
    hist_smooth = smooth_histogram(hist_raw, sigma=hist_sigma)
    
    if use_bilateral:
        denoised = denoise_bilateral(gray, bilateral_d, sigma_color, sigma_space)
    else:
        denoised = denoise_median(gray, median_ksize)
    
    return {
        "denoised": denoised,
        "histogram": hist_raw,
        "histogram_smooth": hist_smooth,
    }
