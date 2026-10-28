"""
CAMMP-SFSE: Automated Thresholding Engine

Provides modular thresholding methods for separating bright tear ridges from
darker dimple/valley regions in SEM fractographs.

Supported methods:
- "otsu": Otsu's maximum between-class variance thresholding
- "yen": Yen's maximum correlation criterion thresholding
- "triangle": Zack's triangle geometric thresholding
- "li": Li's minimum cross-entropy thresholding
- "adaptive": Local adaptive Gaussian threshold surface T(x, y)
- "gmm": Gaussian Mixture Model intensity clustering with posterior probability cutoff

Reference:
    Sezgin, M., & Sankur, B. (2004). Survey over image thresholding
    techniques and quantitative performance evaluation. Journal of Electronic
    Imaging, 13(1), 146-165.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
from skimage.filters import (
    threshold_li,
    threshold_local,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)
from sklearn.mixture import GaussianMixture

SUPPORTED_METHODS = ("otsu", "yen", "triangle", "li", "adaptive", "gmm")


def _is_uniform_or_constant(img: np.ndarray) -> bool:
    """Check if the image is empty, uniform, or has zero variance."""
    if img.size == 0:
        return True
    return bool(np.min(img) == np.max(img))


def _safe_scalar_fallback(img: np.ndarray) -> float:
    """Return a safe fallback threshold value for uniform or pathological images."""
    if img.size == 0:
        return 0.0
    return float(np.mean(img))


def threshold_otsu_method(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """Apply Otsu's global thresholding.

    Parameters
    ----------
    img : np.ndarray
        Grayscale input image.

    Returns
    -------
    tuple of (np.ndarray, float)
        Binary mask (uint8, 0 or 255) and scalar threshold value.
    """
    if _is_uniform_or_constant(img):
        t = _safe_scalar_fallback(img)
        return (img > t).astype(np.uint8) * 255, t

    try:
        t = float(threshold_otsu(img))
    except (ValueError, RuntimeError):
        t = _safe_scalar_fallback(img)

    binary = (img > t).astype(np.uint8) * 255
    return binary, t


def threshold_yen_method(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """Apply Yen's maximum correlation criterion thresholding.

    Parameters
    ----------
    img : np.ndarray
        Grayscale input image.

    Returns
    -------
    tuple of (np.ndarray, float)
        Binary mask (uint8, 0 or 255) and scalar threshold value.
    """
    if _is_uniform_or_constant(img):
        t = _safe_scalar_fallback(img)
        return (img > t).astype(np.uint8) * 255, t

    try:
        t = float(threshold_yen(img))
    except (ValueError, RuntimeError):
        t = _safe_scalar_fallback(img)

    binary = (img > t).astype(np.uint8) * 255
    return binary, t


def threshold_triangle_method(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """Apply Zack's triangle thresholding.

    Parameters
    ----------
    img : np.ndarray
        Grayscale input image.

    Returns
    -------
    tuple of (np.ndarray, float)
        Binary mask (uint8, 0 or 255) and scalar threshold value.
    """
    if _is_uniform_or_constant(img):
        t = _safe_scalar_fallback(img)
        return (img > t).astype(np.uint8) * 255, t

    try:
        t = float(threshold_triangle(img))
    except (ValueError, RuntimeError):
        t = _safe_scalar_fallback(img)

    binary = (img > t).astype(np.uint8) * 255
    return binary, t


def threshold_li_method(img: np.ndarray) -> Tuple[np.ndarray, float]:
    """Apply Li's minimum cross-entropy thresholding.

    Parameters
    ----------
    img : np.ndarray
        Grayscale input image.

    Returns
    -------
    tuple of (np.ndarray, float)
        Binary mask (uint8, 0 or 255) and scalar threshold value.
    """
    if _is_uniform_or_constant(img):
        t = _safe_scalar_fallback(img)
        return (img > t).astype(np.uint8) * 255, t

    try:
        t = float(threshold_li(img))
        if np.isnan(t) or np.isinf(t):
            t = _safe_scalar_fallback(img)
    except (ValueError, RuntimeError):
        t = _safe_scalar_fallback(img)

    binary = (img > t).astype(np.uint8) * 255
    return binary, t


def threshold_adaptive_method(
    img: np.ndarray,
    block_size: int = 35,
    offset: float = -5.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Apply local adaptive Gaussian thresholding.

    Calculates a threshold surface T(x, y) = Gaussian(img) - offset.
    block_size is automatically enforced to be an odd integer >= 3 and capped
    to the minimum spatial dimension of img.

    Parameters
    ----------
    img : np.ndarray
        Grayscale input image.
    block_size : int, default=35
        Neighborhood size for Gaussian filter (must be odd >= 3).
    offset : float, default=-5.0
        Constant subtracted from weighted mean.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, float)
        Binary mask (uint8, 0 or 255), 2D threshold surface, and mean threshold summary.
    """
    if img.size == 0:
        empty = np.zeros_like(img, dtype=np.uint8)
        empty_t = np.zeros_like(img, dtype=np.float64)
        return empty, empty_t, 0.0

    if _is_uniform_or_constant(img):
        t_scalar = _safe_scalar_fallback(img)
        t_surf = np.full(img.shape, t_scalar, dtype=np.float64)
        binary = (img > t_surf).astype(np.uint8) * 255
        return binary, t_surf, t_scalar

    h, w = img.shape[:2]
    min_dim = min(h, w)

    if min_dim < 3:
        t_scalar = _safe_scalar_fallback(img)
        t_surf = np.full(img.shape, t_scalar, dtype=np.float64)
        binary = (img > t_surf).astype(np.uint8) * 255
        return binary, t_surf, t_scalar

    b = int(block_size)
    if b < 3:
        b = 3
    if b % 2 == 0:
        b += 1
    if b > min_dim:
        b = min_dim if (min_dim % 2 == 1) else min_dim - 1
    if b < 3:
        b = 3

    try:
        t_surf = threshold_local(
            img,
            block_size=b,
            method="gaussian",
            offset=float(offset),
        )
    except Exception:
        t_scalar = _safe_scalar_fallback(img)
        t_surf = np.full(img.shape, t_scalar, dtype=np.float64)

    binary = (img > t_surf).astype(np.uint8) * 255
    t_summary = float(np.mean(t_surf))
    return binary, t_surf, t_summary


def threshold_gmm_method(
    img: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
    max_iter: int = 200,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """Apply Gaussian Mixture Model (GMM) thresholding.

    Fits an n-component GMM on flattened pixel intensities, sorts components
    by mean intensity, and searches candidate integer intensities between the
    second-highest and highest component means where the highest component's
    posterior probability takes over.

    Parameters
    ----------
    img : np.ndarray
        Grayscale input image.
    n_components : int, default=2
        Number of Gaussian components to fit.
    random_state : int, default=42
        Random seed for reproducible GMM fitting.
    max_iter : int, default=200
        Maximum EM iterations.
    max_samples : int or None, optional
        Maximum number of pixel intensity samples used for fitting GMM (accelerates sweeps).

    Returns
    -------
    tuple of (np.ndarray, float)
        Binary mask (uint8, 0 or 255) and scalar cutoff threshold.
    """
    if _is_uniform_or_constant(img) or img.size < max(n_components, 2):
        t = _safe_scalar_fallback(img)
        return (img > t).astype(np.uint8) * 255, t

    pixels = img.astype(np.float64).reshape(-1, 1)

    try:
        gmm = GaussianMixture(
            n_components=int(n_components),
            random_state=random_state,
            max_iter=max_iter,
        )
        if max_samples is not None and pixels.shape[0] > int(max_samples):
            rng = np.random.RandomState(random_state)
            sample_indices = rng.choice(pixels.shape[0], size=int(max_samples), replace=False)
            gmm.fit(pixels[sample_indices])
        else:
            gmm.fit(pixels)

        means = gmm.means_.flatten()
        sorted_indices = np.argsort(means)
        highest_idx = int(sorted_indices[-1])
        second_highest_idx = int(sorted_indices[-2])

        mu_highest = float(means[highest_idx])
        mu_second_highest = float(means[second_highest_idx])
        fallback_threshold = (mu_second_highest + mu_highest) / 2.0

        low = int(np.floor(mu_second_highest))
        high = int(np.ceil(mu_highest))

        if low >= high:
            t = float(fallback_threshold)
        else:
            x_candidates = np.arange(low, high + 1, dtype=np.float64).reshape(-1, 1)
            probs = gmm.predict_proba(x_candidates)

            # Cutoff where highest-mean component takes over
            takes_over = (probs[:, highest_idx] >= probs[:, second_highest_idx]) & (
                np.argmax(probs, axis=1) == highest_idx
            )
            takeover_indices = np.where(takes_over)[0]

            if len(takeover_indices) > 0:
                t = float(x_candidates[takeover_indices[0], 0])
            else:
                t = float(fallback_threshold)

    except Exception:
        t = _safe_scalar_fallback(img)

    binary = (img > t).astype(np.uint8) * 255
    return binary, t


def apply_threshold(
    img: np.ndarray,
    method: str = "otsu",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Apply a selected thresholding method to an image.

    Parameters
    ----------
    img : np.ndarray
        2D grayscale input image.
    method : str, default="otsu"
        Thresholding method: "otsu", "yen", "triangle", "li", "adaptive", "gmm".
    **kwargs : Any
        Method-specific keyword arguments:
        - adaptive: block_size (or adaptive_block_size), offset (or adaptive_offset)
        - gmm: n_components (or gmm_components), random_state, max_iter

    Returns
    -------
    dict
        - "binary": 2D uint8 foreground ridge mask (0 or 255, where img > threshold)
        - "threshold": float scalar for global/GMM methods, or 2D np.ndarray surface for "adaptive"
        - "threshold_summary": float (scalar threshold or mean of threshold surface)
        - "runtime_ms": execution time in milliseconds measured via time.perf_counter()
        - "method": normalized method name
    """
    norm_method = str(method).strip().lower()
    if norm_method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported thresholding method '{method}'. "
            f"Supported methods are: {list(SUPPORTED_METHODS)}"
        )

    t0 = time.perf_counter()

    if norm_method == "otsu":
        binary, thresh = threshold_otsu_method(img)
        thresh_summary = thresh
    elif norm_method == "yen":
        binary, thresh = threshold_yen_method(img)
        thresh_summary = thresh
    elif norm_method == "triangle":
        binary, thresh = threshold_triangle_method(img)
        thresh_summary = thresh
    elif norm_method == "li":
        binary, thresh = threshold_li_method(img)
        thresh_summary = thresh
    elif norm_method == "adaptive":
        block_size = kwargs.get("block_size", kwargs.get("adaptive_block_size", 35))
        offset = kwargs.get("offset", kwargs.get("adaptive_offset", -5.0))
        binary, thresh, thresh_summary = threshold_adaptive_method(
            img, block_size=int(block_size), offset=float(offset)
        )
    elif norm_method == "gmm":
        n_components = kwargs.get(
            "n_components", kwargs.get("gmm_components", 2)
        )
        random_state = kwargs.get("random_state", 42)
        max_iter = kwargs.get("max_iter", 200)
        max_samples = kwargs.get("max_samples", None)
        binary, thresh = threshold_gmm_method(
            img,
            n_components=int(n_components),
            random_state=int(random_state),
            max_iter=int(max_iter),
            max_samples=max_samples,
        )
        thresh_summary = thresh

    t1 = time.perf_counter()
    runtime_ms = float((t1 - t0) * 1000.0)

    return {
        "binary": binary,
        "threshold": thresh,
        "threshold_summary": float(thresh_summary),
        "runtime_ms": runtime_ms,
        "method": norm_method,
    }
