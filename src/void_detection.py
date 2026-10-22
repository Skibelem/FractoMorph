"""
CAMMP-SFSE: Void Detection Module

Stage A: Void-Tip Detection by locating the first local maximum/cutoff
in the dark low-intensity histogram regime.
Stage B: Void-Tip Extension by growing void seeds locally within
closed contours while distinguishing deep vs. shallow voids.
"""

import numpy as np
import cv2
from scipy.signal import argrelextrema
from scipy import ndimage


def find_void_tip_intensity(hist_smooth: np.ndarray,
                            search_range: tuple = (5, 100),
                            min_prominence: float = 0.01) -> int:
    """Stage A: Find the void-tip cutoff intensity from the smoothed histogram.
    
    Analyzes the low-intensity (dark) regime of the histogram to locate
    the first local maximum corresponding to deep void bottoms.
    
    Parameters
    ----------
    hist_smooth : np.ndarray
        Smoothed 256-bin grayscale histogram.
    search_range : tuple of (int, int)
        Intensity range (lo, hi) to search for the void-tip peak.
    min_prominence : float
        Minimum relative prominence (fraction of max histogram value)
        for a peak to be considered.
    
    Returns
    -------
    int
        Void-tip cutoff intensity I_tip (0-255).
    """
    lo, hi = search_range
    # Normalize histogram for consistent analysis
    h = hist_smooth.copy()
    h_max = h.max()
    if h_max == 0:
        return lo
    
    # Look for local maxima in the dark regime
    local_max_indices = argrelextrema(h[lo:hi], np.greater, order=3)[0]
    
    if len(local_max_indices) > 0:
        # Offset by search_range start
        candidates = local_max_indices + lo
        # Filter by prominence
        prominent = [
            idx for idx in candidates
            if h[idx] > min_prominence * h_max
        ]
        
        if prominent:
            # Return the first prominent peak in the dark regime
            return int(prominent[0])
    
    # Fallback: find the first local minimum after the dark tail,
    # which indicates the transition from void bottoms to surface
    local_min_indices = argrelextrema(h[lo:hi], np.less, order=5)[0]
    if len(local_min_indices) > 0:
        return int(local_min_indices[0] + lo)
    
    # Last fallback: use 15th percentile of the histogram CDF
    cdf = np.cumsum(h)
    cdf_norm = cdf / cdf[-1]
    tip_idx = np.searchsorted(cdf_norm, 0.15)
    return int(np.clip(tip_idx, lo, hi))


def create_void_tip_seeds(gray: np.ndarray, i_tip: int) -> np.ndarray:
    """Create initial void-tip seed mask from intensity cutoff.
    
    All pixels with intensity <= I_tip are marked as void-tip seeds.
    
    Parameters
    ----------
    gray : np.ndarray
        2D uint8 grayscale image.
    i_tip : int
        Void-tip cutoff intensity.
    
    Returns
    -------
    np.ndarray
        Binary void-tip seed mask (uint8, 0 or 255).
    """
    seeds = (gray <= i_tip).astype(np.uint8) * 255
    return seeds


def extend_void_seeds(gray: np.ndarray, seeds: np.ndarray,
                      label_map: np.ndarray, num_regions: int,
                      i_tip: int, extension_factor: float = 1.5,
                      gradient_threshold: float = 15.0) -> np.ndarray:
    """Stage B: Extend void-tip seeds outward within closed contours.
    
    Iteratively grow void regions from seeds into neighboring pixels
    that have gentle intensity gradients and remain below the upper
    void cutoff I_ext = i_tip * extension_factor.
    
    Parameters
    ----------
    gray : np.ndarray
        2D uint8 grayscale image.
    seeds : np.ndarray
        Initial void-tip seed mask (uint8).
    label_map : np.ndarray
        Region label map.
    num_regions : int
        Number of regions.
    i_tip : int
        Void-tip cutoff intensity.
    extension_factor : float
        Multiplier for upper void extension cutoff.
    gradient_threshold : float
        Maximum local gradient magnitude for seed extension.
    
    Returns
    -------
    np.ndarray
        Extended void mask (uint8, 0 or 255).
    """
    i_ext = min(int(i_tip * extension_factor), 255)
    
    # Compute gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    extended = seeds.copy()
    
    # Iterative dilation within constraints
    struct = ndimage.generate_binary_structure(2, 2)  # 8-connectivity
    
    for _ in range(20):  # Max iterations
        # Dilate by one pixel
        dilated = ndimage.binary_dilation(
            extended > 0, structure=struct
        ).astype(np.uint8) * 255
        
        # Candidate pixels: newly added by dilation
        candidates = (dilated > 0) & (extended == 0)
        
        if not np.any(candidates):
            break
        
        # Apply constraints:
        # 1. Must be within the same region (not on boundaries)
        valid_region = label_map > 0
        # 2. Must have intensity below I_ext
        valid_intensity = gray <= i_ext
        # 3. Must have gentle gradient
        valid_gradient = gradient_mag < gradient_threshold
        
        # Accept only candidates meeting all constraints
        accepted = candidates & valid_region & valid_intensity & valid_gradient
        
        if not np.any(accepted):
            break
        
        extended[accepted] = 255
    
    return extended


def compute_void_fractions(extended_voids: np.ndarray,
                           label_map: np.ndarray,
                           num_regions: int) -> dict:
    """Compute the void fraction for each labeled region.
    
    Parameters
    ----------
    extended_voids : np.ndarray
        Extended void mask (uint8, 0 or 255).
    label_map : np.ndarray
        Region label map.
    num_regions : int
        Number of labeled regions.
    
    Returns
    -------
    dict
        Mapping from region label to void fraction (0.0-1.0).
    """
    fractions = {}
    void_bool = extended_voids > 0
    
    for label in range(1, num_regions + 1):
        region_mask = label_map == label
        region_area = np.sum(region_mask)
        
        if region_area == 0:
            fractions[label] = 0.0
            continue
        
        void_area = np.sum(region_mask & void_bool)
        fractions[label] = float(void_area) / float(region_area)
    
    return fractions


def detect_voids(gray: np.ndarray, hist_smooth: np.ndarray,
                 label_map: np.ndarray, num_regions: int,
                 extension_factor: float = 1.5,
                 gradient_threshold: float = 15.0) -> dict:
    """Run the full void detection pipeline (Stage A + Stage B).
    
    Parameters
    ----------
    gray : np.ndarray
        2D uint8 grayscale image.
    hist_smooth : np.ndarray
        Smoothed 256-bin histogram.
    label_map : np.ndarray
        Refined region label map.
    num_regions : int
        Number of regions.
    extension_factor : float
        Multiplier for upper void extension cutoff.
    gradient_threshold : float
        Maximum gradient magnitude for void extension.
    
    Returns
    -------
    dict
        Keys:
        - 'i_tip': Void-tip cutoff intensity
        - 'void_seeds': Initial void-tip seed mask
        - 'void_extended': Extended void mask
        - 'void_fractions': Per-region void fractions
    """
    # Stage A: Find void-tip intensity
    i_tip = find_void_tip_intensity(hist_smooth)
    
    # Create void-tip seeds
    void_seeds = create_void_tip_seeds(gray, i_tip)
    
    # Stage B: Extend void seeds
    void_extended = extend_void_seeds(
        gray, void_seeds, label_map, num_regions,
        i_tip, extension_factor, gradient_threshold
    )
    
    # Compute per-region void fractions
    void_fractions = compute_void_fractions(
        void_extended, label_map, num_regions
    )
    
    return {
        "i_tip": i_tip,
        "void_seeds": void_seeds,
        "void_extended": void_extended,
        "void_fractions": void_fractions,
    }
