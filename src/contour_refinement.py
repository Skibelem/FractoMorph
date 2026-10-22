"""
CAMMP-SFSE: Contour Refinement Module

Compute region area mean and standard deviation. Flag candidate regions
with A_i > mu_A + sigma_A, apply local re-thresholding inside each
flagged region mask, merge internal sub-contours into the master
skeleton, re-prune, and filter out regions smaller than min_area.
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu


def compute_area_stats(label_map: np.ndarray, num_regions: int) -> dict:
    """Compute area statistics for all labeled regions.
    
    Parameters
    ----------
    label_map : np.ndarray
        2D int32 label map (0 = boundary, 1..N = regions).
    num_regions : int
        Number of regions.
    
    Returns
    -------
    dict
        Keys: 'areas' (dict label->area), 'mean', 'std'
    """
    areas = {}
    for label in range(1, num_regions + 1):
        areas[label] = int(np.sum(label_map == label))
    
    area_values = np.array(list(areas.values()), dtype=np.float64)
    
    if len(area_values) == 0:
        return {"areas": areas, "mean": 0.0, "std": 0.0}
    
    return {
        "areas": areas,
        "mean": float(np.mean(area_values)),
        "std": float(np.std(area_values)),
    }


def find_oversized_regions(area_stats: dict) -> list:
    """Find regions with area > mu_A + sigma_A.
    
    Parameters
    ----------
    area_stats : dict
        Output from compute_area_stats.
    
    Returns
    -------
    list of int
        Labels of oversized regions.
    """
    threshold = area_stats["mean"] + area_stats["std"]
    oversized = [
        label for label, area in area_stats["areas"].items()
        if area > threshold
    ]
    return oversized


def local_rethreshold(gray: np.ndarray, label_map: np.ndarray,
                      region_label: int) -> np.ndarray:
    """Apply local re-thresholding inside a single oversized region.
    
    Extract the local grayscale intensity distribution within the region
    mask, compute a local Otsu threshold, and return a binary sub-skeleton
    of internal boundaries.
    
    Parameters
    ----------
    gray : np.ndarray
        Original 2D uint8 grayscale image.
    label_map : np.ndarray
        Region label map.
    region_label : int
        Label of the region to re-threshold.
    
    Returns
    -------
    np.ndarray
        Binary sub-contour mask (uint8, 0 or 255) for the region,
        same shape as gray.
    """
    mask = (label_map == region_label).astype(np.uint8)
    
    # Extract local intensities
    local_pixels = gray[mask > 0]
    
    if len(local_pixels) < 10:
        return np.zeros_like(gray, dtype=np.uint8)
    
    # Check if there's enough intensity variation for thresholding
    if np.std(local_pixels) < 5:
        return np.zeros_like(gray, dtype=np.uint8)
    
    try:
        local_thresh = threshold_otsu(local_pixels)
    except ValueError:
        return np.zeros_like(gray, dtype=np.uint8)
    
    # Create local binary inside the region mask
    local_binary = np.zeros_like(gray, dtype=np.uint8)
    region_coords = mask > 0
    local_binary[region_coords] = (
        (gray[region_coords] > local_thresh).astype(np.uint8) * 255
    )
    
    # Skeletonize the local binary to get thin sub-contours
    if np.sum(local_binary > 0) < 5:
        return np.zeros_like(gray, dtype=np.uint8)
    
    local_skel = skeletonize(local_binary > 0).astype(np.uint8) * 255
    
    return local_skel


def merge_subcontours(master_skeleton: np.ndarray,
                      sub_skeletons: list) -> np.ndarray:
    """Merge sub-contour skeletons into the master skeleton.
    
    Parameters
    ----------
    master_skeleton : np.ndarray
        Main skeleton mask (uint8, 0 or 255).
    sub_skeletons : list of np.ndarray
        Sub-contour skeletons to merge.
    
    Returns
    -------
    np.ndarray
        Merged skeleton.
    """
    merged = master_skeleton.copy()
    for sub in sub_skeletons:
        merged = np.maximum(merged, sub)
    return merged


def filter_small_regions(label_map: np.ndarray, num_regions: int,
                         min_area: int = 25) -> tuple:
    """Remove regions smaller than min_area from the label map.
    
    Small regions are absorbed into the skeleton (boundary).
    
    Parameters
    ----------
    label_map : np.ndarray
        Region label map.
    num_regions : int
        Number of regions.
    min_area : int
        Minimum allowed region area in pixels.
    
    Returns
    -------
    tuple of (np.ndarray, int)
        Filtered label map (re-labeled sequentially) and new region count.
    """
    filtered = np.zeros_like(label_map)
    new_label = 0
    
    for label in range(1, num_regions + 1):
        region_mask = (label_map == label)
        area = np.sum(region_mask)
        if area >= min_area:
            new_label += 1
            filtered[region_mask] = new_label
    
    return filtered, new_label


def refine_contours(gray: np.ndarray, skeleton: np.ndarray,
                    label_map: np.ndarray, num_regions: int,
                    min_area: int = 25) -> dict:
    """Run the full contour refinement pipeline.
    
    Parameters
    ----------
    gray : np.ndarray
        Original 2D uint8 grayscale image.
    skeleton : np.ndarray
        Initial skeleton from contour detection.
    label_map : np.ndarray
        Initial region label map.
    num_regions : int
        Initial number of regions.
    min_area : int
        Minimum region area to keep.
    
    Returns
    -------
    dict
        Keys:
        - 'skeleton': Refined skeleton
        - 'label_map': Refined label map
        - 'num_regions': Refined region count
        - 'area_stats': Area statistics dict
        - 'oversized_labels': List of originally oversized region labels
    """
    # Import here to avoid circular dependency
    from src.contour_detection import prune_spurs, label_regions
    
    # Compute area stats
    area_stats = compute_area_stats(label_map, num_regions)
    
    # Find oversized regions
    oversized = find_oversized_regions(area_stats)
    
    # Local re-threshold each oversized region
    sub_skeletons = []
    for label in oversized:
        sub_skel = local_rethreshold(gray, label_map, label)
        if np.any(sub_skel > 0):
            sub_skeletons.append(sub_skel)
    
    # Merge sub-contours into master skeleton
    if sub_skeletons:
        refined_skeleton = merge_subcontours(skeleton, sub_skeletons)
        # Re-prune to remove new spurs
        refined_skeleton = prune_spurs(refined_skeleton, max_spur_length=5)
    else:
        refined_skeleton = skeleton.copy()
    
    # Re-label regions
    new_label_map, new_num = label_regions(refined_skeleton)
    
    # Filter small regions
    final_label_map, final_num = filter_small_regions(
        new_label_map, new_num, min_area=min_area
    )
    
    return {
        "skeleton": refined_skeleton,
        "label_map": final_label_map,
        "num_regions": final_num,
        "area_stats": area_stats,
        "oversized_labels": oversized,
    }
