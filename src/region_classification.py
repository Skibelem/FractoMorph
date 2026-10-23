"""
CAMMP-SFSE: Classification Module

Classify each refined closed region into Surface (f_void = 0),
Void (f_void >= tau_void), or Mixed (0 < f_void < tau_void).
"""

import numpy as np

# Region class constants
CLASS_SURFACE = "Surface"
CLASS_VOID = "Void"
CLASS_MIXED = "Mixed"

# Color mapping for visualization (BGR format for OpenCV)
CLASS_COLORS_BGR = {
    CLASS_SURFACE: (255, 0, 0),    # Blue
    CLASS_VOID: (0, 0, 255),       # Red
    CLASS_MIXED: (0, 255, 255),    # Yellow
}

# Color mapping (RGB format for matplotlib)
CLASS_COLORS_RGB = {
    CLASS_SURFACE: (0.0, 0.0, 1.0),    # Blue
    CLASS_VOID: (1.0, 0.0, 0.0),       # Red
    CLASS_MIXED: (1.0, 1.0, 0.0),      # Yellow
}


def classify_region(f_void: float, tau_void: float = 0.50) -> str:
    """Classify a single region based on its void fraction.
    
    Parameters
    ----------
    f_void : float
        Fraction of region pixels covered by extended voids (0.0-1.0).
    tau_void : float
        Void dominance threshold. Regions with f_void >= tau_void
        are classified as Void.
    
    Returns
    -------
    str
        One of 'Surface', 'Void', or 'Mixed'.
    """
    if f_void <= 0.0:
        return CLASS_SURFACE
    elif f_void >= tau_void:
        return CLASS_VOID
    else:
        return CLASS_MIXED


def classify_all_regions(void_fractions: dict,
                         tau_void: float = 0.50) -> dict:
    """Classify all labeled regions.
    
    Parameters
    ----------
    void_fractions : dict
        Mapping from region label to void fraction.
    tau_void : float
        Void dominance threshold.
    
    Returns
    -------
    dict
        Mapping from region label to class string.
    """
    return {
        label: classify_region(frac, tau_void)
        for label, frac in void_fractions.items()
    }


def create_classification_mask(label_map: np.ndarray,
                               classifications: dict) -> np.ndarray:
    """Create a color-coded classification mask image.
    
    Parameters
    ----------
    label_map : np.ndarray
        2D int32 region label map.
    classifications : dict
        Mapping from region label to class string.
    
    Returns
    -------
    np.ndarray
        3-channel uint8 color image (RGB).
        Void = Red, Mixed = Yellow, Surface = Blue.
        Boundaries (label 0) = White.
    """
    h, w = label_map.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Set boundary pixels to white
    color_mask[label_map == 0] = [255, 255, 255]
    
    for label, cls in classifications.items():
        region_mask = label_map == label
        rgb = CLASS_COLORS_RGB[cls]
        color_mask[region_mask] = [
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255),
        ]
    
    return color_mask


def compute_class_summary(classifications: dict,
                          label_map: np.ndarray) -> dict:
    """Compute summary statistics for each class.
    
    Parameters
    ----------
    classifications : dict
        Mapping from region label to class string.
    label_map : np.ndarray
        Region label map.
    
    Returns
    -------
    dict
        Summary with keys:
        - 'counts': dict of class -> count
        - 'areas': dict of class -> total pixel area
        - 'fractions': dict of class -> area fraction
    """
    counts = {CLASS_SURFACE: 0, CLASS_VOID: 0, CLASS_MIXED: 0}
    areas = {CLASS_SURFACE: 0, CLASS_VOID: 0, CLASS_MIXED: 0}
    
    for label, cls in classifications.items():
        counts[cls] += 1
        areas[cls] += int(np.sum(label_map == label))
    
    total_area = sum(areas.values())
    if total_area == 0:
        fractions = {cls: 0.0 for cls in counts}
    else:
        fractions = {
            cls: float(a) / float(total_area)
            for cls, a in areas.items()
        }
    
    return {
        "counts": counts,
        "areas": areas,
        "fractions": fractions,
    }
