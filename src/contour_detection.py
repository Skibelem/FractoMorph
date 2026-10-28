"""
CAMMP-SFSE: Contour Detection Module

Global thresholding, binary boundary extraction, morphological
skeletonization (skimage.morphology.skeletonize to 1-px width),
endpoint detection + edge linking across small gaps, spur pruning,
and closed-region connected-component labeling.
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu

from src.thresholding import apply_threshold


def global_threshold(
    gray: np.ndarray,
    method: str = "otsu",
    threshold_kwargs: dict | None = None,
) -> np.ndarray:
    """Apply global or adaptive thresholding to separate ridges from valleys.
    
    Parameters
    ----------
    gray : np.ndarray
        2D uint8 grayscale image.
    method : str, default="otsu"
        Thresholding method ("otsu", "yen", "triangle", "li", "adaptive", "gmm").
    threshold_kwargs : dict or None, optional
        Additional keyword arguments passed to the thresholding method.
    
    Returns
    -------
    np.ndarray
        Binary mask (uint8, 0 or 255). Bright ridges = 255.
    """
    kwargs = threshold_kwargs or {}
    result = apply_threshold(gray, method=method, **kwargs)
    return result["binary"]


def skeletonize_binary(binary: np.ndarray) -> np.ndarray:
    """Thin binary ridge boundaries to single-pixel-wide skeletons.
    
    Uses skimage.morphology.skeletonize for proper topology-preserving
    thinning.
    
    Parameters
    ----------
    binary : np.ndarray
        Binary mask (uint8, 0 or 255).
    
    Returns
    -------
    np.ndarray
        Skeleton mask (uint8, 0 or 255), 1-pixel-wide lines.
    """
    bool_skel = skeletonize(binary > 0)
    return bool_skel.astype(np.uint8) * 255


def find_endpoints(skeleton: np.ndarray) -> list:
    """Find skeleton endpoints (pixels with exactly 1 neighbor).
    
    An endpoint is a foreground pixel with exactly one 8-connected
    foreground neighbor.
    
    Parameters
    ----------
    skeleton : np.ndarray
        Binary skeleton (uint8, 0 or 255).
    
    Returns
    -------
    list of tuple
        List of (row, col) endpoint coordinates.
    """
    skel_bool = (skeleton > 0).astype(np.uint8)
    # Count 8-connected neighbors using convolution
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_bool, -1, kernel)
    
    # Endpoints: foreground pixels with exactly 1 neighbor
    endpoints = np.argwhere((skel_bool == 1) & (neighbor_count == 1))
    return [tuple(pt) for pt in endpoints]


def link_endpoints(skeleton: np.ndarray, max_gap: int = 10) -> np.ndarray:
    """Link nearby skeleton endpoints across small gaps.
    
    For each endpoint, search for the nearest other endpoint within
    max_gap distance and draw a thin line between them.
    
    Parameters
    ----------
    skeleton : np.ndarray
        Binary skeleton (uint8, 0 or 255).
    max_gap : int
        Maximum distance (pixels) to bridge between endpoints.
    
    Returns
    -------
    np.ndarray
        Updated skeleton with linked endpoints.
    """
    result = skeleton.copy()
    endpoints = find_endpoints(result)
    
    if len(endpoints) < 2:
        return result
    
    from scipy.spatial import cKDTree
    
    eps = np.array(endpoints, dtype=np.float64)
    tree = cKDTree(eps)
    pairs = tree.query_pairs(r=max_gap, output_type="ndarray")
    
    if len(pairs) == 0:
        return result
        
    # Sort pairs by distance so closest endpoints are linked first
    diffs = eps[pairs[:, 0]] - eps[pairs[:, 1]]
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    sorted_indices = np.argsort(dists)
    
    linked = set()
    for idx in sorted_indices:
        i, j = pairs[idx]
        if i in linked or j in linked:
            continue
        r1, c1 = endpoints[i]
        r2, c2 = endpoints[j]
        cv2.line(result, (int(c1), int(r1)), (int(c2), int(r2)), 255, 1)
        linked.add(i)
        linked.add(j)
    
    return result


def prune_spurs(skeleton: np.ndarray, max_spur_length: int = 8) -> np.ndarray:
    """Remove dangling open branches (spurs) from the skeleton.
    
    Iteratively remove endpoints and their short branches that do not
    enclose any region.
    
    Parameters
    ----------
    skeleton : np.ndarray
        Binary skeleton (uint8, 0 or 255).
    max_spur_length : int
        Maximum length of spurs to remove.
    
    Returns
    -------
    np.ndarray
        Pruned skeleton.
    """
    result = skeleton.copy()
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    nbr_offsets = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                   (1, 0), (1, -1), (0, -1), (-1, -1)]

    for _ in range(max_spur_length):
        skel_bool = (result > 0).astype(np.uint8)
        neighbor_count = cv2.filter2D(skel_bool, -1, kernel)

        # Standard endpoints (exactly 1 neighbor)
        std_endpoints = np.argwhere((skel_bool == 1) & (neighbor_count == 1))

        if len(std_endpoints) > 0:
            for r, c in std_endpoints:
                result[r, c] = 0
        else:
            # Check for stub spur pixels adjacent to junctions (crossing number == 1)
            candidates = np.argwhere((skel_bool == 1) & (neighbor_count >= 2) & (neighbor_count <= 3))
            stub_pts = []
            h, w = result.shape
            for r, c in candidates:
                if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                    continue
                seq = [skel_bool[r + dr, c + dc] for dr, dc in nbr_offsets]
                seq.append(seq[0])
                transitions = sum((seq[i] == 0 and seq[i + 1] == 1) for i in range(8))
                if transitions == 1:
                    stub_pts.append((r, c))

            if stub_pts:
                for r, c in stub_pts:
                    result[r, c] = 0
            else:
                break

    return result


def label_regions(skeleton: np.ndarray) -> tuple:
    """Label closed regions enclosed by the skeleton boundaries.
    
    The skeleton pixels form the boundaries; the spaces between them
    are the closed regions, labeled using connected-component analysis.
    
    Parameters
    ----------
    skeleton : np.ndarray
        Binary skeleton (uint8, 0 or 255).
    
    Returns
    -------
    tuple of (np.ndarray, int)
        label_map : 2D int32 array where each pixel has its region label
                    (0 = boundary, 1..N = region labels)
        num_regions : number of distinct regions found
    """
    # Invert: regions are where skeleton is 0
    background = (skeleton == 0).astype(np.uint8)
    label_map, num_regions = ndimage.label(background)
    return label_map, num_regions


def detect_contours(
    gray: np.ndarray,
    max_gap: int = 10,
    max_spur_length: int = 8,
    threshold_method: str = "otsu",
    threshold_kwargs: dict | None = None,
) -> dict:
    """Run the full contour detection pipeline.
    
    Parameters
    ----------
    gray : np.ndarray
        2D uint8 grayscale image (denoised).
    max_gap : int
        Maximum endpoint gap to bridge.
    max_spur_length : int
        Maximum spur length to prune.
    threshold_method : str, default="otsu"
        Thresholding method ("otsu", "yen", "triangle", "li", "adaptive", "gmm").
    threshold_kwargs : dict or None, optional
        Additional keyword arguments for thresholding.
    
    Returns
    -------
    dict
        Keys:
        - 'binary': Binary thresholded image
        - 'skeleton_raw': Initial skeleton before linking/pruning
        - 'skeleton': Final skeleton after linking and pruning
        - 'label_map': Region label array
        - 'num_regions': Number of closed regions
    """
    binary = global_threshold(
        gray, method=threshold_method, threshold_kwargs=threshold_kwargs
    )
    skeleton_raw = skeletonize_binary(binary)
    skeleton_linked = link_endpoints(skeleton_raw, max_gap=max_gap)
    skeleton_pruned = prune_spurs(skeleton_linked, max_spur_length=max_spur_length)
    label_map, num_regions = label_regions(skeleton_pruned)
    
    return {
        "binary": binary,
        "skeleton_raw": skeleton_raw,
        "skeleton": skeleton_pruned,
        "label_map": label_map,
        "num_regions": num_regions,
    }
