"""
CAMMP-SFSE: Evaluation and Reference Mask Generation Module

Provides reference mask generators (synthetic ground truth & multi-scale
consensus voting for real SEM micrographs) and segmentation accuracy metrics
(IoU, Dice, Precision, Recall, and tolerance-based Boundary F1).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology
from skimage.filters import meijering, sato, sobel, threshold_otsu
from skimage.morphology import skeletonize


def generate_synthetic_ground_truth(
    height: int = 512,
    width: int = 512,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct exact deterministic ground truth tear-ridge mask and skeleton.

    Parameters
    ----------
    height, width : int
        Dimensions of the synthetic image.
    seed : int
        Seed matching the synthetic SEM generator in main.py.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        ref_binary : 2D uint8 ground-truth ridge mask (0 or 255)
        ref_skel : 2D uint8 1-pixel wide skeleton (0 or 255)
    """
    rng = np.random.RandomState(seed)

    # In main.py:
    # 3 iterations of Gaussian blurred noise are generated first
    for _ in range(3):
        _ = rng.normal(0, 8, (height, width)).astype(np.float64)

    # Next: create bright tear-ridge network
    ridge_mask = np.zeros((height, width), dtype=np.uint8)

    n_horiz = rng.randint(6, 10)
    n_vert = rng.randint(6, 10)

    for i in range(n_horiz):
        y = int(height * (i + 0.5) / n_horiz)
        pts = []
        for x in range(0, width, 20):
            jy = y + rng.randint(-15, 16)
            pts.append([x, np.clip(jy, 5, height - 5)])
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(ridge_mask, [pts], False, 255, thickness=2)

    for i in range(n_vert):
        x = int(width * (i + 0.5) / n_vert)
        pts = []
        for y in range(0, height, 20):
            jx = x + rng.randint(-15, 16)
            pts.append([np.clip(jx, 5, width - 5), y])
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(ridge_mask, [pts], False, 255, thickness=2)

    # Diagonal ridges
    for _ in range(4):
        x1, y1 = rng.randint(0, width), rng.randint(0, height)
        x2, y2 = rng.randint(0, width), rng.randint(0, height)
        cv2.line(ridge_mask, (x1, y1), (x2, y2), 255, 2)

    ref_binary = (ridge_mask > 0).astype(np.uint8) * 255
    ref_skel = skeletonize(ref_binary > 0).astype(np.uint8) * 255
    return ref_binary, ref_skel


def generate_real_consensus_reference(
    img: np.ndarray,
    consensus_votes: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate high-confidence consensus reference mask for real SEM images.

    Combines CLAHE contrast normalization, multi-scale Meijering, Sato,
    Sobel gradient, and morphological top-hat filters with a pixel-wise
    consensus voting threshold (>= 4 of 6 detectors agree) + morphological thinning.

    Parameters
    ----------
    img : np.ndarray
        Grayscale SEM image.
    consensus_votes : int, default=4
        Minimum number of detectors that must agree for a pixel to be classified
        as a reference ridge.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        ref_binary : 2D uint8 consensus ridge mask (0 or 255)
        ref_skel : 2D uint8 1-pixel wide skeleton (0 or 255)
    """
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    c_img = clahe.apply(img)

    # 1. Meijering fine scale (sigma=1.0)
    m1 = meijering(c_img, sigmas=[1.0], black_ridges=False)
    t1 = float(threshold_otsu(m1)) if np.std(m1) > 1e-4 else float(np.mean(m1))
    d1 = m1 > t1

    # 2. Meijering medium scale (sigma=2.0)
    m2 = meijering(c_img, sigmas=[2.0], black_ridges=False)
    t2 = float(threshold_otsu(m2)) if np.std(m2) > 1e-4 else float(np.mean(m2))
    d2 = m2 > t2

    # 3. Sato tubeness / ridge filter (sigmas=[1.0, 2.0])
    s = sato(c_img, sigmas=[1.0, 2.0], black_ridges=False)
    ts = float(threshold_otsu(s)) if np.std(s) > 1e-4 else float(np.mean(s))
    d3 = s > ts

    # 4. Sobel edge magnitude
    sob = sobel(c_img)
    t_sob = float(threshold_otsu(sob)) if np.std(sob) > 1e-4 else float(np.mean(sob))
    d4 = sob > t_sob

    # 5. Morphological white top-hat (5x5 kernel)
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    top5 = cv2.morphologyEx(c_img, cv2.MORPH_TOPHAT, k5)
    t_top5 = float(threshold_otsu(top5)) if np.std(top5) > 1e-4 else float(np.mean(top5))
    d5 = top5 > t_top5

    # 6. Morphological white top-hat (9x9 kernel)
    k9 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    top9 = cv2.morphologyEx(c_img, cv2.MORPH_TOPHAT, k9)
    t_top9 = float(threshold_otsu(top9)) if np.std(top9) > 1e-4 else float(np.mean(top9))
    d6 = top9 > t_top9

    # Pixel-wise consensus voting
    vote_count = (
        d1.astype(np.uint8)
        + d2.astype(np.uint8)
        + d3.astype(np.uint8)
        + d4.astype(np.uint8)
        + d5.astype(np.uint8)
        + d6.astype(np.uint8)
    )
    consensus = vote_count >= consensus_votes

    # Morphological refinement
    try:
        cleaned = morphology.remove_small_objects(consensus, max_size=9)
    except TypeError:
        cleaned = morphology.remove_small_objects(consensus, min_size=10)
    cleaned = morphology.closing(cleaned, footprint=morphology.disk(1))
    ref_binary = cleaned.astype(np.uint8) * 255
    ref_skel = skeletonize(ref_binary > 0).astype(np.uint8) * 255

    return ref_binary, ref_skel


def generate_reference_mask(
    img: np.ndarray,
    sample_name: str,
    output_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate and optionally save reference binary mask and skeleton.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image array.
    sample_name : str
        Sample name identifier (e.g. "synthetic_sem", "sem_real_1").
    output_dir : str or None, optional
        Directory where `<sample>_ref_binary.png` and `<sample>_ref_skeleton.png`
        will be saved.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        ref_binary : 2D uint8 mask (0 or 255)
        ref_skel : 2D uint8 skeleton (0 or 255)
    """
    sname = sample_name.lower().strip()
    if "synthetic" in sname:
        h, w = img.shape[:2]
        ref_binary, ref_skel = generate_synthetic_ground_truth(height=h, width=w)
    else:
        ref_binary, ref_skel = generate_real_consensus_reference(img, consensus_votes=4)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        bin_path = os.path.join(output_dir, f"{sample_name}_ref_binary.png")
        skel_path = os.path.join(output_dir, f"{sample_name}_ref_skeleton.png")
        cv2.imwrite(bin_path, ref_binary)
        cv2.imwrite(skel_path, ref_skel)

    return ref_binary, ref_skel


def compute_segmentation_metrics(
    pred_binary: np.ndarray,
    ref_binary: np.ndarray,
    pred_skel: np.ndarray,
    ref_skel: np.ndarray,
    tolerance_px: int = 2,
) -> Dict[str, float]:
    """Compute segmentation accuracy metrics between prediction and reference.

    Calculates:
    - Pixel-level IoU (Jaccard Index)
    - Pixel-level Dice Coefficient (F1-score)
    - Pixel-level Precision
    - Pixel-level Recall
    - Boundary F1 with morphological dilation tolerance band

    Parameters
    ----------
    pred_binary : np.ndarray
        Predicted binary foreground mask.
    ref_binary : np.ndarray
        Reference ground-truth binary foreground mask.
    pred_skel : np.ndarray
        Predicted 1-pixel skeleton mask.
    ref_skel : np.ndarray
        Reference 1-pixel skeleton mask.
    tolerance_px : int, default=2
        Tolerance radius in pixels for contour boundary alignment.

    Returns
    -------
    dict
        Dictionary with keys: "iou", "dice", "precision", "recall", "boundary_f1".
    """
    pred_b = pred_binary > 0
    ref_b = ref_binary > 0

    tp = int(np.logical_and(pred_b, ref_b).sum())
    fp = int(np.logical_and(pred_b, ~ref_b).sum())
    fn = int(np.logical_and(~pred_b, ref_b).sum())

    n_pred_b = int(pred_b.sum())
    n_ref_b = int(ref_b.sum())

    if n_pred_b == 0 and n_ref_b == 0:
        iou = 1.0
        dice = 1.0
        precision = 1.0
        recall = 1.0
    else:
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        iou = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0
        dice = float((2.0 * tp) / (2.0 * tp + fp + fn)) if (2.0 * tp + fp + fn) > 0 else 0.0

    # Boundary F1 using morphological tolerance band
    pred_s = pred_skel > 0
    ref_s = ref_skel > 0

    n_pred_s = int(pred_s.sum())
    n_ref_s = int(ref_s.sum())

    if n_pred_s == 0 and n_ref_s == 0:
        boundary_f1 = 1.0
    elif n_pred_s == 0 or n_ref_s == 0:
        boundary_f1 = 0.0
    else:
        footprint = morphology.disk(tolerance_px)
        ref_dilated = ndimage.binary_dilation(ref_s, structure=footprint)
        pred_dilated = ndimage.binary_dilation(pred_s, structure=footprint)

        matched_pred = int(np.logical_and(pred_s, ref_dilated).sum())
        prec_b = float(matched_pred / n_pred_s)

        matched_ref = int(np.logical_and(ref_s, pred_dilated).sum())
        rec_b = float(matched_ref / n_ref_s)

        if (prec_b + rec_b) > 0:
            boundary_f1 = float(2.0 * prec_b * rec_b / (prec_b + rec_b))
        else:
            boundary_f1 = 0.0

    metrics = {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "boundary_f1": float(boundary_f1),
    }

    for k, v in metrics.items():
        assert not (np.isnan(v) or np.isinf(v)), f"Metric {k} contains NaN or Inf: {v}"

    return metrics
