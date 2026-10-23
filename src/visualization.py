"""
CAMMP-SFSE: Visualization Module

Generate the 6-panel publication-quality diagnostic figure (_pipeline.png)
and color-coded region masks.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2


def create_pipeline_figure(gray: np.ndarray,
                           skeleton_raw: np.ndarray,
                           skeleton_refined: np.ndarray,
                           void_seeds: np.ndarray,
                           void_extended: np.ndarray,
                           classification_mask: np.ndarray,
                           df_regions,
                           output_path: str,
                           dpi: int = 150) -> str:
    """Generate the 6-panel publication-quality diagnostic figure.

    Panels:
        1. Original SEM image
        2. Initial Skeleton
        3. Refined Skeleton
        4. Void Tips & Extension
        5. Classified Regions (Void=Red, Mixed=Yellow, Surface=Blue)
        6. ECD Size Distribution Histogram

    Parameters
    ----------
    gray : np.ndarray
        Original 2D uint8 grayscale image.
    skeleton_raw : np.ndarray
        Initial skeleton from contour detection (uint8).
    skeleton_refined : np.ndarray
        Refined skeleton after contour refinement (uint8).
    void_seeds : np.ndarray
        Void-tip seed mask (uint8).
    void_extended : np.ndarray
        Extended void mask (uint8).
    classification_mask : np.ndarray
        3-channel RGB classification mask.
    df_regions : pd.DataFrame
        Per-region metrics (must have 'ECD_px' column).
    output_path : str
        Path to save the figure.
    dpi : int
        Figure resolution.

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("CAMMP-SFSE Pipeline Diagnostic", fontsize=16, fontweight="bold")

    # Panel 1: Original SEM
    ax = axes[0, 0]
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax.set_title("(a) Original SEM Image")
    ax.axis("off")

    # Panel 2: Initial Skeleton
    ax = axes[0, 1]
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255, alpha=0.5)
    skel_overlay = np.zeros((*gray.shape, 4), dtype=np.float32)
    skel_overlay[skeleton_raw > 0] = [0, 1, 0, 1]  # green
    ax.imshow(skel_overlay)
    ax.set_title("(b) Initial Skeleton")
    ax.axis("off")

    # Panel 3: Refined Skeleton
    ax = axes[0, 2]
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255, alpha=0.5)
    skel_overlay2 = np.zeros((*gray.shape, 4), dtype=np.float32)
    skel_overlay2[skeleton_refined > 0] = [0, 1, 0, 1]  # green
    ax.imshow(skel_overlay2)
    ax.set_title("(c) Refined Skeleton")
    ax.axis("off")

    # Panel 4: Void Tips & Extension
    ax = axes[1, 0]
    void_vis = np.stack([gray, gray, gray], axis=-1).copy()
    # Seeds in cyan
    void_vis[void_seeds > 0] = [0, 255, 255]
    # Extended (non-seed) in magenta
    extension_only = (void_extended > 0) & (void_seeds == 0)
    void_vis[extension_only] = [255, 0, 255]
    ax.imshow(void_vis)
    ax.set_title("(d) Void Tips (cyan) & Extension (magenta)")
    ax.axis("off")

    # Panel 5: Classified Regions
    ax = axes[1, 1]
    ax.imshow(classification_mask)
    ax.set_title("(e) Classified Regions")
    ax.axis("off")
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="red", label="Void"),
        Patch(facecolor="yellow", label="Mixed"),
        Patch(facecolor="blue", label="Surface"),
        Patch(facecolor="white", edgecolor="black", label="Boundary"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    # Panel 6: ECD Size Distribution
    ax = axes[1, 2]
    if df_regions is not None and len(df_regions) > 0 and "ECD_px" in df_regions.columns:
        ecd_values = df_regions["ECD_px"].values
        ecd_values = ecd_values[ecd_values > 0]
        if len(ecd_values) > 0:
            ax.hist(ecd_values, bins=min(30, max(5, len(ecd_values) // 3)),
                    color="steelblue", edgecolor="black", alpha=0.8)
            ax.set_xlabel("ECD (px)")
            ax.set_ylabel("Count")
    ax.set_title("(f) ECD Size Distribution")

    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return os.path.abspath(output_path)


def save_classification_mask(classification_mask: np.ndarray,
                             output_path: str) -> str:
    """Save the color-coded classification mask as an image.

    Parameters
    ----------
    classification_mask : np.ndarray
        3-channel RGB classification mask.
    output_path : str
        Output file path.

    Returns
    -------
    str
        Absolute path to the saved image.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor(classification_mask, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr)
    return os.path.abspath(output_path)


def save_skeleton_overlay(gray: np.ndarray, skeleton: np.ndarray,
                          output_path: str) -> str:
    """Save a grayscale image with skeleton overlay in green.

    Parameters
    ----------
    gray : np.ndarray
        2D uint8 grayscale image.
    skeleton : np.ndarray
        Binary skeleton (uint8).
    output_path : str
        Output file path.

    Returns
    -------
    str
        Absolute path to the saved image.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    vis[skeleton > 0] = [0, 255, 0]
    cv2.imwrite(output_path, vis)
    return os.path.abspath(output_path)


def save_binary_mask(mask: np.ndarray, output_path: str) -> str:
    """Save a binary mask as a monochrome image (0 or 255).

    Parameters
    ----------
    mask : np.ndarray
        2D binary or boolean mask.
    output_path : str
        Output file path.

    Returns
    -------
    str
        Absolute path to the saved image.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out = (mask > 0).astype(np.uint8) * 255
    cv2.imwrite(output_path, out)
    return os.path.abspath(output_path)

