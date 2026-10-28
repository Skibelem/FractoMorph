"""
CAMMP-SFSE: Robustness, Sensitivity, and Statistical Analysis Module

Implements image perturbation transforms, hyperparameter sweeps,
paired statistical significance tests vs Otsu baseline, 95% confidence
intervals, and publication-quality diagnostic plotting.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from skimage.morphology import skeletonize

from src.evaluation import compute_segmentation_metrics, generate_reference_mask
from src.thresholding import SUPPORTED_METHODS, apply_threshold


# ---------------------------------------------------------------------------
# Image Perturbations
# ---------------------------------------------------------------------------

def apply_brightness_shift(img: np.ndarray, delta: int) -> np.ndarray:
    """Shift image brightness by additive delta in [-30, +30]."""
    return np.clip(img.astype(np.int16) + int(delta), 0, 255).astype(np.uint8)


def apply_contrast_scale(img: np.ndarray, alpha: float) -> np.ndarray:
    """Scale image contrast by multiplicative alpha in [0.6, 1.4]."""
    return np.clip(float(alpha) * img.astype(np.float64), 0, 255).astype(np.uint8)


def apply_gaussian_noise(img: np.ndarray, sigma: float, seed: int = 42) -> np.ndarray:
    """Add zero-mean Gaussian noise with standard deviation sigma."""
    if sigma <= 0:
        return img.copy()
    rng = np.random.RandomState(seed)
    noise = rng.normal(0, float(sigma), img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def apply_resolution_scale(img: np.ndarray, scale: float) -> np.ndarray:
    """Scale image spatial resolution by scale factor in (0, 1]."""
    if scale >= 0.999:
        return img.copy()
    h, w = img.shape[:2]
    new_w = max(int(round(w * scale)), 16)
    new_h = max(int(round(h * scale)), 16)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Task 6: Perturbation and Sensitivity Analysis
# ---------------------------------------------------------------------------

def evaluate_threshold_accuracy_fast(
    img_perturbed: np.ndarray,
    ref_binary: np.ndarray,
    ref_skel: np.ndarray,
    method: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Apply thresholding to a perturbed image and evaluate against original reference."""
    h_orig, w_orig = ref_binary.shape[:2]
    h_cur, w_cur = img_perturbed.shape[:2]

    # Subsample GMM pixel fitting if max_samples provided
    res = apply_threshold(img_perturbed, method=method, **kwargs)
    pred_bin = res["binary"]
    t_val = float(res["threshold_summary"])
    runtime_ms = float(res["runtime_ms"])

    # If image was resolution scaled, resize binary back to original dimensions
    if (h_cur != h_orig) or (w_cur != w_orig):
        pred_bin = cv2.resize(pred_bin, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    pred_skel = skeletonize(pred_bin > 0).astype(np.uint8) * 255
    metrics = compute_segmentation_metrics(pred_bin, ref_binary, pred_skel, ref_skel, tolerance_px=2)

    return {
        "threshold_val": t_val,
        "runtime_ms": runtime_ms,
        "iou": metrics["iou"],
        "dice": metrics["dice"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "boundary_f1": metrics["boundary_f1"],
    }


def run_perturbation_sweeps(
    samples: List[Tuple[str, np.ndarray]],
    methods: Optional[List[str]] = None,
    gmm_samples: int = 25000,
) -> pd.DataFrame:
    """Run perturbation robustness sweeps across brightness, contrast, noise, and scale."""
    if methods is None:
        methods = list(SUPPORTED_METHODS)

    perturbation_specs = [
        ("brightness", [-30, -15, 15, 30]),
        ("contrast", [0.6, 0.8, 1.2, 1.4]),
        ("noise", [5.0, 10.0, 20.0]),
        ("resolution", [0.5, 0.75, 1.0]),
    ]

    records = []

    for sample_name, img in samples:
        ref_binary, ref_skel = generate_reference_mask(img, sample_name=sample_name)

        for p_type, p_values in perturbation_specs:
            for p_val in p_values:
                # Apply specific perturbation
                if p_type == "brightness":
                    p_img = apply_brightness_shift(img, delta=int(p_val))
                elif p_type == "contrast":
                    p_img = apply_contrast_scale(img, alpha=float(p_val))
                elif p_type == "noise":
                    p_img = apply_gaussian_noise(img, sigma=float(p_val))
                elif p_type == "resolution":
                    p_img = apply_resolution_scale(img, scale=float(p_val))
                else:
                    p_img = img.copy()

                for method in methods:
                    kwargs: Dict[str, Any] = {}
                    if method == "gmm":
                        kwargs = {"max_samples": gmm_samples, "n_components": 2, "random_state": 42}
                    elif method == "adaptive":
                        kwargs = {"block_size": 35, "offset": -5.0}

                    eval_res = evaluate_threshold_accuracy_fast(
                        p_img, ref_binary, ref_skel, method=method, **kwargs
                    )

                    records.append({
                        "Sample": sample_name,
                        "Method": method,
                        "Perturbation_Type": p_type,
                        "Parameter_Value": p_val,
                        "Threshold_T": eval_res["threshold_val"],
                        "IoU": eval_res["iou"],
                        "Dice": eval_res["dice"],
                        "Precision": eval_res["precision"],
                        "Recall": eval_res["recall"],
                        "Boundary_F1": eval_res["boundary_f1"],
                        "Runtime_ms": eval_res["runtime_ms"],
                    })

    return pd.DataFrame(records)


def run_hyperparameter_sweeps(
    samples: List[Tuple[str, np.ndarray]],
    gmm_samples: int = 25000,
) -> pd.DataFrame:
    """Run hyperparameter sensitivity sweeps for adaptive and GMM methods."""
    records = []

    adaptive_block_sizes = [15, 31, 51, 75]
    adaptive_offsets = [-10.0, -5.0, 0.0, 5.0]
    gmm_components = [2, 3, 4]

    for sample_name, img in samples:
        ref_binary, ref_skel = generate_reference_mask(img, sample_name=sample_name)

        # 1. Sweep adaptive block_size and offset
        for bs in adaptive_block_sizes:
            for off in adaptive_offsets:
                eval_res = evaluate_threshold_accuracy_fast(
                    img, ref_binary, ref_skel, method="adaptive", block_size=bs, offset=off
                )
                records.append({
                    "Sample": sample_name,
                    "Method": "adaptive",
                    "Parameter_Name": f"bs_{bs}_off_{off}",
                    "Block_Size": bs,
                    "Offset": off,
                    "N_Components": np.nan,
                    "Threshold_T": eval_res["threshold_val"],
                    "IoU": eval_res["iou"],
                    "Dice": eval_res["dice"],
                    "Precision": eval_res["precision"],
                    "Recall": eval_res["recall"],
                    "Boundary_F1": eval_res["boundary_f1"],
                    "Runtime_ms": eval_res["runtime_ms"],
                })

        # 2. Sweep GMM components
        for nc in gmm_components:
            eval_res = evaluate_threshold_accuracy_fast(
                img, ref_binary, ref_skel, method="gmm", n_components=nc, max_samples=gmm_samples, random_state=42
            )
            records.append({
                "Sample": sample_name,
                "Method": "gmm",
                "Parameter_Name": f"n_components_{nc}",
                "Block_Size": np.nan,
                "Offset": np.nan,
                "N_Components": nc,
                "Threshold_T": eval_res["threshold_val"],
                "IoU": eval_res["iou"],
                "Dice": eval_res["dice"],
                "Precision": eval_res["precision"],
                "Recall": eval_res["recall"],
                "Boundary_F1": eval_res["boundary_f1"],
                "Runtime_ms": eval_res["runtime_ms"],
            })

    return pd.DataFrame(records)


def plot_robustness_and_sensitivity(
    df_pert: pd.DataFrame,
    df_hyper: pd.DataFrame,
    output_path: str,
) -> str:
    """Generate multi-panel sensitivity and robustness figures."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    methods = ["otsu", "yen", "triangle", "li", "adaptive", "gmm"]
    colors = {
        "otsu": "#1f77b4",
        "yen": "#ff7f0e",
        "triangle": "#2ca02c",
        "li": "#d62728",
        "adaptive": "#9467bd",
        "gmm": "#8c564b",
    }

    # Row 1: Perturbations (averaged across samples)
    p_configs = [
        ("brightness", "Brightness Shift", "Brightness Delta", axes[0, 0]),
        ("contrast", "Contrast Scaling", "Contrast Factor (\u03b1)", axes[0, 1]),
        ("noise", "Gaussian Noise", "Noise Sigma (\u03c3)", axes[0, 2]),
        ("resolution", "Resolution Scaling", "Scale Factor", axes[0, 3]),
    ]

    for p_type, title, xlabel, ax in p_configs:
        sub = df_pert[df_pert["Perturbation_Type"] == p_type]
        for m in methods:
            m_sub = sub[sub["Method"] == m]
            if len(m_sub) == 0:
                continue
            grp = m_sub.groupby("Parameter_Value")["IoU"].mean().sort_index()
            ax.plot(grp.index, grp.values, marker="o", label=m, color=colors.get(m, "black"), linewidth=1.8)
        ax.set_title(f"(a) {title} vs IoU", fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Mean IoU", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        if ax == axes[0, 0]:
            ax.legend(fontsize=8, loc="upper right")

    # Row 2: Hyperparameter Sweeps
    # Panel 5: Adaptive Block Size vs IoU across Offsets
    ax = axes[1, 0]
    sub_adap = df_hyper[df_hyper["Method"] == "adaptive"]
    for off in [-10.0, -5.0, 0.0, 5.0]:
        off_sub = sub_adap[sub_adap["Offset"] == off]
        if len(off_sub) > 0:
            grp = off_sub.groupby("Block_Size")["IoU"].mean().sort_index()
            ax.plot(grp.index, grp.values, marker="s", label=f"Offset = {off:+.1f}", linewidth=1.8)
    ax.set_title("(b) Adaptive: Block Size vs IoU", fontsize=11, fontweight="bold")
    ax.set_xlabel("Block Size (px)", fontsize=10)
    ax.set_ylabel("Mean IoU", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)

    # Panel 6: Adaptive Offset vs Boundary F1
    ax = axes[1, 1]
    for bs in [15, 31, 51, 75]:
        bs_sub = sub_adap[sub_adap["Block_Size"] == bs]
        if len(bs_sub) > 0:
            grp = bs_sub.groupby("Offset")["Boundary_F1"].mean().sort_index()
            ax.plot(grp.index, grp.values, marker="^", label=f"Block = {bs}", linewidth=1.8)
    ax.set_title("(c) Adaptive: Offset vs Boundary F1", fontsize=11, fontweight="bold")
    ax.set_xlabel("Offset Constant", fontsize=10)
    ax.set_ylabel("Mean Boundary F1", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)

    # Panel 7: Adaptive Offset vs IoU
    ax = axes[1, 2]
    for bs in [15, 31, 51, 75]:
        bs_sub = sub_adap[sub_adap["Block_Size"] == bs]
        if len(bs_sub) > 0:
            grp = bs_sub.groupby("Offset")["IoU"].mean().sort_index()
            ax.plot(grp.index, grp.values, marker="d", label=f"Block = {bs}", linewidth=1.8)
    ax.set_title("(d) Adaptive: Offset vs IoU", fontsize=11, fontweight="bold")
    ax.set_xlabel("Offset Constant", fontsize=10)
    ax.set_ylabel("Mean IoU", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)

    # Panel 8: GMM n_components vs IoU and Boundary F1
    ax = axes[1, 3]
    sub_gmm = df_hyper[df_hyper["Method"] == "gmm"]
    if len(sub_gmm) > 0:
        grp_iou = sub_gmm.groupby("N_Components")["IoU"].mean().sort_index()
        grp_f1 = sub_gmm.groupby("N_Components")["Boundary_F1"].mean().sort_index()
        ax.plot(grp_iou.index, grp_iou.values, marker="o", color="#8c564b", linewidth=2.0, label="IoU")
        ax.plot(grp_f1.index, grp_f1.values, marker="s", color="#17becf", linewidth=2.0, label="Boundary F1")
    ax.set_title("(e) GMM Components vs Accuracy", fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of Components (k)", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_xticks([2, 3, 4])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Task 7: Statistical Analysis & Confidence Intervals
# ---------------------------------------------------------------------------

def compute_detailed_statistical_analysis(df_acc: pd.DataFrame) -> pd.DataFrame:
    """Compute 95% Confidence Intervals and paired statistical tests against Otsu baseline.

    Parameters
    ----------
    df_acc : pd.DataFrame
        Table from per_image_accuracy_metrics.csv containing per-image metrics.

    Returns
    -------
    pd.DataFrame
        Summary statistics table with means, SDs, 95% CIs, and paired p-values vs Otsu.
    """
    methods = ["otsu", "yen", "triangle", "li", "adaptive", "gmm"]
    metrics_to_test = ["IoU", "Dice", "Precision", "Recall", "Boundary_F1", "Runtime_ms"]

    otsu_sub = df_acc[df_acc["Method"] == "otsu"].sort_values("Sample")
    otsu_iou = otsu_sub["IoU"].values
    otsu_dice = otsu_sub["Dice"].values
    otsu_f1 = otsu_sub["Boundary_F1"].values

    n_samples = len(otsu_sub)
    df_freedom = max(n_samples - 1, 1)
    t_crit = stats.t.ppf(0.975, df=df_freedom)

    stat_rows = []

    for m in methods:
        sub = df_acc[df_acc["Method"] == m].sort_values("Sample")
        m_iou = sub["IoU"].values
        m_dice = sub["Dice"].values
        m_f1 = sub["Boundary_F1"].values

        # Paired tests vs Otsu
        if m == "otsu":
            p_val_t_iou, p_val_w_iou = 1.0, 1.0
            p_val_t_dice, p_val_w_dice = 1.0, 1.0
            p_val_t_f1, p_val_w_f1 = 1.0, 1.0
        else:
            # Paired t-test
            p_val_t_iou = float(stats.ttest_rel(m_iou, otsu_iou)[1])
            p_val_t_dice = float(stats.ttest_rel(m_dice, otsu_dice)[1])
            p_val_t_f1 = float(stats.ttest_rel(m_f1, otsu_f1)[1])

            # Wilcoxon signed-rank test
            try:
                p_val_w_iou = float(stats.wilcoxon(m_iou, otsu_iou)[1])
            except Exception:
                p_val_w_iou = 1.0

            try:
                p_val_w_dice = float(stats.wilcoxon(m_dice, otsu_dice)[1])
            except Exception:
                p_val_w_dice = 1.0

            try:
                p_val_w_f1 = float(stats.wilcoxon(m_f1, otsu_f1)[1])
            except Exception:
                p_val_w_f1 = 1.0

        row: Dict[str, Any] = {
            "Method": m,
            "N_Samples": n_samples,
        }

        # Calculate mean, SD, and 95% CI for each metric
        for col in metrics_to_test:
            vals = sub[col].values
            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            sem_val = sd_val / np.sqrt(len(vals)) if len(vals) > 0 else 0.0
            ci_margin = float(t_crit * sem_val)

            row[f"{col}_mean"] = mean_val
            row[f"{col}_sd"] = sd_val
            row[f"{col}_ci95_low"] = mean_val - ci_margin
            row[f"{col}_ci95_high"] = mean_val + ci_margin

        # Append paired test results
        row["IoU_paired_ttest_p"] = p_val_t_iou
        row["IoU_wilcoxon_p"] = p_val_w_iou
        row["Dice_paired_ttest_p"] = p_val_t_dice
        row["Dice_wilcoxon_p"] = p_val_w_dice
        row["Boundary_F1_paired_ttest_p"] = p_val_t_f1
        row["Boundary_F1_wilcoxon_p"] = p_val_w_f1

        stat_rows.append(row)

    return pd.DataFrame(stat_rows)


def plot_benchmark_boxplots(df_acc: pd.DataFrame, output_path: str) -> str:
    """Generate publication-quality 2x2 box plot figure with sample overlays."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    methods = ["otsu", "yen", "triangle", "li", "adaptive", "gmm"]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    panels = [
        ("IoU", "Panel (a): Intersection-over-Union (IoU)", "IoU Score", axes[0, 0], False),
        ("Dice", "Panel (b): Dice Coefficient (F1)", "Dice Score", axes[0, 1], False),
        ("Boundary_F1", "Panel (c): Boundary F1 Score (\u00b12px tolerance)", "Boundary F1", axes[1, 0], False),
        ("Runtime_ms", "Panel (d): Processing Time (ms, log-scale)", "Execution Time [ms] (log\u2081\u2080)", axes[1, 1], True),
    ]

    for col, title, ylabel, ax, log_scale in panels:
        data_by_method = [df_acc[df_acc["Method"] == m][col].values for m in methods]

        # Draw boxplots
        bp = ax.boxplot(
            data_by_method,
            patch_artist=True,
            tick_labels=[m.upper() for m in methods],
            widths=0.55,
            medianprops=dict(color="black", linewidth=2.0),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
        )

        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        # Overlay individual sample jitter points
        rng = np.random.RandomState(42)
        for idx, vals in enumerate(data_by_method):
            x_jitter = (idx + 1) + rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(x_jitter, vals, color=palette[idx], edgecolor="black", s=45, zorder=4, alpha=0.9)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)

        if log_scale:
            ax.set_yscale("log")
        else:
            ax.set_ylim(-0.02, 1.05)

    plt.tight_layout(pad=2.5)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(output_path)
