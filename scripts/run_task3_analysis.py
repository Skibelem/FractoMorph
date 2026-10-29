#!/usr/bin/env python3
"""
CAMMP-SFSE: Task 3 Six-Model Comparative Benchmark Runner

Processes three blind-validation SEM micrographs:
- sample_A_0  (Image A, 0 deg orientation)
- sample_B_60 (Image B, 60 deg orientation)
- sample_C_90 (Image C, 90 deg orientation)
from `data/task3_cropped/` through the full FractoMorph pipeline
(min_area=30, void_ratio=0.50) across all 6 automated thresholding models:
["otsu", "yen", "triangle", "li", "adaptive", "gmm"]
holding all upstream bilateral filtering and downstream skeletonization,
cKDTree gap linking, Stage 2 local refinement, void detection, and
3-class classification strictly constant.

Outputs generated under `outputs/task3_analysis/`:
- `reference_masks/`: Multi-scale consensus reference masks
- `<method>/`: Binary masks, skeleton overlays, classified region maps, region & summary CSVs
- `figures/`: 6-model comparison panels, comparative charts, 2x2 boxplots
- `tables/`: Per-image accuracy CSV, per-image morphometrics CSV, model mean summary CSV
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.contour_detection import (
    label_regions,
    link_endpoints,
    prune_spurs,
    skeletonize_binary,
)
from src.contour_refinement import refine_contours
from src.evaluation import compute_segmentation_metrics, generate_reference_mask
from src.image_io import load_image, to_grayscale_uint8
from src.preprocessing import preprocess
from src.quantification import (
    compute_region_morphometrics,
    compute_summary_metrics,
    export_csv,
)
from src.region_classification import (
    classify_all_regions,
    create_classification_mask,
)
from src.thresholding import SUPPORTED_METHODS, apply_threshold
from src.visualization import (
    save_binary_mask,
    save_classification_mask,
    save_skeleton_overlay,
)
from src.void_detection import detect_voids


def load_or_generate_reference(
    gray: np.ndarray,
    sample_name: str,
    ref_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load existing reference mask or generate high-confidence consensus reference."""
    os.makedirs(ref_dir, exist_ok=True)
    bin_path = os.path.join(ref_dir, f"{sample_name}_ref_binary.png")
    skel_path = os.path.join(ref_dir, f"{sample_name}_ref_skeleton.png")

    if os.path.isfile(bin_path) and os.path.isfile(skel_path):
        ref_bin = cv2.imread(bin_path, cv2.IMREAD_GRAYSCALE)
        ref_skel = cv2.imread(skel_path, cv2.IMREAD_GRAYSCALE)
        if ref_bin is not None and ref_skel is not None:
            print(f"  [Reference] Loaded cached consensus reference mask for {sample_name}", flush=True)
            return ref_bin, ref_skel

    print(f"  [Reference] Generating multi-scale consensus reference mask for {sample_name}...", flush=True)
    t0 = time.perf_counter()
    ref_bin, ref_skel = generate_reference_mask(gray, sample_name=sample_name, output_dir=ref_dir)
    print(f"  [Reference] Generated reference mask in {time.perf_counter() - t0:.1f}s", flush=True)
    return ref_bin, ref_skel


def plot_task3_comparison_charts(
    df_acc: pd.DataFrame,
    df_morph: pd.DataFrame,
    output_path: str,
) -> None:
    """Generate multi-panel bar/line comparison charts across all orientations and models."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Merge accuracy and morphometrics
    df = pd.merge(df_acc, df_morph, on=["Sample", "Orientation", "Method"])

    methods = ["otsu", "yen", "triangle", "li", "adaptive", "gmm"]
    orientations = [0, 60, 90]
    orient_labels = ["0° (A)", "60° (B)", "90° (C)"]

    palette = {
        "otsu": "#1f77b4",
        "yen": "#ff7f0e",
        "triangle": "#2ca02c",
        "li": "#d62728",
        "adaptive": "#9467bd",
        "gmm": "#8c564b",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=150)
    fig.suptitle(
        "Task 3: Comparative Model Benchmark Across Orientations (0°, 60°, 90°)\n"
        "Segmentation Accuracy & Quantitative Microstructural Fractions",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # 1. IoU
    ax = axes[0, 0]
    x = np.arange(len(orientations))
    width = 0.13
    for idx, m in enumerate(methods):
        sub = df[df["Method"] == m].sort_values("Orientation")
        vals = [sub[sub["Orientation"] == o]["IoU"].values[0] for o in orientations]
        ax.bar(x + (idx - 2.5) * width, vals, width, label=m.upper(), color=palette[m], alpha=0.9)
    ax.set_title("Intersection over Union (IoU)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(orient_labels)
    ax.set_ylabel("IoU Score")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    # 2. Dice Coefficient
    ax = axes[0, 1]
    for idx, m in enumerate(methods):
        sub = df[df["Method"] == m].sort_values("Orientation")
        vals = [sub[sub["Orientation"] == o]["Dice"].values[0] for o in orientations]
        ax.bar(x + (idx - 2.5) * width, vals, width, label=m.upper(), color=palette[m], alpha=0.9)
    ax.set_title("Dice Coefficient (F1-Score)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(orient_labels)
    ax.set_ylabel("Dice Score")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 3. Boundary F1
    ax = axes[0, 2]
    for idx, m in enumerate(methods):
        sub = df[df["Method"] == m].sort_values("Orientation")
        vals = [sub[sub["Orientation"] == o]["Boundary_F1"].values[0] for o in orientations]
        ax.bar(x + (idx - 2.5) * width, vals, width, label=m.upper(), color=palette[m], alpha=0.9)
    ax.set_title("Boundary F1 (Contour Alignment ±2px)", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(orient_labels)
    ax.set_ylabel("Boundary F1")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # 4. Total Region Count
    ax = axes[1, 0]
    for idx, m in enumerate(methods):
        sub = df[df["Method"] == m].sort_values("Orientation")
        vals = [sub[sub["Orientation"] == o]["Total_Regions"].values[0] for o in orientations]
        ax.plot(orient_labels, vals, marker="o", linewidth=2, label=m.upper(), color=palette[m])
    ax.set_title("Total Dimple Region Count", fontsize=11, fontweight="bold")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle="--", alpha=0.4)

    # 5. Void Area Fraction (AF_Void)
    ax = axes[1, 1]
    for idx, m in enumerate(methods):
        sub = df[df["Method"] == m].sort_values("Orientation")
        vals = [sub[sub["Orientation"] == o]["AF_Void"].values[0] for o in orientations]
        ax.plot(orient_labels, vals, marker="s", linewidth=2, label=m.upper(), color=palette[m])
    ax.set_title("Ductile Micro-Void Area Fraction (AF_Void)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Area Fraction")
    ax.grid(True, linestyle="--", alpha=0.4)

    # 6. Surface Area Fraction (AF_Surface)
    ax = axes[1, 2]
    for idx, m in enumerate(methods):
        sub = df[df["Method"] == m].sort_values("Orientation")
        vals = [sub[sub["Orientation"] == o]["AF_Surface"].values[0] for o in orientations]
        ax.plot(orient_labels, vals, marker="^", linewidth=2, label=m.upper(), color=palette[m])
    ax.set_title("Cleavage / Surface Area Fraction (AF_Surface)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Area Fraction")
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] Saved model comparison charts to: {output_path}")


def plot_task3_boxplots(df_acc: pd.DataFrame, output_path: str) -> None:
    """Generate 2x2 publication boxplots with sample dot overlays for IoU, Dice, Boundary F1, and log Runtime."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    methods = ["otsu", "yen", "triangle", "li", "adaptive", "gmm"]
    metrics_info = [
        ("IoU", "Intersection over Union (IoU)", False),
        ("Dice", "Dice Coefficient (F1)", False),
        ("Boundary_F1", "Boundary F1 (Contour Alignment)", False),
        ("Runtime_ms", "Thresholding Runtime (ms, Log Scale)", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=150)
    fig.suptitle(
        "Task 3: Distribution of Segmentation Accuracy & Runtime Across 6 Models\n"
        "Evaluation on Blind-Validation Micrographs A (0°), B (60°), and C (90°)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for idx, (metric_key, metric_title, is_log) in enumerate(metrics_info):
        r = idx // 2
        c = idx % 2
        ax = axes[r, c]

        data = [df_acc[df_acc["Method"] == m][metric_key].values for m in methods]

        bp = ax.boxplot(
            data,
            tick_labels=[m.upper() for m in methods],
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "D", "markeredgecolor": "black", "markerfacecolor": "gold", "markersize": 6},
            medianprops={"color": "red", "linewidth": 1.5},
            boxprops={"facecolor": "#e6f2ff", "edgecolor": "#1f77b4"},
        )

        # Overlay jittered data points for the 3 samples
        rng = np.random.RandomState(42)
        for i_m, m in enumerate(methods):
            vals = df_acc[df_acc["Method"] == m][metric_key].values
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                np.full_like(vals, i_m + 1) + jitter,
                vals,
                color="#003366",
                alpha=0.8,
                s=35,
                zorder=3,
            )

        ax.set_title(metric_title, fontsize=11, fontweight="bold")
        if is_log:
            ax.set_yscale("log")
            ax.set_ylabel("Runtime (ms, log scale)")
        else:
            ax.set_ylabel("Score")
            ax.set_ylim(-0.05, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Figure] Saved publication boxplots to: {output_path}")


def run_task3_analysis():
    task3_base_dir = os.path.join("outputs", "task3_analysis")
    ref_dir = os.path.join(task3_base_dir, "reference_masks")
    fig_dir = os.path.join(task3_base_dir, "figures")
    table_dir = os.path.join(task3_base_dir, "tables")

    for d in [task3_base_dir, ref_dir, fig_dir, table_dir]:
        os.makedirs(d, exist_ok=True)

    dataset_configs = [
        {
            "name": "sample_A_0",
            "path": "data/task3_cropped/sample_A_0.png",
            "orientation": 0,
            "min_area": 30,
            "void_ratio": 0.50,
            "scale": 1.0,
        },
        {
            "name": "sample_B_60",
            "path": "data/task3_cropped/sample_B_60.png",
            "orientation": 60,
            "min_area": 30,
            "void_ratio": 0.50,
            "scale": 1.0,
        },
        {
            "name": "sample_C_90",
            "path": "data/task3_cropped/sample_C_90.png",
            "orientation": 90,
            "min_area": 30,
            "void_ratio": 0.50,
            "scale": 1.0,
        },
    ]

    methods = list(SUPPORTED_METHODS)
    for m in methods:
        os.makedirs(os.path.join(task3_base_dir, m), exist_ok=True)

    accuracy_records: List[Dict[str, Any]] = []
    morphometrics_records: List[Dict[str, Any]] = []

    print("\n" + "=" * 75)
    print("  FRACTOMORPH: TASK 3 SIX-MODEL COMPARATIVE BENCHMARK")
    print(f"  Target directory: {task3_base_dir}")
    print("=" * 75)

    for cfg in dataset_configs:
        sample_name = cfg["name"]
        img_path = cfg["path"]
        orientation = cfg["orientation"]
        min_area = cfg["min_area"]
        void_ratio = cfg["void_ratio"]
        scale = cfg["scale"]

        print(f"\n>>> Processing Task 3 Dataset: {sample_name} (Orientation: {orientation}°)")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Cropped image not found: {img_path}. Run extract_and_crop_task3.py first.")

        raw_img = load_image(img_path)
        gray = to_grayscale_uint8(raw_img)

        # Stage 0: Preprocessing (Bilateral filtering & Histogram smoothing)
        prep = preprocess(gray)
        denoised = prep["denoised"]
        hist_smooth = prep["histogram_smooth"]

        # Multi-scale consensus reference mask
        ref_binary, ref_skel = load_or_generate_reference(
            gray, sample_name=sample_name, ref_dir=ref_dir
        )

        sample_comparison_panels: Dict[str, Dict[str, Any]] = {}

        for method in methods:
            method_dir = os.path.join(task3_base_dir, method)

            thresh_kwargs: Dict[str, Any] = {}
            if method == "adaptive":
                thresh_kwargs = {"block_size": 35, "offset": -5.0}
            elif method == "gmm":
                thresh_kwargs = {"n_components": 2, "random_state": 42}

            # Stage 1: Automated Thresholding
            thresh_res = apply_threshold(denoised, method=method, **thresh_kwargs)
            runtime_ms = thresh_res["runtime_ms"]
            binary = thresh_res["binary"]
            thresh_val = float(thresh_res["threshold_summary"])

            # Downstream morphological operations held strictly constant
            skeleton_raw = skeletonize_binary(binary)
            skeleton_linked = link_endpoints(skeleton_raw, max_gap=10)
            skeleton_pruned = prune_spurs(skeleton_linked, max_spur_length=8)
            label_map, num_regions = label_regions(skeleton_pruned)

            # Stage 2: Contour Refinement
            refined = refine_contours(
                gray,
                skeleton_pruned,
                label_map,
                num_regions,
                min_area=min_area,
                threshold_method=method,
                threshold_kwargs=thresh_kwargs,
            )
            final_skel = refined["skeleton"]
            final_label_map = refined["label_map"]
            final_num_regions = refined["num_regions"]

            # Stage 3: Void Detection
            voids = detect_voids(
                denoised, hist_smooth, final_label_map, final_num_regions
            )

            # Stage 4: 3-Class Classification
            classifications = classify_all_regions(
                voids["void_fractions"], tau_void=void_ratio
            )
            class_mask = create_classification_mask(
                final_label_map, classifications
            )

            # Stage 5: Quantification
            df_regions = compute_region_morphometrics(
                final_label_map, classifications, voids["void_fractions"], scale=scale
            )
            image_area_px = int(np.sum(final_label_map > 0))
            df_summary = compute_summary_metrics(df_regions, image_area_px, scale=scale)

            # Accuracy Metrics against consensus reference
            metrics = compute_segmentation_metrics(
                binary, ref_binary, final_skel, ref_skel, tolerance_px=2
            )

            # Save per-method per-sample output artifacts
            save_binary_mask(
                binary, os.path.join(method_dir, f"{sample_name}_binary.png")
            )
            save_skeleton_overlay(
                gray, final_skel, os.path.join(method_dir, f"{sample_name}_skeleton.png")
            )
            save_classification_mask(
                class_mask, os.path.join(method_dir, f"{sample_name}_classified.png")
            )
            export_csv(
                df_regions, os.path.join(method_dir, f"{sample_name}_regions.csv")
            )
            export_csv(
                df_summary, os.path.join(method_dir, f"{sample_name}_summary.csv")
            )

            if method == "adaptive":
                t_surf = thresh_res["threshold"]
                t_norm = cv2.normalize(
                    t_surf, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
                cv2.imwrite(
                    os.path.join(method_dir, f"{sample_name}_adaptive_threshold_map.png"),
                    t_norm,
                )

            # Extract summary counts and fractions
            sum_row = df_summary.iloc[0] if len(df_summary) > 0 else {}
            n_void = sum(1 for c in classifications.values() if c == "Void")
            n_surface = sum(1 for c in classifications.values() if c == "Surface")
            n_mixed = sum(1 for c in classifications.values() if c == "Mixed")

            af_void = float(sum_row.get("AF_Void", 0.0))
            af_surface = float(sum_row.get("AF_Surface", 0.0))
            af_mixed = float(sum_row.get("AF_Mixed", 0.0))
            sum_af = af_void + af_surface + af_mixed

            mean_ecd = float(df_regions["ECD_um"].mean()) if "ECD_um" in df_regions and len(df_regions) > 0 else 0.0
            ar_col = "Aspect_Ratio" if "Aspect_Ratio" in df_regions.columns else ("aspect_ratio" if "aspect_ratio" in df_regions.columns else None)
            circ_col = "Circularity" if "Circularity" in df_regions.columns else ("circularity" if "circularity" in df_regions.columns else None)
            mean_ar = float(df_regions[ar_col].mean()) if ar_col is not None and len(df_regions) > 0 else 0.0
            mean_circ = float(df_regions[circ_col].mean()) if circ_col is not None and len(df_regions) > 0 else 0.0

            accuracy_records.append({
                "Sample": sample_name,
                "Orientation": orientation,
                "Method": method,
                "Threshold_T": thresh_val,
                "IoU": metrics["iou"],
                "Dice": metrics["dice"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "Boundary_F1": metrics["boundary_f1"],
                "Runtime_ms": runtime_ms,
            })

            morphometrics_records.append({
                "Sample": sample_name,
                "Orientation": orientation,
                "Method": method,
                "Total_Regions": final_num_regions,
                "Void_Count": n_void,
                "Surface_Count": n_surface,
                "Mixed_Count": n_mixed,
                "AF_Void": af_void,
                "AF_Surface": af_surface,
                "AF_Mixed": af_mixed,
                "Sum_AF": sum_af,
                "Mean_ECD": mean_ecd,
                "Mean_Aspect_Ratio": mean_ar,
                "Mean_Circularity": mean_circ,
            })

            sample_comparison_panels[method] = {
                "class_mask": class_mask,
                "skeleton": final_skel,
                "thresh_val": thresh_val,
                "iou": metrics["iou"],
                "dice": metrics["dice"],
                "boundary_f1": metrics["boundary_f1"],
                "regions": final_num_regions,
                "af_void": af_void,
            }

            print(
                f"    [{method:8s}] T={thresh_val:5.1f} | IoU={metrics['iou']:.3f} | "
                f"Dice={metrics['dice']:.3f} | B-F1={metrics['boundary_f1']:.3f} | "
                f"Regions={final_num_regions:4d} | AF_Void={af_void:.3f}",
                flush=True,
            )

        # 6-panel side-by-side comparison figure for this sample
        comp_fig_path = os.path.join(fig_dir, f"{sample_name}_6models_comparison.png")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
        fig.suptitle(
            f"Task 3 Automated Thresholding Benchmark: {sample_name} ({orientation}° Orientation)\n"
            "3-Class Fracture Region Map (Red=Void, Blue=Surface, Yellow=Mixed) + Skeleton Contours",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        for idx, method in enumerate(methods):
            r = idx // 3
            c = idx % 3
            ax = axes[r, c]

            panel_info = sample_comparison_panels[method]
            mask_rgb = panel_info["class_mask"].copy()
            # Overlay 1-px skeleton lines in white
            mask_rgb[panel_info["skeleton"] > 0] = [255, 255, 255]

            ax.imshow(mask_rgb)
            title = (
                f"{method.upper()} (T = {panel_info['thresh_val']:.1f})\n"
                f"IoU: {panel_info['iou']:.3f} | Dice: {panel_info['dice']:.3f} | "
                f"B-F1: {panel_info['boundary_f1']:.3f}\n"
                f"Regions: {panel_info['regions']} | AF_Void: {panel_info['af_void']:.3f}"
            )
            ax.set_title(title, fontsize=11, fontweight="semibold")
            ax.axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(comp_fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Figure] Saved 6-model comparison grid to: {comp_fig_path}")

    # Build and export Task 3 tables
    df_acc = pd.DataFrame(accuracy_records)
    df_morph = pd.DataFrame(morphometrics_records)

    acc_csv_path = os.path.join(table_dir, "task3_per_image_accuracy.csv")
    morph_csv_path = os.path.join(table_dir, "task3_per_image_morphometrics.csv")
    export_csv(df_acc, acc_csv_path)
    export_csv(df_morph, morph_csv_path)
    print(f"\n[Tables] Exported per-image accuracy table to:\n  -> {acc_csv_path}")
    print(f"[Tables] Exported per-image morphometrics table to:\n  -> {morph_csv_path}")

    # Compute model mean summary across the 3 images
    df_combined = pd.merge(df_acc, df_morph, on=["Sample", "Orientation", "Method"])

    summary_rows = []
    for method in methods:
        sub = df_combined[df_combined["Method"] == method]
        summary_rows.append({
            "Method": method,
            "IoU_mean": float(sub["IoU"].mean()),
            "IoU_std": float(sub["IoU"].std()),
            "Dice_mean": float(sub["Dice"].mean()),
            "Dice_std": float(sub["Dice"].std()),
            "Precision_mean": float(sub["Precision"].mean()),
            "Precision_std": float(sub["Precision"].std()),
            "Recall_mean": float(sub["Recall"].mean()),
            "Recall_std": float(sub["Recall"].std()),
            "Boundary_F1_mean": float(sub["Boundary_F1"].mean()),
            "Boundary_F1_std": float(sub["Boundary_F1"].std()),
            "Runtime_ms_mean": float(sub["Runtime_ms"].mean()),
            "Runtime_ms_std": float(sub["Runtime_ms"].std()),
            "Total_Regions_mean": float(sub["Total_Regions"].mean()),
            "Total_Regions_std": float(sub["Total_Regions"].std()),
            "AF_Void_mean": float(sub["AF_Void"].mean()),
            "AF_Void_std": float(sub["AF_Void"].std()),
            "AF_Surface_mean": float(sub["AF_Surface"].mean()),
            "AF_Surface_std": float(sub["AF_Surface"].std()),
            "AF_Mixed_mean": float(sub["AF_Mixed"].mean()),
            "AF_Mixed_std": float(sub["AF_Mixed"].std()),
            "Mean_ECD_mean": float(sub["Mean_ECD"].mean()),
            "Mean_ECD_std": float(sub["Mean_ECD"].std()),
            "Mean_Aspect_Ratio_mean": float(sub["Mean_Aspect_Ratio"].mean()),
            "Mean_Aspect_Ratio_std": float(sub["Mean_Aspect_Ratio"].std()),
            "Mean_Circularity_mean": float(sub["Mean_Circularity"].mean()),
            "Mean_Circularity_std": float(sub["Mean_Circularity"].std()),
        })

    df_summary_models = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(table_dir, "task3_model_mean_summary.csv")
    export_csv(df_summary_models, summary_csv_path)
    print(f"[Tables] Exported model mean summary table to:\n  -> {summary_csv_path}")

    # Generate cross-sample comparison charts & publication boxplots
    charts_fig_path = os.path.join(fig_dir, "task3_model_comparison_charts.png")
    plot_task3_comparison_charts(df_acc, df_morph, charts_fig_path)

    boxplots_fig_path = os.path.join(fig_dir, "task3_boxplots.png")
    plot_task3_boxplots(df_acc, boxplots_fig_path)

    print("\n" + "=" * 75)
    print("  TASK 3 BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"  All outputs saved under: {os.path.abspath(task3_base_dir)}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_task3_analysis()
