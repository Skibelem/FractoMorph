#!/usr/bin/env python3
"""
CAMMP-SFSE: Tasks 3, 4, and 5 Benchmark Runner

Executes the automated thresholding benchmark across all 6 methods:
["otsu", "yen", "triangle", "li", "adaptive", "gmm"]
on 6 SEM fractograph datasets (1 synthetic with ground-truth geometry,
5 real SEM images with multi-scale consensus reference masks).

Outputs:
- Reference masks in `outputs/threshold_benchmark/reference_masks/`
- Method-specific outputs in `outputs/threshold_benchmark/<method>/`
- 6-panel side-by-side comparison figures in `outputs/threshold_benchmark/figures/`
- Aggregated CSV tables in `outputs/threshold_benchmark/tables/`
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add repo root to sys.path
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


def run_benchmark():
    benchmark_dir = os.path.join("outputs", "threshold_benchmark")
    ref_dir = os.path.join(benchmark_dir, "reference_masks")
    fig_dir = os.path.join(benchmark_dir, "figures")
    table_dir = os.path.join(benchmark_dir, "tables")

    for d in [benchmark_dir, ref_dir, fig_dir, table_dir]:
        os.makedirs(d, exist_ok=True)

    dataset_configs = [
        {
            "name": "synthetic_sem",
            "path": "data/sample/synthetic_sem.png",
            "min_area": 25,
            "void_ratio": 0.20,
            "scale": 1.0,
        },
        {
            "name": "sem_real_1",
            "path": "data/raw/sem_real_1.png",
            "min_area": 30,
            "void_ratio": 0.50,
            "scale": 1.0,
        },
        {
            "name": "sem_real_2",
            "path": "data/raw/sem_real_2.png",
            "min_area": 30,
            "void_ratio": 0.50,
            "scale": 1.0,
        },
        {
            "name": "sem_real_3",
            "path": "data/raw/sem_real_3.png",
            "min_area": 30,
            "void_ratio": 0.50,
            "scale": 1.0,
        },
        {
            "name": "sem_real_4",
            "path": "data/raw/sem_real_4.png",
            "min_area": 30,
            "void_ratio": 0.50,
            "scale": 1.0,
        },
        {
            "name": "sem_real_5",
            "path": "data/raw/sem_real_5.png",
            "min_area": 30,
            "void_ratio": 0.50,
            "scale": 1.0,
        },
    ]

    methods = list(SUPPORTED_METHODS)
    for m in methods:
        os.makedirs(os.path.join(benchmark_dir, m), exist_ok=True)

    accuracy_records: List[Dict[str, Any]] = []
    morphometrics_records: List[Dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("  FRACTOMORPH: TASK 2 BENCHMARK (TASKS 3, 4, AND 5)")
    print(f"  Target directory: {benchmark_dir}")
    print("=" * 70)

    for cfg in dataset_configs:
        sample_name = cfg["name"]
        img_path = cfg["path"]
        min_area = cfg["min_area"]
        void_ratio = cfg["void_ratio"]
        scale = cfg["scale"]

        print(f"\n>>> Processing dataset: {sample_name} ({img_path})")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        raw_img = load_image(img_path)
        gray = to_grayscale_uint8(raw_img)

        # Stage 0: Preprocessing
        prep = preprocess(gray)
        denoised = prep["denoised"]
        hist_smooth = prep["histogram_smooth"]

        # Task 3: Reference mask generation
        print(f"  [Task 3] Generating reference mask for {sample_name}...")
        ref_binary, ref_skel = generate_reference_mask(
            gray, sample_name=sample_name, output_dir=ref_dir
        )

        sample_comparison_panels: Dict[str, Dict[str, Any]] = {}

        for method in methods:
            method_dir = os.path.join(benchmark_dir, method)

            thresh_kwargs: Dict[str, Any] = {}
            if method == "adaptive":
                thresh_kwargs = {"block_size": 35, "offset": -5.0}
            elif method == "gmm":
                thresh_kwargs = {"n_components": 2, "random_state": 42}

            # Stage 1: Thresholding & Contour Detection
            t_thresh_start = time.perf_counter()
            thresh_res = apply_threshold(denoised, method=method, **thresh_kwargs)
            runtime_ms = thresh_res["runtime_ms"]

            binary = thresh_res["binary"]
            thresh_val = float(thresh_res["threshold_summary"])

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

            # Stage 4: Classification
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

            # Task 5: Accuracy Metrics
            metrics = compute_segmentation_metrics(
                binary, ref_binary, final_skel, ref_skel, tolerance_px=2
            )

            # Save per-method per-image outputs
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

            # Counts and area fractions
            sum_row = df_summary.iloc[0] if len(df_summary) > 0 else {}
            n_void = sum(1 for c in classifications.values() if c == "Void")
            n_surface = sum(1 for c in classifications.values() if c == "Surface")
            n_mixed = sum(1 for c in classifications.values() if c == "Mixed")

            af_void = float(sum_row.get("AF_Void", 0.0))
            af_surface = float(sum_row.get("AF_Surface", 0.0))
            af_mixed = float(sum_row.get("AF_Mixed", 0.0))
            sum_af = af_void + af_surface + af_mixed

            mean_ecd = float(df_regions["ECD_um"].mean()) if "ECD_um" in df_regions else 0.0
            mean_ar = float(df_regions["aspect_ratio"].mean()) if "aspect_ratio" in df_regions else 0.0
            mean_circ = float(df_regions["circularity"].mean()) if "circularity" in df_regions else 0.0

            accuracy_records.append({
                "Sample": sample_name,
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
                f"Dice={metrics['dice']:.3f} | B_F1={metrics['boundary_f1']:.3f} | "
                f"Regions={final_num_regions:3d} | AF_Void={af_void:.3f}"
            )

        # Generate 6-panel visual comparison figure for this sample
        comp_fig_path = os.path.join(fig_dir, f"{sample_name}_6methods_comparison.png")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"Automated Thresholding Comparison: {sample_name}",
            fontsize=16,
            fontweight="bold",
        )

        for idx, method in enumerate(methods):
            r = idx // 3
            c = idx % 3
            ax = axes[r, c]

            panel_info = sample_comparison_panels[method]
            mask_rgb = panel_info["class_mask"].copy()
            # Overlay skeleton lines in white on the classification mask
            mask_rgb[panel_info["skeleton"] > 0] = [255, 255, 255]

            ax.imshow(mask_rgb)
            title = (
                f"{method.upper()} (T = {panel_info['thresh_val']:.1f})\n"
                f"IoU: {panel_info['iou']:.3f} | Dice: {panel_info['dice']:.3f} | "
                f"B-F1: {panel_info['boundary_f1']:.3f}\n"
                f"Regions: {panel_info['regions']} | AF_Void: {panel_info['af_void']:.3f}"
            )
            ax.set_title(title, fontsize=11)
            ax.axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(comp_fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Figure] Saved 6-method comparison to: {comp_fig_path}")

    # Build and save Task 5 tables
    df_acc = pd.DataFrame(accuracy_records)
    df_morph = pd.DataFrame(morphometrics_records)

    acc_csv_path = os.path.join(table_dir, "per_image_accuracy_metrics.csv")
    morph_csv_path = os.path.join(table_dir, "per_image_morphometrics.csv")
    export_csv(df_acc, acc_csv_path)
    export_csv(df_morph, morph_csv_path)
    print(f"\n[Tables] Exported {acc_csv_path}")
    print(f"[Tables] Exported {morph_csv_path}")

    # Method Mean Summary (Aggregated Mean ± SD across all 6 images for each of 6 methods)
    summary_rows = []
    # Merge df_acc and df_morph for complete cross-metric aggregation
    df_combined = pd.merge(df_acc, df_morph, on=["Sample", "Method"])

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
        })

    df_summary_methods = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(table_dir, "method_mean_summary.csv")
    export_csv(df_summary_methods, summary_csv_path)
    print(f"[Tables] Exported {summary_csv_path}")

    print("\n" + "=" * 70)
    print("  TASK 2 BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"  All outputs saved in: {os.path.abspath(benchmark_dir)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_benchmark()
