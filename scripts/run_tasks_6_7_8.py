#!/usr/bin/env python3
"""
CAMMP-SFSE: Tasks 6 and 7 Benchmark Execution Script

Executes:
- Task 6: Controlled perturbation sweeps (brightness, contrast, noise, scale)
  and hyperparameter sensitivity analysis (adaptive block_size/offset, GMM components).
- Task 7: Statistical significance testing (paired t-test & Wilcoxon signed-rank vs Otsu),
  95% confidence intervals, and publication 2x2 boxplots with sample dot overlays.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.image_io import load_image, to_grayscale_uint8
from src.quantification import export_csv
from src.robustness import (
    compute_detailed_statistical_analysis,
    plot_benchmark_boxplots,
    plot_robustness_and_sensitivity,
    run_hyperparameter_sweeps,
    run_perturbation_sweeps,
)


def main():
    benchmark_dir = os.path.join("outputs", "threshold_benchmark")
    table_dir = os.path.join(benchmark_dir, "tables")
    fig_dir = os.path.join(benchmark_dir, "figures")

    for d in [table_dir, fig_dir]:
        os.makedirs(d, exist_ok=True)

    print("\n" + "=" * 70)
    print("  FRACTOMORPH: TASKS 6 & 7 BENCHMARK EXECUTION")
    print(f"  Target directory: {benchmark_dir}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Task 6: Perturbation Robustness & Hyperparameter Sweeps
    # -----------------------------------------------------------------------
    print("\n[Task 6] Loading representative images for sensitivity analysis...")
    sample_paths = [
        ("synthetic_sem", "data/sample/synthetic_sem.png"),
        ("sem_real_1", "data/raw/sem_real_1.png"),
    ]

    samples = []
    for name, path in sample_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        raw = load_image(path)
        gray = to_grayscale_uint8(raw)
        samples.append((name, gray))

    print("  Running image perturbation sweeps (brightness, contrast, noise, resolution)...")
    df_pert = run_perturbation_sweeps(samples, gmm_samples=25000)
    pert_csv_path = os.path.join(table_dir, "perturbation_robustness.csv")
    export_csv(df_pert, pert_csv_path)
    print(f"  Exported: {pert_csv_path} ({len(df_pert)} evaluations)")

    print("  Running hyperparameter sensitivity sweeps (adaptive block/offset, GMM k)...")
    df_hyper = run_hyperparameter_sweeps(samples, gmm_samples=25000)
    hyper_csv_path = os.path.join(table_dir, "hyperparameter_sensitivity.csv")
    export_csv(df_hyper, hyper_csv_path)
    print(f"  Exported: {hyper_csv_path} ({len(df_hyper)} evaluations)")

    print("  Generating multi-panel sensitivity figures...")
    sens_fig_path = os.path.join(fig_dir, "robustness_and_sensitivity.png")
    plot_robustness_and_sensitivity(df_pert, df_hyper, sens_fig_path)
    print(f"  Saved figure: {sens_fig_path}")

    # -----------------------------------------------------------------------
    # Task 7: Statistical Analysis & Publication Boxplots
    # -----------------------------------------------------------------------
    print("\n[Task 7] Computing paired statistical tests and 95% Confidence Intervals...")
    acc_csv_path = os.path.join(table_dir, "per_image_accuracy_metrics.csv")
    if not os.path.exists(acc_csv_path):
        raise FileNotFoundError(f"Missing {acc_csv_path}. Please run Tasks 3, 4, 5 first.")

    df_acc = pd.read_csv(acc_csv_path)
    df_stats = compute_detailed_statistical_analysis(df_acc)
    stats_csv_path = os.path.join(table_dir, "statistical_tests_vs_otsu.csv")
    export_csv(df_stats, stats_csv_path)
    print(f"  Exported: {stats_csv_path}")

    print("  Generating publication 2x2 box plot figure...")
    boxplot_fig_path = os.path.join(fig_dir, "benchmark_boxplots.png")
    plot_benchmark_boxplots(df_acc, boxplot_fig_path)
    print(f"  Saved figure: {boxplot_fig_path}")

    print("\n" + "=" * 70)
    print("  TASKS 6 & 7 COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    # Print statistical table to console
    print("\n--- STATISTICAL TESTS VS OTSU (TABLE) ---")
    print(df_stats.to_string(index=False))
    print("\n")


if __name__ == "__main__":
    main()
