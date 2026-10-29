"""
Quick fix script to update Mean_Aspect_Ratio and Mean_Circularity in Task 3
morphometric tables without re-running any thresholding or segmentation.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = REPO_ROOT / "outputs" / "task3_analysis"
TABLE_DIR = BASE_DIR / "tables"

morph_path = TABLE_DIR / "task3_per_image_morphometrics.csv"
acc_path = TABLE_DIR / "task3_per_image_accuracy.csv"
summary_path = TABLE_DIR / "task3_model_mean_summary.csv"

df_morph = pd.read_csv(morph_path)
df_acc = pd.read_csv(acc_path)

print(f"Loaded {len(df_morph)} morphometrics rows.")

updated_ar = []
updated_circ = []

for idx, row in df_morph.iterrows():
    sample = row["Sample"]
    method = row["Method"]
    
    reg_csv = BASE_DIR / method / f"{sample}_regions.csv"
    if not reg_csv.is_file():
        raise FileNotFoundError(f"Missing region CSV: {reg_csv}")
    
    df_reg = pd.read_csv(reg_csv)
    
    ar_col = "Aspect_Ratio" if "Aspect_Ratio" in df_reg.columns else ("aspect_ratio" if "aspect_ratio" in df_reg.columns else None)
    circ_col = "Circularity" if "Circularity" in df_reg.columns else ("circularity" if "circularity" in df_reg.columns else None)
    
    mean_ar = float(df_reg[ar_col].mean()) if (ar_col is not None and len(df_reg) > 0) else 0.0
    mean_circ = float(df_reg[circ_col].mean()) if (circ_col is not None and len(df_reg) > 0) else 0.0
    
    updated_ar.append(mean_ar)
    updated_circ.append(mean_circ)
    print(f"[{sample:12s} | {method:8s}] Mean AR = {mean_ar:.4f}, Mean Circ = {mean_circ:.4f}")

df_morph["Mean_Aspect_Ratio"] = updated_ar
df_morph["Mean_Circularity"] = updated_circ

# Save updated task3_per_image_morphometrics.csv
df_morph.to_csv(morph_path, index=False)
print(f"\nSaved updated morphometrics table to: {morph_path}")

# Recompute summary table across all 3 images per model
df_merged = pd.merge(df_acc, df_morph, on=["Sample", "Orientation", "Method"])

methods = ["otsu", "yen", "triangle", "li", "adaptive", "gmm"]
numeric_metrics = [
    "IoU", "Dice", "Precision", "Recall", "Boundary_F1", "Runtime_ms",
    "Total_Regions", "AF_Void", "AF_Surface", "AF_Mixed",
    "Mean_ECD", "Mean_Aspect_Ratio", "Mean_Circularity",
]

summary_rows = []
for m in methods:
    sub = df_merged[df_merged["Method"] == m]
    row_data = {"Method": m}
    for metric in numeric_metrics:
        row_data[f"{metric}_mean"] = float(sub[metric].mean())
        row_data[f"{metric}_std"] = float(sub[metric].std(ddof=1)) if len(sub) > 1 else 0.0
    summary_rows.append(row_data)

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(summary_path, index=False)
print(f"Saved updated summary table to: {summary_path}")

# Re-generate comparison charts
try:
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.run_task3_analysis import plot_task3_comparison_charts
    chart_path = BASE_DIR / "figures" / "task3_model_comparison_charts.png"
    plot_task3_comparison_charts(df_acc, df_morph, str(chart_path))
    print(f"Updated comparison charts at: {chart_path}")
except Exception as e:
    print(f"Note on charts: {e}")

print("\n--- UPDATED task3_per_image_morphometrics.csv ---")
print(df_morph.to_string(index=False))
