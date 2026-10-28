"""
Tests for src/robustness.py (Task 6 & Task 7 functionality).
"""

import numpy as np
import pandas as pd
import pytest

from src.robustness import (
    apply_brightness_shift,
    apply_contrast_scale,
    apply_gaussian_noise,
    apply_resolution_scale,
    compute_detailed_statistical_analysis,
    evaluate_threshold_accuracy_fast,
    run_hyperparameter_sweeps,
    run_perturbation_sweeps,
)


@pytest.fixture
def synthetic_pair():
    """Create a pair of test image and reference mask."""
    img = np.full((64, 64), 50, dtype=np.uint8)
    img[25:35, :] = 200
    ref_binary = (img > 120).astype(np.uint8) * 255
    ref_skel = np.zeros((64, 64), dtype=np.uint8)
    ref_skel[30, :] = 255
    return img, ref_binary, ref_skel


class TestImagePerturbations:
    def test_brightness_shift(self):
        img = np.full((10, 10), 100, dtype=np.uint8)
        shifted_pos = apply_brightness_shift(img, delta=20)
        assert np.all(shifted_pos == 120)
        shifted_neg = apply_brightness_shift(img, delta=-30)
        assert np.all(shifted_neg == 70)
        # Clipping
        clipped = apply_brightness_shift(img, delta=200)
        assert np.all(clipped == 255)

    def test_contrast_scale(self):
        img = np.full((10, 10), 100, dtype=np.uint8)
        scaled = apply_contrast_scale(img, alpha=1.2)
        assert np.all(scaled == 120)
        scaled_down = apply_contrast_scale(img, alpha=0.5)
        assert np.all(scaled_down == 50)

    def test_gaussian_noise(self):
        img = np.full((50, 50), 100, dtype=np.uint8)
        noisy = apply_gaussian_noise(img, sigma=10.0, seed=42)
        assert noisy.shape == img.shape
        assert noisy.dtype == np.uint8
        assert not np.array_equal(noisy, img)
        assert abs(float(np.mean(noisy)) - 100.0) < 2.0

    def test_resolution_scale(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        scaled = apply_resolution_scale(img, scale=0.5)
        assert scaled.shape == (50, 50)
        full = apply_resolution_scale(img, scale=1.0)
        assert full.shape == (100, 100)


class TestFastEvaluation:
    def test_evaluate_threshold_accuracy_fast(self, synthetic_pair):
        img, ref_bin, ref_skel = synthetic_pair
        res = evaluate_threshold_accuracy_fast(img, ref_bin, ref_skel, method="otsu")
        assert "iou" in res
        assert "dice" in res
        assert "boundary_f1" in res
        assert 0.0 <= res["iou"] <= 1.0
        assert 0.0 <= res["boundary_f1"] <= 1.0


class TestStatisticalAnalysis:
    def test_compute_detailed_statistical_analysis(self):
        # Create mock accuracy records for 6 samples and 6 methods
        records = []
        samples = [f"sample_{i}" for i in range(6)]
        methods = ["otsu", "yen", "triangle", "li", "adaptive", "gmm"]

        rng = np.random.RandomState(42)
        for s in samples:
            for m in methods:
                base_iou = 0.5 if m != "adaptive" else 0.65
                records.append({
                    "Sample": s,
                    "Method": m,
                    "Threshold_T": 100.0,
                    "IoU": float(np.clip(base_iou + rng.normal(0, 0.05), 0, 1)),
                    "Dice": float(np.clip(base_iou + 0.15 + rng.normal(0, 0.05), 0, 1)),
                    "Precision": 0.6,
                    "Recall": 0.8,
                    "Boundary_F1": float(np.clip(base_iou + 0.2 + rng.normal(0, 0.05), 0, 1)),
                    "Runtime_ms": 50.0,
                })

        df_mock = pd.DataFrame(records)
        df_stats = compute_detailed_statistical_analysis(df_mock)

        assert len(df_stats) == 6
        assert "IoU_mean" in df_stats.columns
        assert "IoU_ci95_low" in df_stats.columns
        assert "IoU_ci95_high" in df_stats.columns
        assert "IoU_paired_ttest_p" in df_stats.columns
        assert "IoU_wilcoxon_p" in df_stats.columns

        otsu_row = df_stats[df_stats["Method"] == "otsu"].iloc[0]
        assert otsu_row["IoU_paired_ttest_p"] == 1.0
