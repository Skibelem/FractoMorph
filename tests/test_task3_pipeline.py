"""
CAMMP-SFSE: Unit Tests for Task 3 Blind-Validation Pipeline

Tests:
1. Extraction & Automated Banner Cropping from Task 3.pptx.
2. 100% Elimination of bottom SEM metadata/scale-bar text and numbers.
3. Existence and integrity of all Task 3 output directories, masks, figures, and CSVs.
4. Strict conservation of microstructural area fractions: Sum_AF == 1.0000 across all 18 runs (3 samples x 6 models).
5. Numerical sanity and validity of accuracy metrics (IoU, Dice, Precision, Recall, Boundary F1, Runtime).
6. Model mean summary table completeness and statistical consistency.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

from scripts.extract_and_crop_task3 import detect_banner_boundary


class TestTask3BannerCropping:
    """Verify raw micrograph extraction and automated banner removal."""

    def test_raw_micrographs_exist_and_dimensions(self):
        raw_dir = REPO_ROOT / "data" / "task3_raw"
        samples = ["sample_A_0", "sample_B_60", "sample_C_90"]
        for s in samples:
            raw_path = raw_dir / f"{s}.png"
            assert raw_path.is_file(), f"Missing raw image: {raw_path}"
            img = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
            assert img is not None, f"Failed to load raw image: {raw_path}"
            assert img.shape == (1920, 2560), f"Unexpected raw dimensions for {s}: {img.shape}"

    def test_cropped_micrographs_dimensions_and_cleanliness(self):
        crop_dir = REPO_ROOT / "data" / "task3_cropped"
        samples = ["sample_A_0", "sample_B_60", "sample_C_90"]
        for s in samples:
            crop_path = crop_dir / f"{s}.png"
            assert crop_path.is_file(), f"Missing cropped image: {crop_path}"
            img = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
            assert img is not None, f"Failed to load cropped image: {crop_path}"
            assert img.shape == (1800, 2560), f"Expected cropped shape (1800, 2560), got {img.shape}"

            # Verify that the bottom 120 pixels of the original raw image contained the banner text/lines
            raw_path = REPO_ROOT / "data" / "task3_raw" / f"{s}.png"
            raw = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
            banner = raw[1800:, :]
            # The removed banner must contain bright scale bar/text pixels
            assert np.sum(banner >= 240) > 1000, f"Expected banner in removed section of {s}"

            # The bottom of the cropped image must be free of horizontal scale-bar lines
            crop_bottom = img[-30:, :]
            assert np.sum(crop_bottom >= 254) < 50, f"Detected residual scale bar artifacts in {s}"

    def test_detect_banner_boundary_algorithm(self):
        raw_dir = REPO_ROOT / "data" / "task3_raw"
        for s in ["sample_A_0", "sample_B_60", "sample_C_90"]:
            raw_path = raw_dir / f"{s}.png"
            raw = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
            detected_y = detect_banner_boundary(raw, search_start_frac=0.75, margin_px=2)
            assert detected_y == 1800, f"Expected detected crop row 1800 for {s}, got {detected_y}"

    def test_verification_figure_exists(self):
        fig_path = REPO_ROOT / "outputs" / "task3_analysis" / "figures" / "banner_removal_verification.png"
        assert fig_path.is_file(), f"Missing verification figure: {fig_path}"
        assert fig_path.stat().st_size > 100000, "Verification figure file size unexpectedly small"


class TestTask3OutputsIntegrity:
    """Verify that all Task 3 model output directories, reference masks, and figures exist."""

    @pytest.fixture
    def expected_methods(self):
        return ["otsu", "yen", "triangle", "li", "adaptive", "gmm"]

    @pytest.fixture
    def expected_samples(self):
        return ["sample_A_0", "sample_B_60", "sample_C_90"]

    def test_reference_masks_exist(self, expected_samples):
        ref_dir = REPO_ROOT / "outputs" / "task3_analysis" / "reference_masks"
        for s in expected_samples:
            bin_path = ref_dir / f"{s}_ref_binary.png"
            skel_path = ref_dir / f"{s}_ref_skeleton.png"
            assert bin_path.is_file(), f"Missing reference binary mask: {bin_path}"
            assert skel_path.is_file(), f"Missing reference skeleton: {skel_path}"

            bin_img = cv2.imread(str(bin_path), cv2.IMREAD_GRAYSCALE)
            skel_img = cv2.imread(str(skel_path), cv2.IMREAD_GRAYSCALE)
            assert bin_img.shape == (1800, 2560)
            assert skel_img.shape == (1800, 2560)
            assert np.sum(bin_img > 0) > 0
            assert np.sum(skel_img > 0) > 0

    def test_all_method_outputs_exist(self, expected_methods, expected_samples):
        base_dir = REPO_ROOT / "outputs" / "task3_analysis"
        for m in expected_methods:
            m_dir = base_dir / m
            assert m_dir.is_dir(), f"Missing method directory: {m_dir}"
            for s in expected_samples:
                bin_path = m_dir / f"{s}_binary.png"
                skel_path = m_dir / f"{s}_skeleton.png"
                class_path = m_dir / f"{s}_classified.png"
                reg_csv = m_dir / f"{s}_regions.csv"
                sum_csv = m_dir / f"{s}_summary.csv"

                assert bin_path.is_file(), f"Missing {bin_path}"
                assert skel_path.is_file(), f"Missing {skel_path}"
                assert class_path.is_file(), f"Missing {class_path}"
                assert reg_csv.is_file(), f"Missing {reg_csv}"
                assert sum_csv.is_file(), f"Missing {sum_csv}"

    def test_all_figures_exist(self, expected_samples):
        fig_dir = REPO_ROOT / "outputs" / "task3_analysis" / "figures"
        for s in expected_samples:
            comp_fig = fig_dir / f"{s}_6models_comparison.png"
            assert comp_fig.is_file(), f"Missing comparison figure: {comp_fig}"
            assert comp_fig.stat().st_size > 100000

        charts_fig = fig_dir / "task3_model_comparison_charts.png"
        assert charts_fig.is_file(), f"Missing charts figure: {charts_fig}"

        boxplots_fig = fig_dir / "task3_boxplots.png"
        assert boxplots_fig.is_file(), f"Missing boxplots figure: {boxplots_fig}"


class TestTask3QuantificationConservation:
    """Verify strict conservation of area fractions and metric validity across all 18 runs."""

    @pytest.fixture
    def df_morph(self):
        csv_path = REPO_ROOT / "outputs" / "task3_analysis" / "tables" / "task3_per_image_morphometrics.csv"
        assert csv_path.is_file(), f"Missing morphometrics table: {csv_path}"
        return pd.read_csv(csv_path)

    @pytest.fixture
    def df_acc(self):
        csv_path = REPO_ROOT / "outputs" / "task3_analysis" / "tables" / "task3_per_image_accuracy.csv"
        assert csv_path.is_file(), f"Missing accuracy table: {csv_path}"
        return pd.read_csv(csv_path)

    @pytest.fixture
    def df_summary(self):
        csv_path = REPO_ROOT / "outputs" / "task3_analysis" / "tables" / "task3_model_mean_summary.csv"
        assert csv_path.is_file(), f"Missing summary table: {csv_path}"
        return pd.read_csv(csv_path)

    def test_run_count(self, df_morph, df_acc, df_summary):
        # 3 samples x 6 methods = 18 evaluations
        assert len(df_morph) == 18, f"Expected 18 morphometrics records, found {len(df_morph)}"
        assert len(df_acc) == 18, f"Expected 18 accuracy records, found {len(df_acc)}"
        assert len(df_summary) == 6, f"Expected 6 model summary records, found {len(df_summary)}"

    def test_exact_sum_af_conservation(self, df_morph):
        """Sum_AF == 1.0000 conservation constraint across all 18 runs."""
        for idx, row in df_morph.iterrows():
            sample = row["Sample"]
            method = row["Method"]
            af_void = row["AF_Void"]
            af_surface = row["AF_Surface"]
            af_mixed = row["AF_Mixed"]
            sum_af = row["Sum_AF"]

            # Mathematical conservation
            recalc_sum = af_void + af_surface + af_mixed
            assert np.isclose(recalc_sum, 1.0, atol=1e-4), (
                f"Area fractions do not sum to 1.0 for {sample} ({method}): "
                f"AF_Void={af_void}, AF_Surface={af_surface}, AF_Mixed={af_mixed}, Sum={recalc_sum}"
            )
            assert np.isclose(sum_af, 1.0, atol=1e-4), (
                f"Sum_AF column is not 1.0 for {sample} ({method}): {sum_af}"
            )

            # Bounded within [0, 1]
            assert 0.0 <= af_void <= 1.0
            assert 0.0 <= af_surface <= 1.0
            assert 0.0 <= af_mixed <= 1.0

            # Region count partition consistency
            total_r = row["Total_Regions"]
            n_v = row["Void_Count"]
            n_s = row["Surface_Count"]
            n_m = row["Mixed_Count"]
            assert total_r == n_v + n_s + n_m, f"Region count mismatch for {sample} ({method}): {total_r} != {n_v}+{n_s}+{n_m}"

    def test_accuracy_metrics_validity(self, df_acc):
        for idx, row in df_acc.iterrows():
            sample = row["Sample"]
            method = row["Method"]
            for metric in ["IoU", "Dice", "Precision", "Recall", "Boundary_F1"]:
                val = row[metric]
                assert not np.isnan(val), f"{metric} is NaN for {sample} ({method})"
                assert not np.isinf(val), f"{metric} is Inf for {sample} ({method})"
                assert 0.0 <= val <= 1.0, f"{metric} out of bounds for {sample} ({method}): {val}"

            assert row["Runtime_ms"] > 0, f"Runtime <= 0 for {sample} ({method})"

    def test_summary_metrics_validity(self, df_summary):
        methods_in_summary = set(df_summary["Method"].str.lower())
        expected = {"otsu", "yen", "triangle", "li", "adaptive", "gmm"}
        assert methods_in_summary == expected

        for col in df_summary.columns:
            if col == "Method":
                continue
            assert not df_summary[col].isna().any(), f"Column {col} has NaN in summary table"
            assert not np.isinf(df_summary[col]).any(), f"Column {col} has Inf in summary table"
