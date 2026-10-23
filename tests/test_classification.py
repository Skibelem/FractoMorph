"""
Tests for src/region_classification.py, src/void_detection.py, and src/quantification.py
"""

import numpy as np
import pandas as pd
import pytest

from src.region_classification import (
    CLASS_SURFACE,
    CLASS_VOID,
    CLASS_MIXED,
    classify_region,
    classify_all_regions,
    create_classification_mask,
    compute_class_summary,
)
from src.void_detection import (
    find_void_tip_intensity,
    create_void_tip_seeds,
    compute_void_fractions,
)
from src.quantification import (
    compute_region_morphometrics,
    compute_summary_metrics,
)


# ---------------------------------------------------------------------------
# classify_region
# ---------------------------------------------------------------------------

class TestClassifyRegion:
    def test_surface(self):
        """f_void = 0 → Surface."""
        assert classify_region(0.0) == CLASS_SURFACE

    def test_void_at_threshold(self):
        """f_void = tau → Void."""
        assert classify_region(0.50, tau_void=0.50) == CLASS_VOID

    def test_void_above_threshold(self):
        """f_void > tau → Void."""
        assert classify_region(0.80, tau_void=0.50) == CLASS_VOID

    def test_mixed(self):
        """0 < f_void < tau → Mixed."""
        assert classify_region(0.25, tau_void=0.50) == CLASS_MIXED

    def test_custom_threshold(self):
        """Custom tau_void should be respected."""
        assert classify_region(0.60, tau_void=0.70) == CLASS_MIXED
        assert classify_region(0.70, tau_void=0.70) == CLASS_VOID

    def test_negative_treated_as_surface(self):
        """Negative void fraction → Surface."""
        assert classify_region(-0.1) == CLASS_SURFACE


# ---------------------------------------------------------------------------
# classify_all_regions
# ---------------------------------------------------------------------------

class TestClassifyAllRegions:
    def test_multiple_regions(self):
        fracs = {1: 0.0, 2: 0.60, 3: 0.30}
        result = classify_all_regions(fracs, tau_void=0.50)
        assert result[1] == CLASS_SURFACE
        assert result[2] == CLASS_VOID
        assert result[3] == CLASS_MIXED

    def test_empty_dict(self):
        result = classify_all_regions({})
        assert result == {}


# ---------------------------------------------------------------------------
# create_classification_mask
# ---------------------------------------------------------------------------

class TestCreateClassificationMask:
    def test_mask_shape_and_dtype(self):
        label_map = np.zeros((50, 50), dtype=np.int32)
        label_map[10:20, 10:20] = 1
        label_map[30:40, 30:40] = 2
        classifications = {1: CLASS_VOID, 2: CLASS_SURFACE}
        mask = create_classification_mask(label_map, classifications)
        assert mask.shape == (50, 50, 3)
        assert mask.dtype == np.uint8

    def test_void_is_red(self):
        """Void regions should be red (255, 0, 0) in RGB."""
        label_map = np.zeros((30, 30), dtype=np.int32)
        label_map[5:25, 5:25] = 1
        classifications = {1: CLASS_VOID}
        mask = create_classification_mask(label_map, classifications)
        # Check a pixel inside the void region
        assert list(mask[15, 15]) == [255, 0, 0]

    def test_surface_is_blue(self):
        """Surface regions should be blue (0, 0, 255) in RGB."""
        label_map = np.zeros((30, 30), dtype=np.int32)
        label_map[5:25, 5:25] = 1
        classifications = {1: CLASS_SURFACE}
        mask = create_classification_mask(label_map, classifications)
        assert list(mask[15, 15]) == [0, 0, 255]

    def test_boundary_is_white(self):
        """Boundary pixels (label 0) should be white."""
        label_map = np.zeros((30, 30), dtype=np.int32)
        label_map[10:20, 10:20] = 1
        classifications = {1: CLASS_VOID}
        mask = create_classification_mask(label_map, classifications)
        # (0, 0) is boundary → white
        assert list(mask[0, 0]) == [255, 255, 255]


# ---------------------------------------------------------------------------
# compute_class_summary
# ---------------------------------------------------------------------------

class TestComputeClassSummary:
    def test_fractions_sum_to_one(self):
        """Area fractions should sum to 1.0."""
        label_map = np.zeros((100, 100), dtype=np.int32)
        label_map[0:30, 0:30] = 1    # 900 px
        label_map[0:30, 40:70] = 2   # 900 px
        label_map[50:80, 50:80] = 3  # 900 px
        classifications = {
            1: CLASS_VOID,
            2: CLASS_SURFACE,
            3: CLASS_MIXED,
        }
        summary = compute_class_summary(classifications, label_map)
        total = sum(summary["fractions"].values())
        assert abs(total - 1.0) < 1e-6

    def test_counts(self):
        label_map = np.zeros((50, 50), dtype=np.int32)
        label_map[0:10, 0:10] = 1
        label_map[0:10, 20:30] = 2
        label_map[20:30, 0:10] = 3
        classifications = {
            1: CLASS_VOID,
            2: CLASS_VOID,
            3: CLASS_SURFACE,
        }
        summary = compute_class_summary(classifications, label_map)
        assert summary["counts"][CLASS_VOID] == 2
        assert summary["counts"][CLASS_SURFACE] == 1
        assert summary["counts"][CLASS_MIXED] == 0


# ---------------------------------------------------------------------------
# find_void_tip_intensity
# ---------------------------------------------------------------------------

class TestFindVoidTipIntensity:
    def test_returns_int(self):
        hist = np.zeros(256)
        hist[20] = 500   # dark peak
        hist[150] = 3000  # main peak
        from scipy.ndimage import gaussian_filter1d
        hist_smooth = gaussian_filter1d(hist, sigma=2.0)
        i_tip = find_void_tip_intensity(hist_smooth)
        assert isinstance(i_tip, int)

    def test_within_bounds(self):
        hist = np.random.rand(256) * 100
        from scipy.ndimage import gaussian_filter1d
        hist_smooth = gaussian_filter1d(hist, sigma=2.0)
        i_tip = find_void_tip_intensity(hist_smooth)
        assert 0 <= i_tip <= 255


# ---------------------------------------------------------------------------
# create_void_tip_seeds
# ---------------------------------------------------------------------------

class TestCreateVoidTipSeeds:
    def test_seeds_below_threshold(self):
        """Pixels <= i_tip should be seeds."""
        gray = np.array([[10, 50, 100],
                         [200, 30, 150]], dtype=np.uint8)
        seeds = create_void_tip_seeds(gray, i_tip=50)
        expected = np.array([[255, 255, 0],
                             [0, 255, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(seeds, expected)


# ---------------------------------------------------------------------------
# compute_void_fractions
# ---------------------------------------------------------------------------

class TestComputeVoidFractions:
    def test_full_void_region(self):
        """Region entirely covered by voids → fraction = 1.0."""
        label_map = np.zeros((20, 20), dtype=np.int32)
        label_map[5:15, 5:15] = 1
        voids = np.zeros((20, 20), dtype=np.uint8)
        voids[5:15, 5:15] = 255
        fracs = compute_void_fractions(voids, label_map, 1)
        assert abs(fracs[1] - 1.0) < 1e-6

    def test_no_void_region(self):
        """Region with no void pixels → fraction = 0.0."""
        label_map = np.zeros((20, 20), dtype=np.int32)
        label_map[5:15, 5:15] = 1
        voids = np.zeros((20, 20), dtype=np.uint8)
        fracs = compute_void_fractions(voids, label_map, 1)
        assert fracs[1] == 0.0

    def test_partial_void(self):
        """Half-void region → fraction ≈ 0.5."""
        label_map = np.zeros((20, 20), dtype=np.int32)
        label_map[0:20, 0:10] = 1  # 200 px
        voids = np.zeros((20, 20), dtype=np.uint8)
        voids[0:20, 0:5] = 255  # 100 px void
        fracs = compute_void_fractions(voids, label_map, 1)
        assert abs(fracs[1] - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# compute_region_morphometrics
# ---------------------------------------------------------------------------

class TestComputeRegionMorphometrics:
    def _make_simple_label_map(self):
        label_map = np.zeros((100, 100), dtype=np.int32)
        label_map[10:40, 10:40] = 1   # ~900 px
        label_map[50:80, 50:80] = 2   # ~900 px
        return label_map

    def test_no_nan_inf(self):
        """Output should contain no NaN or Inf."""
        label_map = self._make_simple_label_map()
        classifications = {1: CLASS_VOID, 2: CLASS_SURFACE}
        void_fractions = {1: 0.8, 2: 0.0}
        df = compute_region_morphometrics(
            label_map, classifications, void_fractions
        )
        assert not df.isnull().any().any()
        assert not np.isinf(df.select_dtypes(include=[np.number]).values).any()

    def test_expected_columns(self):
        label_map = self._make_simple_label_map()
        classifications = {1: CLASS_VOID, 2: CLASS_SURFACE}
        void_fractions = {1: 0.8, 2: 0.0}
        df = compute_region_morphometrics(
            label_map, classifications, void_fractions
        )
        for col in ("Region_ID", "Class", "Void_Fraction", "Area_px",
                     "ECD_px", "Aspect_Ratio", "Circularity"):
            assert col in df.columns

    def test_ecd_positive(self):
        """ECD should be positive for non-empty regions."""
        label_map = self._make_simple_label_map()
        classifications = {1: CLASS_VOID, 2: CLASS_SURFACE}
        void_fractions = {1: 0.8, 2: 0.0}
        df = compute_region_morphometrics(
            label_map, classifications, void_fractions
        )
        assert (df["ECD_px"] > 0).all()

    def test_circularity_bounded(self):
        """Circularity should be in [0, 1]."""
        label_map = self._make_simple_label_map()
        classifications = {1: CLASS_VOID, 2: CLASS_SURFACE}
        void_fractions = {1: 0.8, 2: 0.0}
        df = compute_region_morphometrics(
            label_map, classifications, void_fractions
        )
        assert (df["Circularity"] >= 0).all()
        assert (df["Circularity"] <= 1.0).all()


# ---------------------------------------------------------------------------
# compute_summary_metrics
# ---------------------------------------------------------------------------

class TestComputeSummaryMetrics:
    def test_area_fractions_sum_to_one(self):
        """AF_Void + AF_Surface + AF_Mixed should sum to 1.0."""
        label_map = np.zeros((100, 100), dtype=np.int32)
        label_map[0:30, 0:30] = 1
        label_map[0:30, 40:70] = 2
        label_map[50:80, 50:80] = 3
        classifications = {1: CLASS_VOID, 2: CLASS_SURFACE, 3: CLASS_MIXED}
        void_fractions = {1: 0.9, 2: 0.0, 3: 0.3}

        df_regions = compute_region_morphometrics(
            label_map, classifications, void_fractions
        )
        df_summary = compute_summary_metrics(
            df_regions, image_area_px=2700
        )
        row = df_summary.iloc[0]
        af_sum = row["AF_Void"] + row["AF_Surface"] + row["AF_Mixed"]
        assert abs(af_sum - 1.0) < 1e-4

    def test_no_nan_inf(self):
        label_map = np.zeros((50, 50), dtype=np.int32)
        label_map[5:20, 5:20] = 1
        classifications = {1: CLASS_VOID}
        void_fractions = {1: 1.0}
        df_regions = compute_region_morphometrics(
            label_map, classifications, void_fractions
        )
        df_summary = compute_summary_metrics(df_regions, image_area_px=225)
        assert not df_summary.isnull().any().any()
        assert not np.isinf(
            df_summary.select_dtypes(include=[np.number]).values
        ).any()
