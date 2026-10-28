"""
Unit tests for CAMMP-SFSE thresholding engine (src/thresholding.py)
and its integration into contour detection and contour refinement.
"""

import numpy as np
import pytest

from src.thresholding import (
    SUPPORTED_METHODS,
    apply_threshold,
    threshold_adaptive_method,
    threshold_gmm_method,
    threshold_li_method,
    threshold_otsu_method,
    threshold_triangle_method,
    threshold_yen_method,
)
from src.contour_detection import detect_contours, global_threshold
from src.contour_refinement import local_rethreshold, refine_contours


# ---------------------------------------------------------------------------
# Test fixtures and image generators
# ---------------------------------------------------------------------------

@pytest.fixture
def bimodal_image():
    """Create a synthetic bimodal image (dark valley=40, bright ridge=210)."""
    img = np.full((120, 120), 40, dtype=np.uint8)
    # Bright ridge network
    img[20:25, :] = 210
    img[60:65, :] = 210
    img[:, 40:45] = 210
    img[:, 80:85] = 210
    # Add mild noise
    rng = np.random.RandomState(42)
    noise = rng.normal(0, 5, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def uniform_image():
    """Create a constant uniform image (all pixels = 128)."""
    return np.full((64, 64), 128, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Individual method unit tests
# ---------------------------------------------------------------------------

class TestThresholdMethods:
    @pytest.mark.parametrize("method", ["otsu", "yen", "triangle", "li", "gmm"])
    def test_global_methods_return_scalar_and_binary(self, bimodal_image, method):
        res = apply_threshold(bimodal_image, method=method)
        assert res["method"] == method
        assert isinstance(res["threshold"], float)
        assert isinstance(res["threshold_summary"], float)
        assert res["threshold"] == res["threshold_summary"]
        assert 40 <= res["threshold"] <= 210
        assert res["binary"].shape == bimodal_image.shape
        assert res["binary"].dtype == np.uint8
        assert set(np.unique(res["binary"])).issubset({0, 255})
        assert res["runtime_ms"] >= 0.0

    def test_adaptive_method_returns_surface(self, bimodal_image):
        res = apply_threshold(bimodal_image, method="adaptive", block_size=35, offset=-5.0)
        assert res["method"] == "adaptive"
        assert isinstance(res["threshold"], np.ndarray)
        assert res["threshold"].shape == bimodal_image.shape
        assert isinstance(res["threshold_summary"], float)
        assert res["binary"].shape == bimodal_image.shape
        assert res["binary"].dtype == np.uint8
        assert set(np.unique(res["binary"])).issubset({0, 255})
        assert res["runtime_ms"] >= 0.0

    def test_unsupported_method_raises_value_error(self, bimodal_image):
        with pytest.raises(ValueError, match="Unsupported thresholding method"):
            apply_threshold(bimodal_image, method="nonexistent_method")

    def test_case_insensitivity_and_whitespace(self, bimodal_image):
        res = apply_threshold(bimodal_image, method="  OTSU  ")
        assert res["method"] == "otsu"
        assert set(np.unique(res["binary"])).issubset({0, 255})


# ---------------------------------------------------------------------------
# Uniform / constant image edge case tests
# ---------------------------------------------------------------------------

class TestUniformEdgeCases:
    @pytest.mark.parametrize("method", SUPPORTED_METHODS)
    def test_uniform_image_safe_fallback(self, uniform_image, method):
        """All methods must handle uniform/constant images safely without throwing."""
        res = apply_threshold(uniform_image, method=method)
        assert res["binary"].shape == uniform_image.shape
        assert res["binary"].dtype == np.uint8
        assert isinstance(res["threshold_summary"], float)
        assert res["threshold_summary"] == pytest.approx(128.0, rel=1e-3)
        # On uniform 128 image, img > 128 is everywhere False -> binary all 0
        assert np.all(res["binary"] == 0)

    @pytest.mark.parametrize("method", SUPPORTED_METHODS)
    def test_all_black_image(self, method):
        black = np.zeros((50, 50), dtype=np.uint8)
        res = apply_threshold(black, method=method)
        assert res["binary"].shape == (50, 50)
        assert np.all(res["binary"] == 0)

    @pytest.mark.parametrize("method", SUPPORTED_METHODS)
    def test_all_white_image(self, method):
        white = np.full((50, 50), 255, dtype=np.uint8)
        res = apply_threshold(white, method=method)
        assert res["binary"].shape == (50, 50)
        assert np.all(res["binary"] == 0)


# ---------------------------------------------------------------------------
# Parameter override tests
# ---------------------------------------------------------------------------

class TestParameterOverrides:
    def test_adaptive_even_block_size_enforced_odd(self, bimodal_image):
        # Even block_size 36 should be converted to odd
        res = apply_threshold(bimodal_image, method="adaptive", block_size=36)
        assert res["binary"].shape == bimodal_image.shape
        assert set(np.unique(res["binary"])).issubset({0, 255})

    def test_adaptive_small_block_size_enforced_min(self, bimodal_image):
        # block_size < 3 enforced to 3
        res = apply_threshold(bimodal_image, method="adaptive", block_size=1)
        assert res["binary"].shape == bimodal_image.shape

    def test_adaptive_block_size_capped_to_image_dim(self):
        tiny = np.random.randint(50, 200, (15, 20), dtype=np.uint8)
        # block_size 35 exceeds 15, should be capped to 15 without error
        res = apply_threshold(tiny, method="adaptive", block_size=35)
        assert res["binary"].shape == (15, 20)

    def test_gmm_three_components(self, bimodal_image):
        # Add a third intensity level to simulate 3 distinct modes
        trimodal = bimodal_image.copy()
        trimodal[80:110, 80:110] = 130
        res = apply_threshold(trimodal, method="gmm", n_components=3, random_state=42)
        assert res["method"] == "gmm"
        assert isinstance(res["threshold"], float)
        assert 40 <= res["threshold"] <= 210
        assert set(np.unique(res["binary"])).issubset({0, 255})

    def test_gmm_reproducibility(self, bimodal_image):
        res1 = apply_threshold(bimodal_image, method="gmm", random_state=42)
        res2 = apply_threshold(bimodal_image, method="gmm", random_state=42)
        assert res1["threshold"] == res2["threshold"]
        assert np.array_equal(res1["binary"], res2["binary"])


# ---------------------------------------------------------------------------
# Integration tests with contour detection & refinement
# ---------------------------------------------------------------------------

class TestThresholdIntegration:
    @pytest.mark.parametrize("method", SUPPORTED_METHODS)
    def test_global_threshold_wrapper(self, bimodal_image, method):
        binary = global_threshold(bimodal_image, method=method)
        assert binary.shape == bimodal_image.shape
        assert binary.dtype == np.uint8
        assert set(np.unique(binary)).issubset({0, 255})

    @pytest.mark.parametrize("method", SUPPORTED_METHODS)
    def test_detect_contours_with_threshold_method(self, bimodal_image, method):
        res = detect_contours(bimodal_image, threshold_method=method)
        assert "binary" in res
        assert "skeleton" in res
        assert "label_map" in res
        assert "num_regions" in res
        assert res["num_regions"] >= 1
        assert set(np.unique(res["skeleton"])).issubset({0, 255})

    @pytest.mark.parametrize("method", SUPPORTED_METHODS)
    def test_refine_contours_with_threshold_method(self, bimodal_image, method):
        initial = detect_contours(bimodal_image, threshold_method=method)
        refined = refine_contours(
            bimodal_image,
            initial["skeleton"],
            initial["label_map"],
            initial["num_regions"],
            min_area=10,
            threshold_method=method,
        )
        assert "skeleton" in refined
        assert "label_map" in refined
        assert "num_regions" in refined
        assert "area_stats" in refined
        assert set(np.unique(refined["skeleton"])).issubset({0, 255})

    def test_backward_compatibility_defaults(self, bimodal_image):
        """Omitting threshold_method defaults to 'otsu' matching legacy behavior."""
        legacy_binary = global_threshold(bimodal_image)
        otsu_binary = global_threshold(bimodal_image, method="otsu")
        assert np.array_equal(legacy_binary, otsu_binary)

        legacy_contours = detect_contours(bimodal_image)
        otsu_contours = detect_contours(bimodal_image, threshold_method="otsu")
        assert legacy_contours["num_regions"] == otsu_contours["num_regions"]
        assert np.array_equal(legacy_contours["skeleton"], otsu_contours["skeleton"])
