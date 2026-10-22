"""
Tests for src/preprocessing.py
"""

import numpy as np
import pytest
from src.preprocessing import (
    compute_histogram,
    smooth_histogram,
    denoise_bilateral,
    denoise_median,
    preprocess,
)


class TestComputeHistogram:
    """Tests for compute_histogram()."""

    def test_histogram_shape(self):
        """Histogram should have 256 bins by default."""
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        hist = compute_histogram(gray)
        assert hist.shape == (256,)

    def test_histogram_sum(self):
        """Sum of histogram bins should equal total pixel count."""
        gray = np.random.randint(0, 256, (50, 80), dtype=np.uint8)
        hist = compute_histogram(gray)
        assert int(hist.sum()) == 50 * 80

    def test_uniform_image(self):
        """Uniform image should have all counts in one bin."""
        gray = np.full((64, 64), 128, dtype=np.uint8)
        hist = compute_histogram(gray)
        assert hist[128] == 64 * 64
        assert hist.sum() == 64 * 64

    def test_black_image(self):
        """All-black image: all counts in bin 0."""
        gray = np.zeros((32, 32), dtype=np.uint8)
        hist = compute_histogram(gray)
        assert hist[0] == 32 * 32

    def test_histogram_dtype(self):
        """Histogram should be float64."""
        gray = np.zeros((10, 10), dtype=np.uint8)
        hist = compute_histogram(gray)
        assert hist.dtype == np.float64


class TestSmoothHistogram:
    """Tests for smooth_histogram()."""

    def test_output_shape_preserved(self):
        """Smoothed histogram should have same length."""
        hist = np.random.rand(256)
        smoothed = smooth_histogram(hist, sigma=2.0)
        assert smoothed.shape == hist.shape

    def test_smoothing_reduces_peaks(self):
        """Smoothing should reduce sharp peaks."""
        hist = np.zeros(256)
        hist[128] = 1000.0
        smoothed = smooth_histogram(hist, sigma=3.0)
        # Peak should be reduced
        assert smoothed[128] < 1000.0
        # But total area approximately preserved
        assert abs(smoothed.sum() - 1000.0) < 1.0

    def test_already_smooth(self):
        """A constant histogram should remain constant after smoothing."""
        hist = np.ones(256) * 50.0
        smoothed = smooth_histogram(hist, sigma=2.0)
        np.testing.assert_allclose(smoothed, hist, atol=1.0)


class TestDenoiseBilateral:
    """Tests for denoise_bilateral()."""

    def test_output_shape(self):
        """Output should have same shape as input."""
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = denoise_bilateral(gray)
        assert result.shape == gray.shape

    def test_output_dtype(self):
        """Output should be uint8."""
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = denoise_bilateral(gray)
        assert result.dtype == np.uint8

    def test_uniform_unchanged(self):
        """Uniform image should not change after bilateral filter."""
        gray = np.full((64, 64), 100, dtype=np.uint8)
        result = denoise_bilateral(gray)
        np.testing.assert_array_equal(result, gray)


class TestDenoiseMedian:
    """Tests for denoise_median()."""

    def test_output_shape(self):
        """Output shape should match input."""
        gray = np.random.randint(0, 256, (80, 80), dtype=np.uint8)
        result = denoise_median(gray, ksize=3)
        assert result.shape == gray.shape

    def test_even_ksize_corrected(self):
        """Even ksize should be auto-corrected to odd."""
        gray = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        # Should not raise even with even ksize
        result = denoise_median(gray, ksize=4)
        assert result.shape == gray.shape


class TestPreprocess:
    """Tests for the combined preprocess() function."""

    def test_returns_dict_keys(self):
        """preprocess should return dict with expected keys."""
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = preprocess(gray)
        assert "denoised" in result
        assert "histogram" in result
        assert "histogram_smooth" in result

    def test_denoised_shape(self):
        """Denoised image should match input shape."""
        gray = np.random.randint(0, 256, (80, 80), dtype=np.uint8)
        result = preprocess(gray)
        assert result["denoised"].shape == gray.shape
        assert result["denoised"].dtype == np.uint8

    def test_histogram_length(self):
        """Histograms should have 256 entries."""
        gray = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        result = preprocess(gray)
        assert len(result["histogram"]) == 256
        assert len(result["histogram_smooth"]) == 256

    def test_median_mode(self):
        """preprocess with use_bilateral=False should work."""
        gray = np.random.randint(0, 256, (60, 60), dtype=np.uint8)
        result = preprocess(gray, use_bilateral=False)
        assert result["denoised"].shape == gray.shape
