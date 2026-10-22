"""
Tests for src/contour_detection.py and src/contour_refinement.py
"""

import numpy as np
import pytest
from src.contour_detection import (
    global_threshold,
    skeletonize_binary,
    find_endpoints,
    link_endpoints,
    prune_spurs,
    label_regions,
    detect_contours,
)
from src.contour_refinement import (
    compute_area_stats,
    find_oversized_regions,
    filter_small_regions,
    refine_contours,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rectangle_image(h=128, w=128, rect_val=40, bg_val=180):
    """Create a test image with a dark rectangle on a bright background.

    The bright background simulates tear ridges; the dark interior
    simulates a dimple region.
    """
    img = np.full((h, w), bg_val, dtype=np.uint8)
    img[30:90, 30:90] = rect_val
    return img


def _make_ring_skeleton(h=64, w=64, cy=32, cx=32, r=15):
    """Create a skeleton image containing a single closed ring."""
    skel = np.zeros((h, w), dtype=np.uint8)
    for angle in np.linspace(0, 2 * np.pi, 500):
        y = int(cy + r * np.sin(angle))
        x = int(cx + r * np.cos(angle))
        if 0 <= y < h and 0 <= x < w:
            skel[y, x] = 255
    return skel


# ---------------------------------------------------------------------------
# global_threshold
# ---------------------------------------------------------------------------

class TestGlobalThreshold:
    def test_output_binary(self):
        """Output should only contain 0 and 255."""
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        binary = global_threshold(gray)
        unique = set(np.unique(binary))
        assert unique.issubset({0, 255})

    def test_shape_preserved(self):
        gray = np.random.randint(0, 256, (80, 120), dtype=np.uint8)
        binary = global_threshold(gray)
        assert binary.shape == gray.shape

    def test_all_bright(self):
        """All-white image should be all 255 (or all 0 — single-class)."""
        gray = np.full((50, 50), 200, dtype=np.uint8)
        binary = global_threshold(gray)
        # Uniform image → threshold can't separate, any result is acceptable
        assert binary.dtype == np.uint8


# ---------------------------------------------------------------------------
# skeletonize_binary
# ---------------------------------------------------------------------------

class TestSkeletonize:
    def test_output_binary(self):
        binary = np.zeros((64, 64), dtype=np.uint8)
        binary[10:50, 25:35] = 255  # thick vertical bar
        skel = skeletonize_binary(binary)
        assert set(np.unique(skel)).issubset({0, 255})

    def test_thinner_than_input(self):
        """Skeleton should have fewer foreground pixels than the thick input."""
        binary = np.zeros((64, 64), dtype=np.uint8)
        binary[10:50, 20:40] = 255
        skel = skeletonize_binary(binary)
        assert np.sum(skel > 0) < np.sum(binary > 0)


# ---------------------------------------------------------------------------
# find_endpoints
# ---------------------------------------------------------------------------

class TestFindEndpoints:
    def test_line_has_two_endpoints(self):
        """A straight line segment should have exactly 2 endpoints."""
        skel = np.zeros((64, 64), dtype=np.uint8)
        skel[32, 10:50] = 255  # horizontal line
        eps = find_endpoints(skel)
        assert len(eps) == 2

    def test_ring_has_no_endpoints(self):
        """A closed ring should have 0 endpoints."""
        skel = _make_ring_skeleton()
        eps = find_endpoints(skel)
        assert len(eps) == 0


# ---------------------------------------------------------------------------
# link_endpoints
# ---------------------------------------------------------------------------

class TestLinkEndpoints:
    def test_links_close_gap(self):
        """Two line segments with a small gap should be linked."""
        skel = np.zeros((64, 64), dtype=np.uint8)
        skel[32, 10:28] = 255
        skel[32, 33:50] = 255
        gap = 33 - 28  # = 5

        linked = link_endpoints(skel, max_gap=10)
        # After linking, the gap row should be filled
        assert np.sum(linked[32, 28:33] > 0) > 0

    def test_does_not_link_distant(self):
        """Endpoints farther than max_gap should not be linked."""
        skel = np.zeros((100, 100), dtype=np.uint8)
        skel[50, 10:20] = 255
        skel[50, 80:90] = 255
        linked = link_endpoints(skel, max_gap=5)
        assert np.sum(linked[50, 20:80] > 0) == 0


# ---------------------------------------------------------------------------
# prune_spurs
# ---------------------------------------------------------------------------

class TestPruneSpurs:
    def test_removes_short_spur(self):
        """A short dangling branch should be removed."""
        skel = _make_ring_skeleton()
        # Add a 4-pixel spur
        cy, cx, r = 32, 32, 15
        top_y = cy - r
        for dy in range(1, 5):
            if top_y - dy >= 0:
                skel[top_y - dy, cx] = 255

        pruned = prune_spurs(skel, max_spur_length=6)
        # Spur pixels should be gone
        for dy in range(1, 5):
            if top_y - dy >= 0:
                assert pruned[top_y - dy, cx] == 0


# ---------------------------------------------------------------------------
# label_regions
# ---------------------------------------------------------------------------

class TestLabelRegions:
    def test_ring_makes_two_regions(self):
        """A closed ring should create exactly 2 regions (inside + outside)."""
        skel = _make_ring_skeleton()
        label_map, num = label_regions(skel)
        assert num == 2

    def test_boundary_is_zero(self):
        """Skeleton pixels should have label 0."""
        skel = _make_ring_skeleton()
        label_map, _ = label_regions(skel)
        assert np.all(label_map[skel > 0] == 0)


# ---------------------------------------------------------------------------
# detect_contours (integration)
# ---------------------------------------------------------------------------

class TestDetectContours:
    def test_returns_expected_keys(self):
        gray = _make_rectangle_image()
        result = detect_contours(gray)
        for key in ("binary", "skeleton_raw", "skeleton",
                     "label_map", "num_regions"):
            assert key in result

    def test_finds_regions(self):
        """Rectangle image should produce at least 2 regions."""
        gray = _make_rectangle_image()
        result = detect_contours(gray)
        assert result["num_regions"] >= 2

    def test_skeleton_is_binary(self):
        gray = _make_rectangle_image()
        result = detect_contours(gray)
        assert set(np.unique(result["skeleton"])).issubset({0, 255})


# ---------------------------------------------------------------------------
# compute_area_stats / find_oversized_regions
# ---------------------------------------------------------------------------

class TestAreaStats:
    def test_mean_std(self):
        """Verify mean and std computation."""
        label_map = np.zeros((10, 10), dtype=np.int32)
        label_map[0:3, 0:3] = 1    # area = 9
        label_map[5:10, 5:10] = 2  # area = 25
        stats = compute_area_stats(label_map, 2)
        assert stats["areas"][1] == 9
        assert stats["areas"][2] == 25
        assert abs(stats["mean"] - 17.0) < 0.01

    def test_oversized_detection(self):
        """Region with area > mean+std should be flagged."""
        label_map = np.zeros((100, 100), dtype=np.int32)
        # Region 1: small
        label_map[0:5, 0:5] = 1     # 25
        # Region 2: small
        label_map[0:5, 10:15] = 2   # 25
        # Region 3: huge
        label_map[20:80, 20:80] = 3  # 3600

        stats = compute_area_stats(label_map, 3)
        oversized = find_oversized_regions(stats)
        assert 3 in oversized
        assert 1 not in oversized


# ---------------------------------------------------------------------------
# filter_small_regions
# ---------------------------------------------------------------------------

class TestFilterSmallRegions:
    def test_removes_tiny_regions(self):
        label_map = np.zeros((50, 50), dtype=np.int32)
        label_map[0:2, 0:2] = 1    # area = 4  → should be removed
        label_map[10:30, 10:30] = 2  # area = 400  → should stay
        filtered, n = filter_small_regions(label_map, 2, min_area=25)
        assert n == 1
        assert np.sum(filtered == 1) == 400  # relabeled to 1
        assert np.sum(filtered == 2) == 0    # no label 2


# ---------------------------------------------------------------------------
# refine_contours (integration)
# ---------------------------------------------------------------------------

class TestRefineContours:
    def test_returns_expected_keys(self):
        gray = _make_rectangle_image()
        contours = detect_contours(gray)
        result = refine_contours(
            gray, contours["skeleton"],
            contours["label_map"], contours["num_regions"],
            min_area=10,
        )
        for key in ("skeleton", "label_map", "num_regions",
                     "area_stats", "oversized_labels"):
            assert key in result

    def test_small_regions_filtered(self):
        """Refinement with high min_area should drop small regions."""
        gray = _make_rectangle_image()
        contours = detect_contours(gray)
        result = refine_contours(
            gray, contours["skeleton"],
            contours["label_map"], contours["num_regions"],
            min_area=50000,  # very large → drop everything
        )
        assert result["num_regions"] == 0
