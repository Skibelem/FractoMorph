"""
Tests for src/evaluation.py (Reference mask generation & accuracy metrics).
"""

import numpy as np
import pytest

from src.evaluation import (
    compute_segmentation_metrics,
    generate_real_consensus_reference,
    generate_reference_mask,
    generate_synthetic_ground_truth,
)


class TestSegmentationMetrics:
    def test_identical_masks(self):
        """Identical binary masks and skeletons must yield 1.0 for all metrics."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255
        skel = np.zeros((100, 100), dtype=np.uint8)
        skel[50, 30:70] = 255

        metrics = compute_segmentation_metrics(mask, mask, skel, skel)
        assert metrics["iou"] == pytest.approx(1.0)
        assert metrics["dice"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["boundary_f1"] == pytest.approx(1.0)

    def test_completely_disjoint_masks(self):
        """Disjoint masks must yield 0.0 for all metrics."""
        pred_mask = np.zeros((100, 100), dtype=np.uint8)
        pred_mask[10:30, 10:30] = 255
        ref_mask = np.zeros((100, 100), dtype=np.uint8)
        ref_mask[70:90, 70:90] = 255

        pred_skel = np.zeros((100, 100), dtype=np.uint8)
        pred_skel[20, 10:30] = 255
        ref_skel = np.zeros((100, 100), dtype=np.uint8)
        ref_skel[80, 70:90] = 255

        metrics = compute_segmentation_metrics(pred_mask, ref_mask, pred_skel, ref_skel)
        assert metrics["iou"] == pytest.approx(0.0)
        assert metrics["dice"] == pytest.approx(0.0)
        assert metrics["precision"] == pytest.approx(0.0)
        assert metrics["recall"] == pytest.approx(0.0)
        assert metrics["boundary_f1"] == pytest.approx(0.0)

    def test_shifted_skeleton_within_tolerance(self):
        """A skeleton shifted by 1 px must yield Boundary_F1 = 1.0 under 2-px tolerance."""
        ref_skel = np.zeros((100, 100), dtype=np.uint8)
        ref_skel[50, 20:80] = 255

        pred_skel = np.zeros((100, 100), dtype=np.uint8)
        pred_skel[51, 20:80] = 255  # 1-pixel shift down

        ref_mask = np.zeros((100, 100), dtype=np.uint8)
        ref_mask[48:53, 20:80] = 255
        pred_mask = np.zeros((100, 100), dtype=np.uint8)
        pred_mask[49:54, 20:80] = 255

        metrics = compute_segmentation_metrics(
            pred_mask, ref_mask, pred_skel, ref_skel, tolerance_px=2
        )
        assert metrics["boundary_f1"] == pytest.approx(1.0)
        assert 0.0 < metrics["iou"] < 1.0

    def test_empty_masks_no_division_by_zero(self):
        """All-zero masks should handle division safely and return 1.0."""
        empty = np.zeros((50, 50), dtype=np.uint8)
        metrics = compute_segmentation_metrics(empty, empty, empty, empty)
        assert metrics["iou"] == 1.0
        assert metrics["dice"] == 1.0
        assert metrics["boundary_f1"] == 1.0

    def test_no_nan_or_inf(self):
        """Metrics should never contain NaN or Inf on arbitrary binary inputs."""
        rng = np.random.RandomState(42)
        m1 = (rng.rand(64, 64) > 0.7).astype(np.uint8) * 255
        m2 = (rng.rand(64, 64) > 0.7).astype(np.uint8) * 255
        s1 = (rng.rand(64, 64) > 0.9).astype(np.uint8) * 255
        s2 = (rng.rand(64, 64) > 0.9).astype(np.uint8) * 255

        metrics = compute_segmentation_metrics(m1, m2, s1, s2)
        for k, v in metrics.items():
            assert not np.isnan(v), f"{k} is NaN"
            assert not np.isinf(v), f"{k} is Inf"
            assert 0.0 <= v <= 1.0, f"{k} = {v} out of [0, 1] range"


class TestReferenceMaskGeneration:
    def test_synthetic_ground_truth(self):
        ref_binary, ref_skel = generate_synthetic_ground_truth(height=256, width=256)
        assert ref_binary.shape == (256, 256)
        assert ref_skel.shape == (256, 256)
        assert ref_binary.dtype == np.uint8
        assert ref_skel.dtype == np.uint8
        assert set(np.unique(ref_binary)).issubset({0, 255})
        assert set(np.unique(ref_skel)).issubset({0, 255})
        assert np.sum(ref_binary > 0) > 0
        assert np.sum(ref_skel > 0) > 0

    def test_real_consensus_reference(self):
        # Create a synthetic image with ridge pattern to test consensus detector
        img = np.full((128, 128), 100, dtype=np.uint8)
        img[30:35, :] = 220
        img[:, 50:55] = 220

        ref_bin, ref_skel = generate_real_consensus_reference(img, consensus_votes=3)
        assert ref_bin.shape == (128, 128)
        assert ref_skel.shape == (128, 128)
        assert np.sum(ref_bin > 0) > 0

    def test_generate_reference_mask_router(self, tmp_path):
        dummy = np.zeros((100, 100), dtype=np.uint8)
        out_dir = str(tmp_path / "ref_masks")

        # Test synthetic routing
        b1, s1 = generate_reference_mask(dummy, "synthetic_sem", output_dir=out_dir)
        assert b1.shape == (100, 100)
        assert (tmp_path / "ref_masks" / "synthetic_sem_ref_binary.png").exists()
        assert (tmp_path / "ref_masks" / "synthetic_sem_ref_skeleton.png").exists()

        # Test real routing
        b2, s2 = generate_reference_mask(dummy, "sem_real_1", output_dir=out_dir)
        assert b2.shape == (100, 100)
        assert (tmp_path / "ref_masks" / "sem_real_1_ref_binary.png").exists()
