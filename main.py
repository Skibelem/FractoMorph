#!/usr/bin/env python3
"""
CAMMP-SFSE: SEM Fractograph Segmentation Engine

Command-line interface for automated segmentation of SEM fractographs
into Void, Surface, and Mixed regions.

Author: Adelugba Joel Oluwaseyi
"""

import argparse
import os
import sys
import numpy as np
import cv2
from pathlib import Path


def generate_synthetic_sem(output_path: str = "data/sample/synthetic_sem.png",
                           width: int = 512, height: int = 512,
                           seed: int = 42) -> str:
    """Generate a realistic synthetic grayscale SEM fractograph.

    Creates an image containing:
    - Dark deep dimples (voids) of varying sizes
    - Bright tear ridges forming a network
    - Large multi-dimple regions that need refinement
    - Flat surface patches (medium intensity)

    Parameters
    ----------
    output_path : str
        Path to save the synthetic image.
    width, height : int
        Image dimensions in pixels.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    str
        Path to the saved image.
    """
    rng = np.random.RandomState(seed)

    # Start with medium-gray base (surface)
    img = np.full((height, width), 140, dtype=np.uint8)

    # --- Add smooth background texture ---
    for _ in range(3):
        noise = rng.normal(0, 8, (height, width)).astype(np.float64)
        noise_smooth = cv2.GaussianBlur(noise, (31, 31), 5)
        img = np.clip(img.astype(np.float64) + noise_smooth, 0, 255).astype(np.uint8)

    # --- Create bright tear-ridge network ---
    ridge_mask = np.zeros((height, width), dtype=np.uint8)

    # Grid-like ridges with some jitter
    n_horiz = rng.randint(6, 10)
    n_vert = rng.randint(6, 10)

    for i in range(n_horiz):
        y = int(height * (i + 0.5) / n_horiz)
        pts = []
        for x in range(0, width, 20):
            jy = y + rng.randint(-15, 16)
            pts.append([x, np.clip(jy, 5, height - 5)])
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(ridge_mask, [pts], False, 255, thickness=2)

    for i in range(n_vert):
        x = int(width * (i + 0.5) / n_vert)
        pts = []
        for y in range(0, height, 20):
            jx = x + rng.randint(-15, 16)
            pts.append([np.clip(jx, 5, width - 5), y])
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(ridge_mask, [pts], False, 255, thickness=2)

    # Add some diagonal ridges
    for _ in range(4):
        x1, y1 = rng.randint(0, width), rng.randint(0, height)
        x2, y2 = rng.randint(0, width), rng.randint(0, height)
        cv2.line(ridge_mask, (x1, y1), (x2, y2), 255, 2)

    # Brighten ridges
    img[ridge_mask > 0] = np.clip(
        img[ridge_mask > 0].astype(np.int16) + rng.randint(60, 100),
        180, 255
    ).astype(np.uint8)

    # --- Add deep dark dimples (voids) ---
    n_deep_voids = rng.randint(15, 30)
    for _ in range(n_deep_voids):
        cx = rng.randint(30, width - 30)
        cy = rng.randint(30, height - 30)
        rx = rng.randint(8, 30)
        ry = rng.randint(8, 30)
        angle = rng.randint(0, 180)
        intensity = rng.randint(15, 55)  # Very dark

        void_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(void_mask, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)

        # Don't place void on top of ridges
        valid = (void_mask > 0) & (ridge_mask == 0)
        if np.sum(valid) > 20:
            # Create gradient from center (darkest) to edge
            yy, xx = np.where(valid)
            dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            max_dist = max(dist.max(), 1)
            gradient = (dist / max_dist * 50).astype(np.int16)
            img[valid] = np.clip(
                intensity + gradient,
                10, 120
            ).astype(np.uint8)

    # --- Add a large multi-dimple region (needs refinement) ---
    large_cx, large_cy = width // 3, height // 3
    large_r = 60

    for offset in [(-20, -20), (20, -15), (0, 25), (-25, 15), (20, 20)]:
        cx = large_cx + offset[0]
        cy = large_cy + offset[1]
        rx = rng.randint(12, 22)
        ry = rng.randint(12, 22)
        void_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(void_mask, (cx, cy), (rx, ry), rng.randint(0, 180),
                    0, 360, 255, -1)
        valid = (void_mask > 0) & (ridge_mask == 0)
        if np.sum(valid) > 0:
            img[valid] = np.clip(
                rng.randint(20, 50) + rng.randint(0, 30, size=np.sum(valid)),
                10, 90
            ).astype(np.uint8)

    # --- Add shallow voids (not deep enough for void-tip) ---
    n_shallow = rng.randint(8, 15)
    for _ in range(n_shallow):
        cx = rng.randint(20, width - 20)
        cy = rng.randint(20, height - 20)
        rx = rng.randint(5, 15)
        ry = rng.randint(5, 15)
        void_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(void_mask, (cx, cy), (rx, ry), rng.randint(0, 180),
                    0, 360, 255, -1)
        valid = (void_mask > 0) & (ridge_mask == 0)
        if np.sum(valid) > 0:
            img[valid] = np.clip(
                rng.randint(80, 110) + rng.randint(0, 20, size=np.sum(valid)),
                60, 130
            ).astype(np.uint8)

    # --- Add flat surface patches (high brightness, uniform) ---
    n_flat = rng.randint(3, 6)
    for _ in range(n_flat):
        cx = rng.randint(50, width - 50)
        cy = rng.randint(50, height - 50)
        rx = rng.randint(20, 45)
        ry = rng.randint(20, 45)
        flat_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(flat_mask, (cx, cy), (rx, ry), rng.randint(0, 180),
                    0, 360, 255, -1)
        valid = (flat_mask > 0) & (ridge_mask == 0)
        if np.sum(valid) > 0:
            base_intensity = rng.randint(140, 170)
            img[valid] = np.clip(
                base_intensity + rng.randint(-5, 6, size=np.sum(valid)),
                120, 190
            ).astype(np.uint8)

    # --- Final smoothing for realism ---
    img = cv2.GaussianBlur(img, (3, 3), 0.8)

    # Re-sharpen ridges slightly
    img[ridge_mask > 0] = np.clip(
        img[ridge_mask > 0].astype(np.int16) + 20,
        0, 255
    ).astype(np.uint8)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"[CAMMP-SFSE] Synthetic SEM image saved to: {output_path}")
    return output_path


def process_image(input_path: str, output_dir: str, scale: float = 1.0,
                  crop_banner_px: int = 0, min_area: int = 25,
                  void_ratio: float = 0.50):
    """Run the full CAMMP-SFSE pipeline on a single SEM image.

    Parameters
    ----------
    input_path : str
        Path to the input SEM image.
    output_dir : str
        Root output directory.
    scale : float
        Spatial calibration (um/px).
    crop_banner_px : int
        Pixels to crop from bottom.
    min_area : int
        Minimum region area in pixels.
    void_ratio : float
        Void fraction threshold for Void vs Mixed classification.
    """
    from src.image_io import load_image, to_grayscale_uint8, crop_banner, save_image
    from src.preprocessing import preprocess
    from src.contour_detection import detect_contours
    from src.contour_refinement import refine_contours
    from src.void_detection import detect_voids
    from src.region_classification import classify_all_regions, create_classification_mask
    from src.quantification import (compute_region_morphometrics,
                                     compute_summary_metrics, export_csv)
    from src.visualization import (create_pipeline_figure,
                                    save_classification_mask,
                                    save_skeleton_overlay,
                                    save_binary_mask)

    image_name = Path(input_path).stem
    contour_dir = os.path.join(output_dir, "contours")
    mask_dir = os.path.join(output_dir, "masks")
    fig_dir = os.path.join(output_dir, "figures")
    meas_dir = os.path.join(output_dir, "measurements")

    for d in [contour_dir, mask_dir, fig_dir, meas_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CAMMP-SFSE Processing: {image_name}")
    print(f"{'='*60}")

    # --- Stage 0: Load & Preprocess ---
    print("[Stage 0] Loading and preprocessing image...")
    raw_img = load_image(input_path)
    gray = to_grayscale_uint8(raw_img)

    if crop_banner_px > 0:
        gray = crop_banner(gray, crop_banner_px)
        print(f"  Cropped {crop_banner_px}px banner from bottom")

    prep = preprocess(gray)
    denoised = prep["denoised"]
    hist_smooth = prep["histogram_smooth"]
    print(f"  Image size: {gray.shape[1]}x{gray.shape[0]} px")

    # --- Stage 1: Contour Detection ---
    print("[Stage 1] Detecting region contours...")
    contours = detect_contours(denoised)
    print(f"  Initial regions: {contours['num_regions']}")

    # --- Stage 2: Contour Refinement ---
    print("[Stage 2] Refining contours...")
    refined = refine_contours(
        gray, contours["skeleton"],
        contours["label_map"], contours["num_regions"],
        min_area=min_area
    )
    print(f"  Refined regions: {refined['num_regions']}")
    print(f"  Oversized regions re-thresholded: {len(refined['oversized_labels'])}")

    # --- Stage 3: Void Detection ---
    print("[Stage 3] Detecting voids...")
    voids = detect_voids(
        denoised, hist_smooth,
        refined["label_map"], refined["num_regions"]
    )
    print(f"  Void-tip intensity (I_tip): {voids['i_tip']}")

    # --- Stage 4: Classification ---
    print("[Stage 4] Classifying regions...")
    classifications = classify_all_regions(
        voids["void_fractions"], tau_void=void_ratio
    )

    n_void = sum(1 for c in classifications.values() if c == "Void")
    n_surface = sum(1 for c in classifications.values() if c == "Surface")
    n_mixed = sum(1 for c in classifications.values() if c == "Mixed")
    print(f"  Void: {n_void}, Surface: {n_surface}, Mixed: {n_mixed}")

    class_mask = create_classification_mask(
        refined["label_map"], classifications
    )

    # --- Stage 5: Quantification ---
    print("[Stage 5] Computing metrics and generating outputs...")

    df_regions = compute_region_morphometrics(
        refined["label_map"], classifications,
        voids["void_fractions"], scale=scale
    )

    # Image area = total non-boundary pixels
    image_area_px = int(np.sum(refined["label_map"] > 0))
    df_summary = compute_summary_metrics(df_regions, image_area_px, scale=scale)

    # Export CSVs to outputs/measurements/
    regions_csv = os.path.join(meas_dir, f"{image_name}_regions.csv")
    summary_csv = os.path.join(meas_dir, f"{image_name}_summary.csv")
    export_csv(df_regions, regions_csv)
    export_csv(df_summary, summary_csv)
    print(f"  Regions CSV: {regions_csv}")
    print(f"  Summary CSV: {summary_csv}")

    # Verify no NaN/Inf
    assert not df_regions.isnull().any().any(), "NaN found in regions CSV!"
    assert not df_summary.isnull().any().any(), "NaN found in summary CSV!"

    # --- Stage 6: Visualization ---
    print("[Stage 6] Generating visualizations...")

    # Diagnostic figures to outputs/figures/
    pipeline_fig = os.path.join(fig_dir, f"{image_name}_pipeline.png")
    create_pipeline_figure(
        gray, contours["skeleton_raw"], refined["skeleton"],
        voids["void_seeds"], voids["void_extended"],
        class_mask, df_regions, pipeline_fig
    )
    print(f"  Pipeline figure: {pipeline_fig}")

    # Contours to outputs/contours/
    init_contour_path = os.path.join(contour_dir, f"{image_name}_initial_contours.png")
    save_skeleton_overlay(gray, contours["skeleton_raw"], init_contour_path)
    print(f"  Initial contours: {init_contour_path}")

    skel_path = os.path.join(contour_dir, f"{image_name}_skeleton.png")
    save_skeleton_overlay(gray, refined["skeleton"], skel_path)
    print(f"  Refined skeleton: {skel_path}")

    # Masks to outputs/masks/
    void_mask_path = os.path.join(mask_dir, f"{image_name}_void_mask.png")
    save_binary_mask(voids["void_extended"], void_mask_path)
    print(f"  Void mask: {void_mask_path}")

    mask_path = os.path.join(mask_dir, f"{image_name}_classified.png")
    save_classification_mask(class_mask, mask_path)
    print(f"  Classification mask: {mask_path}")

    # Print summary
    print(f"\n--- Summary for {image_name} ---")
    print(f"  Total regions: {len(df_regions)}")
    if len(df_summary) > 0:
        row = df_summary.iloc[0]
        print(f"  AF_Void:    {row['AF_Void']:.4f}")
        print(f"  AF_Surface: {row['AF_Surface']:.4f}")
        print(f"  AF_Mixed:   {row['AF_Mixed']:.4f}")
        af_sum = row['AF_Void'] + row['AF_Surface'] + row['AF_Mixed']
        print(f"  AF Sum:     {af_sum:.4f}")

    print(f"{'='*60}\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CAMMP-SFSE: Automated SEM Fractograph Segmentation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --generate-sample
  python main.py --input data/sample/synthetic_sem.png --output outputs/
  python main.py --input data/raw/sample_sem.tif --output outputs/ --scale 0.15
  python main.py --input data/raw/ --output outputs/ --crop-banner 40
        """,
    )

    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="Path to a single SEM image or directory of images."
    )
    parser.add_argument(
        "--output", "-o", type=str, default="outputs/",
        help="Output directory path (default: outputs/)."
    )
    parser.add_argument(
        "--scale", "-s", type=float, default=1.0,
        help="Spatial calibration in um/px (default: 1.0)."
    )
    parser.add_argument(
        "--crop-banner", type=int, default=0,
        help="Pixels to trim from image bottom to remove SEM info bar (default: 0)."
    )
    parser.add_argument(
        "--min-area", type=int, default=25,
        help="Minimum region area in pixels to filter noise (default: 25)."
    )
    parser.add_argument(
        "--void-ratio", type=float, default=0.50,
        help="Void fraction threshold for Void vs Mixed classification (default: 0.50)."
    )
    parser.add_argument(
        "--generate-sample", action="store_true",
        help="Generate a synthetic SEM test image in data/sample/."
    )

    args = parser.parse_args()

    if args.generate_sample:
        generate_synthetic_sem("data/sample/synthetic_sem.png")
        if args.input is None:
            print("\nUse --input data/sample/synthetic_sem.png to process it.")
            return

    if args.input is None:
        parser.error("--input is required (or use --generate-sample).")

    from src.image_io import collect_image_paths

    image_paths = collect_image_paths(args.input)
    print(f"[CAMMP-SFSE] Found {len(image_paths)} image(s) to process.")

    for img_path in image_paths:
        process_image(
            input_path=img_path,
            output_dir=args.output,
            scale=args.scale,
            crop_banner_px=args.crop_banner,
            min_area=args.min_area,
            void_ratio=args.void_ratio,
        )

    print("[CAMMP-SFSE] All processing complete.")


if __name__ == "__main__":
    main()
