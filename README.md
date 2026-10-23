# CAMMP-SFSE: SEM Fractograph Segmentation Engine

**Automated Segmentation of SEM Fractographs into Void and Surface Regions**

Center for Advanced Modeling of Materials and Manufacturing Processes (CAMMP)

## Overview

This pipeline automatically segments grayscale SEM fractographs of ductile materials into closed regions and classifies each region as **Void**, **Surface**, or **Mixed**. It extracts quantitative morphometric measurements and produces publication-quality diagnostic figures.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Generate a synthetic test image and run the full pipeline
python main.py --generate-sample
python main.py --input data/sample/synthetic_sem.png --output outputs/
```

## CLI Usage

```bash
# Process a single SEM image
python main.py --input data/raw/sample_sem.tif --output outputs/ --scale 0.15

# Process all images in a folder
python main.py --input data/raw/ --output outputs/ --crop-banner 40
```

### CLI Arguments

| Argument | Short | Description | Default |
|---|---|---|---|
| `--input` | `-i` | Path to a single SEM image or directory of images | *required* |
| `--output` | `-o` | Output directory path | `outputs/` |
| `--scale` | `-s` | Spatial calibration (µm/px) | `1.0` |
| `--crop-banner` | | Pixels to trim from image bottom (SEM info bar) | `0` |
| `--min-area` | | Minimum region area (px) to filter noise | `25` |
| `--void-ratio` | | Void fraction threshold for Void vs Mixed classification | `0.50` |
| `--generate-sample` | | Generate synthetic SEM test image in `data/sample/` | `false` |

```text
sem-fractograph-segmentation/
├── PRD_AND_ARCHITECTURE.md      # Master specification & architecture blueprint
├── hole_extract.pdf             # Reference research paper (Banerjee et al.)
├── README.md                    # Setup instructions, CLI usage, and methodology notes
├── requirements.txt             # Python package dependencies
├── main.py                      # Command-line interface (CLI) entry point
├── scripts/
│   └── extract_paper_images.py  # PDF image extractor for experimental micrographs
├── data/
│   ├── raw/                     # Input SEM fractograph images (.tif, .png, .jpg)
│   └── sample/                  # Cropped or synthetic test images for quick validation
├── outputs/
│   ├── contours/                # Initial contours and refined skeleton overlays
│   ├── figures/                 # Diagnostic multi-panel publication plots
│   ├── masks/                   # Binary void masks and color-coded classified region masks
│   └── measurements/            # CSV files (region-level metrics & summary stats)
├── src/
│   ├── __init__.py
│   ├── image_io.py              # Image loading, grayscale conversion, metadata/scale handling
│   ├── preprocessing.py         # Intensity histograms, gentle denoising, contrast stretching
│   ├── contour_detection.py     # Global thresholding, skeletonization, edge linking, pruning
│   ├── contour_refinement.py    # Area statistics, outlier detection, local re-thresholding
│   ├── void_detection.py        # Stage A (void-tip peak detection) & Stage B (void extension)
│   ├── region_classification.py # 3-class assignment (Surface, Void, Mixed)
│   ├── quantification.py        # Morphometric metrics (ECD, aspect ratio, circularity, area fraction)
│   └── visualization.py         # Multi-panel diagnostic plots & color-coded region maps
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py
    ├── test_contours.py
    └── test_classification.py
```

## Pipeline Stages

1. **Image I/O & Preprocessing** (`src/image_io.py`, `src/preprocessing.py`) — Load, convert to grayscale, optional scale-bar crop, histogram, gentle denoising
2. **Contour Detection** (`src/contour_detection.py`) — Global thresholding, skeletonization, cKDTree edge linking, spur pruning, region labeling
3. **Contour Refinement** (`src/contour_refinement.py`) — Statistical outlier detection, local re-thresholding of oversized regions
4. **Void Detection** (`src/void_detection.py`) — Histogram-based void-tip detection (Stage A) and local void extension (Stage B)
5. **Region Classification** (`src/region_classification.py`) — Three-class assignment (Void, Surface, Mixed) per region
6. **Quantification & Visualization** (`src/quantification.py`, `src/visualization.py`) — Morphometrics, CSV reports in `outputs/measurements/`, masks in `outputs/masks/`, contour overlays in `outputs/contours/`, and 6-panel diagnostic figures in `outputs/figures/`

## Testing

```bash
pytest tests/ -v
```

## Experimental Validation on Real Fractographs

Extract the experimental SEM fractographs from Banerjee et al. (`hole_extract.pdf`) and process the entire dataset:

```bash
# Extract high-resolution micrographs from reference paper
python scripts/extract_paper_images.py

# Run batch processing on experimental SEM images
python main.py --input data/raw/ --output outputs/ --min-area 30
```

### Quantitative Segmentation Results

| Image / Sample | Image Dimensions | Total Regions | Void Count | Surface Count | Mixed Count | AF (Void) | AF (Surface) | AF (Mixed) | $\sum$ AF |
|---|---|---|---|---|---|---|---|---|---|
| **Synthetic SEM** | 512 × 512 px | 62 | 4 | 37 | 21 | 0.0722 | 0.4457 | 0.4820 | **1.0000** |
| **Real SEM Sample 1** | 1280 × 832 px | 2,248 | 436 | 949 | 863 | 0.3396 | 0.0964 | 0.5640 | **1.0000** |
| **Real SEM Sample 2** | 1280 × 831 px | 2,310 | 383 | 900 | 1,027 | 0.4358 | 0.0816 | 0.4826 | **1.0000** |
| **Real SEM Sample 3** | 1280 × 830 px | 2,720 | 4 | 2,413 | 303 | 0.0967 | 0.4933 | 0.4100 | **1.0000** |
| **Real SEM Sample 4** | 1280 × 832 px | 2,299 | 389 | 977 | 933 | 0.4200 | 0.1239 | 0.4561 | **1.0000** |
| **Real SEM Sample 5** | 1280 × 832 px | 1,745 | 42 | 1,075 | 628 | 0.3559 | 0.1314 | 0.5127 | **1.0000** |

All outputs (diagnostic 6-panel figures `_pipeline.png`, classified masks `_classified.png`, skeleton overlays `_skeleton.png`, and per-region/summary CSV metrics) are cleanly generated in `outputs/`.

## Reference

Banerjee, S., et al. — *Automated Segmentation of SEM Fractographs into Void and Surface Regions*

## Author

Adelugba Joel Oluwaseyi
