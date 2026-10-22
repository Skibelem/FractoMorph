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

## Pipeline Stages

1. **Image I/O & Preprocessing** — Load, convert to grayscale, optional scale-bar crop, histogram, gentle denoising
2. **Contour Detection** — Global thresholding, skeletonization, edge linking, pruning, region labeling
3. **Contour Refinement** — Statistical outlier detection, local re-thresholding of oversized regions
4. **Void Detection** — Histogram-based void-tip detection (Stage A) and local void extension (Stage B)
5. **Classification** — Three-class assignment (Void, Surface, Mixed) per region
6. **Quantification & Visualization** — Morphometrics, CSV reports, 6-panel diagnostic figure

## Testing

```bash
pytest tests/ -v
```

## Reference

Banerjee, S., et al. — *Automated Segmentation of SEM Fractographs into Void and Surface Regions*

## Author

Adelugba Joel Oluwaseyi
