# SEM Fractograph Segmentation

**Automated Segmentation of SEM Fractographs into Void and Surface Regions**

Center for Advanced Modeling of Materials and Manufacturing Processes (CAMMP)

## Overview

This pipeline automatically segments grayscale SEM fractographs of ductile materials into closed regions and classifies each region as **Void**, **Surface**, or **Mixed**. It extracts quantitative morphometric measurements and produces publication-quality diagnostic and side-by-side figures.

For a detailed breakdown of the algorithm flowchart, mathematical formulations, and experimental walkthrough, see [`Technical_Note.md`](./Technical_Note.md).

## Installation

```bash
pip install -r requirements.txt
```

---

## Running in Visual Studio Code (VS Code)

The engine can be executed from **VS Code** via two simple workflows:

### Method 1: Interactive Runner (Recommended for VS Code)
1. Open the project root in VS Code (`File > Open Folder...`).
2. Open [`main.py`](./main.py) and click the **Run Python File** (Play) button in the top-right corner (or run `python main.py` in the VS Code integrated terminal).
3. A guided interactive menu will prompt you to:
   - **Select an SEM image** from sample or real fractographs (or enter a custom file path / directory for batch processing).
   - **Adjust relevant parameters** (spatial scale, banner cropping, minimum area filter, void ratio threshold) or press Enter to accept optimized defaults.
   - Run the segmentation pipeline and automatically export all masks, CSV measurements, diagnostic figures, and side-by-side comparison plots.

### Method 2: 1-Click Launch Configurations (`.vscode/launch.json`)
Open the **Run & Debug** tab in VS Code (`Ctrl+Shift+D` / `Cmd+Shift+D`) and select any pre-configured profile:
- **`CAMMP-SFSE: Interactive Mode (Recommended)`** — Launches the guided interactive console.
- **`CAMMP-SFSE: Process Sample Image`** — Processes `data/sample/synthetic_sem.png` with default parameters.
- **`CAMMP-SFSE: Batch Process Real SEM Micrographs`** — Processes all experimental images in `data/raw/`.
- **`CAMMP-SFSE: Generate Synthetic SEM Sample`** — Generates a synthetic test fractograph.

---

## Command-Line Interface (CLI) Usage

```bash
# Process a single SEM image
python main.py --input data/raw/sample_sem.tif --output outputs/ --scale 0.15

# Process all images in a folder with banner crop
python main.py --input data/raw/ --output outputs/ --crop-banner 40 --min-area 30

# Generate a synthetic test fractograph
python main.py --generate-sample
```

### CLI Arguments & Default Parameters

| Argument | Short | Description | Default |
|---|---|---|---|
| `--input` | `-i` | Path to a single SEM image or directory of images | *Interactive prompt if omitted* |
| `--output` | `-o` | Output directory path | `outputs/` |
| `--scale` | `-s` | Spatial calibration (µm/px) | `1.0` |
| `--crop-banner` | | Pixels to trim from image bottom (SEM info bar) | `0` |
| `--min-area` | | Minimum region area (px) to filter noise | `25` |
| `--void-ratio` | | Void fraction threshold for Void vs Mixed classification | `0.50` |
| `--interactive` | | Force interactive guided mode | `false` |
| `--generate-sample` | | Generate synthetic SEM test image in `data/sample/` | `false` |

### Internal Pipeline Hyperparameters

| Stage | Parameter | Default Value | Purpose |
|---|---|---|---|
| **Stage 0: Preprocessing** | Histogram smoothing (`sigma`) | `2.0` | 1D Gaussian smoothing of 256-bin intensity histogram |
| **Stage 0: Preprocessing** | Bilateral filter (`d`, `sigmaColor`, `sigmaSpace`) | `5, 25, 25` | Denoises valley floors while preserving tear-ridge gradients |
| **Stage 1: Contours** | Endpoint linking radius (`max_dist`) | `10 px` | Maximum gap bridged between skeleton endpoints via `cKDTree` |
| **Stage 2: Refinement** | Outlier area threshold | `μ_A + 1.0 × σ_A` | Flags abnormally large/under-segmented regions for local Otsu re-thresholding |
| **Stage 3: Void Detection** | Dark regime search window | `I in [5, 100]` | Intensity range scanned for first local peak (`I_tip`) |
| **Stage 3: Void Extension** | Extension cutoff (`I_ext`) | `min(1.5 × I_tip, 255)` | Upper intensity bound for Stage B dilation |
| **Stage 3: Void Extension** | Sobel gradient ceiling | `15.0` | Prevents void extension from bleeding up steep dimple walls |

---

## Exported Outputs

For every processed image (`<name>`), the application automatically exports:

```text
outputs/
├── contours/
│   ├── <name>_initial_contours.png   # Stage 1 initial skeleton overlaid on grayscale image
│   └── <name>_skeleton.png           # Stage 2 refined skeleton overlaid on grayscale image
├── figures/
│   ├── <name>_side_by_side.png       # Direct side-by-side view: Original SEM vs Classified Mask with legend
│   └── <name>_pipeline.png           # 6-panel publication diagnostic (Original, Skel, Refined, Tips, Mask, ECD)
├── masks/
│   ├── <name>_void_mask.png          # Monochrome binary void mask (0 or 255)
│   └── <name>_classified.png         # 3-color RGB classification mask (Void=Red, Mixed=Yellow, Surface=Blue)
└── measurements/
    ├── <name>_regions.csv            # Per-region morphometrics (ECD, Aspect Ratio, Circularity, Void Fraction, Label)
    └── <name>_summary.csv            # Microstructure summary (Area Fractions AF_Void, AF_Surface, AF_Mixed summing to 1.0)
```

---

## Project Structure

```text
SEM_Fractograph_Segmentation/
├── .vscode/
│   └── launch.json              # VS Code 1-click debug & run configurations
├── data/
│   ├── raw/                     # Input SEM fractograph images (.tif, .png, .jpg)
│   └── sample/                  # Cropped or synthetic test images for quick validation
├── outputs/
│   ├── contours/                # Initial contours and refined skeleton overlays
│   ├── masks/                   # Binary void masks and color-coded classified region masks
│   ├── figures/                 # Side-by-side views & diagnostic multi-panel plots
│   └── measurements/            # CSV files (region-level metrics & summary stats)
├── scripts/
│   └── extract_paper_images.py  # PDF image extractor for experimental micrographs
├── src/
│   ├── __init__.py
│   ├── image_io.py              # Image loading, grayscale conversion, metadata/scale handling
│   ├── preprocessing.py         # Intensity histograms, gentle denoising, contrast stretching
│   ├── contour_detection.py     # Global thresholding, skeletonization, edge linking, pruning
│   ├── contour_refinement.py    # Area statistics, outlier detection, local re-thresholding
│   ├── void_detection.py        # Stage A (void-tip peak detection) & Stage B (void extension)
│   ├── region_classification.py # 3-class assignment (Surface, Void, Mixed)
│   ├── quantification.py        # Morphometric metrics (ECD, aspect ratio, circularity, area fraction)
│   └── visualization.py         # Side-by-side figures & multi-panel diagnostic plots
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_contours.py
│   └── test_classification.py
├── main.py                      # Application entry point (Interactive & CLI)
├── requirements.txt             # Python package dependencies
├── Technical_Note.md            # Technical note, algorithm flowchart & validation walkthrough
└── README.md                    # Setup instructions, VS Code guide, and methodology notes
```

---

## Testing

```bash
pytest tests/ -v
```

All 63 unit tests across `test_preprocessing.py`, `test_contours.py`, and `test_classification.py` pass with zero warnings.

---

## Experimental Validation on Real Fractographs

Extract the experimental SEM fractographs from Banerjee et al. (`hole_extract.pdf`) and process the entire dataset:

```bash
# Extract high-resolution micrographs from reference paper
python scripts/extract_paper_images.py

# Run batch processing on experimental SEM images
python main.py --input data/raw/ --output outputs/ --min-area 30
```

### Quantitative Segmentation Results

| Image / Sample | Image Dimensions | Total Regions | Void Count | Surface Count | Mixed Count | AF (Void) | AF (Surface) | AF (Mixed) | Sum AF |
|---|---|---|---|---|---|---|---|---|---|
| **Synthetic SEM** | 512 × 512 px | 62 | 4 | 37 | 21 | 0.0722 | 0.4457 | 0.4820 | **1.0000** |
| **Real SEM Sample 1** | 1280 × 832 px | 2,248 | 436 | 949 | 863 | 0.3396 | 0.0964 | 0.5640 | **1.0000** |
| **Real SEM Sample 2** | 1280 × 831 px | 2,310 | 383 | 900 | 1,027 | 0.4358 | 0.0816 | 0.4826 | **1.0000** |
| **Real SEM Sample 3** | 1280 × 830 px | 2,720 | 4 | 2,413 | 303 | 0.0967 | 0.4933 | 0.4100 | **1.0000** |
| **Real SEM Sample 4** | 1280 × 832 px | 2,299 | 389 | 977 | 933 | 0.4200 | 0.1239 | 0.4561 | **1.0000** |
| **Real SEM Sample 5** | 1280 × 832 px | 1,745 | 42 | 1,075 | 628 | 0.3559 | 0.1314 | 0.5127 | **1.0000** |

All outputs (side-by-side figures `_side_by_side.png`, diagnostic 6-panel figures `_pipeline.png`, classified masks `_classified.png`, skeleton overlays `_skeleton.png`, and per-region/summary CSV metrics) are cleanly generated in `outputs/`.

### Validation Observations & Known Limitations

- **Effect of Stage 2 Refinement:** Across the 5 real SEM fractographs, 16 to 48 oversized regions (`A_i > μ_A + σ_A`) per image were automatically flagged and subdivided via local Otsu re-thresholding, successfully resolving coalesced dimple clusters that global thresholding missed.
- **Adaptive Void-Tip Thresholding:** The histogram peak detector dynamically adapts to global illumination shifts (e.g., `I_tip = 32` on brighter Sample 3 versus `I_tip = 75–95` on darker Samples 2 and 4).
- **Limitations & Sensitivity:**
  - **Low-Contrast Tear Ridges:** When tear ridges are heavily shadowed or blurred, skeleton gaps larger than `max_dist = 10 px` may remain unlinked, merging adjacent dimples until Stage 2 refinement catches them.
  - **Charging Artifacts & Deep Cleavage Shadows:** Extreme bright charging spots or deep topographic shadows in mixed brittle-ductile fracture can skew the dark-regime histogram tail; adjusting `--void-ratio` or pre-cropping scale banners (`--crop-banner`) mitigates this.
  - **Resolution Floor:** Sub-micron dimples smaller than `--min-area` (default `25-30 px`) are filtered out to prevent speckle noise from inflating void number density.

## Reference

Banerjee, S., et al. — *Automated Segmentation of SEM Fractographs into Void and Surface Regions*

## Author

Adelugba Joel Oluwaseyi