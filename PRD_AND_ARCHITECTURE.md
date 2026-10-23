# Product Requirements Document (PRD) & System Architecture

**Project Title:** Automated Segmentation of SEM Fractographs into Void and Surface Regions  
**System Code:** CAMMP-SFSE (SEM Fractograph Segmentation Engine)  
**Target Organization:** Center for Advanced Modeling of Materials and Manufacturing Processes (CAMMP)  
**Reference Methodology:** Banerjee, S., et al. — *Automated Segmentation of SEM Fractographs into Void and Surface Regions*  
**Author:** Adelugba Joel Oluwaseyi  

---

## 1. Project Overview & Goal

Scanning Electron Microscope (SEM) fractographs of ductile materials contain complex microstructural features—primarily dimples (voids), tear ridges, and flat or sloped surface regions. Manual measurement of these features is slow and subjective. 

The goal of this project is to build a clean, modular Python image-processing pipeline that automatically:
1. Detects closed region boundaries (contours) in grayscale SEM fractographs.
2. Refines oversized or under-segmented regions using local statistical re-thresholding.
3. Detects deep and shallow void pixels using histogram-based void-tip detection and local intensity extension.
4. Classifies every segmented region into **Void**, **Surface**, or **Mixed**.
5. Calculates quantitative fractographic metrics and exports publication-ready visualizations and CSV reports.

---

## 2. Project Directory Structure

This structure follows the exact layout recommended in the CAMMP screening task specification:

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
│   ├── contours/                # Contour overlays (initial and refined skeletons)
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

---

## 3. Core Pipeline Stages & Functional Requirements

### Stage 0: Image I/O & Preprocessing (`src/image_io.py`, `src/preprocessing.py`)
* **Input Handling:** Load `.tif`, `.png`, `.jpg`, or `.bmp` images safely. Convert multi-channel images to 8-bit grayscale (`uint8`, range `0-255`).
* **Scale Bar Cropping (Optional):** Provide a parameter to crop out bottom SEM information banners/scale bars so text does not corrupt the intensity histogram.
* **Histogram Calculation:** Compute and smooth the 256-bin grayscale intensity distribution.
* **Gradient-Safe Filtering:** Apply gentle edge-preserving smoothing (e.g., median or bilateral filter) only when needed, ensuring intensity gradients around dimple rims are preserved.

### Stage 1: Initial Region Contour Detection (`src/contour_detection.py`)
* **Global Thresholding:** Apply global thresholding (e.g., Otsu's method or iterative intensity thresholding) to separate bright tear ridges from darker valleys.
* **Morphological Skeletonization:** Thin the binary ridge boundaries to single-pixel-wide skeletons using `skimage.morphology.skeletonize`.
* **Edge Linking:** Identify skeleton endpoints (pixels with only 1 neighbor in an 8-connected neighborhood) and link nearby endpoints across small gaps to form closed contours.
* **Pruning & Region Labeling:** Remove dangling open branches (spurs) that do not enclose a region. Label all closed connected regions using 4- or 8-connectivity (`scipy.ndimage.label` or `skimage.measure.label`).

### Stage 2: Region Contour Refinement (`src/contour_refinement.py`)
* **Area Distribution Statistics:** Compute the area (A_i) of all closed regions from Stage 1, along with the global mean area (mu_A) and standard deviation (sigma_A).
* **Candidate Selection:** Flag abnormally large regions where A_i exceeds mu_A + sigma_A.
* **Local Re-Thresholding:** For each flagged region, extract its local grayscale intensity distribution, compute a local threshold, and extract sub-contours inside that region's mask.
* **Contour Merging & Cleanup:** Merge the newly discovered internal boundaries into the master skeleton, re-run pruning, and prevent over-segmentation (discarding sub-regions smaller than a minimum noise floor, e.g., `min_area_px`).

### Stage 3: Void Detection & Extension (`src/void_detection.py`)
* **Stage A - Void-Tip Detection:**
  * Analyze the smoothed grayscale intensity histogram in the low-intensity (dark) regime.
  * Locate the first local maximum (or inflection point in the dark tail) corresponding to deep void bottoms.
  * Define the void-tip cutoff intensity (I_tip) and mark all pixels with intensity at or below I_tip as initial **void-tip seeds**.
* **Stage B - Void-Tip Extension:**
  * Iteratively grow void-tip seed regions outward into neighboring pixels inside the same closed contour region if their local intensity gradient is gentle and remains below the upper void cutoff (I_ext).
  * Distinguish between **deep voids** (originating from dark void-tip seeds) and **shallow voids** (local intensity depressions that do not reach I_tip).

### Stage 4: Region Classification (`src/region_classification.py`)
For each refined closed region R_i, compute the fraction of pixels covered by extended voids (f_void) and assign one of three classes:
1. **Surface Region:** Contains zero void pixels (f_void = 0).
2. **Void Region:** Predominantly composed of deep extended void pixels (f_void >= tau_void, where tau_void is the void dominance threshold, e.g., 0.50-0.70).
3. **Mixed Region:** Surface regions containing isolated, shallow, or partial voids (0 < f_void < tau_void).

### Stage 5: Quantitative Fractography & Visualization (`src/quantification.py`, `src/visualization.py`)
* **Global Metrics:**
  * Total Void Area Fraction (AF_void)
  * Surface Area Fraction (AF_surface) and Mixed Area Fraction (AF_mixed)
  * Void Number Density (voids per unit area)
* **Per-Void Morphometric Metrics:**
  * **Area (A):** Pixel count (and square micrometers if spatial scale `um_per_px` is provided).
  * **Perimeter (P):** Contour boundary length.
  * **Equivalent Circular Diameter (ECD):** sqrt(4 * A / pi)
  * **Aspect Ratio (AR):** Ratio of major axis length to minor axis length from fitted ellipse.
  * **Circularity (C):** (4 * pi * A) / (P^2)
* **Outputs:**
  * `outputs/measurements/<image_name>_regions.csv`: Per-region measurements and class labels.
  * `outputs/measurements/<image_name>_summary.csv`: Global summary statistics.
  * `outputs/figures/<image_name>_pipeline.png`: 6-panel figure showing (1) Original SEM, (2) Initial Skeleton, (3) Refined Skeleton, (4) Void Tips & Extension, (5) Classified Regions (Void = Red, Mixed = Yellow/Green, Surface = Blue), and (6) ECD Size Distribution Histogram.
  * `outputs/contours/<image_name>_skeleton.png`: Skeleton contour overlays.
  * `outputs/masks/<image_name>_classified.png` & `_void_mask.png`: Binary and classified masks.

---

## 4. Technical Stack & Dependencies (`requirements.txt`)

```text
numpy>=1.24.0
scipy>=1.10.0
scikit-image>=0.21.0
opencv-python>=4.8.0
matplotlib>=3.7.0
pandas>=2.0.0
pytest>=7.4.0
```

---

## 5. CLI Usage Specification (`main.py`)

The project must be runnable from the terminal with a single command:

```bash
# Process a single SEM image
python main.py --input data/raw/sample_sem.tif --output outputs/ --scale 0.15

# Process all images in a folder
python main.py --input data/raw/ --output outputs/ --crop-banner 40
```

### CLI Arguments:
* `--input` (`-i`): Path to a single SEM image or a directory of images.
* `--output` (`-o`): Output directory path (defaults to `outputs/`).
* `--scale` (`-s`): Spatial calibration in micrometers per pixel (`um/px`, default `1.0` for pixel units).
* `--crop-banner`: Number of pixels to trim from the bottom of the image to remove the SEM info bar (default `0`).
* `--min-area`: Minimum region area in pixels to filter out speckle noise (default `25`).
* `--void-ratio`: Void area fraction cutoff inside a contour to classify a region as `Void` vs `Mixed` (default `0.50`).

---

## 6. Acceptance Criteria & Testing Checklist

- [ ] **Single-Pixel Contours:** All extracted region boundaries are strictly 1-pixel wide and form closed loops without dangling spurs.
- [ ] **Refinement Improvement:** Regions with area greater than mu_A + sigma_A are locally re-evaluated and subdivided when internal ridges exist.
- [ ] **Three-Class Partition:** Every pixel (excluding 1-px boundaries) is assigned to `Void`, `Surface`, or `Mixed`.
- [ ] **Valid Metrics:** CSV outputs contain no `NaN` or `Inf` values, and area fractions sum to `1.0` (100%).
- [ ] **Automated Unit Tests:** `pytest tests/` passes all unit tests on a synthetic test fractograph.