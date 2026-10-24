# Technical Note & End-to-End Walkthrough: SEM Fractograph Segmentation

This document details the algorithmic foundations, pipeline architecture, engineering decisions, and experimental validation performed to build and verify the **SEM Fractograph Segmentation** pipeline following the reference methodology (*Banerjee et al., Automated Segmentation of SEM Fractographs into Void and Surface Regions*).

---

## 1. Architectural Blueprint & Repository Structure

The project is organized into modular components matching the CAMMP specification:

```text
SEM_Fractograph_Segmentation/
├── .vscode/
│   └── launch.json                    # VS Code 1-click debug & launch configurations
├── Technical_Note.md                  # Technical note, algorithm flowchart & validation walkthrough
├── README.md                          # Usage, methodology, CLI guide, & results table
├── requirements.txt                   # Production package dependencies
├── main.py                            # Interactive runner, CLI entrypoint & synthetic generator
├── scripts/
│   └── extract_paper_images.py        # PDF image extractor for experimental micrographs
├── data/
│   ├── raw/                           # Real SEM fractographs (sem_real_1..5.png)
│   └── sample/                        # Synthetic fractograph (synthetic_sem.png)
├── outputs/
│   ├── contours/                      # Initial contours and refined skeleton overlays
│   ├── figures/                       # Side-by-side comparisons & 6-panel publication plots
│   ├── masks/                         # Color-coded classified masks & binary void masks
│   └── measurements/                  # Per-region morphometrics & summary CSVs
├── src/
│   ├── __init__.py
│   ├── image_io.py                    # Multi-format I/O, grayscale normalization, banner cropping
│   ├── preprocessing.py               # 256-bin histograms, Gaussian smoothing, bilateral filtering
│   ├── contour_detection.py           # Otsu, skeletonization, spatial edge linking, spur pruning
│   ├── contour_refinement.py          # Area statistics, outlier detection, local re-thresholding
│   ├── void_detection.py              # Stage A (void-tip cutoff) & Stage B (gradient-bounded extension)
│   ├── region_classification.py       # Three-class partition (Void, Surface, Mixed) & color masks
│   ├── quantification.py              # ECD, aspect ratio, circularity, area fractions, sanitized CSVs
│   └── visualization.py               # Side-by-side plots & multi-panel diagnostic figure generation
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py          # 14 unit tests for Stage 0
    ├── test_contours.py               # 20 unit tests for Stage 1 & 2
    └── test_classification.py         # 29 unit tests for Stage 3, 4, & 5
```

---

## 2. Algorithm Flowchart & Pipeline Stages

```text
   [Input SEM Image (.tif, .png, .jpg)]
                    │
                    ▼
       ┌────────────────────────┐
       │ Stage 0: Preprocessing │
       │  - 8-bit grayscale     │
       │  - Banner crop         │
       │  - Bilateral filter    │
       │  - 256-bin histogram   │
       └───────────┬────────────┘
                    │
                    ▼
       ┌────────────────────────┐
       │ Stage 1: Contours      │
       │  - Otsu threshold      │
       │  - Skeletonize (1-px)  │
       │  - cKDTree edge-link   │
       │  - Crossing-num prune  │
       │  - Connected components│
       └───────────┬────────────┘
                    │
                    ▼
       ┌────────────────────────┐
       │ Stage 2: Refinement    │
       │  - Area μ_A, σ_A stats │
       │  - Flag A_i > μ_A+σ_A  │
       │  - Local Otsu per mask │
       │  - Merge sub-skeletons │
       │  - Filter min_area     │
       └───────────┬────────────┘
                    │
                    ▼
       ┌────────────────────────┐
       │ Stage 3: Void Detect   │
       │  - Stage A: I_tip peak │
       │  - Stage B: Dilation   │
       │    (Sobel grad < 15,   │
       │     I <= 1.5 * I_tip)  │
       │  - Per-region f_void   │
       └───────────┬────────────┘
                    │
                    ▼
       ┌────────────────────────┐
       │ Stage 4: Classification│
       │  - Surface: f_void = 0 │
       │  - Void: f_void >= tau │
       │  - Mixed: 0 < f < tau  │
       └───────────┬────────────┘
                    │
                    ▼
       ┌────────────────────────┐
       │ Stage 5: Quantification│
       │  - Region ECD, AR, Circ│
       │  - Area fractions AF_k │
       │  - Validate AF sum = 1 │
       └───────────┬────────────┘
                    │
                    ▼
       ┌────────────────────────┐
       │ Stage 6: Visualization │
       │  - Side-by-side plot   │
       │  - 6-panel diagnostic  │
       │  - Masks & skeletons   │
       │  - CSV export          │
       └────────────────────────┘
```

---

## 3. Mathematical & Algorithmic Formulation

### Stage 1: Contours & Edge Linking
1. **Otsu Global Thresholding:**
   - Minimizes within-class intensity variance across foreground ridges and background valleys: `σ_w²(T) = ω_0(T)·σ_0²(T) + ω_1(T)·σ_1²(T)`.
2. **Skeletonization & Linking:**
   - Zhang-Suen or Lee skeletonization reduces ridges to 1-pixel-wide centerlines.
   - Spatial endpoint linking via `scipy.spatial.cKDTree` finds all skeleton endpoints within radius `R = 10 px` and connects mutual nearest neighbors using Bresenham lines.
   - Spurious spurs with topological branch distance `< 5 px` are pruned.

### Stage 2: Contour Refinement
- Region areas `A_i` are collected, and the mean (`μ_A`) and standard deviation (`σ_A`) are calculated.
- **Outlier condition:** `A_i > μ_A + 1.0 × σ_A`
- Flagged regions undergo localized Otsu thresholding within their bounding box. Resulting sub-skeletons are merged back into the master skeleton.
- Regions with area below `min_area` (default `25 px`) are filtered out to remove speckle noise.

### Stage 3: Void Detection (Stages A & B)
1. **Stage A (Void-Tip Peak):**
   - The smoothed 256-bin histogram `H_smooth` is scanned in the dark regime `I in [5, 100]` for the first local maximum:
     - `I_tip = argmax H_smooth(I)` for `I in [5, 100]`
   - **Seed mask:** `S(x, y) = 1` if `I(x, y) ≤ I_tip` and `(x, y)` is not on the skeleton boundary.
2. **Stage B (Void Extension):**
   - Iterative dilation of seeds constrained by:
     - **Intensity ceiling:** `I(x, y) ≤ min(1.5 × I_tip, 255)`
     - **Sobel gradient ceiling:** `||∇I(x, y)|| < 15.0`
     - **Boundary constraint:** Pixel must not cross the closed skeleton border.
3. **Region Void Fraction:**
   - `f_void,i = (Void Pixels in Region i) / (Total Pixels in Region i)`

### Stage 4: Classification
- **Surface:** `f_void,i = 0`
- **Mixed:** `0 < f_void,i < τ_void` (default `τ_void = 0.50`)
- **Void:** `f_void,i ≥ τ_void`

### Stage 5: Morphometrics & Area Fractions
- **Equivalent Circular Diameter (ECD):** `ECD_i = 2 × sqrt(A_i / π)`
- **Circularity:** `C_i = (4 × π × A_i) / P_i²` (bounded to `[0, 1]`)
- **Area Fraction of class `k`:**
  - `AF_k = (Sum of A_i in Class k) / (Sum of A_j across all regions)`
  - `AF_Void + AF_Surface + AF_Mixed = 1.0000`

---

## 4. Verification & Unit Tests

The test suite consists of 63 unit tests across three modules:
- `test_preprocessing.py`: 14 tests verifying grayscale conversion, 256-bin histograms, Gaussian smoothing, and bilateral filtering.
- `test_contours.py`: 20 tests verifying Otsu thresholding, skeletonization, spatial KDTree linking, spur pruning, region area stats, and local re-thresholding.
- `test_classification.py`: 29 tests verifying void-tip peak detection, morphological extension, 3-class classification thresholds, morphometrics computation, and exact sum of area fractions.

---

## 5. Experimental Validation on Real SEM Fractographs

Using `scripts/extract_paper_images.py`, the 5 primary experimental fractographs were extracted from Banerjee et al. into `data/raw/`:

```bash
python main.py --input data/raw/ --output outputs/ --min-area 30
```

### Full Benchmark Summary

| Sample | Source | Dimensions | Regions | Void Count | Surface Count | Mixed Count | AF (Void) | AF (Surface) | AF (Mixed) | Sum AF |
|---|---|---|---|---|---|---|---|---|---|---|
| **Synthetic SEM** | Programmatic | 512 × 512 | 62 | 4 | 37 | 21 | 0.0722 | 0.4457 | 0.4820 | **1.0000** |
| **Real Sample 1** | Banerjee et al. | 1280 × 832 | 2,248 | 436 | 949 | 863 | 0.3396 | 0.0964 | 0.5640 | **1.0000** |
| **Real Sample 2** | Banerjee et al. | 1280 × 831 | 2,310 | 383 | 900 | 1,027 | 0.4358 | 0.0816 | 0.4826 | **1.0000** |
| **Real Sample 3** | Banerjee et al. | 1280 × 830 | 2,720 | 4 | 2,413 | 303 | 0.0967 | 0.4933 | 0.4100 | **1.0000** |
| **Real Sample 4** | Banerjee et al. | 1280 × 832 | 2,299 | 389 | 977 | 933 | 0.4200 | 0.1239 | 0.4561 | **1.0000** |
| **Real Sample 5** | Banerjee et al. | 1280 × 832 | 1,745 | 42 | 1,075 | 628 | 0.3559 | 0.1314 | 0.5127 | **1.0000** |

### Key Observations
1. **Refinement in Action:** Across experimental images, 16 to 48 oversized regions per image were flagged and subdivided, preventing under-segmentation in dense dimple colonies.
2. **Dynamic Cutoff:** `I_tip` adapted accurately between samples (e.g., `I_tip = 32` for high-brightness Sample 3 vs. `I_tip = 75–95` for darker Samples 2 and 4).
3. **Partition Completeness:** In every sample, area fractions sum to exactly **1.0000** (100% partitioning).

---

## 6. Generated Output Artifacts

All outputs are organized in `outputs/`:
- **Side-by-Side Comparisons:** `outputs/figures/*_side_by_side.png` (Original SEM next to color-coded classified regions with legend).
- **6-Panel Diagnostic Figures:** `outputs/figures/*_pipeline.png`
- **Contour Overlays:** `outputs/contours/*_skeleton.png`
- **Classified & Void Masks:** `outputs/masks/*_classified.png` and `outputs/masks/*_void_mask.png`
- **Tabular Reports:** Per-region metrics (`*_regions.csv`) and summary statistics (`*_summary.csv`) in `outputs/measurements/`.