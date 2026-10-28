# Task 1 Technical Note: Baseline Thresholding & Segmentation Architecture in FractoMorph

**Project:** FractoMorph — Automated Segmentation of SEM Fractographs  
**Organization:** Center for Advanced Modeling of Materials and Manufacturing Processes (CAMMP)  
**Author:** Adelugba Joel Oluwaseyi  

---

## 1. Purpose of This Technical Note

This document reviews the existing `FractoMorph` (v1.0) codebase, documents the mathematical and algorithmic workflow for threshold selection, contour extraction, local refinement, and region classification, and establishes the quantitative and structural baseline for the multi-method thresholding benchmark.

---

## 2. Codebase Review: Where Thresholding & Classification Occur

In the baseline `FractoMorph` architecture, thresholding and region partitioning are executed across four core modules inside `src/`:

| Pipeline Stage | Module (`src/`) | Baseline Role & Key Functions |
|---|---|---|
| **Stage 0: Preprocessing** | `preprocessing.py` | Computes the 256-bin intensity histogram (`sigma = 2.0` Gaussian smoothing) and applies bilateral denoising (`d = 5, sigmaColor = 25, sigmaSpace = 25`) to smooth dimple floors while preserving steep tear-ridge gradients. |
| **Stage 1: Initial Contours** | `contour_detection.py` | Applies **Global Otsu Thresholding** to separate bright tear ridges from dark valleys, followed by 1-px skeletonization, `cKDTree` gap linking (`R <= 10 px`), and spur pruning (`< 5 px`). |
| **Stage 2: Refinement** | `contour_refinement.py` | Identifies under-segmented outlier regions (`A_i > μ_A + 1.0 × σ_A`) and applies **Localized Otsu Re-Thresholding** within each flagged region mask to subdivide coalesced dimple clusters. |
| **Stage 3 & 4: Voids & Classes** | `void_detection.py`, `region_classification.py` | Detects deep void seeds via dark-regime histogram peak (`I_tip in [5, 100]`), performs Sobel gradient-bounded dilation (`||∇I|| < 15.0`), and assigns each closed region to **Void**, **Surface**, or **Mixed**. |

---

## 3. Detailed Baseline Algorithm Workflow

### 3.1 Global Threshold Selection (Stage 1 — `src/contour_detection.py`)
1. Given the bilateral-filtered 8-bit grayscale image `I(x, y) in [0, 255]`, `FractoMorph` computes a single global scalar threshold `T_otsu` using Otsu's variance-minimization method (`skimage.filters.threshold_otsu`).
2. Otsu's algorithm exhaustively searches for the intensity cutoff `T` that minimizes the weighted within-class variance across background valley pixels (`C_0: I <= T`) and foreground ridge pixels (`C_1: I > T`):
   - `σ_w²(T) = ω_0(T) · σ_0²(T) + ω_1(T) · σ_1²(T)`
3. Pixels satisfying `I(x, y) > T_otsu` form the initial binary ridge mask `B_init(x, y)`.

### 3.2 Topological Contour Extraction (Stage 1 — `src/contour_detection.py`)
1. **Morphological Skeletonization:** The thick binary ridge mask `B_init` is thinned to a 1-pixel-wide topological skeleton using Zhang-Suen / Lee thinning (`skimage.morphology.skeletonize`).
2. **Spatial Endpoint Gap Linking:** Broken ridge endpoints (skeleton pixels with exactly 1 neighbor in an 8-connected window) are indexed in a 2D KD-Tree (`scipy.spatial.cKDTree`). Mutual nearest-neighbor endpoints within Euclidean radius `max_dist = 10 px` are connected via Bresenham line segments.
3. **Crossing-Number Spur Pruning:** Open dangling branches shorter than `5 px` are iteratively pruned while preserving closed loop topology.
4. **Connected-Component Labeling:** Closed background cells enclosed by the 1-px skeleton are labeled using `scipy.ndimage.label`.

### 3.3 Statistical Local Contour Refinement (Stage 2 — `src/contour_refinement.py`)
1. From the initial set of closed regions `{R_1, R_2, ..., R_N}`, the pipeline computes the mean region area (`μ_A`) and standard deviation (`σ_A`).
2. Any region whose area exceeds the statistical outlier cutoff:
   - `A_i > μ_A + 1.0 × σ_A`
   is flagged as a coalesced, under-segmented dimple colony.
3. For each flagged region `R_i`, pixel intensities strictly inside its binary mask are isolated, and a **local Otsu threshold** (`T_local,i`) is computed.
4. Internal sub-ridges (`I(x, y) > T_local,i` within `R_i`) are skeletonized, pruned, and unioned into the master skeleton.
5. Sub-micron speckle regions smaller than `min_area` (default `25 px` for synthetic, `30 px` for real SEM) are merged into adjacent regions.

### 3.4 Void Detection & 3-Class Classification (Stages 3 & 4)
1. **Stage A (Void-Tip Seed Detection):** The smoothed 256-bin histogram `H_smooth` is scanned across the dark intensity window `I in [5, 100]` to find the first local maximum `I_tip`. Non-boundary pixels with `I(x, y) <= I_tip` are marked as deep void seeds.
2. **Stage B (Gradient-Bounded Void Extension):** Void seeds grow outward via iterative morphological dilation inside each closed region, halting wherever:
   - `I(x, y) > min(1.5 × I_tip, 255)`, or
   - Local Sobel gradient magnitude `||∇I(x, y)|| >= 15.0`, or
   - A skeleton contour pixel is encountered.
3. **Three-Class Partitioning:** For each closed region `R_i`, the void coverage fraction `f_void,i = A_void,i / A_i` determines its class:
   - **Surface Region:** `f_void,i == 0.0` (Blue)
   - **Mixed Region:** `0.0 < f_void,i < τ_void` (Yellow, default `τ_void = 0.50`)
   - **Void Region:** `f_void,i >= τ_void` (Red)

---

## 4. Established Baseline Benchmark Performance

Across the 6 standard fractographs (1 synthetic + 5 real experimental SEM micrographs from Banerjee et al., processed with `--min-area 30` and default Otsu thresholding), the baseline implementation produces the following reference partition metrics:

| Sample | Dimensions | Total Regions | Void Count | Surface Count | Mixed Count | AF (Void) | AF (Surface) | AF (Mixed) | Sum AF |
|---|---|---|---|---|---|---|---|---|---|
| **Synthetic SEM** | 512 × 512 px | 62 | 4 | 37 | 21 | 0.0722 | 0.4457 | 0.4820 | **1.0000** |
| **Real SEM Sample 1** | 1280 × 832 px | 2,248 | 436 | 949 | 863 | 0.3396 | 0.0964 | 0.5640 | **1.0000** |
| **Real SEM Sample 2** | 1280 × 831 px | 2,310 | 383 | 900 | 1,027 | 0.4358 | 0.0816 | 0.4826 | **1.0000** |
| **Real SEM Sample 3** | 1280 × 830 px | 2,720 | 4 | 2,413 | 303 | 0.0967 | 0.4933 | 0.4100 | **1.0000** |
| **Real SEM Sample 4** | 1280 × 832 px | 2,299 | 389 | 977 | 933 | 0.4200 | 0.1239 | 0.4561 | **1.0000** |
| **Real SEM Sample 5** | 1280 × 832 px | 1,745 | 42 | 1,075 | 628 | 0.3559 | 0.1314 | 0.5127 | **1.0000** |

---

## 5. Technical Limitations of the Otsu Baseline

While Otsu's global threshold combined with Stage 2 local Otsu refinement achieves topological closure (`Sum AF = 1.0000`), code inspection and experimental testing reveal four critical failure modes inherent to Otsu's formulation:

1. **Equal-Variance & Bimodal Histogram Assumption:**
   - Otsu's objective function assumes the image histogram is bimodal with roughly comparable class variances (`σ_0² ≈ σ_1²`). In real SEM fractographs, thin bright tear ridges occupy a small fraction of total pixels (`10%–20%`), creating a heavily skewed or unimodal histogram with a long upper tail. Otsu shifts the threshold toward the dominant background class, over-thickening ridges or merging shallow dimples.
2. **Vulnerability to Non-Uniform SEM Illumination & Shadow Gradients:**
   - Secondary electron (SE) detector geometry and macroscopic fracture surface tilt create gradual brightness gradients across a single micrograph. A single global scalar `T_otsu` over-thresholds shadowed quadrants (leaving wide gaps > `10 px` in tear ridges that `cKDTree` cannot bridge) while under-thresholding brightly lit quadrants.
3. **Dependence on Bounding-Box Stage 2 Recovery:**
   - Although Stage 2 flags regions with `A_i > μ_A + σ_A`, moderately merged dimple pairs (e.g., two medium dimples whose combined area remains below `μ_A + σ_A`) bypass Stage 2 refinement entirely if Stage 1 global Otsu missed the shared ridge between them.
4. **Sensitivity to Charging Artifacts & Extreme Tail Outliers:**
   - Localized electron charging on sharp ductile tear ridges produces saturated bright spikes (`I ≈ 255`) that inflate within-class variance `σ_1²(T)`, pulling `T_otsu` upward and causing fainter secondary tear ridges to drop out of the binary mask.

---

## 6. Motivation for Multi-Method Thresholding Evaluation

To overcome these baseline limitations without altering downstream contour linking, void extension, or morphometric definitions, Task 2 replaces the fixed global Otsu module with a pluggable thresholding engine comparing six distinct paradigms:
- **Global Variance:** Otsu (Baseline)
- **Global Entropic / Cross-Entropy:** Yen and Li
- **Geometric Histogram Shape:** Triangle (specifically suited for skewed single-peak histograms)
- **Spatially Local Surface:** Adaptive Gaussian Thresholding (`T(x, y)`)
- **Probabilistic Multi-Distribution:** Gaussian Mixture Model (GMM)