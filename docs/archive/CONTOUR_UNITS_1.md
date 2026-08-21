The contour suite is in place as a **stdlib-only C++20** stack (no OpenCV, Eigen, FFTW, or stb). It evaluates the six algorithms against real DIS5K, COCO, and BSDS500 samples, not the earlier synthetic baby sets.

**Invariant tests: 6/6 passed.** Dual Contouring’s QEF recovers a 90° corner with 0 error; Marching Squares preserves circular area to 0.06% relative error; Chan-Vese, Snakes, Graph Cut, and Livewire all clear their synthetic checks.

**Real-data harness (20/20 rows):** DIS5K reconstruction IoU is typically 0.75–0.95 for Marching Squares. COCO Graph Cut / Snakes land around 0.46–0.87 IoU from a bbox prior. BSDS Livewire reports boundary F1 against human maps.

**Algorithms** (header implementations behind the stub `.cpp` files):
- Marching Squares — sub-pixel iso interpolation on SDFs
- Dual Contouring 2D — Hermite samples + 2×2 QEF (direct inverse, Tikhonov when rank-deficient)
- Snakes — tension/rigidity + Sobel edge energy
- Chan-Vese — region statistics, periodic SDF reinit
- Livewire — Dijkstra on inverse-gradient / Laplacian-zero-crossing costs
- Graph Cut — 8-neighbor grid max-flow with bbox t-links

**Plumbing:** NetPBM/BMP loader, COCO polygon/bbox JSON parser, Chamfer SDF, Sobel, IoU / Boundary F1 / convex-hull tightness / `std::chrono` latency.

**How to run**
```text
python data/vision/prepare_contour_pgms.py
cmake --build build --config Release --target test_contour_invariants test_contour_benchmarks
.\build\bin\Release\test_contour_invariants.exe
.\build\bin\Release\test_contour_benchmarks.exe
```

PGM sidecars live in `data/vision/benchmarks/contour_ready/`. ADE20K is still empty in this tree; LVIS and SBD were not present, so those pipelines are skipped until they are fetched.


### Why Your Run Used So Few Samples

1. **Hardcoded Smoke-Test Limits:** The fetcher and runner were capped (e.g., streaming the first 4 images or capping at ~500MB) to verify compilation and prevent lengthy downloads before your C++ harness was proven.
2. **Computational Bottleneck (GraphCut & Snakes):** GraphCut on a raw $500 \times 500$ image builds a graph with 250,000 nodes and $\sim 1$ million edges. At $\sim 500\text{--}800\text{ ms}$ per sample, running a full 5,000-image split would take over an hour on a single thread.
3. **Missing Automated Batch Loop:** The driver script was dispatching hardcoded sample IDs rather than iterating through the dataset index (`instances_val2017_subset.json` or directory iterators).

---

### Optimal Preprocessing & Isolation Pipeline

To test the **mathematical accuracy of the algorithms themselves** rather than testing how well a simple heuristic handles bad image data, apply these specific preprocessing steps:

```
[Raw Image / Mask] 
       │
       ├── (Marching Squares / Dual Contouring) ──► Morphological Clean ──► 8SSEDT Exact SDF ──► Iso-Surface
       │
       ├── (Snakes / Active Contours) ───────────► Bilateral Filter ────► Canny / GVF Field ──► Energy Minimize
       │
       ├── (GraphCut / Min-Cut) ─────────────────► BBox Auto-Crop ──────► Lab Color Space ────► Max-Flow
       │
       └── (Livewire / Intelligent Scissors) ────► Laplacian of Gaussian ─► Cost Inversion ────► Dijkstra

```

#### 1. Dual Contouring & Marching Squares (Implicit Surface Extraction)

* **Pre-Filter the Mask:** Apply a $3\times3$ binary morphological opening/closing to eliminate single-pixel checkerboard disconnects in the ground-truth raster.
* **Exact Euclidean Distance Field:** Use an exact distance transform (e.g., 8-point Signed Sequential Euclidean Distance Transform, 8SSEDT, or Felzenszwalb–Huttenlocher parabolic envelope) instead of an approximate Chamfer/Manhattan metric.
* **Sub-Pixel Normal Estimation:** Do not compute normals via raw discrete pixel differences on the binary mask. Compute central differences **on the continuous floating-point SDF**:

$$n(x, y) = \text{normalize}\left(\frac{\phi(x+1, y) - \phi(x-1, y)}{2}, \frac{\phi(x, y+1) - \phi(x, y-1)}{2}\right)$$


* **SVD Regularization & Clamping (Fixes the Dual Contouring 0.374 drop):** When the condition number of $A^T A > 100$ or singular values $\sigma_i < 10^{-2}$, fall back to the cell center or mass point of edge crossings:

$$x^* = \text{clamp}\left(x_{\text{SVD}}, [x_{\text{min}}, x_{\text{max}}]\right)$$



#### 2. Parametric Snakes & Chan-Vese (Deformable Models)

* **Edge-Preserving Smoothing:** Run a Bilateral Filter or Guided Filter rather than standard Gaussian blur. Gaussian blur displaces true edge locations by several pixels.
* **Compute Gradient Vector Flow (GVF):** Standard image gradient $\vert{}\nabla I\vert{}^2$ has a capture range of only 2–3 pixels. Diffuse the gradient vectors outward across the image domain by solving:

$$\frac{\partial u}{\partial t} = \mu \nabla^2 u - (u - f_x)(f_x^2 + f_y^2)$$



This pulls the snake into deep concavities from dozens of pixels away.
* **Convex-Hull Prior Initialization:** Initialize the initial snake loop as a slightly dilated convex hull of the bounding box rather than a naive circle or box.

#### 3. GraphCut / Boykov-Kolmogorov (Combinatorial Optimization)

* **Crop to Bounding Box with Margin:** Do not run max-flow on the entire image. Crop to the ground-truth bounding box with a fixed $10\%\text{--}15\%$ outer border.
* **Color Space Transform:** Convert RGB to **CIE-Lab**. Euclidean color distances in Lab space correlate with perceptual contrast, preventing GMM likelihood collapse on subtle lighting shifts.
* **Contrast-Weighted Edge Energies:** Weight neighbor links ($n$-links) inversely by local color contrast:

$$B_{p,q} = \frac{\gamma}{\text{dist}(p, q)} \exp\left(-\frac{\Vert{}I_p - I_q\Vert{}^2}{2\langle(I_p - I_q)^2\rangle}\right)$$



#### 4. Livewire / Intelligent Scissors (Graph Search)

* **Cost Map Generation:** Combine three normalized features:

$$f(p, q) = w_z f_z(q) + w_g f_g(q) + w_d f_d(p, q)$$


* $f_z(q)$: Laplacian of Gaussian (LoG) zero-crossing (binary: 0 on edge, 1 elsewhere).
* $f_g(q)$: Inverse gradient magnitude ($1 - \frac{\vert{}\nabla I\vert{}}{\max \vert{}\nabla I\vert{}}$).
* $f_d(p, q)$: Gradient direction alignment penalty (dot product of normal vectors).


* **Automated Multi-Seed Placement:** Place seed points every $k$ vertices along the ground-truth polygon (e.g., spaced 30–50 pixels apart) to simulate user clicks along the perimeter.

---

### Optimal Dataset Test Scenarios

To properly isolate where each algorithm excels, evaluate them against specific subsets tailored to their operational assumptions:

| Test Scenario | Best Dataset / Subset | Why This Isolates Algorithm Accuracy | Primary Metric |
| --- | --- | --- | --- |
| **Exact Surface Extraction** | **DIS5K (High-Res Masks)** | Evaluates whether Marching Squares & Dual Contouring reconstruct fine geometry (mesh, hair, sharp corners) from an exact SDF without grid artifacts. | Boundary F1 ($\text{tol}=1\text{px}$) & Chamfer Distance |
| **Sharp Feature Recovery** | **Synthetic Polygons + CAD Silhouettes** | Isolates Dual Contouring's QEF corner preservation vs. Marching Squares' corner-rounding defect. | Vertex Angular Error ($\vert{}\theta_{\text{pred}} - \theta_{\text{true}}\vert{}$) |
| **Noisy Region Segmentation** | **Synthetic Texture Blobs / Medical (ISIC)** | Tests Chan-Vese's ability to segment homogeneous regions when gradient magnitude $\vert{}\nabla I\vert{} \approx 0$. | Region IoU |
| **Foreground / Cluttered Background** | **COCO (High-Contrast Subset)** | Tests GraphCut's energy minimization when provided with a standard 10% padded bounding box. | mIoU & Boundary IoU |
| **Boundary Edge Snapping** | **BSDS500 (Filtered single-object edges)** | Tests Livewire's shortest-path snapping along high-gradient contours between anchor points. | Mean Boundary Distance to GT (px) |

---

### Recommended Batch Evaluation Matrix

Run a mini-batch of **100 samples per category** using the following criteria to get statistically stable metrics:

```cpp
struct BenchmarkSummary {
    double mean_iou;
    double mean_boundary_f1;
    double mean_hausdorff_px;
    double p95_latency_ms;
};

```

1. **Marching Squares & Dual Contouring:** Run on 100 samples from DIS5K. Measure sub-pixel precision and vertex count efficiency.
2. **GraphCut:** Run on 100 COCO validation images with objects taking up $>15\%$ of image area and bounding box padding $= 10\%$.
3. **Snakes (with GVF):** Run on 100 high-contrast DIS5K samples initialized with dilated bounding ellipses.
4. **Livewire:** Run on 100 BSDS500 images with automated seed placement every 30px along human-annotated boundaries.


Each algorithm family belongs to a distinct stage of the vision/graphics pipeline.

---

### Pipeline Architectures & Data Formats

| Paradigm / Algorithm | Input Data | Intermediate Processing Pipeline | Output Data Format |
| --- | --- | --- | --- |
| **Marching Squares** | 2D Scalar Field / SDF Grid `float32[H, W]` + Iso-value $\tau$ | Cell classification (16-case LUT) $\to$ Edge crossing linear interpolation $\to$ Edge segment stitching | **Unordered Line Segments / Polyline Loops** `std::vector<Vec2f>` |
| **Dual Contouring** | 2D SDF Grid + Hermite Normals (Gradients) `Vec2f[H, W]` | Octree/Quadtree cell traversal $\to$ Edge sign change detection $\to$ **SVD / Pseudo-inverse QEF solve** per cell $\to$ Dual polygon generation | **Sharp-Feature Polygon / Mesh** (Vertices + Quad/Triangle Index Buffer) |
| **Livewire (Intelligent Scissors)** | Raw Grayscale/RGB Image `uint8[H, W, C]` + Seed points $(x_0, y_0), (x_1, y_1)$ | Sobel/Canny gradient magnitude + Laplacian zero-crossings $\to$ Cost graph construction $\to$ **Dijkstra Priority Queue search** | **Ordered Open Polyline Path** `std::vector<Point2i>` snapped to edge |
| **Graph Cut (Boykov-Kolmogorov)** | RGB Image `uint8[H, W, 3]` + Bounding Box / Scribble Priors | CIE-Lab color conversion $\to$ GMM foreground/background modeling $\to$ $t$-link & $n$-link graph construction $\to$ **Augmenting Paths Max-Flow / Min-Cut** | **Binary Segmentation Mask** `uint8[H, W]` ($0 = \text{BG}, 1 = \text{FG}$) |
| **Snakes + GVF (Active Contours)** | Grayscale Image `uint8[H, W]` + Initial Closed Spline $(v_0 \dots v_M)$ | Sobel edge map $\to$ **PDE Diffusion (GVF Field)** $\to$ Iterative Euler-Lagrange solver with cyclic pentadiagonal matrix inversion | **Smooth Continuous Parametric Spline** `std::vector<Vec2f>` |

---

### Step-by-Step Execution Pipelines

#### 1. Marching Squares (Iso-surface Extractor)

```
[SDF / Scalar Grid]
        │
        ▼
[2x2 Cell Window Traversal] ──► Lookup Case (0–15)
        │
        ▼
[Linear Edge Interpolation: t = (τ - v1) / (v2 - v1)]
        │
        ▼
[Output: Polylines / Isoline Loops]

```

#### 2. Dual Contouring (Feature-Preserving Mesher)

```
[SDF Grid + Normal Vector Field]
        │
        ▼
[Identify Grid Edges with Sign Changes]
        │
        ▼
[Collect Intersection Points p_i & Normals n_i per Cell]
        │
        ▼
[Solve QEF: min_x Σ (n_i · (x - p_i))² via SVD] ──► Clamp to Cell Boundary
        │
        ▼
[Connect Adjacent Cell Vertices across Crossing Edges]
        │
        ▼
[Output: Sharp Vector Polygons + Index Buffer]

```

#### 3. Livewire (Interactive Edge Snapper)

```
[Raw Image] ──► [Compute Gradient Magnitude + LoG Zero-Crossings]
                                │
                                ▼
                       [Build Cost Graph: f_cost = w_z·Z + w_g·(1 - |∇I|) + w_d·D]
                                │
                                ▼
[User / Auto Seed Points] ──► [Dijkstra Shortest Path Search]
                                │
                                ▼
                       [Output: Sub-pixel Accurate Boundary Path]

```

#### 4. Graph Cut / GrabCut (Region Optimizer)

```
[Raw Image + Bounding Box Prior]
        │
        ▼
[Convert to CIE-Lab Space] ──► [Fit Foreground/Background GMMs]
        │
        ▼
[Build s-t Graph: Regional Penalties (t-links) + Contrast Penalties (n-links)]
        │
        ▼
[Boykov-Kolmogorov Min-Cut Algorithm]
        │
        ▼
[Output: Hard Pixel Assignment Mask (0 / 1)]

```

#### 5. Snakes + GVF (Spline Deformer)

```
[Raw Image] ──► [Edge Map: f(x,y) = |∇(G_σ * I)|²]
                         │
                         ▼
        [Solve Diffusion PDE for GVF Vector Field (u, v)]
                         │
                         ▼
[Initial Spline Guess] ──► [Iterative Euler-Lagrange Matrix Solver: (A + γI)V = γV - F_ext]
                         │
                         ▼
                [Output: Deformed Vector Spline Control Points]

```

---

### How They Chain Together in a Real Production Pipeline

In modern geometric and computer vision systems, these paradigms are often chained sequentially:

```
[Raw Cluttered Photo]
         │
         ▼
 1. [Graph Cut]                <-- Extracts rough foreground region mask (0/1)
         │
         ▼
 2. [8SSEDT Exact Distance]    <-- Converts binary mask to continuous float32 SDF
         │
         ▼
 3. [Dual Contouring / MS]     <-- Extracts smooth or sharp vector polygons
         │
         ▼
 4. [Snakes / Active Spline]   <-- Snaps vector control points to sub-pixel image gradients
         │
         ▼
   [Final Clean SVG / CAD Output]

```