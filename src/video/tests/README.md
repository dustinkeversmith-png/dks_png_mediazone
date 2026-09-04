Here is the consolidated recipe for every unit test in your repository. Each recipe starts strictly from the **raw image** (RGB/Grayscale photo from COCO, SBD, or DIS5K), states the exact chain of preprocessing operations needed to create the optimal test fixture, and defines the validation output.

---

### Group 1: Filters & Color Preprocessing

**`test_bilateral`**

* **From Raw Image:** Convert raw RGB to single-channel luminance $Y$ (Rec. 709).
* **Preprocessing Chain:** Pass raw luminance directly; no denoising upstream.
* **Target Test Stage:** Edge-preserving spatial/range smoothing.
* **Verification Output:** Filtered image `*_bilateral_denoised.pgm` (plus `*_processed_base.pgm` = raw luminance) where flat regions are smoothed ($\sigma_s \approx 3, \sigma_r \approx 0.1 \times 255$) while step-edge gradients $\vert{}\nabla I\vert{}$ preserve at least 90% of their initial magnitude.

**`test_lab_color`**

* **From Raw Image:** Decode raw RGB (gamma-corrected sRGB).
* **Preprocessing Chain:** Convert sRGB $\to$ linear RGB $\to$ CIE-XYZ $\to$ CIE-$L^*a^*b^*$.
* **Target Test Stage:** Decoupling lightness ($L^*$) from chromatic opponent channels ($a^*, b^*$).
* **Verification Output:** Three distinct scalar planes (`*_lab_l.pgm`, `*_lab_a.pgm`, `*_lab_b.pgm`) confirming that chromatic edges between isoluminant colors exist in $a^*/b^*$ even when $L^*$ gradient is zero.

**`test_sobel`**

* **From Raw Image:** Raw RGB $\to$ Luminance ($Y$) $\to$ Bilateral or Gaussian blur ($3 \times 3$, $\sigma = 1.0$).
* **Preprocessing Chain:** Smoothing is mandatory to prevent camera sensor noise from corrupting discrete derivative approximations.
* **Target Test Stage:** Spatial convolution with horizontal ($K_x$) and vertical ($K_y$) Sobel kernels.
* **Verification Output:** Gradient magnitude $G = \sqrt{G_x^2 + G_y^2}$ normalized to $[0, 255]$ and orientation map $\theta = \operatorname{atan2}(G_y, G_x)$.

**`test_canny`**

* **From Raw Image:** Raw RGB $\to$ Luminance ($Y$) $\to$ Gaussian blur ($5 \times 5$, $\sigma = 1.4$).
* **Preprocessing Chain:** Denoise $\to$ Sobel gradient computation $\to$ Non-Maximum Suppression (NMS) along gradient rays $\to$ Hysteresis thresholding ($T_{\text{low}}, T_{\text{high}}$).
* **Target Test Stage:** 1-pixel thin binary edge skeletonization.
* **Verification Output:** Binary mask `*_canny_edges.pgm` consisting strictly of 1-pixel-wide edge centerlines (plus `*_processed_base.pgm` = Gaussian-blurred luma).

**`test_hue_gradient`**

* **From Raw Image:** Raw RGB $\to$ CIE-$L^*a^*b^*$ or HSV color space.
* **Preprocessing Chain:** Compute spatial gradients specifically across chrominance channels ($\nabla a^*, \nabla b^*$).
* **Target Test Stage:** Capturing boundaries that standard luminance detectors miss (e.g., a red stop sign against green foliage under equal daylight).
* **Verification Output:** Chrominance magnitude map $G_{\text{chroma}} = \sqrt{\vert{}\nabla a^*\vert{}^2 + \vert{}\nabla b^*\vert{}^2}$.

**`test_morph_clean`**

* **From Raw Image:** Raw RGB $\to$ Luminance $\to$ Otsu/Adaptive Threshold $\to$ Inversion (so object = 255, background = 0).
* **Preprocessing Chain:** Raw thresholded mask carries salt-and-pepper noise and small topological pinholes.
* **Target Test Stage:** Morphological Opening (Erode $\to$ Dilate) to kill background speckles, followed by Morphological Closing (Dilate $\to$ Erode) with a structuring element ($3 \times 3$ or $5 \times 5$) to fuse internal gaps.
* **Verification Output:** Solid binary mask with high-frequency noise spikes eliminated.

**`test_gvf` (Gradient Vector Flow)**

* **From Raw Image:** Raw RGB $\to$ Bilateral blur $\to$ Canny / Sobel edge magnitude $f(x, y) = \vert{}\nabla I\vert{}^2$.
* **Preprocessing Chain:** Normalize edge map to $[0.0, 1.0]$.
* **Target Test Stage:** Numerical diffusion of edge forces into homogeneous zones via partial differential equations: $\mu \nabla^2 \mathbf{v} - b \cdot (\mathbf{v} - \nabla f) = 0$.
* **Verification Output:** 2-channel vector field $[u, v]$ stored in floating point, where vectors point toward boundary ridges even in flat regions far from edges.

---

### Group 2: Segmentation Helpers

**`test_ccl`** / **`test_bbox_auto`** / **`test_convex_hull`** / **`test_chan_vese`** /
**`test_watershed`** live under `atoms/segmentation/helpers/`.
**`test_suzuki_abe`** lives under `atoms/contour/suzuki_abe/`.

CCL feeds `bbox_auto` (largest component → AABB). Watershed is an optional split helper.

**`test_ccl` (Connected Component Labeling)**

* **From Raw Image:** Raw RGB $\to$ Chrominance + Edge Saliency Gate $\to$ Morphological Clean $\to$ Strict Binary Mask.
* **Preprocessing Chain:** Never feed raw photo luminance directly. Foreground must be explicitly white (255) on black background (0). Components touching outer image borders must be flood-filled out if isolating central objects.


* **Target Test Stage:** 2-pass Union-Find (SAUF) labeling over an 8-connected grid.


* **Verification Output:** Integer label grid `labels[H, W]` + Bounding boxes `[x, y, w, h]` for each component exceeding the minimum area threshold.



**`test_bbox_auto`**

* **From Raw Image:** Raw RGB $\to$ Background removal/thresholding $\to$ Morphological Opening $\to$ CCL.
* **Preprocessing Chain:** Take the single largest connected component from the CCL stage and isolate it into its own single-entity mask.
* **Target Test Stage:** Scanning non-zero pixel coordinate bounds $[\min(x), \min(y), \max(x), \max(y)]$.
* **Verification Output:** Singular axis-aligned bounding box `Rect{x, y, w, h}`.

**`test_convex_hull`**

* **From Raw Image:** Raw RGB $\to$ Inverted Threshold $\to$ CCL $\to$ Moore Boundary Trace.
* **Preprocessing Chain:** Extract the ordered 2D perimeter coordinates `std::vector<Point2i>` of an isolated component. (Do not feed raw cluttered pixels directly).
* **Target Test Stage:** Graham Scan or Monotone Chain algorithm on the discrete vertex set.
* **Verification Output:** Minimal convex polygon vertices enclosing the component without inward dents.

**`test_chan_vese`** (helper under `segmentation/helpers/chan_vese`)

* **From Raw Image:** Raw RGB $\to$ Grayscale luminance $\to$ Gaussian smoothing ($3 \times 3$).
* **Preprocessing Chain:** Normalize pixel intensities to $[0.0, 1.0]$. Initialize a geometric level-set function $\phi(x, y)$ as a regular checkerboard grid of bubbles or an image-spanning bounding box across the domain.
* **Target Test Stage:** Iterative PDE minimization of the Mumford-Shah functional balancing inside variance ($c_1$), outside variance ($c_2$), and contour length ($\mu$).
* **Verification Output:** Zero-level set $\phi(x, y) = 0$ forming smooth closed region loops enclosing textured objects. Artifacts: `*_processed_base.pgm`, `*_chan_vese_phi.pgm`.

**`test_suzuki_abe`** (Suzuki–Abe topological border following — under `contour/suzuki_abe`)

* **From Raw Image:** Raw RGB $\to$ Multi-threshold / Edge-gated binarization $\to$ Binary image containing both outer silhouettes and internal holes (e.g., pants with buttons, or donut shapes).
* **Preprocessing Chain:** Pad image borders with 1 pixel of zero ($0$) to ensure closed external loops.
* **Target Test Stage:** Raster scan walking $0 \to 1$ transitions (outer contours) and $1 \to 0$ transitions (hole contours).
* **Verification Output:** Hierarchical tree structure where each contour stores pointers: `[Next, Prev, First_Child, Parent]`. Artifacts: `*_processed_base.pgm`, `*_suzuki_abe_hierarchy.pgm`.

**`test_watershed`** (helper under `segmentation/helpers/watershed`; optional split before `bbox_auto`)

* **From Raw Image:** Raw RGB $\to$ Clean binary mask containing touching/overlapping objects (e.g., teddy bears, fruit).
* **Preprocessing Chain:**
1. Compute Euclidean Distance Transform ($EDT$) on the binary mask.
2. Local maxima thresholding on the $EDT$ to produce isolated marker seeds for each object center.
3. Invert the distance map (or use Sobel gradient magnitude) as the topographic relief: $H = -EDT$.


* **Target Test Stage:** Meyer's flooding algorithm expanding outward from the seed markers until flood fronts collide.
* **Verification Output:** 1-pixel-wide partition boundaries splitting touching instances along valleys. Artifacts: `*_processed_base.pgm`, `*_watershed_lines.pgm`.

`test_template_ncc` has been removed (not needed for the current segmentation helper stack).

---

### Group 3: Signed Distance Fields (SDF)

**`test_8ssedt` / `test_edt**`

* **From Raw Image:** Raw RGB $\to$ Segmented binary mask (from CCL or ground truth).
* **Preprocessing Chain:** Ensure strict 2-state grid ($0$ and $255$). Border boundaries clamped.
* **Target Test Stage:** 2-pass 8-point Signed Sequential Euclidean Distance Transform or Meijster's separable parabolic algorithm.
* **Verification Output:** Floating-point 2D scalar field where each pixel value equals the exact sub-pixel Euclidean distance to the nearest boundary.

**`test_chamfer`**

* **From Raw Image:** Raw RGB $\to$ Canny edge detection $\to$ Binary edge skeleton.
* **Preprocessing Chain:** Invert edge map so boundary edges are $0$ and empty space is $\infty$.
* **Target Test Stage:** Forward and backward passes with integer distance weights (e.g., Borgefors $3\text{--}4$ metric: orthogonal step = 3, diagonal step = 4).
* **Verification Output:** Integer scalar field approximating Euclidean distance.

---

### Group 4: Contouring & Meshing

**`test_marching_squares`**

* **From Raw Image:** Raw RGB $\to$ Binary Mask $\to$ Exact Distance Transform ($EDT$).
* **Preprocessing Chain:** Generate a continuous scalar field; select target isovalue $\tau = 0.0$ (or target luminance threshold).
* **Target Test Stage:** $2 \times 2$ grid cell evaluation, 16-case lookup table, linear interpolation along cell edges.
* **Verification Output:** Unordered line segments stitched into closed polyline loops `std::vector<Vec2f>`.

**`test_dual_contouring`**

* **From Raw Image:** Raw RGB $\to$ Distance Transform ($EDT$) $\to$ Compute Hermite normals via central difference gradients on the field $\nabla \phi$.
* **Preprocessing Chain:** Create a continuous signed scalar field paired with normal vectors at zero-crossing edges.
* **Target Test Stage:** Octree/quadtree edge-crossing detection $\to$ Quadric Error Function (QEF) solve via SVD per cell.
* **Verification Output:** Polygon mesh (vertex buffer + triangle/quad index buffer) preserving sharp geometric features.

**`test_moore_neighbor`**

* **From Raw Image:** Raw RGB $\to$ Inverted binary threshold $\to$ Morphological close.
* **Preprocessing Chain:** Isolate single connected component mask; ensure background is strictly 0.
* **Target Test Stage:** Clockwise/counter-clockwise 8-neighborhood walking starting from the first non-zero pixel.
* **Verification Output:** Ordered clockwise list of boundary pixels `std::vector<Point2i>`.

**`test_snakes`**

* **From Raw Image:** Raw RGB $\to$ Bilateral blur $\to$ Sobel edge magnitude $\to$ GVF vector field.
* **Preprocessing Chain:** Initialize a coarse, loose polygon/spline around the target object.
* **Target Test Stage:** Euler-Lagrange iterative solver pulling spline vertices under internal stiffness/elasticity and external GVF forces.
* **Verification Output:** Deformed sub-pixel accurate spline contour snapped tight to object boundaries.

**`test_live_wire`**

* **From Raw Image:** Raw RGB $\to$ Grayscale $\to$ Sobel magnitude + Laplacian Zero-Crossings.
* **Preprocessing Chain:** Build a directed cost graph where edge cost is inversely proportional to gradient magnitude and zero-crossing presence: $Cost = w_z Z + w_g (1 - \vert{}\nabla I\vert{})$.
* **Target Test Stage:** Dijkstra / Priority Queue shortest-path search between designated seed points.
* **Verification Output:** Sub-pixel optimal boundary path snapped to the object edge.

**`test_rdp` (Ramer-Douglas-Peucker)**

* **From Raw Image:** Raw RGB $\to$ CCL $\to$ Moore Boundary Trace or Marching Squares.
* **Preprocessing Chain:** Extract dense polyline coordinate list containing hundreds of adjacent boundary pixels.
* **Target Test Stage:** Recursive polyline decimation based on perpendicular distance tolerance $\epsilon$.
* **Verification Output:** Simplified sparse vector polygon retaining only corner vertices.

---

### Group 5: Geometrify & Topology

**`test_fourier_descriptors`**

* **From Raw Image:** Raw RGB $\to$ Clean Mask $\to$ Boundary Trace $\to$ RDP.
* **Preprocessing Chain:** Resample the closed polygon uniformly by arc-length into $N = 2^k$ coordinates. Represent coordinates as complex numbers $s(t) = x(t) + i y(t)$.
* **Target Test Stage:** Fast Fourier Transform (FFT) on the coordinate series; normalize coefficients by $F(1)$ for scale and rotation invariance.
* **Verification Output:** Fixed-length vector of invariant harmonic shape descriptors.

**`test_hu_moments`**

* **From Raw Image:** Raw RGB $\to$ Clean binary mask of isolated component.
* **Preprocessing Chain:** Ensure binary image has exactly one component with zero background noise.
* **Target Test Stage:** Compute raw spatial moments $m_{pq}$, central moments $\mu_{pq}$, and normalized central moments $\eta_{pq}$.
* **Verification Output:** Array of 7 invariant moment values ($H_1 \dots H_7$).

**`test_medial_axis`**

* **From Raw Image:** Raw RGB $\to$ Clean binary mask $\to$ Exact Distance Transform ($EDT$).
* **Preprocessing Chain:** Compute exact continuous distance field.
* **Target Test Stage:** Identify local distance ridge maxima (points where the distance map slope transitions or where maximal inscribed discs touch at least two boundary points).
* **Verification Output:** 1-pixel-wide morphological topological skeleton of the object.

**`test_euler`**

* **From Raw Image:** Raw RGB $\to$ Clean binary mask.
* **Preprocessing Chain:** Ensure discrete binary raster.
* **Target Test Stage:** Bit-quad neighborhood scanning ($2 \times 2$ pixel patterns) to calculate the Euler characteristic:

$$\chi = V - E + F = N_{\text{objects}} - N_{\text{holes}}$$


* **Verification Output:** Scalar integer $\chi$ verifying topological invariants.