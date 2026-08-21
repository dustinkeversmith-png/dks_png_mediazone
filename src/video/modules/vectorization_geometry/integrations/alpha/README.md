1. The "No-Nets" Vision Toolkit: Techniques & Math[Raw Frame / Luma Plane]
         │
         ▼
[Contour / Silhouette Tracing] ────► (Moore-Neighbor / Marching Squares)
         │
         ▼
[Invariant Descriptors (SDF / Hu)] ─► (Scale / Rotation Invariance)
         │
         ▼
[Spatial / Vector Index Query] ────► (VP-Tree / 2D R-Tree / KD-Tree)
         │
         ▼
[Hierarchical Component Binding] ──► (Part-Graph / Relational Constraints)
Distance Transforms & Signed Distance Fields (SDFs):Compute the 2D Euclidean Distance Transform (e.g., Felzenszwalb–Huttenlocher algorithm in $O(N)$). Contours become continuous scalar fields $f(x, y) = d$, enabling gradient descent to boundary edges, skeletonization (Medial Axis Transform), and collision/overlap checks via simple sign comparisons.Hu Moments & Flusser Affine Invariants:Extract 7 central normalized moments ($\eta_{pq}$) from silhouettes. They remain strictly invariant under translation, uniform scaling, and 2D rotation.Radial Basis Functions (RBFs):Fit sparse contour point clouds with thin-plate splines ($\phi(r) = r^2 \ln r$) to represent arbitrary organic deformable silhouettes as continuous, differentiable parametric functions.Shape Dot Products & Fourier Descriptors:
Sample $N$ equidistant points along a closed silhouette perimeter, map coordinates into complex values $z(n) = x(n) + j y(n)$, and run an FFT via fftw3 or PocketFFT. The lower frequency magnitudes yield a scale- and rotation-invariant shape signature that can be compared instantly with Euclidean distance or vector dot products.


