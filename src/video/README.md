# Vision (`src/video`)

## Layout
- `modules/` — algorithms (contour kit, geometry, CCL, trees, …)
- `tests/unit_tests/atoms/` — atom I/O demos (values + images, no GT scoring) → `atoms/artifacts/<suite>/`
- `tests/unit_tests/benchmarks/contour/` — contour invariants + DIS/COCO/BSDS benches → `.../artifacts/`
- `tests/integration_tests/` → `integration_tests/artifacts/`
- Data: `data/vision/atoms/` (real DIS packs), `data/vision/benchmarks/contour_ready/`

## Build / run
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target test_hu_moments test_ccl test_contour_invariants
python data/vision/prepare_atom_datasets.py   # once, from contour_ready
.\build\bin\Release\test_hu_moments.exe --data data/vision/atoms --sample dis5k_val_0002
.\build\bin\Release\test_hu_moments.exe --help
```

See `DEPENDENCIES.md` for third-party libs.

---

4. Low-Level Video Classification FeaturesOnce you have pointer access to raw frames (frame->data), video classification generally relies on temporal and spatial signals:Temporal Differencing & Motion History (MHI):Subtract consecutive Luma frames $\vert{}Y_t(x,y) - Y_{t-1}(x,y)\vert{}$ with SIMD (AVX2 / _mm256_sub_epi8). This gives you instant motion vectors and activity masks without heavy neural networks.Dense Optical Flow (Farnebäck or Lucas-Kanade):Calculates vector displacement $(dx, dy)$ for each pixel between frames to track direction and speed of motion.3D Spatio-Temporal Convolutions / Tensor Batches:Group sequences into shapes like $(N \text{ frames}, C, H, W)$ for classification models (e.g., action recognition, scene cut detection).


Distance Transforms & Signed Distance Fields (SDFs):Compute the 2D Euclidean Distance Transform (e.g., Felzenszwalb–Huttenlocher algorithm in $O(N)$). Contours become continuous scalar fields $f(x, y) = d$, enabling gradient descent to boundary edges, skeletonization (Medial Axis Transform), and collision/overlap checks via simple sign comparisons.Hu Moments & Flusser Affine Invariants:Extract 7 central normalized moments ($\eta_{pq}$) from silhouettes. They remain strictly invariant under translation, uniform scaling, and 2D rotation.Radial Basis Functions (RBFs):Fit sparse contour point clouds with thin-plate splines ($\phi(r) = r^2 \ln r$) to represent arbitrary organic deformable silhouettes as continuous, differentiable parametric functions.Shape Dot Products & Fourier Descriptors:
Sample $N$ equidistant points along a closed silhouette perimeter, map coordinates into complex values $z(n) = x(n) + j y(n)$, and run an FFT via fftw3 or PocketFFT. The lower frequency magnitudes yield a scale- and rotation-invariant shape signature that can be compared instantly with Euclidean distance or vector dot products.
