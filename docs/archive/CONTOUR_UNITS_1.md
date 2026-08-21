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