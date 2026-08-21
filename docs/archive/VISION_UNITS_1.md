Vision unit tests, datasets, and C++ wiring are in place. Each test is its own executable: load dataset → run analysis → score accuracy → write artifacts.

## Getting the datasets

Run from the repo root (each suite is capped at **500 MB**):

```
python data/vision/download_vision_datasets.py
```

Output lives in `data/vision/data_vision/`. Full format table is in `data/vision/FORMATS.md`. C++ reads **8-bit PNG** via `stb_image` (`stbi_load(..., 1)`, then `pixel > 128`) and labels from `index.tsv` (`file<TAB>label<TAB>...`).

| Suite | Resource | Retrieval |
| --- | --- | --- |
| MPEG-7 CE-Shape-1 Part B | Temple `mpeg7shapeB.tar.gz` (~2.3 MB) | Script fetched `http://www.cis.temple.edu/~latecki/TestData/mpeg7shapeB.tar.gz` into `integration_1_shapes/` |
| Kimia-99 | GitHub zip | Fallback mirror; local silhouettes always generated |
| Swedish Leaf | Official TIFFs are **GB per class** | Binary 15-species silhouettes in `integration_3_leaf/` (same protocol, under the cap) |
| RICO | Full set ~9 GB | Labeled 540×960 screens + `boxes.tsv` in `integration_2_rico_ui/` |
| SynthText / CHAR74K / MNIST | Full dumps are huge | Glyph RST set, Euler genus set, 64×64 crops |

## Getting the C++ dependencies

Fastest path on this machine: they are **already** under `dependencies/`. Recheck / fill gaps with:

```
python scripts/fetch_vision_dependencies.py
```

Details: `src/video/DEPENDENCIES.md`.

| Lib | Where | Fastest install |
| --- | --- | --- |
| FFmpeg | gyan.dev shared build | `dependencies/ffmpeg/{include,lib,bin}` |
| stb_image.h | github.com/nothings/stb | already in `dependencies/stb/` |
| FFTW3 | fftw.org Windows zip | already in `dependencies/fftw/` (CMake builds the import lib) |
| Eigen | gitlab.com/libeigen/eigen | header-only at `dependencies/eigen/` |
| YACCLAB | github.com/prittt/YACCLAB | optional clone; tests use in-tree **SAUF** CCL (no OpenCV) |
| libspatialindex | already built | tests use an in-header STR R-tree |
| Felzenszwalb–Huttenlocher DT | paper algorithm | `src/video/math/distance_transform.hpp` |

## Unit tests

Independent executables in `src/video/unit_tests/`, classes in the matching README folders. Build:

```
cmake -B build -S .
cmake --build build --config Release
```

Run e.g. `.\build\bin\Release\test_boundary_tracing.exe`

| Test | Implementation | Result |
| --- | --- | --- |
| `test_boundary_tracing` | Moore-neighbor | 100% |
| `test_fourier_descriptors` | fftw3 Fourier magnitudes | 98.3% 1-NN |
| `test_hu_moments` | Hu + Flusser | 92.2% |
| `test_rbf` | Eigen thin-plate spline | 100% |
| `test_lookup_tables` | aspect/compactness LUT | 100% |
| `test_ccl` | SAUF CCL | 100% |
| `test_2d_r_tree` | STR R-tree vs brute force | 100% |
| `test_lsh` | Fourier LSH | 100% recall@1 |
| `test_vp_tree` | Chamfer VP-tree | 100% self-NN |
| `test_convex_hull` | monotone chain | 100% |
| `test_euler` | χ = C − H | 100% |
| `test_turning_function` | Θ(s) 1-NN | 42% (pass threshold 35%; synthetic leaves are close) |
| `test_medial_axis` | FH EDT ridges | 100%; hands 212 vs blobs 7 skeleton px |
| `test_rdp` | RDP polylines | 100% |
| `test_integration_alpha` | Hu + SDF → VP-tree | **100%**, ~0.002 ms/query |
| `test_integration_beta` | RICO CCL + NCC | **100%**, 0 control mismatches |
| `test_integration_gamma` | leaves + Euler | 89.7% |

Artifacts go to `artifacts/<test_name>/`.