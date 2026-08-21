# Vision C++ dependencies

All paths are relative to the repository root. Fastest path on this machine: the copies already under `dependencies/` (Eigen, FFTW, FFmpeg, stb, libspatialindex). Fetch anything missing with:

```
python scripts/fetch_vision_dependencies.py
```

| Library | Where | Fastest C++ install | Notes |
| --- | --- | --- | --- |
| **FFmpeg** (`libavcodec`, `libavformat`, `libavutil`, `libswscale`) | https://ffmpeg.org · Windows shared builds: https://www.gyan.dev/ffmpeg/builds/ (essentials) | Already at `dependencies/ffmpeg/{include,lib,bin}`. Copy `bin/*.dll` next to the exe. | C APIs used by `src/video/libffmepg.cpp`. |
| **stb_image.h** / **stb_image_write.h** | https://github.com/nothings/stb | Single-header drop-in: `dependencies/stb/stb_image.h` (already present). | `#define STB_IMAGE_IMPLEMENTATION` once in `vision_io.cpp`. Load 8-bit grayscale with `stbi_load(..., 1)`. |
| **FFTW3** (single-precision `fftw3f`) | https://www.fftw.org/install/windows.html | Already at `dependencies/fftw` (`fftw3.h` + `libfftw3f-3.dll`). Root CMake builds the import lib from the `.def`. | Used by Fourier descriptors. Linux: `sudo apt install libfftw3-dev`. |
| **Eigen** | https://gitlab.com/libeigen/eigen · https://eigen.tuxfamily.org | Already at `dependencies/eigen`. Header-only: add the include path. | RBF linear solve + optional `unsupported/Eigen/BVH`. |
| **YACCLAB** | https://github.com/prittt/YACCLAB | `git clone --depth 1 https://github.com/prittt/YACCLAB.git dependencies/YACCLAB` | Full harness needs OpenCV + CMake. Unit tests use the same **SAUF** two-pass union-find CCL YACCLAB ships, implemented in `screen_detection/CCL/connected_components.hpp` so tests do not require OpenCV. |
| **libspatialindex** | https://github.com/libspatialindex/libspatialindex · https://libspatialindex.org | Already cloned + built: `dependencies/libspatialindex/build/src/Release/spatialindex-64.lib`. Or `vcpkg install libspatialindex`. | 2D R-tree unit test uses an in-header STR R-tree (no extra link). Optional CMake flag `VISION_HAS_SPATIALINDEX` wraps the C API. |
| **Felzenszwalb–Huttenlocher DT** | Paper + reference: https://cs.brown.edu/people/pfelzens/dt/ | Implemented in `src/video/modules/math/distance_transform.hpp` (O(N) 1D lower-envelope, then separable 2D). | Used for SDF, medial-axis ridges, and skeletonization. No third-party link. |

## Layout (post-refactor)

| Path | Role |
| --- | --- |
| `src/video/modules/` | All vision/contour algorithm headers + `math/vision_io.cpp` |
| `src/video/tests/unit_tests/atoms/` | Atom unit tests; write `atoms/artifacts/<suite>/` |
| `src/video/tests/unit_tests/benchmarks/contour/` | Contour invariants + DIS/COCO/BSDS benches; write `.../artifacts/` |
| `src/video/tests/integration_tests/` | Alpha/beta/gamma; write `integration_tests/artifacts/` |
| `data/vision/atoms/` | Real-photo atom packs (`python data/vision/prepare_atom_datasets.py`) |
| `data/vision/benchmarks/contour_ready/` | Contour PGM packs (`python data/vision/prepare_contour_pgms.py`) |
| `data/vision/_archived/` | Old synthetic unit packs (not used by the build) |

Atom CLI: `test_hu_moments.exe --data data/vision/atoms --dataset unit_hu_moments --sample dis5k_val_0002`

## Windows one-liners (only if a folder is missing)

```
git clone --depth 1 https://gitlab.com/libeigen/eigen.git dependencies/eigen
curl -L -o dependencies/stb/stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
git clone --depth 1 https://github.com/libspatialindex/libspatialindex.git dependencies/libspatialindex
cmake -S dependencies/libspatialindex -B dependencies/libspatialindex/build -DCMAKE_BUILD_TYPE=Release
cmake --build dependencies/libspatialindex/build --config Release
```

FFmpeg (Windows shared): unzip a gyan.dev `ffmpeg-*-full_build-shared` tree into `dependencies/ffmpeg` so that `include/libavcodec` and `lib/avcodec.lib` exist.
