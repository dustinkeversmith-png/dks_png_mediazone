Each listed module now has its own atom demo under `src/video/tests/unit_tests/atoms`. Convex hull, medial axis, and CCL were already there; the rest are new input→output binaries that write TSV/PGM artifacts (no ground-truth scoring).

**Boundary tracing**
| Module | Atom |
| --- | --- |
| `bbox/bbox_auto.hpp` | `test_bbox_auto` |
| `moore_neighbor.hpp` | `test_moore_neighbor` |
| `snakes.hpp` | `test_snakes` |
| `marching_squares.hpp` | `test_marching_squares` |
| `dual_contouring.hpp` | `test_dual_contouring` |
| `level_set.hpp` | `test_level_set` |
| `live_wire.hpp` | `test_live_wire` |
| `graph_cut.hpp` | `test_graph_cut` |

**Color, edges, SDF, filters, screen**
| Module | Atom |
| --- | --- |
| `color/lab_color_space.hpp` | `test_lab_color` |
| `featurizations/edge/canny.hpp` | `test_canny` |
| `featurizations/laplace_gaussian/` | `test_laplace_gaussian` |
| `featurizations/sdf/chamfer.hpp` | `test_chamfer` |
| `featurizations/sdf/8SSEDT.hpp` | `test_8ssedt` |
| `featurizations/gvh/gvh.hpp` | `test_gvf` |
| `filters/sobel.hpp` | `test_sobel` |
| `filters/bilateral.hpp` | `test_bilateral` |
| `filters/morph_clean/` | `test_morph_clean` |
| `screen_detection/LOOKUP/template_ncc.hpp` | `test_template_ncc` |
| `screen_detection/CCL/` | `test_ccl` (existing) |
| `vectorization_geometry/convex_hull/` | `test_convex_hull` (existing) |
| `vectorization_geometry/medial_axis/` | `test_medial_axis` (existing) |

If a dedicated pack is missing, the new atoms fall back to `unit_contour` (or `unit_ccl` for NCC). Artifacts go to `src/video/tests/unit_tests/atoms/artifacts/<suite>/`.

```
cmake --build build --config Release --target test_moore_neighbor test_snakes test_canny
.\build\bin\Release\test_moore_neighbor.exe --sample dis5k_val_0002
```