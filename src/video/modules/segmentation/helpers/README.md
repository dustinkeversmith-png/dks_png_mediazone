# Segmentation helpers

Small building blocks used by higher-level contour / mission atoms — not end-to-end
photo segmenters.

| Helper | Path | Job |
|--------|------|-----|
| **CCL** | `helpers/ccl/` | Label binary CCs + bboxes (feeds `bbox_auto`) |
| **Watershed** | `helpers/watershed/` | Split touching blobs via EDT markers |
| **BBox auto** | `helpers/bbox_auto/` | Morph-open → CCL largest CC → AABB / crop |
| **Convex hull** | `helpers/convex_hull/` | Hull of an isolated component boundary |
| **Chan–Vese** | `helpers/chan_vese/` | Multiphase Mumford–Shah level-set partition |

`bbox_auto` depends on **CCL** (`largest_component_mask` / `from_binary_via_ccl`).
Watershed is available for optional instance split before boxing.
