# BBox auto (helper)

**Role:** segmentation *helper* — axis-aligned bounding box from a binary mask.

## Pipeline (recipe)
1. Binary mask (threshold / background removal)
2. Morphological opening
3. **CCL** (`helpers/ccl`) → keep largest connected component
4. Scan non-zero extrema → `Rect{x,y,w,h}`
5. Optional crop / uncrop with margin

## API
- `from_mask` — AABB extrema scan
- `largest_component_mask` — open → CCL → largest CC mask
- `from_binary_via_ccl` — full recipe AABB
- `crop` / `uncrop` / `pad`

## Depends on
`segmentation/helpers/ccl/connected_components.hpp`

Watershed (`helpers/watershed`) can optionally split touching blobs *before* taking the
largest component when callers need instance isolation.
