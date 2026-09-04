# Connected Component Labeling (CCL)

**Role:** segmentation *helper* — turns a strict binary mask into instance labels + boxes.
Not a full photo→objects pipeline by itself; upstream atoms (saliency, morph clean, proposals)
must produce FG=255 / BG=0 first.

## Location
`src/video/modules/segmentation/helpers/ccl/`

## API
`vision::ConnectedComponentLabeler::label(GrayImage, thr)` — 2-pass SAUF / Union-Find, 8-connected.

Returns:
- `labels[H*W]` integer grid (0 = background)
- `components[]` with `label`, `area`, `bbox`

Helpers:
- `filter_min_area`
- `merge_boxes` (IoU / near-duplicate merge)

## Used by
- `bbox_auto` — largest CC → axis-aligned bbox
- `convex_hull`, `suzuki_abe`, `object_proposals`, watershed prep
- SDF / contour atoms that need an isolated binary blob

## Recipe (atom `test_ccl`)
Raw RGB → chrominance/edge saliency → morph clean → binary mask → **SAUF CCL** →
`labels` + boxes for components above min area.
