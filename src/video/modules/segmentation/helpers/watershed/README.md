# Watershed (helper)

**Role:** segmentation *helper* — splits a **touching multi-object binary blob** into
instance basins. Not a photo→mask segmenter; feed a clean FG mask (from CCL / morph / GT).

## Location
`src/video/modules/segmentation/helpers/watershed/`

## API (`contour::Watershed`)
- `from_mask(binary)` — EDT maxima markers → Meyer flood on `-EDT`
- `from_gradient(image)` — Sobel-magnitude relief from local minima
- `flood(relief, markers)` — generic Meyer pass

## Used by
- `object_proposals` / CCL demos when large blobs need instance split
- Optional pre-step before `bbox_auto` when multiple objects touch

## Recipe (atom `test_watershed`)
Raw RGB → clean binary of touching objects → EDT → marker seeds → flood →
1-px watershed lines between instances.
