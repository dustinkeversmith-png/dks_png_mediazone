Watershed segmentation (Beucher / Meyer flooding).

Treats a relief (usually −EDT or gradient magnitude) as topography. Markers are flooded
with a priority queue; unlabeled pixels where basins meet are watershed lines.

- `from_mask` — marker-controlled split of a binary blob via EDT maxima
- `from_gradient` — flood Sobel magnitude from local minima
- `flood(relief, markers)` — generic Meyer pass
