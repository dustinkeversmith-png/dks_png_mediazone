Crease / valley (sink) filter.

Dark folds produce two parallel gradient walls. Collapse them to a single spine:

1. Grayscale black top-hat (`close(I) − I`)
2. LoG valley response
3. 1-D non-maximum suppression along the valley gradient
