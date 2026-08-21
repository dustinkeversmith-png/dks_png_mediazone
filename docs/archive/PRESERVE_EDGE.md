[Raw Image: RGB / Grayscale]
                                │
        ┌───────────────────────┴────────────────────────┐
        ▼                                                ▼
 1. [Global Region Segmenter]                   2. [Feature Edge Detector]
    (Graph Cut / SAM / Level Set)                  (Bilateral + Canny / Sobel / LoG)
    Extracts outer domain mask (0 / 1)             Extracts all interior gradients
        │                                                │
        └───────────────────────┬────────────────────────┘
                                │
                                ▼
 3. [Masked Internal Gradient Field]
    $E(x, y) = |\nabla I(x, y)| \cdot \text{Mask}(x,y)$
    (Zeroes out noisy background, preserves all internal details)
                                │
        ┌───────────────────────┴────────────────────────┐
        ▼                                                ▼
 4a. [Topological Contour Extraction]            4b. [Vector Optimization]
     - Suzuki-Abe (Hierarchical Poly Tree)           - Multi-seed Snakes on $E(x,y)$
     - Multi-threshold Marching Squares              - Dual Contouring on multi-material grid
                                │
                                ▼
                 [Hierarchical Vector Output]
                 - Outer Hull (Bag)
                 └── Child Holes / Geometry (Pockets, Buttons)

Do not binarize early. Keep the full-resolution color/intensity data for all gradient calculations.

Use the segmentation mask purely as an attention gate (alpha mask) to kill background clutter while leaving interior pixel values intact.

Feed the gated gradient field into topological or multi-seed vectorizers (Suzuki-Abe, Multi-phase Level Sets, or Multi-seed Snakes) so interior zero-crossings and edges are resolved as nested geometric children.