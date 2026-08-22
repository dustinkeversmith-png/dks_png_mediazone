Constructing the 2D Spatial Tree & Planar Arrangement in $\mathcal{O}(E \log E)$$\mathcal{O}(E \log E)$ via the Bentley–Ottmann sweep-line algorithm is very fast on modern CPUs (a few milliseconds for tens of thousands of line segments).[Raw Lines / Vectors]
         │
         ▼
 1. [Bentley-Ottmann Sweep-Line] ──► Finds all line intersections & splits segments
         │
         ▼
 2. [DCEL Graph Assembly]       ──► Forms closed cycles (Faces) + stores Hue ΔE on edges
         │
         ▼
 3. [2D BVH / Quadtree Index]    ──► Spatial acceleration for point/box classification queries
         │
         ▼
 4. [Raster Integration Pass]    ──► Accumulates mean Lab color & sink metrics per Face
Sweep-Line Intersections: Intersects all detected hue/crease segments into non-overlapping atomic segments.Build DCEL: Sort outgoing edges around each vertex by angle ($\text{atan2}$) to link next pointers and close polygons.Build 2D Spatial Tree (BVH / R-Tree): Index face bounding boxes and edge segments into a bounding volume hierarchy (or 2D segment tree) for instant $\mathcal{O}(\log F)$ spatial lookups.Sample Color into Faces: Run a fast scanline rasterizer or uniform point-sample across the polygon interior to populate mean_lab and color_variance.