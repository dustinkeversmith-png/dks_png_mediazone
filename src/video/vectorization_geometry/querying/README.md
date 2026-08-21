Spatial Querying & Geometrical "Never Re-Querying"Once geometrized into oriented bounding boxes (OBB) or polygons, the layout is organized into spatial structures:[Raw Frame] ──► [CCL / Contours] ──► [RDP Vectorization] ──► [2D BVH / R-Tree]
                                                                    │
                 ┌──────────────────────────────────────────────────┤
                 ▼                                                  ▼
      [Spatial Containment Queries]                    [Shape Hash Tray Lookup]
    (Parent-Child Trees / AABB Trees)                   (Hu / Fourier Descriptors)
Bounding Volume Hierarchies (BVH) & R-TreesStore vectorized objects in a Bounding Volume Hierarchy (BVH) (available natively in unsupported/Eigen/BVH). Queries like "What sub-entities exist inside this window or head?" require no pixel access—only fast AABB interval overlap tests in $O(\log K)$ time.