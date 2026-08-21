Beyond standard raster masks and bounding boxes, several specialized computer vision datasets provide parametric edge geometries, topological graphs, surface textures, and 3D geometric surface normals:

---

### 1. Geometric Edges, Line Segments & Wireframes

Instead of binary pixel edges, these datasets vectorize scenes into parametric geometric primitives (junctions and line segments $[x_1, y_1, x_2, y_2]$).

* **ShanghaiTech Wireframe Dataset (Huang et al., CVPR 2018):** Contains 5,462 indoor/outdoor images annotated with vector wireframes. Ground truth consists of:
* **Junctions:** Exact float coordinate pairs $V = \{(x, y)\}$.
* **Line Segments:** Adjacency pairs $E = \{(v_i, v_j)\}$ defining structural lines (wall intersections, architectural edges).


* **YorkUrban:** 102 calibrated urban images annotated with precise line segments classified along the three orthogonal Manhattan frame vanishing directions.
* **Holicity / Structured3D:** Large-scale synthetic and urban datasets providing full CAD wireframes, polygon face planar layouts, and vanishing point vector lines aligned with RGB frames.

---

### 2. Graph Topology & Network Connectivity

These datasets provide graph ground truths $G = (V, E)$ to evaluate network connectivity, shortest path routing, and topological validity.

* **SpaceNet 3 & 5 (Road Network Extraction):** High-resolution satellite imagery paired with road network graphs. Ground truths are GeoJSON/Shapefiles containing:
* **Centerline Edges:** Linestrings representing physical road paths.
* **Graph Topology:** Connected node junctions with edge attributes (travel time, lane counts, surface type). Evaluated using topological metrics like APLS (Average Path Length Similarity).


* **SketchGraphs:** Over 15 million parametric CAD sketches from Onshape. Annotations represent sketches as geometric constraint graphs (nodes = lines/arcs; edges = topological constraints like *parallel, perpendicular, tangent, coincident, concentric*).

---

### 3. Textures, Color Patterns & Materials

Datasets that label surface material properties and human-perceptible texture descriptions rather than discrete object boundaries.

* **DTD (Describable Textures Dataset):** 5,640 images annotated across **47 perceptual texture attributes** (e.g., *braided, checkered, cobwebbed, flecked, honeycombed, perforated, swirly, wrinkled*).
* **MINC (Materials in Context):** ~3 million point annotations labeling material composition at distinct surface locations (e.g., *fabric, ceramic, leather, polished metal, wood, glass, mirror*).
* **KTH-TIPS2 / Kylberg:** High-resolution texture benchmarks capturing controlled surface patterns under multiple scales, rotations, and illumination angles.

---

### 4. Surface Normals & Geometric Planarity

These datasets map every pixel to a 3D geometric orientation vector rather than a semantic ID.

* **NYU Depth V2 / ScanNet:** RGB-D datasets providing dense **Surface Normal maps** ($[n_x, n_y, n_z] \in [-1, 1]$), mapping the 3D surface orientation facing the camera per pixel.
* **ScanNet / Matterport3D (PlaneRCNN / PlaneNet format):** Segmentations where each surface is represented as a 3D planar equation $n^T X + d = 0$, giving exact geometric plane parameters for floors, walls, and tabletops.

---

### Summary of Data Representations

| Task / Domain | Key Datasets | Ground Truth Format | Target Representation |
| --- | --- | --- | --- |
| **Wireframes / Lines** | ShanghaiTech Wireframe, YorkUrban | `.json` / `.mat` coordinate lists | Line segments $[x_1, y_1, x_2, y_2]$ & junction graphs |
| **Graph Topology** | SpaceNet 3/5, SketchGraphs | `.geojson`, graph edge-lists | $G=(V, E)$ connectivity & parametric constraints |
| **Texture & Patterns** | DTD, MINC, Kylberg | Image class folders, point `.json` | 47 texture descriptors, material classification |
| **Surface Geometry** | NYUv2, Structured3D, Matterport3D | 3-channel float/PNG, plane arrays | Surface normal vectors $[n_x, n_y, n_z]$, 3D planes |