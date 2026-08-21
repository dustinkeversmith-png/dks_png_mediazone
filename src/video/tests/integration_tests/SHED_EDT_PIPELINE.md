**Correct.** None of these algorithms—on their own—perform **instance segmentation** (identifying and separating multiple distinct objects as independent entities).

Here is why they don't do it and where they fall short:

---

### What Each Algorithm Actually Does With Multiple Objects

* **Marching Squares & Dual Contouring:**
If two distinct objects touch by even a single pixel or share the same zero-level isovalue in the SDF, they will extract one giant, merged polygon contour or a single non-manifold mesh. They have no concept of "Object A" vs. "Object B"—only "Inside" vs. "Outside".
* **GraphCut (Standard 2-Terminal $s$-$t$ cut):**
It is strictly **binary** ($0 = \text{Background}$, $1 = \text{Foreground}$). If there are 5 people inside the bounding box, it cuts out all 5 as a single connected white blob on a black background.
* **Snakes (Parametric Active Contours):**
A single snake is a single closed topological loop ($v_0 \dots v_M$). It physically **cannot split** to wrap around two separate objects; it will either stretch awkwardly across the gap between them or collapse into one.
* **Livewire:**
It only follows a 1D path between two specific points you clicked.

---

### How Multiple / Touching Objects Are Actually Separated

To go from a raw image or binary mask to **clean, separated individual instances**, you have to insert an explicit separation layer into the pipeline:

```
[Binary Mask / Cluttered Foreground]
                 │
                 ├──► 1. Connected Component Labeling (CCL) ──► (If objects do NOT touch)
                 │
                 ├──► 2. Distance Transform + Watershed ─────► (If objects overlap/touch)
                 │
                 └──► 3. Multi-Label Alpha-Expansion GraphCut ─► (For N distinct semantic classes)

```

#### 1. Disjoint / Non-Touching Objects: Connected Component Labeling (CCL)

* **How it works:** A single two-pass scan (using a Disjoint-Set / Union-Find structure) assigns a unique integer ID ($1, 2, 3, \dots, K$) to every isolated 8-connected island of pixels.
* **Result:** You run Marching Squares / Dual Contouring on each labeled mask independently, giving you distinct vector paths for each object.

#### 2. Touching / Overlapping Objects: Marker-Controlled Watershed

* **When to use:** Objects overlap or touch (e.g., coins on a table, touching cells, overlapping UI icons).
* **Pipeline:**
1. Compute the Euclidean Distance Transform ($EDT$) on the binary mask.
2. Find local maxima in the distance field to identify the **centers / cores** of each individual object (the "markers").
3. Flood the inverted distance landscape from these markers (the **Watershed Transform**).
4. The "ridges" where the flooding basins collide form the exact separation boundaries (cut lines) between touching objects.



#### 3. Multi-Class Separation: Multi-Label Graph Cuts ($\alpha$-Expansion)

* Instead of a single source-sink min-cut, it solves an energy minimization over $K$ discrete labels simultaneously using a sequence of standard Boykov-Kolmogorov max-flow cuts ($\alpha$-expansion).

---

### The Complete End-to-End Pipeline

To handle multi-object scenes completely from raw image to isolated vectors:

```
[Raw Scene / Image]
        │
        ▼
1. [GraphCut / Mask Generator]    ──► Binary Foreground Mask (All objects merged)
        │
        ▼
2. [Distance Transform + Watershed] ──► Splits touching entities into Instance IDs [1, 2, ..., K]
        │
        ▼
3. [Per-Instance 8SSEDT]           ──► Generates individual SDFs per object ID
        │
        ▼
4. [Dual Contouring / Marching Sq] ──► Extracts separate, clean vector polygons for each instance

```