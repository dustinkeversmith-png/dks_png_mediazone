**Step 1: Grid-Based Center Seeding**

* Divide the canvas into a uniform grid with spacing $S = \sqrt{(W \times H) / K}$, where $K$ is the desired superpixel count.
* Sample initial cluster centers at each grid coordinate, capturing both spatial positions $(x, y)$ and normalized color coordinates $(R, G, B)$ or $(L, a, b)$.

**Step 2: Localized k-Means Distance Assignment**

* For each cluster center, restrict the search region to a local $2S \times 2S$ bounding box rather than the whole image.
* Compute the combined 5D distance $D = d_{\text{color}} + \frac{m^2}{S^2} d_{\text{spatial}}$, where compactness parameter $m$ balances color fidelity against geometric boundary smoothness.
* Assign each pixel within the window to the cluster yielding the lowest $D$.

**Step 3: Centroid Recalculation & Convergence**

* Recompute every cluster center by taking the average spatial coordinates and mean color channels of all constituent pixels.
* Repeat the local search and center update passes for 5 to 10 iterations until pixel assignments stabilize.
* Enforce connectivity with a single connected-component pass to eliminate stray orphan pixels.

**Step 4: RAG Construction & Feature Aggregation**

* Traverse the superpixel label map using a 4-connected sliding window ($[x+1, y]$ and $[x, y+1]$).
* For every adjacent pixel pair with different labels $(u, v)$, insert an undirected edge between nodes $u$ and $v$ in an adjacency list.
* Accumulate pixel area, mean color vectors, and normalized color histograms (e.g., 24-bin RGB) for each active region node.

**Step 5: Edge Weight Computation & Priority Queue Initialization**

* Evaluate a composite similarity metric for every adjacent pair $(u, v)$, typically weighting histogram intersection $S_{\text{color}}$ and size compatibility $S_{\text{size}} = 1 - \frac{\text{area}_u + \text{area}_v}{\text{total\_pixels}}$.
* Invert the similarity into a scalar distance cost: $\text{cost} = 1.0 - (\alpha S_{\text{color}} + \beta S_{\text{size}})$.
* Push all graph edges into a min-priority queue ordered by lowest cost.

**Step 6: Iterative Graph Contraction**

* Pop the minimum-cost edge $(u, v)$ from the priority queue and resolve their current canonical roots using a Disjoint-Set (Union-Find) structure.
* Check termination guards: halt if the cost exceeds a dissimilarity threshold or the remaining active region count reaches the target.
* Merge region $v$ into $u$:
* Mark $v$ inactive and set its union-find parent to $u$.
* Update node $u$'s area, color means, and histograms using an area-weighted average.
* Union the neighbor lists of $u$ and $v$, removing duplicate self-loops.
* Recalculate edge weights between $u$ and its updated neighbors, inserting the new costs into the priority queue.



**Step 7: Final Relabeling & Mask Generation**

* Run path compression across the union-find structure to map every original superpixel ID to its final consolidated region root.
* Re-index surviving roots to dense contiguous integer IDs ($1 \dots M$).
* Remap the per-pixel label buffer to the new dense IDs to produce the segmented output map.