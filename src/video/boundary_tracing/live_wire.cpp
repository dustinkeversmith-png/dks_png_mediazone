Intelligent Scissors / Livewire:
Transforms pixels into graph nodes with edge costs derived from gradient magnitude, direction, and Laplacian zero-crossings. Uses Dijkstra’s algorithm to snap dynamically to optimal feature boundaries in real time.


Livewire / Intelligent Scissors:Precalculating the cost graph takes $\mathcal{O}(N)$.Interactively finding paths from a seed pixel uses a priority-queue Dijkstra: $\mathcal{O}(N \log N)$.Nuance: To make interaction fluid at 60 FPS on large images, Dijkstra is computed lazily in a localized bounding box around the cursor rather than globally across all $N$ pixels.