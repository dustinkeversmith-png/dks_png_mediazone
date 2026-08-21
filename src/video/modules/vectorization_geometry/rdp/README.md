Ramer–Douglas–Peucker (RDP) Polyline Simplification:


Takes raw contour point chains and collapses collinear points based on an error threshold $\epsilon$. A 2,000-pixel perimeter is compressed into an 8- to 16-point convex/concave polygon in $O(N \log N)$.