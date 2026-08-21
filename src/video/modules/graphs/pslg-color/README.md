struct GeometricFace {
    uint32_t id;
    std::vector<uint32_t> half_edge_indices; // Boundary cycle
    
    // Color & Intensity Profile for Classification
    Vec3f mean_lab;              // Interior average color (e.g. brown shirt base)
    Vec3f color_variance;         // Texture / noise variance
    float mean_intensity_sink;   // Value of interior sinks/valleys
    float area;
    Vec2f centroid;
    
    // Structural Metadata
    uint32_t parent_face_id;     // E.g. Pocket belongs inside Shirt Face
    std::vector<uint32_t> child_face_ids; 
};

struct HalfEdge {
    uint32_t origin_vertex;
    uint32_t twin;               // Opposite direction half-edge
    uint32_t next;               // Next half-edge in face cycle
    uint32_t incident_face;      // The face to the left of this edge
    
    // Boundary transition delta
    float hue_delta;             // ΔE / hue distance crossing this line
    float intensity_step;        // Step height of crease/sink
};

That architecture works cleanly if you structure it as a Planar Straight-Line Graph (PSLG) with a Doubly-Connected Edge List (DCEL / Half-Edge structure).

By treating the image as a geometric cell decomposition, you solve your three requirements naturally:

1. Navigable Shapes & Face-Level Color Profiles
If you store edges as directed half-edges with an incident face pointer, every closed polygon (e.g., the front shirt pocket) becomes an explicit face record.

You can store the interior and boundary characteristics directly on that face:

Navigating Features: Crossing edge->twin immediately tells you the adjacent shape and the exact color/hue distance crossed ($\Delta E$).

Boundary edge has high intensity_step (the dark crease sink of the pocket fold).

Geometry = small convex quad nested inside a larger torso bounding box.