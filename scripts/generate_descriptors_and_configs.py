#!/usr/bin/env python3
import json
import sys
from pathlib import Path

# -------------------------------------------------------------------------
# Dataset & Provider Descriptors Metadata
# -------------------------------------------------------------------------
DATASET_DESCRIPTORS = {
    "ade20k": {
        "dataset_desc": {
            "name": "ADE20K (Scene Parsing Benchmark)",
            "data": ".jpgs scene photography (indoor/outdoor contexts)",
            "truths": ".png segmentation masks with class IDs (0-150) and part instance layers",
        },
        "provider_desc": {
            "name": "ade20k",
            "dataset": ".jpgs",
            "ground_truths": "png_masks",
            "renderings": [
                "scene_parsing_overlay()",
                "part_segmentation_overlay()",
                "palette_colormap()",
            ],
        },
    },
    "bsds500": {
        "dataset_desc": {
            "name": "BSDS500 (Berkeley Segmentation Dataset)",
            "data": ".jpgs natural image benchmarks (481x321 landscape/portrait)",
            "truths": ".mat files containing groundTruth cell arrays with segmentations and boundary contours across human annotators",
        },
        "provider_desc": {
            "name": "bsds500",
            "dataset": ".jpgs",
            "ground_truths": "mat_cell_arrays",
            "renderings": [
                "boundary_overlay()",
                "consensus_boundary_map()",
                "segmentation_regions()",
            ],
        },
    },
    "coco": {
        "dataset_desc": {
            "name": "COCO (Common Objects in Context)",
            "data": ".jpgs multi-object complex scenes",
            "truths": ".json format annotations with RLE / polygon boundaries, category ids, and bounding boxes",
        },
        "provider_desc": {
            "name": "coco",
            "dataset": ".jpgs",
            "ground_truths": "json_polygons",
            "renderings": [
                "bounding_box_overlay()",
                "polygon_segmentation_mask()",
                "keypoints_overlay()",
            ],
        },
    },
    "dis5k": {
        "dataset_desc": {
            "name": "DIS5K (Dichotomous Image Segmentation 5K)",
            "data": ".jpgs / .pngs high-resolution complex natural targets",
            "truths": ".png binary masks mapping fine-detail foreground boundaries",
        },
        "provider_desc": {
            "name": "dis5k",
            "dataset": ".pngs",
            "ground_truths": "binary_png_masks",
            "renderings": [
                "alpha_matte_overlay()",
                "boundary_gradient_overlay()",
                "foreground_crop()",
            ],
        },
    },
    "lvis": {
        "dataset_desc": {
            "name": "LVIS (Large Vocabulary Instance Segmentation)",
            "data": ".jpgs detailed object scenes spanning 1200+ entry categories",
            "truths": ".json format cross-referenced bounding boxes, fine boundary contours, and negative classes",
        },
        "provider_desc": {
            "name": "lvis",
            "dataset": ".jpgs",
            "ground_truths": "json_annotations",
            "renderings": [
                "category_instance_overlay()",
                "sparse_mask_render()",
                "hierarchical_boxes()",
            ],
        },
    },
    "sbd": {
        "dataset_desc": {
            "name": "SBD (Semantic Boundaries Dataset)",
            "data": ".jpgs (real life photography) people, objects, planes",
            "truths": ".mats (GTcls) (Semantic Classes) GTinst (for object instances) .Segmentation 2D Integer Matrix of Class IDS, .Boundaries Cell Array Mapping Boundaries",
        },
        "provider_desc": {
            "name": "sbd",
            "dataset": ".jpgs",
            "ground_truths": "maps",
            "renderings": [
                "segmentation_overlay()",
                "boundaries_overlay()",
            ],
        },
    },
}

# -------------------------------------------------------------------------
# Atom Configurations (unique behavior per unit test atom)
# -------------------------------------------------------------------------
ATOM_CONFIGS = {
    # contour
    "contour/dual_contouring": {
        "input_type": "SIGNED_DISTANCE_FIELD",
        "pre_processing_stages": ["Distance Field Inversion", "Zero-Crossing Detection"],
        "artifacts": ["Zero Crossings", "Hermite Data Points", "Dual Contours", "Mesh Topology Evaluation"],
    },
    "contour/laplace_gaussian": {
        "input_type": "GRAYSCALE_IMAGE",
        "pre_processing_stages": ["Gaussian Blur", "Laplacian Convolution"],
        "artifacts": ["LoG Response Map", "Zero Crossing Edges", "Thresholded Extremas"],
    },
    "contour/level_set": {
        "input_type": "RAW_RGB",
        "pre_processing_stages": ["Grayscale", "Distance Map Initialization", "Gradient Vector Field"],
        "artifacts": ["Phi Implicit Surface", "Iterative Interface Steps", "Zero Level Contour", "Convergence Delta"],
    },
    "contour/live_wire": {
        "input_type": "RAW_RGB",
        "pre_processing_stages": ["Grayscale", "Sobel Gradient", "Cost Function Normalization"],
        "artifacts": ["Local Cost Map", "Dijkstra Search Graph", "Shortest Path Wire", "Cumulative Cost Delta"],
    },
    "contour/marching_squares": {
        "input_type": "SCALAR_FIELD",
        "pre_processing_stages": ["Isoline Threshold", "Cell Index Evaluation"],
        "artifacts": ["Cell Topology Masks", "Interpolated Segments", "Closed Isoline Contours"],
    },
    "contour/moore_neighborhood": {
        "input_type": "BINARY_MASK",
        "pre_processing_stages": ["Binary Threshold", "Border Padding"],
        "artifacts": ["Clockwise Tracing Path", "Visited Matrix", "Vector Loop Contours"],
    },
    "contour/rdp": {
        "input_type": "VECTOR_PATH",
        "pre_processing_stages": ["Perpendicular Distance Calculation", "Recursive Path Splitting"],
        "artifacts": ["Key Anchor Vertices", "Simplified Polyline", "Compression Ratio", "Epsilon Deviation Plot"],
    },
    "contour/snakes": {
        "input_type": "RAW_RGB",
        "pre_processing_stages": ["Grayscale", "Image Energy Calculation", "Elastic Matrix Setup"],
        "artifacts": ["Internal Energy Matrix", "External Gradient Force", "Snake Iteration Steps", "Final Converged Contour"],
    },

    # features
    "features/crease_sink": {
        "input_type": "GRAYSCALE_IMAGE",
        "pre_processing_stages": ["Hessian Matrix Construction", "Eigenvalue Decomposition"],
        "artifacts": ["Ridges Mask", "Valley Sinks", "Spine Tophat Overlays", "Evaluation Against SBD"],
    },
    "features/rbf": {
        "input_type": "POINT_CLOUD_2D",
        "pre_processing_stages": ["Basis Function Kernel Setup", "Thin Plate Spline Weight Solver"],
        "artifacts": ["Warp Field Matrix", "Deformed Grid Map", "Spline Control Points"],
    },
    "features/turning_function": {
        "input_type": "VECTOR_CONTOUR",
        "pre_processing_stages": ["Arc Length Normalization", "Cumulative Angle Integration"],
        "artifacts": ["Theta-S Curve Plot", "Cyclic Shift Alignment", "Shape Distance Matrix"],
    },

    # filters
    "filters/bilateral": {
        "input_type": "RAW_RGB",
        "pre_processing_stages": ["Spatial Domain Kernel", "Range Intensity Kernel"],
        "artifacts": ["Domain Distance Weights", "Range Intensity Weights", "Edge-Preserved Output Image"],
    },
    "filters/edge/canny": {
        "input_type": "GRAYSCALE_IMAGE",
        "pre_processing_stages": ["Gaussian Smoothing", "Sobel Gradient Operators", "Non-Maximum Suppression"],
        "artifacts": ["Gradient Magnitude", "Suppressed Thin Edges", "Hysteresis Dual Thresholds"],
    },
    "filters/gvf": {
        "input_type": "GRAYSCALE_IMAGE",
        "pre_processing_stages": ["Edge Map Computation", "Laplacian Diffusion Solver"],
        "artifacts": ["Gradient Vector Field U", "Gradient Vector Field V", "Streamline Vector Overlays"],
    },
    "filters/hue_gradient": {
        "input_type": "RAW_RGB",
        "pre_processing_stages": ["RGB to HSV Conversion", "Circular Hue Difference Operator"],
        "artifacts": ["Circular Difference Map", "Hue Gradient Magnitude", "Chroma Filter Overlays"],
    },
    "filters/lab_color": {
        "input_type": "RAW_RGB",
        "pre_processing_stages": ["sRGB Gamma Linearization", "XYZ Matrix Transform", "CIELAB Mapping"],
        "artifacts": ["Luminance Channel L", "Color Opponent A", "Color Opponent B", "Delta E Heatmap"],
    },
    "filters/morph_clean": {
        "input_type": "BINARY_MASK",
        "pre_processing_stages": ["Structuring Element Kernel Setup", "Erosion and Dilation"],
        "artifacts": ["Morphological Opening", "Morphological Closing", "Isolated Pixel Suppression Map"],
    },
    "filters/sobel": {
        "input_type": "GRAYSCALE_IMAGE",
        "pre_processing_stages": ["Horizontal Kernel Convolution", "Vertical Kernel Convolution"],
        "artifacts": ["Gradient X (Horizontal)", "Gradient Y (Vertical)", "Fused Gradient Magnitude", "Gradient Angle Map"],
    },

    # geometrify
    "geometrify/fourier_descriptors": {
        "input_type": "COMPLEX_CONTOUR",
        "pre_processing_stages": ["Centroid Normalization", "Fast Fourier Transform 1D"],
        "artifacts": ["Fourier Coefficients Spectra", "Harmonic Reconstructions (K=4..32)", "Scale Invariant Descriptors"],
    },
    "geometrify/hu_moments": {
        "input_type": "BINARY_MASK",
        "pre_processing_stages": ["Spatial Moments Integration", "Central Moments Normalization"],
        "artifacts": ["Invariants H1-H7 Summary", "Scale Invariance Delta", "Rotational Invariance Delta"],
    },
    "geometrify/media_axis": {
        "input_type": "BINARY_MASK",
        "pre_processing_stages": ["Distance Transform Computation", "Ridge Local Maxima Extraction"],
        "artifacts": ["Medial Skeleton Map", "Topological Branch Junctions", "Maximal Inscribed Circles"],
    },

    # graphs
    "graphs/graph_cut": {
        "input_type": "RAW_RGB",
        "pre_processing_stages": ["Unary Terminal Weight Setup", "Pairwise Neighbor Smoothness Weights", "Max-Flow Solver"],
        "artifacts": ["Cut Graph Residuals", "Terminal Flow Capacities", "Binarized Partition Mask"],
    },
    "graphs/pslg": {
        "input_type": "SEGMENTED_REGIONS",
        "pre_processing_stages": ["Planar Straight-Line Graph Extraction", "Adjacency Matrix Construction"],
        "artifacts": ["PSLG Vertices & Edges", "Four-Color Theorem Coloring", "Dual Graph Region Overlays"],
    },

    # sdf
    "sdf/8ssedt": {
        "input_type": "BINARY_MASK",
        "pre_processing_stages": ["Forward Sweeping Raster", "Backward Sweeping Raster"],
        "artifacts": ["8-Neighbor Signed Distance Map", "Exact L2 Distance Comparison", "Signed Boundary Field"],
    },
    "sdf/chamfer": {
        "input_type": "BINARY_EDGE_MAP",
        "pre_processing_stages": ["3-4 Chamfer Forward Pass", "3-4 Chamfer Backward Pass"],
        "artifacts": ["Chamfer Distance Transform", "Approximation Error Metric Map"],
    },
    "sdf/edt": {
        "input_type": "BINARY_MASK",
        "pre_processing_stages": ["1D Parabolic Envelope Voronoi Intersection", "Dimension Separation"],
        "artifacts": ["Exact Squared Euclidean Distance", "Signed Distance Transform Map", "Iso-Contour Visualizer"],
    },

    # segmentation
    "segmentation/bbox_auto": {
        "input_type": "BINARY_MASK",
        "pre_processing_stages": ["Coordinate Extrema Search", "Padding Margins Calculation"],
        "artifacts": ["Bounding Box Coordinates", "Cropped Sub-Regions", "Padded Uncrop Overlays"],
    },
    "segmentation/ccl": {
        "input_type": "BINARY_MASK",
        "pre_processing_stages": ["Two-Pass Label Assignment", "Disjoint Set Union-Find Resolution"],
        "artifacts": ["Labeled Components Map", "Connected Bounding Boxes", "Component Area Summary"],
    },
    "segmentation/convex_hull": {
        "input_type": "2D_POINT_SET",
        "pre_processing_stages": ["Monotone Chain Sorting", "Cross Product Convexity Filtering"],
        "artifacts": ["Convex Hull Polygon", "Convexity Defects Depth Map", "Area Enclosure Delta"],
    },
    "segmentation/template_ncc": {
        "input_type": "RAW_RGB",
        "pre_processing_stages": ["Template Mean-Zero Normalization", "Fast Cross Correlation via Frequency Domain"],
        "artifacts": ["Cross-Correlation Score Surface", "Peak Coordinate Matrix", "Detected Box Highlights"],
    },
    "segmentation/watershed": {
        "input_type": "RAW_RGB",
        "preferred_dataset": "ade20k",
        "pre_processing_stages": ["Binary Threshold", "Grayscale", "Gradient Magnitude"],
        "artifacts": ["Binary Threshold", "Gradient Magnitude", "Grayscale", "Basins, Markers, Relief", "Overlay", "Accuracy Evaluation Against the Dataset"],
    },

    # structures
    "structures/2d_r_tree": {
        "input_type": "BOUNDING_BOX_LIST",
        "pre_processing_stages": ["Bounding Box Insertion", "Quadratic Split Optimization"],
        "artifacts": ["R-Tree Hierarchy Bounds", "Spatial Query Raycast Intersections", "Tree Depth Diagnostics"],
    },
    "structures/lookup_tables": {
        "input_type": "LOOKUP_ARRAY",
        "pre_processing_stages": ["Static Table Pre-Computation", "Quantization Level Scaling"],
        "artifacts": ["Geometry LUT Binary Dump", "Interpolation Error Analysis", "Lookup Latency Benchmark"],
    },
    "structures/lsh": {
        "input_type": "HIGH_DIMENSIONAL_DESCRIPTORS",
        "pre_processing_stages": ["Random Hyperplane Generation", "Hash Bucket Indexing"],
        "artifacts": ["Bucket Collision Heatmap", "k-NN Recall Precision Curve", "Bit Hash Signatures"],
    },
    "structures/region_adjacency": {
        "input_type": "SEGMENTED_LABELS",
        "pre_processing_stages": ["Boundary Edge Adjacency Scan", "Topological Half-Edge Construction"],
        "artifacts": ["RAG Graph Topology", "Face Map Boundaries", "Dual Node Adjacency Matrix"],
    },
    "structures/slic": {
        "input_type": "RAW_RGB",
        "pre_processing_stages": ["CIELAB 5D Space Clustering", "Seed Grid Initialization", "K-Means Iteration"],
        "artifacts": ["Superpixel Cluster Labels", "Superpixel Contours Overlay", "Enforce Connectivity Masks"],
    },
    "structures/vp_tree": {
        "input_type": "VANTAGE_POINT_METRIC_SPACE",
        "pre_processing_stages": ["Vantage Point Selection", "Median Distance Partitioning"],
        "artifacts": ["Vantage Hypersphere Partitions", "Spherical Range Query Results", "Search Tree Path Traces"],
    },

    # topology
    "topology/euler": {
        "input_type": "BINARY_GRID",
        "pre_processing_stages": ["Vertex/Edge/Face Quad Counting", "Bitmask Pattern Lookup"],
        "artifacts": ["Bitmask Pattern Matches", "Euler Characteristic (V-E+F)", "Genus / Hole Count Summary"],
    },
}

def write_json_file(path: Path, data: dict, dry_run: bool):
    if dry_run:
        print(f"  [CREATE JSON] {path.as_posix()}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"  [WRITTEN] {path.as_posix()}")

def run_descriptor_and_config_generation(dry_run: bool = True):
    repo_root = Path.cwd()
    data_dir = repo_root / "data" / "vision"
    video_datasets_dir = repo_root / "src" / "video" / "datasets"
    unit_tests_modules_dir = repo_root / "src" / "video" / "tests" / "unit_tests" / "modules" / "atoms"

    mode_label = "DRY RUN (No files created)" if dry_run else "EXECUTING GENERATION"
    print(f"=== {mode_label} ===")

    # 1. Generate Dataset and Provider Descriptors
    print("\n[1/3] Generating dataset.desc and provider.desc in corresponding dataset folders...")
    for ds_key, meta in DATASET_DESCRIPTORS.items():
        # Match case-insensitive folder under data/vision/ (e.g. ADE20K, BSDS500, SDB)
        target_dataset_dir = None
        if data_dir.exists():
            for folder in data_dir.iterdir():
                if folder.is_dir() and folder.name.lower() == ds_key.lower():
                    target_dataset_dir = folder
                    break

        if not target_dataset_dir:
            target_dataset_dir = data_dir / ds_key.upper()

        # Place {dataset_name}.dataset.desc.json in data/vision/<DATASET>/
        dataset_desc_file = target_dataset_dir / f"{ds_key}.dataset.desc.json"
        write_json_file(dataset_desc_file, meta["dataset_desc"], dry_run)

        # Place {dataset_name}.provider.desc.json in src/video/datasets/<dataset_name>/
        provider_subfolder = video_datasets_dir / ds_key
        provider_desc_file = provider_subfolder / f"{ds_key}.provider.desc.json"
        write_json_file(provider_desc_file, meta["provider_desc"], dry_run)

    # 2. Generate Atom Configurations
    print("\n[2/3] Generating unique atom.config.json for every unit test atom...")
    for atom_subpath, config_data in ATOM_CONFIGS.items():
        atom_dir = unit_tests_modules_dir / atom_subpath
        config_file = atom_dir / "atom.config.json"
        write_json_file(config_file, config_data, dry_run)

    # 3. Top-level/isolated tests configuration
    special_atom_configs = {
        unit_tests_modules_dir / "test_template_ncc.atom.config.json": ATOM_CONFIGS["segmentation/template_ncc"]
    }
    for file_path, config_data in special_atom_configs.items():
        write_json_file(file_path, config_data, dry_run)

    if dry_run:
        print("\nDry run completed. Run with `--apply` to generate all files.")
    else:
        print("\nAll descriptors and atom configurations were created.")

if __name__ == "__main__":
    apply_changes = "--apply" in sys.argv
    run_descriptor_and_config_generation(dry_run=not apply_changes)