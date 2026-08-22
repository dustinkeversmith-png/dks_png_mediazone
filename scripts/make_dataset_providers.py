#!/usr/bin/env python3
import sys
from pathlib import Path

# Datasets extracted from the tree
DATASETS = [
    {"name": "ade20k", "class_name": "ADE20KProvider", "domain": "vision"},
    {"name": "bsds500", "class_name": "BSDS500Provider", "domain": "vision"},
    {"name": "coco", "class_name": "COCOProvider", "domain": "vision"},
    {"name": "dis5k", "class_name": "DIS5KProvider", "domain": "vision"},
    {"name": "lvis", "class_name": "LVISProvider", "domain": "vision"},
    {"name": "sbd", "class_name": "SBDProvider", "domain": "vision"},
]

def generate_header_content(class_name: str, dataset_name: str) -> str:
    guard_name = f"{dataset_name.upper()}_PROVIDER_HPP"
    return f"""#ifndef {guard_name}
#define {guard_name}

#include <string>
#include <vector>
#include <memory>
#include <filesystem>

namespace datasets {{

class {class_name} {{
public:
    explicit {class_name}(const std::filesystem::path& root_dir)
        : root_dir_(root_dir) {{}}

    virtual ~{class_name}() = default;

    virtual bool initialize() = 0;
    virtual size_t size() const = 0;

protected:
    std::filesystem::path root_dir_;
}};

}} // namespace datasets

#endif // {guard_name}
"""

def setup_dataset_structure(dry_run: bool = True):
    repo_root = Path.cwd()
    video_dir = repo_root / "src" / "video"
    datasets_src_dir = video_dir / "datasets"
    unit_tests_dir = video_dir / "tests" / "unit_tests"
    dataset_atoms_dir = unit_tests_dir / "datasets" / "atoms"
    module_atoms_dir = unit_tests_dir / "modules" / "atoms"

    if not video_dir.exists():
        print(f"Error: Could not locate 'src/video' from '{repo_root}'.")
        print("Make sure you are executing from the repository root.")
        sys.exit(1)

    mode_label = "DRY RUN" if dry_run else "EXECUTING"
    print(f"=== {mode_label}: Setting up Dataset Providers & Test Artifacts ===\n")

    # 1. Create provider .hpp files in src/video/datasets/
    print("[1/3] Setting up dataset provider headers...")
    for ds in DATASETS:
        header_path = datasets_src_dir / f"{ds['name']}_provider.hpp"
        if dry_run:
            print(f"  [CREATE HEADER] {header_path.relative_to(repo_root)}")
        else:
            datasets_src_dir.mkdir(parents=True, exist_ok=True)
            if not header_path.exists():
                header_path.write_text(generate_header_content(ds["class_name"], ds["name"]))
                print(f"  [WRITTEN] {header_path.relative_to(repo_root)}")
            else:
                print(f"  [EXISTS] {header_path.relative_to(repo_root)}")

    # 2. Create folders + /artifacts in unit_tests/datasets/atoms/
    print("\n[2/3] Setting up unit_tests dataset atom folders & artifacts...")
    for ds in DATASETS:
        provider_atom_dir = dataset_atoms_dir / f"unit_{ds['name']}"
        artifacts_dir = provider_atom_dir / "artifacts"
        if dry_run:
            print(f"  [MKDIR] {artifacts_dir.relative_to(repo_root)}")
        else:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            print(f"  [CREATED] {artifacts_dir.relative_to(repo_root)}")

    # 3. Ensure an empty /artifacts directory exists in every unit_test atom directory
    print("\n[3/3] Ensuring /artifacts exist across all module unit test directories...")
    if module_atoms_dir.exists():
        # Iterate over all category directories (contour, features, filters, etc.)
        for category_dir in [d for d in module_atoms_dir.iterdir() if d.is_dir() and d.name != "artifacts"]:
            for atom_dir in [d for d in category_dir.iterdir() if d.is_dir()]:
                artifacts_path = atom_dir / "artifacts"
                if dry_run:
                    print(f"  [CHECK/MKDIR] {artifacts_path.relative_to(repo_root)}")
                else:
                    artifacts_path.mkdir(parents=True, exist_ok=True)
                    print(f"  [READY] {artifacts_path.relative_to(repo_root)}")

    if dry_run:
        print("\nDry run completed. Run with `--apply` to create the files and directories.")
    else:
        print("\nAll provider interfaces, dataset atoms, and artifacts folders have been set up.")

if __name__ == "__main__":
    apply_changes = "--apply" in sys.argv
    setup_dataset_structure(dry_run=not apply_changes)