#!/usr/bin/env python3
"""Wire mission-table atom tests to provider-based loading and update atom.config.json."""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ATOMS = REPO / "src" / "video" / "tests" / "unit_tests" / "modules" / "atoms"

MISSION = {
    "segmentation/watershed": {"preferred_dataset": "ade20k", "max_side": 128},
    "contour/marching_squares": {"preferred_dataset": "synthetic_grids", "max_side": 128},
    "contour/dual_contouring": {"preferred_dataset": "synthetic_grids", "max_side": 128},
    "contour/snakes": {"preferred_dataset": "bsds500", "max_side": 128},
    "contour/level_set": {"preferred_dataset": "bsds500", "max_side": 128},
    "contour/live_wire": {"preferred_dataset": "dis5k", "max_side": 96},
    "segmentation/ccl": {"preferred_dataset": "coco", "max_side": 256},
    "segmentation/template_ncc": {"preferred_dataset": "coco", "max_side": 128},
    "contour/rdp": {"preferred_dataset": "synthetic_silhouettes", "max_side": 128},
    "segmentation/convex_hull": {"preferred_dataset": "synthetic_silhouettes", "max_side": 128},
    "geometrify/fourier_descriptors": {"preferred_dataset": "synthetic_silhouettes", "max_side": 128},
    "geometrify/hu_moments": {"preferred_dataset": "synthetic_silhouettes", "max_side": 128},
    "topology/euler": {"preferred_dataset": "synthetic_glyphs", "max_side": 64},
    "features/turning_function": {"preferred_dataset": "synthetic_leaf", "max_side": 128},
    "structures/2d_r_tree": {"preferred_dataset": "synthetic_boxes", "max_side": 0},
    "structures/vp_tree": {"preferred_dataset": "synthetic_silhouettes", "max_side": 128},
}

LOAD_REPLACEMENT = """        const auto mission = load_mission_samples(cli, argc > 0 ? argv[0] : nullptr, {max_samples}, {max_side});
        provider_samples = std::move(mission.provider_samples);
        samples = std::move(mission.samples);
        std::cout << "loaded " << samples.size() << " samples via " << mission.provider_name << "\\n";
        report.n_inputs = static_cast<int>(samples.size());
        return !samples.empty();"""

OLD_LOAD_PATTERNS = [
    re.compile(
        r"bool load\(const std::string& root, const std::string& dataset, const std::string& sample_filter\) \{.*?return !samples\.empty\(\);\s*\}",
        re.S,
    ),
    re.compile(
        r"bool load\(const AtomCli& cli\) \{.*?return !samples\.empty\(\);\s*\}",
        re.S,
    ),
]


def patch_test(path: Path, max_samples: int, max_side: int) -> bool:
    text = path.read_text(encoding="utf-8")
    orig = text

    if "std::vector<ProviderLoadedSample> provider_samples;" not in text:
        text = text.replace(
            "std::vector<LoadedSample> samples;",
            "std::vector<ProviderLoadedSample> provider_samples;\n    std::vector<LoadedSample> samples;",
            1,
        )

    replacement = LOAD_REPLACEMENT.format(max_samples=max_samples, max_side=max_side)
    for pat in OLD_LOAD_PATTERNS:
        if pat.search(text):
            text = pat.sub(
                f"bool load(const AtomCli& cli, int argc, char** argv) {{\n        print_banner(\"load mission samples\");\n{replacement}\n    }}",
                text,
                count=1,
            )
            break

    text = re.sub(
        r"if \(!atom\.load\(cli\.data_root, cli\.dataset, cli\.sample_filter\)\)",
        "if (!atom.load(cli, argc, argv))",
        text,
    )
    text = re.sub(
        r"if \(!atom\.load\(cli\.data_root, cli\.dataset, cli\.sample_filter,",
        "if (!atom.load(cli, argc, argv)",
        text,
    )

    if text != orig:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_config(config_path: Path, provider: str) -> None:
    if not config_path.exists():
        return
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data["preferred_dataset"] = provider
    config_path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")


def main() -> None:
    changed = 0
    for rel, meta in MISSION.items():
        if rel.endswith(".cpp"):
            test_path = ATOMS / rel
        else:
            test_path = next(ATOMS.glob(f"{rel}/test_*.cpp"), None)
            config_path = ATOMS / rel / "atom.config.json"
            patch_config(config_path, meta["preferred_dataset"])
        if test_path and test_path.exists():
            if patch_test(test_path, 8, meta["max_side"]):
                print(f"patched {test_path.relative_to(REPO)}")
                changed += 1
    print(f"done ({changed} tests patched)")


if __name__ == "__main__":
    main()
