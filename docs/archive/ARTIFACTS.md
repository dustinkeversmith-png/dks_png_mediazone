Artifacts now land next to each suite, and atom packs are built from real DIS photos.

### What changed
- **CMake** points at `modules/` + `tests/`, with compile-time artifact roots:
  - atoms → `src/video/tests/unit_tests/atoms/artifacts/<suite>/`
  - contour → `.../benchmarks/contour/artifacts/`
  - integration → `.../integration_tests/artifacts/`
- **Atom CLI** on every atom/integration binary: `--data`, `--dataset`, `--sample`, `--artifacts`, `--list`, `--help`
- **Stdout** prints data root, dataset, sample filter, artifact path, per-case lines, and accuracy summary
- **Real-photo atoms** from DIS via `python data/vision/prepare_atom_datasets.py` → `data/vision/atoms/` (archived synthetics unused)
- Contour benches use module includes and local artifacts (no top-level `artifacts/`)

### Verified
- Builds: atoms, integration, `test_contour_invariants`
- CCL 36/36 with per-screen `screen_*_boxes.pgm` + `summary.tsv`
- Hu moments `--sample dis5k_val_0002` → local artifacts
- Contour invariants write under `benchmarks/contour/artifacts/contour_invariants/`

Example:
```powershell
python data/vision/prepare_atom_datasets.py
.\build\bin\Release\test_hu_moments.exe --sample dis5k_val_0002
.\build\bin\Release\test_hu_moments.exe --sample dis5k_val_0002

.\build\bin\Release\test_ccl.exe
.\build\bin\Release\test_contour_invariants.exe
```