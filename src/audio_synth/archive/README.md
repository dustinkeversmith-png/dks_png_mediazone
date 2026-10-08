# Archived speech experiments

These files are preserved for reference and are not part of the active CMake
build. Production work is concentrated on explicit neural inference and the
Piper reference. Matcha was removed rather than archived, following the user's
explicit request.

| Location | Preserved material |
| --- | --- |
| `dsp_paradigms/` | Six DSP/source-filter demo families |
| `homebrew_neural/` | Two deterministic, untrained neural-shaped baselines |
| `app/` | Shared demo CLI implementation, header and tiny runner entry point |
| `tests/baseline_checks.cpp.txt` | Legacy model assertions extracted from the mixed framework test |
| `notes/` | DSP, unit-selection, acoustic-pipeline and neural sketch notes |
| `scratch/` | Combined C++/header snapshots, moved without rewriting their contents |

There is intentionally no archive build option. Original relative include paths
are retained as historical source references; a deliberate restoration would
need its own build configuration. Historical notes may include speculative
speed or quality claims and do not describe the active implementation.

All seven `src/*.cpp` files remain active framework code, so none were moved.
The current metrics, prosody, runtime and CLI tests remain relevant. Generated
build caches and historical audio artifacts are outside the source build graph.
