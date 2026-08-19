# retree

Fast compiled directory tree printer with `.ignore` support.

## Build

From this directory, configure and build with CMake:

```powershell
cmake -S . -B build
cmake --build build --config Release
```

The executable is `build\retree.exe` with single-config generators, or `build\Release\retree.exe` with Visual Studio generators.

Alternatively, use a C++17 compiler directly:

```powershell
g++ -std=c++17 -O3 -DNDEBUG scripts/re_file_structure_tree/retree.cpp -o scripts/re_file_structure_tree/retree.exe
```

On Linux or macOS, omit the `.exe` suffix.

## Use

```powershell
scripts/re_file_structure_tree/retree.exe
scripts/re_file_structure_tree/retree.exe path\to\directory
```

The program reads a `.ignore` file at the root being printed. Rules support comments, `*`, `?`, `**`, directory-only patterns ending in `/`, path patterns, and `!` negation. Rules are applied in order, like `.gitignore`.

Standard defaults ignore source-control metadata, editor folders, dependency trees (`node_modules`, `vendor`, `site-packages`), Python environments and caches, build folders (`build`, `dist`, `out`, `target`, `bin`, `obj`, `lib`), framework caches, compiled binaries, logs, and common generated artifacts. Defaults include Windows case variants such as `Lib`, `Include`, `Build`, and `Scripts`.

```powershell
# Disable standard defaults
retree.exe --no-defaults

# Add a command-line rule (repeatable)
retree.exe --ignore "*.tmp"
```