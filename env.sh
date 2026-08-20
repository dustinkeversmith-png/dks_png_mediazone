alias retree="./scripts/re_file_structure_tree/retree.exe"


fullbuild() {
    # 1. Configure the build
    cmake -B build -S .

    # 2. Build Release configuration
    cmake --build build --config Release
}

build() {
    cmake --build build --config Release
}