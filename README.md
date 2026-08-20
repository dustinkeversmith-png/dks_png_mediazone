# dks_png_mediazone
media and stuff


# 1. Configure the build
cmake -B build -S .

# 2. Build Release configuration
cmake --build build --config Release


# 1. Configure
cmake -B build -S . -G "Ninja" -DCMAKE_BUILD_TYPE=Release

# 2. Compile
cmake --build build