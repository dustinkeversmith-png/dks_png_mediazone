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

cmake --build build --config Release --target test_ade20k_provider test_bsds500_provider test_coco_provider test_dis5k_provider test_lvis_provider test_sbd_provider

.\build\bin\Release\test_dis5k_provider.exe          # render 4 samples
.\build\bin\Release\test_ade20k_provider.exe --list  # enumerate all samples