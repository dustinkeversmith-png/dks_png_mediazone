cmake_minimum_required(VERSION 3.16)
project(DeterministicVisionEngine LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Header-Only Includes
add_library(vision_deps INTERFACE)
target_include_directories(vision_deps INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/eigen
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/stb
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/fftw
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies
)

# 2. Precompile Eigen Template Headers (Fast Compilation)
target_precompile_headers(vision_deps INTERFACE
    <Eigen/Dense>
    <unsupported/Eigen/BVH>
)

# 3. FFTW3 Linkage
if(WIN32)
    target_link_libraries(vision_deps INTERFACE 
        ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/fftw/libfftw3f-3.lib
    )
else()
    target_link_libraries(vision_deps INTERFACE fftw3f)
endif()

# YACCLAB include path
target_include_directories(your_target PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/YACCLAB/include
)

# libspatialindex include path & link library
target_include_directories(your_target PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/libspatialindex/include
)
target_link_libraries(your_target PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/libspatialindex/build/bin/Release/spatialindex.lib
)

# FFmpeg C APIs (avcodec, avformat, avutil, swscale)
target_include_directories(your_target PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/ffmpeg/include
)
target_link_libraries(your_target PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/ffmpeg/lib/avcodec.lib
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/ffmpeg/lib/avformat.lib
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/ffmpeg/lib/avutil.lib
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/ffmpeg/lib/swscale.lib
)

# 4. Binary Pipeline
add_executable(run_vision_tests src/main_vision_pipeline.cpp)
target_link_libraries(run_vision_tests PRIVATE vision_deps)