# Vision geometry / screen-detection unit tests.
# Included from the repository root CMakeLists.txt.

set(VISION_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}/data/vision/data_vision" VISION_DATA_DIR_NORM)

add_library(vision_core STATIC
    ${VISION_SRC_DIR}/math/vision_io.cpp
)
target_include_directories(vision_core PUBLIC
    ${VISION_SRC_DIR}
    ${VISION_SRC_DIR}/math
    ${VISION_SRC_DIR}/unit_tests
    ${CMAKE_SOURCE_DIR}/dependencies/stb
    ${CMAKE_SOURCE_DIR}/dependencies/libspatialindex/include
)
target_compile_definitions(vision_core PUBLIC
    VISION_DATA_DIR="${VISION_DATA_DIR_NORM}"
    NOMINMAX
)
target_link_libraries(vision_core PUBLIC eigen_lib fftw3f_lib)
if(FFTW_IMPORT_TARGET)
    add_dependencies(vision_core ${FFTW_IMPORT_TARGET})
endif()

function(add_vision_unit_test test_name)
    add_executable(${test_name} ${VISION_SRC_DIR}/unit_tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE vision_core)
    if(WIN32 AND DEFINED FFTW_DLL)
        add_custom_command(TARGET ${test_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FFTW_DLL}"
                "$<TARGET_FILE_DIR:${test_name}>"
            COMMENT "Copying FFTW3f runtime DLL for ${test_name}"
        )
    endif()
endfunction()

add_vision_unit_test(test_boundary_tracing)
add_vision_unit_test(test_fourier_descriptors)
add_vision_unit_test(test_hu_moments)
add_vision_unit_test(test_rbf)
add_vision_unit_test(test_lookup_tables)
add_vision_unit_test(test_ccl)
add_vision_unit_test(test_2d_r_tree)
add_vision_unit_test(test_lsh)
add_vision_unit_test(test_vp_tree)
add_vision_unit_test(test_convex_hull)
add_vision_unit_test(test_euler)
add_vision_unit_test(test_turning_function)
add_vision_unit_test(test_medial_axis)
add_vision_unit_test(test_rdp)
add_vision_unit_test(test_integration_alpha)
add_vision_unit_test(test_integration_beta)
add_vision_unit_test(test_integration_gamma)

# Standalone C++20 contour suite: stdlib only (no Eigen / FFTW / OpenCV / stb).
file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}/data/vision/benchmarks" VISION_BENCH_DIR_NORM)
add_library(contour_kit INTERFACE)
target_include_directories(contour_kit INTERFACE ${VISION_SRC_DIR})
target_compile_features(contour_kit INTERFACE cxx_std_20)
target_compile_definitions(contour_kit INTERFACE
    VISION_BENCH_DIR="${VISION_BENCH_DIR_NORM}"
    NOMINMAX
)

function(add_contour_unit_test test_name)
    add_executable(${test_name} ${VISION_SRC_DIR}/unit_tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE contour_kit)
    set_property(TARGET ${test_name} PROPERTY CXX_STANDARD 20)
endfunction()

add_contour_unit_test(test_contour_invariants)
add_contour_unit_test(test_contour_benchmarks)

add_library(contour_impl OBJECT
    ${VISION_SRC_DIR}/boundary_tracing/dual_contouring.cpp
    ${VISION_SRC_DIR}/boundary_tracing/graph_cut.cpp
    ${VISION_SRC_DIR}/boundary_tracing/level_set.cpp
    ${VISION_SRC_DIR}/boundary_tracing/live_wire.cpp
    ${VISION_SRC_DIR}/boundary_tracing/snakes.cpp
    ${VISION_SRC_DIR}/featurizations/sdf/chamfer.cpp
    ${VISION_SRC_DIR}/filters/sobel.cpp
)
target_link_libraries(contour_impl PUBLIC contour_kit)
target_include_directories(contour_impl PUBLIC ${VISION_SRC_DIR})
set_property(TARGET contour_impl PROPERTY CXX_STANDARD 20)

# Optional FFmpeg luma decoder sample (not a unit test).
set(FFMPEG_ROOT "${CMAKE_SOURCE_DIR}/dependencies/ffmpeg")
if(EXISTS "${FFMPEG_ROOT}/include/libavcodec/avcodec.h")
    add_executable(ffmpeg_luma_sample ${VISION_SRC_DIR}/libffmepg.cpp)
    target_include_directories(ffmpeg_luma_sample PRIVATE "${FFMPEG_ROOT}/include")
    target_link_libraries(ffmpeg_luma_sample PRIVATE
        "${FFMPEG_ROOT}/lib/avcodec.lib"
        "${FFMPEG_ROOT}/lib/avformat.lib"
        "${FFMPEG_ROOT}/lib/avutil.lib"
        "${FFMPEG_ROOT}/lib/swscale.lib"
    )
    if(WIN32)
        add_custom_command(TARGET ffmpeg_luma_sample POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FFMPEG_ROOT}/bin/avcodec-63.dll"
                "${FFMPEG_ROOT}/bin/avformat-63.dll"
                "${FFMPEG_ROOT}/bin/avutil-61.dll"
                "${FFMPEG_ROOT}/bin/swscale-10.dll"
                "${FFMPEG_ROOT}/bin/swresample-7.dll"
                "$<TARGET_FILE_DIR:ffmpeg_luma_sample>"
            COMMENT "Copying FFmpeg runtime DLLs"
        )
    endif()
endif()
