# Vision modules + tests. Included from the repository root CMakeLists.txt.

set(VISION_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(VISION_MODULES_DIR ${VISION_SRC_DIR}/modules)
set(VISION_TESTS_DIR ${VISION_SRC_DIR}/tests)
set(VISION_ATOMS_DIR ${VISION_TESTS_DIR}/unit_tests/atoms)
set(VISION_BENCH_CONTOUR_DIR ${VISION_TESTS_DIR}/unit_tests/benchmarks/contour)
set(VISION_INTEGRATION_DIR ${VISION_TESTS_DIR}/integration_tests)

file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}/data/vision/atoms" VISION_DATA_DIR_NORM)
file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}/data/vision/benchmarks" VISION_BENCH_DIR_NORM)
file(TO_CMAKE_PATH "${VISION_ATOMS_DIR}/artifacts" VISION_ATOM_ARTIFACT_ROOT_NORM)
file(TO_CMAKE_PATH "${VISION_BENCH_CONTOUR_DIR}/artifacts" VISION_CONTOUR_ARTIFACT_ROOT_NORM)
file(TO_CMAKE_PATH "${VISION_INTEGRATION_DIR}/artifacts" VISION_INTEGRATION_ARTIFACT_ROOT_NORM)

# ---------------------------------------------------------------------------
# vision_core: Eigen / FFTW / stb path used by atom + integration suites
# ---------------------------------------------------------------------------
add_library(vision_core STATIC
    ${VISION_MODULES_DIR}/math/vision_io.cpp
)
target_include_directories(vision_core PUBLIC
    ${VISION_MODULES_DIR}
    ${VISION_MODULES_DIR}/math
    ${VISION_TESTS_DIR}/unit_tests
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

function(vision_copy_fftw target_name)
    if(WIN32 AND DEFINED FFTW_DLL)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FFTW_DLL}"
                "$<TARGET_FILE_DIR:${target_name}>"
            COMMENT "Copying FFTW3f runtime DLL for ${target_name}"
        )
    endif()
endfunction()

function(add_vision_atom_test test_name)
    add_executable(${test_name} ${VISION_ATOMS_DIR}/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE vision_core)
    target_compile_definitions(${test_name} PRIVATE
        VISION_TEST_ARTIFACT_ROOT="${VISION_ATOM_ARTIFACT_ROOT_NORM}"
        VISION_DEFAULT_DATASET="${test_name}"
    )
    vision_copy_fftw(${test_name})
endfunction()

function(add_vision_integration_test test_name)
    add_executable(${test_name} ${VISION_INTEGRATION_DIR}/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE vision_core)
    target_compile_definitions(${test_name} PRIVATE
        VISION_TEST_ARTIFACT_ROOT="${VISION_INTEGRATION_ARTIFACT_ROOT_NORM}"
    )
    vision_copy_fftw(${test_name})
endfunction()

add_vision_atom_test(test_fourier_descriptors)
add_vision_atom_test(test_hu_moments)
add_vision_atom_test(test_rbf)
add_vision_atom_test(test_lookup_tables)
add_vision_atom_test(test_ccl)
add_vision_atom_test(test_2d_r_tree)
add_vision_atom_test(test_lsh)
add_vision_atom_test(test_vp_tree)
add_vision_atom_test(test_convex_hull)
add_vision_atom_test(test_euler)
add_vision_atom_test(test_turning_function)
add_vision_atom_test(test_medial_axis)
add_vision_atom_test(test_rdp)
add_vision_atom_test(test_moore_neighbor)
add_vision_atom_test(test_template_ncc)

add_vision_integration_test(test_integration_alpha)
add_vision_integration_test(test_integration_beta)
add_vision_integration_test(test_integration_gamma)

# ---------------------------------------------------------------------------
# contour_kit: stdlib-only C++20 contour algorithms + benchmarks
# ---------------------------------------------------------------------------
add_library(contour_kit INTERFACE)
target_include_directories(contour_kit INTERFACE ${VISION_MODULES_DIR})
target_compile_features(contour_kit INTERFACE cxx_std_20)
target_compile_definitions(contour_kit INTERFACE
    VISION_BENCH_DIR="${VISION_BENCH_DIR_NORM}"
    NOMINMAX
)

function(add_vision_contour_atom_test test_name)
    add_vision_atom_test(${test_name})
    target_link_libraries(${test_name} PRIVATE contour_kit)
    set_property(TARGET ${test_name} PROPERTY CXX_STANDARD 20)
endfunction()

add_vision_contour_atom_test(test_bbox_auto)
add_vision_contour_atom_test(test_snakes)
add_vision_contour_atom_test(test_marching_squares)
add_vision_contour_atom_test(test_dual_contouring)
add_vision_contour_atom_test(test_level_set)
add_vision_contour_atom_test(test_live_wire)
add_vision_contour_atom_test(test_graph_cut)
add_vision_contour_atom_test(test_lab_color)
add_vision_contour_atom_test(test_canny)
add_vision_contour_atom_test(test_laplace_gaussian)
add_vision_contour_atom_test(test_chamfer)
add_vision_contour_atom_test(test_8ssedt)
add_vision_contour_atom_test(test_sobel)
add_vision_contour_atom_test(test_bilateral)
add_vision_contour_atom_test(test_morph_clean)
add_vision_contour_atom_test(test_gvf)

function(add_contour_bench_test test_name)
    add_executable(${test_name} ${VISION_BENCH_CONTOUR_DIR}/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE contour_kit)
    target_compile_definitions(${test_name} PRIVATE
        VISION_TEST_ARTIFACT_ROOT="${VISION_CONTOUR_ARTIFACT_ROOT_NORM}"
    )
    set_property(TARGET ${test_name} PROPERTY CXX_STANDARD 20)
endfunction()

add_contour_bench_test(test_contour_invariants)
add_contour_bench_test(test_contour_benchmarks)

add_library(contour_impl OBJECT
    ${VISION_MODULES_DIR}/boundary_tracing/dual_contouring.cpp
    ${VISION_MODULES_DIR}/boundary_tracing/graph_cut.cpp
    ${VISION_MODULES_DIR}/boundary_tracing/level_set.cpp
    ${VISION_MODULES_DIR}/boundary_tracing/live_wire.cpp
    ${VISION_MODULES_DIR}/boundary_tracing/snakes.cpp
    ${VISION_MODULES_DIR}/featurizations/sdf/chamfer.cpp
    ${VISION_MODULES_DIR}/filters/sobel.cpp
)
target_link_libraries(contour_impl PUBLIC contour_kit)
target_include_directories(contour_impl PUBLIC ${VISION_MODULES_DIR})
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
