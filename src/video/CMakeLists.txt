# Vision modules + atomic unit tests. Included from the repository root CMakeLists.txt.

set(VISION_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(VISION_MODULES_DIR ${VISION_SRC_DIR}/modules)
set(VISION_DATASETS_DIR ${VISION_SRC_DIR}/datasets)
set(VISION_TESTS_DIR ${VISION_SRC_DIR}/tests)
set(VISION_ATOMS_ROOT ${VISION_TESTS_DIR}/unit_tests/modules/atoms)

file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}/data/vision" VISION_VISION_ROOT_NORM)
file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}/data/vision/atoms" VISION_DATA_DIR_NORM)

# ---------------------------------------------------------------------------
# vision_core: Eigen / FFTW / stb I/O used by atom suites
# ---------------------------------------------------------------------------
add_library(vision_core STATIC
    ${VISION_DATASETS_DIR}/io/vision_io.cpp
)
target_include_directories(vision_core PUBLIC
    ${VISION_SRC_DIR}
    ${VISION_MODULES_DIR}
    ${VISION_MODULES_DIR}/math
    ${VISION_DATASETS_DIR}
    ${VISION_DATASETS_DIR}/io
    ${VISION_TESTS_DIR}/unit_tests
    ${VISION_TESTS_DIR}/unit_tests/datasets/atoms
    ${CMAKE_SOURCE_DIR}/dependencies/stb
    ${CMAKE_SOURCE_DIR}/dependencies/libspatialindex/include
)
target_compile_definitions(vision_core PUBLIC
    VISION_DATA_DIR="${VISION_DATA_DIR_NORM}"
    VISION_VISION_ROOT="${VISION_VISION_ROOT_NORM}"
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

function(add_vision_atom_test test_name source_file)
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} PRIVATE vision_core)
    get_filename_component(atom_dir ${source_file} DIRECTORY)
    file(TO_CMAKE_PATH "${atom_dir}/artifacts" atom_artifact_root)
    file(TO_CMAKE_PATH "${atom_dir}/atom.config.json" atom_config_path)
    target_compile_definitions(${test_name} PRIVATE
        VISION_TEST_ARTIFACT_ROOT="${atom_artifact_root}"
        VISION_ATOM_CONFIG_PATH="${atom_config_path}"
        VISION_DEFAULT_DATASET="${test_name}"
    )
    set_property(TARGET ${test_name} PROPERTY CXX_STANDARD 17)
    vision_copy_fftw(${test_name})
endfunction()

file(GLOB_RECURSE VISION_ATOM_SOURCES
    "${VISION_ATOMS_ROOT}/test_*.cpp"
    "${VISION_ATOMS_ROOT}/*/test_*.cpp"
    "${VISION_ATOMS_ROOT}/*/*/test_*.cpp"
)

foreach(source_file ${VISION_ATOM_SOURCES})
    get_filename_component(source_name ${source_file} NAME_WE)
    add_vision_atom_test(${source_name} ${source_file})
endforeach()

set(VISION_DATASET_ATOMS_DIR ${VISION_TESTS_DIR}/unit_tests/datasets/atoms)
file(GLOB VISION_DATASET_ATOM_SOURCES "${VISION_DATASET_ATOMS_DIR}/*/test_*.cpp")
foreach(source_file ${VISION_DATASET_ATOM_SOURCES})
    get_filename_component(source_name ${source_file} NAME_WE)
    add_vision_atom_test(${source_name} ${source_file})
endforeach()

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
