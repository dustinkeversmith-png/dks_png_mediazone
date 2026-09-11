cmake_minimum_required(VERSION 3.20)

# Speech synthesis (text -> speech) ONLY.
#
# Captioning (speech -> text) lives in ../captions and is built from the
# repository-root CMakeLists; nothing here depends on it. The two projects
# share only the ONNX Runtime SDK in ../../dependencies/onnxruntime.
#
# Layout:
#   include/vocal/  framework API - audio, features, metrics, dataset,
#                   evaluator, reporting, study, and acoustic_model, the
#                   interface every model family implements
#   src/            framework implementation
#   models/         one directory per model family (see models/README.md):
#                     dsp_paradigms/   six classical source-filter sketches
#                     homebrew_neural/ untrained FastSpeech2-shaped baselines
#                     piper_onnx/      real Piper VITS checkpoints via ONNX
project(vocal_acoustics VERSION 0.1.0 LANGUAGES CXX)

option(VA_BUILD_TESTS "Build the unit tests" ON)
option(VA_ENABLE_ONNX_RUNTIME "Build the ONNX Runtime voice adapter (models/piper_onnx)" OFF)
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/onnxruntime"
    CACHE PATH "ONNX Runtime installation prefix")

set(VA_FRAMEWORK_SOURCES
    src/audio.cpp
    src/dataset.cpp
    src/evaluator.cpp
    src/features.cpp
    src/metrics.cpp
    src/reporting.cpp
    src/study.cpp
)

set(VA_MODEL_SOURCES
    models/dsp_paradigms/synthesizers.cpp
    models/homebrew_neural/neural_models.cpp
    models/piper_onnx/onnx_runtime_model.cpp
    models/piper_onnx/phonemizer.cpp
    models/piper_onnx/piper_voice.cpp
)

add_library(vocal_acoustics ${VA_FRAMEWORK_SOURCES} ${VA_MODEL_SOURCES})
add_library(vocal::acoustics ALIAS vocal_acoustics)

target_include_directories(vocal_acoustics PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/models>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(vocal_acoustics PUBLIC cxx_std_20)

if(VA_ENABLE_ONNX_RUNTIME)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS "${ONNXRUNTIME_ROOT}/include" REQUIRED)
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
        HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64" REQUIRED)
    target_include_directories(vocal_acoustics PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
    target_link_libraries(vocal_acoustics PRIVATE "${ONNXRUNTIME_LIBRARY}")
    target_compile_definitions(vocal_acoustics PRIVATE VA_HAS_ONNX_RUNTIME=1)
    if(WIN32)
        add_custom_command(TARGET vocal_acoustics POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
                "$<TARGET_FILE_DIR:vocal_acoustics>/onnxruntime.dll"
            VERBATIM)
    endif()
else()
    target_compile_definitions(vocal_acoustics PRIVATE VA_HAS_ONNX_RUNTIME=0)
endif()

if(MSVC)
    target_compile_options(vocal_acoustics PRIVATE /W4 /permissive-)
else()
    target_compile_options(vocal_acoustics PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(vocal-eval app/vocal_eval.cpp)
target_link_libraries(vocal-eval PRIVATE vocal::acoustics)

if(VA_BUILD_TESTS)
    include(CTest)
    add_executable(vocal-acoustics-tests tests/metrics_test.cpp)
    target_link_libraries(vocal-acoustics-tests PRIVATE vocal::acoustics)
    target_compile_definitions(vocal-acoustics-tests PRIVATE
        VA_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME vocal-acoustics-tests COMMAND vocal-acoustics-tests)
endif()
