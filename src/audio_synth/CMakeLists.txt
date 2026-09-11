cmake_minimum_required(VERSION 3.20)

project(vocal_acoustics VERSION 0.1.0 LANGUAGES CXX)

option(VA_BUILD_TESTS "Build the unit tests" ON)
option(VA_ENABLE_ONNX_RUNTIME "Build the ONNX Runtime acoustic-model adapter" OFF)
set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime installation prefix")

add_library(vocal_acoustics
    src/audio.cpp
    src/dataset.cpp
    src/evaluator.cpp
    src/features.cpp
    src/metrics.cpp
    src/neural_models.cpp
    src/onnx_runtime_model.cpp
    src/reporting.cpp
    src/study.cpp
    src/synthesizers.cpp
)
add_library(vocal::acoustics ALIAS vocal_acoustics)

target_include_directories(vocal_acoustics PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
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
    add_test(NAME vocal-acoustics-tests COMMAND vocal-acoustics-tests)
endif()
