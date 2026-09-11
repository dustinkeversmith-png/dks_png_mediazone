cmake_minimum_required(VERSION 3.20)

project(vocal_acoustics VERSION 0.1.0 LANGUAGES CXX)

option(VA_BUILD_TESTS "Build the unit tests" ON)
option(VA_ENABLE_ONNX_RUNTIME "Build the ONNX Runtime acoustic-model adapter" OFF)
option(VA_ENABLE_STREAMING_ASR "Build the streaming INT8 Zipformer caption engine" OFF)
set(KALDI_NATIVE_FBANK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/csukuangfj-kaldi-native-fbank-b09e686" CACHE PATH "Pinned kaldi-native-fbank sources")
set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime installation prefix")

add_library(vocal_acoustics
    src/audio.cpp
    src/dataset.cpp
    src/evaluator.cpp
    src/features.cpp
    src/metrics.cpp
    src/neural_models.cpp
    src/onnx_runtime_model.cpp
    src/phonemizer.cpp
    src/piper_voice.cpp
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

if(VA_ENABLE_STREAMING_ASR)
    if(NOT VA_ENABLE_ONNX_RUNTIME)
        message(FATAL_ERROR "Streaming ASR requires VA_ENABLE_ONNX_RUNTIME")
    endif()
    set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF)
    add_subdirectory("${KALDI_NATIVE_FBANK_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/knf" EXCLUDE_FROM_ALL)
    add_library(vocal_streaming_asr src/streaming_asr.cpp)
    target_compile_features(vocal_streaming_asr PUBLIC cxx_std_20)
    target_include_directories(vocal_streaming_asr PUBLIC include PRIVATE
        "${ONNXRUNTIME_INCLUDE_DIR}" "${KALDI_NATIVE_FBANK_ROOT}")
    target_link_libraries(vocal_streaming_asr PRIVATE "${ONNXRUNTIME_LIBRARY}" kaldi-native-fbank-core)
    if(MSVC)
        target_compile_options(vocal_streaming_asr PRIVATE /W4 /permissive-)
    endif()
    add_executable(caption-streaming app/caption_streaming.cpp)
    target_compile_definitions(caption-streaming PRIVATE NOMINMAX)
    target_link_libraries(caption-streaming PRIVATE vocal_streaming_asr vocal::acoustics)
    target_include_directories(caption-streaming PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../audio"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/miniaudio/include")
    if(WIN32)
        add_custom_command(TARGET caption-streaming POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll" "$<TARGET_FILE_DIR:caption-streaming>" VERBATIM)
    endif()
endif()

if(VA_BUILD_TESTS)
    include(CTest)
    add_executable(vocal-acoustics-tests tests/metrics_test.cpp)
    target_link_libraries(vocal-acoustics-tests PRIVATE vocal::acoustics)
    target_compile_definitions(vocal-acoustics-tests PRIVATE
        VA_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME vocal-acoustics-tests COMMAND vocal-acoustics-tests)
    if(VA_ENABLE_STREAMING_ASR AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/models/asr_streaming_int8/compact/tokens.txt" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../data/audio/librispeech/sample_000000.wav")
        add_test(NAME streaming-asr-regression COMMAND caption-streaming
            --models "${CMAKE_CURRENT_SOURCE_DIR}/artifacts/models/asr_streaming_int8/compact"
            --self-test "${CMAKE_CURRENT_SOURCE_DIR}/../../data/audio/librispeech/sample_000000.wav")
    endif()
endif()
