cmake_minimum_required(VERSION 3.25)
project(vocal_acoustics VERSION 0.2.0 LANGUAGES CXX)

option(VA_BUILD_TESTS "Build synthesis and framework tests" ON)
option(VA_ENABLE_ONNX_RUNTIME "Build Piper and explicit neural ONNX inference" OFF)
option(VA_BUILD_ARTICULATION_EXPERIMENT "Build the separate word/phoneme articulation lab" OFF)
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/onnxruntime"
    CACHE PATH "ONNX Runtime installation prefix")

function(va_compile_options target)
    if(MSVC)
        target_compile_options(${target} PUBLIC /EHsc)
        target_compile_options(${target} PRIVATE /W4 /permissive- /MP /utf-8)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

function(va_library target)
    add_library(${target} STATIC ${ARGN})
    target_include_directories(${target} PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/models")
    target_compile_features(${target} PUBLIC cxx_std_20)
    va_compile_options(${target})
endfunction()

va_library(vocal_acoustics src/audio.cpp src/dataset.cpp src/evaluator.cpp
    src/features.cpp src/metrics.cpp src/reporting.cpp src/study.cpp)
add_library(vocal::acoustics ALIAS vocal_acoustics)
va_library(vocal_phonemizer models/piper_onnx/phonemizer.cpp)
va_library(vocal_prosody models/explicit_neural/length_regulator.cpp)
va_library(vocal_piper_models models/piper_onnx/piper_voice.cpp
    models/piper_onnx/onnx_runtime_model.cpp)
target_link_libraries(vocal_piper_models PUBLIC vocal::acoustics vocal_phonemizer)
va_library(vocal_explicit_models models/explicit_neural/explicit_acoustic_gen.cpp
    models/explicit_neural/neural_vocoder.cpp models/explicit_neural/prosody_predictor.cpp)
target_link_libraries(vocal_explicit_models PUBLIC vocal::acoustics vocal_prosody)

if(VA_ENABLE_ONNX_RUNTIME)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS "${ONNXRUNTIME_ROOT}/include" REQUIRED)
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
        HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64" REQUIRED)
endif()
foreach(target vocal_piper_models vocal_explicit_models)
    target_compile_definitions(${target} PRIVATE VA_HAS_ONNX_RUNTIME=$<BOOL:${VA_ENABLE_ONNX_RUNTIME}>)
    if(VA_ENABLE_ONNX_RUNTIME)
        target_include_directories(${target} SYSTEM PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
        target_link_libraries(${target} PRIVATE "${ONNXRUNTIME_LIBRARY}")
    endif()
endforeach()

function(va_runtime_dll target)
    if(WIN32 AND VA_ENABLE_ONNX_RUNTIME)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll" "$<TARGET_FILE_DIR:${target}>"
            VERBATIM)
    endif()
endfunction()

add_executable(piper-tts app/piper_tts.cpp)
va_compile_options(piper-tts)
target_link_libraries(piper-tts PRIVATE vocal_piper_models)
va_runtime_dll(piper-tts)
add_executable(explicit-tts app/tts_cli.cpp)
va_compile_options(explicit-tts)
target_link_libraries(explicit-tts PRIVATE vocal_explicit_models vocal_phonemizer)
va_runtime_dll(explicit-tts)

if(VA_BUILD_ARTICULATION_EXPERIMENT)
    if(NOT VA_ENABLE_ONNX_RUNTIME)
        message(FATAL_ERROR "The articulation experiment requires ONNX Runtime")
    endif()
    va_library(vocal_articulation_experiment experiments/articulation/articulation.cpp)
    target_include_directories(vocal_articulation_experiment PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_include_directories(vocal_articulation_experiment PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/experiments/articulation")
    target_link_libraries(vocal_articulation_experiment PUBLIC vocal_phonemizer vocal_prosody)
    add_executable(articulation-lab experiments/articulation/main.cpp)
    va_compile_options(articulation-lab)
    target_include_directories(articulation-lab PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(articulation-lab PRIVATE vocal_articulation_experiment vocal_explicit_models)
    va_runtime_dll(articulation-lab)
endif()

if(NOT VA_ENABLE_ONNX_RUNTIME)
    # The runtime-free preset builds framework libraries and tests; ONNX stubs stay opt-in.
    set_target_properties(vocal_piper_models vocal_explicit_models
        piper-tts explicit-tts PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

if(VA_BUILD_TESTS)
    include(CTest)
    add_executable(vocal-acoustics-tests tests/metrics_test.cpp)
    va_compile_options(vocal-acoustics-tests)
    target_link_libraries(vocal-acoustics-tests PRIVATE vocal::acoustics vocal_phonemizer)
    target_compile_definitions(vocal-acoustics-tests PRIVATE VA_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME vocal-acoustics-tests COMMAND vocal-acoustics-tests)
    add_executable(vocal-prosody-tests tests/prosody_test.cpp)
    va_compile_options(vocal-prosody-tests)
    target_link_libraries(vocal-prosody-tests PRIVATE vocal_prosody)
    add_test(NAME vocal-prosody-tests COMMAND vocal-prosody-tests)
    if(VA_BUILD_ARTICULATION_EXPERIMENT)
        add_executable(articulation-control-tests experiments/articulation/control_test.cpp)
        va_compile_options(articulation-control-tests)
        target_link_libraries(articulation-control-tests PRIVATE vocal_articulation_experiment)
        target_compile_definitions(articulation-control-tests PRIVATE VA_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
        add_test(NAME articulation-control-tests COMMAND articulation-control-tests)
    endif()
    if(VA_ENABLE_ONNX_RUNTIME)
        add_executable(explicit-runtime-tests tests/explicit_runtime_test.cpp)
        va_compile_options(explicit-runtime-tests)
        target_link_libraries(explicit-runtime-tests PRIVATE vocal_explicit_models)
        target_compile_definitions(explicit-runtime-tests PRIVATE VA_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
        va_runtime_dll(explicit-runtime-tests)
        add_test(NAME explicit-runtime-tests COMMAND explicit-runtime-tests)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            if(VA_BUILD_ARTICULATION_EXPERIMENT AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model_assets/explicit_neural/prosody_predictor.onnx")
                add_test(NAME articulation-cli-tests COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/experiments/articulation/cli_test.py"
                    "$<TARGET_FILE:articulation-lab>" "${CMAKE_CURRENT_SOURCE_DIR}/model_assets/explicit_neural"
                    "${CMAKE_CURRENT_BINARY_DIR}/articulation-cli-artifacts")
            endif()
            add_test(NAME explicit-cli-tests COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/explicit_cli_test.py"
                "$<TARGET_FILE:explicit-tts>" "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/explicit"
                "${CMAKE_CURRENT_BINARY_DIR}/explicit-cli-artifacts")
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model_assets/explicit_neural/prosody_predictor.onnx")
                add_test(NAME explicit-trained-cli-tests COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/tests/trained_cli_test.py"
                    "$<TARGET_FILE:explicit-tts>" "${CMAKE_CURRENT_SOURCE_DIR}/model_assets/explicit_neural"
                    "${CMAKE_CURRENT_BINARY_DIR}/explicit-trained-artifacts")
            endif()
        endif()
    endif()
endif()
