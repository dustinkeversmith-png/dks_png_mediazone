cmake_minimum_required(VERSION 3.20)
project(vocal_acoustics VERSION 0.2.0 LANGUAGES CXX)

option(VA_BUILD_TESTS "Build synthesis and framework tests" ON)
option(VA_ENABLE_ONNX_RUNTIME "Build Piper and explicit neural ONNX inference" OFF)
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/onnxruntime"
    CACHE PATH "ONNX Runtime installation prefix")

function(va_library target)
    add_library(${target} STATIC ${ARGN})
    target_include_directories(${target} PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/models")
    target_compile_features(${target} PUBLIC cxx_std_20)
    if(MSVC)
        target_compile_options(${target} PUBLIC /EHsc)
        target_compile_options(${target} PRIVATE /W4 /permissive- /MP /utf-8)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

va_library(vocal_acoustics src/audio.cpp src/dataset.cpp src/evaluator.cpp
    src/features.cpp src/metrics.cpp src/reporting.cpp src/study.cpp)
add_library(vocal::acoustics ALIAS vocal_acoustics)
va_library(vocal_phonemizer models/piper_onnx/phonemizer.cpp)
va_library(vocal_homebrew_models models/homebrew_neural/neural_models.cpp)
target_link_libraries(vocal_homebrew_models PUBLIC vocal::acoustics)
va_library(vocal_dsp_models models/dsp_paradigms/synthesizers.cpp)
target_link_libraries(vocal_dsp_models PUBLIC vocal::acoustics vocal_homebrew_models)
va_library(vocal_prosody models/explicit_neural/length_regulator.cpp)
va_library(vocal_piper_models models/piper_onnx/piper_voice.cpp
    models/piper_onnx/onnx_runtime_model.cpp)
target_link_libraries(vocal_piper_models PUBLIC vocal::acoustics vocal_phonemizer)
va_library(vocal_explicit_models models/explicit_neural/explicit_acoustic_gen.cpp
    models/explicit_neural/neural_vocoder.cpp)
target_link_libraries(vocal_explicit_models PUBLIC vocal::acoustics vocal_prosody)
va_library(vocal_matcha_models models/matcha_onnx/matcha_acoustic_gen.cpp)
target_link_libraries(vocal_matcha_models PUBLIC vocal::acoustics)

if(VA_ENABLE_ONNX_RUNTIME)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS "${ONNXRUNTIME_ROOT}/include" REQUIRED)
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
        HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64" REQUIRED)
endif()
foreach(target vocal_piper_models vocal_explicit_models vocal_matcha_models)
    target_compile_definitions(${target} PRIVATE VA_HAS_ONNX_RUNTIME=$<BOOL:${VA_ENABLE_ONNX_RUNTIME}>)
    if(VA_ENABLE_ONNX_RUNTIME)
        target_include_directories(${target} PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
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

va_library(vocal_demo_cli app/demo_tts.cpp)
target_link_libraries(vocal_demo_cli PUBLIC vocal_dsp_models)
function(va_demo target kind)
    add_executable(${target} app/demo_main.cpp)
    target_compile_definitions(${target} PRIVATE VA_DEMO_KIND=${kind})
    target_link_libraries(${target} PRIVATE vocal_demo_cli)
endfunction()
va_demo(formant-tts formant)
va_demo(hmm-tts statistical_hmm)
va_demo(articulatory-tts articulatory)
va_demo(lpc-tts lpc)
va_demo(sine-wave-tts sine_wave)
va_demo(unit-selection-tts unit_selection)
va_demo(homebrew-tts homebrew_neural)
va_demo(onnx-style-tts onnx_style)

add_executable(piper-tts app/piper_tts.cpp)
target_link_libraries(piper-tts PRIVATE vocal_piper_models)
va_runtime_dll(piper-tts)
add_executable(explicit-tts app/tts_cli.cpp)
target_link_libraries(explicit-tts PRIVATE vocal_explicit_models vocal_phonemizer)
va_runtime_dll(explicit-tts)
add_executable(matcha-tts app/matcha_tts.cpp)
target_link_libraries(matcha-tts PRIVATE vocal_matcha_models vocal_explicit_models vocal_phonemizer)
va_runtime_dll(matcha-tts)

if(NOT VA_ENABLE_ONNX_RUNTIME)
    # Build only usable baseline runners by default; ONNX stubs stay opt-in.
    set_target_properties(vocal_piper_models vocal_explicit_models vocal_matcha_models
        piper-tts explicit-tts matcha-tts PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

if(VA_BUILD_TESTS)
    include(CTest)
    add_executable(vocal-acoustics-tests tests/metrics_test.cpp)
    target_link_libraries(vocal-acoustics-tests PRIVATE vocal::acoustics vocal_dsp_models vocal_phonemizer)
    target_compile_definitions(vocal-acoustics-tests PRIVATE VA_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME vocal-acoustics-tests COMMAND vocal-acoustics-tests)
    add_executable(vocal-prosody-tests tests/prosody_test.cpp)
    target_link_libraries(vocal-prosody-tests PRIVATE vocal_prosody)
    add_test(NAME vocal-prosody-tests COMMAND vocal-prosody-tests)
    if(VA_ENABLE_ONNX_RUNTIME)
        add_executable(explicit-runtime-tests tests/explicit_runtime_test.cpp)
        target_link_libraries(explicit-runtime-tests PRIVATE vocal_explicit_models)
        target_compile_definitions(explicit-runtime-tests PRIVATE VA_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
        va_runtime_dll(explicit-runtime-tests)
        add_test(NAME explicit-runtime-tests COMMAND explicit-runtime-tests)
        find_package(Python3 COMPONENTS Interpreter)
        if(Python3_Interpreter_FOUND)
            add_test(NAME explicit-cli-tests COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/explicit_cli_test.py"
                "$<TARGET_FILE:explicit-tts>" "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/explicit"
                "${CMAKE_CURRENT_BINARY_DIR}/explicit-cli-artifacts")
        endif()
    endif()
endif()
