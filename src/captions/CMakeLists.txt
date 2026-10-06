# Captions: speech in, text out. Text-to-speech lives in ../audio_synth and
# nothing here links against it.
#
# One engine: the streaming Zipformer2 transducer (INT8, ONNX Runtime CPU) in
# models/int8_zip/, exposed as the CaptionEngine API and the `captions` command
# line (batch files/video and live microphone). See models/int8_zip/README.md.
# Retired GMM-HMM and DSP experiments live in archive/ and are not built.

option(CAPTIONS_ENABLE_STREAMING_ASR "Build the Zipformer2 caption engine (needs ONNX Runtime)" ON)
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/onnxruntime"
    CACHE PATH "ONNX Runtime installation prefix")
set(KALDI_NATIVE_FBANK_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/models/int8_zip/third_party/csukuangfj-kaldi-native-fbank-b09e686"
    CACHE PATH "Pinned kaldi-native-fbank sources")

# Include root for <input/...> (WAV reader) and <benchmarks/...> (WER scorer).
add_library(captions_common INTERFACE)
target_include_directories(captions_common INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(captions_common INTERFACE cxx_std_20)
if(MSVC)
    target_compile_definitions(captions_common INTERFACE NOMINMAX)
endif()

if(CAPTIONS_ENABLE_STREAMING_ASR)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS "${ONNXRUNTIME_ROOT}/include" REQUIRED)
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
        HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64" REQUIRED)

    set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    add_subdirectory("${KALDI_NATIVE_FBANK_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/knf" EXCLUDE_FROM_ALL)

    # Caption engine: streaming (live) and batch (files/video) decoding of the
    # Zipformer2 checkpoint, plus the CaptionEngine API and caption formatting.
    set(CAPTIONS_MODEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/models/int8_zip/models")
    add_library(captions_streaming_asr
        models/int8_zip/streaming_asr.cpp
        models/int8_zip/batch_asr.cpp
        models/int8_zip/caption_engine.cpp
        models/int8_zip/vad.cpp
        models/int8_zip/hotwords.cpp)
    target_compile_features(captions_streaming_asr PUBLIC cxx_std_20)
    target_include_directories(captions_streaming_asr
        PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/models/int8_zip/include"
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/models/int8_zip"
                "${ONNXRUNTIME_INCLUDE_DIR}" "${KALDI_NATIVE_FBANK_ROOT}")
    target_compile_definitions(captions_streaming_asr
        PRIVATE CAPTIONS_MODEL_ROOT="${CAPTIONS_MODEL_ROOT}")
    target_link_libraries(captions_streaming_asr
        PRIVATE "${ONNXRUNTIME_LIBRARY}" kaldi-native-fbank-core captions_common)
    if(MSVC)
        target_compile_options(captions_streaming_asr PRIVATE /W4 /permissive-)
    endif()

    # The `captions` command line: batch and live modes, benchmark, self-test.
    # It compiles miniaudio itself (capture only, MA_NO_DECODING), so it takes
    # the header path rather than linking miniaudio_lib.
    add_executable(captions app/captions.cpp)
    target_link_libraries(captions PRIVATE captions_streaming_asr captions_common)
    target_include_directories(captions PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/miniaudio/include")
    if(MSVC)
        target_compile_options(captions PRIVATE /W4 /permissive-)
    endif()
    if(WIN32)
        add_custom_command(TARGET captions POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll" "$<TARGET_FILE_DIR:captions>"
            VERBATIM)
    endif()

    # Regression test: live-engine contracts (silence, reset/finish, packet
    # boundaries), WER normalizer, hotwords, batch == live parity (Zipformer2,
    # when present) and default batch caption output.
    if(EXISTS "${CAPTIONS_MODEL_ROOT}/parakeet-tdt-110m/tokens.txt" AND
       EXISTS "${CAPTIONS_MODEL_ROOT}/nemo-streaming-480ms/tokens.txt" AND
       EXISTS "${CAPTIONS_MODEL_ROOT}/silero-vad/silero_vad.onnx" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../data/audio/librispeech/sample_000000.wav")
        add_test(NAME captions-self-test COMMAND captions
            --self-test "${CMAKE_CURRENT_SOURCE_DIR}/../../data/audio/librispeech")
    endif()
endif()
