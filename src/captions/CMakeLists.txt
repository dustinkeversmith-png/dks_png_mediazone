# Caption engines. This directory is captions ONLY - speech in, text out.
# Text-to-speech lives in ../audio_synth and nothing here links against it.
#
# Targets by model family (see models/README.md):
#   lpc_method, fourier_method  models/naive/        frame-level vowel lookup
#   train_models, caption,      models/toy_pruned_hmm/  GMM-HMM + Viterbi stack
#   benchmark_models
#   caption-streaming           models/int8_zip/     INT8 Zipformer via ONNX Runtime
#
# The first four need only FFTW/Eigen/miniaudio and build by default. The
# streaming engine additionally needs ONNX Runtime and kaldi-native-fbank, so
# it is behind CAPTIONS_ENABLE_STREAMING_ASR.

option(CAPTIONS_ENABLE_STREAMING_ASR "Build the INT8 Zipformer streaming caption engine" OFF)
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/onnxruntime"
    CACHE PATH "ONNX Runtime installation prefix")
set(KALDI_NATIVE_FBANK_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/models/int8_zip/third_party/csukuangfj-kaldi-native-fbank-b09e686"
    CACHE PATH "Pinned kaldi-native-fbank sources")

# Shared DSP front-end and model headers. One interface target carries the
# include roots so every caption target sees the same layout:
#   <filter/...> <framing/...> <formants/...> <spectral/...> <input/...>
#   <power_spectrum/...> <benchmarks/...>   from this directory
#   <naive/...> <toy_pruned_hmm/...>        from models/
add_library(captions_common INTERFACE)
target_include_directories(captions_common INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/models"
    "${CMAKE_CURRENT_SOURCE_DIR}/input"
    "${CMAKE_CURRENT_SOURCE_DIR}/framing"
    "${CMAKE_CURRENT_SOURCE_DIR}/filter"
    # The toy_pruned_hmm headers include their siblings by bare name.
    "${CMAKE_CURRENT_SOURCE_DIR}/models/toy_pruned_hmm"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/toy_pruned_hmm/dtw"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/toy_pruned_hmm/ngram"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/toy_pruned_hmm/phonemes"
    "${CMAKE_CURRENT_SOURCE_DIR}/models/toy_pruned_hmm/triphone"
)
target_compile_features(captions_common INTERFACE cxx_std_20)
if(MSVC)
    target_compile_definitions(captions_common INTERFACE NOMINMAX)
endif()

# Helper: one caption executable with the common include roots and the DSP
# dependencies the non-neural engines share.
function(captions_add_executable name source)
    add_executable(${name} "${source}")
    target_link_libraries(${name} PRIVATE captions_common miniaudio_lib eigen_lib fftw3f_lib)
    if(FFTW_IMPORT_TARGET)
        add_dependencies(${name} ${FFTW_IMPORT_TARGET})
    endif()
    if(WIN32 AND FFTW_DLL)
        add_custom_command(TARGET ${name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${FFTW_DLL}" "$<TARGET_FILE_DIR:${name}>"
            COMMENT "Copying FFTW3f runtime DLL for ${name}")
    endif()
endfunction()

# --- models/naive: the original frame-level classifiers --------------------
captions_add_executable(lpc_method models/naive/lpc_method.cpp)
captions_add_executable(fourier_method models/naive/fourier_method.cpp)

# --- models/toy_pruned_hmm: trained GMM-HMM + Viterbi decoder ---------------
captions_add_executable(train_models models/train_models.cpp)
captions_add_executable(caption cli/caption.cpp)
captions_add_executable(benchmark_models benchmarks/benchmark_models.cpp)
captions_add_executable(test_model_contexts models/test_model_contexts.cpp)
# Regression coverage for word-boundary context and forced alignment.
add_test(NAME model_contexts COMMAND test_model_contexts)

# --- models/int8_zip: INT8 Zipformer streaming engine ----------------------
if(CAPTIONS_ENABLE_STREAMING_ASR)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        HINTS "${ONNXRUNTIME_ROOT}/include" REQUIRED)
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
        HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64" REQUIRED)

    set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    add_subdirectory("${KALDI_NATIVE_FBANK_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/knf" EXCLUDE_FROM_ALL)

    add_library(captions_streaming_asr models/int8_zip/streaming_asr.cpp)
    target_compile_features(captions_streaming_asr PUBLIC cxx_std_20)
    target_include_directories(captions_streaming_asr
        PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/models/int8_zip/include"
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/models/int8_zip"
                "${ONNXRUNTIME_INCLUDE_DIR}" "${KALDI_NATIVE_FBANK_ROOT}")
    target_link_libraries(captions_streaming_asr
        PRIVATE "${ONNXRUNTIME_LIBRARY}" kaldi-native-fbank-core)
    if(MSVC)
        target_compile_options(captions_streaming_asr PRIVATE /W4 /permissive-)
    endif()

    # The app compiles miniaudio itself (capture only, MA_NO_DECODING), so it
    # takes the header path rather than linking miniaudio_lib.
    add_executable(caption-streaming app/caption_streaming.cpp)
    target_link_libraries(caption-streaming PRIVATE captions_streaming_asr captions_common)
    target_include_directories(caption-streaming PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../dependencies/miniaudio/include")
    if(WIN32)
        add_custom_command(TARGET caption-streaming POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll" "$<TARGET_FILE_DIR:caption-streaming>"
            VERBATIM)
    endif()

    # Regression test: silence rejection, reset/finish contracts, and
    # packet-boundary invariance of the transcript.
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models/int8_zip/models/compact/tokens.txt" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../data/audio/librispeech/sample_000000.wav")
        add_test(NAME streaming-asr-regression COMMAND caption-streaming
            --models "${CMAKE_CURRENT_SOURCE_DIR}/models/int8_zip/models/compact"
            --self-test "${CMAKE_CURRENT_SOURCE_DIR}/../../data/audio/librispeech/sample_000000.wav")
    endif()
endif()
