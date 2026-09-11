cmake_minimum_required(VERSION 3.16)
project(GenerativeMediaResearch LANGUAGES C CXX)

# Standard C++ configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output binaries directly into a clean build/bin folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_library(miniaudio_lib STATIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/miniaudio/include/miniaudio.c
)

# Point to dependencies/miniaudio directory
target_include_directories(miniaudio_lib INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/miniaudio/include
)



# Link platform-specific system audio backends and threading
if(WIN32)
    # Windows multimedia and threading are handled natively by MSVC/MinGW
    target_link_libraries(miniaudio_lib INTERFACE)
elseif(APPLE)
    find_library(CORE_AUDIO CoreAudio REQUIRED)
    find_library(AUDIO_TOOLBOX AudioToolbox REQUIRED)
    find_library(CORE_FOUNDATION CoreFoundation REQUIRED)
    target_link_libraries(miniaudio_lib INTERFACE
        ${CORE_AUDIO}
        ${AUDIO_TOOLBOX}
        ${CORE_FOUNDATION}
        pthread
        m
        dl
    )
elseif(UNIX)
    find_package(Threads REQUIRED)
    target_link_libraries(miniaudio_lib INTERFACE
        Threads::Threads
        m
        dl
    )
endif()


# Add Eigen as a header-only interface
add_library(eigen_lib INTERFACE)
target_include_directories(eigen_lib INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/eigen
)

# Precompile Eigen's heaviest headers so they compile once into a binary cache
target_precompile_headers(eigen_lib INTERFACE
    <Eigen/Dense>
    <unsupported/Eigen/Polynomials>
)

# ==============================================================================
# 3. Dependency: FFTW3 (Imported Shared DLL)
# ==============================================================================
if(WIN32)
    # The repository ships FFTW's DLL and .def, but not a compiler-specific import library.
    set(FFTW_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/fftw")
    set(FFTW_DLL "${FFTW_ROOT}/libfftw3f-3.dll")
    set(FFTW_DEF "${FFTW_ROOT}/libfftw3f-3.def")
    set(FFTW_IMPORT_LIB "${CMAKE_CURRENT_BINARY_DIR}/libfftw3f-3.lib")

    if(EXISTS "${FFTW_ROOT}/libfftw3f-3.lib")
        set(FFTW_LINK_LIB "${FFTW_ROOT}/libfftw3f-3.lib")
        set(FFTW_IMPORT_TARGET)
    else()
        find_program(DLLTOOL_EXECUTABLE NAMES dlltool llvm-dlltool REQUIRED)
        add_custom_command(
            OUTPUT "${FFTW_IMPORT_LIB}"
            COMMAND "${DLLTOOL_EXECUTABLE}"
                --input-def "${FFTW_DEF}"
                --dllname libfftw3f-3.dll
                --output-lib "${FFTW_IMPORT_LIB}"
            DEPENDS "${FFTW_DEF}"
            COMMENT "Generating FFTW3f import library"
            VERBATIM
        )
        add_custom_target(fftw3f_import_lib DEPENDS "${FFTW_IMPORT_LIB}")
        set(FFTW_LINK_LIB "${FFTW_IMPORT_LIB}")
        set(FFTW_IMPORT_TARGET fftw3f_import_lib)
    endif()

    add_library(fftw3f_lib SHARED IMPORTED GLOBAL)

    set_target_properties(fftw3f_lib PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${FFTW_ROOT}"
        IMPORTED_IMPLIB "${FFTW_LINK_LIB}"
        IMPORTED_LOCATION "${FFTW_DLL}"
    )

    add_custom_target(copy_fftw_dll ALL
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${FFTW_DLL}" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
        COMMENT "Copying FFTW3f runtime DLL"
    )
    if(FFTW_IMPORT_TARGET)
        add_dependencies(copy_fftw_dll ${FFTW_IMPORT_TARGET})
    endif()
else()
    # Linux / macOS fallback
    add_library(fftw3f_lib INTERFACE)
    target_link_libraries(fftw3f_lib INTERFACE fftw3f)
endif()

# Target 3: Eigen numerical sample
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/eigen_sample.cpp")
    add_executable(eigen_sample examples/eigen_sample.cpp)
    target_link_libraries(eigen_sample PRIVATE eigen_lib)
endif()

# Target 4: FFTW numerical sample
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/fftw_sample.cpp")
    add_executable(fftw_sample examples/fftw_sample.cpp)
    target_link_libraries(fftw_sample PRIVATE fftw3f_lib)
    if(FFTW_IMPORT_TARGET)
        add_dependencies(fftw_sample ${FFTW_IMPORT_TARGET})
    endif()
endif()

# Target 5: AudioMNIST formant-to-vowel input pipeline (optional, only if exists)
# add_executable(input_pipeline src/audio/input_pipeline.cpp)
# target_include_directories(input_pipeline PRIVATE
#     "${CMAKE_CURRENT_SOURCE_DIR}/src"
#     "${CMAKE_CURRENT_SOURCE_DIR}/src/audio"
#     "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/input"
#     "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/framing"
# )
# target_compile_definitions(input_pipeline PRIVATE NOMINMAX)
# target_link_libraries(input_pipeline PRIVATE miniaudio_lib eigen_lib fftw3f_lib)
# if(FFTW_IMPORT_TARGET)
#     add_dependencies(input_pipeline ${FFTW_IMPORT_TARGET})
# endif()
# if(WIN32)
#     add_custom_command(TARGET input_pipeline POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E copy_if_different
#             "${FFTW_DLL}"
#             "$<TARGET_FILE_DIR:input_pipeline>"
#         COMMENT "Copying FFTW3f runtime DLL for input_pipeline"
#     )
# endif()
# if(WIN32 AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/fftw_sample.cpp")
#     add_custom_command(TARGET fftw_sample POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E copy_if_different
#             "${FFTW_DLL}"
#             "$<TARGET_FILE_DIR:fftw_sample>"
#         COMMENT "Copying FFTW3f runtime DLL for fftw_sample"
#     )
# endif()

# Vision geometry / screen-detection library + unit tests
add_subdirectory(src/video)

# ==========================================
# 2. Example Executables
# ==========================================

# Target 1: miniaudio_normalization
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/miniaudio_normalization.cpp")
    add_executable(miniaudio_normalization
        examples/miniaudio_normalization.cpp
    )
    target_link_libraries(miniaudio_normalization PRIVATE miniaudio_lib)
    target_link_libraries(miniaudio_normalization PRIVATE eigen_lib)
endif()

# Target 2: miniaudio_playback
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/miniaudio_playback.cpp")
    add_executable(miniaudio_playback
        examples/miniaudio_playback.cpp
    )
    target_link_libraries(miniaudio_playback PRIVATE miniaudio_lib)
    target_link_libraries(miniaudio_playback PRIVATE eigen_lib)
endif()

# ==========================================
# 3. Audio Processing Methods (LPC vs Fourier)
# ==========================================

# Target: LPC-based formant extraction and vowel recognition
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/lpc_method.cpp")
    add_executable(lpc_method src/audio/lpc_method.cpp)
    target_include_directories(lpc_method PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/audio"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/input"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/framing"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/formants"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/filter"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/power_spectrum"
    )
    target_compile_definitions(lpc_method PRIVATE NOMINMAX)
    target_link_libraries(lpc_method PRIVATE miniaudio_lib eigen_lib fftw3f_lib)
    if(FFTW_IMPORT_TARGET)
        add_dependencies(lpc_method ${FFTW_IMPORT_TARGET})
    endif()
    if(WIN32)
        add_custom_command(TARGET lpc_method POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FFTW_DLL}"
                "$<TARGET_FILE_DIR:lpc_method>"
            COMMENT "Copying FFTW3f runtime DLL for lpc_method"
        )
    endif()
endif()

# Target: Fourier/Spectral-based analysis and vowel recognition
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/fourier_method.cpp")
    add_executable(fourier_method src/audio/fourier_method.cpp)
    target_include_directories(fourier_method PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/audio"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/input"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/framing"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/spectral"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/filter"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/power_spectrum"
    )
    target_compile_definitions(fourier_method PRIVATE NOMINMAX)
    target_link_libraries(fourier_method PRIVATE miniaudio_lib eigen_lib fftw3f_lib)
    if(FFTW_IMPORT_TARGET)
        add_dependencies(fourier_method ${FFTW_IMPORT_TARGET})
    endif()
    if(WIN32)
        add_custom_command(TARGET fourier_method POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FFTW_DLL}"
                "$<TARGET_FILE_DIR:fourier_method>"
            COMMENT "Copying FFTW3f runtime DLL for fourier_method"
        )
    endif()
endif()

# ==========================================
# 4. Networkless Captioning Models (MFCC + DTW + Viterbi)
# ==========================================

# Shared configuration for every model program.
function(configure_model_target target_name)
    target_include_directories(${target_name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/audio"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/input"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/framing"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/formants"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/spectral"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/filter"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/power_spectrum"
    )
    target_compile_definitions(${target_name} PRIVATE NOMINMAX)
    target_link_libraries(${target_name} PRIVATE miniaudio_lib eigen_lib fftw3f_lib)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /O2 /fp:fast)
    else()
        target_compile_options(${target_name} PRIVATE -O3 -ffast-math)
    endif()
    if(FFTW_IMPORT_TARGET)
        add_dependencies(${target_name} ${FFTW_IMPORT_TARGET})
    endif()
    if(WIN32)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FFTW_DLL}"
                "$<TARGET_FILE_DIR:${target_name}>"
            COMMENT "Copying FFTW3f runtime DLL for ${target_name}"
        )
    endif()
endfunction()

# Target: trains the monophone acoustic model and the n-gram language models
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/models/train_models.cpp")
    add_executable(train_models src/audio/models/train_models.cpp)
    configure_model_target(train_models)
endif()

# Target: end-to-end benchmark across digits / TIMIT / LibriSpeech
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/models/benchmark_models.cpp")
    add_executable(benchmark_models src/audio/models/benchmark_models.cpp)
    configure_model_target(benchmark_models)
endif()

# Target: single-file captioner (wav in, text out)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/audio/models/caption.cpp")
    add_executable(caption src/audio/models/caption.cpp)
    configure_model_target(caption)
endif()

# Regression coverage for word-boundary context and forced alignment.
add_executable(test_model_contexts src/audio/models/test_model_contexts.cpp)
configure_model_target(test_model_contexts)
enable_testing()
add_test(NAME model_contexts COMMAND test_model_contexts)
