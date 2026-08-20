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
    # Define an imported shared library target
    add_library(fftw3f_lib SHARED IMPORTED GLOBAL)

    set_target_properties(fftw3f_lib PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/fftw"
        # 1. Path to the .lib (Import library for linking)
        IMPORTED_IMPLIB "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/fftw/libfftw3f-3.lib"
        # 2. Path to the .dll (Runtime library)
        IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/fftw/libfftw3f-3.dll"
    )

    # Post-build step: automatically copy the DLL next to your executables
    add_custom_target(copy_fftw_dll ALL
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/fftw/libfftw3f-3.dll"
            "$<TARGET_FILE_DIR:miniaudio_normalization>"
        COMMENT "Copying libfftw3f-3.dll to binary output folder..."
    )
else()
    # Linux / macOS fallback
    add_library(fftw3f_lib INTERFACE)
    target_link_libraries(fftw3f_lib INTERFACE fftw3f)
endif()

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
