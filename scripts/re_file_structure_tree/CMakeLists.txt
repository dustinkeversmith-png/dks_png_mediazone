cmake_minimum_required(VERSION 3.16)

project(retree VERSION 1.0.0 LANGUAGES CXX)

add_executable(retree retree.cpp)

target_compile_features(retree PRIVATE cxx_std_17)

if (MSVC)
    target_compile_options(retree PRIVATE /W4)
else()
    target_compile_options(retree PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS retree RUNTIME DESTINATION bin)
