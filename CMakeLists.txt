cmake_minimum_required(VERSION 3.7)

include(CheckLanguage)
check_language(CXX)
check_language(CUDA)

# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel Coverage.")
set(CUDA_ARCHS "86" CACHE STRING "semi-colon seperated Compute Capability codes without dots, like 62;86")

# Enforce C++17 for nvcc
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

project(FaradarsCudaBasics LANGUAGES CUDA CXX)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G -O0 -Xptxas=\"-v\"")  # -G enables device debugging (its a nvcc only flag)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-O3 -Xptxas=\"-v\" --fmad=false")
endif()

add_subdirectory(common)
add_subdirectory(example1)
add_subdirectory(example2)
add_subdirectory(example3)
add_subdirectory(example4)
add_subdirectory(example5)
