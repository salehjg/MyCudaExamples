cmake_minimum_required(VERSION 3.7)

include(CheckLanguage)
check_language(CXX)
check_language(CUDA)

set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel Coverage.")
set(CUDA_ARCHS "86" CACHE STRING "semi-colon seperated compute capability codes without dots, like 62;86")

project(FaradarsCudaBasics LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_subdirectory(common)
add_subdirectory(example1)
