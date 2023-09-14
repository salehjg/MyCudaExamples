cmake_minimum_required(VERSION 3.7)

include(CheckLanguage)
check_language(CXX)
check_language(CUDA)

project(FaradarsCudaBasics LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_subdirectory(common)
add_subdirectory(example1)
