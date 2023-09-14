add_library(common "" ../example1/kernel1.cu ../example1/main.cpp)
target_sources(common
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/cnpy.cpp
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/cnpy.h
        )
target_include_directories(common
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        )
target_link_libraries(common
        PRIVATE
        z
        )
set_target_properties(common PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHS}")