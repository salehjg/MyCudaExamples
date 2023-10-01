add_library(common "")
target_sources(common
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/cnpy.cpp
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/cnpy.h
        ${CMAKE_CURRENT_LIST_DIR}/CTensor.h
        ${CMAKE_CURRENT_LIST_DIR}/common.h
        ${CMAKE_CURRENT_LIST_DIR}/CRandFiller.h
        )
target_include_directories(common
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        )
target_link_libraries(common
        PRIVATE
        z
        )
set_target_properties(common PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHS}")