add_library(common "")
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