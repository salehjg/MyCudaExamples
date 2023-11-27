set(target_name "MatMul")
get_filename_component(DIR_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
set(KERNEL_EXAMPLE kernel_${DIR_NAME})
set(${KERNEL_EXAMPLE} 3 CACHE STRING "Integer specifying the kernel version to be used in the compilation of example ${target_name}.")

set(kernel_version ${${KERNEL_EXAMPLE}})
add_executable(${target_name}
        kernels/${kernel_version}/kernel.cu
        main.cpp
        )
target_include_directories(${target_name}
        PRIVATE
        kernels/${kernel_version}/
        )
target_link_libraries(${target_name} common)
set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHS}")
target_compile_definitions(${target_name} PRIVATE TARGETNAME=\"${target_name}\" TARGETKERNEL=\"${kernel_version}\")