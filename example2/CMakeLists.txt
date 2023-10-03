set(target_name "BasicOps")
set(KERNEL_EXAMPLE2 3 CACHE STRING "Integer specifying the kernel version to be used in the compilation of example ${target_name}.")


set(kernel_version ${KERNEL_EXAMPLE2})
add_executable(${target_name}
        kernels/${kernel_version}/kernel.cu
        main.cpp
        )
target_include_directories(${target_name}
        PRIVATE
        "kernels/${kernel_version}/"
        )
target_link_libraries(${target_name} common)
set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHS}")
target_compile_definitions(${target_name} PRIVATE TARGETNAME=\"${target_name}\" TARGETKERNEL=\"${kernel_version}\")