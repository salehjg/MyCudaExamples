set(KERNEL 1)
add_executable(example1
        "kernels/${KERNEL}/kernel.cu"
        main.cpp
        kernels/1/kernel.h)
target_include_directories(example1
        PRIVATE
        "kernels/${KERNEL}/"
        )
target_link_libraries(example1 common)
set_target_properties(example1 PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHS}")