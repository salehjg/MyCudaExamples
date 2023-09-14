set(KERNEL 1)
add_executable(example1 "kernel${KERNEL}.cu" main.cpp)
target_link_libraries(example1 common)
set_target_properties(example1 PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHS}")