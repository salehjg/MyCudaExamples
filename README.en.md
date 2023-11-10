# Faradars - CUDA Basics

## Incompatible GCC/G++
In case your main `GCC/G++` version is not compatible with `nvcc` (like `nvcc` 12 and `GCC/G++` 13), set these ENV variables before using `cmake ..`:
```
export HOST_COMPILER=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12
```

## Useful Refs
- [Nvidia - Modern CMake and CUDA](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9444-build-systems-exploring-modern-cmake-cuda-v2.pdf)