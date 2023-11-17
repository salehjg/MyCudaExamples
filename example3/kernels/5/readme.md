# Details
**Base Kernel**: 4

## Notes:
- The kernel be optimized further by using Wrap Shuffle instructions in the path for `0<=tid<32`. 
- Replacing atomic instructions with nested kernel launches.
- Using [Thread Fence Reduction](https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/threadFenceReduction/threadFenceReduction_kernel.cuh).

## Modifications
- Fully unrolled loops.
- Mapping of multiple slices per block (`kernel.h: fixedSlicesPerBlock`).


# Results
Per device results are listed below.

## Device `RXT3060-12G`
- **Args**: `67108864`
- **NVCC Opts**: `-O3`
- **Device Time (ms)**: 0.940832


