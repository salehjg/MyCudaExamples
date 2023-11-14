# Details
**Base Kernel**: 3


## Modifications
### Fully Unrolled Loops
All the loops in the kernel are fully unrolled. This is made possible utilizing template functions and `constexpr`s.

## Summary
This kernel uses:
- `__constant__` array for `sliceLens`.
- `__shared__` array for `indices` to avoid local memory space for this array.
- Shared memory padding to avoid bank conflicts (`SMEM_PADDING`).
- Templates to move some computations to compile-time.
- Compile-time for-loops to fully unroll everything and move more computation to compile-time.

# Results
Per device results are listed below.

## Device `RXT3060-12G`
- **Args**: `../../data/golds/ 1000`
- **NVCC Opts**: `-O3`
- **Average (ms)**: 0.0576104
