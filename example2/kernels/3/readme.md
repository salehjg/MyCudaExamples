# Details
**Base Kernel**: 2


## Modifications
### Use of Shared Memory Space For `indices[]`
Since accesses to this array involve runtime indices, `nvcc` places it in Local memory space, meaning that any access to this array should go through all the cache hierarchy to the global memory.
To avoid this, we can allocate a workspace in the shared memory dynamically and use chunks of this workspace locally and privately in each thread. Since we are not sharing data in-between threads in a block, there is no need for memory fences or block-level synchronizations.


# Results
Per device results are listed below.

## Device `RXT3060-12G`
- **Args**: `../../data/golds/ 1000`
- **NVCC Opts**: `-O3`
- **Average (ms)**: 0.0808938


