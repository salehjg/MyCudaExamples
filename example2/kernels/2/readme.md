# Details
**Base Kernel**: 1


## Modifications
### Use of Constant Memory Space for `SliceLen`
Compared to V1, here we declare `SliceLensDevice` statically and globally with `__constant__` qualifier.
Meaning any access to this array (on the global memory / constant region) will be cached and optimized for broadcasting.
`cudaMemcpyToSymbol` is used to copy the data from the host array into the static symbol `SliceLensDevice`.
Please note that the address of this symbol could be passed into the kernel as an argument as well (instead of accessing
it as a global static array), but the address should be acquired with `cudaGetSymbolAddress`.

### Use of Loop-unrolling (only for one loop)
The loop below (in V1):
```c++
for (int axis = 0; axis < rank1; axis++) {
    indices[axis] = ComputeAxisIndex(axis, idx, sliceLens, rank1);
}
```
is converted to the following in this version of the kernel:
```c++
constexpr int UNROLLED_RANK1 = 5;
assert(rank1 < UNROLLED_RANK1);
// ...
#pragma unroll
for (int axis = 0; axis < UNROLLED_RANK1; axis++) {
    if(axis<rank1){
        indices[axis] = ComputeAxisIndex(axis, idx, sliceLens, rank1);
    }
}
```

# Results
Per device results are listed below.

## Device `RXT3060-12G`
- **Args**: `../../data/golds/ 1000`
- **NVCC Opts**: `-O3`
- **Average (ms)**: 0.10433


