# Details
**Base Kernel**: 2


## Modifications
To fix high `warp state stats / stall MIO throttle`, we need to decrease the total number of accesses to the shared memory.
Since, normally each thread calculates one element of the output tensor, the data loaded from a row/column of the shared mem. buffer is not reused for the next element of the output tensor.
In short, here, we try to map `RxR` elements of the output tensor to each thread and reuse the fetched data from the shared mem. buffers by storing them on registers.
There is more behind this approach and [this blog post](https://siboehm.com/articles/22/CUDA-MMM) explains it quite nicely.




# Results
Per device results are listed below.

## Device `RXT3060-12G`
- **Args**: `1024 1024 1024`
- **NVCC Opts**: `-O3`
- **Device Time (ms)**: 0.65472


