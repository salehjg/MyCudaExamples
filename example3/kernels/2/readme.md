# Details
**Base Kernel**: 1


## Modifications
None.

### Issue1
works with `len0 = 16777218`  
``` 
gird: 16385
block: 1024
len: 16778240
Padded Words: 1022
Gold: 16384
UUT: 16384
```

fails with `len = len0 + 1 = 16777219`
```
gird: 16385
block: 1024
len: 16778240
Padded Words: 1021
Gold: 16384
UUT: 16384
```


# Results
Per device results are listed below.

## Device `RXT3060-12G`
- **Args**: `67108864`
- **NVCC Opts**: `-O3`
- **Device Time (ms)**: 117.723


