# Performany Analysis

Build with `sm_80` for Nvidia A100 systems

```
nvcc -arch=sm_80 -o multi multi1.cu
```

Building with a newer sm models results in `Fatal error: execution error (no kernel image is available for execution 
on the device at multi1.cu:94)`, see <https://developer.nvidia.com/cuda/gpus>

# Multi gpu kernels 

```
kernel     device   time 
multi.cu   1xA10    0.018064
multi1.cu  1xA10    Fatal error: allocation error (invalid device ordinal at multi1.cu:76)
multi1.cu  8xA100   0.006094
```
