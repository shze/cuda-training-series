# Preparing lambda.ai instance

```
git pull  # or git clone on first run
bash scripts/install.bash
```

# Building and running

On lambda.ai instances, build (set the arch specific to the GPU in the instance) and run with 

```
nvcc -o hello hello1.cu
nvcc -arch=sm_86 -o hello hello1.cu
./hello
```

The streaming multiprocesser versions sm_X correspond to compute capabilities and differ by GPU, see 
https://developer.nvidia.com/cuda/gpus)

# Debugging

The compute sanitizer can help with debugging, see 
https://developer.nvidia.com/blog/efficient-cuda-debugging-using-compute-sanitizer-with-nvtx-and-creating-custom-tools/, 
and is run with

```
compute-sanitizer ./hello
```
