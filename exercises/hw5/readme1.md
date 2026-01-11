# Profiling

Run profiler with

```
sudo ncu ./matrix_sums
```

# Sum reduction results on one A10

```
kernel     duration_8M duration_160k   duration_32M
atomic_red     28.93ms      568.96us 115.71ms/ERROR
reduce_a       85.73us       10.08us
reduce_ws      84.51us        7.94us
```

# Matrix row sum results on one A 10

```
kernel      duration
hw4_row_sum   6.74ms
hw5_row_sum   2.10ms
col_sum       2.75ms
```
