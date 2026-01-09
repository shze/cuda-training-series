# Profiling

Run profiler with

```
sudo ncu ./matrix_sums
sudo ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./matrix_sums
```

# Results on one A10

```
sum duration gmem_requests gmem_sector_loads
row   6.73m        8388608         268435393
col   2.76ms       8388608          33554432
```


