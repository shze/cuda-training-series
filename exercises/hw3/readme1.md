# Profiling

Run profiler with

```
sudo ncu ./add --gridsize=1 --blocksize=1
sudo ncu --section SpeedOfLight --section MemoryWorkloadAnalysis ./add --gridsize=1 --blocksize=1
```

# Results on one A10

```
gridsize blocksize duration
1        1         4.79s
1        1024      19.20ms
160      1024      899.81us
```


