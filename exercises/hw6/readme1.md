# Performany Analysis

Run Nvidia systems with

```
sudo nsys profile --stats=true array_inc
```

# Array increment results on one A 10

```
kernel             duration
array_inc          594914ns
1_managed        23308182ns
2_prefetch         592737ns
3_10k_prefetch 5918710385ns
```
