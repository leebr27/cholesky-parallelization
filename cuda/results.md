# Part 4: CUDA Cholesky Factorization — Results

## Reproducing
```
sbatch submit.slurm
sbatch submit_nsys.slurm
```
Output is written to `cholesky_cuda_<jobid>.out` and `cholesky_cuda_nsys_<jobid>.out`.

## Results (`cholesky_cuda_108357.out`)

GPU: NVIDIA L4 (CC 8.9, 58 SMs, CUDA 12.8).

| N    | Best time (s) | GFLOP/s | Speedup vs serial |
|------|--------------|---------|-------------------|
| 512  | 0.008844     | 5.059   | 6.6×              |
| 1024 | 0.021121     | 16.946  | 36.8×             |
| 2048 | 0.072146     | 39.688  | 173×              |
| 3072 | 0.240840     | 40.125  | — (no serial run) |

## Nsight Systems Profiling (`cholesky_cuda_nsys_108855.out`)

NVIDIA Nsight Systems 2024.6.2, N=2048, 3 repeats, NVIDIA A10G (sm_86, CUDA 12.8).

### CUDA GPU kernel summary

| Kernel | GPU Time | Instances | Avg (ns) | % of GPU Time |
|---|---|---|---|---|
| `rank1_update_kernel` | 254.0 ms | 8188 | 31,021 | 95.1% |
| `scale_column_kernel` | 13.2 ms | 8188 | 1,607 | 4.9% |

### CUDA memcpy summary (GPU-side transfer time)

| Operation | GPU Time | Count | Avg (ns) |
|---|---|---|---|
| Device-to-Host | 13.86 ms | 8193 | 1,692 |
| Host-to-Device | 13.85 ms | 8196 | 1,690 |

### CUDA API summary (host-side overhead)

| API Call | Host Time | Calls | Avg (ns) | % of API Time |
|---|---|---|---|---|
| `cudaMemcpy` | 392.4 ms | 16,389 | 23,945 | 59.8% |
| `cudaLaunchKernel` | 134.4 ms | 16,376 | 8,210 | 20.5% |
| `cudaMalloc` | 125.7 ms | 1 | 125,728,165 | 19.2% |
