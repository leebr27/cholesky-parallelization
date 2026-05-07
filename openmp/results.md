# Part 2: OpenMP Cholesky Factorization — Results

## Reproducing
```
sbatch submit.slurm
sbatch submit_vtune.slurm
```
Output is written to `cholesky_openmp_<jobid>.out` and `cholesky_openmp_vtune_<jobid>.out`.

## Results (`cholesky_openmp_108269.out`)

### Strong Scaling (N = 1024)

| Threads | Best time (s) | GFLOP/s | Speedup | Efficiency |
|---------|--------------|---------|---------|------------|
| 1       | 0.854345     | 0.419   | 1.00×   | 100%       |
| 2       | 0.350302     | 1.022   | 2.44×   | 122%       |
| 4       | 0.172024     | 2.081   | 4.97×   | 124%       |
| 8       | 0.089407     | 4.003   | 9.56×   | 120%       |
| 16      | 0.083800     | 4.271   | 10.19×  | 64%        |

### Weak Scaling (N³/threads = constant, base N=512 at 1 thread)

| Threads | N    | Best time (s) | GFLOP/s | Efficiency |
|---------|------|--------------|---------|------------|
| 1       | 512  | 0.061153     | 0.732   | 100%       |
| 2       | 645  | 0.024440     | 3.660   | 250%       |
| 4       | 812  | 0.025532     | 6.990   | 239%       |
| 8       | 1024 | 0.091712     | 3.903   | 67%        |
| 16      | 1290 | 0.279336     | 2.562   | 22%        |

## VTune Profiling (`cholesky_openmp_vtune_108828.out`)

Intel VTune Profiler 2025.8.0 hotspots, N=1024, 16 threads.

| Function | CPU Time | % of CPU Time | Type |
|---|---|---|---|
| `generate_spd_matrix` | 9.790 s | 70.1% | Effective (serial) |
| `cholesky_openmp._omp_fn.0` | 2.790 s | 20.0% | Effective (parallel) |
| `verify` | 0.670 s | 4.8% | Effective (serial) |
| `gomp_team_barrier_wait_end` | 0.670 s | 4.8% | Spin / Imbalance |
| Thread creation + other | 0.040 s | 0.2% | Overhead |

- Elapsed wall time: 10.708 s.
- Factorization itself: 0.055 s at 16 threads (6.6 GFLOP/s).
