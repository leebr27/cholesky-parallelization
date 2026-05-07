# Part 3: MPI Cholesky Factorization — Results

## Reproducing
```
sbatch submit.slurm
```
Output is written to `cholesky_mpi_<jobid>.out`.

## Results (`cholesky_mpi_108317.out`)

### Strong Scaling (N = 1024)

| Ranks | Best time (s) | GFLOP/s | Speedup vs serial |
|-------|--------------|---------|-------------------|
| 2     | 0.093178     | 3.841   | 8.3×              |
| 4     | 0.074562     | 4.800   | 10.4×             |
| 8     | 0.046557     | 7.688   | 16.7×             |
| 16    | 0.036596     | 9.780   | 21.2×             |

### Large Problem (N = 2048, 16 ranks across 2 nodes)

| Ranks | N    | Best time (s) | GFLOP/s | Speedup vs serial |
|-------|------|--------------|---------|-------------------|
| 16    | 2048 | 0.149886     | 19.103  | 83.4×             |

All residuals are 1.48e-10, consistent with the serial and OpenMP results.
