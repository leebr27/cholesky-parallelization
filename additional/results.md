# Part 5: Additional Implementation (mpi4py) — Results

## Reproducing
```
sbatch submit.slurm
```
Output is written to `cholesky_mpi4py_<jobid>.out`.

## Results (`cholesky_mpi4py_108385.out`)

### Strong Scaling (N = 1024)

| Ranks | mpi4py time (s) | mpi4py GFLOP/s | C++ MPI time (s) | Overhead |
|-------|----------------|----------------|-----------------|----------|
| 2     | 0.914531       | 0.391          | 0.093178        | 9.8×     |
| 4     | 1.103367       | 0.324          | 0.074562        | 14.8×    |
| 8     | 0.569766       | 0.628          | 0.046557        | 12.2×    |
| 16    | 0.288484       | 1.241          | 0.036596        | 7.9×     |

### Large Problem (N = 2048, 16 ranks across 2 nodes)

| Ranks | N    | mpi4py time (s) | mpi4py GFLOP/s | C++ MPI time (s) | Overhead |
|-------|------|----------------|----------------|-----------------|----------|
| 16    | 2048 | 1.185425       | 2.415          | 0.149886        | 7.9×     |

Residuals: 1.70e-10 for N=1024 strong-scaling runs; 7.14e-10 for N=2048.
