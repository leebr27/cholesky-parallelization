# Part 1: Serial Cholesky Factorization

Serial baseline implementation of `DPOTRF` — the Cholesky factorization
$A = L L^T$ of an $N \times N$ symmetric positive-definite matrix.

## Files
- `cholesky_serial.cpp` — single-threaded, column-by-column right-looking
  Cholesky factorization in C++ with column-major storage.
- `submit.slurm` — compiles with `g++ -O3 -march=native` and runs the
  executable across problem sizes $N \in \{256, 512, 1024, 2048\}$ on one
  CPU core (`OMP_NUM_THREADS=1`) for 3 timed iterations after a warm-up.

## Reproducing
```
sbatch submit.slurm
```
The job writes `cholesky_serial_<jobid>.out` containing best/average wall
times, achieved GFLOP/s (using the standard $N^3/3$ FLOP count), and the
Frobenius residual $\lVert A - L L^T \rVert_F$ for correctness.

## Results (`cholesky_serial_108259.out`)

| N    | Best time (s) | Avg time (s) | GFLOP/s | Residual  |
|------|--------------|-------------|---------|-----------|
| 256  | 0.003062     | 0.003079    | 1.827   | 5.10e-12  |
| 512  | 0.058234     | 0.058305    | 0.768   | 2.65e-11  |
| 1024 | 0.776923     | 0.781225    | 0.461   | 1.48e-10  |
| 2048 | 12.502958    | 12.802376   | 0.229   | 8.40e-10  |

All residuals are well below machine epsilon scaled by $N$, confirming
correctness. GFLOP/s decreases with $N$ due to growing cache pressure — the
working set for the trailing matrix update no longer fits in L2/L3 cache as
$N$ increases, which is expected for a naive column-by-column implementation
and motivates the parallel implementations.

## Algorithm
For each column $j = 0, \dots, N-1$:
$$L_{jj} = \sqrt{A_{jj} - \sum_{k<j} L_{jk}^2}, \qquad
L_{ij} = \frac{1}{L_{jj}} \Big(A_{ij} - \sum_{k<j} L_{ik} L_{jk}\Big), \; i > j.$$
This serves as the correctness reference and timing baseline for the
OpenMP, MPI, CUDA, and additional implementations.
