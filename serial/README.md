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

## Algorithm
For each column $j = 0, \dots, N-1$:
$$L_{jj} = \sqrt{A_{jj} - \sum_{k<j} L_{jk}^2}, \qquad
L_{ij} = \frac{1}{L_{jj}} \Big(A_{ij} - \sum_{k<j} L_{ik} L_{jk}\Big), \; i > j.$$
This serves as the correctness reference and timing baseline for the
OpenMP, MPI, CUDA, and additional implementations.
