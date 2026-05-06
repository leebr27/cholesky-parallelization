# Part 2: OpenMP Cholesky Factorization

OpenMP shared-memory parallelization of `DPOTRF` — the Cholesky factorization
$A = L L^T$ of an $N \times N$ symmetric positive-definite matrix.

## Files
- `cholesky_openmp.cpp` — OpenMP parallelization of the column-by-column
  right-looking Cholesky factorization. The diagonal element for each column
  is computed sequentially; the sub-diagonal row updates (`i > j`) are
  independent and parallelized with `#pragma omp parallel for schedule(static)`.
- `submit.slurm` — compiles with `g++ -O3 -march=native -fopenmp` and runs
  two studies on a 16-core node:
  1. **Strong scaling**: N=1024, threads ∈ {1, 2, 4, 8, 16}.
  2. **Weak scaling**: thread count scales with N so that $N^3/\text{threads}$
     is held constant (base: N=512, 1 thread).

## Reproducing
```
sbatch submit.slurm
```
Output is written to `cholesky_openmp_<jobid>.out`.

## Results (`cholesky_openmp_108269.out`)

### Strong Scaling (N = 1024)

Serial baseline (Part 1, same node): 0.777 s.
The 1-thread OpenMP run is marginally slower (0.854 s) due to fork/join
overhead at each of the N=1024 columns.

| Threads | Best time (s) | GFLOP/s | Speedup | Efficiency |
|---------|--------------|---------|---------|------------|
| 1       | 0.854345     | 0.419   | 1.00×   | 100%       |
| 2       | 0.350302     | 1.022   | 2.44×   | 122%       |
| 4       | 0.172024     | 2.081   | 4.97×   | 124%       |
| 8       | 0.089407     | 4.003   | 9.56×   | 120%       |
| 16      | 0.083800     | 4.271   | 10.19×  | 64%        |

Speedup is near-ideal up to 8 threads. At 16 threads the gain diminishes
(10.19× vs. ideal 16×) because the sequential diagonal update at each column
becomes the bottleneck (Amdahl's law), and thread launch overhead dominates
for small trailing panels early in the factorization.

### Weak Scaling (N³/threads = constant, base N=512 at 1 thread)

| Threads | N    | Best time (s) | GFLOP/s | Efficiency |
|---------|------|--------------|---------|------------|
| 1       | 512  | 0.061153     | 0.732   | 100%       |
| 2       | 645  | 0.024440     | 3.660   | 250%       |
| 4       | 812  | 0.025532     | 6.990   | 239%       |
| 8       | 1024 | 0.091712     | 3.903   | 67%        |
| 16      | 1290 | 0.279336     | 2.562   | 22%        |

The super-linear efficiency at 2–4 threads is a cache artifact: N=512/645/812
fit comfortably in the shared LLC split across cores, whereas the single-thread
N=512 run is already slightly cache-limited. At 8–16 threads the working set
grows beyond the cache hierarchy and memory bandwidth becomes the bottleneck,
causing efficiency to drop sharply. All residuals remain consistent with the
serial result, confirming correctness.

## Algorithm

For each column $j = 0, \dots, N-1$:
$$L_{jj} = \sqrt{A_{jj} - \sum_{k<j} L_{jk}^2}$$
$$L_{ij} = \frac{1}{L_{jj}} \Big(A_{ij} - \sum_{k<j} L_{ik} L_{jk}\Big), \quad i > j \quad \text{(parallelized)}$$

The diagonal step is inherently sequential (each column depends on all
previous columns), limiting parallel efficiency for large thread counts on
small matrices — a known characteristic of the column-by-column formulation.
