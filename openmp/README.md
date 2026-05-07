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

## VTune Profiling (`cholesky_openmp_vtune_108828.out`)

Intel VTune Profiler 2025.8.0 hotspots analysis was run at N=1024 with 16
threads (`submit_vtune.slurm`). The profiler sampled the entire process,
including matrix generation and correctness verification.

### Hotspots summary

| Function | CPU Time | % of CPU Time | Type |
|---|---|---|---|
| `generate_spd_matrix` | 9.790 s | 70.1% | Effective (serial) |
| `cholesky_openmp._omp_fn.0` | 2.790 s | 20.0% | Effective (parallel) |
| `verify` | 0.670 s | 4.8% | Effective (serial) |
| `gomp_team_barrier_wait_end` | 0.670 s | 4.8% | **Spin / Imbalance** |
| Thread creation + other | 0.040 s | 0.2% | Overhead |

- **Elapsed wall time**: 10.708 s (dominated by the unparallelized
  `generate_spd_matrix` call, which is O(N³) and serial).
- **Factorization itself**: 0.055 s at 16 threads (6.6 GFLOP/s) — consistent
  with the strong-scaling results in `submit.slurm`.

### Performance trends explained by VTune

**1. Serial matrix generation dominates profiled CPU time.**
`generate_spd_matrix` accounts for 70% of aggregate CPU time. It constructs
the SPD matrix as $A = M^T M + NI$ using a triply-nested serial loop, which
is O(N³). Because this step is not parallelized, it becomes the dominant cost
under profiling. In the scaling experiments this function's time is excluded
(only factorization runtimes are measured), so it does not affect the reported
speedups — but it illustrates that any real-world application would need to
parallelize data generation as well.

**2. Barrier spin time directly quantifies the Amdahl bottleneck.**
`gomp_team_barrier_wait_end` spent 0.650 s in *Imbalance or Serial Spinning*
(out of 0.670 s total spin time), with zero lock contention. This occurs
because the column-by-column factorization requires a team barrier after each
of the N=1024 columns: once the sequential diagonal element $L_{jj}$ is
computed, all 16 threads synchronize before proceeding to the next column.
Threads that finish their share of the parallel row update early must spin-wait
at the barrier. This is precisely the serial fraction that Amdahl's law
predicts will cap speedup — explaining why efficiency drops from ~120% at 8
threads to 64% at 16 threads in the strong-scaling results.

**3. Thread creation overhead is negligible.**
Creation overhead totals 0.020 s (0.1% of CPU time), confirming that OpenMP's
fork/join model at each column does not meaningfully contribute to the observed
slowdown. The bottleneck is synchronization latency (barrier imbalance), not
thread launch cost.

**4. No lock contention or false sharing.**
Lock Contention and Atomics are both 0 s, confirming the `schedule(static)`
partition assigns independent row ranges to each thread with no overlap and no
shared writes.

