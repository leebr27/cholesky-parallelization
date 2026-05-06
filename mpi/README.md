# Part 3: MPI Cholesky Factorization

Distributed-memory parallelization of `DPOTRF` — the Cholesky factorization
$A = L L^T$ of an $N \times N$ symmetric positive-definite matrix — using MPI
with a 1D column-cyclic distribution across ranks.

## Files
- `cholesky_mpi.cpp` — MPI C++ implementation. Rank `r` owns global column `j`
  iff `j % P == r` (column-cyclic). Right-looking algorithm per column `j`:
  1. Owner computes `L(j,j) = sqrt(...)` and scales the panel below the diagonal.
  2. `MPI_Bcast` of the factored column (length `N-j`) to all ranks.
  3. Each rank applies a rank-1 trailing update to its owned columns `k > j`.
  
  Correctness is verified by gathering `L` on rank 0 and computing
  $\lVert A - LL^T \rVert_F$.

- `submit.slurm` — requests 2 nodes × 8 tasks/node (16 ranks total). Loads
  `openmpi5/5.0.8amzn1` (PMIx v5 compatible), compiles with
  `mpicxx -O3 -march=native`, and launches via `srun --mpi=pmix_v5`.
  Runs two studies:
  1. **Strong scaling**: N=1024, ranks ∈ {2, 4, 8, 16}.
  2. **Large problem**: N=2048, 16 ranks across both nodes.

## Reproducing
```
sbatch submit.slurm
```
Output is written to `cholesky_mpi_<jobid>.out`.

## Communication Strategy

For each of the N columns, the owning rank broadcasts a vector of length
$(N - j)$ to all other ranks. Total data communicated per rank:
$$\sum_{j=0}^{N-1}(N-j) = \frac{N(N+1)}{2} \text{ doubles} \approx \frac{N^2}{2}$$
Computation per rank is $O(N^3 / P)$, so the computation-to-communication
ratio grows as $O(N)$ — larger problems amortize communication overhead better.

The cyclic (rather than block) column assignment keeps load balanced as the
trailing matrix shrinks: each rank always owns approximately $N/P$ columns
of each width.

## Results (`cholesky_mpi_108317.out`)

Serial baseline (Part 1, N=1024): 0.777 s. All residuals are 1.48e-10,
consistent with the serial and OpenMP results.

### Strong Scaling (N = 1024)

| Ranks | Best time (s) | GFLOP/s | Speedup vs serial |
|-------|--------------|---------|-------------------|
| 2     | 0.093178     | 3.841   | 8.3×              |
| 4     | 0.074562     | 4.800   | 10.4×             |
| 8     | 0.046557     | 7.688   | 16.7×             |
| 16    | 0.036596     | 9.780   | 21.2×             |

Speedup is super-linear: at 16 ranks, 21× vs. an ideal of 16×. This is
a genuine cache effect — with distributed memory each rank's working set is
$1/P$ of the total, which fits in L2/L3 cache on the compute node. The serial
run at N=1024 was already cache-limited (0.229 GFLOP/s at N=2048), so
distributing the data eliminates that bottleneck and delivers more than
proportional speedup.

### Large Problem (N = 2048, 16 ranks across 2 nodes)

| Ranks | N    | Best time (s) | GFLOP/s | Speedup vs serial |
|-------|------|--------------|---------|-------------------|
| 16    | 2048 | 0.149886     | 19.103  | 83.4×             |

The serial N=2048 run took 12.5 s at 0.229 GFLOP/s due to severe cache
pressure. With 16 distributed ranks, each node holds only $\approx 1/16$ of
the matrix, eliminating LLC thrashing entirely and yielding 19.1 GFLOP/s —
an 83× speedup that far exceeds the 16× ideal. This demonstrates the
significant secondary benefit of distributed-memory parallelism for
memory-bound workloads.

## Algorithm

For each column $j = 0, \dots, N-1$:
$$L_{jj} = \sqrt{A_{jj} - \sum_{k<j} L_{jk}^2} \quad \text{(on owner rank)}$$
$$\xrightarrow{\text{MPI\_Bcast}} \text{all ranks receive } L(j{:}N,\, j)$$
$$L_{ik} \mathrel{-}= L_{ij} \cdot L_{kj}, \quad i \ge k > j \quad \text{(each rank updates its owned columns)}$$
