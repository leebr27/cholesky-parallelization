# Part 5: Additional Implementation — mpi4py

Python port of the distributed-memory Cholesky factorization using
**mpi4py** (Python MPI bindings) with **NumPy** for local linear algebra.
The algorithm is identical to Part 3 (C++ MPI), enabling a direct
comparison of performance, development effort, and abstraction level.

## Files
- `cholesky_mpi4py.py` — ~115 lines of Python. The same 1D column-cyclic
  right-looking Cholesky from Part 3:
  1. `comm.Bcast` the factored column from its owner to all ranks.
  2. Each rank applies the rank-1 trailing update to its owned columns
     via a single vectorised NumPy slice: `L_loc[k:, lk] -= buf[k-j:] * Lkj`.

  The NumPy operations run inside C/BLAS kernels; Python interpreter overhead
  lives in the per-column loop driver (N iterations per factorization).
  Correctness verified with `np.linalg.norm(A - L @ L.T)`.

- `submit.slurm` — sources the class Spack environment
  (`/shared/spack/share/spack/setup-env.sh`), which provides Python 3.12,
  mpi4py 4.1.1, NumPy 2.3.5, and OpenMPI 5. Launches via
  `srun --mpi=pmix_v5`. Mirrors the Part 3 Slurm script exactly for
  apples-to-apples comparison.

## Reproducing
```
sbatch submit.slurm
```
Output is written to `cholesky_mpi4py_<jobid>.out`.

## Results (`cholesky_mpi4py_108385.out`)

### Strong Scaling (N = 1024)

C++ MPI baseline (Part 3, same cluster) shown for direct comparison.

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

All residuals ≈ 1.70e-10, consistent with expected floating-point error for
this matrix size. The slight difference from the C++ residual (1.48e-10) is
because NumPy's `default_rng(42)` produces a different SPD matrix than C's
`srand(42)/rand()`; both are correct.

## Analysis

### Performance

mpi4py is **8–15× slower** than hand-written C++ MPI for this algorithm.
The overhead has two sources:

1. **Python loop driver**: the outer `for j in range(N)` loop runs 1024
   iterations in the Python interpreter per factorization. Each iteration
   involves Python attribute lookups, NumPy slice creation, and MPI call
   dispatch — roughly constant overhead per column that dwarfs the actual
   BLAS work at small N.

2. **MPI binding overhead**: `comm.Bcast` in mpi4py adds one Python function
   call and a buffer-descriptor lookup per collective, whereas `MPI_Bcast` in
   C is a direct library call.

The 4-rank case is slower than 2-rank (1.10 s vs 0.91 s) because with more
ranks each rank handles fewer columns, so the NumPy trailing update per rank
is smaller and finishes faster — but the Python loop + broadcast overhead
per column stays constant, so communication cost grows relative to compute.
At 16 ranks the larger broadcast fan-out and reduced work-per-rank push
mpi4py toward respectable scaling (2× faster than 8 ranks).

### Development effort

The mpi4py implementation is ~115 lines of Python versus ~215 lines of C++
(Part 3). Key simplifications:
- No manual memory management (`new`/`delete`, raw pointers).
- NumPy handles column layout — no explicit index arithmetic.
- `comm.Bcast` accepts a NumPy array directly; no `MPI_Datatype` boilerplate.
- Built-in `np.linalg.norm` for the residual check.

For prototyping or research code where correctness matters more than raw
throughput, mpi4py offers a compelling trade-off. For production at scale,
the C++ implementation is clearly preferred.
