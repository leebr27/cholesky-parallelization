# Part 4: CUDA Cholesky Factorization

GPU-accelerated implementation of `DPOTRF` — the Cholesky factorization
$A = L L^T$ of an $N \times N$ symmetric positive-definite matrix — using CUDA
kernels for the trailing-matrix rank-1 update on an NVIDIA GPU.

## Files
- `cholesky_cuda.cu` — CUDA C++ implementation. The matrix is held in device
  memory (column-major) and factored in place. For each column `j`:
  1. **Diagonal step** (host): one `double` is read back via `cudaMemcpy`,
     the square root is taken on the CPU, and the value is written back.
  2. **`scale_column_kernel`** (1D, 256 threads/block): divides sub-diagonal
     entries `L(i,j)` for `i > j` by `L(j,j)`.
  3. **`rank1_update_kernel`** (2D, 16×16 blocks): updates the lower triangle
     of the trailing submatrix: `L(i,k) -= L(i,j)·L(k,j)` for `i ≥ k > j`.

  The rank-1 update is the dominant step — $O((N-j)^2)$ parallel work per
  column — and drives the GFLOP/s numbers. Correctness is verified by copying
  `L` back to the host and computing $\lVert A - LL^T \rVert_F$.

- `submit.slurm` — requests 1 GPU node (`--gres=gpu:1`), compiles with
  `nvcc -O3 -arch=sm_70` (sm_70 as a baseline; PTX-JIT to newer archs at
  runtime), and sweeps N ∈ {512, 1024, 2048, 3072} with 3 timed runs each.

## Reproducing
```
sbatch submit.slurm
```
Output is written to `cholesky_cuda_<jobid>.out`.

## Results (`cholesky_cuda_108357.out`)

GPU: **NVIDIA L4** (CC 8.9, 58 SMs, CUDA 12.8). Peak FP64: ~30 TFLOP/s.
Serial baseline at N=1024: 0.777 s. All residuals consistent with serial,
OpenMP, and MPI results.

| N    | Best time (s) | GFLOP/s | Speedup vs serial |
|------|--------------|---------|-------------------|
| 512  | 0.008844     | 5.059   | 0.35× (overhead)  |
| 1024 | 0.021121     | 16.946  | 36.8×             |
| 2048 | 0.072146     | 39.688  | 173×              |
| 3072 | 0.240840     | 40.125  | ~175×             |

GFLOP/s plateaus near 40 at N=2048–3072, indicating the 80 CUDA SMs are
compute-saturated for the rank-1 update kernel. At N=512, kernel-launch
overhead per column dominates and makes the GPU slower than the serial CPU
— a known characteristic of column-by-column Cholesky on GPUs. At N=1024
and beyond, the per-launch work grows as $(N-j)^2$, amortizing the launch
cost and delivering 37–175× speedup over the serial baseline.

## Performance Discussion

The naive column-by-column kernel achieves ~40 GFLOP/s sustained (FP64),
which is ~0.13% of the L4's theoretical FP64 peak. The sequential column
dependency (each column must complete before the next begins) serializes the
N kernel launches and fundamentally limits throughput — this is the same
Amdahl bottleneck seen in the OpenMP results, but amplified by GPU launch
overhead. Production GPU LAPACK libraries (cuSOLVER `cusolverDnDpotrf`)
address this with a blocked formulation where a CPU-resident panel
factorization is interleaved with large GPU `DSYRK`/`DGEMM` calls, achieving
much closer to peak bandwidth/compute. Compared to MPI at N=2048 (0.150 s,
16 ranks), the single-GPU time of 0.072 s is 2× faster, demonstrating the
GPU's advantage for dense linear algebra at this scale.

## Algorithm

For each column $j = 0, \dots, N-1$:
$$L_{jj} = \sqrt{A_{jj} - \sum_{k<j} L_{jk}^2} \quad \text{(host)}$$
$$L_{ij} \mathrel{/}= L_{jj}, \quad i > j \quad \texttt{(scale\_column\_kernel)}$$
$$L_{ik} \mathrel{-}= L_{ij} \cdot L_{kj}, \quad i \ge k > j \quad \texttt{(rank1\_update\_kernel)}$$
