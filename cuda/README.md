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
| 512  | 0.008844     | 5.059   | 6.6×              |
| 1024 | 0.021121     | 16.946  | 36.8×             |
| 2048 | 0.072146     | 39.688  | 173×              |
| 3072 | 0.240840     | 40.125  | — (no serial run) |

Speedups are computed against same-N serial baselines from `serial/`
(0.058234 s, 0.776923 s, 12.502958 s for N=512, 1024, 2048). Serial was
not run at N=3072, so no speedup figure is reported there. GFLOP/s
plateaus near 40 at N=2048–3072, indicating the SMs are compute-saturated
for the rank-1 update kernel. At N=512, the GPU achieves only ~5 GFLOP/s
— about one-eighth of its own asymptotic throughput — because per-column
launch and memcpy overhead is comparable to the actual rank-1 update
work; the 6.6× speedup over serial is real but well below what the same
kernel sustains at larger N. By N=1024 and beyond, the per-launch work
grows as $(N-j)^2$, amortizing the launch cost and delivering 37–173×
speedup over the serial baseline.

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

## Nsight Systems Profiling (`cholesky_cuda_nsys_108855.out`)

NVIDIA Nsight Systems 2024.6.2 was run at N=2048 with 3 repeats on an NVIDIA
A10G (sm_86, CUDA 12.8) via `submit_nsys.slurm`.

### CUDA GPU kernel summary

| Kernel | GPU Time | Instances | Avg (ns) | % of GPU Time |
|---|---|---|---|---|
| `rank1_update_kernel` | 254.0 ms | 8188 | 31,021 | **95.1%** |
| `scale_column_kernel` | 13.2 ms | 8188 | 1,607 | 4.9% |

8188 instances ≈ 2048 columns × 4 runs (1 warmup + 3 timed), consistent with
the column-by-column loop.

### CUDA memcpy summary (GPU-side transfer time)

| Operation | GPU Time | Count | Avg (ns) |
|---|---|---|---|
| Device-to-Host | 13.86 ms | 8193 | 1,692 |
| Host-to-Device | 13.85 ms | 8196 | 1,690 |

The D2H count (8193) matches one scalar `double` read per column per run to
retrieve `L(j,j)` for the host-side square root. The H2D count (8196) includes
one large initial matrix copy (33.6 MB) plus per-column `inv_ljj` writes back.
GPU-side transfer time is small (~28 ms total) — the real cost is on the host.

### CUDA API summary (host-side overhead)

| API Call | Host Time | Calls | Avg (ns) | % of API Time |
|---|---|---|---|---|
| `cudaMemcpy` | 392.4 ms | 16,389 | 23,945 | **59.8%** |
| `cudaLaunchKernel` | 134.4 ms | 16,376 | 8,210 | 20.5% |
| `cudaMalloc` | 125.7 ms | 1 | 125,728,165 | 19.2% |

### Performance bottlenecks identified

**1. Per-column synchronous `cudaMemcpy` is the dominant host bottleneck.**
Each of the N=2048 columns issues a synchronous `cudaMemcpy` D2H to read
`L(j,j)` from the device. Synchronous memcpy implicitly calls
`cudaDeviceSynchronize`, stalling the host until the GPU finishes the previous
column's `rank1_update_kernel` before the host can compute the diagonal and
proceed. The Nsight data shows `cudaMemcpy` consuming 59.8% of total API
time (392 ms) at an average of 23.9 µs per call — far larger than the actual
1.7 µs GPU transfer time. This host round-trip is serialized N times,
preventing any pipelining between columns and is the root cause of the
column-count-driven overhead seen in the N=512 scaling results.

**2. `rank1_update_kernel` correctly dominates GPU time at 95.1%.**
The trailing-matrix rank-1 update does $O((N-j)^2)$ work per column and
accounts for virtually all useful GPU computation. Its average duration grows
from 1.2 µs (final column, tiny 2×2 update) to 132 µs (early columns, large
trailing submatrix), spanning a 100× range — consistent with the $(N-j)^2$
work scaling. This wide variance in kernel duration means early large kernels
fully saturate the A10G's 80 SMs while later small kernels leave most of the
GPU idle, partly explaining the gap between achieved (25 GFLOP/s) and
theoretical peak.

**3. `scale_column_kernel` overhead is negligible at 4.9%.**
The 1D column-scaling kernel averages 1.6 µs and contributes 13 ms across all
instances — small enough to ignore for optimization purposes.

**4. `cudaLaunchKernel` host overhead is measurable but secondary.**
Kernel launch overhead totals 134 ms (avg 8.2 µs × 16,376 launches = 2
launches per column × N × repeats). This is about one-third of the memcpy
overhead and confirms that the synchronization cost of the diagonal memcpy,
not the launch overhead itself, is the primary limiter.

**Summary:** the per-column host sync imposed by the synchronous diagonal
`cudaMemcpy` serializes N=2048 GPU-CPU round-trips, each taking ~24 µs
host-side. Eliminating this (e.g. by computing the diagonal on the GPU with a
small device kernel and passing `inv_ljj` directly to the next kernel) would
remove the dominant bottleneck and allow larger sustained throughput.

