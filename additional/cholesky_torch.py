"""
PyTorch Cholesky factorization benchmark.

Uses torch.linalg.cholesky, which dispatches to:
  - LAPACK (MKL/OpenBLAS) when run on CPU
  - cuSOLVER (NVIDIA's tuned LAPACK port) when run on GPU
Both backends are highly optimized blocked implementations.

This serves as the "additional implementation" for the final project,
contrasting a high-level framework (one library call) with the hand-rolled
serial/OpenMP/MPI/CUDA codes.

Usage:
    python cholesky_torch.py [N] [repeats] [device]
        device = "cpu" or "cuda"  (default: cuda if available else cpu)
"""

import sys
import time
import torch


def generate_spd_matrix(N: int, device: torch.device, dtype=torch.float64) -> torch.Tensor:
    """Generate a random SPD matrix matching the C++ benchmarks.

    A = M^T M + N*I  (well-conditioned, large diagonal dominance)
    """
    g = torch.Generator(device=device).manual_seed(42)
    M = torch.rand((N, N), device=device, dtype=dtype, generator=g)
    A = M.T @ M
    A.diagonal().add_(float(N))
    # Symmetrize defensively (M^T M is already symmetric in exact arithmetic)
    A = 0.5 * (A + A.T)
    return A


def benchmark(N: int, repeats: int, device_str: str) -> None:
    device = torch.device(device_str)
    dtype = torch.float64
    print(f"PyTorch Cholesky: N = {N}, repeats = {repeats}, "
          f"device = {device}, dtype = {dtype}")

    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")

    A = generate_spd_matrix(N, device, dtype)

    # Warm-up (loads the cuSOLVER/MKL kernel and any JIT/autotuning state)
    _ = torch.linalg.cholesky(A)
    if device.type == "cuda":
        torch.cuda.synchronize()

    best_time = float("inf")
    total_time = 0.0
    L = None
    for r in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        L = torch.linalg.cholesky(A)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        total_time += elapsed
        best_time = min(best_time, elapsed)
        print(f"  run {r+1}: {elapsed:.6f} s")

    # Verify: Frobenius norm of (A - L L^T)
    residual = torch.linalg.norm(A - L @ L.T).item()
    gflops = (1.0 / 3.0) * (N ** 3) / best_time / 1e9

    print(f"  Best time:  {best_time:.6f} s")
    print(f"  Avg time:   {total_time / repeats:.6f} s")
    print(f"  GFLOP/s:    {gflops:.3f}")
    print(f"  Residual:   {residual:.2e}")


def main() -> None:
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    if len(sys.argv) > 3:
        device_str = sys.argv[3]
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    benchmark(N, repeats, device_str)


if __name__ == "__main__":
    main()
