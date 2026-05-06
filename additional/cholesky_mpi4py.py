"""
mpi4py distributed Cholesky factorization (DPOTRF): A = L * L^T.

Same column-cyclic right-looking algorithm as the C++ MPI implementation
in Part 3, ported to Python with NumPy for the local kernels.  This is the
"additional implementation" — the comparison axis is ease of development
and performance overhead of a Python MPI binding versus hand-written C++.

Algorithm (1D column-cyclic, P = world size):
  Rank r owns global column j iff (j % P) == r, stored as columns in a
  local (N x my_ncols) NumPy array.

  for j in range(N):
      owner = j % P
      if rank == owner:
          col[j]    = sqrt(col[j])
          col[j+1:] /= col[j]
      Bcast L(j..N-1, j) from owner
      for each owned column k > j:
          col[k:] -= L(k:, j) * L(k, j)        # vectorised by NumPy

The per-column trailing update is a single NumPy outer-product-like
operation, so almost all FLOPs run inside the NumPy/BLAS C kernels rather
than Python interpreter loops.

Usage:
    mpirun -n P python cholesky_mpi4py.py [N] [repeats]
"""

import sys
import time
import numpy as np
from mpi4py import MPI


def generate_spd_matrix(N: int) -> np.ndarray:
    """Generate the same SPD matrix recipe used by the C++ benchmarks."""
    rng = np.random.default_rng(42)
    M = rng.random((N, N), dtype=np.float64)
    A = M.T @ M
    A[np.diag_indices(N)] += float(N)
    return np.asfortranarray(A)  # column-major to match C++ layout


def factorize(L_loc: np.ndarray, N: int, P: int, rank: int,
              comm: MPI.Comm) -> None:
    """In-place Cholesky factorization on the column-cyclically distributed
    matrix L_loc (shape (N, my_ncols), Fortran-ordered)."""
    col_buf = np.empty(N, dtype=np.float64)
    my_ncols = L_loc.shape[1]

    for j in range(N):
        owner = j % P
        length = N - j  # length of L(j..N-1, j)

        if rank == owner:
            lj = j // P
            col = L_loc[:, lj]
            diag = np.sqrt(col[j])
            col[j] = diag
            col[j + 1:] /= diag
            col_buf[:length] = col[j:]

        comm.Bcast([col_buf[:length], MPI.DOUBLE], root=owner)

        # Smallest k > j with k % P == rank
        if rank > j % P:
            k_start = (j // P) * P + rank
        else:
            k_start = (j // P + 1) * P + rank
        lk_start = k_start // P

        # Vectorised rank-1 update over each owned column k > j.
        # col_buf[0:length] holds L(j..N-1, j).  For owned column lk
        # (global k = lk*P + rank), update L(k:N, k) -= L(k:N, j) * L(k, j).
        for lk in range(lk_start, my_ncols):
            k = lk * P + rank          # global column index, > j
            Lkj = col_buf[k - j]       # L(k, j)
            # L_loc[k:, lk] -= col_buf[k-j : length] * Lkj
            L_loc[k:, lk] -= col_buf[k - j:length] * Lkj


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    P = comm.Get_size()

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    if rank == 0:
        print(f"mpi4py Cholesky: N = {N}, ranks = {P}, repeats = {repeats}",
              flush=True)

    # Each rank's local columns (column-cyclic).
    my_ncols = (N - rank + P - 1) // P
    A_loc = np.zeros((N, my_ncols), dtype=np.float64, order="F")

    # Rank 0 generates A and scatters columns to owners.
    if rank == 0:
        A_full = generate_spd_matrix(N)
    else:
        A_full = None

    for j in range(N):
        owner = j % P
        if rank == 0:
            col = np.ascontiguousarray(A_full[:, j])
            if owner == 0:
                A_loc[:, j // P] = col
            else:
                comm.Send([col, MPI.DOUBLE], dest=owner, tag=j)
        elif rank == owner:
            buf = np.empty(N, dtype=np.float64)
            comm.Recv([buf, MPI.DOUBLE], source=0, tag=j)
            A_loc[:, j // P] = buf

    # Warm-up
    L_loc = np.array(A_loc, copy=True, order="F")
    factorize(L_loc, N, P, rank, comm)
    comm.Barrier()

    best = float("inf")
    total = 0.0
    for r in range(repeats):
        # Reset L_loc from A_loc for each timed run.
        np.copyto(L_loc, A_loc)
        comm.Barrier()
        t0 = MPI.Wtime()
        factorize(L_loc, N, P, rank, comm)
        comm.Barrier()
        elapsed = MPI.Wtime() - t0
        total += elapsed
        best = min(best, elapsed)
        if rank == 0:
            print(f"  run {r+1}: {elapsed:.6f} s", flush=True)

    # Gather L on rank 0 to verify.
    if rank == 0:
        L_full = np.zeros((N, N), dtype=np.float64, order="F")

    for j in range(N):
        owner = j % P
        if rank == owner:
            col = np.ascontiguousarray(L_loc[:, j // P])
            if rank == 0:
                # zero strict upper, copy lower
                L_full[:j, j] = 0.0
                L_full[j:, j] = col[j:]
            else:
                comm.Send([col, MPI.DOUBLE], dest=0, tag=j)
        elif rank == 0:
            buf = np.empty(N, dtype=np.float64)
            comm.Recv([buf, MPI.DOUBLE], source=owner, tag=j)
            L_full[:j, j] = 0.0
            L_full[j:, j] = buf[j:]

    if rank == 0:
        residual = np.linalg.norm(A_full - L_full @ L_full.T)
        gflops = (1.0 / 3.0) * (N ** 3) / best / 1e9
        print(f"  Best time:  {best:.6f} s")
        print(f"  Avg time:   {total / repeats:.6f} s")
        print(f"  GFLOP/s:    {gflops:.3f}")
        print(f"  Residual:   {residual:.2e}", flush=True)


if __name__ == "__main__":
    main()
