#!/bin/bash
#SBATCH --job-name=cholesky_openmp
#SBATCH --output=cholesky_openmp_%j.out
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=00:10:00

# Compile with OpenMP support
g++ -O3 -march=native -fopenmp -o cholesky_openmp cholesky_openmp.cpp -lm

# ---------------------------------------------------------------------------
# Strong scaling study: fixed problem size N=1024, vary thread count.
# Matches the serial baseline (N=1024, best time ~0.777 s) for direct speedup
# comparison.
# ---------------------------------------------------------------------------
echo "### Strong Scaling (N=1024) ###"
for t in 1 2 4 8 16; do
    echo "=============================="
    export OMP_NUM_THREADS=$t
    ./cholesky_openmp 1024 3
done
echo "=============================="

# ---------------------------------------------------------------------------
# Weak scaling study: keep work per thread roughly constant by scaling
# N such that N^3 / threads is fixed.  Base: N=512 at 1 thread.
#   1 thread  -> N=512   (N^3 = 1.34e8)
#   2 threads -> N=645   (N^3 ≈ 2x)
#   4 threads -> N=812   (N^3 ≈ 4x)
#   8 threads -> N=1024  (N^3 ≈ 8x)
#  16 threads -> N=1290  (N^3 ≈ 16x)
# ---------------------------------------------------------------------------
echo "### Weak Scaling ###"
configs="1 512 2 645 4 812 8 1024 16 1290"
set -- $configs
while [ $# -ge 2 ]; do
    t=$1; N=$2; shift 2
    echo "=============================="
    export OMP_NUM_THREADS=$t
    ./cholesky_openmp $N 3
done
echo "=============================="
