#!/bin/bash
#SBATCH --job-name=cholesky_mpi
#SBATCH --output=cholesky_mpi_%j.out
#SBATCH --partition=general
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00

# Load MPI (adjust module name if cluster uses a different one)
module load openmpi 2>/dev/null || module load mpich 2>/dev/null || true

# Compile with MPI C++ wrapper
mpicxx -O3 -march=native -o cholesky_mpi cholesky_mpi.cpp -lm

# ---------------------------------------------------------------------------
# Strong scaling at N = 1024 across 2 nodes.
# Total ranks vary; use mpirun's -n flag.  Always at least 2 nodes are
# allocated (per project requirements); when using fewer ranks, ranks may
# not span both nodes — that's fine, the allocation still satisfies the
# "at least two nodes" rule.
# ---------------------------------------------------------------------------
echo "### Strong Scaling (N=1024) ###"
for P in 2 4 8 16; do
    echo "=============================="
    echo "Ranks: $P"
    mpirun -n $P ./cholesky_mpi 1024 3
done
echo "=============================="

# ---------------------------------------------------------------------------
# Larger problem to amortize communication overhead — N = 2048 with full
# 16-rank allocation across both nodes.
# ---------------------------------------------------------------------------
echo "### Large Problem (N=2048, 16 ranks across 2 nodes) ###"
echo "=============================="
mpirun -n 16 ./cholesky_mpi 2048 3
echo "=============================="
