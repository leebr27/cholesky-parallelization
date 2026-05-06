#!/bin/bash
#SBATCH --job-name=cholesky_mpi
#SBATCH --output=cholesky_mpi_%j.out
#SBATCH --partition=general
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:10:00

# Load OpenMPI 5 — built with PMIx v5 support to match the cluster's srun launcher
module load openmpi5/5.0.8amzn1

# Ensure HOME is set and propagated to compute nodes (OpenMPI needs it during init)
export HOME="${HOME:-/home/$USER}"

# Compile with MPI C++ wrapper
mpicxx -O3 -march=native -o cholesky_mpi cholesky_mpi.cpp -lm

# Force PMIx to use the hash GDS component (avoids missing shmem2 component)
export PMIX_MCA_gds=hash

# ---------------------------------------------------------------------------
# Strong scaling at N = 1024 across 2 nodes.
# Use srun (not mpirun) so that Slurm handles inter-node process launch via
# PMI/PMIx — mpirun's ORTE daemon cannot communicate across nodes on this
# cluster configuration.
# ---------------------------------------------------------------------------
echo "### Strong Scaling (N=1024) ###"
for P in 2 4 8 16; do
    echo "=============================="
    echo "Ranks: $P"
    srun --export=ALL --mpi=pmix_v5 -n $P ./cholesky_mpi 1024 3
done
echo "=============================="

# ---------------------------------------------------------------------------
# Larger problem to amortize communication overhead — N = 2048 with full
# 16-rank allocation across both nodes.
# ---------------------------------------------------------------------------
echo "### Large Problem (N=2048, 16 ranks across 2 nodes) ###"
echo "=============================="
srun --export=ALL --mpi=pmix_v5 -n 16 ./cholesky_mpi 2048 3
echo "=============================="
