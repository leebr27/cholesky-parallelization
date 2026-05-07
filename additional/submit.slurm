#!/bin/bash
#SBATCH --job-name=cholesky_mpi4py
#SBATCH --output=cholesky_mpi4py_%j.out
#SBATCH --partition=general
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:10:00

# Source the Spack environment.
source /shared/spack/share/spack/setup-env.sh

# Sanity check: show what we loaded.
which python mpicxx
python -c "import mpi4py, numpy; print('mpi4py', mpi4py.__version__, 'numpy', numpy.__version__)"

# Same PMIx workaround as the C++ MPI job.
export PMIX_MCA_gds=hash
export HOME="${HOME:-/home/$USER}"

# ---------------------------------------------------------------------------
# Strong scaling at N=1024 — direct comparison against the C++ MPI results.
# ---------------------------------------------------------------------------
echo "### Strong Scaling (N=1024) ###"
for P in 2 4 8 16; do
    echo "=============================="
    echo "Ranks: $P"
    srun --export=ALL --mpi=pmix_v5 -n $P python cholesky_mpi4py.py 1024 3
done
echo "=============================="

# ---------------------------------------------------------------------------
# Larger problem at full 16 ranks for direct comparison with cholesky_mpi.
# ---------------------------------------------------------------------------
echo "### Large Problem (N=2048, 16 ranks across 2 nodes) ###"
echo "=============================="
srun --export=ALL --mpi=pmix_v5 -n 16 python cholesky_mpi4py.py 2048 3
echo "=============================="
