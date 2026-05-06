#!/bin/bash
#SBATCH --job-name=cholesky_vtune
#SBATCH --output=cholesky_vtune_%j.out
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:10:00

# ---------------------------------------------------------------------------
# CPU profiling of the OpenMP Cholesky kernel using Intel VTune.
# Two collection types:
#   1. hotspots  — hardware-event-based sampling, identifies the hot loops
#                  and how time is split across threads.
#   2. threading — overhead/spin/wait analysis to quantify the OpenMP
#                  fork/join cost we suspect limits 16-thread scaling.
# Both runs use the existing OpenMP binary at N=2048, 16 threads.
# ---------------------------------------------------------------------------

source /shared/spack/share/spack/setup-env.sh
spack load intel-oneapi-vtune
which vtune || { echo "vtune not on PATH after spack load"; exit 1; }

# Build with debug symbols so VTune can attribute samples to source lines.
g++ -O3 -g -march=native -fopenmp \
    -o cholesky_openmp_prof ../openmp/cholesky_openmp.cpp -lm

export OMP_NUM_THREADS=16

# Hotspots analysis (function-level CPU time + thread breakdown).
echo "### VTune hotspots (N=2048, 16 threads) ###"
rm -rf vtune_hotspots
vtune -collect hotspots -result-dir vtune_hotspots \
      -- ./cholesky_openmp_prof 2048 1
echo "----- top hotspots -----"
vtune -report hotspots -result-dir vtune_hotspots \
      -group-by function -limit 10
echo

# Threading analysis (spin/wait time, fork-join overhead).
echo "### VTune threading (N=2048, 16 threads) ###"
rm -rf vtune_threading
vtune -collect threading -result-dir vtune_threading \
      -- ./cholesky_openmp_prof 2048 1
echo "----- threading summary -----"
vtune -report summary -result-dir vtune_threading
