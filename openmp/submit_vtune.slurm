#!/bin/bash
#SBATCH --job-name=cholesky_openmp_vtune
#SBATCH --output=cholesky_openmp_vtune_%j.out
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=00:10:00

# ---------------------------------------------------------------------------
# Intel VTune profiling for the OpenMP Cholesky implementation.
# Identifies hotspots and OpenMP-region bottlenecks (e.g. the sequential
# diagonal update vs. the parallel sub-diagonal panel) on a 16-core node.
# ---------------------------------------------------------------------------

# Load Intel VTune. Try a few common module names so this works across
# cluster configurations; fall back to whatever `vtune` is already in PATH.
module load intel-oneapi-vtune 2>/dev/null \
    || module load vtune       2>/dev/null \
    || module load intel       2>/dev/null \
    || true

if ! command -v vtune >/dev/null 2>&1; then
    echo "ERROR: 'vtune' not found in PATH after module load." >&2
    echo "Available modules matching vtune/intel:" >&2
    module avail 2>&1 | grep -iE 'vtune|oneapi|intel' >&2 || true
    exit 1
fi

vtune --version

# Build with -g so VTune can attribute samples to source lines.
g++ -O3 -g -march=native -fopenmp -o cholesky_openmp cholesky_openmp.cpp -lm

# Use full physical cores; let OpenMP threads stay pinned for stable profiles.
export OMP_NUM_THREADS=16
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Problem size matches the strong-scaling sweep so results are comparable.
N=1024
REPEATS=3

# ---------------------------------------------------------------------------
# 1. Hotspots analysis — where is wall time spent?
# ---------------------------------------------------------------------------
echo "### VTune: hotspots ###"
rm -rf vtune_hotspots
vtune -collect hotspots \
      -result-dir vtune_hotspots \
      -- ./cholesky_openmp $N $REPEATS

vtune -report summary -result-dir vtune_hotspots
echo
echo "--- Top hotspot functions ---"
vtune -report hotspots -result-dir vtune_hotspots -limit 15

# ---------------------------------------------------------------------------
# 2. Threading / OpenMP analysis — measure parallel efficiency,
#    serial regions, and imbalance in the column loop.
# ---------------------------------------------------------------------------
echo
echo "### VTune: threading ###"
rm -rf vtune_threading
vtune -collect threading \
      -result-dir vtune_threading \
      -- ./cholesky_openmp $N $REPEATS

vtune -report summary -result-dir vtune_threading
