#!/bin/bash -l
#SBATCH --job-name=cholesky_openmp_vtune
#SBATCH --output=cholesky_openmp_vtune_%j.out
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:10:00

# ---------------------------------------------------------------------------
# Intel VTune profiling for the OpenMP Cholesky implementation.
# Identifies hotspots and OpenMP-region bottlenecks (e.g. the sequential
# diagonal update vs. the parallel sub-diagonal panel) on a 16-core node.
# ---------------------------------------------------------------------------

# Set up Spack environment (provides vtune, compilers, etc.).
# The setup script may live at ~/spack/... or ~/UID/spack/... depending on
# how the cluster was configured.  Search for it automatically.
SPACK_SETUP=""
for candidate in \
    "$HOME/spack/share/spack/setup-env.sh" \
    "$HOME"/*/spack/share/spack/setup-env.sh; do
    if [ -f "$candidate" ]; then
        SPACK_SETUP="$candidate"
        break
    fi
done

if [ -z "$SPACK_SETUP" ]; then
    echo "ERROR: Could not find spack/share/spack/setup-env.sh under $HOME" >&2
    echo "Hint: run 'find ~ -name setup-env.sh 2>/dev/null' on the cluster to locate it." >&2
    exit 1
fi

echo "Using Spack setup: $SPACK_SETUP"
. "$SPACK_SETUP"

# `spack load` / `spack env activate` are broken on this account due to a
# stale entry in ~/.spack/package_repos (UnknownNamespaceError: builtin).
# Bypass spack entirely and locate the vtune binary directly inside the
# spack install tree.
SPACK_ROOT="$(dirname "$(dirname "$(dirname "$SPACK_SETUP")")")"
# Prefer the newest installed vtune (sort -V puts e.g. 2025.8 after 2024.3).
VTUNE_BIN="$(ls -d "$SPACK_ROOT"/opt/spack/*/intel-oneapi-vtune-*/vtune/latest/bin64 2>/dev/null \
            | sort -V | tail -n 1)"

if [ -z "$VTUNE_BIN" ] || [ ! -x "$VTUNE_BIN/vtune" ]; then
    # Fallback: search anywhere under the spack root.
    VTUNE_PATH="$(find "$SPACK_ROOT" -type f -name vtune -path '*/bin64/*' 2>/dev/null | head -n 1)"
    if [ -n "$VTUNE_PATH" ]; then
        VTUNE_BIN="$(dirname "$VTUNE_PATH")"
    fi
fi

if [ -z "$VTUNE_BIN" ] || [ ! -x "$VTUNE_BIN/vtune" ]; then
    echo "ERROR: could not locate vtune binary under $SPACK_ROOT" >&2
    exit 1
fi

export PATH="$VTUNE_BIN:$PATH"
echo "Using vtune from: $VTUNE_BIN"
# Source vtune's own env script if present (sets up sampling drivers, etc.)
[ -f "$VTUNE_BIN/../env/vars.sh" ] && . "$VTUNE_BIN/../env/vars.sh" 2>/dev/null || true

if ! command -v vtune >/dev/null 2>&1; then
    echo "ERROR: 'vtune' still not in PATH after manual load (looked in $VTUNE_BIN)" >&2
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
      -knob enable-characterization-insights=false \
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
      -knob enable-characterization-insights=false \
      -- ./cholesky_openmp $N $REPEATS

vtune -report summary -result-dir vtune_threading
