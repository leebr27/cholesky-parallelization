#!/bin/bash
#SBATCH --job-name=cholesky_cuda_nsys
#SBATCH --output=cholesky_cuda_nsys_%j.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

# ---------------------------------------------------------------------------
# Nsight Systems profiling for the CUDA Cholesky implementation.
# Captures CUDA API calls, kernel launches, and host<->device memory copies
# to identify per-column kernel-launch overhead and the dominant rank-1
# update cost.
# ---------------------------------------------------------------------------

# Load CUDA toolchain (nsys ships with the CUDA toolkit).
module load cuda 2>/dev/null || true

# Make sure nsys is available; otherwise fall back to common CUDA install paths.
if ! command -v nsys >/dev/null 2>&1; then
    for d in \
        /usr/local/cuda/bin \
        /usr/local/cuda/nsight-systems*/bin \
        /opt/nvidia/nsight-systems*/bin \
        /usr/local/cuda/targets/x86_64-linux/bin; do
        if [ -x "$d/nsys" ]; then
            export PATH="$d:$PATH"
            break
        fi
    done
fi

if ! command -v nsys >/dev/null 2>&1; then
    echo "ERROR: 'nsys' (Nsight Systems) not found in PATH." >&2
    echo "Hint: try 'find /usr/local/cuda -name nsys 2>/dev/null'." >&2
    exit 1
fi

nvidia-smi || true
nvcc --version
nsys --version

# Compile with line-info so the report can attribute samples to source lines.
# A10G is Ampere sm_86; sm_70 is dropped for offline compilation in CUDA 12.8.
nvcc -O3 -lineinfo -arch=sm_86 -Wno-deprecated-gpu-targets \
     -o cholesky_cuda cholesky_cuda.cu -lm

# Problem size large enough to expose the dominant rank-1 update cost
# without exceeding the 10-minute job window.
N=2048
REPEATS=3

# ---------------------------------------------------------------------------
# Nsight Systems trace: CUDA runtime + driver API + OS runtime calls.
# `--stats=true` prints kernel/memcpy summary tables in the job output.
# ---------------------------------------------------------------------------
echo "### Nsight Systems: profile (N=$N, repeats=$REPEATS) ###"
rm -f nsys_cholesky_cuda.nsys-rep nsys_cholesky_cuda.sqlite

nsys profile \
     --output=nsys_cholesky_cuda \
     --force-overwrite=true \
     --trace=cuda,nvtx,osrt \
     --sample=cpu \
     --stats=true \
     ./cholesky_cuda $N $REPEATS

echo
echo "### Top CUDA kernels by total time ###"
nsys stats --report cuda_gpu_kern_sum --format table nsys_cholesky_cuda.nsys-rep || true

echo
echo "### CUDA memcpy summary ###"
nsys stats --report cuda_gpu_mem_time_sum --format table nsys_cholesky_cuda.nsys-rep || true

echo
echo "### CUDA API call summary (host overhead) ###"
nsys stats --report cuda_api_sum --format table nsys_cholesky_cuda.nsys-rep || true
