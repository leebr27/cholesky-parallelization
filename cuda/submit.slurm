#!/bin/bash
#SBATCH --job-name=cholesky_cuda
#SBATCH --output=cholesky_cuda_%j.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

# Load CUDA toolchain
module load cuda 2>/dev/null || true

# Show GPU and nvcc info
nvidia-smi || true
nvcc --version

# Compile.  -arch=sm_70 covers V100; -arch=sm_80 covers A100.  We use sm_70
# as a safe baseline and let nvcc PTX-JIT to newer GPUs at runtime.
nvcc -O3 -arch=sm_70 -o cholesky_cuda cholesky_cuda.cu -lm

# ---------------------------------------------------------------------------
# Run a sweep of problem sizes to characterize GPU scaling.
# Larger N is needed to amortize kernel-launch overhead and saturate the GPU.
# Capped at N=3072: the naive rank-1-update formulation issues N kernel
# launches with a per-column device sync, so per-launch overhead grows
# linearly and limits achievable size within the 10-minute job window.
# ---------------------------------------------------------------------------
for N in 512 1024 2048 3072; do
    echo "=============================="
    ./cholesky_cuda $N 3
done
echo "=============================="
