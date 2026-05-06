#!/bin/bash
#SBATCH --job-name=cholesky_torch
#SBATCH --output=cholesky_torch_%j.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

# ---------------------------------------------------------------------------
# Bring up a Python environment with PyTorch.
# Strategy: try a class-supplied module first; if not available, install
# torch into a venv at runtime (works on cluster node with internet access).
# ---------------------------------------------------------------------------
module load python 2>/dev/null || module load anaconda 2>/dev/null || true

if ! python -c "import torch" 2>/dev/null; then
    echo "PyTorch not present in environment — creating local venv."
    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --quiet --upgrade pip
    # CUDA 12 build of PyTorch (matches CUDA 12.8 runtime on the cluster GPUs)
    python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu121
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi | head -n 20 || true

# ---------------------------------------------------------------------------
# CPU runs — single thread to compare against the serial baseline, then
# 16 threads to compare against OpenMP.
# ---------------------------------------------------------------------------
echo "### CPU (1 thread) ###"
for N in 512 1024 2048; do
    echo "=============================="
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python cholesky_torch.py $N 3 cpu
done
echo "=============================="

echo "### CPU (4 threads) ###"
for N in 1024 2048; do
    echo "=============================="
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python cholesky_torch.py $N 3 cpu
done
echo "=============================="

# ---------------------------------------------------------------------------
# GPU runs — direct comparison with the hand-rolled CUDA kernel in part 4.
# ---------------------------------------------------------------------------
echo "### GPU (cuSOLVER via PyTorch) ###"
for N in 1024 2048 4096; do
    echo "=============================="
    python cholesky_torch.py $N 3 cuda
done
echo "=============================="
