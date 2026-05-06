#!/bin/bash
#SBATCH --job-name=cholesky_serial
#SBATCH --output=cholesky_serial_%j.out
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00

# Compile the serial Cholesky factorization
g++ -O3 -march=native -o cholesky_serial cholesky_serial.cpp -lm

# Pin to a single thread to ensure a fair serial baseline
export OMP_NUM_THREADS=1

# Sweep across problem sizes to characterize serial scaling.
# Each invocation runs 3 timed iterations after a warm-up.
for N in 256 512 1024 2048; do
    echo "=============================="
    ./cholesky_serial $N 3
done
echo "=============================="
