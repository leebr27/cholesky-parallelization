#!/bin/bash
#SBATCH --job-name=cholesky
#SBATCH --output=cholesky_%j.out
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=00:10:00

# Compile with OpenMP support
g++ -O2 -march=native -fopenmp -o cholesky cholesky.cpp -lm

# Run with 1, 2, 4, 8, and 16 threads
for t in 1 2 4 8 16; do
    echo "=============================="
    export OMP_NUM_THREADS=$t
    ./cholesky 1024
done
echo "=============================="
