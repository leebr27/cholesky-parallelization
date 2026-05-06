# Question 6 Answers

### Checklist

For full credit, you must:
- [x] Provide C++ code implementing some task or algorithm (ideally something related to your project);
- [x] Enable OpenMP parallelization;
- [x] Provide your code in this directory, along with a Slurm script that compiles and runs it on a CPU node with 1, 2, 4, 8, and 16 threads;
- [x] Explain your work below in one paragraph or less.

### Explanation

I implemented the Cholesky factorization (DPOTRF) of a symmetric positive-definite matrix, which is a core LAPACK routine I plan to parallelize for my project. The serial algorithm walks down the diagonal, computing each column of the lower-triangular factor L via a square root and a series of dot-product updates. For the OpenMP version, I parallelized the sub-diagonal row updates within each column using `#pragma omp parallel for`, since rows within a column are independent once the diagonal element is computed. The Slurm script compiles the code with `-fopenmp` and runs it on a 1024×1024 matrix with 1, 2, 4, 8, and 16 threads, reporting the best wall-clock time and a residual norm to verify correctness.

### Results

| Threads | Best Time (s) | Speedup | Residual |
|---------|---------------|---------|----------|
| 1       | 0.877157      | 1.00x   | 1.48e-10 |
| 2       | 0.348698      | 2.52x   | 1.48e-10 |
| 4       | 0.180886      | 4.85x   | 1.48e-10 |
| 8       | 0.101428      | 8.65x   | 1.48e-10 |
| 16      | 0.091995      | 9.53x   | 1.48e-10 |
