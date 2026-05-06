// OpenMP-parallelized Cholesky factorization (DPOTRF): A = L * L^T
//
// A is an N×N symmetric positive-definite matrix stored in column-major order.
// The sub-diagonal row updates within each column are independent and are
// parallelized with #pragma omp parallel for.
//
// Build:  g++ -O3 -march=native -fopenmp -o cholesky_openmp cholesky_openmp.cpp -lm
// Run:    OMP_NUM_THREADS=8 ./cholesky_openmp [N] [repeats]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <omp.h>

// Column-major access macros
#define A(i, j) A[(size_t)(j) * N + (i)]
#define L(i, j) L[(size_t)(j) * N + (i)]

// ---------------------------------------------------------------------------
// OpenMP Cholesky — parallelize the sub-diagonal row updates in each column.
// Rows i > j are fully independent once L(j,j) is known, making this a
// natural fork/join over the trailing column panel.
// ---------------------------------------------------------------------------
void cholesky_openmp(const double* A, double* L, int N) {
    std::memset(L, 0, (size_t)N * N * sizeof(double));
    for (int j = 0; j < N; j++) {
        // Diagonal element (sequential — depends on previous columns)
        double sum = A(j, j);
        for (int k = 0; k < j; k++)
            sum -= L(j, k) * L(j, k);
        L(j, j) = std::sqrt(sum);

        double inv_ljj = 1.0 / L(j, j);

        // Sub-diagonal elements — rows are independent, parallelize over i
        #pragma omp parallel for schedule(static)
        for (int i = j + 1; i < N; i++) {
            double s = A(i, j);
            for (int k = 0; k < j; k++)
                s -= L(i, k) * L(j, k);
            L(i, j) = s * inv_ljj;
        }
    }
}

// ---------------------------------------------------------------------------
// Generate a random symmetric positive-definite matrix:
//   A = M^T * M + N*I
// ---------------------------------------------------------------------------
void generate_spd_matrix(double* A, int N) {
    double* M = new double[(size_t)N * N];
    for (size_t i = 0; i < (size_t)N * N; i++)
        M[i] = (double)rand() / RAND_MAX;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double s = 0.0;
            for (int k = 0; k < N; k++)
                s += M[(size_t)k * N + i] * M[(size_t)k * N + j];
            A(i, j) = s;
        }

    for (int i = 0; i < N; i++)
        A(i, i) += N;

    delete[] M;
}

// ---------------------------------------------------------------------------
// Frobenius norm of (A - L*L^T) over the lower triangle.
// ---------------------------------------------------------------------------
double verify(const double* A, const double* L, int N) {
    double err = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j <= i; j++) {
            double s = 0.0;
            for (int k = 0; k <= j; k++)
                s += L(i, k) * L(j, k);
            double diff = A(i, j) - s;
            err += diff * diff;
        }
    return std::sqrt(err);
}

int main(int argc, char** argv) {
    int N = 1024;
    int repeats = 3;
    if (argc > 1) N = std::atoi(argv[1]);
    if (argc > 2) repeats = std::atoi(argv[2]);

    int num_threads = 0;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    std::printf("OpenMP Cholesky factorization: N = %d, threads = %d, repeats = %d\n",
                N, num_threads, repeats);

    double* A = new double[(size_t)N * N];
    double* L = new double[(size_t)N * N];

    srand(42);
    generate_spd_matrix(A, N);

    // Warm-up
    cholesky_openmp(A, L, N);

    // Timed runs — keep best wall-clock time
    double best_time = 1e30;
    double total_time = 0.0;
    for (int r = 0; r < repeats; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cholesky_openmp(A, L, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        total_time += elapsed;
        if (elapsed < best_time) best_time = elapsed;
        std::printf("  run %d: %.6f s\n", r + 1, elapsed);
    }

    double residual = verify(A, L, N);
    double gflops = (1.0 / 3.0) * (double)N * N * N / best_time / 1e9;

    std::printf("  Best time:  %.6f s\n", best_time);
    std::printf("  Avg time:   %.6f s\n", total_time / repeats);
    std::printf("  GFLOP/s:    %.3f\n", gflops);
    std::printf("  Residual:   %.2e\n", residual);

    delete[] A;
    delete[] L;
    return 0;
}
