#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <omp.h>

// Column-major access macro
#define A(i, j) A[(j)*N + (i)]
#define L(i, j) L[(j)*N + (i)]

// Serial Cholesky factorization: A = L * L^T
// A is N×N symmetric positive-definite (column-major), result written to L.
void cholesky_serial(const double* A, double* L, int N) {
    std::memset(L, 0, (size_t)N * N * sizeof(double));
    for (int j = 0; j < N; j++) {
        // Diagonal element
        double sum = A(j, j);
        for (int k = 0; k < j; k++)
            sum -= L(j, k) * L(j, k);
        L(j, j) = std::sqrt(sum);

        double inv_ljj = 1.0 / L(j, j);

        // Sub-diagonal elements in column j
        for (int i = j + 1; i < N; i++) {
            double s = A(i, j);
            for (int k = 0; k < j; k++)
                s -= L(i, k) * L(j, k);
            L(i, j) = s * inv_ljj;
        }
    }
}

// OpenMP-parallelized Cholesky factorization.
// The inner loops over rows and the dot-product reductions are parallelized.
void cholesky_openmp(const double* A, double* L, int N) {
    std::memset(L, 0, (size_t)N * N * sizeof(double));
    for (int j = 0; j < N; j++) {
        // Diagonal element
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

// Generate a random symmetric positive-definite matrix: A = M * M^T + N*I
void generate_spd_matrix(double* A, int N) {
    double* M = new double[(size_t)N * N];
    for (int i = 0; i < N * N; i++)
        M[i] = (double)rand() / RAND_MAX;

    // A = M^T * M
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double s = 0.0;
            for (int k = 0; k < N; k++)
                s += M[k * N + i] * M[k * N + j];
            A(i, j) = s;
        }

    // Add N*I to ensure well-conditioned
    for (int i = 0; i < N; i++)
        A(i, i) += N;

    delete[] M;
}

// Verify L*L^T ≈ A by computing the Frobenius norm of the residual
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
    if (argc > 1) N = std::atoi(argv[1]);

    int num_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    std::printf("Cholesky factorization: N = %d, threads = %d\n", N, num_threads);

    double* A = new double[(size_t)N * N];
    double* L = new double[(size_t)N * N];

    srand(42);
    generate_spd_matrix(A, N);

    // Warm-up run
    cholesky_openmp(A, L, N);

    // Timed run
    int repeats = 3;
    double best_time = 1e30;
    for (int r = 0; r < repeats; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        cholesky_openmp(A, L, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        if (elapsed < best_time) best_time = elapsed;
    }

    double residual = verify(A, L, N);
    std::printf("  Best time:  %.6f s\n", best_time);
    std::printf("  Residual:   %.2e\n", residual);

    delete[] A;
    delete[] L;
    return 0;
}
