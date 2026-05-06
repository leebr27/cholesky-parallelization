// MPI distributed-memory Cholesky factorization (DPOTRF): A = L * L^T
//
// Strategy: 1D column-cyclic distribution.  Rank r owns global column j
// iff (j % P) == r, where P is the MPI world size.  Each rank stores its
// owned columns in column-major order.
//
// Algorithm (right-looking):
//   for j = 0..N-1:
//     owner = j % P
//     if rank == owner:
//       L(j,j)   = sqrt(L(j,j))                 // diagonal already updated
//       L(i,j)  /= L(j,j) for i > j             // panel scale
//     MPI_Bcast L(j..N-1, j) from owner         // share factored column
//     each rank: for every owned column k > j:
//       L(i,k) -= L(i,j) * L(k,j) for i >= k    // rank-1 trailing update
//
// Communication: one MPI_Bcast per column of length (N-j) doubles, total
//   sum_{j=0}^{N-1} (N-j) = N(N+1)/2 doubles per rank — O(N^2 log P) bits.
// Computation per rank: O(N^3 / P).
//
// Build:  mpicxx -O3 -march=native -o cholesky_mpi cholesky_mpi.cpp -lm
// Run:    mpirun -n P ./cholesky_mpi [N] [repeats]

#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

// Generate a random symmetric positive-definite matrix on rank 0.
//   A = M^T * M + N*I
static void generate_spd_matrix(std::vector<double>& A, int N) {
    std::vector<double> M((size_t)N * N);
    std::srand(42);
    for (size_t i = 0; i < (size_t)N * N; i++)
        M[i] = (double)std::rand() / RAND_MAX;

    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++) {
            double s = 0.0;
            for (int k = 0; k < N; k++)
                s += M[(size_t)k * N + i] * M[(size_t)k * N + j];
            A[(size_t)j * N + i] = s;
        }

    for (int i = 0; i < N; i++)
        A[(size_t)i * N + i] += N;
}

// Frobenius norm of (A - L*L^T) over the lower triangle (rank 0 only).
static double verify(const std::vector<double>& A,
                     const std::vector<double>& L, int N) {
    double err = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j <= i; j++) {
            double s = 0.0;
            for (int k = 0; k <= j; k++)
                s += L[(size_t)k * N + i] * L[(size_t)k * N + j];
            double diff = A[(size_t)j * N + i] - s;
            err += diff * diff;
        }
    return std::sqrt(err);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    int N = 1024;
    int repeats = 3;
    if (argc > 1) N = std::atoi(argv[1]);
    if (argc > 2) repeats = std::atoi(argv[2]);

    if (rank == 0) {
        std::printf("MPI Cholesky factorization: N = %d, ranks = %d, repeats = %d\n",
                    N, P, repeats);
    }

    // ----- Allocate per-rank column storage (column-cyclic distribution) -----
    // Local column index lj corresponds to global column lj*P + rank.
    int my_ncols = (N - rank + P - 1) / P;
    if (my_ncols < 0) my_ncols = 0;

    // A_loc: this rank's columns of A (preserved across timed runs).
    // L_loc: working storage, factored in place.
    std::vector<double> A_loc((size_t)N * my_ncols, 0.0);
    std::vector<double> L_loc((size_t)N * my_ncols, 0.0);

    // ----- Generate A on rank 0 and scatter columns to owners -----
    std::vector<double> A_full;
    if (rank == 0) {
        A_full.resize((size_t)N * N);
        generate_spd_matrix(A_full, N);
    }

    {
        std::vector<double> col(N);
        for (int j = 0; j < N; j++) {
            int owner = j % P;
            if (rank == 0) {
                std::memcpy(col.data(), &A_full[(size_t)j * N],
                            (size_t)N * sizeof(double));
                if (owner == 0) {
                    int lj = j / P;
                    std::memcpy(&A_loc[(size_t)lj * N], col.data(),
                                (size_t)N * sizeof(double));
                } else {
                    MPI_Send(col.data(), N, MPI_DOUBLE, owner, j, MPI_COMM_WORLD);
                }
            } else if (rank == owner) {
                int lj = j / P;
                MPI_Recv(&A_loc[(size_t)lj * N], N, MPI_DOUBLE, 0, j,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    // ----- Timed factorization -----
    auto run_factorize = [&]() {
        // Reset L_loc from A_loc (we factor the trailing matrix in place).
        std::memcpy(L_loc.data(), A_loc.data(),
                    (size_t)N * my_ncols * sizeof(double));

        std::vector<double> col_buf(N);

        for (int j = 0; j < N; j++) {
            int owner = j % P;
            int lj = j / P;
            int len = N - j;            // length of L(j..N-1, j)

            // Owner finalizes column j: sqrt the diagonal and scale below.
            if (rank == owner) {
                double* colp = &L_loc[(size_t)lj * N];
                double djj = std::sqrt(colp[j]);
                colp[j] = djj;
                double inv = 1.0 / djj;
                for (int i = j + 1; i < N; i++)
                    colp[i] *= inv;

                // Pack the factored column into the broadcast buffer.
                std::memcpy(col_buf.data(), &colp[j],
                            (size_t)len * sizeof(double));
            }

            // Share L(j..N-1, j) with all other ranks.
            MPI_Bcast(col_buf.data(), len, MPI_DOUBLE, owner, MPI_COMM_WORLD);

            // Apply rank-1 update to each owned column k > j.
            // Local column lk corresponds to global k = lk*P + rank.
            // Smallest k > j satisfying k%P == rank:
            int k_start;
            if (rank > j % P) k_start = (j / P) * P + rank;
            else              k_start = (j / P + 1) * P + rank;

            int lk_start = k_start / P;
            for (int lk = lk_start; lk < my_ncols; lk++) {
                int k = lk * P + rank;          // global column index, > j
                double Lkj = col_buf[k - j];    // L(k, j)
                double* colp = &L_loc[(size_t)lk * N];
                // colp[i] -= L(i,j) * L(k,j) for i = k..N-1
                for (int i = k; i < N; i++)
                    colp[i] -= col_buf[i - j] * Lkj;
            }
        }
    };

    // Warm-up
    run_factorize();
    MPI_Barrier(MPI_COMM_WORLD);

    double best_time = 1e30;
    double total_time = 0.0;
    for (int r = 0; r < repeats; r++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        run_factorize();
        MPI_Barrier(MPI_COMM_WORLD);
        double elapsed = MPI_Wtime() - t0;
        total_time += elapsed;
        if (elapsed < best_time) best_time = elapsed;
        if (rank == 0)
            std::printf("  run %d: %.6f s\n", r + 1, elapsed);
    }

    // ----- Gather L on rank 0 for residual check -----
    std::vector<double> L_full;
    if (rank == 0) L_full.assign((size_t)N * N, 0.0);

    {
        std::vector<double> col(N);
        for (int j = 0; j < N; j++) {
            int owner = j % P;
            if (rank == owner) {
                int lj = j / P;
                std::memcpy(col.data(), &L_loc[(size_t)lj * N],
                            (size_t)N * sizeof(double));
                if (rank != 0)
                    MPI_Send(col.data(), N, MPI_DOUBLE, 0, j, MPI_COMM_WORLD);
            }
            if (rank == 0) {
                if (owner != 0)
                    MPI_Recv(col.data(), N, MPI_DOUBLE, owner, j,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // Zero the strict upper triangle and copy lower.
                for (int i = 0; i < j; i++)
                    L_full[(size_t)j * N + i] = 0.0;
                for (int i = j; i < N; i++)
                    L_full[(size_t)j * N + i] = col[i];
            }
        }
    }

    if (rank == 0) {
        double residual = verify(A_full, L_full, N);
        double gflops = (1.0 / 3.0) * (double)N * N * N / best_time / 1e9;
        std::printf("  Best time:  %.6f s\n", best_time);
        std::printf("  Avg time:   %.6f s\n", total_time / repeats);
        std::printf("  GFLOP/s:    %.3f\n", gflops);
        std::printf("  Residual:   %.2e\n", residual);
    }

    MPI_Finalize();
    return 0;
}
