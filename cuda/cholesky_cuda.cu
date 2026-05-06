// CUDA Cholesky factorization (DPOTRF): A = L * L^T
//
// Strategy: right-looking factorization with the trailing matrix update on
// the GPU.  The matrix A (column-major) is copied to device memory once and
// factored in place.  For each column j:
//   1. Read A(j,j) back to host, take sqrt -> L(j,j).
//   2. scale_column kernel: divide sub-diagonal entries A(j+1..N-1, j) by L(j,j).
//   3. rank1_update kernel: trailing-matrix rank-1 update over k > j, i >= k:
//        A(i,k) -= L(i,j) * L(k,j)
//
// The rank-1 update is the dominant cost (O((N-j)^2) work per column,
// O(N^3/3) total).  It launches a 2D grid over the lower triangle of the
// trailing submatrix and provides massive parallelism on the GPU.
//
// Build:  nvcc -O3 -arch=sm_70 -o cholesky_cuda cholesky_cuda.cu -lm
// Run:    ./cholesky_cuda [N] [repeats]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                            \
    cudaError_t _err = (call);                                           \
    if (_err != cudaSuccess) {                                           \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(_err));                          \
        std::exit(1);                                                    \
    }                                                                    \
} while (0)

// Column-major indexing on the device (leading dimension N)
__device__ __forceinline__ size_t idx(int i, int j, int N) {
    return (size_t)j * N + i;
}

// Scale sub-diagonal entries of column j by 1/diag.
// Launch: 1D grid covering rows j+1 .. N-1.
__global__ void scale_column_kernel(double* L, int N, int j, double inv_diag) {
    int i = j + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        L[idx(i, j, N)] *= inv_diag;
    }
}

// Trailing rank-1 update: for k > j, i >= k:
//   L(i, k) -= L(i, j) * L(k, j)
// Launch: 2D grid mapping (k - (j+1), i - (j+1)) -> threads in the trailing
// (N-j-1) x (N-j-1) block; each thread tests i >= k.
__global__ void rank1_update_kernel(double* L, int N, int j) {
    int k = j + 1 + blockIdx.x * blockDim.x + threadIdx.x;  // column
    int i = j + 1 + blockIdx.y * blockDim.y + threadIdx.y;  // row
    if (k < N && i < N && i >= k) {
        double Lij = L[idx(i, j, N)];
        double Lkj = L[idx(k, j, N)];
        L[idx(i, k, N)] -= Lij * Lkj;
    }
}

// ---------------------------------------------------------------------------
// Host-side driver: factor A in place on the GPU.
// Returns elapsed seconds.
// ---------------------------------------------------------------------------
double cholesky_cuda(double* d_A, int N) {
    const int BLOCK_1D = 256;
    const dim3 BLOCK_2D(16, 16);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int j = 0; j < N; j++) {
        // 1. Read A(j,j), take sqrt, write back.
        double diag;
        CUDA_CHECK(cudaMemcpy(&diag, d_A + (size_t)j * N + j, sizeof(double),
                              cudaMemcpyDeviceToHost));
        double Ljj = std::sqrt(diag);
        CUDA_CHECK(cudaMemcpy(d_A + (size_t)j * N + j, &Ljj, sizeof(double),
                              cudaMemcpyHostToDevice));

        if (j == N - 1) break;

        // 2. Scale sub-diagonal entries of column j by 1/L(j,j).
        double inv = 1.0 / Ljj;
        int rows = N - j - 1;
        int grid1d = (rows + BLOCK_1D - 1) / BLOCK_1D;
        scale_column_kernel<<<grid1d, BLOCK_1D>>>(d_A, N, j, inv);

        // 3. Rank-1 update of trailing submatrix.
        dim3 grid2d((rows + BLOCK_2D.x - 1) / BLOCK_2D.x,
                    (rows + BLOCK_2D.y - 1) / BLOCK_2D.y);
        rank1_update_kernel<<<grid2d, BLOCK_2D>>>(d_A, N, j);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// Generate SPD matrix (host): A = M^T*M + N*I
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Frobenius norm of (A - L*L^T) over the lower triangle.
// ---------------------------------------------------------------------------
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
    int N = 1024;
    int repeats = 3;
    if (argc > 1) N = std::atoi(argv[1]);
    if (argc > 2) repeats = std::atoi(argv[2]);

    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("CUDA Cholesky factorization: N = %d, repeats = %d\n", N, repeats);
    std::printf("  GPU: %s (CC %d.%d, %d SMs)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    std::vector<double> A((size_t)N * N), L((size_t)N * N);
    generate_spd_matrix(A, N);

    double* d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)N * N * sizeof(double)));

    // Warm-up (kernel JIT, allocator settling, etc.)
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), (size_t)N * N * sizeof(double),
                          cudaMemcpyHostToDevice));
    cholesky_cuda(d_A, N);

    // Timed runs
    double best_time = 1e30;
    double total_time = 0.0;
    for (int r = 0; r < repeats; r++) {
        // Reset device matrix to A before each timed run.
        CUDA_CHECK(cudaMemcpy(d_A, A.data(), (size_t)N * N * sizeof(double),
                              cudaMemcpyHostToDevice));
        double elapsed = cholesky_cuda(d_A, N);
        total_time += elapsed;
        if (elapsed < best_time) best_time = elapsed;
        std::printf("  run %d: %.6f s\n", r + 1, elapsed);
    }

    // Copy factored L back to host (zero strict upper triangle for verify).
    CUDA_CHECK(cudaMemcpy(L.data(), d_A, (size_t)N * N * sizeof(double),
                          cudaMemcpyDeviceToHost));
    for (int j = 0; j < N; j++)
        for (int i = 0; i < j; i++)
            L[(size_t)j * N + i] = 0.0;

    double residual = verify(A, L, N);
    double gflops = (1.0 / 3.0) * (double)N * N * N / best_time / 1e9;

    std::printf("  Best time:  %.6f s\n", best_time);
    std::printf("  Avg time:   %.6f s\n", total_time / repeats);
    std::printf("  GFLOP/s:    %.3f\n", gflops);
    std::printf("  Residual:   %.2e\n", residual);

    CUDA_CHECK(cudaFree(d_A));
    return 0;
}
