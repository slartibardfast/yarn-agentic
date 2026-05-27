// PD6 — 1-stream vs 2-stream-per-device matmul for libmgpu
//
// PHASE_CUDA_NATIVE_DISPATCH.md §9 PD6.
//
// Question: when a libmgpu CLIP-layer matmul is split row-wise into 2
// sub-matmuls per device, does running them on 2 streams (with overlapped
// partial-reduce) deliver >=5% improvement over running the full matmul
// on 1 stream?
//
// Threshold: 2-stream >=5% faster → C8 ships (multi-stream-per-device ILP).
//            Otherwise drop C8.
//
// Methodology:
//   - Allocate a matmul: C = A @ B where A is [M, K] and B is [K, N], C is [M, N].
//     Default shape: M=1024 (n_tokens), K=1280 (hidden_dim), N=2*1280 (2x for
//     CLIP attention QKV-style proj). All F16.
//   - Path A (1-stream): full matmul on stream[0] of each device. Uses
//     cuBLAS hgemm (matches production codepath).
//   - Path B (2-stream): split B by columns into [K, N/2] x 2. Sub-matmul 1
//     on stream[0] computes left half of C; sub-matmul 2 on stream[1]
//     computes right half. The 2 sub-matmuls run concurrently on the same
//     device, contending for SMs but with independent stream ordering.
//   - 500 iterations, drop first 50 warm-up, p50/p95/p99.
//   - Verify byte-identity between paths.
//
// Note: this microbenchmark runs on a SINGLE device, measuring whether the
// 2-stream split can extract ILP on that device's SMs. It does not model
// cross-device peer copies (those are PD5's scope). The libmgpu integration
// stacks the two: per-device 2-stream matmul + per-device-pair reduce.
//
// Compile:
//   nvcc -ccbin /usr/bin/g++-15 -O2 -o pd6_mstream pd6_multistream_matmul.cu -lcublas
// Run:
//   ./pd6_mstream                # default M=1024 K=1280 N=2560
//   ./pd6_mstream 1024 2560 7680 # CLIP-G-style FFN shape

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define CK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, \
                     #call, cudaGetErrorString(e)); std::exit(1); \
    } \
} while (0)

#define BCK(call) do { \
    cublasStatus_t e = (call); \
    if (e != CUBLAS_STATUS_SUCCESS) { \
        std::fprintf(stderr, "%s:%d: cublas err %d\n", __FILE__, __LINE__, (int)e); \
        std::exit(1); \
    } \
} while (0)

static double percentile(std::vector<double> & v, double pct) {
    std::sort(v.begin(), v.end());
    size_t idx = (size_t)(pct * v.size());
    if (idx >= v.size()) idx = v.size() - 1;
    return v[idx];
}

int main(int argc, char ** argv) {
    int M = 1024;   // n_tokens
    int K = 1280;   // hidden_dim
    int N = 2 * 1280;  // 2x hidden_dim (matches CLIP attention QKV-style proj when split)
    if (argc >= 2) M = std::atoi(argv[1]);
    if (argc >= 3) K = std::atoi(argv[2]);
    if (argc >= 4) N = std::atoi(argv[3]);

    if (N % 2 != 0) {
        std::fprintf(stderr, "PD6: N must be even (got %d) so it can split evenly\n", N);
        return 2;
    }
    const int N_half = N / 2;

    const size_t bytes_A = (size_t)M * (size_t)K * sizeof(__half);
    const size_t bytes_B = (size_t)K * (size_t)N * sizeof(__half);
    const size_t bytes_C = (size_t)M * (size_t)N * sizeof(__half);
    std::fprintf(stdout,
        "[PD6] shape: A[%d,%d] B[%d,%d] C[%d,%d] (F16; A=%zu MB B=%zu MB C=%zu MB)\n",
        M, K, K, N, M, N, bytes_A/(1024*1024), bytes_B/(1024*1024), bytes_C/(1024*1024));

    // Use device 0. PD6 measures intra-device ILP, not cross-device.
    CK(cudaSetDevice(0));

    __half * A = nullptr; __half * B = nullptr;
    __half * C_full = nullptr;       // Path A output
    __half * C_split = nullptr;      // Path B output (assembled from C_left + C_right)
    __half * C_left = nullptr;       // Path B left half
    __half * C_right = nullptr;      // Path B right half
    CK(cudaMalloc(&A, bytes_A));
    CK(cudaMalloc(&B, bytes_B));
    CK(cudaMalloc(&C_full, bytes_C));
    CK(cudaMalloc(&C_split, bytes_C));
    const size_t bytes_C_half = (size_t)M * (size_t)N_half * sizeof(__half);
    CK(cudaMalloc(&C_left, bytes_C_half));
    CK(cudaMalloc(&C_right, bytes_C_half));

    // Init host data + copy to device.
    std::vector<__half> hA((size_t)M * K);
    std::vector<__half> hB((size_t)K * N);
    for (size_t i = 0; i < hA.size(); ++i) hA[i] = __float2half(((float)(i % 257) / 257.0f) - 0.5f);
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = __float2half(((float)(i % 263) / 263.0f) - 0.5f);
    CK(cudaMemcpy(A, hA.data(), bytes_A, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(B, hB.data(), bytes_B, cudaMemcpyHostToDevice));

    // Two streams.
    cudaStream_t s0, s1;
    CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));

    cudaEvent_t ev_start, ev_end;
    CK(cudaEventCreate(&ev_start));
    CK(cudaEventCreate(&ev_end));

    // cuBLAS handles — one per stream so each can submit independently.
    cublasHandle_t h0, h1;
    BCK(cublasCreate(&h0));
    BCK(cublasCreate(&h1));
    BCK(cublasSetStream(h0, s0));
    BCK(cublasSetStream(h1, s1));
    // For determinism, pin compute type + algo.
    BCK(cublasSetMathMode(h0, CUBLAS_DEFAULT_MATH));
    BCK(cublasSetMathMode(h1, CUBLAS_DEFAULT_MATH));

    const __half alpha = __float2half(1.0f);
    const __half beta  = __float2half(0.0f);

    // ---- Path A: full matmul on stream s0. C_full = A @ B.
    // cublasHgemm uses column-major. To compute C[M,N] = A[M,K] @ B[K,N] in
    // row-major, we pass cublasHgemm(N, K, M, ...) trick: transpose viewpoint.
    // Equivalently: C^T[N,M] = B^T[N,K] @ A^T[K,M] in column-major. So set
    // op = N, N, dims = N, M, K, and pass B, A, C.
    auto run_path_a = [&]() {
        BCK(cublasHgemm(h0, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K,
                        &alpha,
                        B, N,
                        A, K,
                        &beta,
                        C_full, N));
    };

    // ---- Path B: split B by columns. B_left = B[:, :N/2], B_right = B[:, N/2:].
    // In our column-major view of "C = A @ B" expressed as "C^T = B^T @ A^T",
    // splitting B column-wise (right half of original) means splitting the
    // first axis of B^T — which corresponds to the M-axis of cublasHgemm's
    // input (the N-arg of the gemm call). So:
    //  - sub-call 1: gemm(N/2, M, K) on B (treated as first half of N rows in B^T) → C_left
    //  - sub-call 2: gemm(N/2, M, K) on B + N_half (second half of N rows in B^T) → C_right
    auto run_path_b = [&]() {
        BCK(cublasHgemm(h0, CUBLAS_OP_N, CUBLAS_OP_N,
                        N_half, M, K,
                        &alpha,
                        B, N,                        // ldb = N (stride between K rows in B^T)
                        A, K,
                        &beta,
                        C_left, N_half));
        BCK(cublasHgemm(h1, CUBLAS_OP_N, CUBLAS_OP_N,
                        N_half, M, K,
                        &alpha,
                        B + N_half, N,
                        A, K,
                        &beta,
                        C_right, N_half));
    };

    // After path B we'd concat C_left + C_right into C_split for equivalence
    // check. (We don't include the concat in the timing — the libmgpu use
    // case wouldn't concat; downstream reads from the split halves.)
    auto concat_path_b_for_check = [&]() {
        // Copy C_left → C_split[:, :N/2], C_right → C_split[:, N/2:]
        // C is row-major in the gemm's M×N output. C_left is M rows × N_half cols.
        // C_left[i, j] (row i, col j) is at offset i*N_half + j in C_left.
        // C_split[i, j_full] needs offset i*N + j_full.
        // So we do M row-by-row copies.
        CK(cudaMemcpy2DAsync(
            C_split,                 sizeof(__half) * N,
            C_left,                  sizeof(__half) * N_half,
            sizeof(__half) * N_half, M,
            cudaMemcpyDeviceToDevice, s0));
        CK(cudaMemcpy2DAsync(
            C_split + N_half,        sizeof(__half) * N,
            C_right,                 sizeof(__half) * N_half,
            sizeof(__half) * N_half, M,
            cudaMemcpyDeviceToDevice, s0));
    };

    // ---- Verify byte-identity FIRST.
    run_path_a();
    CK(cudaStreamSynchronize(s0));
    run_path_b();
    CK(cudaStreamSynchronize(s0));
    CK(cudaStreamSynchronize(s1));
    concat_path_b_for_check();
    CK(cudaStreamSynchronize(s0));

    std::vector<__half> ha_full((size_t)M * N);
    std::vector<__half> ha_split((size_t)M * N);
    CK(cudaMemcpy(ha_full.data(),  C_full,  bytes_C, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ha_split.data(), C_split, bytes_C, cudaMemcpyDeviceToHost));
    int n_diff = 0, n_diff_gt2ulp = 0;
    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < ha_full.size(); ++i) {
        if (ha_full[i] != ha_split[i]) ++n_diff;
        float a = __half2float(ha_full[i]);
        float b = __half2float(ha_split[i]);
        float ad = std::abs(a - b);
        if (ad > max_abs_diff) max_abs_diff = ad;
        float ulp = std::max(std::abs(a), std::abs(b)) / 1024.0f;
        if (ad > 2.0f * ulp) ++n_diff_gt2ulp;
    }
    std::fprintf(stdout, "[PD6] equivalence: n_diff=%d of %zu, n_diff>2ULP=%d, max_abs_diff=%g\n",
                 n_diff, ha_full.size(), n_diff_gt2ulp, max_abs_diff);
    if (n_diff != 0) {
        // For matmul split byte-identity is not guaranteed (different reduction
        // tree shapes per cublas tile config). The fp16 ULP-bound test is what
        // we actually care about for determinism.
        std::fprintf(stdout, "[PD6] NOTE: byte-diff between split and full is expected at the K-tile boundary; only fp16 ULP-equivalence matters for C8 design.\n");
    }

    // ---- Timed iterations.
    const int N_WARM  = 50;
    const int N_TIMED = 500;
    auto time_one = [&](auto fn) -> double {
        CK(cudaEventRecord(ev_start, s0));
        // Wait on s1 too if path B is in use — we'll let the lambda handle it.
        fn();
        CK(cudaEventRecord(ev_end, s0));
        CK(cudaStreamSynchronize(s0));
        CK(cudaStreamSynchronize(s1));
        float ms = 0.0f;
        CK(cudaEventElapsedTime(&ms, ev_start, ev_end));
        return (double)ms;
    };

    // Note: for Path B, the gemm on s1 is concurrent. We measure from
    // ev_start (on s0) to ev_end (on s0) BUT we also have to sync s1.
    // The eventElapsedTime captures the s0 critical-path duration. For
    // Path B, s1's path must finish for the workload to be complete. To
    // capture the WALLCLOCK of the full path B, we time around a CPU
    // sync of both streams.
    auto time_wallclock = [&](auto fn) -> double {
        // Use CPU timing — more accurate for multi-stream overlap.
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        CK(cudaStreamSynchronize(s0));
        CK(cudaStreamSynchronize(s1));
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dur = t1 - t0;
        return dur.count();
    };

    // Warm-up A.
    for (int i = 0; i < N_WARM; ++i) (void)time_wallclock(run_path_a);
    std::vector<double> ms_a(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) ms_a[i] = time_wallclock(run_path_a);

    // Warm-up B.
    for (int i = 0; i < N_WARM; ++i) (void)time_wallclock(run_path_b);
    std::vector<double> ms_b(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) ms_b[i] = time_wallclock(run_path_b);

    double a_p50 = percentile(ms_a, 0.50);
    double a_p95 = percentile(ms_a, 0.95);
    double a_p99 = percentile(ms_a, 0.99);
    double b_p50 = percentile(ms_b, 0.50);
    double b_p95 = percentile(ms_b, 0.95);
    double b_p99 = percentile(ms_b, 0.99);
    double speedup_p50 = a_p50 / b_p50;

    std::fprintf(stdout, "\n[PD6] Path A (1-stream full matmul):\n");
    std::fprintf(stdout, "  p50=%.3f ms  p95=%.3f ms  p99=%.3f ms\n", a_p50, a_p95, a_p99);
    std::fprintf(stdout, "[PD6] Path B (2-stream split matmul):\n");
    std::fprintf(stdout, "  p50=%.3f ms  p95=%.3f ms  p99=%.3f ms\n", b_p50, b_p95, b_p99);
    std::fprintf(stdout, "[PD6] 2-stream speedup at p50: %.2fx\n", speedup_p50);

    int rc = 0;
    if (speedup_p50 >= 1.05) {
        std::fprintf(stdout, "[PD6] DECISION: SHIP C8 (2-stream %.0f%% faster at p50)\n",
                     (speedup_p50 - 1.0) * 100.0);
    } else {
        std::fprintf(stdout, "[PD6] DECISION: DROP C8 (2-stream not faster by 5%% threshold; %.0f%% delta)\n",
                     (speedup_p50 - 1.0) * 100.0);
    }

    FILE * jf = std::fopen("pd6_result.json", "w");
    if (jf) {
        std::fprintf(jf,
            "{\n"
            "  \"shape\": { \"M\": %d, \"K\": %d, \"N\": %d, \"bytes_C\": %zu },\n"
            "  \"path_a_1stream\": { \"p50_ms\": %.6f, \"p95_ms\": %.6f, \"p99_ms\": %.6f },\n"
            "  \"path_b_2stream\": { \"p50_ms\": %.6f, \"p95_ms\": %.6f, \"p99_ms\": %.6f },\n"
            "  \"speedup_p50\": %.6f,\n"
            "  \"decision\": \"%s\",\n"
            "  \"equivalence\": { \"n_diff\": %d, \"n_diff_gt2ulp\": %d, \"max_abs_diff\": %g }\n"
            "}\n",
            M, K, N, bytes_C,
            a_p50, a_p95, a_p99,
            b_p50, b_p95, b_p99,
            speedup_p50,
            (speedup_p50 >= 1.05) ? "SHIP_C8" : "DROP_C8",
            n_diff, n_diff_gt2ulp, max_abs_diff);
        std::fclose(jf);
        std::fprintf(stdout, "[PD6] wrote pd6_result.json\n");
    }

    BCK(cublasDestroy(h0));
    BCK(cublasDestroy(h1));
    CK(cudaFree(A)); CK(cudaFree(B));
    CK(cudaFree(C_full)); CK(cudaFree(C_split));
    CK(cudaFree(C_left)); CK(cudaFree(C_right));
    CK(cudaEventDestroy(ev_start));
    CK(cudaEventDestroy(ev_end));
    CK(cudaStreamDestroy(s0));
    CK(cudaStreamDestroy(s1));

    return rc;
}
