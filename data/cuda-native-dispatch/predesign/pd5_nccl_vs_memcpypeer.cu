// PD5 — NCCL ncclAllReduce vs cudaMemcpyPeer + add on libmgpu's reduce shape
//
// PHASE_CUDA_NATIVE_DISPATCH.md §9 PD5.
//
// Question: at libmgpu's CLIP cross-device REDUCE shape (per-device partial
// output, summed across 2 devices), is NCCL's ncclAllReduce >=10% faster
// than the current cudaMemcpyPeer + element-wise add path?
//
// Threshold: NCCL >=10% faster → C9 ships (re-enable NCCL for libmgpu reduce).
//            Otherwise drop C9 from PHASE_CUDA_NATIVE_DISPATCH.
//
// Methodology:
//   - Allocate per-device input tensors of the target shape (F16, default
//     [hidden_dim, n_tokens] = [1280, 1024] = 1.3M elements = 2.6 MB each).
//   - Path A (memcpy-peer + add): for each device d, peer-copy other device's
//     buffer to a scratch tensor on d, then launch an add kernel that sums
//     scratch + local in-place to local. After both devices' adds, both
//     local buffers hold the sum.
//   - Path B (NCCL): ncclAllReduce(local, local, ncclSum) on both devices.
//   - 1000 iterations, drop first 100 warm-up, compute p50/p95/p99.
//   - Verify byte-identity between the two paths (or fp16 ULP equivalence).
//
// Compile:
//   nvcc -ccbin /usr/bin/g++-15 -O2 -o pd5_nccl pd5_nccl_vs_memcpypeer.cu -lnccl
// Run:
//   ./pd5_nccl              # defaults: [1280, 1024] F16
//   ./pd5_nccl 1280 4096    # try at a larger sequence
//   ./pd5_nccl 2560 1024    # try at a larger hidden dim

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <nccl.h>

#define CK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, \
                     #call, cudaGetErrorString(e)); std::exit(1); \
    } \
} while (0)

#define NCK(call) do { \
    ncclResult_t e = (call); \
    if (e != ncclSuccess) { \
        std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, \
                     #call, ncclGetErrorString(e)); std::exit(1); \
    } \
} while (0)

// Element-wise add in fp16: dst[i] = local[i] + remote[i]. One thread per element.
__global__ void add_inplace_fp16(__half * local, const __half * remote, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = __half2float(local[i]);
        float b = __half2float(remote[i]);
        local[i] = __float2half(a + b);
    }
}

// Compute the median of a sorted vector.
static double percentile(std::vector<double> & v, double pct) {
    std::sort(v.begin(), v.end());
    size_t idx = (size_t)(pct * v.size());
    if (idx >= v.size()) idx = v.size() - 1;
    return v[idx];
}

int main(int argc, char ** argv) {
    int hidden_dim = 1280;
    int n_tokens   = 1024;
    if (argc >= 2) hidden_dim = std::atoi(argv[1]);
    if (argc >= 3) n_tokens   = std::atoi(argv[2]);
    const size_t N = (size_t)hidden_dim * (size_t)n_tokens;
    const size_t BYTES = N * sizeof(__half);
    std::fprintf(stdout,
        "[PD5] shape: hidden_dim=%d n_tokens=%d total=%zu elem (%zu MB F16)\n",
        hidden_dim, n_tokens, N, BYTES / (1024*1024));

    int n_dev = 0;
    CK(cudaGetDeviceCount(&n_dev));
    if (n_dev < 2) { std::fprintf(stderr, "PD5 needs >=2 devices; got %d\n", n_dev); return 77; }

    // Enable peer access.
    CK(cudaSetDevice(0));
    cudaError_t pa01 = cudaDeviceEnablePeerAccess(1, 0);
    if (pa01 != cudaSuccess && pa01 != cudaErrorPeerAccessAlreadyEnabled) {
        std::fprintf(stderr, "0->1 peer: %s\n", cudaGetErrorString(pa01));
    }
    CK(cudaSetDevice(1));
    cudaError_t pa10 = cudaDeviceEnablePeerAccess(0, 0);
    if (pa10 != cudaSuccess && pa10 != cudaErrorPeerAccessAlreadyEnabled) {
        std::fprintf(stderr, "1->0 peer: %s\n", cudaGetErrorString(pa10));
    }

    // Allocate input/output buffers per device.
    __half * d0 = nullptr; __half * d1 = nullptr;
    __half * scratch0 = nullptr; __half * scratch1 = nullptr;
    // For NCCL: separate buffers so we can re-init between iterations.
    __half * n0 = nullptr; __half * n1 = nullptr;

    CK(cudaSetDevice(0));
    CK(cudaMalloc(&d0, BYTES));
    CK(cudaMalloc(&scratch0, BYTES));
    CK(cudaMalloc(&n0, BYTES));
    CK(cudaSetDevice(1));
    CK(cudaMalloc(&d1, BYTES));
    CK(cudaMalloc(&scratch1, BYTES));
    CK(cudaMalloc(&n1, BYTES));

    // Initialize host-side input.
    std::vector<__half> h0(N), h1(N);
    for (size_t i = 0; i < N; ++i) {
        h0[i] = __float2half((float)(i & 0xFF) / 256.0f);
        h1[i] = __float2half((float)((i + 1) & 0xFF) / 256.0f);
    }

    // Streams + events.
    cudaStream_t s0, s1;
    CK(cudaSetDevice(0));
    CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    CK(cudaSetDevice(1));
    CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    cudaEvent_t e0_start, e0_end, e1_start, e1_end;
    CK(cudaSetDevice(0));
    CK(cudaEventCreate(&e0_start));
    CK(cudaEventCreate(&e0_end));
    CK(cudaSetDevice(1));
    CK(cudaEventCreate(&e1_start));
    CK(cudaEventCreate(&e1_end));

    // Setup NCCL: one comm per device, both in same group.
    ncclComm_t comms[2];
    int devs[2] = {0, 1};
    NCK(ncclCommInitAll(comms, 2, devs));

    // --- Reset buffers helper. ---
    auto reset = [&]() {
        CK(cudaSetDevice(0));
        CK(cudaMemcpyAsync(d0, h0.data(), BYTES, cudaMemcpyHostToDevice, s0));
        CK(cudaMemcpyAsync(n0, h0.data(), BYTES, cudaMemcpyHostToDevice, s0));
        CK(cudaSetDevice(1));
        CK(cudaMemcpyAsync(d1, h1.data(), BYTES, cudaMemcpyHostToDevice, s1));
        CK(cudaMemcpyAsync(n1, h1.data(), BYTES, cudaMemcpyHostToDevice, s1));
        CK(cudaStreamSynchronize(s0));
        CK(cudaSetDevice(1));
        CK(cudaStreamSynchronize(s1));
    };

    // --- Verify equivalence FIRST: run one Path A and one Path B, compare. ---
    reset();
    // Path A on this iteration: peer copy + add.
    // d0 <- d0 + d1, d1 <- d1 + d0_old (but order matters; for sum it's commutative)
    CK(cudaSetDevice(0));
    CK(cudaMemcpyPeerAsync(scratch0, 0, d1, 1, BYTES, s0));
    {
        dim3 blk(256), grd((N + 255) / 256);
        add_inplace_fp16<<<grd, blk, 0, s0>>>(d0, scratch0, N);
    }
    CK(cudaSetDevice(1));
    CK(cudaMemcpyPeerAsync(scratch1, 1, d0, 0, BYTES, s1));  // d0 already updated on dev 0
    // Actually for fair comparison, use the pre-add d0 for scratch1 source.
    // Reset and re-do correctly:
    reset();
    CK(cudaSetDevice(0));
    CK(cudaMemcpyPeerAsync(scratch0, 0, d1, 1, BYTES, s0));
    CK(cudaSetDevice(1));
    CK(cudaMemcpyPeerAsync(scratch1, 1, d0, 0, BYTES, s1));
    CK(cudaSetDevice(0));
    CK(cudaStreamSynchronize(s0));
    CK(cudaSetDevice(1));
    CK(cudaStreamSynchronize(s1));
    {
        CK(cudaSetDevice(0));
        dim3 blk(256), grd((N + 255) / 256);
        add_inplace_fp16<<<grd, blk, 0, s0>>>(d0, scratch0, N);
        CK(cudaSetDevice(1));
        add_inplace_fp16<<<grd, blk, 0, s1>>>(d1, scratch1, N);
    }
    CK(cudaSetDevice(0));
    CK(cudaStreamSynchronize(s0));
    CK(cudaSetDevice(1));
    CK(cudaStreamSynchronize(s1));

    // Path B: NCCL allreduce on n0/n1.
    NCK(ncclGroupStart());
    NCK(ncclAllReduce(n0, n0, N, ncclHalf, ncclSum, comms[0], s0));
    NCK(ncclAllReduce(n1, n1, N, ncclHalf, ncclSum, comms[1], s1));
    NCK(ncclGroupEnd());
    CK(cudaSetDevice(0));
    CK(cudaStreamSynchronize(s0));
    CK(cudaSetDevice(1));
    CK(cudaStreamSynchronize(s1));

    // Compare A and B (using device 0).
    std::vector<__half> ha(N), hb(N);
    CK(cudaSetDevice(0));
    CK(cudaMemcpy(ha.data(), d0, BYTES, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hb.data(), n0, BYTES, cudaMemcpyDeviceToHost));
    int n_diff = 0, n_diff_gt1ulp = 0;
    float max_abs_diff = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        float a = __half2float(ha[i]);
        float b = __half2float(hb[i]);
        if (ha[i] != hb[i]) ++n_diff;
        float ad = std::abs(a - b);
        if (ad > max_abs_diff) max_abs_diff = ad;
        // fp16 ULP at value v: about |v| * 2^-10 = |v| / 1024
        float ulp = std::max(std::abs(a), std::abs(b)) / 1024.0f;
        if (ad > 2.0f * ulp) ++n_diff_gt1ulp;
    }
    std::fprintf(stdout, "[PD5] equivalence: n_diff=%d of %zu, n_diff>2ULP=%d, max_abs_diff=%g\n",
                 n_diff, N, n_diff_gt1ulp, max_abs_diff);
    if (n_diff_gt1ulp > 0) {
        std::fprintf(stderr, "[PD5] WARN: NCCL vs memcpy+add differ by >2 fp16 ULPs at %d elems\n", n_diff_gt1ulp);
    }

    // --- Now run timed iterations. ---
    const int N_WARM = 100;
    const int N_TIMED = 1000;

    auto time_one = [&](auto fn) -> double {
        // returns elapsed ms across both devices (we use the max of the two)
        reset();
        CK(cudaSetDevice(0));
        CK(cudaEventRecord(e0_start, s0));
        CK(cudaSetDevice(1));
        CK(cudaEventRecord(e1_start, s1));
        fn();
        CK(cudaSetDevice(0));
        CK(cudaEventRecord(e0_end, s0));
        CK(cudaStreamSynchronize(s0));
        CK(cudaSetDevice(1));
        CK(cudaEventRecord(e1_end, s1));
        CK(cudaStreamSynchronize(s1));
        float ms0 = 0.0f, ms1 = 0.0f;
        CK(cudaSetDevice(0));
        CK(cudaEventElapsedTime(&ms0, e0_start, e0_end));
        CK(cudaSetDevice(1));
        CK(cudaEventElapsedTime(&ms1, e1_start, e1_end));
        return std::max((double)ms0, (double)ms1);
    };

    // Warm-up A.
    for (int i = 0; i < N_WARM; ++i) {
        time_one([&]() {
            CK(cudaSetDevice(0));
            CK(cudaMemcpyPeerAsync(scratch0, 0, d1, 1, BYTES, s0));
            CK(cudaSetDevice(1));
            CK(cudaMemcpyPeerAsync(scratch1, 1, d0, 0, BYTES, s1));
            CK(cudaStreamSynchronize(s0));
            CK(cudaSetDevice(0));
            CK(cudaStreamSynchronize(s1));
            CK(cudaSetDevice(0));
            dim3 blk(256), grd((N + 255) / 256);
            add_inplace_fp16<<<grd, blk, 0, s0>>>(d0, scratch0, N);
            CK(cudaSetDevice(1));
            add_inplace_fp16<<<grd, blk, 0, s1>>>(d1, scratch1, N);
        });
    }
    std::vector<double> ms_a(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        ms_a[i] = time_one([&]() {
            CK(cudaSetDevice(0));
            CK(cudaMemcpyPeerAsync(scratch0, 0, d1, 1, BYTES, s0));
            CK(cudaSetDevice(1));
            CK(cudaMemcpyPeerAsync(scratch1, 1, d0, 0, BYTES, s1));
            CK(cudaStreamSynchronize(s0));
            CK(cudaSetDevice(0));
            CK(cudaStreamSynchronize(s1));
            CK(cudaSetDevice(0));
            dim3 blk(256), grd((N + 255) / 256);
            add_inplace_fp16<<<grd, blk, 0, s0>>>(d0, scratch0, N);
            CK(cudaSetDevice(1));
            add_inplace_fp16<<<grd, blk, 0, s1>>>(d1, scratch1, N);
        });
    }

    // Warm-up B.
    for (int i = 0; i < N_WARM; ++i) {
        time_one([&]() {
            NCK(ncclGroupStart());
            NCK(ncclAllReduce(n0, n0, N, ncclHalf, ncclSum, comms[0], s0));
            NCK(ncclAllReduce(n1, n1, N, ncclHalf, ncclSum, comms[1], s1));
            NCK(ncclGroupEnd());
        });
    }
    std::vector<double> ms_b(N_TIMED);
    for (int i = 0; i < N_TIMED; ++i) {
        ms_b[i] = time_one([&]() {
            NCK(ncclGroupStart());
            NCK(ncclAllReduce(n0, n0, N, ncclHalf, ncclSum, comms[0], s0));
            NCK(ncclAllReduce(n1, n1, N, ncclHalf, ncclSum, comms[1], s1));
            NCK(ncclGroupEnd());
        });
    }

    double a_p50 = percentile(ms_a, 0.50);
    double a_p95 = percentile(ms_a, 0.95);
    double a_p99 = percentile(ms_a, 0.99);
    double b_p50 = percentile(ms_b, 0.50);
    double b_p95 = percentile(ms_b, 0.95);
    double b_p99 = percentile(ms_b, 0.99);
    double speedup_p50 = a_p50 / b_p50;

    std::fprintf(stdout, "\n[PD5] Path A (memcpy-peer + add):\n");
    std::fprintf(stdout, "  p50=%.3f ms  p95=%.3f ms  p99=%.3f ms\n", a_p50, a_p95, a_p99);
    std::fprintf(stdout, "[PD5] Path B (NCCL ncclAllReduce):\n");
    std::fprintf(stdout, "  p50=%.3f ms  p95=%.3f ms  p99=%.3f ms\n", b_p50, b_p95, b_p99);
    std::fprintf(stdout, "[PD5] NCCL speedup at p50: %.2fx\n", speedup_p50);

    // Threshold: NCCL >=10% faster (A/B >= 1.10) → SHIP
    int rc = 0;
    if (speedup_p50 >= 1.10) {
        std::fprintf(stdout, "[PD5] DECISION: SHIP C9 (NCCL %.0f%% faster at p50)\n",
                     (speedup_p50 - 1.0) * 100.0);
    } else {
        std::fprintf(stdout, "[PD5] DECISION: DROP C9 (NCCL not faster by 10%% threshold; %.0f%% delta)\n",
                     (speedup_p50 - 1.0) * 100.0);
    }

    // Write a json summary next to the binary.
    FILE * jf = std::fopen("pd5_result.json", "w");
    if (jf) {
        std::fprintf(jf,
            "{\n"
            "  \"shape\": { \"hidden_dim\": %d, \"n_tokens\": %d, \"bytes_fp16\": %zu },\n"
            "  \"path_a_memcpypeer_add\": { \"p50_ms\": %.6f, \"p95_ms\": %.6f, \"p99_ms\": %.6f },\n"
            "  \"path_b_nccl\": { \"p50_ms\": %.6f, \"p95_ms\": %.6f, \"p99_ms\": %.6f },\n"
            "  \"speedup_p50\": %.6f,\n"
            "  \"decision\": \"%s\",\n"
            "  \"equivalence\": { \"n_diff\": %d, \"n_diff_gt2ulp\": %d, \"max_abs_diff\": %g }\n"
            "}\n",
            hidden_dim, n_tokens, BYTES,
            a_p50, a_p95, a_p99,
            b_p50, b_p95, b_p99,
            speedup_p50,
            (speedup_p50 >= 1.10) ? "SHIP_C9" : "DROP_C9",
            n_diff, n_diff_gt1ulp, max_abs_diff);
        std::fclose(jf);
        std::fprintf(stdout, "[PD5] wrote pd5_result.json\n");
    }

    // Cleanup.
    NCK(ncclCommDestroy(comms[0]));
    NCK(ncclCommDestroy(comms[1]));
    CK(cudaSetDevice(0));
    CK(cudaFree(d0)); CK(cudaFree(scratch0)); CK(cudaFree(n0));
    CK(cudaEventDestroy(e0_start)); CK(cudaEventDestroy(e0_end));
    CK(cudaStreamDestroy(s0));
    CK(cudaSetDevice(1));
    CK(cudaFree(d1)); CK(cudaFree(scratch1)); CK(cudaFree(n1));
    CK(cudaEventDestroy(e1_start)); CK(cudaEventDestroy(e1_end));
    CK(cudaStreamDestroy(s1));

    return rc;
}
