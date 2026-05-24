// AsyncReduce overlap microbench (Candidate B Step 1)
//
// Question: does cudaMemcpyPeerAsync on a dedicated comm stream
// overlap with kernel execution on a separate compute stream on TU102?
//
// Answer is binary:
//   YES → AsyncReduce Option B viable, can hide ~9% NCCL kernel-time
//         behind compute → green-light PHASE_ASYNC_REDUCE T1-T10 impl.
//   NO  → Option B degrades to Option A bandwidth cost (still byte-deterministic
//         but no perf win) → kill B, pivot to Candidate A's ncu probe.
//
// Methodology: 2 GPUs with peer access. On GPU0: launch a long-running
// compute kernel (~5 ms) on stream_compute. While it runs, issue a
// cudaMemcpyPeerAsync (2 MiB, mimicking AsyncReduce per-layer transfer
// size from DESIGN.md §"Hidden cost") on stream_comm. nsys-trace both.
// Inspect timeline: if memcpy fires while kernel is on the SM, overlap
// works.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

// Compute kernel: spin for a target wall time by doing busy-work.
// 1 block, 256 threads; each thread does a fused-multiply-add chain.
// Tuned so 5 ms of compute is reached at the inner-loop count below
// on a TU102 SM @ default clocks.
__global__ void busy_compute(float * __restrict__ x, int iters) {
    float a = (float)threadIdx.x * 0.001f;
    float b = (float)blockIdx.x  * 0.0001f;
    for (int i = 0; i < iters; ++i) {
        a = a * 1.000001f + b;
        b = b * 0.999999f + a;
    }
    // Sink to prevent dead-code elimination.
    if (a + b == -123456.789f) x[0] = a;
}

int main(int argc, char **argv) {
    const size_t TRANSFER_BYTES = 2 * 1024 * 1024; // 2 MiB, matches DESIGN.md per-layer reduce
    const int    COMPUTE_ITERS  = 5'000'000;       // ~5 ms on TU102 @ default clocks
    const int    N_ROUNDS       = 10;

    int n_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_devices));
    if (n_devices < 2) {
        fprintf(stderr, "Need at least 2 CUDA devices; got %d\n", n_devices);
        return 1;
    }

    // Enable peer access GPU0 ↔ GPU1.
    int can01 = 0, can10 = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can01, 0, 1));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can10, 1, 0));
    if (!can01 || !can10) {
        fprintf(stderr, "P2P not available 0->1=%d 1->0=%d\n", can01, can10);
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(0)); CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));
    CUDA_CHECK(cudaSetDevice(1)); CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0));

    // Allocate transfer buffers (one on each device).
    CUDA_CHECK(cudaSetDevice(0));
    void * buf0 = nullptr;
    float * x0 = nullptr;
    CUDA_CHECK(cudaMalloc(&buf0, TRANSFER_BYTES));
    CUDA_CHECK(cudaMalloc(&x0,   256 * sizeof(float)));

    CUDA_CHECK(cudaSetDevice(1));
    void * buf1 = nullptr;
    CUDA_CHECK(cudaMalloc(&buf1, TRANSFER_BYTES));

    // Two streams on GPU0: compute + comm. Comm uses non-blocking flag
    // so it doesn't serialize on the legacy default stream (matches
    // what AsyncReduce DESIGN.md §Architecture proposes).
    CUDA_CHECK(cudaSetDevice(0));
    cudaStream_t stream_compute, stream_comm;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_comm,    cudaStreamNonBlocking));

    // Warm-up: one round to amortize CUDA context init + kernel JIT.
    busy_compute<<<1, 256, 0, stream_compute>>>(x0, 100);
    CUDA_CHECK(cudaMemcpyPeerAsync(buf0, 0, buf1, 1, TRANSFER_BYTES, stream_comm));
    CUDA_CHECK(cudaStreamSynchronize(stream_compute));
    CUDA_CHECK(cudaStreamSynchronize(stream_comm));

    fprintf(stderr, "warm-up done; firing %d overlapped rounds\n", N_ROUNDS);

    // The actual test: for each round, launch compute + comm at the
    // same epoch, both on their own streams. If TU102 supports overlap,
    // nsys will show them as concurrent. If not, comm will serialize
    // after compute (or vice versa).
    for (int r = 0; r < N_ROUNDS; ++r) {
        // Long-running compute on GPU0 stream_compute.
        busy_compute<<<1, 256, 0, stream_compute>>>(x0, COMPUTE_ITERS);
        // Short transfer on GPU0 stream_comm (pulls 2 MiB from GPU1).
        CUDA_CHECK(cudaMemcpyPeerAsync(buf0, 0, buf1, 1, TRANSFER_BYTES, stream_comm));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_compute));
    CUDA_CHECK(cudaStreamSynchronize(stream_comm));

    fprintf(stderr, "done\n");

    CUDA_CHECK(cudaFree(buf0));
    CUDA_CHECK(cudaFree(x0));
    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaFree(buf1));
    return 0;
}
