// PD2 — verify cudaStreamCaptureModeRelaxed supports cross-device dependencies
//
// PHASE_CUDA_NATIVE_DISPATCH.md §9 PD2.
//
// Question: can a single cudaStreamBeginCapture(...,Relaxed) on stream A
// of device 0 capture a graph that includes:
//   - kernel launches on stream B of device 1
//   - cross-device event wait edges (A waits for B, B waits for A)
// resulting in ONE cudaGraph_t that, when launched, executes ops on both
// devices with proper ordering?
//
// If yes:        Stage C3-C5 use single-graph capture (best perf, simplest)
// If no:         Stage C3-C5 use per-device subgraph capture +
//                cudaGraphAddDependencies stitching at parent-graph level
//                (more code, similar perf)
//
// Compile:
//   nvcc -ccbin /usr/bin/g++-15 -O2 -o pd2_capture pd2_multi_device_capture.cu
// Run:
//   ./pd2_capture

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, \
                     #call, cudaGetErrorString(e)); \
        std::exit(1); \
    } \
} while (0)

// Trivial kernel: add a scalar in-place. One thread.
__global__ void add_scalar(float * x, float v) {
    *x += v;
}

int main() {
    int n_dev = 0;
    CK(cudaGetDeviceCount(&n_dev));
    if (n_dev < 2) {
        std::fprintf(stderr, "PD2 needs >=2 devices; got %d\n", n_dev);
        return 77;
    }

    // Enable peer access both ways. Required for cross-device kernels
    // launched within a single captured graph in some configurations.
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

    // Allocate one float on each device, initialised to 0.
    float * d0_buf = nullptr;
    float * d1_buf = nullptr;
    CK(cudaSetDevice(0));
    CK(cudaMalloc(&d0_buf, sizeof(float)));
    CK(cudaMemset(d0_buf, 0, sizeof(float)));
    CK(cudaSetDevice(1));
    CK(cudaMalloc(&d1_buf, sizeof(float)));
    CK(cudaMemset(d1_buf, 0, sizeof(float)));

    // Create one non-default stream per device.
    cudaStream_t s0, s1;
    CK(cudaSetDevice(0));
    CK(cudaStreamCreateWithFlags(&s0, cudaStreamNonBlocking));
    CK(cudaSetDevice(1));
    CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));

    // Create cross-device events.
    cudaEvent_t e0_done, e1_done;
    CK(cudaSetDevice(0));
    CK(cudaEventCreateWithFlags(&e0_done, cudaEventDisableTiming));
    CK(cudaSetDevice(1));
    CK(cudaEventCreateWithFlags(&e1_done, cudaEventDisableTiming));

    // ----- The pivotal test: begin-capture on s0 (CUDA0), then enqueue
    // ----- work on s1 (CUDA1) inside the capture region.

    std::fprintf(stdout, "[PD2] Begin capture on s0 (CUDA0) mode=Relaxed\n");
    CK(cudaSetDevice(0));
    CK(cudaStreamBeginCapture(s0, cudaStreamCaptureModeRelaxed));

    // (a) Step on CUDA0: d0_buf += 1.0
    std::fprintf(stdout, "[PD2] enqueue kernel on s0 (CUDA0): d0_buf += 1.0\n");
    add_scalar<<<1, 1, 0, s0>>>(d0_buf, 1.0f);
    CK(cudaEventRecord(e0_done, s0));

    // (b) Cross to CUDA1; have s1 wait for e0_done; then d1_buf += 2.0
    std::fprintf(stdout, "[PD2] cudaSetDevice(1); s1.wait(e0_done); kernel: d1_buf += 2.0\n");
    CK(cudaSetDevice(1));
    CK(cudaStreamWaitEvent(s1, e0_done, 0));
    add_scalar<<<1, 1, 0, s1>>>(d1_buf, 2.0f);
    CK(cudaEventRecord(e1_done, s1));

    // (c) Back to CUDA0; have s0 wait for e1_done; then d0_buf += 4.0
    std::fprintf(stdout, "[PD2] cudaSetDevice(0); s0.wait(e1_done); kernel: d0_buf += 4.0\n");
    CK(cudaSetDevice(0));
    CK(cudaStreamWaitEvent(s0, e1_done, 0));
    add_scalar<<<1, 1, 0, s0>>>(d0_buf, 4.0f);

    cudaGraph_t graph = nullptr;
    std::fprintf(stdout, "[PD2] End capture on s0\n");
    CK(cudaStreamEndCapture(s0, &graph));

    if (graph == nullptr) {
        std::fprintf(stderr, "[PD2] FAIL: cudaStreamEndCapture returned null graph\n");
        return 1;
    }

    // Inspect the graph: how many nodes did we get? Should be >= 5
    // (3 kernel nodes + 2 event records + 2 event waits = 7 ideally).
    size_t n_nodes = 0;
    CK(cudaGraphGetNodes(graph, nullptr, &n_nodes));
    std::fprintf(stdout, "[PD2] captured graph has %zu nodes\n", n_nodes);

    // Instantiate + launch.
    cudaGraphExec_t exec = nullptr;
    cudaGraphNode_t err_node = nullptr;
    char err_log[1024] = {};
    cudaError_t inst = cudaGraphInstantiate(&exec, graph, &err_node, err_log, sizeof(err_log));
    if (inst != cudaSuccess) {
        std::fprintf(stderr, "[PD2] FAIL: cudaGraphInstantiate -> %s\n",
                     cudaGetErrorString(inst));
        std::fprintf(stderr, "[PD2]   err_log: %s\n", err_log);
        return 1;
    }
    std::fprintf(stdout, "[PD2] graph instantiated\n");

    // Launch the graph on s0 of CUDA0.
    CK(cudaSetDevice(0));
    CK(cudaGraphLaunch(exec, s0));
    CK(cudaStreamSynchronize(s0));

    // Verify outputs.
    float h_d0 = -1.0f, h_d1 = -1.0f;
    CK(cudaMemcpy(&h_d0, d0_buf, sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaSetDevice(1));
    CK(cudaMemcpy(&h_d1, d1_buf, sizeof(float), cudaMemcpyDeviceToHost));

    std::fprintf(stdout, "[PD2] d0_buf = %f (expected 5.0 = 1.0 + 4.0)\n", h_d0);
    std::fprintf(stdout, "[PD2] d1_buf = %f (expected 2.0)\n", h_d1);

    int rc = 0;
    if (h_d0 != 5.0f || h_d1 != 2.0f) {
        std::fprintf(stderr, "[PD2] FAIL: output mismatch\n");
        rc = 1;
    } else {
        std::fprintf(stdout, "[PD2] PASS: cross-device capture works\n");
    }

    // Cleanup.
    CK(cudaGraphExecDestroy(exec));
    CK(cudaGraphDestroy(graph));
    CK(cudaSetDevice(0));
    CK(cudaEventDestroy(e0_done));
    CK(cudaStreamDestroy(s0));
    CK(cudaFree(d0_buf));
    CK(cudaSetDevice(1));
    CK(cudaEventDestroy(e1_done));
    CK(cudaStreamDestroy(s1));
    CK(cudaFree(d1_buf));

    return rc;
}
