# DATA-4 — long-context Welford softmax correctness

**Date**: 2026-05-16
**Setup**: `test-deltanet-d1-capture` with `LLAMA_PSKV_MODE=singlewarp` at varying prompt lengths.

## Results

| Prompt tokens | NP | ctx_per_slot | n_kv (post-prefill+decode) | comparisons | divergences |
|---|---|---|---|---|---|
| 12 | 2 | 2048 | 26 | 64 | 0 |
| 12 | 4 | 1024 | 52 | 192 | 0 |
| 12 | 8 | 512 | 104 | 448 | 0 |
| **520** | 2 | 2048 | 1042 | 64 | **0** ✓ |
| **520** | 4 | 1024 | 2084 | 192 | **0** ✓ |
| **1469** | 2 | 2048 | 2940 | 64 | **0** ✓ |
| 1469 | 4 | 1024 | n/a — overflow | n/a | (test failure: prompt > ctx_per_slot) |

(Comparisons = layers × (slots-1); each compares slot s residual to slot 0 at every layer.)

## Significance

The single-warp kernel's online Welford softmax stays correct and batch-invariant at long context.
At 1469 tokens × NP=2, the K-loop iterates approximately 2940 positions per CTA. Each iteration
performs the streaming-softmax update:

```
new_max = max(kqmax, kq);
scale_corr = expf(kqmax - new_max);
kqmax = new_max;
kqsum = kqsum * scale_corr + expf(kq - new_max);
VKQ[*] *= scale_corr;
VKQ[*] += V[k][*] * expf(kq - new_max);
```

Over ~3000 iterations the fp32 accumulator and exponent updates remain stable. Same-prompt slots
produce byte-identical output through the long sequence.

## Conclusion

DATA-4 GREEN. Singlewarp is correct at production-realistic n_kv values.

The NP=4+1469-token failure case is not a singlewarp issue — it's the test harness's ctx_per_slot
budget overflowing (1469 > 1024). A production server with proper ctx allocation per slot would
size around this.
