# Phase 1: Peer Host Quickstart — Qwen3.5-9B + MTP on Vega 64 8 GiB

Self-contained. No external file transfers. Run top to bottom.

## 0. Clarification

"Qwen3.5 8B" → use Qwen3.5-9B (closest Qwen3.5 dense variant; has native MTP weights).

## 1. Prereqs

```bash
sudo apt install -y git cmake build-essential python3 python3-venv \
  libvulkan-dev vulkan-tools mesa-vulkan-drivers glslc libcurl4-openssl-dev
# Arch: pacman -S git cmake base-devel python vulkan-headers vulkan-tools mesa vulkan-radeon glslang curl
```

## 2. Source + build (Vulkan)

```bash
git clone git@github.com:slartibardfast/llama.cpp.git ~/src/qwen35-mtp
cd ~/src/qwen35-mtp && git checkout polaris-hybrid-cpu-opt
git log -1 --format='%H %s'    # must show: 7557ddd33 polaris: wire inline MTP producer...

# Sanity: NEXTN classification fix present?
grep -A1 NEXTN_EH_PROJ src/llama-arch.cpp | grep -q LAYER_REPEATING \
  && echo "NEXTN OK" || { echo "REGRESSION — reapply a4ed1af94"; exit 1; }

cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-vk --target llama-server llama-completion llama-quantize llama-batched-bench -j$(nproc)
```

## 3. Conversion venv

```bash
cd ~/src/qwen35-mtp
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install -e gguf-py    # MUST be this tree's gguf-py, editable
.venv/bin/pip install transformers sentencepiece 'protobuf>=4.21,<5' safetensors huggingface_hub
.venv/bin/pip show gguf | grep Editable    # must point at ~/src/qwen35-mtp/gguf-py
```

## 4. Download model

```bash
mkdir -p ~/models
.venv/bin/hf download Qwen/Qwen3.5-9B --local-dir ~/models/Qwen3.5-9B-HF

.venv/bin/python -c "
import json
c = json.load(open('$HOME/models/Qwen3.5-9B-HF/config.json'))
tc = c.get('text_config', c)
print('arch:', c['architectures'], 'layers:', tc['num_hidden_layers'],
      'mtp:', tc.get('mtp_num_hidden_layers',0), 'hidden:', tc['hidden_size'])
"
# Expect: arch=['Qwen3_5ForConditionalGeneration'], mtp=1
```

## 5. Convert HF → GGUF f16 (with MTP)

```bash
.venv/bin/python convert_hf_to_gguf.py \
  ~/models/Qwen3.5-9B-HF --outtype f16 \
  --outfile ~/models/Qwen3.5-9B-mtp-f16.gguf
# Log must show blk.<n_main>.nextn.{eh_proj,enorm,hnorm,shared_head_norm}.weight
```

## 6. Quantize — Vega 64 8 GiB fit recipe

```bash
./build-vk/bin/llama-quantize \
  --output-tensor-type q5_k \
  --token-embedding-type q5_k \
  --tensor-type "attn_q=q5_k" \
  --tensor-type "attn_k=q5_k" \
  --tensor-type "attn_v=q5_k" \
  --tensor-type "attn_output=q5_k" \
  --tensor-type "ssm_=q6_k" \
  --tensor-type "nextn=q8_0" \
  ~/models/Qwen3.5-9B-mtp-f16.gguf \
  ~/models/Qwen3.5-9B-mtp-q4km.gguf \
  Q4_K_M $(nproc)
```

Target: ~5.5 GB weights (leaves ~2.5 GB for compute buffer + KV at n_ctx=4096).

If it OOMs at runtime → drop embed to `q4_k` or re-quantize with plain `Q4_K_M` (no custom `--tensor-type` flags) for ~5.0 GB.

## 7. Validate

```bash
.venv/bin/python - <<'PY' ~/models/Qwen3.5-9B-mtp-q4km.gguf
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
kv = {f.name: f for f in r.fields.values()}
arch = next(a for a in ("qwen35","qwen35moe") if f"{a}.block_count" in kv)
def sc(k): f=kv.get(k); return f.parts[-1][0] if f else None
nextn = sc(f"{arch}.nextn_predict_layers") or 0
bc = sc(f"{arch}.block_count")
names = {t.name for t in r.tensors}
mtp_idx = bc - nextn
need = [f"blk.{mtp_idx}.nextn.{n}.weight" for n in ("eh_proj","enorm","hnorm","shared_head_norm")]
miss = [n for n in need if n not in names]
print(f"arch={arch} block_count={bc} nextn={nextn} tensors={len(list(r.tensors))}")
print("PASS" if nextn==1 and not miss else f"FAIL missing={miss}")
PY
```

## 8. Run

```bash
# Full GPU offload with MTP spec decode (auto-enabled)
./build-vk/bin/llama-server \
  -m ~/models/Qwen3.5-9B-mtp-q4km.gguf \
  -c 4096 -ngl 99 -np 1 -fit off \
  --host 127.0.0.1 --port 9099 --no-warmup
```

In the startup log, confirm:

- `offloaded <n>/<n> layers to GPU` (should be the full layer count, e.g. `34/34` for Qwen3.5-9B including the MTP block)
- **Model weights fit:** `Vulkan0 model buffer size` + `CPU_Mapped model buffer size` sums to ~5665 MiB. This build of the polaris branch keeps the token embedding + output tensors on a CPU `mmap`, so the `Vulkan0` line alone shows ~5000 MiB, not ~5500 MiB. Don't flag that as a regression — check the **sum** of the two buffer-size lines against the quantized file size.
- `auto-enabling MTP speculative decoding`
- `server is listening`

## 9. Byte-identical equivalence check (spec vs non-spec)

```bash
# Server with MTP
./build-vk/bin/llama-server -m ~/models/Qwen3.5-9B-mtp-q4km.gguf \
  -c 4096 -ngl 99 -np 1 -fit off --host 127.0.0.1 --port 9099 --no-warmup &
PID=$!; sleep 30
curl -s localhost:9099/completion -d '{"prompt":"Once upon a time","n_predict":64,"temperature":0,"seed":42,"return_tokens":true}' > /tmp/spec.json
kill $PID; sleep 2

# Same server, MTP disabled via env var
LLAMA_NO_MTP_AUTO=1 ./build-vk/bin/llama-server -m ~/models/Qwen3.5-9B-mtp-q4km.gguf \
  -c 4096 -ngl 99 -np 1 -fit off --host 127.0.0.1 --port 9099 --no-warmup &
PID=$!; sleep 30
curl -s localhost:9099/completion -d '{"prompt":"Once upon a time","n_predict":64,"temperature":0,"seed":42,"return_tokens":true}' > /tmp/nospec.json
kill $PID

# Compare
jq -S '.tokens'  /tmp/spec.json   > /tmp/s.tok
jq -S '.tokens'  /tmp/nospec.json > /tmp/n.tok
diff /tmp/s.tok /tmp/n.tok && echo "BYTE-IDENTICAL ✓" || echo "DIVERGED ✗"
jq '.timings | {predicted_n, draft_n, draft_n_accepted, predicted_per_second}' /tmp/spec.json
jq '.timings | {predicted_n, predicted_per_second}' /tmp/nospec.json
```

**Pass criteria:** tokens identical, `draft_n_accepted > 0`, acceptance 70–90%.

**Performance note** (revised after measurement on this peer host, 2026-04-11): theory says 2-phase MTP should be faster than non-spec on GPU full-offload because the draft pass amortizes on the GPU. In practice on Vega 64 Vulkan with this build, it measures out as neutral-to-slightly-slower — e.g. 27.07 t/s spec vs 27.65 t/s non-spec at 77.78 % acceptance (a ~2 % slowdown). The most likely cause is the ~666 MiB of `CPU_Mapped` tensors (see §8): they pull enough of the forward pass off-GPU to blunt the amortization. **Don't treat "spec is faster" as a pass criterion** — treat token identity and acceptance rate as the hard pass criteria, and measure throughput separately. If the throughput ordering matters for your use case, profile per-pass breakdown before building more spec infrastructure around it.

## 10. Batching feasibility (Plan A go/no-go, optional)

```bash
./build-vk/bin/llama-batched-bench -m ~/models/Qwen3.5-9B-mtp-q4km.gguf \
  -c 4096 -b 2048 -ub 512 -npp 12 -ntg 64 -npl 1,2,4,8 -ngl 99 -t $(nproc)
```

`t(batch=4)/t(batch=1) < 2.0` → Plan A batched spec decode is worth building. `> 3.5` → don't bother.

## 11. Gotchas (read before you hit them)

- **NEXTN regression:** every upstream sync may reset `LLM_TENSOR_NEXTN_*` from `LAYER_REPEATING` back to `LAYER_OUTPUT` in `src/llama-arch.cpp`, silently breaking all Qwen3.5 MTP loads with an abort (`"input/output layer tensor blk.N.nextn.eh_proj.weight used with a layer number"`). Grep before every build. Fix: restore `REPEATING` classification on the six NEXTN entries (see commit `a4ed1af94`).
- **Stale gguf venv:** if `pip show gguf` doesn't point at this tree's `gguf-py`, the converter silently drops MTP tensors. Always install editable from the same tree you're building.
- **`hf download`, NOT `huggingface-cli download`:** in `huggingface_hub >= 1.0`, `huggingface-cli` is deprecated and just prints a help stub — the download appears to "succeed" (exit 0) but nothing lands on disk. The current command is `hf download <repo> --local-dir <path>`. Every tutorial and old doc still shows `huggingface-cli download`; ignore them. Verify a file actually landed before trusting the exit code.
- **`torch~=2.6.0` pin in `requirements-convert_hf_to_gguf.txt`:** breaks on Python 3.13+. Ignore the pin, install latest torch CPU wheel.
- **`LLAMA_NO_MTP_AUTO=1`:** env var to bypass the server's MTP auto-enable. Needed to get a clean non-spec baseline for equivalence testing.
- **`-fit off` is cleaner than auto-fit.** Auto-fit can stall in a resize loop. Set `-ngl` explicitly, use `-fit off`.
- **Quant floor for tool calling is Q4_K_M.** Q3 and below break JSON schema compliance intermittently. Don't go lower.
- **Always use GBNF grammar-constrained output for tool calls.** Free correctness floor — the model literally cannot emit invalid JSON. Pass via the `"grammar"` field in `/completion` requests.
- **`llama-completion` vs `llama-server` disagree on `n_predict` accounting** by 1–2 tokens. Never use `llama-completion` as the non-spec baseline. Always compare server-vs-server via `LLAMA_NO_MTP_AUTO=1`.
- **2-phase MTP is neutral-to-slower on CPU, faster on GPU.** On Vega 64 full offload it *should* win in theory; measured on this peer host it came out ~2 % slower even at 77 % acceptance — likely because the `CPU_Mapped` embed/output tensors (§8) blunt the amortization. Don't assume without measuring. See the §9 performance note.
- **`batched-bench` and `llama-server` report wildly different throughput at the same `B=1`.** Measured on Vega 64 Vulkan with this model: `llama-server -np 1` hits ~27 t/s for single-token gen, while `llama-batched-bench -c 4096 -npl 1,2,4,8` reports ~5 t/s at `B=1`. Same model, same GPU, same `-c`. The batched-bench run shows `graph splits = 118` and reserves the recurrent-state memory for the full max-parallel (8 seqs) even when measuring `B=1`, which trips a much slower scheduling path. **Use `llama-server` throughput as the absolute baseline; `batched-bench` is only trustworthy for *scaling ratios* like `T(batch=4)/T(batch=1)`.** Don't report the batched-bench B=1 number as "the model's throughput."

## 12. What you're here to measure

The mission is **tool-calling accuracy, not throughput.** After the equivalence check passes, write a small harness:

- 5–10 tool schemas (weather, search, file ops, etc.)
- 5–10 positive queries per schema (should call the tool) + 2–3 negatives (shouldn't)
- Hit `/completion` with a GBNF grammar for the tool-call JSON
- Score: function selection accuracy, required-arg presence, arg type correctness, correct abstention on negatives

Run it against `Qwen3.5-9B-mtp-q4km.gguf` and a couple of alternates (Qwen3-14B-Q6_K, Hermes 3 8B Q6_K, Functionary Small). Accuracy ranking drives any further hardware/model decision.

**Do not build more spec-decode infrastructure until the accuracy ranking is done.** That's the lesson the previous host paid for.
