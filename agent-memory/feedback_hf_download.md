---
name: Use `hf download`, not `huggingface-cli download`
description: Persistent model habit to unlearn — the HuggingFace CLI entry point is now `hf`, the old one is a deprecated help stub that looks like it worked.
type: feedback
originSessionId: a5a02c13-3d1d-4998-badf-36c74b87c680
---
When fetching models from HuggingFace, always use `hf download`, never `huggingface-cli download`.

**Why:** In `huggingface_hub >= 1.0`, `huggingface-cli` is deprecated and only prints a help stub — it exits 0 with no error, but downloads nothing. This silently wastes time and masquerades as success until someone notices the target directory doesn't exist. The user has flagged this as a persistent mistake my training data biases me toward. Old tutorials and docs (including upstream llama.cpp quickstarts) still show `huggingface-cli download` because the rename is recent.

**How to apply:** Any time the task involves downloading from HuggingFace — models, datasets, anything — use `hf download <repo> --local-dir <path>`. Same rule for other hf subcommands: `hf auth login`, `hf upload`, etc. Don't just trust that the exit code is 0; verify the target files actually landed. When reading or writing quickstart docs that other people will run, explicitly use `hf` rather than copying forward the old command.
