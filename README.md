# yarn-agentic

A solo developer plus agentic AI assistants building a byte-deterministic, multi-GPU LLM inference server on a pair of Quadro RTX 6000s (sm_75 / TU102), with the full work record published as the [project ledger](https://blog.david.connol.ly/).

Start at [`docs/home.md`](docs/home.md). Active phases are under [`docs/active/`](docs/active/); closed phases live under [`docs/archive/phases/<topic>/`](docs/archive/phases/). The chronological journal is `MEMORY.md` (append-only, 11 028+ lines).

Documentation is rendered from `docs/` via [mdBook](https://rust-lang.github.io/mdBook/) and auto-published to GitHub Pages on push to `main`. Formal-spec gates (`45 .allium` + `39 .tla` files under `specs/`) run via `.github/workflows/spec-tla-gate.yml`.
