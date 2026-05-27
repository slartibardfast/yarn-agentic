#!/usr/bin/env bash
# scripts/deploy-llama-server.sh
#
# Productionized deploy for ik_llama.cpp llama-server.
#
# Why this exists: an earlier "deploy" copied the binary alone and forgot
# the .so files. /opt/llm-server/lib/libggml.so stayed stale, the service
# loaded the unpatched libggml at runtime, and the production wedge from
# 2026-05-25 07:52 UTC repeated twice (10:27, 10:37) after the supposed
# fix was "deployed". See MEMORY.md 2026-05-25 + PHASE_GGML_SCHED_DYNSPLITS.md.
#
# This script is the only sanctioned way to push a new llama-server build
# to /opt/llm-server. Do NOT sudo install the binary manually.
#
# Defaults (overridable via env):
#   BUILD=/home/dconnolly/yarn-agentic/ik_llama.cpp/build
#   PREFIX=/opt/llm-server
#   SERVICE=llama-server.service

set -euo pipefail

BUILD=${BUILD:-/home/dconnolly/yarn-agentic/ik_llama.cpp/build}
PREFIX=${PREFIX:-/opt/llm-server}
SERVICE=${SERVICE:-llama-server.service}

ALLOW_NO_MMPROJ_MGPU=0
for arg in "$@"; do
    case "$arg" in
        --allow-no-mmproj-mgpu) ALLOW_NO_MMPROJ_MGPU=1 ;;
        *) printf 'unknown flag: %s\n' "$arg" >&2; exit 2 ;;
    esac
done

log() { printf '%s %s\n' "[$(date -u '+%H:%M:%S')]" "$*"; }

# ---------------------------------------------------------------------------
# Pre-flight: the build must be complete and fresh.
# ---------------------------------------------------------------------------
if [[ ! -x "$BUILD/bin/llama-server" ]]; then
    log "ABORT: $BUILD/bin/llama-server does not exist"
    log "  Rebuild first: cmake --build $BUILD -j --target llama-server"
    exit 1
fi
for so in "$BUILD/ggml/src/libggml.so" "$BUILD/src/libllama.so" \
          "$BUILD/examples/mtmd/libmtmd.so"; do
    if [[ ! -f "$so" ]]; then
        log "ABORT: $so is missing — libs must be rebuilt alongside the binary"
        exit 1
    fi
done
# libmgpu.so is Phase 46+; required ONLY when not in --allow-no-mmproj-mgpu
# rollback mode (a pre-Phase-46 build legitimately won't have it).
if [[ "$ALLOW_NO_MMPROJ_MGPU" != "1" ]]; then
    if [[ ! -f "$BUILD/mgpu/libmgpu.so" ]]; then
        log "ABORT: $BUILD/mgpu/libmgpu.so is missing — libs must be rebuilt alongside the binary"
        log "  Pass --allow-no-mmproj-mgpu if intentionally deploying a pre-Phase-46 build."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# PHASE 46 §11.4 — multi-GPU CLIP regression guard.
#
# Two byte-level checks on the build tree before install:
#   1. The binary must contain the "multi-backend init" LOG_INF string emitted
#      by clip.cpp's Path-B multi-backend parser. A pre-Phase-46 binary lacks
#      this string entirely.
#   2. libggml.so must export ggml_mgpu_create_split — the shared row-chunk
#      split entry point introduced by submodule commit f2704241. A
#      pre-Phase-46 libggml.so has no such symbol.
#
# Either failure aborts deploy. Pass --allow-no-mmproj-mgpu for the emergency
# rollback path (deploying a pre-Phase-46 binary on purpose).
# ---------------------------------------------------------------------------
if [[ "$ALLOW_NO_MMPROJ_MGPU" == "1" ]]; then
    log "WARN: --allow-no-mmproj-mgpu set; skipping Phase 46 multi-GPU CLIP guards"
else
    # NOTE: `strings|grep -q` and `nm|grep -q` send SIGPIPE to the producer
    # when grep exits early on first match; with `set -o pipefail` the
    # pipeline then returns non-zero EVEN THOUGH THE MATCH WAS FOUND. Use
    # `grep -c ... >/dev/null` so grep consumes the full stream and the
    # exit code reflects whether the match was zero or non-zero. Same
    # trap verify-multigpu-clip.sh hit (see its lines 60-64).
    if [[ "$(strings "$BUILD/examples/mtmd/libmtmd.so" \
             | grep -c 'multi-backend init' || true)" == "0" ]]; then
        log "ABORT: $BUILD/examples/mtmd/libmtmd.so missing 'multi-backend init' string"
        log "  Phase 46 multi-backend CLIP path not present — refusing to deploy"
        log "  Pass --allow-no-mmproj-mgpu to override for rollback."
        exit 1
    fi
    if [[ "$(nm -D --defined-only "$BUILD/ggml/src/libggml.so" 2>/dev/null \
             | grep -c 'ggml_mgpu_create_split' || true)" == "0" ]]; then
        log "ABORT: $BUILD/ggml/src/libggml.so missing ggml_mgpu_create_split export"
        log "  Phase 46 shared mgpu_split_config infra not present — refusing to deploy"
        log "  Pass --allow-no-mmproj-mgpu to override for rollback."
        exit 1
    fi
    log "OK: Phase 46 multi-GPU CLIP guards clean"
fi

# Warn (do not block) on dirty submodule — caller may be testing local changes.
SUBMOD_DIR="$(cd "$BUILD"/.. && pwd)"
if [[ -n "$(git -C "$SUBMOD_DIR" status --porcelain 2>/dev/null)" ]]; then
    log "WARN: $SUBMOD_DIR working tree has uncommitted changes"
    git -C "$SUBMOD_DIR" status --short | sed 's/^/    /' | head -5
fi

# ---------------------------------------------------------------------------
# Atomic install: binary + ALL libs together, ownership root:llm.
# ---------------------------------------------------------------------------
log "installing to $PREFIX (sudo required)"
sudo install -m 0755 -o root -g llm "$BUILD/bin/llama-server" "$PREFIX/bin/llama-server"
sudo install -m 0755 -o root -g llm \
    "$BUILD/ggml/src/libggml.so" \
    "$BUILD/src/libllama.so" \
    "$BUILD/examples/mtmd/libmtmd.so" \
    "$PREFIX/lib/"
# libmgpu.so is Phase 46+; only install when present (rollback builds lack it)
if [[ -f "$BUILD/mgpu/libmgpu.so" ]]; then
    sudo install -m 0755 -o root -g llm \
        "$BUILD/mgpu/libmgpu.so" \
        "$PREFIX/lib/"
else
    # Best-effort: remove any stale libmgpu.so so the binary doesn't link
    # against a leftover from a prior forward deploy. Rollback binaries do
    # not reference libmgpu.so so deletion is safe.
    sudo rm -f "$PREFIX/lib/libmgpu.so"
fi

# ---------------------------------------------------------------------------
# Post-install verification — refuse to restart if anything is wrong.
# ---------------------------------------------------------------------------
log "verifying installed artifacts"
# Base libs always verified; libmgpu.so verified only when the build has it.
VERIFY_NAMES=(libggml.so libllama.so libmtmd.so)
if [[ -f "$BUILD/mgpu/libmgpu.so" ]]; then
    VERIFY_NAMES+=(libmgpu.so)
fi
for name in "${VERIFY_NAMES[@]}"; do
    case "$name" in
        libggml.so)  src="$BUILD/ggml/src/$name" ;;
        libllama.so) src="$BUILD/src/$name" ;;
        libmtmd.so)  src="$BUILD/examples/mtmd/$name" ;;
        libmgpu.so)  src="$BUILD/mgpu/$name" ;;
    esac
    bh=$(sha256sum "$src" | cut -d' ' -f1)
    ih=$(sha256sum "$PREFIX/lib/$name" | cut -d' ' -f1)
    if [[ "$bh" != "$ih" ]]; then
        log "ABORT: $name hash mismatch (build=$bh, install=$ih)"
        exit 1
    fi
done
bh=$(sha256sum "$BUILD/bin/llama-server" | cut -d' ' -f1)
ih=$(sha256sum "$PREFIX/bin/llama-server" | cut -d' ' -f1)
if [[ "$bh" != "$ih" ]]; then
    log "ABORT: llama-server hash mismatch (build=$bh, install=$ih)"
    exit 1
fi

# Regression guard: the GGML_SCHED_MAX_SPLITS legacy assert must be absent
# from the installed libggml.so. This is the specific bug from 2026-05-25;
# its presence in an installed lib means the patch from ik_llama.cpp commit
# 252217d8 (or its replacement) did not land in this build.
if [[ "$(strings "$PREFIX/lib/libggml.so" \
         | grep -c 'i_split < GGML_SCHED_MAX_SPLITS' || true)" != "0" ]]; then
    log "ABORT: legacy GGML_SCHED_MAX_SPLITS assert string is present in $PREFIX/lib/libggml.so"
    log "  the patch from commit 252217d8 has not landed in this build — refusing to deploy"
    exit 1
fi
log "OK: hashes match, regression guard clean"

# ---------------------------------------------------------------------------
# Ensure systemd drop-in sets LD_LIBRARY_PATH=/opt/llm-server/lib so the
# loader prefers the installed libs over the binary's build-tree RUNPATH.
#
# Without this drop-in, the installed binary at $PREFIX/bin/llama-server
# resolves libggml/libllama/libmtmd from its embedded RUNPATH — which points
# at the build tree (e.g. /home/dconnolly/yarn-agentic/ik_llama.cpp/build/
# ggml/src/libggml.so) because CMake stores BUILD_RPATH in the linked binary
# and we install via `install -m 0755` which does not rewrite the ELF.
# Setting LD_LIBRARY_PATH makes the loader check $PREFIX/lib first; that's
# where the deploy step above just placed the fresh patched libs.
#
# A cleaner alternative (binary self-contained via patchelf --set-rpath
# '$ORIGIN/../lib') is available if patchelf is installed; this script
# prefers the systemd drop-in because it works on any host without an
# extra build dependency.
# ---------------------------------------------------------------------------
DROPIN_DIR=/etc/systemd/system/"$SERVICE".d
DROPIN_FILE="$DROPIN_DIR/00-lib-path.conf"
log "ensuring systemd drop-in at $DROPIN_FILE"
sudo install -d -m 0755 "$DROPIN_DIR"
if [[ ! -f "$DROPIN_FILE" ]] \
    || ! grep -q "LD_LIBRARY_PATH=$PREFIX/lib" "$DROPIN_FILE" 2>/dev/null; then
    sudo tee "$DROPIN_FILE" >/dev/null <<EOF
# Force the loader to prefer the installed libs over the binary's build-tree
# RUNPATH. Managed by scripts/deploy-llama-server.sh — do not hand-edit.
[Service]
Environment=LD_LIBRARY_PATH=$PREFIX/lib
EOF
    log "wrote $DROPIN_FILE"
    sudo systemctl daemon-reload
else
    log "drop-in already in place"
fi

# ---------------------------------------------------------------------------
# Restart and wait for /health.
# ---------------------------------------------------------------------------
log "restarting $SERVICE"
sudo systemctl restart "$SERVICE"

log "waiting up to 120s for /health = 200"
for i in $(seq 1 60); do
    sleep 2
    code=$(curl -sS -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/health 2>/dev/null || echo 000)
    if [[ "$code" == "200" ]]; then
        log "OK: /health = 200 after $((i*2))s"
        sleep 1
        log "running build stamp:"
        journalctl -u "$SERVICE" --no-pager --since '1 min ago' 2>&1 \
            | grep -oE 'build=[0-9]+ commit="[a-f0-9]+"' | tail -1 | sed 's/^/    /'
        log "service status:"
        systemctl status "$SERVICE" --no-pager -n 0 | head -8 | sed 's/^/    /'
        exit 0
    fi
done

log "ABORT: /health did not return 200 within 120s"
systemctl status "$SERVICE" --no-pager -n 0 | head -10
journalctl -u "$SERVICE" --no-pager -n 20
exit 1
