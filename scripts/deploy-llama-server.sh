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

log() { printf '%s %s\n' "[$(date -u '+%H:%M:%S')]" "$*"; }

# ---------------------------------------------------------------------------
# Pre-flight: the build must be complete and fresh.
# ---------------------------------------------------------------------------
if [[ ! -x "$BUILD/bin/llama-server" ]]; then
    log "ABORT: $BUILD/bin/llama-server does not exist"
    log "  Rebuild first: cmake --build $BUILD -j --target llama-server"
    exit 1
fi
for so in "$BUILD/ggml/src/libggml.so" "$BUILD/src/libllama.so" "$BUILD/examples/mtmd/libmtmd.so"; do
    if [[ ! -f "$so" ]]; then
        log "ABORT: $so is missing — libs must be rebuilt alongside the binary"
        exit 1
    fi
done

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

# ---------------------------------------------------------------------------
# Post-install verification — refuse to restart if anything is wrong.
# ---------------------------------------------------------------------------
log "verifying installed artifacts"
for name in libggml.so libllama.so libmtmd.so; do
    case "$name" in
        libggml.so)  src="$BUILD/ggml/src/$name" ;;
        libllama.so) src="$BUILD/src/$name" ;;
        libmtmd.so)  src="$BUILD/examples/mtmd/$name" ;;
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
if strings "$PREFIX/lib/libggml.so" | grep -q 'i_split < GGML_SCHED_MAX_SPLITS'; then
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
