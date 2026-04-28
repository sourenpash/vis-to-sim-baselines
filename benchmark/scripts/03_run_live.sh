#!/usr/bin/env bash
# Live-mode FPS / latency for every enabled adapter (each owns the D435i in
# its own env, sequentially - the camera is single-tenant).
#
# Usage:
#   bash scripts/03_run_live.sh sessions/<live_root> [--max-frames N] [--adapters a,b]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <session_dir> [--max-frames N] [--adapters a,b]" >&2
  exit 2
fi

SESSION_DIR="$1"
mkdir -p "$SESSION_DIR"
SESSION_DIR="$(cd "$SESSION_DIR" && pwd)"
shift || true

ONLY=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --adapters) ONLY="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

# Always limit live runs unless caller already passed --max-frames.
HAS_MAX=0
for a in "${EXTRA[@]:-}"; do
  case "$a" in --max-frames|--max-frames=*) HAS_MAX=1 ;; esac
done
if [[ "$HAS_MAX" -eq 0 ]]; then EXTRA+=(--max-frames 300); fi

run_live () {
  local env_name="$1"
  local adapter="$2"
  echo
  echo "===================================================================="
  echo " LIVE: adapter '$adapter' in env '$env_name'"
  echo "===================================================================="
  if ! conda env list | awk '{print $1}' | grep -qx "$env_name"; then
    echo "  [skip] env '$env_name' missing"
    return 0
  fi
  conda run --no-capture-output -n "$env_name" \
    python -m benchmark.runners.run_model \
      --adapter "$adapter" --session "$SESSION_DIR" --live "${EXTRA[@]}" \
    || echo "  [warn] adapter '$adapter' failed (rc=$?); continuing"
}

cd "$BENCH_ROOT"

for entry in "${ADAPTERS[@]}"; do
  env_name="${entry%%:*}"
  adapter="${entry##*:}"
  if [[ -n "$ONLY" ]] && [[ ",${ONLY}," != *",${adapter},"* ]]; then
    continue
  fi
  run_live "$env_name" "$adapter"
done

echo
echo "[live] all adapters finished. Inspect outputs/<adapter>/timings.csv."
