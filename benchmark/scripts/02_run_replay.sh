#!/usr/bin/env bash
# Replay a recorded session through every enabled adapter, each in its own env.
#
# Usage:
#   bash scripts/02_run_replay.sh sessions/<name> [--max-frames N] [--adapters ffs,s2m2,...]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <session_dir> [--max-frames N] [--adapters a,b,c]" >&2
  exit 2
fi

SESSION_DIR="$(cd "$1" && pwd)"
shift || true

ONLY=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --adapters) ONLY="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

run_adapter () {
  local env_name="$1"
  local adapter="$2"
  echo
  echo "===================================================================="
  echo " Running adapter '$adapter' in env '$env_name'"
  echo "===================================================================="
  if ! conda env list | awk '{print $1}' | grep -qx "$env_name"; then
    echo "  [skip] env '$env_name' missing"
    return 0
  fi
  conda run --no-capture-output -n "$env_name" \
    python -m benchmark.runners.run_model \
      --adapter "$adapter" --session "$SESSION_DIR" "${EXTRA[@]}" \
    || echo "  [warn] adapter '$adapter' failed (rc=$?); continuing"
}

cd "$BENCH_ROOT"

for entry in "${ADAPTERS[@]}"; do
  env_name="${entry%%:*}"
  adapter="${entry##*:}"
  if [[ -n "$ONLY" ]] && [[ ",${ONLY}," != *",${adapter},"* ]]; then
    continue
  fi
  run_adapter "$env_name" "$adapter"
done

echo
echo "[replay] all adapters finished. Run scripts/04_evaluate.sh next."
