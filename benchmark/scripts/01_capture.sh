#!/usr/bin/env bash
# Record a D435i session in the `bench` conda env.
#
# Usage:
#   bash scripts/01_capture.sh sessions/<name> [--duration 30] [--emitter off|on]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <session_dir> [extra args forwarded to capture.py]" >&2
  exit 2
fi

SESSION_DIR="$1"
shift || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bench

cd "$BENCH_ROOT"
python -m benchmark.runners.capture --out "$SESSION_DIR" "$@"
