#!/usr/bin/env bash
# Compute metrics across all adapters that produced outputs in <session>/outputs/
# and write results.md + results.csv + comparison images.
#
# Usage:
#   bash scripts/04_evaluate.sh sessions/<name>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <session_dir>" >&2
  exit 2
fi

SESSION_DIR="$(cd "$1" && pwd)"
shift || true

cd "$BENCH_ROOT"
conda run --no-capture-output -n "$BENCH_ENV" \
  python -m benchmark.runners.evaluate --session "$SESSION_DIR" "$@"
