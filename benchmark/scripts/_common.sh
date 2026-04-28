#!/usr/bin/env bash
# Source-only helper: sets BENCH_ROOT, REPO_ROOT, and common env names.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$BENCH_ROOT/.." && pwd)"

BENCH_ENV="${BENCH_ENV:-bench}"
FFS_ENV="${FFS_ENV:-ffs}"
S2M2_ENV="${S2M2_ENV:-s2m2}"
LINGBOT_ENV="${LINGBOT_ENV:-lingbot-depth}"

# All adapters in the standard ordering (env name, adapter name).
ADAPTERS=(
  "$BENCH_ENV:realsense_hw"
  "$FFS_ENV:ffs"
  "$S2M2_ENV:s2m2"
  "$LINGBOT_ENV:lingbot"
)

source "$(conda info --base)/etc/profile.d/conda.sh"

run_in_env () {
  local env_name="$1"; shift
  conda activate "$env_name"
  ( cd "$BENCH_ROOT" && "$@" )
  local rc=$?
  conda deactivate
  return $rc
}
