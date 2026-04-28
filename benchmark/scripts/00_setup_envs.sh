#!/usr/bin/env bash
# Create the `bench` env (capture + evaluation) and install pyrealsense2
# into the existing model envs so each can also acquire the camera in --live
# mode.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

ensure_bench_env () {
  if conda env list | awk '{print $1}' | grep -qx "$BENCH_ENV"; then
    echo "[setup] env '$BENCH_ENV' already exists"
  else
    echo "[setup] creating env '$BENCH_ENV' from $BENCH_ROOT/env.yml"
    conda env create -n "$BENCH_ENV" -f "$BENCH_ROOT/env.yml"
  fi
}

install_pyrealsense_in () {
  local env_name="$1"
  if ! conda env list | awk '{print $1}' | grep -qx "$env_name"; then
    echo "[setup] env '$env_name' not found, skipping pyrealsense2 install"
    return
  fi
  echo "[setup] ensuring pyrealsense2 in '$env_name'"
  conda run -n "$env_name" python -c "import pyrealsense2" >/dev/null 2>&1 \
    || conda run -n "$env_name" python -m pip install pyrealsense2
}

install_bench_pkg_in () {
  local env_name="$1"
  if ! conda env list | awk '{print $1}' | grep -qx "$env_name"; then return; fi
  echo "[setup] installing benchmark package (editable) in '$env_name'"
  conda run -n "$env_name" python -m pip install -e "$BENCH_ROOT" >/dev/null
}

ensure_bench_env

for entry in "${ADAPTERS[@]}"; do
  env_name="${entry%%:*}"
  install_pyrealsense_in "$env_name"
  install_bench_pkg_in "$env_name"
done

echo "[setup] done."
