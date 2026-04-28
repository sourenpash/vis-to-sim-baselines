"""Cross-adapter evaluation: read every ``outputs/<adapter>/`` and produce metrics.

Writes ``<session>/results.csv`` (long-format), ``<session>/results.md`` (a
comparison table), and a small set of side-by-side comparison PNGs under
``<session>/outputs/_compare/``.

This script does not import any model code - it only consumes the
universal ``depth/*.npy`` + ``timings.csv`` outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml
from tabulate import tabulate

from ..io.session import Manifest, SessionReader
from ..metrics.completeness import completeness_metrics
from ..metrics.nearfield import ground_plane_metrics
from ..metrics.runtime import runtime_metrics
from ..metrics.spatial import aggregate_plane_fit
from ..metrics.temporal import temporal_metrics
from ..viz.compare import write_compare_grid


def _list_adapter_dirs(session_root: Path) -> List[Path]:
    out = session_root / "outputs"
    if not out.exists():
        return []
    return sorted(p for p in out.iterdir() if p.is_dir() and not p.name.startswith("_"))


def _read_meta(adapter_dir: Path) -> Dict[str, Any]:
    p = adapter_dir / "meta.json"
    if not p.exists():
        return {}
    with open(p, "r") as f:
        return json.load(f)


def _depth_iter(
    adapter_dir: Path,
    start: int = 0,
    end: Optional[int] = None,
) -> Iterable[np.ndarray]:
    depth_dir = adapter_dir / "depth"
    if not depth_dir.exists():
        return iter(())
    paths = sorted(depth_dir.glob("*.npy"))
    if end is not None:
        paths = paths[start:end]
    else:
        paths = paths[start:]

    def _gen():
        for p in paths:
            arr = np.load(p)
            if arr.ndim == 3:
                arr = arr.squeeze()
            yield arr.astype(np.float32, copy=False)

    return _gen()


def _load_depth_list(
    adapter_dir: Path,
    start: int = 0,
    end: Optional[int] = None,
) -> List[np.ndarray]:
    return list(_depth_iter(adapter_dir, start, end))


def _resolve_K(manifest: Manifest, target_frame: str) -> np.ndarray:
    """Returns the 3x3 K of the stream the depth is aligned to."""
    return manifest.streams[target_frame].K()


def _resolve_dims(manifest: Manifest, target_frame: str) -> tuple:
    s = manifest.streams[target_frame]
    return s.width, s.height


def _load_eval_cfg(cfg_path: Optional[Path]) -> Dict[str, Any]:
    if cfg_path is None or not cfg_path.exists():
        return {}
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("metrics", {}) or {}


def evaluate_session(session_path: Path, cfg_path: Optional[Path]) -> int:
    session = SessionReader(session_path)
    manifest = session.manifest
    eval_cfg = _load_eval_cfg(cfg_path)

    near = eval_cfg.get("near_band", {}) or {}
    near_min = float(near.get("min_m", 0.2))
    near_max = float(near.get("max_m", 3.0))
    flat_roi = eval_cfg.get("flat_plane_roi")
    ground_roi = eval_cfg.get("ground_plane_roi")
    static = eval_cfg.get("static_segment", {}) or {}
    static_start = int(static.get("start_frame", 0))
    static_end = static.get("end_frame")
    if static_end is not None:
        static_end = int(static_end)

    adapter_dirs = _list_adapter_dirs(session_path)
    if not adapter_dirs:
        print(f"[evaluate] no outputs/<adapter>/ found under {session_path}", file=sys.stderr)
        return 2

    print(f"[evaluate] session: {session_path}")
    print(f"[evaluate] adapters: {[p.name for p in adapter_dirs]}")

    long_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for adir in adapter_dirs:
        name = adir.name
        meta = _read_meta(adir)
        target_frame = meta.get("target_frame", "color")
        if target_frame not in manifest.streams:
            print(f"  [warn] adapter '{name}' target_frame='{target_frame}' "
                  f"not in manifest.streams; falling back to 'color'")
            target_frame = "color"
        K = _resolve_K(manifest, target_frame)
        W, H = _resolve_dims(manifest, target_frame)

        timings = runtime_metrics(adir / "timings.csv")
        comp = completeness_metrics(
            _depth_iter(adir),
            near_min_m=near_min,
            near_max_m=near_max,
        )

        depths_static = _load_depth_list(adir, start=static_start, end=static_end)
        temp = temporal_metrics(
            depths_static, near_min_m=near_min, near_max_m=near_max,
        )

        plane = aggregate_plane_fit(
            depths_static if depths_static else _load_depth_list(adir),
            K, flat_roi,
        )
        ground = ground_plane_metrics(
            depths_static if depths_static else _load_depth_list(adir),
            K, W, H, roi=ground_roi,
        )

        all_metrics: Dict[str, Any] = {
            **timings,
            **comp,
            **temp,
            **{f"flat_{k}": v for k, v in plane.items()},
            **ground,
        }

        for metric, value in all_metrics.items():
            if isinstance(value, (int, float, np.floating, np.integer)):
                long_rows.append({"adapter": name, "metric": metric, "value": float(value)})

        summary_rows.append({
            "adapter": name,
            "target_frame": target_frame,
            "n_frames": timings.get("n_frames", 0),
            "infer_p50_ms": timings.get("infer_ms_p50", float("nan")),
            "fps": timings.get("fps_from_total", float("nan")),
            "jitter_ms": timings.get("jitter_ms_std", float("nan")),
            "gpu_peak_mb": timings.get("gpu_mem_mb_peak", 0.0),
            "valid_pct": comp.get("valid_pct", float("nan")),
            "near_band_valid_pct": comp.get("near_band_valid_pct", float("nan")),
            "temporal_std_mm": temp.get("temporal_std_mm_mean", float("nan")),
            "frame_diff_mm": temp.get("frame_to_frame_diff_mm_mean", float("nan")),
            "flat_rmse_mm": plane.get("plane_rmse_mm_mean", float("nan")),
            "ground_rmse_mm": ground.get("ground_plane_rmse_mm_mean", float("nan")),
            "ground_normal_std_deg": ground.get("ground_plane_normal_std_deg", float("nan")),
        })

    summary = pd.DataFrame(summary_rows)
    long = pd.DataFrame(long_rows)

    csv_path = session_path / "results.csv"
    long.to_csv(csv_path, index=False)
    print(f"[evaluate] wrote {csv_path}")

    summary_path = session_path / "results_summary.csv"
    summary.to_csv(summary_path, index=False)

    md_table = tabulate(
        summary, headers="keys", tablefmt="github", showindex=False, floatfmt=".2f",
    )
    md_path = session_path / "results.md"
    title = f"# Depth Benchmark - {session_path.name}\n\n"
    notes_block = ""
    if manifest.notes:
        notes_block = f"_Notes: {manifest.notes}_\n\n"
    md_path.write_text(title + notes_block + md_table + "\n")
    print(f"[evaluate] wrote {md_path}")

    compare_dir = session_path / "outputs" / "_compare"
    sample_indices = []
    if len(session) > 0:
        n = len(session)
        sample_indices = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    for idx in sample_indices:
        write_compare_grid(
            session_path, idx,
            [p.name for p in adapter_dirs],
            compare_dir,
        )
    if sample_indices:
        print(f"[evaluate] wrote {len(sample_indices)} compare PNGs to {compare_dir}")

    print()
    print(md_table)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="benchmark.runners.evaluate",
        description="Aggregate metrics across adapters for one session.",
    )
    parser.add_argument("--session", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg_path = args.config
    if cfg_path is None:
        default = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
        if default.exists():
            cfg_path = default
    return evaluate_session(args.session, cfg_path)


if __name__ == "__main__":
    sys.exit(main())
