#!/usr/bin/env python3
"""Repeat GPU benchmark variants and summarize run-to-run variability."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Iterable

from benchmark_sweep import DEFAULT_VARIANTS, Variant, run_variant


MAIN_VARIANT_NAMES = (
    "gpu_split_baseline",
    "gpu_all_fused",
    "gpu_graph_all_fused",
)

POISSON_VARIANT_NAMES = (
    "gpu_split_baseline",
    "gpu_all_fused",
    "gpu_graph_all_fused",
    "gpu_poisson_onepass",
    "gpu_graph_onepass",
)

SUMMARY_METRICS = (
    "kernel_total_ms_per_step",
    "wall_avg_ms_per_step",
    "kernel_mlups",
    "poisson_ms_per_step",
    "poisson_iters",
    "speedup_vs_gpu_split",
)


def finite_float(value: object) -> float:
    if value == "":
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def is_finite(value: float) -> bool:
    return not math.isnan(value) and not math.isinf(value)


def select_variants(label: str, include_cpu: bool) -> list[Variant]:
    variants_by_name = {variant.name: variant for variant in DEFAULT_VARIANTS}
    if label == "main":
        variants = [variants_by_name[name] for name in MAIN_VARIANT_NAMES]
    elif label == "poisson":
        variants = [variants_by_name[name] for name in POISSON_VARIANT_NAMES]
    elif label == "gpu":
        variants = [variant for variant in DEFAULT_VARIANTS if variant.mode == "gpu"]
    elif label == "all":
        variants = list(DEFAULT_VARIANTS)
    else:
        raise ValueError(f"unknown variant set: {label}")

    if include_cpu and variants_by_name["cpu_reference"] not in variants:
        variants = [variants_by_name["cpu_reference"], *variants]
    if not include_cpu:
        variants = [variant for variant in variants if variant.mode != "cpu"]
    return variants


def metric_values(rows: Iterable[dict[str, object]], metric: str) -> list[float]:
    values = [finite_float(row.get(metric, "")) for row in rows]
    return [value for value in values if is_finite(value)]


def add_stats(out: dict[str, object], prefix: str, values: list[float]) -> None:
    if not values:
        out[f"{prefix}_mean"] = ""
        out[f"{prefix}_std"] = ""
        out[f"{prefix}_min"] = ""
        out[f"{prefix}_max"] = ""
        return

    out[f"{prefix}_mean"] = statistics.fmean(values)
    out[f"{prefix}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    out[f"{prefix}_min"] = min(values)
    out[f"{prefix}_max"] = max(values)


def write_csv(path: Path, rows: list[dict[str, object]], keys: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    variants = list(dict.fromkeys(str(row["variant"]) for row in rows))
    out_rows: list[dict[str, object]] = []
    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant]
        ok_rows = [row for row in variant_rows if int(row.get("exit_code", 1)) == 0]
        out: dict[str, object] = {
            "variant": variant,
            "n": len(ok_rows),
            "total_runs": len(variant_rows),
            "exit_code_failures": len(variant_rows) - len(ok_rows),
            "mode": variant_rows[0].get("mode", ""),
            "steps": variant_rows[0].get("steps", ""),
            "poisson": variant_rows[0].get("poisson", ""),
            "pressure_boundary": variant_rows[0].get("pressure_boundary", ""),
            "cuda_graph": variant_rows[0].get("cuda_graph", ""),
        }
        for metric in SUMMARY_METRICS:
            add_stats(out, metric, metric_values(ok_rows, metric))
        out_rows.append(out)
    return out_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=Path("./lbm_gpu"))
    parser.add_argument("--params", type=Path, default=Path("../CPU/params_baseline_8000_stride1.in"))
    parser.add_argument("--gpu-steps", type=int, default=50)
    parser.add_argument("--cpu-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--variants", choices=("main", "poisson", "gpu", "all"), default="main")
    parser.add_argument("--include-cpu", action="store_true")
    parser.add_argument("--raw-csv", type=Path, default=Path("benchmark_repeat_raw.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("benchmark_repeat_summary.csv"))
    args = parser.parse_args()

    variants = select_variants(args.variants, args.include_cpu)
    rows: list[dict[str, object]] = []
    any_failure = False

    for repeat in range(1, args.repeats + 1):
        repeat_rows: list[dict[str, object]] = []
        print(f"repeat {repeat}/{args.repeats}")
        for variant in variants:
            print(f"  running {variant.name} ...", flush=True)
            row = run_variant(
                args.exe,
                args.params,
                variant,
                args.gpu_steps,
                args.cpu_steps,
                args.warmup_steps,
                Path("."),
            )
            row["repeat"] = repeat
            repeat_rows.append(row)
            any_failure = any_failure or int(row.get("exit_code", 1)) != 0
            print(
                f"  {variant.name}: exit={row.get('exit_code')} "
                f"kernel_ms={row.get('kernel_total_ms_per_step', 'NA')} "
                f"mlups={row.get('kernel_mlups', row.get('wall_mlups', 'NA'))}",
                flush=True,
            )

        split_ms = next(
            (
                finite_float(row.get("kernel_total_ms_per_step", ""))
                for row in repeat_rows
                if row["variant"] == "gpu_split_baseline"
            ),
            float("nan"),
        )
        for row in repeat_rows:
            ms = finite_float(row.get("kernel_total_ms_per_step", row.get("wall_avg_ms_per_step", "")))
            row["speedup_vs_gpu_split"] = split_ms / ms if is_finite(split_ms) and ms > 0.0 else ""
        rows.extend(repeat_rows)

    raw_keys = [
        "repeat", "variant", "mode", "steps", "poisson", "pressure_boundary",
        "cuda_graph", "exit_code", "wall_total_ms", "wall_avg_ms_per_step",
        "wall_mlups", "collision_ms_per_step", "stream_ms_per_step",
        "macro_ms_per_step", "poisson_ms_per_step", "poisson_iters",
        "kernel_total_ms_per_step", "kernel_mlups", "collision_percent",
        "stream_percent", "macro_percent", "poisson_percent",
        "speedup_vs_gpu_split", "command",
    ]
    summary_keys = [
        "variant", "n", "total_runs", "exit_code_failures", "mode", "steps",
        "poisson", "pressure_boundary", "cuda_graph",
    ]
    for metric in SUMMARY_METRICS:
        summary_keys.extend([
            f"{metric}_mean",
            f"{metric}_std",
            f"{metric}_min",
            f"{metric}_max",
        ])

    summary = summarize(rows)
    write_csv(args.raw_csv, rows, raw_keys)
    write_csv(args.summary_csv, summary, summary_keys)
    print(f"wrote {args.raw_csv}")
    print(f"wrote {args.summary_csv}")
    return 1 if any_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
