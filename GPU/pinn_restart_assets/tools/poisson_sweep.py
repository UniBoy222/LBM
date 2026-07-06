#!/usr/bin/env python3
"""Run Poisson convergence sweeps and collect accuracy/performance evidence."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


FIELD_RE = re.compile(r"^(fei|rho|u|v|w|p),([^,]+),([^,]+),")
MASS_RE = re.compile(r"rel_diff=([0-9eE+\-.]+)")
POISSON_RE = re.compile(r"Poisson:\s+([0-9eE+\-.]+)")
ITERS_RE = re.compile(r"Poisson iters:\s+([0-9eE+\-.]+)")
TOTAL_RE = re.compile(r"Total per step:\s+([0-9eE+\-.]+)")
MLUPS_RE = re.compile(r"MLUPS:\s+([0-9eE+\-.]+)")


@dataclass(frozen=True)
class Variant:
    name: str
    poisson: str = "fused"
    pressure_boundary: str = "fused"
    check_interval: int = 100
    poisson_tolerance: float = 1.0e-3
    graph: bool = False


DEFAULT_VARIANTS = [
    Variant("strict_all_fused", check_interval=100, poisson_tolerance=1.0e-3),
    Variant("graph_all_fused", check_interval=100, poisson_tolerance=1.0e-3, graph=True),
    Variant("strict_onepass", poisson="onepass", check_interval=100, poisson_tolerance=1.0e-3),
    Variant("graph_onepass", poisson="onepass", check_interval=100, poisson_tolerance=1.0e-3, graph=True),
    Variant("check50_tol1e-3", check_interval=50, poisson_tolerance=1.0e-3),
    Variant("check50_tol5e-4", check_interval=50, poisson_tolerance=5.0e-4),
    Variant("check50_tol1e-4", check_interval=50, poisson_tolerance=1.0e-4),
    Variant("check25_tol1e-4", check_interval=25, poisson_tolerance=1.0e-4),
]


def parse_variant(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = FIELD_RE.match(line.strip())
        if m:
            field, max_abs, rel_l2 = m.groups()
            out[f"{field}_max_abs"] = float(max_abs)
            out[f"{field}_rel_l2"] = float(rel_l2)
            continue
        if line.startswith("phase_mass_cpu="):
            m = MASS_RE.search(line)
            if m:
                out["mass_rel_diff"] = float(m.group(1))
            continue
        if "  Poisson:" in line:
            m = POISSON_RE.search(line)
            if m:
                out["poisson_ms_per_step"] = float(m.group(1))
            continue
        if "Poisson iters:" in line:
            m = ITERS_RE.search(line)
            if m:
                out["poisson_iters"] = float(m.group(1))
            continue
        if "Total per step:" in line:
            m = TOTAL_RE.search(line)
            if m:
                out["gpu_metric_total_ms_per_step"] = float(m.group(1))
            continue
        if "  MLUPS:" in line:
            m = MLUPS_RE.search(line)
            if m:
                out["gpu_metric_mlups"] = float(m.group(1))
    return out


def run_variant(exe: Path, params: Path, steps: int, variant: Variant) -> tuple[int, dict[str, float], str]:
    exe_str = str(exe)
    if exe.parent == Path(".") and "/" not in exe_str:
        exe_str = f"./{exe_str}"
    cmd = [
        exe_str,
        "--params", str(params),
        "--steps", str(steps),
        "--poisson", variant.poisson,
        "--pressure-boundary", variant.pressure_boundary,
        "--poisson-check-interval", str(variant.check_interval),
        "--poisson-tolerance", str(variant.poisson_tolerance),
        "--tolerance", "1e9",
        "--pressure-tolerance", "1e9",
    ]
    if variant.graph:
        cmd.append("--poisson-graph")

    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, parse_variant(proc.stdout), proc.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=Path("./lbm_compare"))
    parser.add_argument("--params", type=Path, default=Path("params_small.in"))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--strict-tol", type=float, default=1.0e-8)
    parser.add_argument("--physics-tol", type=float, default=1.0e-4)
    parser.add_argument("--csv", type=Path, default=Path("poisson_sweep.csv"))
    parser.add_argument("--show-output", action="store_true")
    args = parser.parse_args()

    field_keys = ["fei", "rho", "u", "v", "w", "p"]
    rows: list[dict[str, object]] = []

    for variant in DEFAULT_VARIANTS:
        code, metrics, raw = run_variant(args.exe, args.params, args.steps, variant)
        if args.show_output:
            print(raw)

        strict_pass = all(metrics.get(f"{f}_rel_l2", float("inf")) <= args.strict_tol for f in field_keys)
        strict_pass = strict_pass and metrics.get("mass_rel_diff", float("inf")) <= args.strict_tol
        physics_pass = all(metrics.get(f"{f}_rel_l2", float("inf")) <= args.physics_tol for f in field_keys)
        physics_pass = physics_pass and metrics.get("mass_rel_diff", float("inf")) <= args.physics_tol

        row: dict[str, object] = {
            "variant": variant.name,
            "steps": args.steps,
            "poisson": variant.poisson,
            "pressure_boundary": variant.pressure_boundary,
            "check_interval": variant.check_interval,
            "poisson_tolerance": variant.poisson_tolerance,
            "graph": int(variant.graph),
            "compare_exit_code": code,
            "strict_pass": int(strict_pass),
            "physics_pass": int(physics_pass),
        }
        row.update(metrics)
        rows.append(row)

        print(
            f"{variant.name}: strict={strict_pass} physics={physics_pass} "
            f"iters={metrics.get('poisson_iters', float('nan')):.1f} "
            f"p_rel={metrics.get('p_rel_l2', float('nan')):.3e} "
            f"total_ms={metrics.get('gpu_metric_total_ms_per_step', float('nan')):.3f}"
        )

    keys = [
        "variant", "steps", "poisson", "pressure_boundary", "check_interval",
        "poisson_tolerance", "graph", "compare_exit_code", "strict_pass", "physics_pass",
        "gpu_metric_total_ms_per_step", "poisson_ms_per_step", "poisson_iters",
        "gpu_metric_mlups", "mass_rel_diff",
    ]
    for field in field_keys:
        keys.extend([f"{field}_max_abs", f"{field}_rel_l2"])

    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
