#!/usr/bin/env python3
"""Run one long GPU validation and summarize selected checkpoint frames."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from pathlib import Path

from validation_metrics import read_params, read_tecplot, summarize


POISSON_RE = re.compile(r"Poisson:\s+([0-9eE+\-.]+)")
ITERS_RE = re.compile(r"Poisson iters:\s+([0-9eE+\-.]+)")
TOTAL_RE = re.compile(r"Total per step:\s+([0-9eE+\-.]+)")
MLUPS_RE = re.compile(r"MLUPS:\s+([0-9eE+\-.]+)")
WALL_RE = re.compile(r"GPU wall_total_ms=([0-9eE+\-.]+)\s+wall_avg_ms_per_step=([0-9eE+\-.]+)\s+wall_MLUPS=([0-9eE+\-.]+)")


def parse_int_list(text: str) -> list[int]:
    values = sorted({int(item) for item in text.replace(",", " ").split() if item})
    if any(value < 0 for value in values):
        raise ValueError("checkpoint steps must be non-negative")
    return values


def parse_perf(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    wall = WALL_RE.search(text)
    if wall:
        out["wall_total_ms"] = float(wall.group(1))
        out["wall_avg_ms_per_step"] = float(wall.group(2))
        out["wall_mlups"] = float(wall.group(3))
    for line in text.splitlines():
        if "  Poisson:" in line:
            m = POISSON_RE.search(line)
            if m:
                out["poisson_ms_per_step"] = float(m.group(1))
        elif "Poisson iters:" in line:
            m = ITERS_RE.search(line)
            if m:
                out["poisson_iters"] = float(m.group(1))
        elif "Total per step:" in line:
            m = TOTAL_RE.search(line)
            if m:
                out["gpu_metric_total_ms_per_step"] = float(m.group(1))
        elif "  MLUPS:" in line:
            m = MLUPS_RE.search(line)
            if m:
                out["gpu_metric_mlups"] = float(m.group(1))
    return out


def rel_diff(value: float, ref: float, eps: float = 1.0e-30) -> float:
    return abs(value - ref) / max(abs(ref), eps)


def output_name(step: int) -> str:
    return f"3D{step:09d}.plt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=Path("./lbm_gpu"))
    parser.add_argument(
        "--params",
        type=Path,
        default=Path("double_droplet_stability_candidates_moving/params_tanh_w2_u0p01_cxdefault.in"),
    )
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--checkpoints", default="0,50,100,200,500,1000,2000,4000,8000")
    parser.add_argument("--out-dir", type=Path, default=Path("long_run_validation_outputs"))
    parser.add_argument("--csv", type=Path, default=Path("long_run_validation.csv"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--poisson", default="fused", choices=["split", "fused"])
    parser.add_argument("--pressure-boundary", default="fused", choices=["split", "fused"])
    parser.add_argument("--poisson-graph", action="store_true")
    args = parser.parse_args()

    checkpoints = parse_int_list(args.checkpoints)
    if not checkpoints or checkpoints[-1] != args.steps:
        checkpoints = sorted(set(checkpoints + [args.steps]))

    exe_abs = args.exe.resolve()
    params_abs = args.params.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "out").exists():
        shutil.rmtree(args.out_dir / "out")
    shutil.copy2(params_abs, args.out_dir / "params_used.in")

    cmd = [
        str(exe_abs),
        "--mode", "gpu",
        "--params", str(params_abs),
        "--steps", str(args.steps),
        "--output-steps", ",".join(str(step) for step in checkpoints),
        "--write-output",
        "--no-roofline",
        "--poisson", args.poisson,
        "--pressure-boundary", args.pressure_boundary,
    ]
    if args.poisson_graph:
        cmd.append("--poisson-graph")

    proc = subprocess.run(cmd, cwd=args.out_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (args.out_dir / "run.log").write_text(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"long run failed with code {proc.returncode}; see {args.out_dir / 'run.log'}")

    params_values = read_params(params_abs)
    perf = parse_perf(proc.stdout)
    rows: list[dict[str, object]] = []
    initial: dict[str, object] | None = None

    for step in checkpoints:
        plt = args.out_dir / "out" / output_name(step)
        if not plt.exists():
            raise RuntimeError(f"missing checkpoint file: {plt}")
        metrics = summarize(read_tecplot(plt), params_values, args.threshold)
        if initial is None:
            initial = metrics

        phase_mass = float(metrics["phase_mass"])
        initial_mass = float(initial["phase_mass"])
        liquid_voxels = float(metrics["liquid_voxels"])
        initial_liquid_voxels = float(initial["liquid_voxels"])
        centroid_distance = float(metrics["component_centroid_distance"])
        initial_centroid_distance = float(initial["component_centroid_distance"])

        row: dict[str, object] = {
            "step": step,
            "regime": metrics["regime"],
            "component_count": metrics["component_count"],
            "phase_mass": phase_mass,
            "phase_mass_rel_change": rel_diff(phase_mass, initial_mass),
            "liquid_voxels": liquid_voxels,
            "liquid_voxels_rel_change": rel_diff(liquid_voxels, initial_liquid_voxels),
            "largest_component_voxels": metrics["largest_component_voxels"],
            "second_component_voxels": metrics["second_component_voxels"],
            "centroid_distance": centroid_distance,
            "centroid_distance_rel_change": rel_diff(centroid_distance, initial_centroid_distance),
            "max_speed": metrics["max_speed"],
            "max_gas_speed": metrics["max_gas_speed"],
            "laplace_delta_p": metrics["laplace_delta_p"],
            "checkpoint_file": str(plt),
            "run_exit_code": proc.returncode,
        }
        row.update(perf)
        rows.append(row)
        print(
            f"step={step}: regime={row['regime']} comps={row['component_count']} "
            f"mass={float(row['phase_mass_rel_change']):.3e} "
            f"liq={float(row['liquid_voxels_rel_change']):.3e} "
            f"dist={float(row['centroid_distance']):.3e} "
            f"max_u={float(row['max_speed']):.3e}"
        )

    keys = [
        "step", "regime", "component_count", "phase_mass", "phase_mass_rel_change",
        "liquid_voxels", "liquid_voxels_rel_change", "largest_component_voxels",
        "second_component_voxels", "centroid_distance", "centroid_distance_rel_change",
        "max_speed", "max_gas_speed", "laplace_delta_p", "wall_total_ms",
        "wall_avg_ms_per_step", "wall_mlups", "gpu_metric_total_ms_per_step",
        "gpu_metric_mlups", "poisson_ms_per_step", "poisson_iters",
        "checkpoint_file", "run_exit_code",
    ]
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.csv}")
    print(f"outputs in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
