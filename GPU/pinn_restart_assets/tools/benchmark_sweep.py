#!/usr/bin/env python3
"""Run CPU/GPU LBM benchmark variants and save reproducible timing rows."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


CPU_RE = re.compile(
    r"CPU total_ms=([0-9eE+\-.]+)\s+avg_ms_per_step=([0-9eE+\-.]+)\s+MLUPS=([0-9eE+\-.]+)"
)
GPU_WALL_RE = re.compile(
    r"GPU wall_total_ms=([0-9eE+\-.]+)\s+wall_avg_ms_per_step=([0-9eE+\-.]+)\s+wall_MLUPS=([0-9eE+\-.]+)"
)
KERNEL_RE = re.compile(r"^\s+(Collision|Stream|Macro|Poisson|Total per step|MLUPS):\s+([0-9eE+\-.]+)")
ITERS_RE = re.compile(r"Poisson iters:\s+([0-9eE+\-.]+)")


@dataclass(frozen=True)
class Variant:
    name: str
    mode: str
    poisson: str = "split"
    pressure_boundary: str = "split"
    graph: bool = False


DEFAULT_VARIANTS = [
    Variant("cpu_reference", "cpu"),
    Variant("gpu_split_baseline", "gpu", "split", "split"),
    Variant("gpu_all_fused", "gpu", "fused", "fused"),
    Variant("gpu_graph_all_fused", "gpu", "fused", "fused", True),
    Variant("gpu_poisson_onepass", "gpu", "onepass", "fused"),
    Variant("gpu_graph_onepass", "gpu", "onepass", "fused", True),
]


def run_cmd(cmd: list[str], cwd: Path | None) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout


def parse_output(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    cpu = CPU_RE.search(text)
    if cpu:
        out["wall_total_ms"] = float(cpu.group(1))
        out["wall_avg_ms_per_step"] = float(cpu.group(2))
        out["wall_mlups"] = float(cpu.group(3))

    gpu = GPU_WALL_RE.search(text)
    if gpu:
        out["wall_total_ms"] = float(gpu.group(1))
        out["wall_avg_ms_per_step"] = float(gpu.group(2))
        out["wall_mlups"] = float(gpu.group(3))

    section = ""
    for line in text.splitlines():
        if line.startswith("Average time per kernel"):
            section = "average"
            continue
        if line.startswith("Kernel time distribution"):
            section = "distribution"
            continue

        iters = ITERS_RE.search(line)
        if iters:
            out["poisson_iters"] = float(iters.group(1))
            continue

        m = KERNEL_RE.match(line)
        if not m:
            continue
        key = m.group(1)
        value = float(m.group(2))
        if section == "distribution":
            if key == "Collision":
                out["collision_percent"] = value
            elif key == "Stream":
                out["stream_percent"] = value
            elif key == "Macro":
                out["macro_percent"] = value
            elif key == "Poisson":
                out["poisson_percent"] = value
            continue
        if section != "average":
            continue

        if key == "Collision":
            out["collision_ms_per_step"] = value
        elif key == "Stream":
            out["stream_ms_per_step"] = value
        elif key == "Macro":
            out["macro_ms_per_step"] = value
        elif key == "Poisson":
            out["poisson_ms_per_step"] = value
        elif key == "Total per step":
            out["kernel_total_ms_per_step"] = value
        elif key == "MLUPS":
            out["kernel_mlups"] = value

    return out


def build_command(exe: Path, params: Path, variant: Variant, steps: int) -> list[str]:
    cmd = [
        str(exe.resolve()),
        "--mode", variant.mode,
        "--params", str(params.resolve()),
        "--steps", str(steps),
        "--poisson", variant.poisson,
        "--pressure-boundary", variant.pressure_boundary,
        "--no-roofline",
    ]
    if variant.graph:
        cmd.append("--poisson-graph")
    return cmd


def run_variant(
    exe: Path,
    params: Path,
    variant: Variant,
    gpu_steps: int,
    cpu_steps: int,
    warmup_steps: int,
    cwd: Path | None,
) -> dict[str, object]:
    steps = cpu_steps if variant.mode == "cpu" else gpu_steps

    if variant.mode == "gpu" and warmup_steps > 0:
        warmup = build_command(exe, params, variant, warmup_steps)
        run_cmd(warmup, cwd)

    cmd = build_command(exe, params, variant, steps)
    code, text = run_cmd(cmd, cwd)
    row: dict[str, object] = {
        "variant": variant.name,
        "mode": variant.mode,
        "steps": steps,
        "poisson": variant.poisson,
        "pressure_boundary": variant.pressure_boundary,
        "cuda_graph": int(variant.graph),
        "exit_code": code,
        "command": " ".join(cmd),
    }
    row.update(parse_output(text))
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=Path("./lbm_gpu"))
    parser.add_argument("--params", type=Path, default=Path("params_small.in"))
    parser.add_argument("--gpu-steps", type=int, default=20)
    parser.add_argument("--cpu-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--csv", type=Path, default=Path("benchmark_sweep.csv"))
    parser.add_argument("--skip-cpu", action="store_true")
    args = parser.parse_args()

    variants = DEFAULT_VARIANTS[1:] if args.skip_cpu else DEFAULT_VARIANTS
    rows = [
        run_variant(
            args.exe,
            args.params,
            variant,
            args.gpu_steps,
            args.cpu_steps,
            args.warmup_steps,
            Path("."),
        )
        for variant in variants
    ]

    baseline_ms = next(
        (float(row["kernel_total_ms_per_step"]) for row in rows if row["variant"] == "gpu_split_baseline"
         and "kernel_total_ms_per_step" in row),
        float("nan"),
    )
    cpu_ms = next(
        (float(row["wall_avg_ms_per_step"]) for row in rows if row["variant"] == "cpu_reference"
         and "wall_avg_ms_per_step" in row),
        float("nan"),
    )
    for row in rows:
        ms = row.get("kernel_total_ms_per_step", row.get("wall_avg_ms_per_step"))
        if isinstance(ms, (float, int)) and ms > 0.0:
            row["speedup_vs_gpu_split"] = baseline_ms / float(ms) if baseline_ms == baseline_ms else ""
            row["speedup_vs_cpu"] = cpu_ms / float(ms) if cpu_ms == cpu_ms else ""

    keys = [
        "variant", "mode", "steps", "poisson", "pressure_boundary", "cuda_graph",
        "exit_code", "wall_total_ms", "wall_avg_ms_per_step", "wall_mlups",
        "collision_ms_per_step", "stream_ms_per_step", "macro_ms_per_step",
        "poisson_ms_per_step", "poisson_iters", "kernel_total_ms_per_step",
        "kernel_mlups", "collision_percent", "stream_percent", "macro_percent",
        "poisson_percent", "speedup_vs_gpu_split", "speedup_vs_cpu", "command",
    ]
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['variant']}: exit={row['exit_code']} "
            f"wall_ms={row.get('wall_avg_ms_per_step', 'NA')} "
            f"kernel_ms={row.get('kernel_total_ms_per_step', 'NA')} "
            f"mlups={row.get('kernel_mlups', row.get('wall_mlups', 'NA'))}"
        )
    print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
