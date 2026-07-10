#!/usr/bin/env python3
"""Sweep pressure-initializer export scales and run the residual gate."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(proc.stdout)
        raise SystemExit(proc.returncode)


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("input_plt", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference-plt", type=Path, required=True)
    parser.add_argument("--mode", choices=("absolute", "delta"), default="delta")
    parser.add_argument("--scales", default="0.25,0.5,0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--poisson", choices=("split", "fused", "onepass", "scalar"), default="onepass")
    parser.add_argument("--pressure-boundary", choices=("split", "fused"), default="fused")
    parser.add_argument("--label-prefix", default="scale_sweep")
    parser.add_argument("--csv", type=Path, default=root / "PINN_Poisson" / "results" / "benchmarks" / "pressure_scale_sweep.csv")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for text in args.scales.split(","):
        scale = float(text.strip())
        tag = str(scale).replace(".", "p").replace("-", "m")
        pressure_bin = root / "PINN_Poisson" / "data" / "pressure_init" / f"{args.label_prefix}_{tag}.bin"
        gate_label = f"{args.label_prefix}_{tag}_gate"
        run([
            str(root / "PINN_Poisson" / ".venv" / "bin" / "python"),
            str(root / "PINN_Poisson" / "scripts" / "predict_pressure_initializer.py"),
            str(args.input_plt),
            "--model", str(args.model),
            "--out", str(pressure_bin),
            "--mode", args.mode,
            "--scale", str(scale),
            "--device", args.device,
        ], root)
        run([
            str(root / "PINN_Poisson" / ".venv" / "bin" / "python"),
            str(root / "PINN_Poisson" / "scripts" / "run_pressure_init_gate.py"),
            "--steps", str(args.steps),
            "--poisson", args.poisson,
            "--pressure-boundary", args.pressure_boundary,
            "--pressure-init-file", str(pressure_bin),
            "--pressure-init-mode", args.mode,
            "--reference-plt", str(args.reference_plt),
            "--label", gate_label,
        ], root)
        gate_csv = root / "PINN_Poisson" / "results" / "gates" / gate_label / "gate_summary.csv"
        row = next(csv.DictReader(gate_csv.open()))
        row["scale"] = scale
        row["pressure_init_file"] = str(pressure_bin)
        rows.append(row)
        print(
            f"scale={scale:g} accepts={row.get('pressure_init_accepts')} "
            f"fallbacks={row.get('pressure_init_fallbacks')} "
            f"iters={row.get('poisson_iters')} rel={row.get('pressure_rel_l2')}"
        )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "scale", "label", "pressure_init_accepts", "pressure_init_fallbacks",
        "poisson_iters", "total_ms_per_step", "pressure_rel_l2",
        "pressure_max_abs", "phase_mass_rel_error", "component_count",
        "reference_component_count", "pressure_init_file",
    ]
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

