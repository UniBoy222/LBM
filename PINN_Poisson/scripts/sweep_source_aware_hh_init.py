#!/usr/bin/env python3
"""Sweep source-aware HH initialization for existing pressure-init files."""

from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    label: str
    path: Path
    mode: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_candidate(text: str) -> Candidate:
    parts = text.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("candidate must be LABEL:FILE:absolute|delta")
    label, path, mode = parts
    if mode not in {"absolute", "delta"}:
        raise argparse.ArgumentTypeError("candidate mode must be absolute or delta")
    return Candidate(label, Path(path), mode)


def scale_tag(scale: float | None) -> str:
    if scale is None:
        return "plain"
    return str(scale).replace(".", "p").replace("-", "m")


def run(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(proc.stdout)
        raise SystemExit(proc.returncode)


def default_candidates(root: Path) -> list[Candidate]:
    pressure_dir = root / "PINN_Poisson" / "data" / "pressure_init"
    return [
        Candidate("pinn_default_step1", pressure_dir / "pinn_default_step1_pred.bin", "absolute"),
        Candidate("pinn_paired_delta_step1", pressure_dir / "pinn_paired_delta_step1.bin", "delta"),
        Candidate("pinn_residual_delta_0p25_step1", pressure_dir / "pinn_residual_delta_step1_scale_0p25.bin", "delta"),
    ]


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        help="LABEL:FILE:absolute|delta; can be passed multiple times",
    )
    parser.add_argument("--scales", default="none,0.25,0.5,1.0,1.5,2.0")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--poisson", choices=("split", "fused", "onepass", "scalar"), default="onepass")
    parser.add_argument("--pressure-boundary", choices=("split", "fused"), default="fused")
    parser.add_argument("--poisson-check-interval", type=int, default=100)
    parser.add_argument("--poisson-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--reference-plt", type=Path,
                        default=root / "PINN_Poisson" / "data" / "runs" / "oracle_smoke_step1" / "out" / "3D000000001.plt")
    parser.add_argument("--label-prefix", default="hh_sweep")
    parser.add_argument("--csv", type=Path,
                        default=root / "PINN_Poisson" / "results" / "benchmarks" / "source_aware_hh_sweep.csv")
    args = parser.parse_args()

    candidates = args.candidate or default_candidates(root)
    scale_values: list[float | None] = []
    for text in args.scales.split(","):
        value = text.strip()
        if value in {"", "none", "plain", "off"}:
            scale_values.append(None)
        else:
            scale_values.append(float(value))

    rows: list[dict[str, object]] = []
    for candidate in candidates:
        pressure_path = candidate.path if candidate.path.is_absolute() else root / candidate.path
        if not pressure_path.exists():
            raise SystemExit(f"missing pressure-init file: {pressure_path}")
        for scale in scale_values:
            label = f"{args.label_prefix}_{candidate.label}_{scale_tag(scale)}"
            cmd = [
                str(root / "PINN_Poisson" / ".venv" / "bin" / "python"),
                str(root / "PINN_Poisson" / "scripts" / "run_pressure_init_gate.py"),
                "--steps", str(args.steps),
                "--poisson", args.poisson,
                "--pressure-boundary", args.pressure_boundary,
                "--poisson-check-interval", str(args.poisson_check_interval),
                "--poisson-tolerance", str(args.poisson_tolerance),
                "--pressure-init-file", str(pressure_path),
                "--pressure-init-mode", candidate.mode,
                "--reference-plt", str(args.reference_plt),
                "--label", label,
            ]
            if scale is not None:
                cmd.extend(["--source-aware-hh-init", "--source-aware-hh-scale", str(scale)])
            run(cmd, root)

            gate_csv = root / "PINN_Poisson" / "results" / "gates" / label / "gate_summary.csv"
            row = next(csv.DictReader(gate_csv.open()))
            row["candidate"] = candidate.label
            row["hh_scale"] = "" if scale is None else scale
            row["source_aware_hh_init"] = 0 if scale is None else 1
            rows.append(row)
            print(
                f"{candidate.label} hh={scale_tag(scale)} accepts={row.get('pressure_init_accepts')} "
                f"fallbacks={row.get('pressure_init_fallbacks')} iters={row.get('poisson_iters')} "
                f"total_ms={row.get('total_ms_per_step')} rel={row.get('pressure_rel_l2')}"
            )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "candidate", "hh_scale", "source_aware_hh_init", "source_aware_hh_scale",
        "label", "pressure_init_mode", "poisson_check_interval",
        "pressure_init_accepts", "pressure_init_fallbacks",
        "poisson_iters", "poisson_ms_per_step", "total_ms_per_step",
        "pressure_rel_l2", "pressure_max_abs", "phase_mass_rel_error",
        "component_count", "reference_component_count", "pressure_init_file",
        "diagnostics", "stdout",
    ]
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
