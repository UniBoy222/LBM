#!/usr/bin/env python3
"""Run the safe pressure-initializer pipeline for one pre-Poisson field."""

from __future__ import annotations

import argparse
import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from input_quality_gate import field_metrics, metric_ranges, score


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def safe_label(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def run(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(proc.stdout)
        raise SystemExit(proc.returncode)
    return proc.stdout


def read_gate_summary(path: Path) -> dict[str, object]:
    with path.open(newline="") as f:
        return dict(next(csv.DictReader(f)))


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("input_plt", type=Path, help="pre-Poisson field used by the PINN")
    parser.add_argument("--params", type=Path, default=root / "GPU" / "params_small.in")
    parser.add_argument("--model", type=Path,
                        default=root / "PINN_Poisson" / "models" / "pressure_initializer_augmented_abs_residual32.pt")
    parser.add_argument("--quality-manifest", type=Path,
                        default=root / "PINN_Poisson" / "data" / "augmented_manifest.csv")
    parser.add_argument("--quality-margin", type=float, default=0.25)
    parser.add_argument("--disable-input-quality-gate", action="store_true")
    parser.add_argument("--pressure-init-max-iterations", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--poisson", choices=("split", "fused", "onepass", "scalar"), default="onepass")
    parser.add_argument("--pressure-boundary", choices=("split", "fused"), default="fused")
    parser.add_argument("--poisson-check-interval", type=int, default=100)
    parser.add_argument("--poisson-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--reference-plt", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--label", default=None)
    parser.add_argument("--summary-csv", type=Path,
                        default=root / "PINN_Poisson" / "results" / "benchmarks" / "safe_pressure_initializer.csv")
    args = parser.parse_args()

    label = args.label or f"safe_pressure_init_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    pressure_dir = root / "PINN_Poisson" / "data" / "pressure_init"
    pressure_bin = pressure_dir / f"{safe_label(label)}.bin"

    quality_accept = True
    quality_reasons = ""
    if not args.disable_input_quality_gate:
        ranges = metric_ranges(args.quality_manifest, args.quality_margin)
        quality_accept, reasons = score(field_metrics(args.input_plt), ranges)
        quality_reasons = "; ".join(reasons)

    used_pressure_init = False
    if quality_accept:
        run([
            str(root / "PINN_Poisson" / ".venv" / "bin" / "python"),
            str(root / "PINN_Poisson" / "scripts" / "predict_pressure_initializer.py"),
            str(args.input_plt),
            "--model", str(args.model),
            "--out", str(pressure_bin),
            "--mode", "absolute",
            "--device", args.device,
        ], root)
        used_pressure_init = True

    gate_label = f"{label}_gate"
    gate_cmd = [
        str(root / "PINN_Poisson" / ".venv" / "bin" / "python"),
        str(root / "PINN_Poisson" / "scripts" / "run_pressure_init_gate.py"),
        "--params", str(args.params),
        "--steps", str(args.steps),
        "--poisson", args.poisson,
        "--pressure-boundary", args.pressure_boundary,
        "--poisson-check-interval", str(args.poisson_check_interval),
        "--poisson-tolerance", str(args.poisson_tolerance),
        "--label", gate_label,
    ]
    if used_pressure_init:
        gate_cmd.extend(["--pressure-init-file", str(pressure_bin), "--pressure-init-mode", "absolute"])
        if args.pressure_init_max_iterations > 0:
            gate_cmd.extend(["--pressure-init-max-iterations", str(args.pressure_init_max_iterations)])
    if args.reference_plt is not None:
        gate_cmd.extend(["--reference-plt", str(args.reference_plt)])
    run(gate_cmd, root)

    gate_csv = root / "PINN_Poisson" / "results" / "gates" / gate_label / "gate_summary.csv"
    row = read_gate_summary(gate_csv)
    row.update({
        "safe_label": label,
        "input_plt": str(args.input_plt),
        "model": str(args.model),
        "quality_manifest": str(args.quality_manifest),
        "input_quality_accept": int(quality_accept),
        "input_quality_reasons": quality_reasons,
        "used_pressure_initializer": int(used_pressure_init),
        "safe_pressure_init_file": str(pressure_bin) if used_pressure_init else "",
    })

    keys = [
        "safe_label", "input_plt", "model", "quality_manifest",
        "input_quality_accept", "input_quality_reasons", "used_pressure_initializer",
        "pressure_init_attempts", "pressure_init_accepts", "pressure_init_fallbacks",
        "pressure_init_max_iterations", "poisson_iters", "total_ms_per_step",
        "pressure_rel_l2", "pressure_max_abs", "phase_mass_rel_error",
        "component_count", "reference_component_count",
        "safe_pressure_init_file", "plt", "diagnostics", "stdout", "command",
    ]
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = args.summary_csv.exists() and args.summary_csv.stat().st_size > 0
    with args.summary_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    print(
        f"{label}: quality={int(quality_accept)} used_init={int(used_pressure_init)} "
        f"accepts={row.get('pressure_init_accepts')} fallbacks={row.get('pressure_init_fallbacks')} "
        f"iters={row.get('poisson_iters')} rel={row.get('pressure_rel_l2', '')}"
    )
    if quality_reasons:
        print(f"quality_reasons={quality_reasons}")
    print(f"updated {args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
