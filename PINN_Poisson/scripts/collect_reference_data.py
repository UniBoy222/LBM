#!/usr/bin/env python3
"""Collect solver reference fields for PINN pressure-initializer training/gates."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from tecplot_io import read_tecplot, write_pressure_initializer


ITERS_RE = re.compile(r"Poisson iters:\s+([0-9eE+\-.]+)")
TOTAL_RE = re.compile(r"Total per step:\s+([0-9eE+\-.]+)")
POISSON_RE = re.compile(r"Poisson:\s+([0-9eE+\-.]+)")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_perf(text: str) -> dict[str, object]:
    out: dict[str, object] = {}
    section = ""
    for line in text.splitlines():
        if line.startswith("Average time per kernel"):
            section = "average"
            continue
        if line.startswith("Kernel time distribution"):
            section = "distribution"
            continue
        if "Poisson iters:" in line:
            match = ITERS_RE.search(line)
            if match:
                out["poisson_iters"] = float(match.group(1))
        elif "Total per step:" in line:
            match = TOTAL_RE.search(line)
            if match:
                out["total_ms_per_step"] = float(match.group(1))
        elif section == "average" and line.strip().startswith("Poisson:"):
            match = POISSON_RE.search(line)
            if match:
                out["poisson_ms_per_step"] = float(match.group(1))
    return out


def run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout


def parse_step_list(text: str | None, final_step: int) -> list[int]:
    if text is None or text.strip() == "":
        return [final_step]
    steps: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        step = int(item)
        if step < 0 or step > final_step:
            raise ValueError(f"sample step {step} outside [0, {final_step}]")
        steps.append(step)
    if not steps:
        raise ValueError("no sample steps selected")
    return sorted(dict.fromkeys(steps))


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=root / "GPU" / "lbm_gpu")
    parser.add_argument("--params", type=Path, default=root / "GPU" / "params_small.in")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--poisson", choices=("split", "fused", "onepass", "scalar"), default="onepass")
    parser.add_argument("--pressure-boundary", choices=("split", "fused"), default="fused")
    parser.add_argument("--poisson-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--sample-steps", default=None,
                        help="comma-separated output steps to export; default is final --steps")
    parser.add_argument("--label", default=None)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.build:
        code, text = run(["make", "-C", str(root / "GPU"), "gpu"], root)
        if code != 0:
            print(text)
            return code

    sample_steps = parse_step_list(args.sample_steps, args.steps)
    label = args.label or f"{args.poisson}_step{args.steps}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = root / "PINN_Poisson" / "data" / "runs" / label
    pressure_dir = root / "PINN_Poisson" / "data" / "pressure_init"
    gates_dir = root / "PINN_Poisson" / "results" / "gates"
    run_dir.mkdir(parents=True, exist_ok=True)
    gates_dir.mkdir(parents=True, exist_ok=True)

    diag_csv = gates_dir / f"{label}_poisson_diagnostics.csv"
    cmd = [
        str(args.exe.resolve()),
        "--mode", "gpu",
        "--params", str(args.params.resolve()),
        "--steps", str(args.steps),
        "--poisson", args.poisson,
        "--pressure-boundary", args.pressure_boundary,
        "--poisson-check-interval", "100",
        "--poisson-tolerance", str(args.poisson_tolerance),
        "--poisson-diagnostics", str(diag_csv.resolve()),
        "--output-steps", ",".join(str(step) for step in sample_steps),
        "--write-output",
        "--no-roofline",
    ]
    code, text = run(cmd, run_dir)
    (run_dir / "stdout.log").write_text(text)
    if code != 0:
        print(text)
        return code

    manifest = root / "PINN_Poisson" / "data" / "reference_manifest.csv"
    exists = manifest.exists()
    keys = [
        "label", "params", "run_steps", "sample_step", "poisson", "pressure_boundary", "poisson_tolerance",
        "exit_code", "poisson_iters", "poisson_ms_per_step", "total_ms_per_step",
        "plt", "pressure_initializer", "diagnostics", "stdout", "command",
    ]
    if exists:
        old_header = manifest.open().readline().strip().split(",")
        if old_header != keys:
            backup = manifest.with_suffix(".legacy.csv")
            manifest.replace(backup)
            print(f"moved incompatible manifest to {backup}")
            exists = False
    perf = parse_perf(text)
    with manifest.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for sample_step in sample_steps:
            sample_plt = run_dir / "out" / f"3D{sample_step:09d}.plt"
            if not sample_plt.exists():
                raise SystemExit(f"missing solver output: {sample_plt}")

            data = read_tecplot(sample_plt)
            fields = data["fields"]  # type: ignore[assignment]
            sample_label = f"{label}_step{sample_step}"
            pressure_bin = pressure_dir / f"{sample_label}_pressure.bin"
            write_pressure_initializer(
                pressure_bin,
                int(data["lx"]),
                int(data["ly"]),
                int(data["lz"]),
                fields["press"],  # type: ignore[index]
            )
            row: dict[str, object] = {
                "label": sample_label,
                "params": str(args.params),
                "run_steps": args.steps,
                "sample_step": sample_step,
                "poisson": args.poisson,
                "pressure_boundary": args.pressure_boundary,
                "poisson_tolerance": args.poisson_tolerance,
                "exit_code": code,
                "plt": str(sample_plt),
                "pressure_initializer": str(pressure_bin),
                "diagnostics": str(diag_csv),
                "stdout": str(run_dir / "stdout.log"),
                "command": " ".join(cmd),
            }
            row.update(perf)
            writer.writerow(row)
            print(f"wrote {pressure_bin}")

    print(f"updated {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
