#!/usr/bin/env python3
"""Run the residual-controlled pressure-initializer gate."""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from tecplot_io import read_tecplot


ITERS_RE = re.compile(r"Poisson iters:\s+([0-9eE+\-.]+)")
TOTAL_RE = re.compile(r"Total per step:\s+([0-9eE+\-.]+)")
POISSON_RE = re.compile(r"Poisson:\s+([0-9eE+\-.]+)")
PINN_RE = re.compile(r"attempts=([0-9]+), accepts=([0-9]+), fallbacks=([0-9]+)")
GPU_WALL_RE = re.compile(r"GPU wall_total_ms=([0-9eE+\-.]+)\s+wall_avg_ms_per_step=([0-9eE+\-.]+)")
CPU_WALL_RE = re.compile(r"CPU total_ms=([0-9eE+\-.]+)\s+avg_ms_per_step=([0-9eE+\-.]+)")


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
        wall_match = GPU_WALL_RE.search(line) or CPU_WALL_RE.search(line)
        if wall_match:
            out["wall_total_ms"] = float(wall_match.group(1))
            out["wall_ms_per_step"] = float(wall_match.group(2))
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
        elif "PINN pressure initializer:" in line:
            match = PINN_RE.search(line)
            if match:
                out["pressure_init_attempts"] = int(match.group(1))
                out["pressure_init_accepts"] = int(match.group(2))
                out["pressure_init_fallbacks"] = int(match.group(3))
    return out


def rel_l2_and_max(values: list[float], refs: list[float]) -> tuple[float, float]:
    sum_diff2 = 0.0
    sum_ref2 = 0.0
    max_abs = 0.0
    for value, ref in zip(values, refs):
        diff = value - ref
        sum_diff2 += diff * diff
        sum_ref2 += ref * ref
        max_abs = max(max_abs, abs(diff))
    rel = math.sqrt(sum_diff2 / sum_ref2) if sum_ref2 > 0.0 else math.sqrt(sum_diff2)
    return rel, max_abs


def component_count(fei: list[float], lx: int, ly: int, lz: int, threshold: float) -> int:
    mask = [value >= threshold for value in fei]
    seen = bytearray(len(mask))

    def index(x: int, y: int, z: int) -> int:
        return (z * ly + y) * lx + x

    count = 0
    for seed, keep in enumerate(mask):
        if not keep or seen[seed]:
            continue
        count += 1
        q: deque[int] = deque([seed])
        seen[seed] = 1
        while q:
            idx = q.popleft()
            x = idx % lx
            y = (idx // lx) % ly
            z = idx // (lx * ly)
            neighbors = []
            if x > 0:
                neighbors.append(index(x - 1, y, z))
            if x + 1 < lx:
                neighbors.append(index(x + 1, y, z))
            neighbors.append(index(x, (y - 1) % ly, z))
            neighbors.append(index(x, (y + 1) % ly, z))
            neighbors.append(index(x, y, (z - 1) % lz))
            neighbors.append(index(x, y, (z + 1) % lz))
            for nb in neighbors:
                if mask[nb] and not seen[nb]:
                    seen[nb] = 1
                    q.append(nb)
    return count


def add_reference_metrics(row: dict[str, object], candidate_plt: Path, reference_plt: Path) -> None:
    cand = read_tecplot(candidate_plt)
    ref = read_tecplot(reference_plt)
    if (cand["lx"], cand["ly"], cand["lz"]) != (ref["lx"], ref["ly"], ref["lz"]):
        raise ValueError("candidate/reference grid dimensions do not match")

    cand_fields = cand["fields"]  # type: ignore[assignment]
    ref_fields = ref["fields"]  # type: ignore[assignment]
    press_rel, press_max = rel_l2_and_max(cand_fields["press"], ref_fields["press"])  # type: ignore[index]
    row["pressure_rel_l2"] = press_rel
    row["pressure_max_abs"] = press_max
    ref_mass = sum(ref_fields["fei"])  # type: ignore[index]
    cand_mass = sum(cand_fields["fei"])  # type: ignore[index]
    row["phase_mass_rel_error"] = abs(cand_mass - ref_mass) / abs(ref_mass) if ref_mass else abs(cand_mass - ref_mass)
    threshold = 0.5 * (max(ref_fields["fei"]) + min(ref_fields["fei"]))  # type: ignore[index]
    row["component_count"] = component_count(
        cand_fields["fei"], int(cand["lx"]), int(cand["ly"]), int(cand["lz"]), threshold  # type: ignore[index]
    )
    row["reference_component_count"] = component_count(
        ref_fields["fei"], int(ref["lx"]), int(ref["ly"]), int(ref["lz"]), threshold  # type: ignore[index]
    )


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=root / "GPU" / "lbm_gpu")
    parser.add_argument("--params", type=Path, default=root / "GPU" / "params_small.in")
    parser.add_argument("--pressure-init-file", type=Path, default=None)
    parser.add_argument("--pressure-init-mode", choices=("absolute", "delta"), default="absolute")
    parser.add_argument("--pressure-init-max-iterations", type=int, default=0,
                        help="cap only the PINN-initialized attempt; 0 keeps the solver default")
    parser.add_argument("--pressure-init-check-interval", type=int, default=0,
                        help="0 uses --poisson-check-interval for the PINN attempt")
    parser.add_argument("--reference-plt", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--poisson", choices=("split", "fused", "onepass", "scalar"), default="onepass")
    parser.add_argument("--pressure-boundary", choices=("split", "fused"), default="fused")
    parser.add_argument("--poisson-check-interval", type=int, default=100)
    parser.add_argument("--poisson-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--source-aware-hh-init", action="store_true")
    parser.add_argument("--source-aware-hh-scale", type=float, default=1.0)
    parser.add_argument("--label", default=None)
    parser.add_argument("--summary-csv", type=Path,
                        default=root / "PINN_Poisson" / "results" / "gates" / "pressure_init_gate_summary.csv")
    args = parser.parse_args()

    label = args.label or f"pressure_init_gate_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = root / "PINN_Poisson" / "results" / "gates" / label
    run_dir.mkdir(parents=True, exist_ok=True)
    diag_csv = run_dir / "poisson_diagnostics.csv"
    cmd = [
        str(args.exe.resolve()),
        "--mode", "gpu",
        "--params", str(args.params.resolve()),
        "--steps", str(args.steps),
        "--poisson", args.poisson,
        "--pressure-boundary", args.pressure_boundary,
        "--poisson-check-interval", str(args.poisson_check_interval),
        "--poisson-tolerance", str(args.poisson_tolerance),
        "--poisson-diagnostics", str(diag_csv.resolve()),
        "--output-frequency", str(args.steps),
        "--write-output",
        "--no-roofline",
    ]
    if args.pressure_init_file is not None:
        cmd.extend([
            "--pressure-init-file", str(args.pressure_init_file.resolve()),
            "--pressure-init-mode", args.pressure_init_mode,
        ])
        if args.pressure_init_max_iterations > 0:
            cmd.extend(["--pressure-init-max-iterations", str(args.pressure_init_max_iterations)])
        if args.pressure_init_check_interval > 0:
            cmd.extend(["--pressure-init-check-interval", str(args.pressure_init_check_interval)])
    if args.source_aware_hh_init:
        cmd.extend(["--source-aware-hh-init", "--source-aware-hh-scale", str(args.source_aware_hh_scale)])
    proc = subprocess.run(cmd, cwd=run_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = run_dir / "stdout.log"
    stdout.write_text(proc.stdout)

    row: dict[str, object] = {
        "label": label,
        "exit_code": proc.returncode,
        "steps": args.steps,
        "poisson": args.poisson,
        "pressure_boundary": args.pressure_boundary,
        "poisson_check_interval": args.poisson_check_interval,
        "pressure_init_file": str(args.pressure_init_file) if args.pressure_init_file is not None else "",
        "pressure_init_mode": args.pressure_init_mode,
        "pressure_init_max_iterations": args.pressure_init_max_iterations,
        "pressure_init_check_interval": args.pressure_init_check_interval,
        "source_aware_hh_init": int(args.source_aware_hh_init),
        "source_aware_hh_scale": args.source_aware_hh_scale,
        "diagnostics": str(diag_csv),
        "stdout": str(stdout),
        "command": " ".join(cmd),
    }
    row.update(parse_perf(proc.stdout))

    final_plt = run_dir / "out" / f"3D{args.steps:09d}.plt"
    row["plt"] = str(final_plt) if final_plt.exists() else ""
    if proc.returncode == 0 and args.reference_plt is not None and final_plt.exists():
        add_reference_metrics(row, final_plt, args.reference_plt)

    out_csv = run_dir / "gate_summary.csv"
    keys = [
        "label", "exit_code", "steps", "poisson", "pressure_boundary", "poisson_check_interval",
        "pressure_init_attempts", "pressure_init_accepts", "pressure_init_fallbacks",
        "pressure_init_max_iterations", "pressure_init_check_interval",
        "poisson_iters", "poisson_ms_per_step", "total_ms_per_step",
        "wall_ms_per_step", "wall_total_ms",
        "source_aware_hh_init", "source_aware_hh_scale",
        "pressure_rel_l2", "pressure_max_abs", "phase_mass_rel_error",
        "component_count", "reference_component_count",
        "pressure_init_file", "pressure_init_mode", "plt", "diagnostics", "stdout", "command",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(row)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: list[dict[str, object]] = []
    rewrite_summary = False
    if args.summary_csv.exists() and args.summary_csv.stat().st_size > 0:
        with args.summary_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            rewrite_summary = reader.fieldnames != keys
            if rewrite_summary:
                existing_rows = list(reader)
    if rewrite_summary:
        with args.summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(row)
    else:
        summary_exists = args.summary_csv.exists() and args.summary_csv.stat().st_size > 0
        with args.summary_csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            if not summary_exists:
                writer.writeheader()
            writer.writerow(row)

    print(f"wrote {out_csv}")
    print(f"updated {args.summary_csv}")
    if proc.returncode != 0:
        print(proc.stdout)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
