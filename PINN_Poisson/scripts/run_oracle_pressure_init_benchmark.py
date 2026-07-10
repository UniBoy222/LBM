#!/usr/bin/env python3
"""End-to-end oracle pressure-initializer benchmark without PyTorch."""

from __future__ import annotations

import argparse
import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(proc.stdout)
        raise SystemExit(proc.returncode)


def read_last_matching(path: Path, label: str) -> dict[str, str]:
    rows = [row for row in csv.DictReader(path.open()) if row.get("label") == label]
    if not rows:
        raise SystemExit(f"missing label {label} in {path}")
    return rows[-1]


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--poisson", choices=("split", "fused", "onepass", "scalar"), default="onepass")
    parser.add_argument("--pressure-boundary", choices=("split", "fused"), default="fused")
    parser.add_argument("--poisson-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--label", default=None)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.build:
        run(["make", "-C", str(root / "GPU"), "gpu"], root)

    base_label = args.label or f"oracle_{args.poisson}_step{args.steps}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    collect_cmd = [
        "python3", str(root / "PINN_Poisson" / "scripts" / "collect_reference_data.py"),
        "--steps", str(args.steps),
        "--sample-steps", str(args.steps),
        "--poisson", args.poisson,
        "--pressure-boundary", args.pressure_boundary,
        "--poisson-tolerance", str(args.poisson_tolerance),
        "--label", base_label,
    ]
    run(collect_cmd, root)

    sample_label = f"{base_label}_step{args.steps}"
    manifest = root / "PINN_Poisson" / "data" / "reference_manifest.csv"
    reference = read_last_matching(manifest, sample_label)
    gate_label = f"{base_label}_gate"
    gate_cmd = [
        "python3", str(root / "PINN_Poisson" / "scripts" / "run_pressure_init_gate.py"),
        "--steps", str(args.steps),
        "--poisson", args.poisson,
        "--pressure-boundary", args.pressure_boundary,
        "--poisson-tolerance", str(args.poisson_tolerance),
        "--pressure-init-file", reference["pressure_initializer"],
        "--pressure-init-mode", "absolute",
        "--reference-plt", reference["plt"],
        "--label", gate_label,
    ]
    run(gate_cmd, root)

    gate_csv = root / "PINN_Poisson" / "results" / "gates" / gate_label / "gate_summary.csv"
    gate = next(csv.DictReader(gate_csv.open()))

    def f(row: dict[str, str], key: str) -> float:
        value = row.get(key, "")
        return float(value) if value else float("nan")

    ref_iters = f(reference, "poisson_iters")
    gate_iters = f(gate, "poisson_iters")
    ref_ms = f(reference, "total_ms_per_step")
    gate_ms = f(gate, "total_ms_per_step")
    summary = {
        "label": base_label,
        "steps": args.steps,
        "poisson": args.poisson,
        "pressure_boundary": args.pressure_boundary,
        "reference_iters": ref_iters,
        "oracle_init_iters": gate_iters,
        "iter_reduction": (ref_iters - gate_iters) / ref_iters if ref_iters > 0 else "",
        "reference_ms_per_step": ref_ms,
        "oracle_init_ms_per_step": gate_ms,
        "speedup_vs_reference": ref_ms / gate_ms if gate_ms > 0 else "",
        "accepts": gate.get("pressure_init_accepts", ""),
        "fallbacks": gate.get("pressure_init_fallbacks", ""),
        "pressure_rel_l2": gate.get("pressure_rel_l2", ""),
        "phase_mass_rel_error": gate.get("phase_mass_rel_error", ""),
        "component_count": gate.get("component_count", ""),
        "reference_component_count": gate.get("reference_component_count", ""),
        "reference_plt": reference["plt"],
        "pressure_initializer": reference["pressure_initializer"],
        "gate_summary": str(gate_csv),
    }

    out = root / "PINN_Poisson" / "results" / "benchmarks" / "oracle_pressure_init_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(summary)
    exists = out.exists()
    with out.open("a", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=keys)
        if not exists:
            writer.writeheader()
        writer.writerow(summary)

    print(f"updated {out}")
    print(
        "oracle pressure-init: "
        f"iters {ref_iters:.1f}->{gate_iters:.1f}, "
        f"speedup={summary['speedup_vs_reference']}, "
        f"accepts={summary['accepts']} fallbacks={summary['fallbacks']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

