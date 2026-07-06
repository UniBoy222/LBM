#!/usr/bin/env python3
"""Compare Poisson algorithm candidates against a GPU onepass reference field."""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from poisson_algorithm_candidates import Candidate, default_candidates
from validation_metrics import read_tecplot


ITERS_RE = re.compile(r"Poisson iters:\s+([0-9eE+\-.]+)")
TOTAL_RE = re.compile(r"Total per step:\s+([0-9eE+\-.]+)")
MLUPS_RE = re.compile(r"MLUPS:\s+([0-9eE+\-.]+)")

FIELD_KEYS = ("fei", "rho", "u", "v", "w", "press")


@dataclass(frozen=True)
class RunResult:
    candidate: Candidate
    exit_code: int
    metrics: dict[str, float]
    fields: dict[str, list[float]]
    command: str
    plt: Path


def parse_perf(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if "Poisson iters:" in line:
            match = ITERS_RE.search(line)
            if match:
                out["poisson_iters"] = float(match.group(1))
        elif "Total per step:" in line:
            match = TOTAL_RE.search(line)
            if match:
                out["total_ms_per_step"] = float(match.group(1))
        elif "  MLUPS:" in line:
            match = MLUPS_RE.search(line)
            if match:
                out["mlups"] = float(match.group(1))
    return out


def candidate_args(candidate: Candidate) -> list[str]:
    args = [
        "--poisson", candidate.poisson,
        "--pressure-boundary", candidate.pressure_boundary,
    ]
    if candidate.scalar_source_scale is not None:
        args.extend(["--scalar-source-scale", str(candidate.scalar_source_scale)])
    if candidate.source_aware_hh_init:
        args.append("--source-aware-hh-init")
        args.extend(["--source-aware-hh-scale", str(candidate.source_aware_hh_scale)])
    if candidate.pressure_relax_scale != 1.0:
        args.extend(["--pressure-relax-scale", str(candidate.pressure_relax_scale)])
    if candidate.poisson_fixed_point_relax != 1.0:
        args.extend(["--poisson-fixed-point-relax", str(candidate.poisson_fixed_point_relax)])
    if candidate.poisson_anderson_m1:
        args.append("--poisson-anderson-m1")
        args.extend(["--poisson-anderson-beta-max", str(candidate.poisson_anderson_beta_max)])
    if candidate.poisson_two_grid_correction:
        args.append("--poisson-two-grid-correction")
        args.extend(["--poisson-two-grid-strength", str(candidate.poisson_two_grid_strength)])
    return args


def run_candidate(
    exe: Path,
    params: Path,
    steps: int,
    candidate: Candidate,
    poisson_tolerance: float,
    work_root: Path,
    keep_outputs: Path | None,
) -> RunResult:
    run_dir = work_root / candidate.name
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(exe.resolve()),
        "--mode", "gpu",
        "--params", str(params.resolve()),
        "--steps", str(steps),
        "--output-frequency", str(steps),
        "--write-output",
        "--no-roofline",
        "--poisson-check-interval", "100",
        "--poisson-tolerance", str(poisson_tolerance),
        *candidate_args(candidate),
    ]
    proc = subprocess.run(cmd, cwd=run_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    final_plt = run_dir / "out" / f"3D{steps:09d}.plt"
    if not final_plt.exists():
        raise RuntimeError(f"{candidate.name} did not produce {final_plt}\n{proc.stdout}")

    data = read_tecplot(final_plt)
    fields = data["fields"]  # type: ignore[assignment]
    copied_plt = final_plt
    if keep_outputs is not None:
        keep_outputs.mkdir(parents=True, exist_ok=True)
        copied_plt = keep_outputs / f"{candidate.name}_step{steps}.plt"
        shutil.copy2(final_plt, copied_plt)

    return RunResult(
        candidate=candidate,
        exit_code=proc.returncode,
        metrics=parse_perf(proc.stdout),
        fields={key: list(fields[key]) for key in FIELD_KEYS},  # type: ignore[index]
        command=" ".join(cmd),
        plt=copied_plt,
    )


def compare_fields(fields: dict[str, list[float]], ref: dict[str, list[float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in FIELD_KEYS:
        values = fields[key]
        refs = ref[key]
        if len(values) != len(refs):
            raise ValueError(f"field size mismatch for {key}: {len(values)} != {len(refs)}")
        sum_diff2 = 0.0
        sum_ref2 = 0.0
        max_abs = 0.0
        for value, ref_value in zip(values, refs):
            diff = value - ref_value
            sum_diff2 += diff * diff
            sum_ref2 += ref_value * ref_value
            max_abs = max(max_abs, abs(diff))
        out[f"{key}_max_abs"] = max_abs
        out[f"{key}_rel_l2"] = math.sqrt(sum_diff2 / sum_ref2) if sum_ref2 > 0.0 else math.sqrt(sum_diff2)
    return out


def field_pass(row: dict[str, object], accuracy_tol: float, pressure_accuracy_tol: float) -> bool:
    for key in FIELD_KEYS:
        tol = pressure_accuracy_tol if key == "press" else accuracy_tol
        try:
            value = float(row[f"{key}_rel_l2"])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(value) or value > tol:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=Path("./lbm_gpu"))
    parser.add_argument("--params", type=Path, default=Path("params_small.in"))
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--poisson-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--accuracy-tol", type=float, default=1.0e-4)
    parser.add_argument("--pressure-accuracy-tol", type=float, default=1.0e-4)
    parser.add_argument("--csv", type=Path, default=Path("poisson_reference_gate.csv"))
    parser.add_argument("--keep-outputs", type=Path, default=None)
    parser.add_argument("--candidates", default="all",
                        help="comma-separated candidate names, or all")
    args = parser.parse_args()

    if args.steps <= 0:
        raise SystemExit("--steps must be positive")

    all_candidates = default_candidates()
    if args.candidates != "all":
        wanted = {item.strip() for item in args.candidates.split(",") if item.strip()}
        candidates = [candidate for candidate in all_candidates if candidate.name in wanted]
        missing = sorted(wanted - {candidate.name for candidate in candidates})
        if missing:
            raise SystemExit(f"unknown candidates: {', '.join(missing)}")
    else:
        candidates = all_candidates

    reference = next((candidate for candidate in candidates if candidate.name == "onepass_reference"), None)
    if reference is None:
        reference = next(candidate for candidate in all_candidates if candidate.name == "onepass_reference")
        candidates = [reference, *candidates]

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="poisson_reference_gate_") as tmp:
        work_root = Path(tmp)
        ref_run = run_candidate(
            args.exe, args.params, args.steps, reference,
            args.poisson_tolerance, work_root, args.keep_outputs)
        baseline_ms = ref_run.metrics.get("total_ms_per_step", float("nan"))

        for candidate in candidates:
            run = ref_run if candidate.name == reference.name else run_candidate(
                args.exe, args.params, args.steps, candidate,
                args.poisson_tolerance, work_root, args.keep_outputs)
            errors = compare_fields(run.fields, ref_run.fields)
            total_ms = run.metrics.get("total_ms_per_step", float("nan"))
            speedup = baseline_ms / total_ms if math.isfinite(baseline_ms) and total_ms > 0.0 else ""
            row: dict[str, object] = {
                "candidate": candidate.name,
                "poisson": candidate.poisson,
                "steps": args.steps,
                "exit_code": run.exit_code,
                "reference_candidate": reference.name,
                "poisson_tolerance": args.poisson_tolerance,
                "reference_accuracy_pass": 0,
                "speedup_vs_reference": speedup,
                "total_ms_per_step": total_ms,
                "poisson_iters": run.metrics.get("poisson_iters", ""),
                "mlups": run.metrics.get("mlups", ""),
                "source_aware_hh_init": int(candidate.source_aware_hh_init),
                "source_aware_hh_scale": candidate.source_aware_hh_scale if candidate.source_aware_hh_init else "",
                "scalar_source_scale": candidate.scalar_source_scale if candidate.scalar_source_scale is not None else "",
                "pressure_relax_scale": candidate.pressure_relax_scale,
                "poisson_fixed_point_relax": candidate.poisson_fixed_point_relax,
                "poisson_anderson_m1": int(candidate.poisson_anderson_m1),
                "poisson_anderson_beta_max": candidate.poisson_anderson_beta_max if candidate.poisson_anderson_m1 else "",
                "poisson_two_grid_correction": int(candidate.poisson_two_grid_correction),
                "poisson_two_grid_strength": candidate.poisson_two_grid_strength if candidate.poisson_two_grid_correction else "",
                "saved_plt": str(run.plt) if args.keep_outputs is not None else "",
                "command": run.command,
            }
            row.update(errors)
            row["reference_accuracy_pass"] = int(field_pass(row, args.accuracy_tol, args.pressure_accuracy_tol))
            rows.append(row)
            print(
                f"{candidate.name}: ref_pass={row['reference_accuracy_pass']} "
                f"speedup={speedup if speedup != '' else 'NA'} "
                f"iters={run.metrics.get('poisson_iters', float('nan')):.1f} "
                f"press_rel={row.get('press_rel_l2', float('nan')):.3e}"
            )

    keys = [
        "candidate", "poisson", "steps", "exit_code", "reference_candidate",
        "poisson_tolerance", "reference_accuracy_pass", "speedup_vs_reference",
        "total_ms_per_step", "poisson_iters", "mlups",
        "source_aware_hh_init", "source_aware_hh_scale", "scalar_source_scale",
        "pressure_relax_scale", "poisson_fixed_point_relax",
        "poisson_anderson_m1", "poisson_anderson_beta_max",
        "poisson_two_grid_correction", "poisson_two_grid_strength",
    ]
    for key in FIELD_KEYS:
        keys.extend([f"{key}_max_abs", f"{key}_rel_l2"])
    keys.extend(["saved_plt", "command"])

    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
