#!/usr/bin/env python3
"""Batch benchmark for the safe pressure-initializer pipeline."""

from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

from run_pressure_init_gate import add_reference_metrics
from tecplot_io import read_tecplot, write_pressure_initializer


@dataclass(frozen=True)
class Case:
    label: str
    params: Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_case(text: str) -> Case:
    parts = text.split(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("case must be LABEL:PARAMS")
    return Case(parts[0], Path(parts[1]))


def default_cases(root: Path, case_set: str) -> list[Case]:
    groups: list[Case] = []
    if case_set in {"generalization", "both"}:
        config_dir = root / "PINN_Poisson" / "configs" / "generalization"
        groups.extend([
            Case("velocity_low", config_dir / "velocity_low.in"),
            Case("velocity_high", config_dir / "velocity_high.in"),
            Case("single_droplet", config_dir / "single_droplet.in"),
            Case("tanh_interface", config_dir / "tanh_interface.in"),
        ])
    if case_set in {"validation", "both"}:
        config_dir = root / "PINN_Poisson" / "configs" / "validation"
        groups.extend([
            Case("velocity_extreme", config_dir / "velocity_extreme.in"),
            Case("offset_pair", config_dir / "offset_pair.in"),
            Case("single_droplet_shifted", config_dir / "single_droplet_shifted.in"),
            Case("tanh_interface_wide", config_dir / "tanh_interface_wide.in"),
        ])
    return groups


def run(cmd: list[str], cwd: Path, log: Path | None = None) -> str:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(proc.stdout)
    if proc.returncode != 0:
        print(proc.stdout)
        raise SystemExit(proc.returncode)
    return proc.stdout


def read_one_row(path: Path) -> dict[str, object]:
    with path.open(newline="") as f:
        return dict(next(csv.DictReader(f)))


def read_last_matching(path: Path, key: str, value: str) -> dict[str, object]:
    rows = list(csv.DictReader(path.open()))
    for row in reversed(rows):
        if row.get(key) == value:
            return dict(row)
    raise SystemExit(f"missing row {key}={value} in {path}")


def collect_pair(root: Path, args: argparse.Namespace, case: Case, params: Path) -> tuple[Path, Path, Path]:
    run_dir = root / "PINN_Poisson" / "data" / "safe_pipeline_runs" / case.label
    pair_dir = run_dir / "pairs"
    run_dir.mkdir(parents=True, exist_ok=True)
    diag_csv = root / "PINN_Poisson" / "results" / "gates" / f"safe_pipeline_{case.label}_pairs_poisson_diagnostics.csv"
    cmd = [
        str(args.exe),
        "--mode", "gpu",
        "--params", str(params),
        "--steps", str(args.steps),
        "--poisson", args.poisson,
        "--pressure-boundary", args.pressure_boundary,
        "--poisson-check-interval", str(args.poisson_check_interval),
        "--poisson-tolerance", str(args.poisson_tolerance),
        "--poisson-diagnostics", str(diag_csv),
        "--write-poisson-pairs",
        "--poisson-pair-dir", str(pair_dir),
        "--no-roofline",
    ]
    run(cmd, run_dir, run_dir / "stdout.log")
    pre_plt = pair_dir / "pre_poisson" / f"3D{args.steps:09d}.plt"
    post_plt = pair_dir / "post_poisson" / f"3D{args.steps:09d}.plt"
    if not pre_plt.exists() or not post_plt.exists():
        raise SystemExit(f"missing pair outputs for {case.label}")

    data = read_tecplot(post_plt)
    fields = data["fields"]  # type: ignore[assignment]
    oracle_bin = root / "PINN_Poisson" / "data" / "pressure_init" / f"safe_pipeline_{case.label}_oracle_post_pressure.bin"
    write_pressure_initializer(
        oracle_bin,
        int(data["lx"]),
        int(data["ly"]),
        int(data["lz"]),
        fields["press"],  # type: ignore[index]
    )
    return pre_plt, post_plt, oracle_bin


def run_gate(
    root: Path,
    args: argparse.Namespace,
    label: str,
    params: Path,
    pressure_init: Path | None,
    reference_plt: Path | None,
) -> dict[str, object]:
    cmd = [
        str(root / "PINN_Poisson" / ".venv" / "bin" / "python"),
        str(root / "PINN_Poisson" / "scripts" / "run_pressure_init_gate.py"),
        "--params", str(params),
        "--steps", str(args.steps),
        "--poisson", args.poisson,
        "--pressure-boundary", args.pressure_boundary,
        "--poisson-check-interval", str(args.poisson_check_interval),
        "--poisson-tolerance", str(args.poisson_tolerance),
        "--label", label,
        "--summary-csv", str(args.gate_summary_csv),
    ]
    if pressure_init is not None:
        cmd.extend(["--pressure-init-file", str(pressure_init), "--pressure-init-mode", "absolute"])
    if reference_plt is not None:
        cmd.extend(["--reference-plt", str(reference_plt)])
    run(cmd, root)
    return read_one_row(root / "PINN_Poisson" / "results" / "gates" / label / "gate_summary.csv")


def run_safe_pipeline(
    root: Path,
    args: argparse.Namespace,
    label: str,
    params: Path,
    pre_plt: Path,
    reference_plt: Path,
) -> dict[str, object]:
    cmd = [
        str(root / "PINN_Poisson" / ".venv" / "bin" / "python"),
        str(root / "PINN_Poisson" / "scripts" / "run_safe_pressure_initializer.py"),
        str(pre_plt),
        "--params", str(params),
        "--model", str(args.model),
        "--quality-manifest", str(args.quality_manifest),
        "--quality-margin", str(args.quality_margin),
        "--steps", str(args.steps),
        "--poisson", args.poisson,
        "--pressure-boundary", args.pressure_boundary,
        "--poisson-check-interval", str(args.poisson_check_interval),
        "--poisson-tolerance", str(args.poisson_tolerance),
        "--reference-plt", str(reference_plt),
        "--label", label,
        "--summary-csv", str(args.safe_summary_csv),
    ]
    if args.disable_input_quality_gate:
        cmd.append("--disable-input-quality-gate")
    if args.pressure_init_max_iterations > 0:
        cmd.extend(["--pressure-init-max-iterations", str(args.pressure_init_max_iterations)])
    run(cmd, root)
    return read_last_matching(args.safe_summary_csv, "safe_label", label)


def numeric(row: dict[str, object], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else 0.0


def normalize(
    case: Case,
    method: str,
    row: dict[str, object],
    baseline_iters: float,
    baseline_ms: float,
) -> dict[str, object]:
    poisson_iters = numeric(row, "poisson_iters")
    total_ms = numeric(row, "total_ms_per_step")
    return {
        "case": case.label,
        "method": method,
        "input_quality_accept": row.get("input_quality_accept", ""),
        "input_quality_reasons": row.get("input_quality_reasons", ""),
        "used_pressure_initializer": row.get("used_pressure_initializer", 1 if method == "oracle" else 0),
        "pressure_init_attempts": row.get("pressure_init_attempts", ""),
        "pressure_init_accepts": row.get("pressure_init_accepts", ""),
        "pressure_init_fallbacks": row.get("pressure_init_fallbacks", ""),
        "poisson_iters": row.get("poisson_iters", ""),
        "total_ms_per_step": row.get("total_ms_per_step", ""),
        "iter_reduction_vs_baseline": 1.0 - poisson_iters / baseline_iters if baseline_iters > 0.0 else "",
        "speedup_vs_baseline": baseline_ms / total_ms if total_ms > 0.0 else "",
        "pressure_rel_l2": row.get("pressure_rel_l2", ""),
        "pressure_max_abs": row.get("pressure_max_abs", ""),
        "phase_mass_rel_error": row.get("phase_mass_rel_error", ""),
        "component_count": row.get("component_count", ""),
        "reference_component_count": row.get("reference_component_count", ""),
        "plt": row.get("plt", ""),
        "diagnostics": row.get("diagnostics", ""),
        "stdout": row.get("stdout", ""),
    }


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=root / "GPU" / "lbm_gpu")
    parser.add_argument("--case", action="append", type=parse_case)
    parser.add_argument("--case-set", choices=("generalization", "validation", "both"), default="validation")
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
    parser.add_argument("--no-oracle", dest="include_oracle", action="store_false")
    parser.add_argument("--label-prefix", default="safe_pipeline")
    parser.add_argument("--csv", type=Path,
                        default=root / "PINN_Poisson" / "results" / "benchmarks" / "safe_pipeline_benchmark.csv")
    parser.add_argument("--safe-summary-csv", type=Path,
                        default=root / "PINN_Poisson" / "results" / "benchmarks" / "safe_pressure_initializer_batch.csv")
    parser.add_argument("--gate-summary-csv", type=Path,
                        default=root / "PINN_Poisson" / "results" / "benchmarks" / "safe_pipeline_gate_summary.csv")
    args = parser.parse_args()

    cases = args.case or default_cases(root, args.case_set)
    rows: list[dict[str, object]] = []
    for case in cases:
        params = case.params if case.params.is_absolute() else root / case.params
        if not params.exists():
            raise SystemExit(f"missing params for {case.label}: {params}")
        pre_plt, _post_plt, oracle_bin = collect_pair(root, args, case, params)

        baseline_label = f"{args.label_prefix}_{case.label}_baseline"
        baseline = run_gate(root, args, baseline_label, params, None, None)
        reference_plt = Path(str(baseline["plt"]))
        baseline["pressure_rel_l2"] = 0.0
        baseline["pressure_max_abs"] = 0.0
        baseline["phase_mass_rel_error"] = 0.0
        add_reference_metrics(baseline, reference_plt, reference_plt)

        safe_label_text = f"{args.label_prefix}_{case.label}_safe"
        safe = run_safe_pipeline(root, args, safe_label_text, params, pre_plt, reference_plt)

        baseline_iters = numeric(baseline, "poisson_iters")
        baseline_ms = numeric(baseline, "total_ms_per_step")
        rows.append(normalize(case, "baseline", baseline, baseline_iters, baseline_ms))
        rows.append(normalize(case, "safe_pipeline", safe, baseline_iters, baseline_ms))

        if args.include_oracle:
            oracle_label = f"{args.label_prefix}_{case.label}_oracle"
            oracle = run_gate(root, args, oracle_label, params, oracle_bin, reference_plt)
            rows.append(normalize(case, "oracle", oracle, baseline_iters, baseline_ms))

        print(
            f"{case.label}: baseline={baseline.get('poisson_iters')} "
            f"safe={safe.get('poisson_iters')} quality={safe.get('input_quality_accept')} "
            f"used={safe.get('used_pressure_initializer')} rel={safe.get('pressure_rel_l2')}"
        )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "case", "method", "input_quality_accept", "input_quality_reasons",
        "used_pressure_initializer", "pressure_init_attempts", "pressure_init_accepts",
        "pressure_init_fallbacks", "poisson_iters", "total_ms_per_step",
        "iter_reduction_vs_baseline", "speedup_vs_baseline",
        "pressure_rel_l2", "pressure_max_abs", "phase_mass_rel_error",
        "component_count", "reference_component_count", "plt", "diagnostics", "stdout",
    ]
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
