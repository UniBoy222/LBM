#!/usr/bin/env python3
"""Run held-out pressure-initializer generalization gates."""

from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path

from input_quality_gate import field_metrics, metric_ranges, score
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


def safe_label(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def default_cases(root: Path) -> list[Case]:
    config_dir = root / "PINN_Poisson" / "configs" / "generalization"
    return [
        Case("velocity_low", config_dir / "velocity_low.in"),
        Case("velocity_high", config_dir / "velocity_high.in"),
        Case("single_droplet", config_dir / "single_droplet.in"),
        Case("tanh_interface", config_dir / "tanh_interface.in"),
    ]


def run(cmd: list[str], cwd: Path, log: Path | None = None) -> str:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(proc.stdout)
    if proc.returncode != 0:
        print(proc.stdout)
        raise SystemExit(proc.returncode)
    return proc.stdout


def read_gate_row(path: Path) -> dict[str, object]:
    with path.open(newline="") as f:
        return dict(next(csv.DictReader(f)))


def run_gate(
    root: Path,
    args: argparse.Namespace,
    label: str,
    params: Path,
    pressure_init: Path | None,
    mode: str,
    reference_plt: Path | None,
    pressure_init_max_iterations: int = 0,
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
        cmd.extend(["--pressure-init-file", str(pressure_init), "--pressure-init-mode", mode])
        if pressure_init_max_iterations > 0:
            cmd.extend(["--pressure-init-max-iterations", str(pressure_init_max_iterations)])
    if reference_plt is not None:
        cmd.extend(["--reference-plt", str(reference_plt)])
    run(cmd, root)
    return read_gate_row(root / "PINN_Poisson" / "results" / "gates" / label / "gate_summary.csv")


def collect_pair(root: Path, args: argparse.Namespace, case: Case, params: Path) -> tuple[Path, Path, Path]:
    run_dir = root / "PINN_Poisson" / "data" / "generalization_runs" / case.label
    pair_dir = run_dir / "pairs"
    run_dir.mkdir(parents=True, exist_ok=True)
    diag_csv = root / "PINN_Poisson" / "results" / "gates" / f"generalization_{case.label}_pairs_poisson_diagnostics.csv"
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
    oracle_bin = root / "PINN_Poisson" / "data" / "pressure_init" / f"generalization_{case.label}_oracle_post_pressure.bin"
    write_pressure_initializer(
        oracle_bin,
        int(data["lx"]),
        int(data["ly"]),
        int(data["lz"]),
        fields["press"],  # type: ignore[index]
    )
    return pre_plt, post_plt, oracle_bin


def predict(root: Path, args: argparse.Namespace, case: Case, pre_plt: Path, method_label: str) -> Path:
    out = root / "PINN_Poisson" / "data" / "pressure_init" / f"generalization_{case.label}_{safe_label(method_label)}.bin"
    cmd = [
        str(root / "PINN_Poisson" / ".venv" / "bin" / "python"),
        str(root / "PINN_Poisson" / "scripts" / "predict_pressure_initializer.py"),
        str(pre_plt),
        "--model", str(args.model),
        "--out", str(out),
        "--mode", "absolute",
        "--device", args.device,
    ]
    run(cmd, root)
    return out


def normalize_row(
    case: Case,
    method: str,
    row: dict[str, object],
    params: Path,
    baseline_iters: float,
    baseline_ms: float,
    input_quality_accept: bool = True,
    input_quality_reasons: str = "",
) -> dict[str, object]:
    poisson_iters = float(row.get("poisson_iters", 0.0) or 0.0)
    total_ms = float(row.get("total_ms_per_step", 0.0) or 0.0)
    return {
        "case": case.label,
        "method": method,
        "params": str(params),
        "input_quality_accept": int(input_quality_accept),
        "input_quality_reasons": input_quality_reasons,
        "poisson_check_interval": row.get("poisson_check_interval", ""),
        "pressure_init_max_iterations": row.get("pressure_init_max_iterations", ""),
        "accepts": row.get("pressure_init_accepts", ""),
        "fallbacks": row.get("pressure_init_fallbacks", ""),
        "poisson_iters": row.get("poisson_iters", ""),
        "poisson_ms_per_step": row.get("poisson_ms_per_step", ""),
        "total_ms_per_step": row.get("total_ms_per_step", ""),
        "iter_reduction_vs_baseline": 1.0 - poisson_iters / baseline_iters if baseline_iters > 0.0 else "",
        "speedup_vs_baseline": baseline_ms / total_ms if total_ms > 0.0 else "",
        "pressure_rel_l2": row.get("pressure_rel_l2", ""),
        "pressure_max_abs": row.get("pressure_max_abs", ""),
        "phase_mass_rel_error": row.get("phase_mass_rel_error", ""),
        "component_count": row.get("component_count", ""),
        "reference_component_count": row.get("reference_component_count", ""),
        "pressure_init_file": row.get("pressure_init_file", ""),
        "plt": row.get("plt", ""),
        "diagnostics": row.get("diagnostics", ""),
        "stdout": row.get("stdout", ""),
    }


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=root / "GPU" / "lbm_gpu")
    parser.add_argument("--case", action="append", type=parse_case)
    parser.add_argument("--model", type=Path,
                        default=root / "PINN_Poisson" / "models" / "pressure_initializer_paired_abs_residual32.pt")
    parser.add_argument("--method-label", default=None)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--poisson", choices=("split", "fused", "onepass", "scalar"), default="onepass")
    parser.add_argument("--pressure-boundary", choices=("split", "fused"), default="fused")
    parser.add_argument("--poisson-check-interval", type=int, default=100)
    parser.add_argument("--poisson-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--pressure-init-max-iterations", type=int, default=0)
    parser.add_argument("--use-input-quality-gate", action="store_true")
    parser.add_argument("--input-quality-manifest", type=Path,
                        default=root / "PINN_Poisson" / "data" / "paired_manifest.csv")
    parser.add_argument("--input-quality-margin", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--label-prefix", default="generalization")
    parser.add_argument("--csv", type=Path,
                        default=root / "PINN_Poisson" / "results" / "benchmarks" / "generalization_pressure_init.csv")
    parser.add_argument("--gate-summary-csv", type=Path,
                        default=root / "PINN_Poisson" / "results" / "benchmarks" / "generalization_gate_summary.csv")
    args = parser.parse_args()

    cases = args.case or default_cases(root)
    method_label = args.method_label or args.model.stem.removeprefix("pressure_initializer_")
    quality_ranges = None
    if args.use_input_quality_gate:
        quality_ranges = metric_ranges(args.input_quality_manifest, args.input_quality_margin)

    rows: list[dict[str, object]] = []
    for case in cases:
        params = case.params if case.params.is_absolute() else root / case.params
        if not params.exists():
            raise SystemExit(f"missing params for {case.label}: {params}")

        pre_plt, _post_plt, oracle_bin = collect_pair(root, args, case, params)
        quality_accept = True
        quality_reasons = ""
        if quality_ranges is not None:
            quality_accept, reasons = score(field_metrics(pre_plt), quality_ranges)
            quality_reasons = "; ".join(reasons)

        baseline_label = f"{args.label_prefix}_{case.label}_baseline"
        baseline = run_gate(root, args, baseline_label, params, None, "absolute", None)
        reference_plt = Path(str(baseline["plt"]))
        baseline["pressure_rel_l2"] = 0.0
        baseline["pressure_max_abs"] = 0.0
        baseline["phase_mass_rel_error"] = 0.0
        add_reference_metrics(baseline, reference_plt, reference_plt)

        pinn_label = f"{args.label_prefix}_{case.label}_{safe_label(method_label)}"
        if quality_accept:
            pinn_bin = predict(root, args, case, pre_plt, method_label)
            pinn = run_gate(
                root, args, pinn_label, params, pinn_bin, "absolute", reference_plt,
                args.pressure_init_max_iterations,
            )
        else:
            pinn = dict(baseline)
            pinn["pressure_init_attempts"] = 0
            pinn["pressure_init_accepts"] = 0
            pinn["pressure_init_fallbacks"] = 0
            pinn["pressure_init_file"] = ""
            pinn["pressure_init_max_iterations"] = args.pressure_init_max_iterations
        oracle_label = f"{args.label_prefix}_{case.label}_oracle"
        oracle = run_gate(root, args, oracle_label, params, oracle_bin, "absolute", reference_plt)

        baseline_iters = float(baseline.get("poisson_iters", 0.0) or 0.0)
        baseline_ms = float(baseline.get("total_ms_per_step", 0.0) or 0.0)
        for method, row in [("baseline", baseline), (method_label, pinn), ("oracle", oracle)]:
            rows.append(normalize_row(
                case, method, row, params, baseline_iters, baseline_ms,
                quality_accept if method == method_label else True,
                quality_reasons if method == method_label else "",
            ))
        print(
            f"{case.label}: baseline iters={baseline.get('poisson_iters')} "
            f"pinn iters={pinn.get('poisson_iters')} fallback={pinn.get('pressure_init_fallbacks')} "
            f"rel={pinn.get('pressure_rel_l2')} quality={int(quality_accept)} "
            f"oracle iters={oracle.get('poisson_iters')}"
        )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "case", "method", "params", "input_quality_accept", "input_quality_reasons",
        "poisson_check_interval", "pressure_init_max_iterations",
        "accepts", "fallbacks",
        "poisson_iters", "poisson_ms_per_step", "total_ms_per_step",
        "iter_reduction_vs_baseline", "speedup_vs_baseline",
        "pressure_rel_l2", "pressure_max_abs", "phase_mass_rel_error",
        "component_count", "reference_component_count", "pressure_init_file",
        "plt", "diagnostics", "stdout",
    ]
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
