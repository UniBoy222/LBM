#!/usr/bin/env python3
"""Multi-step replay benchmark for the safe pressure initializer.

This is not a closed-loop Python-in-the-solver rollout. It generates PINN
initializers from a baseline pre-Poisson trajectory, then replays those
initializers step-by-step through the GPU solver.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from input_quality_gate import field_metrics, metric_ranges, score
from run_pressure_init_gate import parse_perf, rel_l2_and_max
from tecplot_io import read_tecplot, write_pressure_initializer


FIELDS = ("u", "v", "w", "rho", "fei", "press")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def step_name(step: int, suffix: str) -> str:
    return f"3D{step:09d}.{suffix}"


def parse_steps(text: str | None, final_step: int) -> list[int]:
    if text is None or text.strip() == "":
        return list(range(1, final_step + 1))
    out: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        step = int(item)
        if step <= 0 or step > final_step:
            raise ValueError(f"step {step} outside [1, {final_step}]")
        out.append(step)
    return sorted(dict.fromkeys(out))


def run(cmd: list[str], cwd: Path, log: Path | None = None) -> str:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(proc.stdout)
    if proc.returncode != 0:
        print(proc.stdout)
        raise SystemExit(proc.returncode)
    return proc.stdout


def run_solver(
    args: argparse.Namespace,
    root: Path,
    run_dir: Path,
    diagnostics: Path,
    output_steps: list[int],
    pressure_init_dir: Path | None = None,
    pair_dir: Path | None = None,
    source_aware_hh_init: bool = False,
) -> dict[str, object]:
    cmd = [
        str(args.exe.resolve()),
        "--mode", "gpu",
        "--params", str(args.params.resolve()),
        "--steps", str(args.steps),
        "--poisson", args.poisson,
        "--pressure-boundary", args.pressure_boundary,
        "--poisson-check-interval", str(args.poisson_check_interval),
        "--poisson-tolerance", str(args.poisson_tolerance),
        "--poisson-diagnostics", str(diagnostics.resolve()),
        "--write-output",
        "--output-steps", ",".join(str(step) for step in output_steps),
        "--no-roofline",
    ]
    if pressure_init_dir is not None:
        cmd.extend(["--pressure-init-dir", str(pressure_init_dir.resolve())])
        cmd.extend(["--pressure-init-mode", "absolute"])
        if args.pressure_init_max_iterations > 0:
            cmd.extend(["--pressure-init-max-iterations", str(args.pressure_init_max_iterations)])
        if args.pressure_init_check_interval > 0:
            cmd.extend(["--pressure-init-check-interval", str(args.pressure_init_check_interval)])
    if source_aware_hh_init:
        cmd.extend(["--source-aware-hh-init", "--source-aware-hh-scale", str(args.source_aware_hh_scale)])
    if pair_dir is not None:
        cmd.extend(["--write-poisson-pairs", "--poisson-pair-dir", str(pair_dir.resolve())])

    run_dir.mkdir(parents=True, exist_ok=True)
    stdout = run_dir / "stdout.log"
    text = run(cmd, run_dir, stdout)
    perf = parse_perf(text)
    perf.update({
        "steps": args.steps,
        "diagnostics": str(diagnostics),
        "stdout": str(stdout),
        "command": " ".join(cmd),
    })
    return perf


def parse_bool(text: object) -> bool:
    return str(text).strip().lower() in {"1", "true", "yes"}


def parse_step_diagnostics(path: Path) -> dict[int, dict[str, object]]:
    if not path.exists():
        return {}
    rows_by_step: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows_by_step.setdefault(int(row["step"]), []).append(row)

    out: dict[int, dict[str, object]] = {}
    for step, rows in rows_by_step.items():
        total_iters = 0
        segments = 0
        converged_segments = 0
        current_last_iter = 0
        current_converged = False
        previous_iter: int | None = None

        def finish_segment() -> None:
            nonlocal total_iters, segments, converged_segments, current_last_iter, current_converged
            if current_last_iter <= 0:
                return
            total_iters += current_last_iter
            segments += 1
            if current_converged:
                converged_segments += 1
            current_last_iter = 0
            current_converged = False

        for row in rows:
            iteration = int(float(row["iteration"]))
            converged = parse_bool(row["converged"])
            if previous_iter is not None and iteration <= previous_iter:
                finish_segment()
            current_last_iter = iteration
            current_converged = current_converged or converged
            previous_iter = iteration
            if converged:
                finish_segment()
                previous_iter = None
        finish_segment()
        out[step] = {
            "poisson_iters_step": total_iters,
            "poisson_segments_step": segments,
            "poisson_converged_segments_step": converged_segments,
            "poisson_converged_step": int(converged_segments > 0),
        }
    return out


def compare_fields(candidate_plt: Path, reference_plt: Path) -> dict[str, object]:
    cand = read_tecplot(candidate_plt)
    ref = read_tecplot(reference_plt)
    if (cand["lx"], cand["ly"], cand["lz"]) != (ref["lx"], ref["ly"], ref["lz"]):
        raise ValueError("candidate/reference grid dimensions do not match")

    cand_fields = cand["fields"]  # type: ignore[assignment]
    ref_fields = ref["fields"]  # type: ignore[assignment]
    out: dict[str, object] = {"field_health_pass": 1, "field_health_reasons": ""}
    health_reasons: list[str] = []
    for owner, fields in (("candidate", cand_fields), ("reference", ref_fields)):
        for field in FIELDS:
            values = fields[field]  # type: ignore[index]
            bad = sum(1 for value in values if not math.isfinite(value))
            if bad:
                health_reasons.append(f"{owner}.{field} nonfinite={bad}")
    if health_reasons:
        out["field_health_pass"] = 0
        out["field_health_reasons"] = "; ".join(health_reasons)
        for field in FIELDS:
            prefix = "pressure" if field == "press" else field
            out[f"{prefix}_rel_l2"] = math.inf
            out[f"{prefix}_max_abs"] = math.inf
        out["phase_mass_rel_error"] = math.inf
        out["component_count"] = ""
        out["reference_component_count"] = ""
        for key in ("interface_fraction", "mid_fraction", "liquid_fraction", "fei_grad_mean", "fei_grad_max"):
            out[f"{key}_abs_error"] = math.inf
        return out

    for field in FIELDS:
        rel, max_abs = rel_l2_and_max(cand_fields[field], ref_fields[field])  # type: ignore[index]
        prefix = "pressure" if field == "press" else field
        out[f"{prefix}_rel_l2"] = rel
        out[f"{prefix}_max_abs"] = max_abs

    ref_fei = ref_fields["fei"]  # type: ignore[index]
    cand_fei = cand_fields["fei"]  # type: ignore[index]
    ref_mass = math.fsum(ref_fei)
    cand_mass = math.fsum(cand_fei)
    out["phase_mass_rel_error"] = (
        abs(cand_mass - ref_mass) / abs(ref_mass) if ref_mass else abs(cand_mass - ref_mass)
    )

    cand_metrics = field_metrics(candidate_plt)
    ref_metrics = field_metrics(reference_plt)
    out["component_count"] = cand_metrics["component_count"]
    out["reference_component_count"] = ref_metrics["component_count"]
    for key in ("interface_fraction", "mid_fraction", "liquid_fraction", "fei_grad_mean", "fei_grad_max"):
        out[f"{key}_abs_error"] = abs(float(cand_metrics[key]) - float(ref_metrics[key]))
    return out


def write_oracle_schedule(pair_dir: Path, schedule_dir: Path, steps: list[int]) -> None:
    schedule_dir.mkdir(parents=True, exist_ok=True)
    for step in steps:
        post_plt = pair_dir / "post_poisson" / step_name(step, "plt")
        data = read_tecplot(post_plt)
        fields = data["fields"]  # type: ignore[assignment]
        write_pressure_initializer(
            schedule_dir / step_name(step, "bin"),
            int(data["lx"]),
            int(data["ly"]),
            int(data["lz"]),
            fields["press"],  # type: ignore[index]
        )


def build_safe_schedule(
    args: argparse.Namespace,
    root: Path,
    pair_dir: Path,
    schedule_dir: Path,
    output_steps: list[int],
) -> dict[int, dict[str, object]]:
    ranges = metric_ranges(args.quality_manifest, args.quality_margin)
    python = root / "PINN_Poisson" / ".venv" / "bin" / "python"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    per_step: dict[int, dict[str, object]] = {}
    for step in output_steps:
        pre_plt = pair_dir / "pre_poisson" / step_name(step, "plt")
        if not pre_plt.exists():
            raise SystemExit(f"missing pre-Poisson pair: {pre_plt}")
        quality_accept = True
        quality_reasons = ""
        if not args.disable_input_quality_gate:
            quality_accept, reasons = score(field_metrics(pre_plt), ranges)
            quality_reasons = "; ".join(reasons)

        pressure_bin = schedule_dir / step_name(step, "bin")
        used = False
        if quality_accept:
            run([
                str(python),
                str(root / "PINN_Poisson" / "scripts" / "predict_pressure_initializer.py"),
                str(pre_plt),
                "--model", str(args.model),
                "--out", str(pressure_bin),
                "--mode", "absolute",
                "--device", args.device,
            ], root)
            used = True

        per_step[step] = {
            "input_plt": str(pre_plt),
            "input_quality_accept": int(quality_accept),
            "input_quality_reasons": quality_reasons,
            "used_pressure_initializer": int(used),
            "pressure_init_file": str(pressure_bin) if used else "",
        }
    return per_step


def method_rows(
    label: str,
    method: str,
    steps: list[int],
    method_dir: Path,
    reference_dir: Path,
    perf: dict[str, object],
    baseline_perf: dict[str, object],
    per_step: dict[int, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    baseline_iters = float(baseline_perf.get("poisson_iters", 0.0) or 0.0)
    method_iters = float(perf.get("poisson_iters", 0.0) or 0.0)
    baseline_ms = float(baseline_perf.get("total_ms_per_step", 0.0) or 0.0)
    method_ms = float(perf.get("total_ms_per_step", 0.0) or 0.0)
    baseline_wall_ms = float(baseline_perf.get("wall_ms_per_step", baseline_ms) or 0.0)
    method_wall_ms = float(perf.get("wall_ms_per_step", method_ms) or 0.0)
    run_steps = int(perf.get("steps", max(steps)) or max(steps))
    step_diag = parse_step_diagnostics(Path(str(perf.get("diagnostics", ""))))
    rows: list[dict[str, object]] = []
    for step in steps:
        candidate_plt = method_dir / "out" / step_name(step, "plt")
        reference_plt = reference_dir / "out" / step_name(step, "plt")
        step_meta = dict(per_step.get(step, {}) if per_step is not None else {})
        step_perf = step_diag.get(step, {})
        used_pressure_initializer = int(step_meta.get("used_pressure_initializer", 0) or 0)
        row: dict[str, object] = {
            "label": label,
            "method": method,
            "step": step,
            "steps": run_steps,
            "poisson_iters_avg": method_iters,
            "poisson_iters_total_est": method_iters * run_steps,
            "total_ms_per_step": method_ms,
            "total_ms_est": method_ms * run_steps,
            "wall_ms_per_step": method_wall_ms,
            "wall_total_ms": float(perf.get("wall_total_ms", method_wall_ms * run_steps) or 0.0),
            "pressure_init_attempts": perf.get("pressure_init_attempts", ""),
            "pressure_init_accepts": perf.get("pressure_init_accepts", ""),
            "pressure_init_fallbacks": perf.get("pressure_init_fallbacks", ""),
            "pressure_init_accept_step": int(used_pressure_initializer == 1 and int(step_perf.get("poisson_segments_step", 0) or 0) == 1 and int(step_perf.get("poisson_converged_step", 0) or 0) == 1),
            "pressure_init_fallback_step": int(used_pressure_initializer == 1 and int(step_perf.get("poisson_segments_step", 0) or 0) > 1),
            "iter_reduction_vs_baseline": 1.0 - method_iters / baseline_iters if baseline_iters else "",
            "speedup_vs_baseline": baseline_ms / method_ms if method_ms else "",
            "wall_speedup_vs_baseline": baseline_wall_ms / method_wall_ms if method_wall_ms else "",
            "plt": str(candidate_plt),
            "reference_plt": str(reference_plt),
            "diagnostics": perf.get("diagnostics", ""),
            "stdout": perf.get("stdout", ""),
        }
        row.update(step_meta)
        row.update(step_perf)
        row.update(compare_fields(candidate_plt, reference_plt))
        rows.append(row)
    return rows


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=root / "GPU" / "lbm_gpu")
    parser.add_argument("--params", type=Path, default=root / "GPU" / "params_small.in")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output-steps", default=None)
    parser.add_argument("--poisson", choices=("split", "fused", "onepass", "scalar"), default="onepass")
    parser.add_argument("--pressure-boundary", choices=("split", "fused"), default="fused")
    parser.add_argument("--poisson-check-interval", type=int, default=100)
    parser.add_argument("--poisson-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--pressure-init-max-iterations", type=int, default=0)
    parser.add_argument("--pressure-init-check-interval", type=int, default=0,
                        help="0 uses --poisson-check-interval for the PINN attempt")
    parser.add_argument("--source-aware-hh-init", action="store_true")
    parser.add_argument("--source-aware-hh-scale", type=float, default=1.0)
    parser.add_argument("--model", type=Path,
                        default=root / "PINN_Poisson" / "models" / "pressure_initializer_augmented_abs_residual32.pt")
    parser.add_argument("--quality-manifest", type=Path,
                        default=root / "PINN_Poisson" / "data" / "augmented_manifest.csv")
    parser.add_argument("--quality-margin", type=float, default=0.25)
    parser.add_argument("--disable-input-quality-gate", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-oracle", dest="include_oracle", action="store_false")
    parser.add_argument("--label", default=None)
    parser.add_argument("--csv", type=Path,
                        default=root / "PINN_Poisson" / "results" / "benchmarks" / "multistep_safe_replay.csv")
    args = parser.parse_args()

    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    output_steps = parse_steps(args.output_steps, args.steps)
    label = args.label or f"multistep_safe_replay_step{args.steps}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    work_dir = root / "PINN_Poisson" / "data" / "multistep_safe_replay" / label

    baseline_dir = work_dir / "baseline"
    baseline_pair_dir = baseline_dir / "pairs"
    baseline_perf = run_solver(
        args,
        root,
        baseline_dir,
        baseline_dir / "poisson_diagnostics.csv",
        output_steps,
        pair_dir=baseline_pair_dir,
    )

    safe_schedule_dir = work_dir / "safe_pressure_init_schedule"
    safe_step_meta = build_safe_schedule(args, root, baseline_pair_dir, safe_schedule_dir, output_steps)
    safe_dir = work_dir / "safe_replay"
    safe_perf = run_solver(
        args,
        root,
        safe_dir,
        safe_dir / "poisson_diagnostics.csv",
        output_steps,
        pressure_init_dir=safe_schedule_dir,
        source_aware_hh_init=args.source_aware_hh_init,
    )

    rows: list[dict[str, object]] = []
    rows.extend(method_rows(label, "baseline", output_steps, baseline_dir, baseline_dir, baseline_perf, baseline_perf))
    rows.extend(method_rows(label, "safe_replay", output_steps, safe_dir, baseline_dir, safe_perf, baseline_perf, safe_step_meta))

    if args.include_oracle:
        oracle_schedule_dir = work_dir / "oracle_pressure_init_schedule"
        write_oracle_schedule(baseline_pair_dir, oracle_schedule_dir, output_steps)
        oracle_dir = work_dir / "oracle_replay"
        oracle_perf = run_solver(
            args,
            root,
            oracle_dir,
            oracle_dir / "poisson_diagnostics.csv",
            output_steps,
            pressure_init_dir=oracle_schedule_dir,
            source_aware_hh_init=args.source_aware_hh_init,
        )
        oracle_meta = {
            step: {
                "input_plt": str(baseline_pair_dir / "pre_poisson" / step_name(step, "plt")),
                "input_quality_accept": "",
                "input_quality_reasons": "",
                "used_pressure_initializer": 1,
                "pressure_init_file": str(oracle_schedule_dir / step_name(step, "bin")),
            }
            for step in output_steps
        }
        rows.extend(method_rows(label, "oracle_replay", output_steps, oracle_dir, baseline_dir, oracle_perf, baseline_perf, oracle_meta))

    keys = [
        "label", "method", "step", "steps",
        "input_quality_accept", "input_quality_reasons", "used_pressure_initializer",
        "pressure_init_attempts", "pressure_init_accepts", "pressure_init_fallbacks",
        "pressure_init_accept_step", "pressure_init_fallback_step",
        "poisson_iters_avg", "poisson_iters_total_est", "total_ms_per_step", "total_ms_est",
        "wall_ms_per_step", "wall_total_ms",
        "poisson_iters_step", "poisson_segments_step", "poisson_converged_segments_step", "poisson_converged_step",
        "iter_reduction_vs_baseline", "speedup_vs_baseline", "wall_speedup_vs_baseline",
        "rho_rel_l2", "rho_max_abs", "fei_rel_l2", "fei_max_abs",
        "u_rel_l2", "u_max_abs", "v_rel_l2", "v_max_abs", "w_rel_l2", "w_max_abs",
        "pressure_rel_l2", "pressure_max_abs", "phase_mass_rel_error",
        "field_health_pass", "field_health_reasons",
        "component_count", "reference_component_count",
        "interface_fraction_abs_error", "mid_fraction_abs_error", "liquid_fraction_abs_error",
        "fei_grad_mean_abs_error", "fei_grad_max_abs_error",
        "input_plt", "pressure_init_file", "plt", "reference_plt", "diagnostics", "stdout",
    ]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    safe_accepts = sum(int(meta["used_pressure_initializer"]) for meta in safe_step_meta.values())
    print(f"{label}: baseline_iters={baseline_perf.get('poisson_iters')} safe_iters={safe_perf.get('poisson_iters')} "
          f"safe_used_steps={safe_accepts}/{len(output_steps)} fallbacks={safe_perf.get('pressure_init_fallbacks', '')}")
    print(f"wrote {args.csv}")
    print(f"work_dir {work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
