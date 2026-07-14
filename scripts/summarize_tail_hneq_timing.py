#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


def quantile_type7(values, probability):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values):
    q1 = quantile_type7(values, 0.25)
    median = quantile_type7(values, 0.50)
    q3 = quantile_type7(values, 0.75)
    return {
        "values": values,
        "min": min(values),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": max(values),
        "iqr": q3 - q1,
    }


def read_run(path, expected_route):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20:
        raise RuntimeError(f"{path}: expected 20 rows, got {len(rows)}")
    if any(row["route"] != expected_route for row in rows):
        raise RuntimeError(f"{path}: route mismatch")
    return rows


def as_float(row, key):
    value = float(row[key])
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite {key}")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-dir", required=True, type=Path)
    parser.add_argument("--correctness-dir", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--per-step-output", required=True, type=Path)
    args = parser.parse_args()

    baseline_runs = []
    replay_runs = []
    for repeat in range(1, 6):
        baseline_runs.append(read_run(args.timing_dir / f"baseline_r{repeat}" / "steps.csv", "baseline"))
        replay_runs.append(read_run(args.timing_dir / f"replay_r{repeat}" / "steps.csv", "replay"))

    capture_correctness = read_run(args.correctness_dir / "capture" / "steps.csv", "capture")
    replay_correctness = read_run(args.correctness_dir / "replay" / "steps.csv", "replay")

    if list(args.timing_dir.glob("**/poisson_diagnostics.csv")):
        raise RuntimeError("benchmark timing contains forbidden poisson diagnostics I/O")

    baseline_totals = [sum(as_float(row, "e2e_ms") for row in rows) for rows in baseline_runs]
    replay_totals = [sum(as_float(row, "e2e_ms") for row in rows) for rows in replay_runs]
    replay_io_totals = [sum(as_float(row, "warm_start_io_ms") for row in rows) for rows in replay_runs]
    baseline_iterations = [sum(int(row["iterations"]) for row in rows) for rows in baseline_runs]
    replay_iterations = [sum(int(row["iterations"]) for row in rows) for rows in replay_runs]

    for rows in replay_runs:
        for row in rows:
            if not (
                int(row["iterations"]) == 100
                and int(row["fallback"]) == 0
                and int(row["warm_start_used"]) == 1
                and int(row["warm_start_first_check_converged"]) == 1
                and int(row["pressure_converged"]) == 1
                and int(row["fixed_point_converged"]) == 1
                and int(row["converged"]) == 1
                and int(row["finite"]) == 1
                and as_float(row, "residual") < 1.0e-3
                and as_float(row, "fixed_point_relative") < 1.0e-3
            ):
                raise RuntimeError("replay timing gate failed")

    baseline_total_stats = stats(baseline_totals)
    replay_total_stats = stats(replay_totals)
    gain = 1.0 - replay_total_stats["median"] / baseline_total_stats["median"]
    speedup = baseline_total_stats["median"] / replay_total_stats["median"]

    per_step = []
    for step in range(20):
        baseline_values = [as_float(rows[step], "e2e_ms") for rows in baseline_runs]
        replay_values = [as_float(rows[step], "e2e_ms") for rows in replay_runs]
        baseline_step = stats(baseline_values)
        replay_step = stats(replay_values)
        per_step.append({
            "step": step + 1,
            "baseline": baseline_step,
            "replay": replay_step,
            "gain_fraction": 1.0 - replay_step["median"] / baseline_step["median"],
            "non_degraded": replay_step["median"] <= baseline_step["median"],
        })

    correctness_fields = [
        "p_raw_rel_l2", "p_demeaned_rel_l2", "pressure_gradient_rel_l2",
        "pressure_correction_rel_l2", "velocity_rel_l2", "u_rel_l2", "v_rel_l2",
        "w_rel_l2", "rho_rel_l2", "fei_rel_l2", "mass_rel", "shape_mismatch_fraction",
        "shape_dice_error", "largest_component_rel", "centroid_distance_normalized",
        "h_moment_rel_l2", "h_p_consistency_max_abs", "h_equilibrium_max_abs", "max_field_rel",
    ]
    correctness = {
        "baseline_total_iterations": sum(int(row["iterations"]) for row in capture_correctness),
        "replay_total_iterations": sum(int(row["iterations"]) for row in replay_correctness),
        "first_check_count": sum(int(row["warm_start_first_check_converged"]) for row in replay_correctness),
        "fallback_count": sum(int(row["fallback"]) for row in replay_correctness),
        "finite_count": sum(int(row["finite"]) for row in replay_correctness),
        "max_pressure_residual": max(as_float(row, "residual") for row in replay_correctness),
        "max_fixed_point_residual": max(as_float(row, "fixed_point_relative") for row in replay_correctness),
        "max_restore_abs": max(as_float(row, "warm_start_restore_max_abs") for row in replay_correctness),
        "maxima": {
            field: max(as_float(row, field) for row in replay_correctness)
            for field in correctness_fields
        },
    }
    correctness["iteration_reduction_fraction"] = (
        1.0 - correctness["replay_total_iterations"] / correctness["baseline_total_iterations"]
    )
    correctness["gate_passed"] = (
        correctness["first_check_count"] == 20
        and correctness["fallback_count"] == 0
        and correctness["finite_count"] == 20
        and correctness["max_pressure_residual"] < 1.0e-3
        and correctness["max_fixed_point_residual"] < 1.0e-3
        and correctness["max_restore_abs"] <= 1.0e-12
        and correctness["maxima"]["max_field_rel"] <= 1.0e-10
    )

    summary = {
        "schema": "tail-hneq-warmstart-timing-v1",
        "warmup_runs_per_route": 1,
        "formal_repeats_per_route": 5,
        "quantiles": "Hyndman-Fan Type 7",
        "timer": "sum of 20 per-step performTimeStepGPU pre-call to post-synchronize intervals",
        "benchmark_io_contract": {
            "baseline_capture_disabled": True,
            "poisson_diagnostics_disabled": True,
            "replay_artifact_read_restore_included": True,
            "reference_validation_excluded_from_timing": True,
        },
        "baseline_total_e2e_ms": baseline_total_stats,
        "replay_total_e2e_ms": replay_total_stats,
        "replay_warm_start_io_ms": stats(replay_io_totals),
        "baseline_total_iterations": baseline_iterations,
        "replay_total_iterations": replay_iterations,
        "iteration_reduction_fraction": 1.0 - replay_iterations[0] / baseline_iterations[0],
        "e2e_reduction_fraction": gain,
        "speedup": speedup,
        "per_step_non_degraded_count": sum(item["non_degraded"] for item in per_step),
        "all_20_steps_non_degraded": all(item["non_degraded"] for item in per_step),
        "all_replay_gates_passed": True,
        "correctness_20_step": correctness,
    }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with args.json_output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with args.per_step_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["step", "baseline_median_ms", "baseline_iqr_ms", "replay_median_ms", "replay_iqr_ms", "gain_fraction", "non_degraded"])
        for item in per_step:
            writer.writerow([
                item["step"], item["baseline"]["median"], item["baseline"]["iqr"],
                item["replay"]["median"], item["replay"]["iqr"],
                item["gain_fraction"], int(item["non_degraded"]),
            ])


if __name__ == "__main__":
    main()
