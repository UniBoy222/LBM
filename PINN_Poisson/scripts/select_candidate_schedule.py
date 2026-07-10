#!/usr/bin/env python3
"""Build a candidate schedule from closed-loop candidate scan results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ERROR_KEYS = (
    "pressure_rel_l2",
    "rho_rel_l2",
    "fei_rel_l2",
    "phase_mass_rel_error",
)


def as_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value == "":
        return 0
    return int(float(value))


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def combined_delta(row: dict[str, str]) -> int:
    return as_int(row, "iter_delta_step") + as_int(row, "future_until_next_iter_delta")


def parse_steps(text: str) -> list[int]:
    steps: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            steps.append(int(item))
    return sorted(dict.fromkeys(steps))


def read_steps_file(path: Path | None) -> list[int]:
    if path is None:
        return []
    return parse_steps(path.read_text().replace("\n", ","))


def accuracy_reasons(row: dict[str, str], args: argparse.Namespace) -> list[str]:
    if as_int(row, "field_error_available") != 1:
        return []
    thresholds = {
        "pressure_rel_l2": args.max_pressure_rel_l2,
        "rho_rel_l2": args.max_rho_rel_l2,
        "fei_rel_l2": args.max_fei_rel_l2,
        "phase_mass_rel_error": args.max_phase_mass_rel_error,
    }
    reasons: list[str] = []
    for key, threshold in thresholds.items():
        value = as_float(row, key)
        if threshold is not None and value is not None and value > threshold:
            reasons.append(f"{key}>{threshold:g}")
    if args.require_field_health and row.get("field_health_pass") not in ("", "1"):
        reasons.append("field_health_fail")
    return reasons


def in_accuracy_backoff(step: int, violations: list[tuple[int, str]], backoff_steps: int) -> str:
    if backoff_steps <= 0:
        return ""
    for violation_step, reason in violations:
        if violation_step - backoff_steps < step <= violation_step:
            return f"accuracy_backoff@{violation_step}:{reason}"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_csv", type=Path)
    parser.add_argument("--min-combined-delta", type=int, default=1,
                        help="select candidates with step_delta + until_next_delta >= this")
    parser.add_argument("--hard-positive-delta", type=int, default=300,
                        help="export strong positive candidates as training hard cases")
    parser.add_argument("--min-candidate-step", type=int, default=1)
    parser.add_argument("--max-candidate-step", type=int, default=0,
                        help="0 keeps all steps from the scan")
    parser.add_argument("--max-pressure-rel-l2", type=float, default=None)
    parser.add_argument("--max-rho-rel-l2", type=float, default=None)
    parser.add_argument("--max-fei-rel-l2", type=float, default=None)
    parser.add_argument("--max-phase-mass-rel-error", type=float, default=None)
    parser.add_argument("--require-field-health", action="store_true")
    parser.add_argument("--accuracy-backoff-steps", type=int, default=0,
                        help="reject scanned candidates in (violation_step-backoff, violation_step]")
    parser.add_argument("--stop-after-accuracy-violation", action="store_true",
                        help="reject scanned candidates after the first checkpoint violating an accuracy threshold")
    parser.add_argument("--extra-steps", default="",
                        help="comma-separated steps to append after filtering, e.g. sparse validation anchors")
    parser.add_argument("--extra-steps-file", type=Path, default=None)
    parser.add_argument("--out-steps", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--hard-cases-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.candidate_csv.open()))
    accuracy_violations: list[tuple[int, str]] = []
    for row in rows:
        reasons = accuracy_reasons(row, args)
        if reasons:
            accuracy_violations.append((as_int(row, "candidate_step"), ";".join(reasons)))
    first_accuracy_violation = accuracy_violations[0][0] if accuracy_violations else 0

    selected: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    hard_cases: list[dict[str, object]] = []

    for row in rows:
        step = as_int(row, "candidate_step")
        combined = combined_delta(row)
        accepted = as_int(row, "pressure_init_accept_step") == 1
        fallback = as_int(row, "pressure_init_fallback_step") == 1
        record: dict[str, object] = {
            "candidate_step": step,
            "combined_iter_delta": combined,
            "iter_delta_step": as_int(row, "iter_delta_step"),
            "future_until_next_iter_delta": as_int(row, "future_until_next_iter_delta"),
            "baseline_iter": as_int(row, "baseline_iter"),
            "safe_iter": as_int(row, "safe_iter"),
            "accepted": int(accepted),
            "fallback": int(fallback),
            "field_error_available": as_int(row, "field_error_available"),
            "pressure_rel_l2": row.get("pressure_rel_l2", ""),
            "rho_rel_l2": row.get("rho_rel_l2", ""),
            "fei_rel_l2": row.get("fei_rel_l2", ""),
            "phase_mass_rel_error": row.get("phase_mass_rel_error", ""),
            "field_health_pass": row.get("field_health_pass", ""),
            "input_plt": row.get("input_plt", ""),
            "pressure_init_file": row.get("pressure_init_file", ""),
        }
        reject_reasons: list[str] = []
        if not accepted:
            reject_reasons.append("not_accepted")
        if fallback:
            reject_reasons.append("fallback")
        if combined < args.min_combined_delta:
            reject_reasons.append(f"combined_delta<{args.min_combined_delta}")
        if step < args.min_candidate_step:
            reject_reasons.append(f"step<{args.min_candidate_step}")
        if args.max_candidate_step > 0 and step > args.max_candidate_step:
            reject_reasons.append(f"step>{args.max_candidate_step}")
        direct_accuracy_reasons = accuracy_reasons(row, args)
        if direct_accuracy_reasons:
            reject_reasons.append("accuracy_violation:" + ";".join(direct_accuracy_reasons))
        backoff_reason = in_accuracy_backoff(step, accuracy_violations, args.accuracy_backoff_steps)
        if backoff_reason:
            reject_reasons.append(backoff_reason)
        if args.stop_after_accuracy_violation and first_accuracy_violation and step > first_accuracy_violation:
            reject_reasons.append(f"after_accuracy_violation@{first_accuracy_violation}")

        record["reject_reason"] = ";".join(reject_reasons)
        if reject_reasons:
            rejected.append(record)
        else:
            selected.append(record)

        if combined < 0 or combined >= args.hard_positive_delta or fallback or direct_accuracy_reasons or backoff_reason:
            hard_case_type = "negative" if combined < 0 else ("fallback" if fallback else "strong_positive")
            if direct_accuracy_reasons:
                hard_case_type = "accuracy_violation"
            elif backoff_reason:
                hard_case_type = "accuracy_backoff"
            hard_cases.append({
                **record,
                "hard_case_type": hard_case_type,
            })

    extra_steps = parse_steps(args.extra_steps) + read_steps_file(args.extra_steps_file)
    selected_by_step = {int(row["candidate_step"]): row for row in selected}
    for step in extra_steps:
        if step not in selected_by_step:
            selected_by_step[step] = {
                "candidate_step": step,
                "combined_iter_delta": "",
                "iter_delta_step": "",
                "future_until_next_iter_delta": "",
                "baseline_iter": "",
                "safe_iter": "",
                "accepted": "",
                "fallback": "",
                "field_error_available": "",
                "pressure_rel_l2": "",
                "rho_rel_l2": "",
                "fei_rel_l2": "",
                "phase_mass_rel_error": "",
                "field_health_pass": "",
                "input_plt": "",
                "pressure_init_file": "",
                "reject_reason": "extra_step",
            }
    selected = [selected_by_step[step] for step in sorted(selected_by_step)]
    selected_steps = [int(row["candidate_step"]) for row in selected]
    args.out_steps.parent.mkdir(parents=True, exist_ok=True)
    args.out_steps.write_text(",".join(str(step) for step in selected_steps) + "\n")

    keys = [
        "candidate_step", "selected", "combined_iter_delta",
        "iter_delta_step", "future_until_next_iter_delta",
        "baseline_iter", "safe_iter", "accepted", "fallback",
        "field_error_available", "pressure_rel_l2", "rho_rel_l2", "fei_rel_l2",
        "phase_mass_rel_error", "field_health_pass", "reject_reason",
        "input_plt", "pressure_init_file",
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in selected:
            writer.writerow({**row, "selected": 1})
        for row in rejected:
            writer.writerow({**row, "selected": 0})

    hard_keys = [
        "candidate_step", "hard_case_type", "combined_iter_delta",
        "iter_delta_step", "future_until_next_iter_delta",
        "baseline_iter", "safe_iter", "accepted", "fallback",
        "field_error_available", "pressure_rel_l2", "rho_rel_l2", "fei_rel_l2",
        "phase_mass_rel_error", "field_health_pass", "reject_reason",
        "input_plt", "pressure_init_file",
    ]
    args.hard_cases_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.hard_cases_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=hard_keys)
        writer.writeheader()
        writer.writerows(hard_cases)

    print(f"selected {len(selected)}/{len(rows)} candidates")
    print(f"rejected {len(rejected)} candidates")
    if extra_steps:
        print(f"extra_steps {len(sorted(set(extra_steps)))}")
    if accuracy_violations:
        print(f"accuracy_violations {len(accuracy_violations)} first={first_accuracy_violation}")
    print(f"hard_cases {len(hard_cases)}")
    print(f"wrote {args.out_steps}")
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.hard_cases_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
