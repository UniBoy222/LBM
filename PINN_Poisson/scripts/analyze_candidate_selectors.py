#!/usr/bin/env python3
"""Analyze PINN candidate scans and emit selector schedules."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


ERROR_KEYS = ("pressure_rel_l2", "rho_rel_l2", "fei_rel_l2", "phase_mass_rel_error")


def as_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    return int(float(value)) if value not in ("", None) else 0


def as_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else None


def fmt_steps(steps: list[int]) -> str:
    return ",".join(str(step) for step in steps) + ("\n" if steps else "")


@dataclass(frozen=True)
class Profile:
    name: str
    min_score: int
    min_step_delta: int
    min_f20: int
    min_f50: int
    min_f100: int
    min_until: int
    max_pressure: float
    max_rho: float
    max_fei: float
    max_d_pressure: float
    max_d_rho: float
    max_d_fei: float
    min_gap: int
    max_count_per_500: int


PROFILES = (
    Profile("conservative", 250, 50, 0, 0, 0, 0, 0.012, 5.0e-4, 3.0e-4, 0.006, 3.0e-4, 2.0e-4, 100, 3),
    Profile("balanced", 200, 50, 0, 0, -50, -50, 0.020, 1.0e-3, 6.0e-4, 0.012, 6.0e-4, 4.0e-4, 50, 5),
    Profile("aggressive", 100, 50, -100, -100, -150, -150, 0.040, 1.5e-3, 9.0e-4, 0.025, 1.0e-3, 7.0e-4, 20, 10),
)


def source_name(path: Path) -> str:
    return path.stem.removesuffix("_candidates")


def annotated_rows(path: Path) -> list[dict[str, object]]:
    rows = list(csv.DictReader(path.open()))
    field_points: list[tuple[int, dict[str, float]]] = []
    for row in rows:
        if as_int(row, "field_error_available") != 1:
            continue
        errors = {key: as_float(row, key) for key in ERROR_KEYS}
        if all(value is not None for value in errors.values()):
            field_points.append((as_int(row, "candidate_step"), errors))  # type: ignore[arg-type]

    out: list[dict[str, object]] = []
    for row in rows:
        step = as_int(row, "candidate_step")
        prev_errors: dict[str, float] | None = None
        next_errors: dict[str, float] | None = None
        next_checkpoint = 0
        for checkpoint, errors in field_points:
            if checkpoint <= step:
                prev_errors = errors
            if checkpoint >= step:
                next_errors = errors
                next_checkpoint = checkpoint
                break

        deltas = {
            "iter_delta_step": as_int(row, "iter_delta_step"),
            "future_20_iter_delta": as_int(row, "future_20_iter_delta"),
            "future_50_iter_delta": as_int(row, "future_50_iter_delta"),
            "future_100_iter_delta": as_int(row, "future_100_iter_delta"),
            "future_until_next_iter_delta": as_int(row, "future_until_next_iter_delta"),
        }
        score = (
            deltas["iter_delta_step"]
            + min(
                deltas["future_20_iter_delta"],
                deltas["future_50_iter_delta"],
                deltas["future_100_iter_delta"],
                deltas["future_until_next_iter_delta"],
            )
        )
        record: dict[str, object] = {
            "source": source_name(path),
            "candidate_step": step,
            "next_candidate_step": row.get("next_candidate_step", ""),
            "baseline_iter": as_int(row, "baseline_iter"),
            "safe_iter": as_int(row, "safe_iter"),
            **deltas,
            "net_until_next": deltas["iter_delta_step"] + deltas["future_until_next_iter_delta"],
            "net_20": deltas["iter_delta_step"] + deltas["future_20_iter_delta"],
            "net_50": deltas["iter_delta_step"] + deltas["future_50_iter_delta"],
            "net_100": deltas["iter_delta_step"] + deltas["future_100_iter_delta"],
            "robust_score": score,
            "accepted": as_int(row, "pressure_init_accept_step"),
            "fallback": as_int(row, "pressure_init_fallback_step"),
            "field_health_pass": row.get("field_health_pass", ""),
            "checkpoint_step": next_checkpoint,
        }
        for key in ERROR_KEYS:
            current = as_float(row, key)
            prev = prev_errors.get(key) if prev_errors else None
            nxt = next_errors.get(key) if next_errors else None
            record[key] = current if current is not None else ""
            record[f"next_{key}"] = nxt if nxt is not None else ""
            record[f"checkpoint_delta_{key}"] = (nxt - prev) if nxt is not None and prev is not None else ""
        out.append(record)
    return out


def passes(row: dict[str, object], profile: Profile, max_step: int) -> bool:
    if int(row["candidate_step"]) > max_step:
        return False
    if int(row["accepted"]) != 1 or int(row["fallback"]) != 0:
        return False
    checks = (
        int(row["robust_score"]) >= profile.min_score,
        int(row["iter_delta_step"]) >= profile.min_step_delta,
        int(row["future_20_iter_delta"]) >= profile.min_f20,
        int(row["future_50_iter_delta"]) >= profile.min_f50,
        int(row["future_100_iter_delta"]) >= profile.min_f100,
        int(row["future_until_next_iter_delta"]) >= profile.min_until,
    )
    if not all(checks):
        return False
    pressure = row.get("next_pressure_rel_l2")
    rho = row.get("next_rho_rel_l2")
    fei = row.get("next_fei_rel_l2")
    d_pressure = row.get("checkpoint_delta_pressure_rel_l2")
    d_rho = row.get("checkpoint_delta_rho_rel_l2")
    d_fei = row.get("checkpoint_delta_fei_rel_l2")
    if pressure != "" and float(pressure) > profile.max_pressure:
        return False
    if rho != "" and float(rho) > profile.max_rho:
        return False
    if fei != "" and float(fei) > profile.max_fei:
        return False
    if d_pressure != "" and float(d_pressure) > profile.max_d_pressure:
        return False
    if d_rho != "" and float(d_rho) > profile.max_d_rho:
        return False
    if d_fei != "" and float(d_fei) > profile.max_d_fei:
        return False
    return True


def select(rows: list[dict[str, object]], profile: Profile, max_step: int) -> list[int]:
    eligible = [row for row in rows if passes(row, profile, max_step)]
    best_by_step: dict[int, dict[str, object]] = {}
    for row in eligible:
        step = int(row["candidate_step"])
        old = best_by_step.get(step)
        if old is None or int(row["robust_score"]) > int(old["robust_score"]):
            best_by_step[step] = row

    ranked = sorted(
        best_by_step.values(),
        key=lambda row: (int(row["robust_score"]), int(row["net_until_next"]), int(row["iter_delta_step"])),
        reverse=True,
    )
    selected: list[int] = []
    bins: dict[int, int] = {}
    for row in ranked:
        step = int(row["candidate_step"])
        bucket = step // 500
        if bins.get(bucket, 0) >= profile.max_count_per_500:
            continue
        if any(abs(step - existing) < profile.min_gap for existing in selected):
            continue
        selected.append(step)
        bins[bucket] = bins.get(bucket, 0) + 1
    return sorted(selected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_csvs", nargs="+", type=Path)
    parser.add_argument("--max-step", type=int, default=8000)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--analysis-csv", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for path in args.candidate_csvs:
        rows.extend(annotated_rows(path))

    keys = [
        "source", "candidate_step", "next_candidate_step", "baseline_iter", "safe_iter",
        "iter_delta_step", "future_20_iter_delta", "future_50_iter_delta",
        "future_100_iter_delta", "future_until_next_iter_delta",
        "net_20", "net_50", "net_100", "net_until_next", "robust_score",
        "accepted", "fallback", "field_health_pass", "checkpoint_step",
        "pressure_rel_l2", "rho_rel_l2", "fei_rel_l2", "phase_mass_rel_error",
        "next_pressure_rel_l2", "next_rho_rel_l2", "next_fei_rel_l2", "next_phase_mass_rel_error",
        "checkpoint_delta_pressure_rel_l2", "checkpoint_delta_rho_rel_l2",
        "checkpoint_delta_fei_rel_l2", "checkpoint_delta_phase_mass_rel_error",
    ]
    args.analysis_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.analysis_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in keys} for row in rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for profile in PROFILES:
        steps = select(rows, profile, args.max_step)
        (args.out_dir / f"{profile.name}_steps.txt").write_text(fmt_steps(steps))
        print(f"{profile.name}: {len(steps)} steps {steps[:20]}{'...' if len(steps) > 20 else ''}")
    print(f"wrote {args.analysis_csv}")
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
