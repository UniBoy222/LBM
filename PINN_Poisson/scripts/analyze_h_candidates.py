#!/usr/bin/env python3
"""Classify closed-loop h_i candidates using per-step field and iteration errors."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from tecplot_io import read_feature_snapshot


def relative_l2(actual, reference) -> tuple[float, float]:
    actual_np = np.asarray(actual, dtype=np.float64)
    reference_np = np.asarray(reference, dtype=np.float64)
    diff = actual_np - reference_np
    return float(np.linalg.norm(diff) / max(np.linalg.norm(reference_np), 1.0e-300)), float(np.max(np.abs(diff)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--candidate-post-dir", type=Path, required=True)
    parser.add_argument("--baseline-post-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hard-steps", type=Path, required=True)
    parser.add_argument("--pressure-limit", type=float, default=2.1e-2)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.candidate_csv.open()))
    output = []
    hard_steps = []
    for row in rows:
        step = int(row["candidate_step"])
        name = f"3D{step:09d}.bin"
        candidate = read_feature_snapshot(args.candidate_post_dir / name)["fields"]
        baseline = read_feature_snapshot(args.baseline_post_dir / name)["fields"]
        metrics = {}
        for field in ("press", "rho", "fei"):
            rel, max_abs = relative_l2(candidate[field], baseline[field])  # type: ignore[index]
            metrics[f"{field}_rel_l2"] = rel
            metrics[f"{field}_max_abs"] = max_abs
        fallback = int(row["pressure_init_fallback_step"] or 0)
        accepted = int(row["pressure_init_accept_step"] or 0)
        direct = int(row["iter_delta_step"] or 0)
        future20 = int(row["future_20_iter_delta"] or 0)
        reasons = []
        if fallback:
            reasons.append("fallback")
        if metrics["press_rel_l2"] > args.pressure_limit:
            reasons.append("pressure_peak")
        if accepted and (metrics["press_rel_l2"] > args.pressure_limit or direct <= 0):
            reasons.append("false_accept")
        if direct < 0:
            reasons.append("negative_direct")
        if future20 < 0:
            reasons.append("negative_future20")
        hard = bool(reasons)
        if hard:
            hard_steps.append(step)
        output.append({**row, **metrics, "hard_case": int(hard), "hard_reasons": ";".join(reasons)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    args.hard_steps.write_text("\n".join(str(step) for step in hard_steps) + "\n")
    counts = {
        reason: sum(reason in row["hard_reasons"].split(";") for row in output)
        for reason in ("fallback", "pressure_peak", "false_accept", "negative_direct", "negative_future20")
    }
    print({"candidates": len(output), "hard_cases": len(hard_steps), **counts})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
