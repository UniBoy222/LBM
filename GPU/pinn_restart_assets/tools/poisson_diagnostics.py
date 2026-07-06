#!/usr/bin/env python3
"""Summarize Poisson residual diagnostics emitted by --poisson-diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def finite(value: float) -> bool:
    return math.isfinite(value)


def safe_ratio(num: float, den: float) -> float:
    if not finite(num) or not finite(den) or den == 0.0:
        return float("nan")
    return num / den


def safe_log10_drop(first: float, final: float) -> float:
    if not finite(first) or not finite(final) or first <= 0.0 or final <= 0.0:
        return float("nan")
    return math.log10(first / final)


def median(values: list[float]) -> float:
    vals = [v for v in values if finite(v)]
    return statistics.median(vals) if vals else float("nan")


def mean(values: list[float]) -> float:
    vals = [v for v in values if finite(v)]
    return statistics.fmean(vals) if vals else float("nan")


def fmt(value: float, digits: int = 4) -> str:
    if not finite(value):
        return "NA"
    if value == 0.0:
        return "0"
    if abs(value) < 1.0e-3 or abs(value) >= 1.0e4:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}g}"


def classify_step(final_rel: float, late_ratio: float, converged: bool, tolerance: float) -> str:
    if converged:
        return "converged"
    if final_rel <= tolerance:
        return "below_tolerance"
    if finite(late_ratio) and late_ratio > 0.85:
        return "stalled_tail"
    if finite(late_ratio) and late_ratio > 0.60:
        return "slow_tail"
    return "active_decay"


def summarize(rows: list[dict[str, str]], tolerance: float, max_iterations: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_step: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_step[as_int(row, "step")].append(row)

    step_rows: list[dict[str, object]] = []
    for step in sorted(by_step):
        checks = sorted(by_step[step], key=lambda r: as_int(r, "iteration"))
        rels = [as_float(r, "relative_error") for r in checks]
        iters = [as_int(r, "iteration") for r in checks]
        converged_rows = [r for r in checks if as_int(r, "converged") == 1]
        converged = bool(converged_rows)
        stop_row = converged_rows[0] if converged else checks[-1]
        final_rel = as_float(stop_row, "relative_error")
        final_iter = as_int(stop_row, "iteration")
        first_rel = rels[0] if rels else float("nan")
        prev_rel = rels[-2] if len(rels) >= 2 else float("nan")
        observed_final_rel = rels[-1] if rels else float("nan")
        late_ratio = safe_ratio(observed_final_rel, prev_rel)
        total_drop = safe_log10_drop(first_rel, observed_final_rel)
        block_low_freqs = [as_float(r, "block_low_frequency_fraction") for r in checks]
        final_block_low_freq = as_float(stop_row, "block_low_frequency_fraction")
        maxed = (not converged) and final_iter >= max_iterations
        step_rows.append({
            "step": step,
            "checks": len(checks),
            "first_iteration": iters[0] if iters else "",
            "first_relative_error": first_rel,
            "final_iteration": final_iter,
            "final_relative_error": final_rel,
            "observed_last_relative_error": observed_final_rel,
            "min_relative_error": min((v for v in rels if finite(v)), default=float("nan")),
            "late_ratio": late_ratio,
            "log10_drop": total_drop,
            "final_block_low_frequency_fraction": final_block_low_freq,
            "mean_block_low_frequency_fraction": mean(block_low_freqs),
            "converged": int(converged),
            "maxed_out": int(maxed),
            "tail_class": classify_step(final_rel, late_ratio, converged, tolerance),
        })

    converged_count = sum(int(r["converged"]) for r in step_rows)
    maxed_count = sum(int(r["maxed_out"]) for r in step_rows)
    slow_tail_count = sum(1 for r in step_rows if r["tail_class"] in {"slow_tail", "stalled_tail"})
    aggregate: dict[str, object] = {
        "steps": len(step_rows),
        "tolerance": tolerance,
        "max_iterations": max_iterations,
        "converged_steps": converged_count,
        "maxed_out_steps": maxed_count,
        "slow_or_stalled_tail_steps": slow_tail_count,
        "mean_final_iteration": mean([float(r["final_iteration"]) for r in step_rows]),
        "mean_final_relative_error": mean([float(r["final_relative_error"]) for r in step_rows]),
        "median_late_ratio": median([float(r["late_ratio"]) for r in step_rows]),
        "median_log10_drop": median([float(r["log10_drop"]) for r in step_rows]),
        "median_final_block_low_frequency_fraction": median([
            float(r["final_block_low_frequency_fraction"]) for r in step_rows
        ]),
        "median_mean_block_low_frequency_fraction": median([
            float(r["mean_block_low_frequency_fraction"]) for r in step_rows
        ]),
    }
    return step_rows, aggregate


def write_csv(path: Path, rows: list[dict[str, object]], keys: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, aggregate: dict[str, object], step_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Poisson Residual Diagnostics Summary",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key, value in aggregate.items():
        text = fmt(float(value)) if isinstance(value, float) else str(value)
        lines.append(f"| {key} | {text} |")

    lines += [
        "",
        "## Interpretation",
        "",
    ]
    steps = int(aggregate.get("steps", 0))
    maxed = int(aggregate.get("maxed_out_steps", 0))
    slow = int(aggregate.get("slow_or_stalled_tail_steps", 0))
    median_ratio = float(aggregate.get("median_late_ratio", float("nan")))
    if steps > 0 and maxed / steps > 0.3:
        lines.append("- Many steps hit the 1000-iteration cap; reducing launch overhead alone cannot solve this.")
    if steps > 0 and slow / steps > 0.3 and finite(median_ratio) and median_ratio > 0.6:
        lines.append("- Late residual decay is slow, so the next candidate should target fixed-point acceleration or a coarse-grid correction.")
    else:
        lines.append("- Residual decay is not dominated by a uniform late tail; inspect per-step rows before choosing a solver accelerator.")
    low_freq = float(aggregate.get("median_final_block_low_frequency_fraction", float("nan")))
    if finite(low_freq):
        if low_freq > 0.65:
            lines.append("- Block residual cancellation is weak, suggesting coarse-grid correction is plausible.")
        elif low_freq < 0.35:
            lines.append("- Block residual cancellation is strong, so a simple coarse-grid correction is unlikely to help.")
        else:
            lines.append("- Block residual low-frequency content is mixed; two-grid should start as a gated candidate, not a mainline change.")

    lines += [
        "",
        "## Per-Step Summary",
        "",
        "| step | final iter | final rel error | late ratio | block low freq | class |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in step_rows:
        lines.append(
            f"| {row['step']} | {row['final_iteration']} | "
            f"{fmt(float(row['final_relative_error']))} | {fmt(float(row['late_ratio']))} | "
            f"{fmt(float(row['final_block_low_frequency_fraction']))} | {row['tail_class']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path)
    parser.add_argument("--tolerance", type=float, default=1.0e-3)
    parser.add_argument("--max-iterations", type=int, default=1000)
    args = parser.parse_args()

    rows = read_rows(args.input)
    step_rows, aggregate = summarize(rows, args.tolerance, args.max_iterations)
    keys = [
        "step", "checks", "first_iteration", "first_relative_error",
        "final_iteration", "final_relative_error",
        "observed_last_relative_error", "min_relative_error",
        "late_ratio", "log10_drop",
        "final_block_low_frequency_fraction", "mean_block_low_frequency_fraction",
        "converged", "maxed_out", "tail_class",
    ]
    write_csv(args.summary_csv, step_rows, keys)
    if args.summary_md:
        write_markdown(args.summary_md, aggregate, step_rows)

    print(
        "steps={steps} converged={converged_steps} maxed={maxed_out_steps} "
        "slow_tail={slow_or_stalled_tail_steps} mean_iters={mean_final_iteration} "
        "median_late_ratio={median_late_ratio}".format(
            steps=aggregate["steps"],
            converged_steps=aggregate["converged_steps"],
            maxed_out_steps=aggregate["maxed_out_steps"],
            slow_or_stalled_tail_steps=aggregate["slow_or_stalled_tail_steps"],
            mean_final_iteration=fmt(float(aggregate["mean_final_iteration"])),
            median_late_ratio=fmt(float(aggregate["median_late_ratio"])),
        )
    )
    print(f"wrote {args.summary_csv}")
    if args.summary_md:
        print(f"wrote {args.summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
