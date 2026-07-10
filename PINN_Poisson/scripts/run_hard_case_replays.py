#!/usr/bin/env python3
"""Solve each captured hard state to a converged book-consistent h_i target."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


RESULT_RE = re.compile(r"iterations=(\d+) converged=(\d+) relative_error=([0-9.eE+-]+)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-exe", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--pre-state-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int, default=5000)
    args = parser.parse_args()
    diagnostics = {int(row["candidate_step"]): row for row in csv.DictReader(args.diagnostics.open()) if row["hard_case"] == "1"}
    args.target_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in sorted(diagnostics):
        name = f"3D{step:09d}.bin"
        feature = (args.feature_dir / name).resolve()
        pre_state = (args.pre_state_dir / name).resolve()
        target = (args.target_dir / name).resolve()
        result = subprocess.run(
            [str(args.replay_exe.resolve()), str(feature), str(pre_state), str(target), str(args.max_iterations)],
            text=True, capture_output=True,
        )
        match = RESULT_RE.search(result.stdout)
        if match is None:
            raise RuntimeError(f"replay failed at step {step}: {result.stderr.strip()}")
        iterations, converged, residual = int(match.group(1)), int(match.group(2)), float(match.group(3))
        rows.append({
            "trajectory": "h15_p50_closed_loop_hard_replay",
            "step": step,
            "input_snapshot": feature,
            "pre_state": pre_state,
            "target_state": target,
            "hard_reasons": diagnostics[step]["hard_reasons"],
            "full_poisson_iterations": iterations,
            "full_poisson_converged": converged,
            "full_poisson_relative_error": residual,
            "gate_pass": int(converged == 1 and residual < 1.0e-3),
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    failed = [row["step"] for row in rows if row["gate_pass"] != 1]
    print({"states": len(rows), "passed": len(rows) - len(failed), "failed": failed})
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
