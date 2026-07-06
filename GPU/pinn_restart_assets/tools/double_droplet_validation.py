#!/usr/bin/env python3
"""Run double-droplet collision validation for baseline and optimized GPU kernels."""

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

from validation_metrics import read_params, read_tecplot, summarize


POISSON_RE = re.compile(r"Poisson:\s+([0-9eE+\-.]+)")
ITERS_RE = re.compile(r"Poisson iters:\s+([0-9eE+\-.]+)")
TOTAL_RE = re.compile(r"Total per step:\s+([0-9eE+\-.]+)")
MLUPS_RE = re.compile(r"MLUPS:\s+([0-9eE+\-.]+)")


@dataclass(frozen=True)
class Variant:
    name: str
    poisson: str
    pressure_boundary: str
    graph: bool = False


VARIANTS = [
    Variant("gpu_split_baseline", "split", "split", False),
    Variant("gpu_all_fused", "fused", "fused", False),
    Variant("gpu_graph_all_fused", "fused", "fused", True),
]


def parse_perf(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if "  Poisson:" in line:
            m = POISSON_RE.search(line)
            if m:
                out["poisson_ms_per_step"] = float(m.group(1))
        elif "Poisson iters:" in line:
            m = ITERS_RE.search(line)
            if m:
                out["poisson_iters"] = float(m.group(1))
        elif "Total per step:" in line:
            m = TOTAL_RE.search(line)
            if m:
                out["gpu_metric_total_ms_per_step"] = float(m.group(1))
        elif "  MLUPS:" in line:
            m = MLUPS_RE.search(line)
            if m:
                out["gpu_metric_mlups"] = float(m.group(1))
    return out


def rel_diff(a: float, b: float, eps: float = 1.0e-30) -> float:
    if math.isnan(a) and math.isnan(b):
        return 0.0
    return abs(a - b) / max(abs(b), eps)


def run_variant(
    exe: Path,
    params: Path,
    steps: int,
    variant: Variant,
    threshold: float | None,
    save_outputs_dir: Path | None,
) -> dict[str, object]:
    exe_abs = exe.resolve()
    params_abs = params.resolve()
    params_values = read_params(params_abs)

    with tempfile.TemporaryDirectory(prefix=f"lbm_collision_{variant.name}_") as tmp:
        tmpdir = Path(tmp)
        cmd = [
            str(exe_abs),
            "--mode", "gpu",
            "--params", str(params_abs),
            "--steps", str(steps),
            "--output-frequency", str(steps),
            "--write-output",
            "--no-roofline",
            "--poisson", variant.poisson,
            "--pressure-boundary", variant.pressure_boundary,
        ]
        if variant.graph:
            cmd.append("--poisson-graph")

        proc = subprocess.run(cmd, cwd=tmpdir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        initial_plt = tmpdir / "out" / "3D000000000.plt"
        final_plt = tmpdir / "out" / f"3D{steps:09d}.plt"
        if not initial_plt.exists() or not final_plt.exists():
            raise RuntimeError(f"{variant.name} did not produce expected Tecplot files\n{proc.stdout}")

        initial = summarize(read_tecplot(initial_plt), params_values, threshold)
        final = summarize(read_tecplot(final_plt), params_values, threshold)
        perf = parse_perf(proc.stdout)

        row: dict[str, object] = {
            "variant": variant.name,
            "steps": steps,
            "poisson": variant.poisson,
            "pressure_boundary": variant.pressure_boundary,
            "cuda_graph": int(variant.graph),
            "run_exit_code": proc.returncode,
            "initial_regime": initial["regime"],
            "final_regime": final["regime"],
            "initial_component_count": initial["component_count"],
            "final_component_count": final["component_count"],
            "initial_phase_mass": initial["phase_mass"],
            "final_phase_mass": final["phase_mass"],
            "phase_mass_rel_change": rel_diff(float(final["phase_mass"]), float(initial["phase_mass"])),
            "initial_liquid_voxels": initial["liquid_voxels"],
            "final_liquid_voxels": final["liquid_voxels"],
            "liquid_voxels_rel_change": rel_diff(float(final["liquid_voxels"]), float(initial["liquid_voxels"])),
            "final_largest_component_voxels": final["largest_component_voxels"],
            "final_second_component_voxels": final["second_component_voxels"],
            "initial_centroid_distance": initial["component_centroid_distance"],
            "final_centroid_distance": final["component_centroid_distance"],
            "final_largest_cx": final["largest_component_cx"],
            "final_largest_cy": final["largest_component_cy"],
            "final_largest_cz": final["largest_component_cz"],
            "final_second_cx": final["second_component_cx"],
            "final_second_cy": final["second_component_cy"],
            "final_second_cz": final["second_component_cz"],
            "final_max_speed": final["max_speed"],
            "final_max_gas_speed": final["max_gas_speed"],
            "final_laplace_delta_p": final["laplace_delta_p"],
        }
        row.update(perf)

        if save_outputs_dir is not None:
            save_outputs_dir.mkdir(parents=True, exist_ok=True)
            init_dst = save_outputs_dir / f"{variant.name}_step0.plt"
            final_dst = save_outputs_dir / f"{variant.name}_step{steps}.plt"
            shutil.copy2(initial_plt, init_dst)
            shutil.copy2(final_plt, final_dst)
            row["saved_initial_plt"] = str(init_dst)
            row["saved_final_plt"] = str(final_dst)
        else:
            row["saved_initial_plt"] = ""
            row["saved_final_plt"] = ""

        return row


def add_baseline_diffs(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    baseline = rows[0]
    for row in rows:
        row["match_baseline_regime"] = int(row["final_regime"] == baseline["final_regime"])
        row["component_count_diff_vs_baseline"] = (
            int(row["final_component_count"]) - int(baseline["final_component_count"])
        )
        for key in [
            "final_phase_mass",
            "final_liquid_voxels",
            "final_largest_component_voxels",
            "final_second_component_voxels",
            "final_centroid_distance",
            "final_max_speed",
            "final_laplace_delta_p",
        ]:
            row[f"{key}_rel_to_baseline"] = rel_diff(float(row[key]), float(baseline[key]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=Path("./lbm_gpu"))
    parser.add_argument("--params", type=Path, default=Path("params_small.in"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--csv", type=Path, default=Path("double_droplet_validation.csv"))
    parser.add_argument("--save-outputs-dir", type=Path, default=Path("double_droplet_outputs"))
    args = parser.parse_args()

    rows = [
        run_variant(args.exe, args.params, args.steps, variant, args.threshold, args.save_outputs_dir)
        for variant in VARIANTS
    ]
    add_baseline_diffs(rows)

    keys = [
        "variant", "steps", "poisson", "pressure_boundary", "cuda_graph",
        "run_exit_code", "match_baseline_regime", "component_count_diff_vs_baseline",
        "initial_regime", "final_regime", "initial_component_count", "final_component_count",
        "initial_phase_mass", "final_phase_mass", "phase_mass_rel_change",
        "final_phase_mass_rel_to_baseline", "initial_liquid_voxels", "final_liquid_voxels",
        "liquid_voxels_rel_change", "final_liquid_voxels_rel_to_baseline",
        "final_largest_component_voxels", "final_largest_component_voxels_rel_to_baseline",
        "final_second_component_voxels", "final_second_component_voxels_rel_to_baseline",
        "initial_centroid_distance", "final_centroid_distance",
        "final_centroid_distance_rel_to_baseline", "final_largest_cx", "final_largest_cy",
        "final_largest_cz", "final_second_cx", "final_second_cy", "final_second_cz",
        "final_max_speed", "final_max_speed_rel_to_baseline",
        "final_max_gas_speed", "final_laplace_delta_p",
        "final_laplace_delta_p_rel_to_baseline", "gpu_metric_total_ms_per_step",
        "poisson_ms_per_step", "poisson_iters", "gpu_metric_mlups",
        "saved_initial_plt", "saved_final_plt",
    ]
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['variant']}: regime={row['final_regime']} "
            f"components={row['final_component_count']} "
            f"mass_change={float(row['phase_mass_rel_change']):.3e} "
            f"centroid_distance={float(row['final_centroid_distance']):.3e} "
            f"match={row['match_baseline_regime']}"
        )
    print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
