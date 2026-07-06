#!/usr/bin/env python3
"""Build a Poisson error-performance boundary with accuracy and morphology gates."""

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


FIELD_RE = re.compile(r"^(fei|rho|u|v|w|p),([^,]+),([^,]+),")
MASS_RE = re.compile(r"rel_diff=([0-9eE+\-.]+)")
POISSON_RE = re.compile(r"Poisson:\s+([0-9eE+\-.]+)")
ITERS_RE = re.compile(r"Poisson iters:\s+([0-9eE+\-.]+)")
TOTAL_RE = re.compile(r"Total per step:\s+([0-9eE+\-.]+)")
MLUPS_RE = re.compile(r"MLUPS:\s+([0-9eE+\-.]+)")


@dataclass(frozen=True)
class Variant:
    name: str
    poisson: str
    pressure_boundary: str
    check_interval: int
    poisson_tolerance: float
    graph: bool = False


FIELD_KEYS = ("fei", "rho", "u", "v", "w", "p")


def tol_label(value: float) -> str:
    text = f"{value:.0e}"
    return text.replace("e-0", "e-").replace("e+0", "e+").replace("+", "p").replace("-", "m")


def build_variants(poissons: list[str], check_intervals: list[int], tolerances: list[float]) -> list[Variant]:
    variants: list[Variant] = []
    for poisson in poissons:
        pressure_boundary = "fused" if poisson in {"fused", "onepass"} else "split"
        for check_interval in check_intervals:
            for tolerance in tolerances:
                variants.append(Variant(
                    f"{poisson}_c{check_interval}_tol{tol_label(tolerance)}",
                    poisson,
                    pressure_boundary,
                    check_interval,
                    tolerance,
                ))
    return variants


def parse_compare(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        stripped = line.strip()
        field = FIELD_RE.match(stripped)
        if field:
            name, max_abs, rel_l2 = field.groups()
            out[f"{name}_max_abs"] = float(max_abs)
            out[f"{name}_rel_l2"] = float(rel_l2)
            continue
        if stripped.startswith("phase_mass_cpu="):
            mass = MASS_RE.search(stripped)
            if mass:
                out["mass_rel_diff"] = float(mass.group(1))
            continue
        if "  Poisson:" in line:
            match = POISSON_RE.search(line)
            if match:
                out["compare_poisson_ms_per_step"] = float(match.group(1))
            continue
        if "Poisson iters:" in line:
            match = ITERS_RE.search(line)
            if match:
                out["compare_poisson_iters"] = float(match.group(1))
            continue
        if "Total per step:" in line:
            match = TOTAL_RE.search(line)
            if match:
                out["compare_total_ms_per_step"] = float(match.group(1))
            continue
        if "  MLUPS:" in line:
            match = MLUPS_RE.search(line)
            if match:
                out["compare_mlups"] = float(match.group(1))
    return out


def parse_perf(text: str, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if "  Poisson:" in line:
            match = POISSON_RE.search(line)
            if match:
                out[f"{prefix}_poisson_ms_per_step"] = float(match.group(1))
        elif "Poisson iters:" in line:
            match = ITERS_RE.search(line)
            if match:
                out[f"{prefix}_poisson_iters"] = float(match.group(1))
        elif "Total per step:" in line:
            match = TOTAL_RE.search(line)
            if match:
                out[f"{prefix}_total_ms_per_step"] = float(match.group(1))
        elif "  MLUPS:" in line:
            match = MLUPS_RE.search(line)
            if match:
                out[f"{prefix}_mlups"] = float(match.group(1))
    return out


def exe_string(path: Path) -> str:
    text = str(path)
    if path.parent == Path(".") and "/" not in text:
        return f"./{text}"
    return text


def run_compare(exe: Path, params: Path, steps: int, variant: Variant) -> tuple[int, dict[str, float]]:
    cmd = [
        exe_string(exe),
        "--params", str(params),
        "--steps", str(steps),
        "--poisson", variant.poisson,
        "--pressure-boundary", variant.pressure_boundary,
        "--poisson-check-interval", str(variant.check_interval),
        "--poisson-tolerance", str(variant.poisson_tolerance),
        "--tolerance", "1e9",
        "--pressure-tolerance", "1e9",
    ]
    if variant.graph:
        cmd.append("--poisson-graph")
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, parse_compare(proc.stdout)


def run_physics(
    exe: Path,
    params: Path,
    steps: int,
    variant: Variant,
    threshold: float | None,
    save_outputs_dir: Path | None,
) -> tuple[int, dict[str, object]]:
    exe_abs = exe.resolve()
    params_abs = params.resolve()
    params_values = read_params(params_abs)

    with tempfile.TemporaryDirectory(prefix=f"poisson_boundary_{variant.name}_") as tmp:
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
            "--poisson-check-interval", str(variant.check_interval),
            "--poisson-tolerance", str(variant.poisson_tolerance),
        ]
        if variant.graph:
            cmd.append("--poisson-graph")

        proc = subprocess.run(cmd, cwd=tmpdir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        final_plt = tmpdir / "out" / f"3D{steps:09d}.plt"
        if not final_plt.exists():
            raise RuntimeError(f"{variant.name} did not produce {final_plt}\n{proc.stdout}")

        row = summarize(read_tecplot(final_plt), params_values, threshold)
        row.update(parse_perf(proc.stdout, "physics"))

        if save_outputs_dir is not None:
            save_outputs_dir.mkdir(parents=True, exist_ok=True)
            dst = save_outputs_dir / f"{variant.name}_step{steps}.plt"
            shutil.copy2(final_plt, dst)
            row["saved_physics_plt"] = str(dst)
        else:
            row["saved_physics_plt"] = ""
        return proc.returncode, row


def finite(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def rel_diff(value: object, ref: object, eps: float = 1.0e-30) -> float:
    a = finite(value)
    b = finite(ref)
    if math.isnan(a) and math.isnan(b):
        return 0.0
    if not math.isfinite(a) or not math.isfinite(b):
        return float("inf")
    return abs(a - b) / max(abs(b), eps)


def add_derived(rows: list[dict[str, object]], strict_tol: float, accuracy_tol: float, physics_tol: float) -> None:
    if not rows:
        return

    baseline = rows[0]
    baseline_ms = finite(baseline.get("compare_total_ms_per_step"))
    for row in rows:
        strict_pass = all(finite(row.get(f"{field}_rel_l2")) <= strict_tol for field in FIELD_KEYS)
        strict_pass = strict_pass and finite(row.get("mass_rel_diff")) <= strict_tol

        accuracy_pass = all(finite(row.get(f"{field}_rel_l2")) <= accuracy_tol for field in FIELD_KEYS)
        accuracy_pass = accuracy_pass and finite(row.get("mass_rel_diff")) <= accuracy_tol

        row["strict_pass"] = int(strict_pass)
        row["accuracy_pass"] = int(accuracy_pass)
        ms = finite(row.get("compare_total_ms_per_step"))
        row["speedup_vs_baseline"] = baseline_ms / ms if math.isfinite(baseline_ms) and ms > 0.0 else ""

        same_regime = row.get("regime") == baseline.get("regime")
        same_components = int(finite(row.get("component_count"))) == int(finite(baseline.get("component_count")))
        for key in [
            "phase_mass", "liquid_voxels", "largest_component_voxels",
            "second_component_voxels", "component_centroid_distance",
            "max_speed", "laplace_delta_p",
        ]:
            row[f"{key}_rel_to_baseline"] = rel_diff(row.get(key), baseline.get(key))

        physics_pass = (
            same_regime and same_components and
            finite(row["phase_mass_rel_to_baseline"]) <= physics_tol and
            finite(row["liquid_voxels_rel_to_baseline"]) <= physics_tol and
            finite(row["largest_component_voxels_rel_to_baseline"]) <= physics_tol and
            finite(row["second_component_voxels_rel_to_baseline"]) <= physics_tol and
            finite(row["max_speed_rel_to_baseline"]) <= physics_tol and
            finite(row["laplace_delta_p_rel_to_baseline"]) <= physics_tol
        )
        row["physical_match_baseline"] = int(physics_pass)
        row["paper_candidate"] = int(accuracy_pass and physics_pass)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-exe", type=Path, default=Path("./lbm_compare"))
    parser.add_argument("--gpu-exe", type=Path, default=Path("./lbm_gpu"))
    parser.add_argument("--compare-params", type=Path, default=Path("params_small.in"))
    parser.add_argument("--physics-params", type=Path, default=Path("double_droplet_stability_candidates_moving/params_tanh_w2_u0p01_cxdefault.in"))
    parser.add_argument("--compare-steps", type=int, default=20)
    parser.add_argument("--physics-steps", type=int, default=20)
    parser.add_argument("--poissons", default="fused,onepass")
    parser.add_argument("--check-intervals", default="100,50,25")
    parser.add_argument("--tolerances", default="1e-3,5e-4,1e-4")
    parser.add_argument("--strict-tol", type=float, default=1.0e-8)
    parser.add_argument("--accuracy-tol", type=float, default=1.0e-4)
    parser.add_argument("--physics-tol", type=float, default=1.0e-3)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--csv", type=Path, default=Path("poisson_error_performance.csv"))
    parser.add_argument("--save-outputs-dir", type=Path, default=None)
    args = parser.parse_args()

    poissons = [item.strip() for item in args.poissons.split(",") if item.strip()]
    check_intervals = [int(item) for item in args.check_intervals.split(",") if item.strip()]
    tolerances = [float(item) for item in args.tolerances.split(",") if item.strip()]
    variants = build_variants(poissons, check_intervals, tolerances)

    rows: list[dict[str, object]] = []
    for idx, variant in enumerate(variants, start=1):
        print(f"[{idx}/{len(variants)}] {variant.name}", flush=True)
        compare_code, compare_metrics = run_compare(
            args.compare_exe, args.compare_params, args.compare_steps, variant)
        physics_code, physics_metrics = run_physics(
            args.gpu_exe, args.physics_params, args.physics_steps,
            variant, args.threshold, args.save_outputs_dir)

        row: dict[str, object] = {
            "variant": variant.name,
            "poisson": variant.poisson,
            "pressure_boundary": variant.pressure_boundary,
            "check_interval": variant.check_interval,
            "poisson_tolerance": variant.poisson_tolerance,
            "graph": int(variant.graph),
            "compare_steps": args.compare_steps,
            "physics_steps": args.physics_steps,
            "compare_exit_code": compare_code,
            "physics_exit_code": physics_code,
        }
        row.update(compare_metrics)
        row.update(physics_metrics)
        rows.append(row)

    add_derived(rows, args.strict_tol, args.accuracy_tol, args.physics_tol)

    keys = [
        "variant", "poisson", "pressure_boundary", "check_interval",
        "poisson_tolerance", "graph", "compare_steps", "physics_steps",
        "compare_exit_code", "physics_exit_code", "strict_pass",
        "accuracy_pass", "physical_match_baseline", "paper_candidate",
        "speedup_vs_baseline", "compare_total_ms_per_step",
        "compare_poisson_ms_per_step", "compare_poisson_iters", "compare_mlups",
        "physics_total_ms_per_step", "physics_poisson_ms_per_step",
        "physics_poisson_iters", "physics_mlups", "mass_rel_diff",
    ]
    for field in FIELD_KEYS:
        keys.extend([f"{field}_max_abs", f"{field}_rel_l2"])
    keys.extend([
        "regime", "component_count", "phase_mass", "phase_mass_rel_to_baseline",
        "liquid_voxels", "liquid_voxels_rel_to_baseline",
        "largest_component_voxels", "largest_component_voxels_rel_to_baseline",
        "second_component_voxels", "second_component_voxels_rel_to_baseline",
        "component_centroid_distance", "component_centroid_distance_rel_to_baseline",
        "max_speed", "max_speed_rel_to_baseline", "laplace_delta_p",
        "laplace_delta_p_rel_to_baseline", "saved_physics_plt",
    ])

    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['variant']}: candidate={row['paper_candidate']} "
            f"strict={row['strict_pass']} acc={row['accuracy_pass']} "
            f"phys={row['physical_match_baseline']} "
            f"speedup={finite(row.get('speedup_vs_baseline')):.3f} "
            f"p_rel={finite(row.get('p_rel_l2')):.3e} "
            f"mass={finite(row.get('mass_rel_diff')):.3e}"
        )
    print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
