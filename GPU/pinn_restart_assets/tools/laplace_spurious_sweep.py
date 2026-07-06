#!/usr/bin/env python3
"""Run single-droplet Laplace-law and spurious-current validation sweeps."""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from validation_metrics import read_params, read_tecplot, summarize


POISSON_RE = re.compile(r"Poisson:\s+([0-9eE+\-.]+)")
ITERS_RE = re.compile(r"Poisson iters:\s+([0-9eE+\-.]+)")
TOTAL_RE = re.compile(r"Total per step:\s+([0-9eE+\-.]+)")
MLUPS_RE = re.compile(r"MLUPS:\s+([0-9eE+\-.]+)")


def parse_float_list(text: str) -> list[float]:
    return [float(item) for item in text.replace(",", " ").split() if item]


def parse_grid(text: str) -> tuple[int, int, int]:
    parts = [int(item) for item in text.replace(",", " ").split() if item]
    if len(parts) != 3:
        raise ValueError("--grid must contain exactly three integers")
    return parts[0], parts[1], parts[2]


def base_numeric_tokens(path: Path) -> list[str]:
    tokens = path.read_text().split()
    numeric = tokens[:22]
    if len(numeric) < 22:
        raise ValueError(f"{path} does not contain the 22 base numeric parameters")
    return numeric


def write_single_droplet_params(
    path: Path,
    base: list[str],
    grid: tuple[int, int, int],
    radius: float,
    steps: int,
    interface_profile: str,
    interface_width: float,
) -> None:
    lx, ly, lz = grid
    values = list(base)
    values[0:3] = [str(lx), str(ly), str(lz)]
    values[4] = str(steps)
    values[5] = str(steps)
    values[21] = f"{2.0 * radius:.12g}"

    path.write_text(
        "\n".join([
            f"{values[0]} {values[1]} {values[2]}",
            values[3],
            values[4],
            values[5],
            f"{values[6]} {values[7]}",
            f"{values[8]} {values[9]}",
            f"{values[10]} {values[11]}",
            f"{values[12]} {values[13]}",
            values[14],
            values[15],
            values[16],
            f"{values[17]} {values[18]}",
            f"{values[19]} {values[20]}",
            values[21],
            "",
            "init_mode=single_droplet",
            f"init_profile={interface_profile}",
            "init_velocity=0.0",
            f"interface_width={interface_width:.12g}",
            f"init_center_x={0.5 * lx:.12g}",
            f"init_center_y={0.5 * ly:.12g}",
            f"init_center_z={0.5 * lz:.12g}",
        ]) + "\n"
    )


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


def run_case(
    exe: Path,
    template: Path,
    base: list[str],
    grid: tuple[int, int, int],
    radius: float,
    steps: int,
    interface_profile: str,
    interface_width: float,
    use_graph: bool,
    save_plots_dir: Path | None,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix=f"lbm_laplace_R{radius:g}_") as tmp:
        tmpdir = Path(tmp)
        params_path = tmpdir / f"params_R{radius:g}.in"
        write_single_droplet_params(
            params_path,
            base,
            grid,
            radius,
            steps,
            interface_profile,
            interface_width,
        )

        cmd = [
            str(exe.resolve()),
            "--mode", "gpu",
            "--params", str(params_path),
            "--steps", str(steps),
            "--output-frequency", str(steps),
            "--write-output",
            "--no-roofline",
            "--poisson", "fused",
            "--pressure-boundary", "fused",
        ]
        if use_graph:
            cmd.append("--poisson-graph")

        proc = subprocess.run(cmd, cwd=tmpdir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        plt = tmpdir / "out" / f"3D{steps:09d}.plt"
        if not plt.exists():
            raise RuntimeError(f"R={radius:g} did not produce {plt}\n{proc.stdout}")

        params_values = read_params(params_path)
        row = summarize(read_tecplot(plt), params_values, threshold=None)
        row.update(parse_perf(proc.stdout))
        row.update({
            "radius": radius,
            "diameter": 2.0 * radius,
            "inverse_radius": 1.0 / radius,
            "steps": steps,
            "run_exit_code": proc.returncode,
            "template": str(template),
            "grid": f"{grid[0]}x{grid[1]}x{grid[2]}",
            "interface_profile": interface_profile,
            "interface_width": interface_width,
            "poisson": "fused",
            "pressure_boundary": "fused",
            "cuda_graph": int(use_graph),
        })

        if save_plots_dir is not None:
            save_plots_dir.mkdir(parents=True, exist_ok=True)
            dst = save_plots_dir / f"single_droplet_R{radius:g}_step{steps}.plt"
            shutil.copy2(plt, dst)
            shutil.copy2(params_path, save_plots_dir / f"params_R{radius:g}.in")
            row["saved_plt"] = str(dst)
        else:
            row["saved_plt"] = ""

        return row


def linear_fit(rows: list[dict[str, object]]) -> dict[str, float]:
    points = []
    for row in rows:
        x = float(row["inverse_radius"])
        y = float(row["laplace_delta_p"])
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y))
    n = len(points)
    if n < 2:
        return {"n": n, "slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}

    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    sxx = sum((x - mean_x) ** 2 for x, _ in points)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in points)
    slope = sxy / sxx if sxx > 0.0 else float("nan")
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for _, y in points)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in points)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return {"n": n, "slope": slope, "intercept": intercept, "r2": r2}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=Path("./lbm_gpu"))
    parser.add_argument("--template", type=Path, default=Path("params_small.in"))
    parser.add_argument("--grid", default="32x64x64")
    parser.add_argument("--radii", default="6,8,10")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--interface-profile", default="tanh", choices=["sharp", "tanh"])
    parser.add_argument("--interface-width", type=float, default=2.0)
    parser.add_argument("--csv", type=Path, default=Path("laplace_spurious.csv"))
    parser.add_argument("--fit-csv", type=Path, default=None)
    parser.add_argument("--poisson-graph", action="store_true")
    parser.add_argument("--save-plots-dir", type=Path, default=None)
    args = parser.parse_args()

    grid = parse_grid(args.grid.replace("x", " "))
    radii = parse_float_list(args.radii)
    base = base_numeric_tokens(args.template)

    rows = [
        run_case(
            args.exe,
            args.template,
            base,
            grid,
            radius,
            args.steps,
            args.interface_profile,
            args.interface_width,
            args.poisson_graph,
            args.save_plots_dir,
        )
        for radius in radii
    ]
    fit = linear_fit(rows)

    keys = [
        "radius", "diameter", "inverse_radius", "steps", "grid",
        "interface_profile", "interface_width",
        "poisson", "pressure_boundary", "cuda_graph", "run_exit_code",
        "regime", "component_count", "phase_mass", "liquid_voxels",
        "largest_component_voxels", "max_speed", "max_gas_speed",
        "mean_pressure_liquid", "mean_pressure_gas", "laplace_delta_p",
        "gpu_metric_total_ms_per_step", "poisson_ms_per_step", "poisson_iters",
        "gpu_metric_mlups", "saved_plt", "template",
    ]
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    fit_csv = args.fit_csv or args.csv.with_name(args.csv.stem + "_fit.csv")
    with fit_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "slope", "intercept", "r2"])
        writer.writeheader()
        writer.writerow(fit)

    for row in rows:
        print(
            f"R={float(row['radius']):g}: dp={float(row['laplace_delta_p']):.6e} "
            f"1/R={float(row['inverse_radius']):.6e} "
            f"max_u={float(row['max_speed']):.6e} comps={row['component_count']}"
        )
    print(f"fit: slope={fit['slope']:.6e} intercept={fit['intercept']:.6e} r2={fit['r2']:.6f}")
    print(f"wrote {args.csv}")
    print(f"wrote {fit_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
