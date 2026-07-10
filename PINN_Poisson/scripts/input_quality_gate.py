#!/usr/bin/env python3
"""Score pressure-initializer inputs against a training manifest."""

from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from pathlib import Path

from tecplot_io import read_tecplot


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def component_count(fei: list[float], lx: int, ly: int, lz: int, threshold: float) -> int:
    mask = [value >= threshold for value in fei]
    seen = bytearray(len(mask))

    def index(x: int, y: int, z: int) -> int:
        return (z * ly + y) * lx + x

    count = 0
    for seed, keep in enumerate(mask):
        if not keep or seen[seed]:
            continue
        count += 1
        q: deque[int] = deque([seed])
        seen[seed] = 1
        while q:
            idx = q.popleft()
            x = idx % lx
            y = (idx // lx) % ly
            z = idx // (lx * ly)
            neighbors = []
            if x > 0:
                neighbors.append(index(x - 1, y, z))
            if x + 1 < lx:
                neighbors.append(index(x + 1, y, z))
            neighbors.append(index(x, (y - 1) % ly, z))
            neighbors.append(index(x, (y + 1) % ly, z))
            neighbors.append(index(x, y, (z - 1) % lz))
            neighbors.append(index(x, y, (z + 1) % lz))
            for nb in neighbors:
                if mask[nb] and not seen[nb]:
                    seen[nb] = 1
                    q.append(nb)
    return count


def field_metrics_from_data(data: dict[str, object]) -> dict[str, float | int]:
    lx = int(data["lx"])
    ly = int(data["ly"])
    lz = int(data["lz"])
    fields = data["fields"]  # type: ignore[assignment]
    fei = list(fields["fei"])  # type: ignore[index]
    fei_min = min(fei)
    fei_max = max(fei)
    width = max(fei_max - fei_min, 1.0e-20)
    threshold = 0.5 * (fei_min + fei_max)
    interface_fraction = sum(1 for value in fei if fei_min + 0.1 * width < value < fei_max - 0.1 * width) / len(fei)
    mid_fraction = sum(1 for value in fei if abs(value - threshold) < 0.2 * width) / len(fei)
    liquid_fraction = sum(1 for value in fei if value >= threshold) / len(fei)

    def idx(x: int, y: int, z: int) -> int:
        return (z * ly + y) * lx + x

    grad_sum = 0.0
    grad_max = 0.0
    for z in range(lz):
        for y in range(ly):
            for x in range(lx):
                gx = 0.5 * (fei[idx((x + 1) % lx, y, z)] - fei[idx((x - 1) % lx, y, z)])
                gy = 0.5 * (fei[idx(x, (y + 1) % ly, z)] - fei[idx(x, (y - 1) % ly, z)])
                gz = 0.5 * (fei[idx(x, y, (z + 1) % lz)] - fei[idx(x, y, (z - 1) % lz)])
                grad = math.sqrt(gx * gx + gy * gy + gz * gz)
                grad_sum += grad
                grad_max = max(grad_max, grad)

    return {
        "component_count": component_count(fei, lx, ly, lz, threshold),
        "interface_fraction": interface_fraction,
        "mid_fraction": mid_fraction,
        "liquid_fraction": liquid_fraction,
        "fei_grad_mean": grad_sum / len(fei),
        "fei_grad_max": grad_max,
    }


def field_metrics(path: Path) -> dict[str, float | int]:
    return field_metrics_from_data(read_tecplot(path))


def metric_ranges(manifest: Path, margin: float) -> dict[str, tuple[float, float]]:
    rows = list(csv.DictReader(manifest.open()))
    if not rows:
        raise SystemExit(f"empty manifest: {manifest}")
    values: dict[str, list[float]] = {}
    for row in rows:
        path = Path(row.get("input_plt") or row["plt"])
        metrics = field_metrics(path)
        for key, value in metrics.items():
            values.setdefault(key, []).append(float(value))

    ranges: dict[str, tuple[float, float]] = {}
    for key, vals in values.items():
        lo = min(vals)
        hi = max(vals)
        span = max(hi - lo, abs(hi), 1.0e-12)
        if key == "component_count":
            ranges[key] = (lo, hi)
        else:
            ranges[key] = (lo - margin * span, hi + margin * span)
    return ranges


def score(metrics: dict[str, float | int], ranges: dict[str, tuple[float, float]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for key, value in metrics.items():
        lo, hi = ranges[key]
        if key == "component_count":
            if int(value) < int(lo) or int(value) > int(hi):
                reasons.append(f"{key}={value} outside [{int(lo)}, {int(hi)}]")
        elif float(value) < lo or float(value) > hi:
            reasons.append(f"{key}={float(value):.6g} outside [{lo:.6g}, {hi:.6g}]")
    return not reasons, reasons


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--manifest", type=Path, default=root / "PINN_Poisson" / "data" / "paired_manifest.csv")
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    ranges = metric_ranges(args.manifest, args.margin)
    rows: list[dict[str, object]] = []
    for path in args.inputs:
        metrics = field_metrics(path)
        ok, reasons = score(metrics, ranges)
        row: dict[str, object] = {
            "input_plt": str(path),
            "accept_input": int(ok),
            "reasons": "; ".join(reasons),
        }
        row.update(metrics)
        rows.append(row)
        print(f"{path}: accept={int(ok)} reasons={row['reasons']}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        keys = [
            "input_plt", "accept_input", "reasons", "component_count",
            "interface_fraction", "mid_fraction", "liquid_fraction",
            "fei_grad_mean", "fei_grad_max",
        ]
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
