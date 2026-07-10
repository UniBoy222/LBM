#!/usr/bin/env python3
"""Build and gate true post-Poisson h_i manifests with time-block splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
from pathlib import Path

import numpy as np


FEATURE_MAGIC = b"PINNF2\0\0"
STATE_MAGIC = b"PINNS1\0\0"
INPUT_ORDER = (3, 0, 1, 2, 4, 5, 6)  # rho,u,v,w,fei,press,div_u_source
INPUT_NAMES = ("rho", "u", "v", "w", "fei", "press", "div_u_source")
STEP_RE = re.compile(r"3D(\d{9})\.bin$")


def step_files(directory: Path) -> dict[int, Path]:
    result = {}
    for path in directory.glob("3D*.bin"):
        match = STEP_RE.search(path.name)
        if match:
            result[int(match.group(1))] = path.resolve()
    return result


def feature_map(path: Path):
    with path.open("rb") as handle:
        if handle.read(8) != FEATURE_MAGIC:
            raise ValueError(f"invalid feature magic: {path}")
        lx, ly, lz, nfields = struct.unpack("<iiii", handle.read(16))
    if nfields != 7:
        raise ValueError(f"expected 7 feature fields: {path}")
    n = lx * ly * lz
    expected = 24 + 7 * n * 4
    if path.stat().st_size != expected:
        raise ValueError(f"feature size mismatch: {path}")
    values = np.memmap(path, dtype="<f4", mode="r", offset=24, shape=(7, n))
    return (lx, ly, lz), values


def state_map(path: Path):
    with path.open("rb") as handle:
        if handle.read(8) != STATE_MAGIC:
            raise ValueError(f"invalid state magic: {path}")
        lx, ly, lz = struct.unpack("<iii", handle.read(12))
    n = lx * ly * lz
    expected = 20 + n * 8 + n * 15 * 8
    if path.stat().st_size != expected:
        raise ValueError(f"state size mismatch: {path}")
    pressure = np.memmap(path, dtype="<f8", mode="r", offset=20, shape=(n,))
    hh = np.memmap(path, dtype="<f8", mode="r", offset=20 + n * 8, shape=(n, 15))
    return (lx, ly, lz), pressure, hh


class Moments:
    def __init__(self, channels: int) -> None:
        self.total = np.zeros(channels, dtype=np.float64)
        self.total2 = np.zeros(channels, dtype=np.float64)
        self.count = 0

    def add(self, values: np.ndarray) -> None:
        values64 = np.asarray(values, dtype=np.float64)
        self.total += values64.sum(axis=0)
        self.total2 += np.square(values64).sum(axis=0)
        self.count += values64.shape[0]

    def result(self) -> tuple[list[float], list[float]]:
        mean = self.total / self.count
        variance = np.maximum(self.total2 / self.count - np.square(mean), 1.0e-20)
        return mean.tolist(), np.sqrt(variance).tolist()


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-dir", type=Path, required=True)
    parser.add_argument("--post-feature-dir", type=Path, required=True)
    parser.add_argument("--post-state-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=root / "PINN_Poisson/data/h_manifest.csv")
    parser.add_argument("--stats", type=Path, default=root / "PINN_Poisson/data/h_manifest_stats.json")
    parser.add_argument("--train-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--val-fraction", type=float, default=1.0 / 6.0)
    parser.add_argument("--p-sum-max", type=float, default=1.0e-12)
    parser.add_argument("--post-pressure-max", type=float, default=2.0e-7)
    args = parser.parse_args()

    pre = step_files(args.pre_dir)
    post = step_files(args.post_feature_dir)
    states = step_files(args.post_state_dir)
    steps = sorted(set(pre) & set(post) & set(states))
    if not steps or set(steps) != set(pre) or set(steps) != set(post) or set(steps) != set(states):
        raise SystemExit("pre/post/state step sets are empty or do not match exactly")
    train_end = max(1, int(len(steps) * args.train_fraction))
    val_count = max(1, int(round(len(steps) * args.val_fraction)))
    val_end = train_end + val_count
    val_end = min(val_end, len(steps) - 1)

    x_moments = Moments(7)
    h_moments = Moments(15)
    rows = []
    chunk = 65536
    for index, step in enumerate(steps):
        split = "train" if index < train_end else ("val" if index < val_end else "test")
        pre_dims, pre_values = feature_map(pre[step])
        post_dims, post_values = feature_map(post[step])
        state_dims, state_pressure, hh = state_map(states[step])
        if pre_dims != post_dims or pre_dims != state_dims:
            raise ValueError(f"dimension mismatch at step {step}")
        n = math.prod(pre_dims)
        sum_diff2 = 0.0
        sum_ref2 = 0.0
        p_sum_max = 0.0
        post_diff2 = 0.0
        post_ref2 = 0.0
        post_max = 0.0
        finite = True
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            h_chunk = np.asarray(hh[start:end], dtype=np.float64)
            p_chunk = np.asarray(state_pressure[start:end], dtype=np.float64)
            h_sum = h_chunk.sum(axis=1)
            diff = h_sum - p_chunk
            p_sum_max = max(p_sum_max, float(np.max(np.abs(diff))))
            sum_diff2 += float(np.dot(diff, diff))
            sum_ref2 += float(np.dot(p_chunk, p_chunk))
            post_p = np.asarray(post_values[5, start:end], dtype=np.float64)
            post_diff = p_chunk - post_p
            post_max = max(post_max, float(np.max(np.abs(post_diff))))
            post_diff2 += float(np.dot(post_diff, post_diff))
            post_ref2 += float(np.dot(post_p, post_p))
            finite = finite and bool(np.isfinite(h_chunk).all() and np.isfinite(p_chunk).all())
            finite = finite and bool(np.isfinite(post_p).all())
            if split == "train":
                h_moments.add(h_chunk)
                x_chunk = np.asarray(pre_values[:, start:end], dtype=np.float64).T[:, INPUT_ORDER]
                x_moments.add(x_chunk)
                finite = finite and bool(np.isfinite(x_chunk).all())
        p_sum_rel = math.sqrt(sum_diff2 / max(sum_ref2, 1.0e-300))
        post_rel = math.sqrt(post_diff2 / max(post_ref2, 1.0e-300))
        passed = finite and p_sum_max <= args.p_sum_max and post_max <= args.post_pressure_max
        rows.append({
            "step": step,
            "split": split,
            "input_snapshot": pre[step],
            "post_snapshot": post[step],
            "target_state": states[step],
            "lx": pre_dims[0], "ly": pre_dims[1], "lz": pre_dims[2],
            "state_p_sum_max_abs": p_sum_max,
            "state_p_sum_rel_l2": p_sum_rel,
            "post_pressure_max_abs": post_max,
            "post_pressure_rel_l2": post_rel,
            "finite": int(finite),
            "gate_pass": int(passed),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    x_mean, x_std = x_moments.result()
    h_mean, h_std = h_moments.result()
    stats = {
        "input_channels": INPUT_NAMES,
        "output_channels": [f"h{i}" for i in range(15)],
        "x_mean": x_mean, "x_std": x_std,
        "h_mean": h_mean, "h_std": h_std,
        "splits": {name: sum(row["split"] == name for row in rows) for name in ("train", "val", "test")},
        "all_gate_pass": all(row["gate_pass"] == 1 for row in rows),
    }
    args.stats.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({
        "samples": len(rows),
        "splits": stats["splits"],
        "all_gate_pass": stats["all_gate_pass"],
        "max_state_p_sum": max(row["state_p_sum_max_abs"] for row in rows),
        "max_post_pressure": max(row["post_pressure_max_abs"] for row in rows),
        "manifest": str(args.out),
        "stats": str(args.stats),
    }))
    return 0 if stats["all_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
