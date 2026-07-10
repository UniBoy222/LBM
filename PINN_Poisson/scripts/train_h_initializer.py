#!/usr/bin/env python3
"""Train a full-volume 15-channel h_i initializer with the exact book operator."""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from book_pressure_operator import fixed_point_loss
from h_initializer_model import HInitializer, INPUT_CHANNELS, OUTPUT_CHANNELS, pressure_from_h


FEATURE_MAGIC = b"PINNF2\0\0"
STATE_MAGIC = b"PINNS1\0\0"
FEATURE_ORDER = (3, 0, 1, 2, 4, 5, 6)


def read_feature(path: Path) -> torch.Tensor:
    with path.open("rb") as handle:
        if handle.read(8) != FEATURE_MAGIC:
            raise ValueError(f"invalid feature snapshot: {path}")
        lx, ly, lz, nfields = struct.unpack("<iiii", handle.read(16))
    if nfields != 7:
        raise ValueError(f"expected seven input fields: {path}")
    n = lx * ly * lz
    data = np.memmap(path, dtype="<f4", mode="r", offset=24, shape=(7, n))
    array = np.asarray(data[list(FEATURE_ORDER)], dtype=np.float32).reshape(7, lz, ly, lx).copy()
    return torch.from_numpy(array)


def read_h_state(path: Path) -> torch.Tensor:
    with path.open("rb") as handle:
        if handle.read(8) != STATE_MAGIC:
            raise ValueError(f"invalid Poisson state: {path}")
        lx, ly, lz = struct.unpack("<iii", handle.read(12))
    n = lx * ly * lz
    offset = 20 + n * 8
    data = np.memmap(path, dtype="<f8", mode="r", offset=offset, shape=(n, 15))
    array = np.asarray(data, dtype=np.float32).reshape(lz, ly, lx, 15).transpose(3, 0, 1, 2).copy()
    return torch.from_numpy(array)


def load_rows(manifest: Path, split: str) -> list[dict[str, str]]:
    rows = [row for row in csv.DictReader(manifest.open()) if row["split"] == split]
    if not rows or any(row.get("gate_pass") != "1" for row in rows):
        raise ValueError(f"split {split} is empty or contains failed data gates")
    return rows


def load_stats(path: Path, device: torch.device):
    stats = json.loads(path.read_text())
    if not stats.get("all_gate_pass"):
        raise ValueError("manifest consistency gate did not pass")
    if tuple(stats["input_channels"]) != INPUT_CHANNELS or tuple(stats["output_channels"]) != OUTPUT_CHANNELS:
        raise ValueError("normalization channel order mismatch")
    shape_x = (1, 7, 1, 1, 1)
    shape_h = (1, 15, 1, 1, 1)
    return {
        "x_mean": torch.tensor(stats["x_mean"], device=device).view(shape_x),
        "x_std": torch.tensor(stats["x_std"], device=device).view(shape_x),
        "h_mean": torch.tensor(stats["h_mean"], device=device).view(shape_h),
        "h_std": torch.tensor(stats["h_std"], device=device).view(shape_h),
        "raw": stats,
    }


def losses(model, row, norm, device, weights, training: bool):
    x = read_feature(Path(row["input_snapshot"])).unsqueeze(0).to(device, non_blocking=True)
    target_h = read_h_state(Path(row["target_state"])).unsqueeze(0).to(device, non_blocking=True)
    x_norm = (x - norm["x_mean"]) / norm["x_std"]
    target_norm = (target_h - norm["h_mean"]) / norm["h_std"]
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        pred_norm = model(x_norm)
        pred_h = pred_norm * norm["h_std"] + norm["h_mean"]
        hh_loss = F.mse_loss(pred_norm, target_norm)
        target_p = pressure_from_h(target_h)
        pred_p = pressure_from_h(pred_h)
        p_scale2 = target_p.square().mean().clamp_min(1.0e-20)
        p_loss = (pred_p - target_p).square().mean() / p_scale2
        fp_scale2 = target_h.square().mean().clamp_min(1.0e-20)
        fp_loss = fixed_point_loss(
            pred_h, x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4],
            divergence=x[:, 6:7],
        ) / fp_scale2
        total = weights[0] * hh_loss + weights[1] * p_loss + weights[2] * fp_loss
    return total, hh_loss, p_loss, fp_loss


def evaluate(model, rows, norm, device, weights, limit: int = 0):
    model.eval()
    selected = rows[:limit] if limit > 0 else rows
    totals = np.zeros(4, dtype=np.float64)
    for row in selected:
        values = losses(model, row, norm, device, weights, False)
        totals += [value.detach().item() for value in values]
    return (totals / len(selected)).tolist()


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=root / "PINN_Poisson/data/h_true_post_manifest.csv")
    parser.add_argument("--stats", type=Path, default=root / "PINN_Poisson/data/h_true_post_stats.json")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--hh-weight", type=float, default=1.0)
    parser.add_argument("--pressure-weight", type=float, default=1.0)
    parser.add_argument("--fixed-point-weight", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--overfit-step", type=int, default=0)
    parser.add_argument("--overfit-min-reduction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--output-dir", type=Path, default=root / "PINN_Poisson/models/h_runs")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; local CPU training is forbidden")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    norm = load_stats(args.stats, device)
    train_rows = load_rows(args.manifest, "train")
    val_rows = load_rows(args.manifest, "val")
    test_rows = load_rows(args.manifest, "test")
    if args.overfit_step:
        matches = [row for row in train_rows if int(row["step"]) == args.overfit_step]
        if len(matches) != 1:
            raise SystemExit("--overfit-step must select exactly one training sample")
        train_rows = matches
        val_rows = matches
    elif args.max_train_samples > 0:
        train_rows = train_rows[:args.max_train_samples]

    run_dir = args.output_dir / args.run_name
    if run_dir.exists():
        raise SystemExit(f"run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    weights = (args.hh_weight, args.pressure_weight, args.fixed_point_weight)
    model = HInitializer(args.width, args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    metrics_path = run_dir / "metrics.csv"
    initial = evaluate(model, val_rows, norm, device, weights, args.max_val_samples)
    best_val = math.inf
    best_path = run_dir / "best.pt"
    start_time = time.monotonic()
    with metrics_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_total", "train_hh", "train_p", "train_fixed", "val_total", "val_hh", "val_p", "val_fixed", "seconds"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = torch.randperm(len(train_rows)).tolist()
            totals = np.zeros(4, dtype=np.float64)
            for index in order:
                optimizer.zero_grad(set_to_none=True)
                values = losses(model, train_rows[index], norm, device, weights, True)
                values[0].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                totals += [value.detach().item() for value in values]
            train_metric = (totals / len(train_rows)).tolist()
            val_metric = evaluate(model, val_rows, norm, device, weights, args.max_val_samples)
            elapsed = time.monotonic() - start_time
            writer.writerow([epoch, *train_metric, *val_metric, elapsed])
            handle.flush()
            print(json.dumps({"epoch": epoch, "train": train_metric, "val": val_metric, "seconds": elapsed}), flush=True)
            if val_metric[0] < best_val:
                best_val = val_metric[0]
                torch.save({
                    "model_state": model.state_dict(),
                    "width": args.width, "depth": args.depth,
                    "input_channels": INPUT_CHANNELS, "output_channels": OUTPUT_CHANNELS,
                    "normalization": norm["raw"],
                    "loss_weights": weights,
                    "epoch": epoch, "val_metrics": val_metric,
                    "operator": "book_eq_6_42_6_44",
                }, best_path)

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metric = evaluate(model, test_rows, norm, device, weights, args.max_val_samples)
    summary = {
        "run_name": args.run_name,
        "device": torch.cuda.get_device_name(0),
        "train_samples": len(train_rows), "val_samples": len(val_rows), "test_samples": len(test_rows),
        "initial": initial, "best_val": checkpoint["val_metrics"], "test": test_metric,
        "best_epoch": checkpoint["epoch"], "elapsed_seconds": time.monotonic() - start_time,
        "model": str(best_path),
    }
    if args.overfit_step:
        reduction = 1.0 - summary["best_val"][1] / max(initial[1], 1.0e-30)
        summary["overfit_hh_reduction"] = reduction
        summary["overfit_pass"] = reduction >= args.overfit_min_reduction
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if args.overfit_step and not summary["overfit_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
