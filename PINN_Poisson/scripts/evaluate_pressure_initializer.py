#!/usr/bin/env python3
"""Evaluate a trained pressure initializer against a reference manifest."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from tecplot_io import read_tecplot
from train_pressure_initializer import CHANNELS, make_model, require_torch


def rel_l2_and_max(values: list[float], refs: list[float]) -> tuple[float, float]:
    sum_diff2 = 0.0
    sum_ref2 = 0.0
    max_abs = 0.0
    for value, ref in zip(values, refs):
        diff = value - ref
        sum_diff2 += diff * diff
        sum_ref2 += ref * ref
        max_abs = max(max_abs, abs(diff))
    rel = math.sqrt(sum_diff2 / sum_ref2) if sum_ref2 > 0.0 else math.sqrt(sum_diff2)
    return rel, max_abs


def main() -> int:
    torch, nn, _ = require_torch()
    root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=root / "data" / "paired_manifest.csv")
    parser.add_argument("--model", type=Path, default=root / "models" / "pressure_initializer.pt")
    parser.add_argument("--csv", type=Path, default=root / "results" / "benchmarks" / "pressure_initializer_eval.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location=args.device)
    channels = tuple(checkpoint.get("channels", CHANNELS))
    target_mode = checkpoint.get("target_mode", "absolute")
    model = make_model(
        nn,
        len(channels),
        checkpoint.get("architecture", "simple"),
        int(checkpoint.get("width", 32)),
        int(checkpoint.get("depth", 4)),
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    norm = checkpoint["normalization"]

    rows = list(csv.DictReader(args.manifest.open()))
    out_rows: list[dict[str, object]] = []
    for row in rows:
        input_plt = Path(row.get("input_plt") or row["plt"])
        target_plt = Path(row.get("target_plt") or row.get("plt") or row["input_plt"])
        input_data = read_tecplot(input_plt)
        target_data = read_tecplot(target_plt)
        lx = int(input_data["lx"])
        ly = int(input_data["ly"])
        lz = int(input_data["lz"])
        fields = input_data["fields"]  # type: ignore[assignment]
        target_fields = target_data["fields"]  # type: ignore[assignment]

        x = torch.stack([
            torch.tensor(fields[name], dtype=torch.float32).view(lz, ly, lx)  # type: ignore[index]
            for name in channels
        ]).unsqueeze(0).to(args.device)
        x_norm = (x - norm["x_mean"].to(args.device)) / norm["x_std"].to(args.device)
        with torch.no_grad():
            pred_norm = model(x_norm)
            pred_quantity = pred_norm * norm["y_std"].to(args.device) + norm["y_mean"].to(args.device)

        predicted = pred_quantity.squeeze(0).squeeze(0).cpu().reshape(-1).tolist()
        current_press = list(fields["press"])  # type: ignore[index]
        if target_mode == "delta":
            predicted_press = [current + delta for current, delta in zip(current_press, predicted)]
        else:
            predicted_press = predicted
        target_press = list(target_fields["press"])  # type: ignore[index]

        press_rel, press_max = rel_l2_and_max(predicted_press, target_press)
        out_rows.append({
            "label": row.get("label", input_plt.name),
            "model": str(args.model),
            "target_mode": target_mode,
            "channels": ",".join(channels),
            "pressure_rel_l2": press_rel,
            "pressure_max_abs": press_max,
            "input_plt": str(input_plt),
            "target_plt": str(target_plt),
        })

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "label", "model", "target_mode", "channels",
        "pressure_rel_l2", "pressure_max_abs", "input_plt", "target_plt",
    ]
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(out_rows)

    rels = [float(row["pressure_rel_l2"]) for row in out_rows]
    maxes = [float(row["pressure_max_abs"]) for row in out_rows]
    if rels:
        print(
            f"evaluated {len(rels)} samples: "
            f"rel_l2 mean={sum(rels) / len(rels):.6e} "
            f"min={min(rels):.6e} max={max(rels):.6e} "
            f"max_abs_max={max(maxes):.6e}"
        )
    print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
