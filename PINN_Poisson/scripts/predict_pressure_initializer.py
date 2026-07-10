#!/usr/bin/env python3
"""Run a trained pressure initializer and export a CUDA-readable pressure file."""

from __future__ import annotations

import argparse
from pathlib import Path

from tecplot_io import read_tecplot, write_pressure_initializer
from train_pressure_initializer import CHANNELS, make_model, require_torch


def main() -> int:
    torch, nn, _ = require_torch()
    root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("plt", type=Path)
    parser.add_argument("--model", type=Path, default=root / "models" / "pressure_initializer.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("absolute", "delta"), default="absolute")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="multiply exported values; intended for delta sweeps")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location=args.device)
    channels = tuple(checkpoint.get("channels", CHANNELS))
    model = make_model(
        nn,
        len(channels),
        checkpoint.get("architecture", "simple"),
        int(checkpoint.get("width", 32)),
        int(checkpoint.get("depth", 4)),
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    data = read_tecplot(args.plt)
    lx = int(data["lx"])
    ly = int(data["ly"])
    lz = int(data["lz"])
    fields = data["fields"]  # type: ignore[assignment]
    x = torch.stack([
        torch.tensor(fields[name], dtype=torch.float32).view(lz, ly, lx)  # type: ignore[index]
        for name in channels
    ]).unsqueeze(0).to(args.device)

    norm = checkpoint["normalization"]
    x_norm = (x - norm["x_mean"].to(args.device)) / norm["x_std"].to(args.device)
    with torch.no_grad():
        pred_norm = model(x_norm)
        pred = pred_norm * norm["y_std"].to(args.device) + norm["y_mean"].to(args.device)

    predicted_quantity = pred.squeeze(0).squeeze(0).cpu().reshape(-1).tolist()
    checkpoint_mode = checkpoint.get("target_mode", "absolute")
    current = list(fields["press"])  # type: ignore[index]
    if checkpoint_mode == "delta":
        values = predicted_quantity if args.mode == "delta" else [
            c + delta for c, delta in zip(current, predicted_quantity)
        ]
    else:
        values = predicted_quantity if args.mode == "absolute" else [
            p - c for p, c in zip(predicted_quantity, current)
        ]
    if args.scale != 1.0:
        values = [args.scale * value for value in values]

    write_pressure_initializer(args.out, lx, ly, lz, values)
    print(f"wrote {args.out} mode={args.mode} scale={args.scale} cells={len(values)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
