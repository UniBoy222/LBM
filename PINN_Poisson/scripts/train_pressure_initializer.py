#!/usr/bin/env python3
"""Train a small physics-informed 3D CNN pressure initializer."""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

from tecplot_io import read_tecplot


CHANNELS = ("rho", "u", "v", "w", "fei", "press")


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for training. Install torch in this environment, "
            "then rerun train_pressure_initializer.py."
        ) from exc
    return torch, nn, F


def load_samples(manifest: Path, torch, target_mode: str):
    rows = list(csv.DictReader(manifest.open()))
    samples = []
    for row in rows:
        input_plt = Path(row.get("input_plt") or row["plt"])
        target_plt = Path(row.get("target_plt") or row.get("plt") or row["input_plt"])
        input_data = read_tecplot(input_plt)
        target_data = read_tecplot(target_plt)
        if (input_data["lx"], input_data["ly"], input_data["lz"]) != (
            target_data["lx"], target_data["ly"], target_data["lz"]
        ):
            raise ValueError(f"grid mismatch for {input_plt} and {target_plt}")
        lx = int(input_data["lx"])
        ly = int(input_data["ly"])
        lz = int(input_data["lz"])
        fields = input_data["fields"]  # type: ignore[assignment]
        target_fields = target_data["fields"]  # type: ignore[assignment]
        x = torch.stack([
            torch.tensor(fields[name], dtype=torch.float32).view(lz, ly, lx)  # type: ignore[index]
            for name in CHANNELS
        ])
        target_press = list(target_fields["press"])  # type: ignore[index]
        if target_mode == "delta":
            input_press = fields["press"]  # type: ignore[index]
            target_press = [target - current for target, current in zip(target_press, input_press)]
        y = torch.tensor(target_press, dtype=torch.float32).view(1, lz, ly, lx)
        samples.append((x, y, row.get("label", input_plt.name)))
    if not samples:
        raise SystemExit(f"no samples in {manifest}")
    return samples


def make_model(
    nn,
    n_channels: int | None = None,
    architecture: str = "simple",
    width: int = 32,
    depth: int = 4,
):
    if n_channels is None:
        n_channels = len(CHANNELS)
    if architecture == "simple":
        return nn.Sequential(
            nn.Conv3d(n_channels, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(24, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(32, 24, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(24, 1, kernel_size=1),
        )
    if architecture != "residual":
        raise ValueError(f"unknown architecture: {architecture}")

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            )
            self.act = nn.GELU()

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.act(x + self.net(x))

    layers: list[nn.Module] = [
        nn.Conv3d(n_channels, width, kernel_size=3, padding=1),
        nn.GELU(),
    ]
    for _ in range(depth):
        layers.append(ResidualBlock(width))
    layers.append(nn.Conv3d(width, 1, kernel_size=1))
    return nn.Sequential(*layers)


def normalize_batch(x, y):
    x_mean = x.mean(dim=(0, 2, 3, 4), keepdim=True)
    x_std = x.std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1.0e-8)
    y_mean = y.mean(dim=(0, 2, 3, 4), keepdim=True)
    y_std = y.std(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1.0e-8)
    return (x - x_mean) / x_std, (y - y_mean) / y_std, {
        "x_mean": x_mean.cpu(),
        "x_std": x_std.cpu(),
        "y_mean": y_mean.cpu(),
        "y_std": y_std.cpu(),
        "channels": CHANNELS,
    }


def normalize_samples(samples, torch):
    x_sum = None
    x_sum2 = None
    y_sum = None
    y_sum2 = None
    count = 0
    for x, y, _ in samples:
        x64 = x.double()
        y64 = y.double()
        if x_sum is None:
            x_sum = torch.zeros(x.shape[0], dtype=torch.float64)
            x_sum2 = torch.zeros(x.shape[0], dtype=torch.float64)
            y_sum = torch.zeros(1, dtype=torch.float64)
            y_sum2 = torch.zeros(1, dtype=torch.float64)
        x_sum += x64.sum(dim=(1, 2, 3))
        x_sum2 += x64.square().sum(dim=(1, 2, 3))
        y_sum += y64.sum()
        y_sum2 += y64.square().sum()
        count += x.shape[1] * x.shape[2] * x.shape[3]

    assert x_sum is not None and x_sum2 is not None and y_sum is not None and y_sum2 is not None
    x_mean = (x_sum / count).float().view(1, -1, 1, 1, 1)
    x_var = (x_sum2 / count - (x_sum / count).square()).clamp_min(1.0e-16)
    x_std = x_var.sqrt().float().view(1, -1, 1, 1, 1)
    y_mean = (y_sum / count).float().view(1, 1, 1, 1, 1)
    y_var = (y_sum2 / count - (y_sum / count).square()).clamp_min(1.0e-16)
    y_std = y_var.sqrt().float().view(1, 1, 1, 1, 1)
    return {
        "x_mean": x_mean.cpu(),
        "x_std": x_std.cpu(),
        "y_mean": y_mean.cpu(),
        "y_std": y_std.cpu(),
        "channels": CHANNELS,
    }


def parse_patch_size(text: str) -> tuple[int, int, int] | None:
    if text.strip() in {"", "0"}:
        return None
    parts = [int(part) for part in text.replace("x", ",").split(",") if part.strip()]
    if len(parts) == 1:
        return (parts[0], parts[0], parts[0])
    if len(parts) == 3:
        return (parts[0], parts[1], parts[2])
    raise SystemExit("--patch-size must be empty, one integer, or z,y,x")


def random_patch(sample, patch_size: tuple[int, int, int], torch, generator):
    x, y, _ = sample
    pz, py, px = patch_size
    _, lz, ly, lx = x.shape
    if pz > lz or py > ly or px > lx:
        raise SystemExit(f"patch {patch_size} larger than sample grid {(lz, ly, lx)}")
    z0 = int(torch.randint(0, lz - pz + 1, (1,), generator=generator).item())
    y0 = int(torch.randint(0, ly - py + 1, (1,), generator=generator).item())
    x0 = int(torch.randint(0, lx - px + 1, (1,), generator=generator).item())
    return (
        x[:, z0:z0 + pz, y0:y0 + py, x0:x0 + px],
        y[:, z0:z0 + pz, y0:y0 + py, x0:x0 + px],
    )


def make_patch_batch(samples, patch_size: tuple[int, int, int], batch_size: int, torch, generator):
    xs = []
    ys = []
    for _ in range(batch_size):
        sample = samples[int(torch.randint(0, len(samples), (1,), generator=generator).item())]
        x_patch, y_patch = random_patch(sample, patch_size, torch, generator)
        xs.append(x_patch)
        ys.append(y_patch)
    return torch.stack(xs), torch.stack(ys)


def load_initial_model(model, init_model: Path | None, torch) -> None:
    if init_model is None:
        return
    checkpoint = torch.load(init_model, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])


def shifted_neighbor(var, ex: int, ey: int, ez: int, torch):
    """GPU gradient boundary: x slip/symmetric, y/z periodic."""
    out = torch.roll(var, shifts=(-ez, -ey), dims=(2, 3))
    if ex == 1:
        return torch.cat([out[..., 1:], out[..., -2:-1]], dim=4)
    if ex == -1:
        return torch.cat([out[..., 1:2], out[..., :-1]], dim=4)
    return out


def d3q15_divergence(u, v, w, torch):
    ex = (0, 1, 0, 0, -1, 0, 0, 1, -1, 1, 1, -1, 1, -1, -1)
    ey = (0, 0, 1, 0, 0, -1, 0, 1, 1, -1, 1, -1, -1, 1, -1)
    ez = (0, 0, 0, 1, 0, 0, -1, 1, 1, 1, -1, -1, -1, -1, 1)
    grad_u_x = u.new_zeros(u.shape)
    grad_v_y = v.new_zeros(v.shape)
    grad_w_z = w.new_zeros(w.shape)
    for q in range(1, 15):
        grad_u_x = grad_u_x + shifted_neighbor(u, ex[q], ey[q], ez[q], torch) * float(ex[q])
        grad_v_y = grad_v_y + shifted_neighbor(v, ex[q], ey[q], ez[q], torch) * float(ey[q])
        grad_w_z = grad_w_z + shifted_neighbor(w, ex[q], ey[q], ez[q], torch) * float(ez[q])
    return (grad_u_x + grad_v_y + grad_w_z) / 10.0


def d3q15_solver_fixed_point_loss(pred_pressure, x_physical, F, torch):
    ex = (0, 1, 0, 0, -1, 0, 0, 1, -1, 1, 1, -1, 1, -1, -1)
    ey = (0, 0, 1, 0, 0, -1, 0, 1, 1, -1, 1, -1, -1, 1, -1)
    ez = (0, 0, 0, 1, 0, 0, -1, 1, 1, 1, -1, -1, -1, -1, 1)
    ei = pred_pressure.new_tensor(
        [
            2.0 / 9.0,
            1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
            1.0 / 72.0, 1.0 / 72.0, 1.0 / 72.0, 1.0 / 72.0,
            1.0 / 72.0, 1.0 / 72.0, 1.0 / 72.0, 1.0 / 72.0,
        ]
    )
    rho = x_physical[:, 0:1].clamp_min(1.0e-8)
    u = x_physical[:, 1:2]
    v = x_physical[:, 2:3]
    w = x_physical[:, 3:4]
    tauh = 1.0 / rho + 0.5
    div_u = d3q15_divergence(u, v, w, torch)

    streamed = []
    for q in range(15):
        local_h = ei[q] * pred_pressure - tauh * (ei[q] / 3.0) * div_u
        streamed.append(torch.roll(local_h, shifts=(ez[q], ey[q], ex[q]), dims=(2, 3, 4)))

    # Same fused x-wall bounce-back as collisionStreamBoundaryPressureKernel.
    original = list(streamed)
    left_pairs = ((1, 4), (7, 8), (9, 14), (10, 13), (12, 11))
    right_pairs = ((4, 1), (8, 7), (14, 9), (13, 10), (11, 12))
    for dst, src in left_pairs:
        streamed[dst] = torch.cat([original[src][..., :1], original[dst][..., 1:]], dim=4)
    for dst, src in right_pairs:
        streamed[dst] = torch.cat([original[dst][..., :-1], original[src][..., -1:]], dim=4)

    p_next = torch.stack(streamed, dim=0).sum(dim=0)
    residual = p_next - pred_pressure
    scale = pred_pressure.detach().square().mean().sqrt().clamp_min(1.0e-6)
    return F.mse_loss(residual / scale, torch.zeros_like(residual))


def simple_poisson_residual_loss(pred_pressure, x_physical, F):
    rho = x_physical[:, 0:1]
    u = x_physical[:, 1:2]
    v = x_physical[:, 2:3]
    w = x_physical[:, 3:4]
    du = 0.5 * (u.roll(-1, dims=4) - u.roll(1, dims=4))
    dv = 0.5 * (v.roll(-1, dims=3) - v.roll(1, dims=3))
    dw = 0.5 * (w.roll(-1, dims=2) - w.roll(1, dims=2))
    rhs = rho * (du + dv + dw)
    lap = (
        pred_pressure.roll(1, dims=2) + pred_pressure.roll(-1, dims=2) +
        pred_pressure.roll(1, dims=3) + pred_pressure.roll(-1, dims=3) +
        pred_pressure.roll(1, dims=4) + pred_pressure.roll(-1, dims=4) -
        6.0 * pred_pressure
    )
    return F.mse_loss(lap, rhs)


def poisson_residual_loss(pred_quantity, x_physical, F, torch, residual_weight: float, target_mode: str, residual_mode: str):
    if residual_weight <= 0.0:
        return pred_quantity.new_tensor(0.0)
    pred_pressure = pred_quantity
    if target_mode == "delta":
        pred_pressure = x_physical[:, 5:6] + pred_quantity
    if residual_mode == "simple":
        residual = simple_poisson_residual_loss(pred_pressure, x_physical, F)
    elif residual_mode == "solver":
        residual = d3q15_solver_fixed_point_loss(pred_pressure, x_physical, F, torch)
    else:
        raise ValueError(f"unknown residual mode: {residual_mode}")
    return residual_weight * residual


def pressure_relative_loss(pred_quantity, target_quantity, x_physical, target_mode: str):
    pred_pressure = pred_quantity
    target_pressure = target_quantity
    if target_mode == "delta":
        current_pressure = x_physical[:, 5:6]
        pred_pressure = current_pressure + pred_quantity
        target_pressure = current_pressure + target_quantity
    diff2 = (pred_pressure - target_pressure).square().flatten(1).sum(dim=1)
    ref2 = target_pressure.square().flatten(1).sum(dim=1).clamp_min(1.0e-20)
    return (diff2 / ref2).mean()


def main() -> int:
    raise SystemExit(
        "legacy one-channel pressure training is disabled; "
        "the book-consistent direction requires a 15-channel h_i target"
    )

    # Historical implementation remains importable for old checkpoint replay only.
    torch, nn, F = require_torch()
    root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=root / "data" / "reference_manifest.csv")
    parser.add_argument("--out", type=Path, default=root / "models" / "pressure_initializer.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--residual-weight", type=float, default=1.0e-3)
    parser.add_argument("--residual-mode", choices=("solver", "simple"), default="solver")
    parser.add_argument("--pressure-rel-weight", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--target-mode", choices=("absolute", "delta"), default="absolute")
    parser.add_argument("--architecture", choices=("simple", "residual"), default="simple")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--patch-size", default="",
                        help="empty/0 trains on full volumes; otherwise use one integer or z,y,x patches")
    parser.add_argument("--patches-per-epoch", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--init-model", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    samples = load_samples(args.manifest, torch, args.target_mode)
    model = make_model(nn, len(CHANNELS), args.architecture, args.width, args.depth).to(args.device)
    load_initial_model(model, args.init_model, torch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_metric = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patch_size = parse_patch_size(args.patch_size)
    if patch_size is None:
        x = torch.stack([sample[0] for sample in samples]).to(args.device)
        y = torch.stack([sample[1] for sample in samples]).to(args.device)
        x_norm, y_norm, norm = normalize_batch(x, y)

        for epoch in range(1, args.epochs + 1):
            pred_norm = model(x_norm)
            supervised = F.mse_loss(pred_norm, y_norm)
            pred_quantity = pred_norm * norm["y_std"].to(args.device) + norm["y_mean"].to(args.device)
            residual = poisson_residual_loss(
                pred_quantity, x, F, torch, args.residual_weight, args.target_mode, args.residual_mode
            )
            rel_pressure2 = pressure_relative_loss(pred_quantity, y, x, args.target_mode)
            loss = supervised + residual + args.pressure_rel_weight * rel_pressure2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            metric = float(rel_pressure2.detach().cpu())
            if metric < best_metric:
                best_metric = metric
                best_state = copy.deepcopy(model.state_dict())
            if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
                rel_pressure_rms = rel_pressure2.sqrt()
                print(
                    f"epoch={epoch} loss={loss.item():.6e} "
                    f"pressure_mse={supervised.item():.6e} "
                    f"pressure_rel2={rel_pressure2.item():.6e} "
                    f"pressure_rel_rms={rel_pressure_rms.item():.6e} "
                    f"residual={residual.item():.6e}",
                    flush=True,
                )
    else:
        if args.batch_size <= 0:
            raise SystemExit("--batch-size must be positive")
        if args.patches_per_epoch <= 0:
            raise SystemExit("--patches-per-epoch must be positive")
        norm = normalize_samples(samples, torch)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed)
        batches_per_epoch = (args.patches_per_epoch + args.batch_size - 1) // args.batch_size

        for epoch in range(1, args.epochs + 1):
            loss_sum = 0.0
            supervised_sum = 0.0
            residual_sum = 0.0
            rel2_sum = 0.0
            for batch_idx in range(batches_per_epoch):
                remaining = args.patches_per_epoch - batch_idx * args.batch_size
                current_batch = min(args.batch_size, remaining)
                x, y = make_patch_batch(samples, patch_size, current_batch, torch, generator)
                x = x.to(args.device)
                y = y.to(args.device)
                x_norm = (x - norm["x_mean"].to(args.device)) / norm["x_std"].to(args.device)
                y_norm = (y - norm["y_mean"].to(args.device)) / norm["y_std"].to(args.device)
                pred_norm = model(x_norm)
                supervised = F.mse_loss(pred_norm, y_norm)
                pred_quantity = pred_norm * norm["y_std"].to(args.device) + norm["y_mean"].to(args.device)
                residual = poisson_residual_loss(
                    pred_quantity, x, F, torch, args.residual_weight, args.target_mode, args.residual_mode
                )
                rel_pressure2 = pressure_relative_loss(pred_quantity, y, x, args.target_mode)
                loss = supervised + residual + args.pressure_rel_weight * rel_pressure2
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                loss_sum += float(loss.detach().cpu())
                supervised_sum += float(supervised.detach().cpu())
                residual_sum += float(residual.detach().cpu())
                rel2_sum += float(rel_pressure2.detach().cpu())

            loss_avg = loss_sum / batches_per_epoch
            supervised_avg = supervised_sum / batches_per_epoch
            residual_avg = residual_sum / batches_per_epoch
            rel2_avg = rel2_sum / batches_per_epoch
            if rel2_avg < best_metric:
                best_metric = rel2_avg
                best_state = copy.deepcopy(model.state_dict())
            if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
                print(
                    f"epoch={epoch} loss={loss_avg:.6e} "
                    f"pressure_mse={supervised_avg:.6e} "
                    f"pressure_rel2={rel2_avg:.6e} "
                    f"pressure_rel_rms={rel2_avg ** 0.5:.6e} "
                    f"residual={residual_avg:.6e}",
                    flush=True,
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save({
        "model_state": model.cpu().state_dict(),
        "normalization": norm,
        "channels": CHANNELS,
        "target_mode": args.target_mode,
        "residual_mode": args.residual_mode,
        "residual_weight": args.residual_weight,
        "architecture": args.architecture,
        "width": args.width,
        "depth": args.depth,
        "patch_size": patch_size,
        "patches_per_epoch": args.patches_per_epoch if patch_size is not None else 0,
        "init_model": str(args.init_model) if args.init_model is not None else "",
        "seed": args.seed,
        "best_pressure_rel2": best_metric,
        "samples": [sample[2] for sample in samples],
    }, args.out)
    print(f"wrote {args.out} best_pressure_rel2={best_metric:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
