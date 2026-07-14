#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch


EX = (0, 1, 0, 0, -1, 0, 0, 1, -1, 1, 1, -1, 1, -1, -1)
EY = (0, 0, 1, 0, 0, -1, 0, 1, 1, -1, 1, -1, -1, 1, -1)
EZ = (0, 0, 0, 1, 0, 0, -1, 1, 1, 1, -1, -1, -1, -1, 1)
EI = (2 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9,
      1 / 72, 1 / 72, 1 / 72, 1 / 72, 1 / 72, 1 / 72, 1 / 72, 1 / 72)
BOUNCE = ((1, 4), (7, 8), (9, 14), (10, 13), (12, 11))


def read_dump(path: Path):
    with path.open("rb") as stream:
        magic = stream.read(8)
        if magic != b"TBOOKA1\0":
            raise RuntimeError(f"bad T_book dump magic: {magic!r}")
        nx, ny, nz, count = struct.unpack("<4I", stream.read(16))
        if count != 16:
            raise RuntimeError(f"unexpected array count: {count}")
        n = nx * ny * nz
        sizes = (n, n, n * 15, n, n, n, n * 15, n * 15,
                 n * 15, n, n * 15, n * 15, n * 15, n, n * 15, n)
        arrays = []
        for size in sizes:
            raw = stream.read(size * 8)
            if len(raw) != size * 8:
                raise RuntimeError("truncated T_book dump")
            arrays.append(np.frombuffer(raw, dtype="<f8").copy())
        if stream.read(1):
            raise RuntimeError("unexpected trailing bytes in T_book dump")
    return (nx, ny, nz), arrays


def max_abs(a: torch.Tensor, b: np.ndarray) -> float:
    reference = torch.from_numpy(b).to(device=a.device, dtype=torch.float64).reshape(a.shape)
    return float(torch.max(torch.abs(a - reference)).cpu())


def run_backend(device: torch.device, dims, arrays):
    nx, ny, nz = dims
    (rho_np, p_np, h_np, ux_np, vy_np, wz_np,
     cpu_collision, cpu_stream, cpu_boundary, cpu_pressure,
     gpu_collision, gpu_stream, gpu_boundary, gpu_pressure,
     gpu_onepass_h, gpu_onepass_p) = arrays
    shape = (nz, ny, nx)
    rho = torch.from_numpy(rho_np).to(device=device, dtype=torch.float64).reshape(shape)
    pressure = torch.from_numpy(p_np).to(device=device, dtype=torch.float64).reshape(shape)
    h = torch.from_numpy(h_np).to(device=device, dtype=torch.float64).reshape(*shape, 15)
    div_u = (
        torch.from_numpy(ux_np).to(device=device, dtype=torch.float64).reshape(shape)
        + torch.from_numpy(vy_np).to(device=device, dtype=torch.float64).reshape(shape)
        + torch.from_numpy(wz_np).to(device=device, dtype=torch.float64).reshape(shape)
    )
    ei = torch.tensor(EI, device=device, dtype=torch.float64)
    tau_h = 1.0 / rho + 0.5
    collision = h - (h - ei * pressure[..., None]) / tau_h[..., None] - ei * div_u[..., None] / 3.0
    streamed = torch.empty_like(collision)
    for q in range(15):
        streamed[..., q] = torch.roll(collision[..., q], shifts=(EZ[q], EY[q], EX[q]), dims=(0, 1, 2))
    boundary = streamed.clone()
    for positive, negative in BOUNCE:
        boundary[:, :, 0, positive] = boundary[:, :, 0, negative]
        boundary[:, :, nx - 1, negative] = boundary[:, :, nx - 1, positive]
    pressure_out = boundary.sum(dim=-1)
    source_sum = (-ei * div_u[..., None] / 3.0).sum(dim=-1)

    projected_div_u = div_u - div_u.mean()
    projected_collision = h - (h - ei * pressure[..., None]) / tau_h[..., None] - ei * projected_div_u[..., None] / 3.0
    projected_stream = torch.empty_like(projected_collision)
    for q in range(15):
        projected_stream[..., q] = torch.roll(projected_collision[..., q], shifts=(EZ[q], EY[q], EX[q]), dims=(0, 1, 2))
    projected_boundary = projected_stream.clone()
    for positive, negative in BOUNCE:
        projected_boundary[:, :, 0, positive] = projected_boundary[:, :, 0, negative]
        projected_boundary[:, :, nx - 1, negative] = projected_boundary[:, :, nx - 1, positive]
    projected_pressure = projected_boundary.sum(dim=-1)

    def gradient(field):
        gx = torch.empty_like(field)
        gx[:, :, 0] = 0.0
        gx[:, :, nx - 1] = 0.0
        gx[:, :, 1:nx - 1] = 0.5 * (field[:, :, 2:nx] - field[:, :, 0:nx - 2])
        gy = 0.5 * (torch.roll(field, -1, 1) - torch.roll(field, 1, 1))
        gz = 0.5 * (torch.roll(field, -1, 0) - torch.roll(field, 1, 0))
        return torch.stack((gx, gy, gz), dim=-1)

    raw_gradient = gradient(pressure_out)
    projected_gradient = gradient(projected_pressure)
    gauge_shift = torch.tensor(0.123456789, device=device, dtype=torch.float64)
    gauged_pressure = projected_pressure - gauge_shift
    gauged_h = projected_boundary - ei * gauge_shift
    gauged_gradient = gradient(gauged_pressure)

    fixed_h = h - boundary
    fixed_p = pressure - h.sum(dim=-1)
    cpu_fixed_h = h_np.reshape(*shape, 15) - cpu_boundary.reshape(*shape, 15)
    cpu_fixed_p = p_np.reshape(shape) - h_np.reshape(*shape, 15).sum(axis=-1)
    gauge = pressure.mean()

    full_gauge_shift = torch.tensor(0.2718281828459045, device=device, dtype=torch.float64)
    shifted_h = h + ei * full_gauge_shift
    shifted_pressure = pressure + full_gauge_shift
    shifted_collision = (
        shifted_h
        - (shifted_h - ei * shifted_pressure[..., None]) / tau_h[..., None]
        - ei * div_u[..., None] / 3.0
    )
    shifted_stream = torch.empty_like(shifted_collision)
    for q in range(15):
        shifted_stream[..., q] = torch.roll(
            shifted_collision[..., q], shifts=(EZ[q], EY[q], EX[q]), dims=(0, 1, 2)
        )
    shifted_boundary = shifted_stream.clone()
    for positive, negative in BOUNCE:
        shifted_boundary[:, :, 0, positive] = shifted_boundary[:, :, 0, negative]
        shifted_boundary[:, :, nx - 1, negative] = shifted_boundary[:, :, nx - 1, positive]
    shifted_fixed_h = shifted_h - shifted_boundary
    shifted_fixed_p = shifted_pressure - shifted_h.sum(dim=-1)

    def fixed_relative(live_h, image_h, live_p, live_gauge):
        h_num = torch.sum(torch.abs(live_h - image_h))
        h_den = torch.maximum(
            torch.sum(torch.abs(live_h - ei * live_gauge)),
            torch.sum(torch.abs(image_h - ei * live_gauge)),
        )
        p_sum = live_h.sum(dim=-1)
        p_num = torch.sum(torch.abs(live_p - p_sum))
        p_den = torch.maximum(
            torch.sum(torch.abs(live_p - live_gauge)),
            torch.sum(torch.abs(p_sum - live_gauge)),
        )
        h_rel = torch.where(h_den > 0, h_num / h_den,
                            torch.where(h_num == 0, h_num, torch.full_like(h_num, float("inf"))))
        p_rel = torch.where(p_den > 0, p_num / p_den,
                            torch.where(p_num == 0, p_num, torch.full_like(p_num, float("inf"))))
        return torch.maximum(h_rel, p_rel)

    fixed_relative_raw = fixed_relative(h, boundary, pressure, gauge)
    fixed_relative_shifted = fixed_relative(
        shifted_h, shifted_boundary, shifted_pressure, gauge + full_gauge_shift
    )
    return {
        "collision_vs_cpu": max_abs(collision, cpu_collision),
        "stream_vs_cpu": max_abs(streamed, cpu_stream),
        "boundary_vs_cpu": max_abs(boundary, cpu_boundary),
        "pressure_vs_cpu": max_abs(pressure_out, cpu_pressure),
        "boundary_vs_gpu_split": max_abs(boundary, gpu_boundary),
        "pressure_vs_gpu_split": max_abs(pressure_out, gpu_pressure),
        "boundary_vs_gpu_onepass": max_abs(boundary, gpu_onepass_h),
        "pressure_vs_gpu_onepass": max_abs(pressure_out, gpu_onepass_p),
        "source_moment_max_abs": float(torch.max(torch.abs(source_sum + div_u / 3.0)).cpu()),
        "projected_source_mean_abs": float(torch.abs(projected_div_u.mean() / 3.0).cpu()),
        "source_projection_gradient_max_abs": float(torch.max(torch.abs(raw_gradient - projected_gradient)).cpu()),
        "gauge_gradient_max_abs": float(torch.max(torch.abs(projected_gradient - gauged_gradient)).cpu()),
        "gauge_correct_uvw_max_abs": float(torch.max(torch.abs((projected_gradient - gauged_gradient) / rho[..., None])).cpu()),
        "gauge_p_sum_h_max_abs": float(torch.max(torch.abs(gauged_h.sum(dim=-1) - gauged_pressure)).cpu()),
        "fixed_point_h_vs_cpu_max_abs": max_abs(fixed_h, cpu_fixed_h),
        "fixed_point_h_vs_gpu_max_abs": max_abs(
            fixed_h, h_np.reshape(*shape, 15) - gpu_boundary.reshape(*shape, 15)
        ),
        "fixed_point_p_vs_cpu_max_abs": max_abs(fixed_p, cpu_fixed_p),
        "fixed_point_gauge_h_max_abs": float(
            torch.max(torch.abs(fixed_h - shifted_fixed_h)).cpu()
        ),
        "fixed_point_gauge_p_max_abs": float(
            torch.max(torch.abs(fixed_p - shifted_fixed_p)).cpu()
        ),
        "fixed_point_relative_gauge_abs": float(
            torch.abs(fixed_relative_raw - fixed_relative_shifted).cpu()
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    dims, arrays = read_dump(args.dump)
    results = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cpu": run_backend(torch.device("cpu"), dims, arrays),
    }
    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is required for the staged audit")
    results["torch_cuda_device"] = torch.cuda.get_device_name(0)
    results["torch_cuda"] = run_backend(torch.device("cuda"), dims, arrays)
    worst = max(value for key in ("torch_cpu", "torch_cuda") for value in results[key].values())
    results["worst_max_abs"] = worst
    results["tolerance"] = 1.0e-12
    results["pass"] = worst <= results["tolerance"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    raise SystemExit(0 if results["pass"] else 2)


if __name__ == "__main__":
    main()
