#!/usr/bin/env python3
"""Stage-by-stage Torch vs book CPU/GPU pressure operator gate."""

from __future__ import annotations

import argparse
import array
import struct
import subprocess
import tempfile
from pathlib import Path

import torch

from book_pressure_operator import EI, BookPressureStages, fixed_point_loss, t_book
from h_initializer_model import HInitializer, pressure_from_h


def deterministic_inputs(lx: int, ly: int, lz: int):
    hh = torch.empty((1, 15, lz, ly, lx), dtype=torch.float64)
    rho = torch.empty((1, 1, lz, ly, lx), dtype=torch.float64)
    u = torch.empty_like(rho)
    v = torch.empty_like(rho)
    w = torch.empty_like(rho)
    for z in range(lz):
        for y in range(ly):
            for x in range(lx):
                rho[0, 0, z, y, x] = 1.0 + ((17 * x + 11 * y + 5 * z) % 37) / 10.0
                u[0, 0, z, y, x] = 0.01 * (x + 2.0 * y - 3.0 * z)
                v[0, 0, z, y, x] = -0.015 * (2.0 * x - y + z)
                w[0, 0, z, y, x] = 0.02 * (-x + 3.0 * y + 2.0 * z)
                base_p = 0.2 + 0.003 * x - 0.002 * y + 0.004 * z
                for q in range(15):
                    hh[0, q, z, y, x] = (
                        EI[q] * base_p
                        + 1.0e-4 * (q + 1.0) * (1.0 + x + 2.0 * y + 3.0 * z)
                    )
    return hh, rho, u, v, w


def read_values(handle, count: int) -> torch.Tensor:
    values = array.array("d")
    values.fromfile(handle, count)
    if len(values) != count:
        raise ValueError("truncated book operator stage dump")
    return torch.frombuffer(values, dtype=torch.float64).clone()


def read_stages(handle, lx: int, ly: int, lz: int) -> BookPressureStages:
    n = lx * ly * lz

    def scalar() -> torch.Tensor:
        return read_values(handle, n).view(1, 1, lz, ly, lx)

    def distribution() -> torch.Tensor:
        return read_values(handle, n * 15).view(lz, ly, lx, 15).permute(3, 0, 1, 2).unsqueeze(0)

    return BookPressureStages(scalar(), distribution(), distribution(), distribution(), scalar())


def load_dump(path: Path):
    with path.open("rb") as handle:
        if handle.read(8) != b"BOOKT1\0\0":
            raise ValueError("invalid book operator stage dump")
        lx, ly, lz = struct.unpack("<iii", handle.read(12))
        cpu = read_stages(handle, lx, ly, lz)
        gpu = read_stages(handle, lx, ly, lz)
        if handle.read(1):
            raise ValueError("unexpected trailing bytes in book operator stage dump")
    return lx, ly, lz, cpu, gpu


def compare(label: str, actual: BookPressureStages, expected: BookPressureStages, tolerance: float) -> None:
    for stage in BookPressureStages._fields:
        error = (getattr(actual, stage).cpu() - getattr(expected, stage)).abs().max().item()
        status = "PASS" if error <= tolerance else "FAIL"
        print(f"{label},{stage},{error:.12e},{status}")
        if error > tolerance:
            raise AssertionError(f"{label} {stage} max error {error} > {tolerance}")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp-test", type=Path, default=root / "GPU" / "book_pressure_step_test")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="book_pressure_") as tmp:
        dump = Path(tmp) / "stages.bin"
        subprocess.run([str(args.cpp_test.resolve()), str(dump)], check=True)
        lx, ly, lz, cpu_expected, gpu_expected = load_dump(dump)

    inputs = deterministic_inputs(lx, ly, lz)
    torch_cpu = t_book(*inputs)
    compare("torch_cpu_vs_book_cpu", torch_cpu, cpu_expected, 1.0e-12)
    compare("torch_cpu_vs_solver_gpu", torch_cpu, gpu_expected, 1.0e-12)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Torch GPU consistency gate")
    torch_gpu = t_book(*(value.cuda() for value in inputs))
    compare("torch_gpu_vs_book_cpu", torch_gpu, cpu_expected, 1.0e-12)

    hh_grad = inputs[0].cuda().requires_grad_(True)
    loss = fixed_point_loss(hh_grad, *(value.cuda() for value in inputs[1:]))
    loss.backward()
    if hh_grad.grad is None or not torch.isfinite(hh_grad.grad).all():
        raise AssertionError("book fixed-point loss gradient is not finite")
    print(f"autograd,fixed_point_loss,{loss.item():.12e},PASS")

    model = HInitializer(width=8, depth=1).cuda()
    div_source = torch_cpu.divergence
    model_input = torch.cat((inputs[1], inputs[2], inputs[3], inputs[4], inputs[1], inputs[0].sum(1, keepdim=True), div_source), dim=1).float().cuda()
    hh_pred = model(model_input)
    pressure = pressure_from_h(hh_pred)
    if hh_pred.shape[1] != 15 or not torch.equal(pressure, hh_pred.sum(dim=1, keepdim=True)):
        raise AssertionError("15-channel model pressure is not the exact h_i sum")
    print("model,15_channel_h_and_exact_pressure_sum,0.000000000000e+00,PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
