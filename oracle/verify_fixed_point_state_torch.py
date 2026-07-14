#!/usr/bin/env python3
"""Torch float64 audit of a real production D3Q15 Poisson checkpoint.

The binary layout is intentionally identical to the CUDA physical-domain
layout: scalar cell index ``(z * ny + y) * nx + x`` and distribution index
``cell * 15 + q``.  The audited map is the production split path
collision -> periodic pull stream -> x-slip boundary -> p=sum_q(h_q).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


MAGIC = b"CLBMK01\0"
HEADER = struct.Struct("<10I3Q2d")
ENDIAN_TAG = 0x01020304
FLOAT_BYTES = 8
Q = 15
FIELD_COUNT = 6
LAYOUT_CELL_MAJOR_Q_MINOR = 1
TOLERANCE = 1.0e-12

EX = (0, 1, 0, 0, -1, 0, 0, 1, -1, 1, 1, -1, 1, -1, -1)
EY = (0, 0, 1, 0, 0, -1, 0, 1, 1, -1, 1, -1, -1, 1, -1)
EZ = (0, 0, 0, 1, 0, 0, -1, 1, 1, 1, -1, -1, -1, -1, 1)
EI = (
    2.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 72.0,
    1.0 / 72.0,
    1.0 / 72.0,
    1.0 / 72.0,
    1.0 / 72.0,
    1.0 / 72.0,
    1.0 / 72.0,
    1.0 / 72.0,
)
BOUNCE = ((1, 4), (7, 8), (9, 14), (10, 13), (12, 11))
POS_TO_NEG = {positive: negative for positive, negative in BOUNCE}
NEG_TO_POS = {negative: positive for positive, negative in BOUNCE}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_state(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        magic = stream.read(8)
        if magic != MAGIC:
            raise RuntimeError(f"{path}: bad magic {magic!r}, expected {MAGIC!r}")
        raw_header = stream.read(HEADER.size)
        if len(raw_header) != HEADER.size:
            raise RuntimeError(f"{path}: truncated header")
        (
            endian,
            float_bytes,
            nx,
            ny,
            nz,
            nz_total,
            q_count,
            step,
            field_count,
            layout,
            iteration,
            cells,
            payload_values,
            gauge_target,
            pressure_relax_scale,
        ) = HEADER.unpack(raw_header)

        expected_cells = nx * ny * nz
        expected_values = 20 * expected_cells
        checks = {
            "endian": endian == ENDIAN_TAG,
            "float_bytes": float_bytes == FLOAT_BYTES,
            "positive_dimensions": nx > 1 and ny > 0 and nz > 0,
            "nz_total": nz_total == nz + 2,
            "q": q_count == Q,
            "field_count": field_count == FIELD_COUNT,
            "layout": layout == LAYOUT_CELL_MAJOR_Q_MINOR,
            "cell_count": cells == expected_cells,
            "payload_values": payload_values == expected_values,
            "gauge_target_finite": math.isfinite(gauge_target),
            "pressure_relax_scale_finite": math.isfinite(pressure_relax_scale),
            "strict_unit_pressure_relax": pressure_relax_scale == 1.0,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(f"{path}: header contract failed: {', '.join(failed)}")

        payload = stream.read(payload_values * FLOAT_BYTES)
        if len(payload) != payload_values * FLOAT_BYTES:
            raise RuntimeError(f"{path}: truncated payload")
        if stream.read(1):
            raise RuntimeError(f"{path}: unexpected trailing bytes")

    values = np.frombuffer(payload, dtype="<f8").copy()
    offset = 0

    def scalar_field() -> np.ndarray:
        nonlocal offset
        result = values[offset : offset + cells].reshape(nz, ny, nx)
        offset += cells
        return result

    pressure = scalar_field()
    rho = scalar_field()
    u_x = scalar_field()
    v_y = scalar_field()
    w_z = scalar_field()
    h = values[offset : offset + Q * cells].reshape(nz, ny, nx, Q)
    offset += Q * cells
    if offset != payload_values:
        raise AssertionError("internal payload parser mismatch")

    arrays = {
        "p": pressure,
        "rho": rho,
        "u_x": u_x,
        "v_y": v_y,
        "w_z": w_z,
        "h": h,
    }
    finite = {name: bool(np.isfinite(value).all()) for name, value in arrays.items()}
    if not all(finite.values()):
        bad = [name for name, passed in finite.items() if not passed]
        raise RuntimeError(f"{path}: NaN/Inf in {', '.join(bad)}")
    if np.any(rho == 0.0):
        raise RuntimeError(f"{path}: rho contains zero")

    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "header": {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "nz_total": nz_total,
            "q": q_count,
            "step": step,
            "iteration": iteration,
            "cells": cells,
            "payload_values": payload_values,
            "gauge_target": gauge_target,
            "pressure_relax_scale": pressure_relax_scale,
            "layout": "cell-major/q-minor",
        },
        "header_contract": checks,
        "input_finite": finite,
        "arrays": arrays,
    }


def sum_q(values: torch.Tensor) -> torch.Tensor:
    """Match computePressureKernel's explicit q=0..14 addition order."""
    result = values[..., 0]
    for q in range(1, Q):
        result = result + values[..., q]
    return result


def fixed_map(
    h: torch.Tensor,
    pressure: torch.Tensor,
    rho: torch.Tensor,
    div_source: torch.Tensor,
    ei: torch.Tensor,
    pressure_relax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tau_h = 1.0 / rho + 0.5
    collision = (
        h
        - pressure_relax_scale
        * (h - ei * pressure[..., None])
        / tau_h[..., None]
        - ei * div_source[..., None] / 3.0
    )

    # CUDA streamKernel is periodic pull: dst(x,y,z,q)=src(x-e_q,y-e_q,z-e_q,q).
    streamed = torch.stack(
        [
            torch.roll(
                collision[..., q],
                shifts=(EZ[q], EY[q], EX[q]),
                dims=(0, 1, 2),
            )
            for q in range(Q)
        ],
        dim=-1,
    )

    # Match slipBounceBackKernel without in-place aliases, retaining autograd.
    boundary_fields = []
    for q in range(Q):
        field = streamed[..., q]
        if q in POS_TO_NEG:
            field = torch.cat(
                (streamed[:, :, 0, POS_TO_NEG[q]].unsqueeze(2), field[:, :, 1:]),
                dim=2,
            )
        elif q in NEG_TO_POS:
            field = torch.cat(
                (field[:, :, :-1], streamed[:, :, -1, NEG_TO_POS[q]].unsqueeze(2)),
                dim=2,
            )
        boundary_fields.append(field)
    boundary = torch.stack(boundary_fields, dim=-1)
    pressure_image = sum_q(boundary)
    return collision, streamed, boundary, pressure_image


def relative(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(
        denominator > 0.0,
        numerator / denominator,
        torch.where(
            numerator == 0.0,
            torch.zeros_like(numerator),
            torch.full_like(numerator, float("inf")),
        ),
    )


def residual_terms(
    h: torch.Tensor,
    image_h: torch.Tensor,
    pressure: torch.Tensor,
    gauge_target: float,
    ei: torch.Tensor,
) -> tuple[dict[str, float | bool], torch.Tensor, torch.Tensor]:
    h_sum = sum_q(h)
    r_h = h - image_h
    r_p = pressure - h_sum
    gauge = torch.as_tensor(gauge_target, dtype=torch.float64, device=h.device)
    h_l1 = torch.sum(torch.abs(r_h))
    h_scale = torch.maximum(
        torch.sum(torch.abs(h - ei * gauge)),
        torch.sum(torch.abs(image_h - ei * gauge)),
    )
    p_l1 = torch.sum(torch.abs(r_p))
    p_scale = torch.maximum(
        torch.sum(torch.abs(pressure - gauge)),
        torch.sum(torch.abs(h_sum - gauge)),
    )
    h_rel = relative(h_l1, h_scale)
    p_rel = relative(p_l1, p_scale)
    combined = torch.maximum(h_rel, p_rel)
    all_finite = bool(torch.isfinite(r_h).all().item() and torch.isfinite(r_p).all().item())
    terms: dict[str, float | bool] = {
        "r_h_l1": float(h_l1.detach().cpu()),
        "r_h_scale": float(h_scale.detach().cpu()),
        "r_h_relative": float(h_rel.detach().cpu()),
        "r_h_linf": float(torch.max(torch.abs(r_h)).detach().cpu()),
        "r_h_rms": float(torch.sqrt(torch.mean(r_h * r_h)).detach().cpu()),
        "r_p_l1": float(p_l1.detach().cpu()),
        "r_p_scale": float(p_scale.detach().cpu()),
        "r_p_relative": float(p_rel.detach().cpu()),
        "r_p_linf": float(torch.max(torch.abs(r_p)).detach().cpu()),
        "r_p_rms": float(torch.sqrt(torch.mean(r_p * r_p)).detach().cpu()),
        "combined_relative": float(combined.detach().cpu()),
        "finite": all_finite and bool(torch.isfinite(combined).item()),
    }
    return terms, r_h, r_p


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a - b)).detach().cpu())


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_backend(
    device: torch.device,
    state: dict[str, Any],
    direction_np: np.ndarray,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    arrays = state["arrays"]
    header = state["header"]
    pressure = torch.from_numpy(arrays["p"]).to(device=device, dtype=torch.float64)
    rho = torch.from_numpy(arrays["rho"]).to(device=device, dtype=torch.float64)
    u_x = torch.from_numpy(arrays["u_x"]).to(device=device, dtype=torch.float64)
    v_y = torch.from_numpy(arrays["v_y"]).to(device=device, dtype=torch.float64)
    w_z = torch.from_numpy(arrays["w_z"]).to(device=device, dtype=torch.float64)
    h = torch.from_numpy(arrays["h"]).to(device=device, dtype=torch.float64)
    direction = torch.from_numpy(direction_np).to(device=device, dtype=torch.float64)
    ei = torch.tensor(EI, device=device, dtype=torch.float64)
    div_source = u_x + v_y + w_z
    relax = float(header["pressure_relax_scale"])
    gauge_target = float(header["gauge_target"])

    synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        stages = fixed_map(h, pressure, rho, div_source, ei, relax)
        collision, streamed, boundary, pressure_image = stages
        terms, r_h, r_p = residual_terms(h, boundary, pressure, gauge_target, ei)

        gauge_shift = torch.as_tensor(
            0.2718281828459045, device=device, dtype=torch.float64
        )
        shifted_stages = fixed_map(
            h + ei * gauge_shift,
            pressure + gauge_shift,
            rho,
            div_source,
            ei,
            relax,
        )
        shifted_collision, shifted_stream, shifted_boundary, shifted_pressure_image = shifted_stages
        shifted_terms, shifted_r_h, shifted_r_p = residual_terms(
            h + ei * gauge_shift,
            shifted_boundary,
            pressure + gauge_shift,
            gauge_target + float(gauge_shift.cpu()),
            ei,
        )
        gauge_errors = {
            "collision_covariance_max_abs": max_abs(
                shifted_collision, collision + ei * gauge_shift
            ),
            "stream_covariance_max_abs": max_abs(
                shifted_stream, streamed + ei * gauge_shift
            ),
            "boundary_covariance_max_abs": max_abs(
                shifted_boundary, boundary + ei * gauge_shift
            ),
            "pressure_image_covariance_max_abs": max_abs(
                shifted_pressure_image, pressure_image + gauge_shift
            ),
            "r_h_invariance_max_abs": max_abs(shifted_r_h, r_h),
            "r_p_invariance_max_abs": max_abs(shifted_r_p, r_p),
            "combined_relative_invariance_abs": abs(
                float(shifted_terms["combined_relative"])
                - float(terms["combined_relative"])
            ),
        }
        pressure_mean = torch.mean(pressure)
        gauge_target_error = float(torch.abs(pressure_mean - gauge_target).cpu())

    def residual_function(candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image = fixed_map(h, candidate, rho, div_source, ei, relax)[2]
        return h - image, candidate - sum_q(h)

    # A fixed, nonconstant, zero-mean direction tests the pressure Jacobian.
    (_, _), (jvp_h, jvp_p) = torch.func.jvp(
        residual_function, (pressure,), (direction,)
    )
    jvp_finite = bool(torch.isfinite(jvp_h).all().item() and torch.isfinite(jvp_p).all().item())
    jvp_linf = max(
        float(torch.max(torch.abs(jvp_h)).detach().cpu()),
        float(torch.max(torch.abs(jvp_p)).detach().cpu()),
    )
    jvp_l2 = float(
        torch.sqrt(torch.sum(jvp_h * jvp_h) + torch.sum(jvp_p * jvp_p))
        .detach()
        .cpu()
    )

    # An explicit VJP probe establishes a finite, nonzero pressure gradient
    # independently of the current residual magnitude.
    pressure_probe = pressure.detach().clone().requires_grad_(True)
    probe_h, probe_p = residual_function(pressure_probe)
    q_probe = torch.arange(1, Q + 1, device=device, dtype=torch.float64) / Q
    objective = torch.mean(probe_h * direction[..., None] * q_probe) + torch.mean(
        probe_p * direction
    )
    (gradient,) = torch.autograd.grad(objective, pressure_probe)
    gradient_finite = bool(torch.isfinite(gradient).all().item())
    gradient_linf = float(torch.max(torch.abs(gradient)).detach().cpu())
    gradient_l2 = float(torch.linalg.vector_norm(gradient).detach().cpu())

    synchronize(device)
    elapsed = time.perf_counter() - start
    metrics = {
        "device": str(device),
        "residual_terms": terms,
        "gauge_target": {
            "pressure_mean": float(pressure_mean.cpu()),
            "target": gauge_target,
            "absolute_error": gauge_target_error,
        },
        "gauge_full_map": gauge_errors,
        "pressure_jvp": {
            "finite": jvp_finite,
            "nonzero": jvp_linf > 0.0,
            "linf": jvp_linf,
            "l2": jvp_l2,
        },
        "pressure_vjp_gradient": {
            "finite": gradient_finite,
            "nonzero": gradient_linf > 0.0,
            "linf": gradient_linf,
            "l2": gradient_l2,
        },
        "elapsed_seconds": elapsed,
    }
    named_tensors = {
        "collision": collision.detach(),
        "stream": streamed.detach(),
        "boundary": boundary.detach(),
        "pressure_image": pressure_image.detach(),
        "r_h": r_h.detach(),
        "r_p": r_p.detach(),
    }
    return metrics, named_tensors


def audit_state(state: dict[str, Any]) -> dict[str, Any]:
    nz, ny, nx = state["arrays"]["p"].shape
    z, y, x = np.indices((nz, ny, nx), dtype=np.int64)
    direction = (((3 * x + 5 * y + 7 * z) % 17).astype(np.float64) - 8.0) / 8.0
    direction -= np.mean(direction, dtype=np.float64)
    if not np.isfinite(direction).all() or np.max(np.abs(direction)) == 0.0:
        raise RuntimeError("invalid deterministic JVP direction")

    cpu_metrics, cpu_tensors = run_backend(torch.device("cpu"), state, direction)
    cuda_metrics, cuda_tensors = run_backend(torch.device("cuda"), state, direction)

    component_errors: dict[str, float] = {}
    for name, cpu_value in cpu_tensors.items():
        component_errors[name + "_max_abs"] = max_abs(
            cuda_tensors[name], cpu_value.to(device="cuda")
        )

    gauge_errors = list(cpu_metrics["gauge_full_map"].values()) + list(
        cuda_metrics["gauge_full_map"].values()
    )
    numerical_errors = list(component_errors.values()) + gauge_errors + [
        cpu_metrics["gauge_target"]["absolute_error"],
        cuda_metrics["gauge_target"]["absolute_error"],
    ]
    worst = max(numerical_errors)
    finite_and_nonzero = all(
        bool(metrics[section][key])
        for metrics in (cpu_metrics, cuda_metrics)
        for section in ("pressure_jvp", "pressure_vjp_gradient")
        for key in ("finite", "nonzero")
    )
    residuals_finite = bool(
        cpu_metrics["residual_terms"]["finite"]
        and cuda_metrics["residual_terms"]["finite"]
    )
    passed = worst <= TOLERANCE and finite_and_nonzero and residuals_finite
    return {
        "state_file": state["path"],
        "state_sha256": state["sha256"],
        "header": state["header"],
        "header_contract": state["header_contract"],
        "input_finite": state["input_finite"],
        "torch_cpu": cpu_metrics,
        "torch_cuda": cuda_metrics,
        "cpu_cuda_component_max_abs": component_errors,
        "worst_gated_numerical_error": worst,
        "tolerance": TOLERANCE,
        "finite_nonzero_gate": finite_and_nonzero and residuals_finite,
        "pass": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--state-file",
        type=Path,
        action="append",
        required=True,
        help="CLBMK01 real-state dump; repeat for multiple development states",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Torch CUDA is required for the CPU/CUDA Stage 0 gate")
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)

    reports = [audit_state(read_state(path)) for path in args.state_file]
    result = {
        "schema": "fixed-point-real-state-torch-audit-v1",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_device": torch.cuda.get_device_name(0),
        "dtype": "float64",
        "tolerance": TOLERANCE,
        "source_indexing": {
            "cell": "(z * ny + y) * nx + x",
            "distribution": "cell * 15 + q",
            "stream": "periodic pull from (x-ex[q], y-ey[q], z-ez[q])",
            "boundary": "x-slip pairs (1,4),(7,8),(9,14),(10,13),(12,11)",
        },
        "formula": {
            "collision": "h-E_relax*(h-E_i*p)/(1/rho+1/2)-(E_i/3)*(u_x+v_y+w_z)",
            "r_h": "h-BS[C(h,p)]",
            "r_p": "p-sum_i(h_i)",
        },
        "states": reports,
        "worst_gated_numerical_error": max(
            report["worst_gated_numerical_error"] for report in reports
        ),
        "pass": all(report["pass"] for report in reports),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if result["pass"] else 2)


if __name__ == "__main__":
    main()
