#!/usr/bin/env python3
"""Closed-loop multi-step benchmark for the safe pressure initializer."""

from __future__ import annotations

import argparse
import csv
import math
import os
import struct
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from input_quality_gate import field_metrics_from_data, metric_ranges, score
from h_initializer_model import HInitializer, pressure_from_h
from run_pressure_init_gate import parse_perf
from run_multistep_safe_replay import (
    compare_fields,
    method_rows,
    parse_step_diagnostics,
    parse_steps,
    run_solver,
    step_name,
)
from tecplot_io import (
    read_feature_snapshot,
    read_tecplot,
    write_poisson_state,
    write_pressure_initializer,
)
from train_pressure_initializer import CHANNELS, make_model, require_torch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def wait_for_stable_file(path: Path, proc: subprocess.Popen[object], timeout_s: float) -> None:
    start = time.monotonic()
    last_size = -1
    stable_count = 0
    while True:
        if path.exists():
            size = path.stat().st_size
            expected_size = 0
            if path.suffix == ".bin" and size >= 24:
                with path.open("rb") as f:
                    header = f.read(24)
                if len(header) == 24 and header[:8] in (b"PINNF1\0\0", b"PINNF2\0\0"):
                    lx, ly, lz, nfields = struct.unpack("<iiii", header[8:])
                    expected_size = 24 + lx * ly * lz * nfields * 4
            if expected_size > 0 and size < expected_size:
                stable_count = 0
                last_size = size
                time.sleep(0.05)
                continue
            if size > 0 and size == last_size:
                stable_count += 1
                if stable_count >= 2:
                    return
            else:
                stable_count = 0
                last_size = size
        if proc.poll() is not None:
            raise RuntimeError(f"solver exited before writing {path}")
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.05)


def write_skip(path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("skip\n")
    tmp.replace(path)


def candidate_steps(args: argparse.Namespace) -> list[int]:
    if args.pinn_steps.strip():
        steps: list[int] = []
        for item in args.pinn_steps.split(","):
            item = item.strip()
            if not item:
                continue
            step = int(item)
            if step <= 0 or step > args.steps:
                raise SystemExit(f"--pinn-steps value {step} outside [1, {args.steps}]")
            steps.append(step)
        return sorted(dict.fromkeys(steps))
    if args.pinn_start_step <= 0:
        raise SystemExit("--pinn-start-step must be positive")
    if args.pinn_interval <= 0:
        raise SystemExit("--pinn-interval must be positive")
    last_step = args.pinn_max_step if args.pinn_max_step > 0 else args.steps
    last_step = min(last_step, args.steps)
    if args.pinn_start_step > last_step:
        return []
    return list(range(args.pinn_start_step, last_step + 1, args.pinn_interval))


def fields_are_finite(data: dict[str, object]) -> tuple[bool, str]:
    fields = data["fields"]  # type: ignore[assignment]
    reasons: list[str] = []
    for name, values in fields.items():
        bad = sum(1 for value in values if not math.isfinite(value))  # type: ignore[union-attr]
        if bad:
            reasons.append(f"{name} nonfinite={bad}")
    return not reasons, "; ".join(reasons)


def write_dict_rows(path: Path, rows: list[dict[str, object]], keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pair_snapshot_suffix(pair_format: str) -> str:
    if pair_format == "features":
        return "bin"
    if pair_format == "tecplot":
        return "plt"
    raise ValueError(f"unsupported Poisson pair format: {pair_format}")


def read_pair_snapshot(path: Path, pair_format: str) -> dict[str, object]:
    if pair_format == "features":
        return read_feature_snapshot(path)
    if pair_format == "tecplot":
        return read_tecplot(path)
    raise ValueError(f"unsupported Poisson pair format: {pair_format}")


class PressurePredictor:
    def __init__(self, model_path: Path, device: str):
        torch, nn, _ = require_torch()
        self.torch = torch
        self.device = device
        checkpoint = torch.load(model_path, map_location=device)
        self.channels = tuple(checkpoint.get("channels", CHANNELS))
        self.model = make_model(
            nn,
            len(self.channels),
            checkpoint.get("architecture", "simple"),
            int(checkpoint.get("width", 32)),
            int(checkpoint.get("depth", 4)),
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.norm = checkpoint["normalization"]
        self.target_mode = checkpoint.get("target_mode", "absolute")

    def write_absolute_from_data(
        self,
        data: dict[str, object],
        out_bin: Path,
        prediction_scale: float = 1.0,
    ) -> None:
        lx = int(data["lx"])
        ly = int(data["ly"])
        lz = int(data["lz"])
        fields = data["fields"]  # type: ignore[assignment]
        torch = self.torch
        x = torch.stack([
            torch.tensor(fields[name], dtype=torch.float32).view(lz, ly, lx)  # type: ignore[index]
            for name in self.channels
        ]).unsqueeze(0).to(self.device)
        x_norm = (x - self.norm["x_mean"].to(self.device)) / self.norm["x_std"].to(self.device)
        with torch.no_grad():
            pred_norm = self.model(x_norm)
            pred = pred_norm * self.norm["y_std"].to(self.device) + self.norm["y_mean"].to(self.device)

        predicted = pred.squeeze(0).squeeze(0).cpu().reshape(-1).tolist()
        current = list(fields["press"])  # type: ignore[index]
        if self.target_mode == "delta":
            values = [c + prediction_scale * delta for c, delta in zip(current, predicted)]
        else:
            values = [c + prediction_scale * (value - c) for c, value in zip(current, predicted)]

        tmp = out_bin.with_suffix(out_bin.suffix + ".tmp")
        write_pressure_initializer(tmp, lx, ly, lz, values)
        tmp.replace(out_bin)

    def write_absolute(self, pre_plt: Path, out_bin: Path) -> None:
        self.write_absolute_from_data(read_tecplot(pre_plt), out_bin)


class HStatePredictor:
    def __init__(self, model_path: Path, device: str):
        torch, _, _ = require_torch()
        self.torch = torch
        self.device = device
        checkpoint = torch.load(model_path, map_location=device)
        if tuple(checkpoint.get("output_channels", ())) != tuple(f"h{i}" for i in range(15)):
            raise ValueError("checkpoint is not a 15-channel h_i model")
        self.channels = tuple(checkpoint["input_channels"])
        self.model = HInitializer(
            int(checkpoint["width"]), int(checkpoint["depth"]), len(self.channels)).to(device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        stats = checkpoint["normalization"]
        self.x_mean = torch.tensor(stats["x_mean"], device=device).view(1, len(self.channels), 1, 1, 1)
        self.x_std = torch.tensor(stats["x_std"], device=device).view(1, len(self.channels), 1, 1, 1)
        self.h_mean = torch.tensor(stats["h_mean"], device=device).view(1, 15, 1, 1, 1)
        self.h_std = torch.tensor(stats["h_std"], device=device).view(1, 15, 1, 1, 1)

    def write_state_from_data(self, data: dict[str, object], out_bin: Path) -> None:
        lx, ly, lz = int(data["lx"]), int(data["ly"]), int(data["lz"])
        fields = data["fields"]  # type: ignore[assignment]
        torch = self.torch
        inputs = torch.stack([
            torch.tensor(fields[name], dtype=torch.float32).view(lz, ly, lx)  # type: ignore[index]
            for name in self.channels
        ]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred_norm = self.model((inputs - self.x_mean) / self.x_std)
            hh = pred_norm * self.h_std + self.h_mean
            pressure = pressure_from_h(hh)
        pressure_values = pressure.squeeze(0).squeeze(0).cpu().reshape(-1).tolist()
        hh_values = hh.squeeze(0).permute(1, 2, 3, 0).cpu().reshape(-1).tolist()
        tmp = out_bin.with_suffix(out_bin.suffix + ".tmp")
        write_poisson_state(tmp, lx, ly, lz, pressure_values, hh_values)
        tmp.replace(out_bin)


def write_oracle_pressure(post_snapshot: Path, pair_format: str, out_bin: Path) -> None:
    data = read_pair_snapshot(post_snapshot, pair_format)
    fields = data["fields"]  # type: ignore[assignment]
    tmp = out_bin.with_suffix(out_bin.suffix + ".tmp")
    write_pressure_initializer(
        tmp,
        int(data["lx"]),
        int(data["ly"]),
        int(data["lz"]),
        fields["press"],  # type: ignore[index]
    )
    tmp.replace(out_bin)


def link_oracle_state(state_snapshot: Path, out_bin: Path) -> None:
    if not state_snapshot.exists():
        raise FileNotFoundError(state_snapshot)
    tmp = out_bin.with_suffix(out_bin.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    os.link(state_snapshot, tmp)
    tmp.replace(out_bin)


def run_closed_loop(
    root: Path,
    args: argparse.Namespace,
    run_dir: Path,
    output_steps: list[int],
) -> tuple[dict[str, object], dict[int, dict[str, object]]]:
    pair_dir = run_dir / "pairs"
    init_dir = run_dir / "pressure_init_wait"
    diag_csv = run_dir / "poisson_diagnostics.csv"
    stdout = run_dir / "stdout.log"
    pair_dir.mkdir(parents=True, exist_ok=True)
    init_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    candidates = candidate_steps(args)

    cmd = [
        str(args.exe.resolve()),
        "--mode", "gpu",
        "--params", str(args.params.resolve()),
        "--steps", str(args.steps),
        "--poisson", args.poisson,
        "--pressure-boundary", args.pressure_boundary,
        "--poisson-check-interval", str(args.poisson_check_interval),
        "--poisson-tolerance", str(args.poisson_tolerance),
        "--poisson-diagnostics", str(diag_csv.resolve()),
        "--write-output",
        "--output-steps", ",".join(str(step) for step in output_steps),
        "--write-poisson-pairs",
        "--poisson-pair-dir", str(pair_dir.resolve()),
        "--poisson-pair-phase", args.poisson_pair_phase,
        "--poisson-pair-format", args.poisson_pair_format,
        "--pressure-init-wait-dir", str(init_dir.resolve()),
        "--pressure-init-wait-timeout-ms", str(args.pressure_init_wait_timeout_ms),
        "--pressure-init-mode", "absolute",
        "--no-roofline",
    ]
    if candidates:
        if len(candidates) <= 512:
            cmd.extend(["--poisson-pair-steps", ",".join(str(step) for step in candidates)])
        else:
            cmd.extend([
                "--poisson-pair-start-step", str(args.pinn_start_step),
                "--poisson-pair-interval", str(args.pinn_interval),
            ])
            if args.pinn_max_step > 0:
                cmd.extend(["--poisson-pair-max-step", str(args.pinn_max_step)])
    else:
        cmd.extend(["--poisson-pair-steps", ""])
    if args.pressure_init_max_iterations > 0:
        cmd.extend(["--pressure-init-max-iterations", str(args.pressure_init_max_iterations)])
    if args.pressure_init_check_interval > 0:
        cmd.extend(["--pressure-init-check-interval", str(args.pressure_init_check_interval)])
    if args.source_aware_hh_init:
        cmd.extend(["--source-aware-hh-init", "--source-aware-hh-scale", str(args.source_aware_hh_scale)])
    if args.state_export_dir is not None:
        cmd.extend([
            "--poisson-state-export-dir", str(args.state_export_dir.resolve()),
            "--poisson-state-export-steps", args.state_export_steps,
            "--poisson-state-export-phase", args.state_export_phase,
        ])

    ranges = metric_ranges(args.quality_manifest, args.quality_margin)
    predictor = PressurePredictor(args.model, args.device) if (
        candidates and args.h_model is None and args.oracle_post_dir is None and args.oracle_state_dir is None
    ) else None
    h_predictor = HStatePredictor(args.h_model, args.device) if candidates and args.h_model is not None else None
    per_step: dict[int, dict[str, object]] = {}
    cooldown_until_step = 0
    with stdout.open("w") as log:
        proc: subprocess.Popen[object] = subprocess.Popen(
            cmd,
            cwd=run_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            for step in candidates:
                candidate_start = time.perf_counter()
                pre_snapshot = pair_dir / "pre_poisson" / step_name(step, pair_snapshot_suffix(args.poisson_pair_format))
                wait_start = time.perf_counter()
                wait_for_stable_file(pre_snapshot, proc, args.pre_poisson_timeout_s)
                wait_ms = (time.perf_counter() - wait_start) * 1000.0
                ready_start = time.perf_counter()
                read_start = time.perf_counter()
                pre_data = read_pair_snapshot(pre_snapshot, args.poisson_pair_format)
                read_ms = (time.perf_counter() - read_start) * 1000.0
                quality_accept = True
                quality_reasons = ""
                adaptive_ms = 0.0
                finite_ms = 0.0
                quality_ms = 0.0
                infer_write_ms = 0.0
                skip_write_ms = 0.0
                if args.adaptive_residual_gate and step > 1:
                    adaptive_start = time.perf_counter()
                    previous = parse_step_diagnostics(diag_csv).get(step - 1, {})
                    previous_used = int(per_step.get(step - 1, {}).get("used_pressure_initializer", 0) or 0)
                    previous_segments = int(previous.get("poisson_segments_step", 0) or 0)
                    previous_converged = int(previous.get("poisson_converged_step", 0) or 0)
                    previous_fallback = previous_used and previous_segments > 1
                    if previous_fallback or not previous_converged:
                        cooldown_until_step = max(cooldown_until_step, step + args.fallback_cooldown_steps)
                    adaptive_ms = (time.perf_counter() - adaptive_start) * 1000.0

                finite_start = time.perf_counter()
                finite_ok, finite_reasons = fields_are_finite(pre_data)
                finite_ms = (time.perf_counter() - finite_start) * 1000.0
                if not finite_ok:
                    quality_accept = False
                    quality_reasons = finite_reasons
                if quality_accept and args.adaptive_residual_gate and step <= cooldown_until_step:
                    quality_accept = False
                    quality_reasons = f"adaptive cooldown until step={cooldown_until_step}"
                if quality_accept and not args.disable_input_quality_gate:
                    quality_start = time.perf_counter()
                    quality_accept, reasons = score(field_metrics_from_data(pre_data), ranges)
                    quality_reasons = "; ".join(reasons)
                    quality_ms = (time.perf_counter() - quality_start) * 1000.0
                out_bin = init_dir / step_name(step, "bin")
                skip_path = init_dir / step_name(step, "skip")
                used = False
                if quality_accept:
                    infer_write_start = time.perf_counter()
                    if h_predictor is not None:
                        h_predictor.write_state_from_data(pre_data, out_bin)
                    elif args.oracle_state_dir is not None:
                        link_oracle_state(
                            args.oracle_state_dir / step_name(step, "bin"), out_bin)
                    elif args.oracle_post_dir is not None:
                        oracle_snapshot = args.oracle_post_dir / step_name(
                            step, pair_snapshot_suffix(args.poisson_pair_format))
                        write_oracle_pressure(oracle_snapshot, args.poisson_pair_format, out_bin)
                    else:
                        assert predictor is not None
                        predictor.write_absolute_from_data(pre_data, out_bin, args.prediction_scale)
                    infer_write_ms = (time.perf_counter() - infer_write_start) * 1000.0
                    used = True
                else:
                    skip_write_start = time.perf_counter()
                    write_skip(skip_path)
                    skip_write_ms = (time.perf_counter() - skip_write_start) * 1000.0

                per_step[step] = {
                    "input_plt": str(pre_snapshot),
                    "input_quality_accept": int(quality_accept),
                    "input_quality_reasons": quality_reasons,
                    "used_pressure_initializer": int(used),
                    "pressure_init_file": str(out_bin) if used else "",
                    "candidate_total_ms": (time.perf_counter() - candidate_start) * 1000.0,
                    "pre_snapshot_wait_ms": wait_ms,
                    "python_ready_ms": (time.perf_counter() - ready_start) * 1000.0,
                    "read_pair_ms": read_ms,
                    "adaptive_gate_ms": adaptive_ms,
                    "finite_check_ms": finite_ms,
                    "quality_gate_ms": quality_ms,
                    "infer_write_ms": infer_write_ms,
                    "skip_write_ms": skip_write_ms,
                    "snapshot_bytes": pre_snapshot.stat().st_size if pre_snapshot.exists() else 0,
                    "pair_format": args.poisson_pair_format,
                }

            return_code = proc.wait(timeout=args.solver_finish_timeout_s)
        except Exception:
            proc.kill()
            proc.wait()
            raise
    if return_code != 0:
        print(stdout.read_text())
        raise SystemExit(return_code)
    if args.pinn_max_step > 0 or args.pinn_start_step > 1 or args.pinn_interval > 1:
        for step in output_steps:
            if step not in candidates and step not in per_step:
                per_step[step] = {
                    "input_plt": "",
                    "input_quality_accept": 0,
                    "input_quality_reasons": "step not in PINN candidate schedule",
                    "used_pressure_initializer": 0,
                    "pressure_init_file": "",
                }

    text = stdout.read_text()
    perf = parse_perf(text)
    perf.update({
        "steps": args.steps,
        "diagnostics": str(diag_csv),
        "stdout": str(stdout),
        "command": " ".join(cmd),
    })
    return perf, per_step


def step_iters(diag: dict[int, dict[str, object]], step: int) -> int:
    return int(diag.get(step, {}).get("poisson_iters_step", 0) or 0)


def window_iter_delta(
    baseline_diag: dict[int, dict[str, object]],
    safe_diag: dict[int, dict[str, object]],
    start_step: int,
    end_step: int,
) -> int:
    if end_step < start_step:
        return 0
    baseline = sum(step_iters(baseline_diag, step) for step in range(start_step, end_step + 1))
    safe = sum(step_iters(safe_diag, step) for step in range(start_step, end_step + 1))
    return baseline - safe


def write_candidate_scan(
    path: Path,
    label: str,
    args: argparse.Namespace,
    candidates: list[int],
    baseline_dir: Path,
    closed_dir: Path,
    closed_step_meta: dict[int, dict[str, object]],
) -> None:
    baseline_diag = parse_step_diagnostics(baseline_dir / "poisson_diagnostics.csv")
    safe_diag = parse_step_diagnostics(closed_dir / "poisson_diagnostics.csv")
    rows: list[dict[str, object]] = []
    windows = (1, 5, 20, 50, 100)
    for idx, step in enumerate(candidates):
        meta = dict(closed_step_meta.get(step, {}))
        safe_step = safe_diag.get(step, {})
        used = int(meta.get("used_pressure_initializer", 0) or 0)
        safe_segments = int(safe_step.get("poisson_segments_step", 0) or 0)
        safe_converged = int(safe_step.get("poisson_converged_step", 0) or 0)
        baseline_iter = step_iters(baseline_diag, step)
        safe_iter = step_iters(safe_diag, step)
        next_candidate = candidates[idx + 1] if idx + 1 < len(candidates) else args.steps + 1
        until_next_end = min(args.steps, next_candidate - 1)
        row: dict[str, object] = {
            "label": label,
            "candidate_step": step,
            "next_candidate_step": next_candidate if next_candidate <= args.steps else "",
            "input_quality_accept": meta.get("input_quality_accept", ""),
            "input_quality_reasons": meta.get("input_quality_reasons", ""),
            "used_pressure_initializer": used,
            "pressure_init_accept_step": int(used == 1 and safe_segments == 1 and safe_converged == 1),
            "pressure_init_fallback_step": int(used == 1 and safe_segments > 1),
            "baseline_iter": baseline_iter,
            "safe_iter": safe_iter,
            "iter_delta_step": baseline_iter - safe_iter,
            "safe_segments": safe_segments,
            "safe_converged": safe_converged,
            "future_until_next_steps": max(0, until_next_end - step),
            "future_until_next_iter_delta": window_iter_delta(baseline_diag, safe_diag, step + 1, until_next_end),
            "input_plt": meta.get("input_plt", ""),
            "pressure_init_file": meta.get("pressure_init_file", ""),
            "pair_format": meta.get("pair_format", ""),
            "snapshot_bytes": meta.get("snapshot_bytes", ""),
            "candidate_total_ms": meta.get("candidate_total_ms", ""),
            "pre_snapshot_wait_ms": meta.get("pre_snapshot_wait_ms", ""),
            "python_ready_ms": meta.get("python_ready_ms", ""),
            "read_pair_ms": meta.get("read_pair_ms", ""),
            "adaptive_gate_ms": meta.get("adaptive_gate_ms", ""),
            "finite_check_ms": meta.get("finite_check_ms", ""),
            "quality_gate_ms": meta.get("quality_gate_ms", ""),
            "infer_write_ms": meta.get("infer_write_ms", ""),
            "skip_write_ms": meta.get("skip_write_ms", ""),
        }
        for window in windows:
            row[f"future_{window}_iter_delta"] = window_iter_delta(
                baseline_diag, safe_diag, step + 1, min(args.steps, step + window))

        candidate_plt = closed_dir / "out" / step_name(step, "plt")
        reference_plt = baseline_dir / "out" / step_name(step, "plt")
        row["field_error_available"] = int(candidate_plt.exists() and reference_plt.exists())
        if candidate_plt.exists() and reference_plt.exists():
            row.update(compare_fields(candidate_plt, reference_plt))
        rows.append(row)

    keys = [
        "label", "candidate_step", "next_candidate_step",
        "input_quality_accept", "input_quality_reasons", "used_pressure_initializer",
        "pressure_init_accept_step", "pressure_init_fallback_step",
        "baseline_iter", "safe_iter", "iter_delta_step",
        "safe_segments", "safe_converged",
        "future_until_next_steps", "future_until_next_iter_delta",
        "future_1_iter_delta", "future_5_iter_delta", "future_20_iter_delta",
        "future_50_iter_delta", "future_100_iter_delta",
        "field_error_available",
        "rho_rel_l2", "rho_max_abs", "fei_rel_l2", "fei_max_abs",
        "u_rel_l2", "u_max_abs", "v_rel_l2", "v_max_abs", "w_rel_l2", "w_max_abs",
        "pressure_rel_l2", "pressure_max_abs", "phase_mass_rel_error",
        "field_health_pass", "field_health_reasons",
        "component_count", "reference_component_count",
        "interface_fraction_abs_error", "mid_fraction_abs_error", "liquid_fraction_abs_error",
        "fei_grad_mean_abs_error", "fei_grad_max_abs_error",
        "candidate_total_ms", "pre_snapshot_wait_ms", "python_ready_ms",
        "read_pair_ms", "adaptive_gate_ms", "finite_check_ms",
        "quality_gate_ms", "infer_write_ms", "skip_write_ms",
        "pair_format", "snapshot_bytes", "input_plt", "pressure_init_file",
    ]
    write_dict_rows(path, rows, keys)


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=root / "GPU" / "lbm_gpu")
    parser.add_argument("--params", type=Path, default=root / "GPU" / "params_small.in")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output-steps", default=None)
    parser.add_argument("--poisson", choices=("split", "fused", "onepass", "scalar"), default="onepass")
    parser.add_argument("--pressure-boundary", choices=("split", "fused"), default="fused")
    parser.add_argument("--poisson-check-interval", type=int, default=100)
    parser.add_argument("--poisson-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--pressure-init-max-iterations", type=int, default=0)
    parser.add_argument("--pressure-init-check-interval", type=int, default=0,
                        help="0 uses --poisson-check-interval for the PINN attempt")
    parser.add_argument("--pressure-init-wait-timeout-ms", type=int, default=600000)
    parser.add_argument("--poisson-pair-phase", choices=("pre", "post", "both"), default="pre",
                        help="Poisson pair snapshots to write during closed-loop runs")
    parser.add_argument("--poisson-pair-format", choices=("tecplot", "features"), default="features",
                        help="Poisson pair snapshot encoding for closed-loop inputs")
    parser.add_argument("--source-aware-hh-init", action="store_true")
    parser.add_argument("--source-aware-hh-scale", type=float, default=1.0)
    parser.add_argument("--state-export-dir", type=Path, default=None)
    parser.add_argument("--state-export-steps", default="")
    parser.add_argument("--state-export-phase", choices=("pre", "post", "both"), default="pre")
    parser.add_argument("--model", type=Path,
                        default=root / "PINN_Poisson" / "models" / "pressure_initializer_augmented_abs_residual32.pt")
    parser.add_argument("--h-model", type=Path, default=None,
                        help="15-channel h_i checkpoint; writes a complete p=sum(h_i),hh state")
    parser.add_argument("--oracle-post-dir", type=Path, default=None,
                        help="use post-Poisson pressure snapshots as oracle initializers")
    parser.add_argument("--oracle-state-dir", type=Path, default=None,
                        help="use exported post-Poisson p+hh state snapshots as oracle initializers")
    parser.add_argument("--quality-manifest", type=Path,
                        default=root / "PINN_Poisson" / "data" / "augmented_manifest.csv")
    parser.add_argument("--quality-margin", type=float, default=0.25)
    parser.add_argument("--disable-input-quality-gate", action="store_true")
    parser.add_argument("--pinn-max-step", type=int, default=0,
                        help="0 allows every step; positive values skip PINN after this step")
    parser.add_argument("--pinn-steps", default="",
                        help="explicit comma-separated PINN candidate steps; overrides start/interval")
    parser.add_argument("--pinn-start-step", type=int, default=1)
    parser.add_argument("--pinn-interval", type=int, default=1)
    parser.add_argument("--adaptive-residual-gate", action="store_true")
    parser.add_argument("--fallback-cooldown-steps", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prediction-scale", type=float, default=1.0,
                        help="blend model pressure correction into the current pressure")
    parser.add_argument("--pre-poisson-timeout-s", type=float, default=120.0)
    parser.add_argument("--solver-finish-timeout-s", type=float, default=120.0)
    parser.add_argument("--label", default=None)
    parser.add_argument("--csv", type=Path,
                        default=root / "PINN_Poisson" / "results" / "benchmarks" / "multistep_closed_loop.csv")
    parser.add_argument("--candidate-csv", type=Path, default=None,
                        help="per-candidate scan CSV; defaults to --csv with _candidates suffix")
    parser.add_argument("--reuse-baseline-dir", type=Path, default=None,
                        help="reuse an existing baseline directory with stdout.log, poisson_diagnostics.csv, and outputs")
    args = parser.parse_args()

    if args.oracle_post_dir is not None:
        args.oracle_post_dir = args.oracle_post_dir.resolve()
    if args.oracle_state_dir is not None:
        args.oracle_state_dir = args.oracle_state_dir.resolve()
    if args.oracle_post_dir is not None and args.oracle_state_dir is not None:
        raise SystemExit("--oracle-post-dir and --oracle-state-dir are mutually exclusive")
    if args.h_model is not None and (args.oracle_post_dir is not None or args.oracle_state_dir is not None):
        raise SystemExit("--h-model cannot be combined with an oracle initializer")
    if args.state_export_dir is not None and not args.state_export_steps.strip():
        raise SystemExit("--state-export-steps is required with --state-export-dir")

    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if not (0.0 <= args.prediction_scale <= 1.0):
        raise SystemExit("--prediction-scale must be in [0, 1]")
    output_steps = parse_steps(args.output_steps, args.steps)
    label = args.label or f"multistep_closed_loop_step{args.steps}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    work_dir = root / "PINN_Poisson" / "data" / "multistep_closed_loop" / label
    candidates = candidate_steps(args)
    candidate_csv = args.candidate_csv
    if candidate_csv is None:
        candidate_csv = args.csv.with_name(f"{args.csv.stem}_candidates{args.csv.suffix}")

    if args.reuse_baseline_dir is not None:
        baseline_dir = args.reuse_baseline_dir.resolve()
        baseline_stdout = baseline_dir / "stdout.log"
        baseline_diag = baseline_dir / "poisson_diagnostics.csv"
        if not baseline_stdout.exists():
            raise SystemExit(f"missing reused baseline stdout: {baseline_stdout}")
        if not baseline_diag.exists():
            raise SystemExit(f"missing reused baseline diagnostics: {baseline_diag}")
        missing_outputs = [
            step for step in output_steps
            if not (baseline_dir / "out" / step_name(step, "plt")).exists()
        ]
        if missing_outputs:
            raise SystemExit(f"reused baseline missing outputs for steps: {missing_outputs}")
        baseline_perf = parse_perf(baseline_stdout.read_text())
        baseline_perf.update({
            "steps": args.steps,
            "diagnostics": str(baseline_diag),
            "stdout": str(baseline_stdout),
            "command": "reused baseline",
        })
    else:
        baseline_dir = work_dir / "baseline"
        baseline_perf = run_solver(
            args,
            root,
            baseline_dir,
            baseline_dir / "poisson_diagnostics.csv",
            output_steps,
        )
    closed_dir = work_dir / "safe_closed_loop"
    closed_perf, closed_step_meta = run_closed_loop(root, args, closed_dir, output_steps)

    rows: list[dict[str, object]] = []
    rows.extend(method_rows(label, "baseline", output_steps, baseline_dir, baseline_dir, baseline_perf, baseline_perf))
    rows.extend(method_rows(label, "safe_closed_loop", output_steps, closed_dir, baseline_dir, closed_perf, baseline_perf, closed_step_meta))

    keys = [
        "label", "method", "step", "steps",
        "input_quality_accept", "input_quality_reasons", "used_pressure_initializer",
        "pressure_init_attempts", "pressure_init_accepts", "pressure_init_fallbacks",
        "pressure_init_accept_step", "pressure_init_fallback_step",
        "poisson_iters_avg", "poisson_iters_total_est", "total_ms_per_step", "total_ms_est",
        "wall_ms_per_step", "wall_total_ms",
        "poisson_iters_step", "poisson_segments_step", "poisson_converged_segments_step", "poisson_converged_step",
        "iter_reduction_vs_baseline", "speedup_vs_baseline", "wall_speedup_vs_baseline",
        "rho_rel_l2", "rho_max_abs", "fei_rel_l2", "fei_max_abs",
        "u_rel_l2", "u_max_abs", "v_rel_l2", "v_max_abs", "w_rel_l2", "w_max_abs",
        "pressure_rel_l2", "pressure_max_abs", "phase_mass_rel_error",
        "field_health_pass", "field_health_reasons",
        "component_count", "reference_component_count",
        "interface_fraction_abs_error", "mid_fraction_abs_error", "liquid_fraction_abs_error",
        "fei_grad_mean_abs_error", "fei_grad_max_abs_error",
        "candidate_total_ms", "pre_snapshot_wait_ms", "python_ready_ms",
        "read_pair_ms", "adaptive_gate_ms", "finite_check_ms",
        "quality_gate_ms", "infer_write_ms", "skip_write_ms",
        "pair_format", "snapshot_bytes",
        "input_plt", "pressure_init_file", "plt", "reference_plt", "diagnostics", "stdout",
    ]
    write_dict_rows(args.csv, rows, keys)
    write_candidate_scan(candidate_csv, label, args, candidates, baseline_dir, closed_dir, closed_step_meta)

    used_steps = sum(int(meta["used_pressure_initializer"]) for meta in closed_step_meta.values())
    print(f"{label}: baseline_iters={baseline_perf.get('poisson_iters')} "
          f"closed_loop_iters={closed_perf.get('poisson_iters')} "
          f"used_steps={used_steps}/{args.steps} fallbacks={closed_perf.get('pressure_init_fallbacks', '')}")
    print(f"wrote {args.csv}")
    print(f"wrote {candidate_csv}")
    print(f"work_dir {work_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
