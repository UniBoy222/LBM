#!/usr/bin/env python3
"""Dump Torch-CPU real-state stages for an independent C++ cross-audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from verify_fixed_point_state_torch import Q, read_state, run_backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    state = read_state(args.state_file)
    nz, ny, nx = state["arrays"]["p"].shape
    z, y, x = np.indices((nz, ny, nx), dtype=np.int64)
    direction = (((3 * x + 5 * y + 7 * z) % 17).astype(np.float64) - 8.0) / 8.0
    direction -= np.mean(direction, dtype=np.float64)
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    metrics, tensors = run_backend(torch.device("cpu"), state, direction)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = (
        f"step{state['header']['step']:04d}_"
        f"iter{state['header']['iteration']:08d}"
    )
    files: dict[str, dict[str, object]] = {}
    for name, tensor in tensors.items():
        values = tensor.detach().cpu().contiguous().numpy().astype("<f8", copy=False)
        path = args.output_dir / f"{prefix}_{name}.bin"
        values.tofile(path)
        files[name] = {
            "path": str(path.resolve()),
            "shape": list(values.shape),
            "values": int(values.size),
        }

    metadata = {
        "schema": "torch-cpu-real-state-stages-v1",
        "state_file": state["path"],
        "state_sha256": state["sha256"],
        "step": state["header"]["step"],
        "iteration": state["header"]["iteration"],
        "q": Q,
        "dtype": "float64-little-endian",
        "files": files,
        "torch_cpu_metrics": metrics,
    }
    metadata_path = args.output_dir / f"{prefix}_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(metadata_path)


if __name__ == "__main__":
    main()
