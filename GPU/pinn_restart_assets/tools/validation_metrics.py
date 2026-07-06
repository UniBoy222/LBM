#!/usr/bin/env python3
"""Post-process Inamuro Tecplot binary files for paper-grade validation metrics."""

from __future__ import annotations

import argparse
import csv
import math
import struct
from collections import deque
from pathlib import Path


def _read_int(buf: bytes, off: int) -> tuple[int, int]:
    return struct.unpack_from("<i", buf, off)[0], off + 4


def _read_float(buf: bytes, off: int) -> tuple[float, int]:
    return struct.unpack_from("<f", buf, off)[0], off + 4


def _read_dump_string(buf: bytes, off: int) -> tuple[str, int]:
    chars: list[str] = []
    while True:
        code, off = _read_int(buf, off)
        if code == 0:
            break
        chars.append(chr(code))
    return "".join(chars), off


def read_params(path: Path) -> dict[str, float]:
    values = path.read_text().split()
    if len(values) < 22:
        return {}
    return {
        "lx": int(values[0]),
        "ly": int(values[1]),
        "lz": int(values[2]),
        "rho_L": float(values[6]),
        "rho_G": float(values[7]),
        "fei_max": float(values[17]),
        "fei_min": float(values[18]),
        "fei_L": float(values[19]),
        "fei_G": float(values[20]),
        "DD": float(values[21]),
    }


def read_tecplot(path: Path) -> dict[str, object]:
    buf = path.read_bytes()
    off = 0
    magic = buf[off:off + 8].decode("ascii", errors="replace")
    off += 8
    if magic != "#!TDV101":
        raise ValueError(f"unsupported Tecplot magic: {magic!r}")

    _, off = _read_int(buf, off)
    _, off = _read_dump_string(buf, off)
    nvars, off = _read_int(buf, off)
    variables = []
    for _ in range(nvars):
        name, off = _read_dump_string(buf, off)
        variables.append(name)

    _, off = _read_float(buf, off)
    _, off = _read_dump_string(buf, off)
    for _ in range(5):
        _, off = _read_int(buf, off)
    lx, off = _read_int(buf, off)
    ly, off = _read_int(buf, off)
    lz, off = _read_int(buf, off)
    _, off = _read_int(buf, off)
    _, off = _read_float(buf, off)
    _, off = _read_float(buf, off)
    for _ in range(nvars):
        _, off = _read_int(buf, off)
    _, off = _read_int(buf, off)
    _, off = _read_int(buf, off)

    n = lx * ly * lz
    fields = {name: [0.0] * n for name in ("u", "v", "w", "rho", "fei", "press")}
    for idx in range(n):
        _, off = _read_int(buf, off)
        _, off = _read_int(buf, off)
        _, off = _read_int(buf, off)
        fields["u"][idx], off = _read_float(buf, off)
        fields["v"][idx], off = _read_float(buf, off)
        fields["w"][idx], off = _read_float(buf, off)
        fields["rho"][idx], off = _read_float(buf, off)
        fields["fei"][idx], off = _read_float(buf, off)
        fields["press"][idx], off = _read_float(buf, off)

    return {"lx": lx, "ly": ly, "lz": lz, "fields": fields, "variables": variables}


def connected_components(mask: list[bool], lx: int, ly: int, lz: int) -> list[dict[str, float]]:
    seen = bytearray(len(mask))
    comps: list[dict[str, float]] = []

    def index(x: int, y: int, z: int) -> int:
        return (z * ly + y) * lx + x

    for seed, is_liquid in enumerate(mask):
        if not is_liquid or seen[seed]:
            continue
        q: deque[int] = deque([seed])
        seen[seed] = 1
        count = 0
        sx = sy = sz = 0.0
        while q:
            idx = q.popleft()
            x = idx % lx
            y = (idx // lx) % ly
            z = idx // (lx * ly)
            count += 1
            sx += x
            sy += y
            sz += z

            neighbors = []
            if x > 0:
                neighbors.append(index(x - 1, y, z))
            if x + 1 < lx:
                neighbors.append(index(x + 1, y, z))
            neighbors.append(index(x, (y - 1) % ly, z))
            neighbors.append(index(x, (y + 1) % ly, z))
            neighbors.append(index(x, y, (z - 1) % lz))
            neighbors.append(index(x, y, (z + 1) % lz))

            for nb in neighbors:
                if mask[nb] and not seen[nb]:
                    seen[nb] = 1
                    q.append(nb)

        comps.append({
            "voxels": float(count),
            "cx": sx / count,
            "cy": sy / count,
            "cz": sz / count,
        })

    comps.sort(key=lambda c: c["voxels"], reverse=True)
    return comps


def summarize(data: dict[str, object], params: dict[str, float], threshold: float | None) -> dict[str, float | str]:
    lx = int(data["lx"])
    ly = int(data["ly"])
    lz = int(data["lz"])
    fields = data["fields"]  # type: ignore[assignment]
    fei = fields["fei"]  # type: ignore[index]
    press = fields["press"]  # type: ignore[index]
    u = fields["u"]  # type: ignore[index]
    v = fields["v"]  # type: ignore[index]
    w = fields["w"]  # type: ignore[index]

    if threshold is None:
        threshold = 0.5 * (params.get("fei_L", 0.092) + params.get("fei_G", 0.015))

    mask = [val >= threshold for val in fei]
    comps = connected_components(mask, lx, ly, lz)
    liquid = [i for i, keep in enumerate(mask) if keep]
    gas = [i for i, keep in enumerate(mask) if not keep]

    def mean(values: list[float], idxs: list[int]) -> float:
        return sum(values[i] for i in idxs) / len(idxs) if idxs else float("nan")

    speeds = [math.sqrt(u[i] * u[i] + v[i] * v[i] + w[i] * w[i]) for i in range(len(fei))]
    max_speed = max(speeds) if speeds else 0.0
    max_gas_speed = max((speeds[i] for i in gas), default=0.0)
    p_liq = mean(press, liquid)
    p_gas = mean(press, gas)
    delta_p = p_liq - p_gas if not math.isnan(p_liq) and not math.isnan(p_gas) else float("nan")
    radius = 0.5 * params.get("DD", float("nan"))
    inv_radius = 1.0 / radius if radius and not math.isnan(radius) else float("nan")

    if len(comps) == 1:
        regime = "coalesced"
    elif len(comps) == 2:
        regime = "two_droplets"
    elif len(comps) > 2:
        regime = "satellite_or_fragmented"
    else:
        regime = "no_liquid"

    c0 = comps[0] if len(comps) > 0 else {"voxels": 0.0, "cx": float("nan"), "cy": float("nan"), "cz": float("nan")}
    c1 = comps[1] if len(comps) > 1 else {"voxels": 0.0, "cx": float("nan"), "cy": float("nan"), "cz": float("nan")}
    if len(comps) > 1:
        dx = c0["cx"] - c1["cx"]
        dy = c0["cy"] - c1["cy"]
        dz = c0["cz"] - c1["cz"]
        component_centroid_distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    else:
        component_centroid_distance = float("nan")

    return {
        "lx": lx,
        "ly": ly,
        "lz": lz,
        "threshold": threshold,
        "phase_mass": sum(fei),
        "liquid_voxels": len(liquid),
        "component_count": len(comps),
        "largest_component_voxels": c0["voxels"],
        "largest_component_cx": c0["cx"],
        "largest_component_cy": c0["cy"],
        "largest_component_cz": c0["cz"],
        "second_component_voxels": c1["voxels"],
        "second_component_cx": c1["cx"],
        "second_component_cy": c1["cy"],
        "second_component_cz": c1["cz"],
        "component_centroid_distance": component_centroid_distance,
        "regime": regime,
        "max_speed": max_speed,
        "max_gas_speed": max_gas_speed,
        "mean_pressure_liquid": p_liq,
        "mean_pressure_gas": p_gas,
        "laplace_delta_p": delta_p,
        "inverse_radius": inv_radius,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plt", type=Path)
    parser.add_argument("--params", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    params = read_params(args.params) if args.params else {}
    data = read_tecplot(args.plt)
    row = summarize(data, params, args.threshold)
    row["file"] = str(args.plt)

    keys = [
        "file", "lx", "ly", "lz", "threshold", "phase_mass", "liquid_voxels",
        "component_count", "largest_component_voxels", "second_component_voxels",
        "largest_component_cx", "largest_component_cy", "largest_component_cz",
        "second_component_cx", "second_component_cy", "second_component_cz",
        "component_centroid_distance",
        "regime", "max_speed", "max_gas_speed", "mean_pressure_liquid",
        "mean_pressure_gas", "laplace_delta_p", "inverse_radius",
    ]

    for key in keys:
        print(f"{key}={row[key]}")

    if args.csv:
        exists = args.csv.exists()
        with args.csv.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if not exists:
                writer.writeheader()
            writer.writerow({key: row[key] for key in keys})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
