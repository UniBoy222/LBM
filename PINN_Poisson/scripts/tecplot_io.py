#!/usr/bin/env python3
"""Small Tecplot/PINN pressure IO helpers for the PINN Poisson track."""

from __future__ import annotations

import array
import math
import struct
import sys
from pathlib import Path
from typing import Iterable


PRESSURE_INIT_MAGIC = b"PINNP1\0\0"
POISSON_STATE_MAGIC = b"PINNS1\0\0"
FEATURE_SNAPSHOT_MAGIC = b"PINNF1\0\0"
FEATURE_SNAPSHOT_MAGIC_V2 = b"PINNF2\0\0"
FIELD_NAMES = ("u", "v", "w", "rho", "fei", "press")
FIELD_NAMES_V2 = FIELD_NAMES + ("div_u_source",)


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


def read_tecplot(path: Path) -> dict[str, object]:
    buf = path.read_bytes()
    off = 0
    magic = buf[off:off + 8].decode("ascii", errors="replace")
    off += 8
    if magic != "#!TDV101":
        raise ValueError(f"unsupported Tecplot magic in {path}: {magic!r}")

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
    fields = {name: [0.0] * n for name in FIELD_NAMES}
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


def read_feature_snapshot(path: Path) -> dict[str, object]:
    with path.open("rb") as f:
        magic = f.read(8)
        if magic not in (FEATURE_SNAPSHOT_MAGIC, FEATURE_SNAPSHOT_MAGIC_V2):
            raise ValueError(f"invalid feature snapshot magic in {path}")
        header = f.read(16)
        if len(header) != 16:
            raise ValueError(f"truncated feature snapshot header in {path}")
        lx, ly, lz, nfields = struct.unpack("<iiii", header)
        names = FIELD_NAMES_V2 if magic == FEATURE_SNAPSHOT_MAGIC_V2 else FIELD_NAMES
        if nfields != len(names):
            raise ValueError(f"unsupported feature count in {path}: {nfields}")
        n = lx * ly * lz
        fields: dict[str, array.array] = {}
        for name in names:
            vals = array.array("f")
            vals.fromfile(f, n)
            if len(vals) != n:
                raise ValueError(f"truncated feature field {name!r} in {path}")
            if sys.byteorder != "little":
                vals.byteswap()
            fields[name] = vals
        trailing = f.read(1)
        if trailing:
            raise ValueError(f"unexpected trailing bytes in {path}")
    return {"lx": lx, "ly": ly, "lz": lz, "fields": fields, "variables": names}


def write_pressure_initializer(path: Path, lx: int, ly: int, lz: int, values: Iterable[float]) -> None:
    vals = array.array("d", values)
    expected = lx * ly * lz
    if len(vals) != expected:
        raise ValueError(f"expected {expected} pressure values, got {len(vals)}")
    if any(not math.isfinite(v) for v in vals):
        raise ValueError("pressure initializer contains non-finite values")
    if sys.byteorder != "little":
        vals.byteswap()

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(PRESSURE_INIT_MAGIC)
        f.write(struct.pack("<iii", lx, ly, lz))
        vals.tofile(f)


def read_poisson_state(path: Path) -> dict[str, object]:
    with path.open("rb") as f:
        magic = f.read(8)
        if magic != POISSON_STATE_MAGIC:
            raise ValueError(f"invalid Poisson state magic in {path}")
        header = f.read(12)
        if len(header) != 12:
            raise ValueError(f"truncated Poisson state header in {path}")
        lx, ly, lz = struct.unpack("<iii", header)
        n = lx * ly * lz
        pressure = array.array("d")
        pressure.fromfile(f, n)
        hh = array.array("d")
        hh.fromfile(f, n * 15)
        if len(pressure) != n or len(hh) != n * 15:
            raise ValueError(f"truncated Poisson state payload in {path}")
        if sys.byteorder != "little":
            pressure.byteswap()
            hh.byteswap()
        if f.read(1):
            raise ValueError(f"unexpected trailing bytes in {path}")
    return {"lx": lx, "ly": ly, "lz": lz, "pressure": pressure, "hh": hh}


def write_poisson_state(
    path: Path,
    lx: int,
    ly: int,
    lz: int,
    pressure_values: Iterable[float],
    hh_values: Iterable[float],
) -> None:
    pressure = array.array("d", pressure_values)
    hh = array.array("d", hh_values)
    n = lx * ly * lz
    if len(pressure) != n or len(hh) != n * 15:
        raise ValueError(f"expected {n} pressure and {n * 15} hh values")
    if any(not math.isfinite(value) for value in pressure) or any(not math.isfinite(value) for value in hh):
        raise ValueError("Poisson state contains non-finite values")
    if sys.byteorder != "little":
        pressure.byteswap()
        hh.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(POISSON_STATE_MAGIC)
        f.write(struct.pack("<iii", lx, ly, lz))
        pressure.tofile(f)
        hh.tofile(f)


def read_pressure_initializer(path: Path) -> dict[str, object]:
    with path.open("rb") as f:
        magic = f.read(8)
        if magic != PRESSURE_INIT_MAGIC:
            raise ValueError(f"invalid pressure initializer magic in {path}")
        lx, ly, lz = struct.unpack("<iii", f.read(12))
        vals = array.array("d")
        vals.fromfile(f, lx * ly * lz)
    if sys.byteorder != "little":
        vals.byteswap()
    return {"lx": lx, "ly": ly, "lz": lz, "values": vals}
