#!/usr/bin/env python3
import argparse
import json
import math
import struct
from pathlib import Path


VALUE_NAMES = ("u", "v", "w", "rho", "fei", "p")


def read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise RuntimeError("truncated Tecplot file")
    return data


def read_i32(stream):
    return struct.unpack("<i", read_exact(stream, 4))[0]


def read_f32(stream):
    return struct.unpack("<f", read_exact(stream, 4))[0]


def read_string(stream):
    chars = []
    while True:
        value = read_i32(stream)
        if value == 0:
            return "".join(chars)
        chars.append(chr(value))


def read_file(path):
    stream = path.open("rb")
    if read_exact(stream, 8) != b"#!TDV101":
        raise RuntimeError(f"bad Tecplot magic: {path}")
    if read_i32(stream) != 1:
        raise RuntimeError(f"unexpected byte-order marker: {path}")
    title = read_string(stream)
    variable_count = read_i32(stream)
    variables = tuple(read_string(stream) for _ in range(variable_count))
    zone_marker = read_f32(stream)
    zone_name = read_string(stream)
    zone_config = tuple(read_i32(stream) for _ in range(5))
    dims = tuple(read_i32(stream) for _ in range(3))
    auxiliary = read_i32(stream)
    end_marker = read_f32(stream)
    data_marker = read_f32(stream)
    formats = tuple(read_i32(stream) for _ in range(variable_count))
    sharing = read_i32(stream)
    connectivity = read_i32(stream)
    expected = (299.0, 357.0, 299.0)
    actual = (zone_marker, end_marker, data_marker)
    if actual != expected or auxiliary != 0 or sharing != 0 or connectivity != -1:
        raise RuntimeError(f"unsupported Tecplot layout: {path}")
    if variables != ("X", "Y", "Z", "u", "v", "w", "rho", "fei", "press"):
        raise RuntimeError(f"unexpected variables: {variables}")
    if formats != (3, 3, 3, 1, 1, 1, 1, 1, 1):
        raise RuntimeError(f"unexpected formats: {formats}")
    return stream, {
        "title": title,
        "zone": zone_name,
        "zone_config": zone_config,
        "dims": dims,
    }


def compare(left_path, right_path):
    left, left_meta = read_file(left_path)
    right, right_meta = read_file(right_path)
    if left_meta != right_meta:
        raise RuntimeError("Tecplot metadata mismatch")
    count = math.prod(left_meta["dims"])
    max_abs = [0.0] * 6
    sum_sq = [0.0] * 6
    ref_sq = [0.0] * 6
    mismatched_values = [0] * 6
    coordinate_mismatches = 0
    for _ in range(count):
        left_coords = struct.unpack("<3i", read_exact(left, 12))
        right_coords = struct.unpack("<3i", read_exact(right, 12))
        coordinate_mismatches += left_coords != right_coords
        left_values = struct.unpack("<6f", read_exact(left, 24))
        right_values = struct.unpack("<6f", read_exact(right, 24))
        for index, (a, b) in enumerate(zip(left_values, right_values)):
            if not math.isfinite(a) or not math.isfinite(b):
                raise RuntimeError("non-finite Tecplot value")
            delta = float(a) - float(b)
            max_abs[index] = max(max_abs[index], abs(delta))
            sum_sq[index] += delta * delta
            ref_sq[index] += float(a) * float(a)
            mismatched_values[index] += a != b
    if left.read(1) or right.read(1):
        raise RuntimeError("unexpected trailing Tecplot bytes")
    left.close()
    right.close()
    fields = {}
    for index, name in enumerate(VALUE_NAMES):
        fields[name] = {
            "max_abs": max_abs[index],
            "rel_l2": math.sqrt(sum_sq[index] / ref_sq[index]) if ref_sq[index] else 0.0,
            "mismatched_values": mismatched_values[index],
        }
    return {
        "left": str(left_path.resolve()),
        "right": str(right_path.resolve()),
        "dims": left_meta["dims"],
        "coordinate_mismatches": coordinate_mismatches,
        "fields": fields,
        "worst_max_abs": max(max_abs),
        "worst_rel_l2": max(item["rel_l2"] for item in fields.values()),
        "finite": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compare(args.left, args.right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
