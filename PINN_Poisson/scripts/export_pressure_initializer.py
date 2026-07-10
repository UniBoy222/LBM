#!/usr/bin/env python3
"""Export pressure predictions/corrections to the GPU pressure-init format."""

from __future__ import annotations

import argparse
from pathlib import Path

from tecplot_io import read_tecplot, write_pressure_initializer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plt", type=Path, help="Tecplot binary field containing pressure")
    parser.add_argument("--out", type=Path, required=True, help="output .bin pressure initializer")
    parser.add_argument("--delta-against", type=Path, default=None,
                        help="optional Tecplot field to subtract, producing delta_p")
    args = parser.parse_args()

    data = read_tecplot(args.plt)
    fields = data["fields"]  # type: ignore[assignment]
    values = list(fields["press"])  # type: ignore[index]
    mode = "absolute"

    if args.delta_against is not None:
        base = read_tecplot(args.delta_against)
        if (data["lx"], data["ly"], data["lz"]) != (base["lx"], base["ly"], base["lz"]):
            raise SystemExit("delta source grid dimensions do not match")
        base_fields = base["fields"]  # type: ignore[assignment]
        values = [v - b for v, b in zip(values, base_fields["press"])]  # type: ignore[index]
        mode = "delta"

    write_pressure_initializer(
        args.out,
        int(data["lx"]),
        int(data["ly"]),
        int(data["lz"]),
        values,
    )
    print(f"wrote {args.out} mode={mode} cells={len(values)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

