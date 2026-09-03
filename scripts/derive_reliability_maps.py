"""Derive auditable optical/SAR reliability proxy rasters on the cloud host."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling

from rhs_rcmtr.utils.reliability import optical_radiometric_usability, sar_local_stability


def derive(optical: Path, sar: Path, output: Path, grid_size: int = 32) -> None:
    if grid_size < 4:
        raise ValueError("grid_size must be at least 4")
    with rasterio.open(optical) as optical_src, rasterio.open(sar) as sar_src:
        if (optical_src.width, optical_src.height) != (sar_src.width, sar_src.height):
            raise ValueError("optical and SAR dimensions differ")
        out_shape = (3, grid_size, grid_size)
        optical_arr = optical_src.read(
            out_shape=out_shape, resampling=Resampling.average, out_dtype="float32"
        )
        sar_arr = sar_src.read(
            1, out_shape=(grid_size, grid_size), resampling=Resampling.average,
            out_dtype="float32"
        )
        optical_reliability = optical_radiometric_usability(optical_arr)
        sar_reliability = sar_local_stability(sar_arr)
        stack = np.stack([optical_reliability, sar_reliability]).astype("float32")
        profile = optical_src.profile.copy()
        scale_x = optical_src.width / grid_size
        scale_y = optical_src.height / grid_size
        profile.update(
            width=grid_size, height=grid_size, count=2, dtype="float32",
            transform=optical_src.transform * Affine.scale(scale_x, scale_y),
            compress="deflate", predictor=2,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **profile) as dst:
            dst.write(stack)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optical", required=True, type=Path)
    parser.add_argument("--sar", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grid-size", type=int, default=32)
    args = parser.parse_args()
    derive(args.optical, args.sar, args.output, grid_size=args.grid_size)
    print(args.output)


if __name__ == "__main__":
    main()
