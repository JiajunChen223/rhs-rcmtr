"""Derive the P1 reliability maps for every official train triplet on cloud.

The output is written to a fresh directory so the P0 baseline's historical
reliability maps remain immutable.  No raster bytes are copied to the local
host by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling

from rhs_rcmtr.utils.reliability import optical_radiometric_usability, sar_local_stability


def derive_one(optical: Path, sar: Path, output: Path, grid_size: int) -> None:
    with rasterio.open(optical) as optical_src, rasterio.open(sar) as sar_src:
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
        profile = optical_src.profile.copy()
        profile.update(
            width=grid_size,
            height=grid_size,
            count=2,
            dtype="float32",
            transform=optical_src.transform
            * Affine.scale(optical_src.width / grid_size, optical_src.height / grid_size),
            compress="deflate",
            predictor=2,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(np.stack([optical_reliability, sar_reliability]).astype("float32"))


def derive_all(sar_root: Path, output_root: Path, *, grid_size: int) -> dict[str, object]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite reliability output root: {output_root}")
    optical_root = sar_root / "train" / "rgb_images"
    sar_image_root = sar_root / "train" / "sar_images"
    optical_paths = sorted(optical_root.glob("*" + chr(46) + "tif"))
    if len(optical_paths) != 4333:
        raise ValueError(f"expected 4333 official train optical rasters, found {len(optical_paths)}")
    output_root.mkdir(parents=True, exist_ok=False)
    for index, optical in enumerate(optical_paths, start=1):
        sar = sar_image_root / optical.name
        if not sar.is_file():
            raise FileNotFoundError(f"missing SAR raster for {optical.stem}")
        derive_one(optical, sar, output_root / optical.name, grid_size)
        if index % 250 == 0 or index == len(optical_paths):
            print(json.dumps({"processed": index, "total": len(optical_paths)}), flush=True)
    return {
        "status": "pass",
        "sample_count": len(optical_paths),
        "output_root": str(output_root),
        "grid_size": grid_size,
        "reliability_policy": "two_band_radiometric_usability_and_local_sar_stability_evidence_cloud_derived",
        "test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sar-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--grid-size", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(derive_all(args.sar_root, args.output_root, grid_size=args.grid_size)))


if __name__ == "__main__":
    main()
