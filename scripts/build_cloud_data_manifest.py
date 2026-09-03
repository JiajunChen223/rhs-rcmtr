"""Build the scheme-A cloud manifest from official train triplets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def split_for(sample_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()
    return "train" if int(digest[:8], 16) / 0xFFFFFFFF < 0.8 else "validation"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sar-root", required=True, type=Path)
    parser.add_argument("--reliability-root", required=True, type=Path)
    parser.add_argument("--cloud-data-root", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.seed != 0:
        raise ValueError("the approved scheme-A internal split is fixed at seed=0")
    separator = chr(47)
    extension = chr(46) + "tif"
    rgb_root = args.sar_root / "train" / "rgb_images"
    sar_root = args.sar_root / "train" / "sar_images"
    label_root = args.sar_root / "train" / "labels"
    samples = []
    for optical in sorted(rgb_root.glob("*" + extension)):
        sample_id = optical.stem
        sar = sar_root / optical.name
        label = label_root / optical.name
        reliability = args.reliability_root / (sample_id + extension)
        if not sar.is_file() or not label.is_file() or not reliability.is_file():
            raise FileNotFoundError(f"incomplete official train triplet or reliability proxy: {sample_id}")
        split = split_for(sample_id, args.seed)
        root = str(args.cloud_data_root).rstrip(separator)
        optical_rel = separator.join(("extracted", "openearthmap_sar", "train", "rgb_images", optical.name))
        sar_rel = separator.join(("extracted", "openearthmap_sar", "train", "sar_images", sar.name))
        label_rel = separator.join(("extracted", "openearthmap_sar", "train", "labels", label.name))
        reliability_rel = separator.join(("derived", "reliability", reliability.name))
        samples.append({
            "sample_id": sample_id, "group_id": sample_id, "split": split,
            "optical_path": root + separator + optical_rel,
            "sar_path": root + separator + sar_rel,
            "label_path": root + separator + label_rel,
            "reliability_path": root + separator + reliability_rel,
        })
    payload = {
        "artifact_type": "cloud_data_manifest", "schema_version": 1,
        "host": "cloud", "transfer_policy": "never_to_local",
        "cloud_data_root": str(args.cloud_data_root).rstrip(separator),
        "target_test_excluded_from_training": True,
        "official_validation_external_role": "sar_only_sealed_external_evaluation",
        "internal_split_policy": "sha256_sample_id_seed0_80_20_of_official_train_triplets",
        "split_seed": 0,
        "group_id_policy": "sample_id_as_trainarea_stem",
        "sample_count": len(samples),
        "label_policy": {"encoding": "one_based_1_to_8_with_zero_ignore", "ignore_index": 255},
        "reliability_policy": "two_band_radiometric_usability_and_local_sar_stability_evidence_cloud_derived",
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "sample_count": len(samples), "output": str(args.output)}))


if __name__ == "__main__":
    main()
