"""Build the sealed official DFC25 validation SAR-only manifest on the cloud."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sar-root", required=True, type=Path)
    parser.add_argument("--label-root", required=True, type=Path)
    parser.add_argument("--cloud-data-root", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    separator = chr(47)
    extension = chr(46) + "tif"
    root = str(args.cloud_data_root).rstrip(separator)
    samples = []
    for sar in sorted(args.sar_root.glob("*" + extension)):
        label = args.label_root / sar.name
        if not label.is_file():
            raise FileNotFoundError(f"missing official validation label: {sar.stem}")
        sar_rel = separator.join(("extracted", "openearthmap_sar", "val", "sar_images", sar.name))
        label_rel = separator.join(("extracted", "openearthmap_sar_val_labels", "val", "labels", label.name))
        samples.append({"sample_id": sar.stem, "sar_path": root + separator + sar_rel, "label_path": root + separator + label_rel})
    payload = {
        "artifact_type": "cloud_external_sar_manifest", "schema_version": 1,
        "host": "cloud", "transfer_policy": "never_to_local",
        "cloud_data_root": root, "evaluation_role": "official_dfc25_val_sar_only_external",
        "target_test_excluded_from_training": True,
        "sample_count": len(samples),
        "label_policy": {"encoding": "one_based_1_to_8_with_zero_ignore", "ignore_index": 255},
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "sample_count": len(samples), "output": str(args.output)}))


if __name__ == "__main__":
    main()
