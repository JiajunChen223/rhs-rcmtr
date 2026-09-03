"""Self-contained release validation shipped with the clean code package."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
EXECUTABLE = {".py", ".ps1", ".sh", ".bat"}
WEIGHT_SUFFIXES = {"." + token for token in ("pt", "pth", "ckpt", "safetensors")}
DATA_SUFFIXES = {"." + token for token in ("tif", "tiff", "h5", "hdf5", "npy", "npz", "zip", "tar")}


def violations(path: Path) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    if path.suffix.casefold() in WEIGHT_SUFFIXES:
        return [{"path": path.relative_to(CODE_ROOT).as_posix(), "type": "weight_binary"}]
    if path.suffix.casefold() != ".py":
        return findings
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            rendered = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
            if rendered.endswith("torch.cuda.is_available") or rendered.endswith("torch.cuda.device_count"):
                findings.append({"path": path.relative_to(CODE_ROOT).as_posix(),
                                 "line": node.lineno, "type": "local_gpu_probe"})
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            lowered = value.casefold()
            if ("nvidia" + "-smi") in lowered or ("cuda" + "_visible_devices") in lowered:
                findings.append({"path": path.relative_to(CODE_ROOT).as_posix(),
                                 "line": node.lineno, "type": "local_gpu_probe"})
            if re.match(r"^[A-Za-z]:[\\/]", value) or Path(value).suffix.casefold() in DATA_SUFFIXES:
                findings.append({"path": path.relative_to(CODE_ROOT).as_posix(),
                                 "line": node.lineno, "type": "local_data_path"})
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    required = ["scripts/train.py", "scripts/evaluate.py", "pyproject.toml", "configs", "tests"]
    problems = [f"missing:{item}" for item in required if not (CODE_ROOT / item).exists()]
    findings: list[dict[str, object]] = []
    scanned = 0
    for path in CODE_ROOT.rglob("*"):
        if not path.is_file() or any(part in {"outputs", "review", "__pycache__", ".pytest_cache"}
                                     for part in path.relative_to(CODE_ROOT).parts):
            continue
        if path.suffix.casefold() in EXECUTABLE | WEIGHT_SUFFIXES:
            scanned += 1
            findings.extend(violations(path))
    if findings:
        problems.append("forbidden executable behavior or binary detected")
    result = {"status": "pass" if not problems else "fail", "scanned": scanned,
              "problems": problems, "violations": findings,
              "local_gpu_probe": "forbidden_not_run", "local_real_data_allowed": False}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if not problems else 3


if __name__ == "__main__":
    raise SystemExit(main())
