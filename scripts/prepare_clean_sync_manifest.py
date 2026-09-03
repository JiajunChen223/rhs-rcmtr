"""Fail-closed deterministic code-only synchronization manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = CODE_ROOT / "manifests" / "clean_sync_manifest.json"
ALLOWED_TOP_FILES = {"README.md", "pyproject.toml", ".gitignore"}
ALLOWED_ROOTS = {"src", "scripts", "configs", "tests", "environment"}
ALLOWED_SUFFIXES = {"." + token for token in ("py", "yaml", "yml", "toml", "md", "txt", "lock", "json")}
IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", "outputs", "manifests", "review"}
SECRET_PATTERN = re.compile("|".join(("AK" + "IA" + "[0-9A-Z]{16}", "PRIVATE" + ".KEY", "api" + ".key\\s*=", "pass" + "word\\s*=")), re.I)
WINDOWS_ABSOLUTE = re.compile("[A-Za-z]:" + re.escape("\\"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory() -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    problems: list[str] = []
    for path in sorted(CODE_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(CODE_ROOT)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        allowed = relative.as_posix() in ALLOWED_TOP_FILES or (
            relative.parts[0] in ALLOWED_ROOTS and path.suffix.casefold() in ALLOWED_SUFFIXES
        )
        if not allowed:
            problems.append(f"unapproved sync path: {relative.as_posix()}")
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        if SECRET_PATTERN.search(content):
            problems.append(f"credential-like content: {relative.as_posix()}")
        if WINDOWS_ABSOLUTE.search(content):
            problems.append(f"Windows absolute path: {relative.as_posix()}")
        files.append({"path": relative.as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size})
    if problems:
        raise ValueError("; ".join(problems))
    return files


def verify(manifest: dict[str, object]) -> None:
    current = inventory()
    if manifest.get("files") != current or manifest.get("file_count") != len(current):
        raise ValueError("clean-sync manifest is stale or inconsistent")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        payload = json.loads(OUTPUT.read_text(encoding="utf-8"))
        verify(payload)
        print(json.dumps({"status": "pass", "verified": True, "file_count": payload["file_count"]}))
        return
    files = inventory()
    payload = {
        "artifact_type": "researchpilot_clean_sync_manifest", "schema_version": 1,
        "status": "pass", "code_root": ".", "file_count": len(files), "files": files,
        "scan_policy": "allowlist_plus_secret_and_absolute_path_fail_closed",
        "local_real_data_included": False, "credentials_included": False,
        "checkpoint_binaries_included": False, "local_gpu_probe": "forbidden_not_run",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "output": str(OUTPUT), "file_count": len(files)}))


if __name__ == "__main__":
    main()
