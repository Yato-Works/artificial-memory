"""Verify that the AM Apex benchmark lab matches its pinned official sources.

This intentionally does not fetch, mutate, or evaluate anything.  It is safe to run
before every frozen benchmark run and reports missing data separately from a code
revision mismatch.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "benchmark_config" / "official_benchmarks.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    failed = False
    print("AM Apex official benchmark lab verification")
    for key, spec in registry["benchmarks"].items():
        checkout = ROOT / spec["checkout"]
        head = git_head(checkout)
        revision_ok = head == spec["pinned_commit"]
        if not revision_ok:
            failed = True
        print(f"\n[{key}] {'OK' if revision_ok else 'REVISION MISMATCH'}")
        print(f"  source: {spec['official_repository']}")
        print(f"  code:   {head or 'MISSING'}")
        requirements = spec.get(
            "dataset_requirements", [{"path": path} for path in spec["dataset_paths"]]
        )
        for requirement in requirements:
            relative = requirement["path"]
            path = ROOT / relative
            if not path.exists():
                failed = True
                print(f"  data:   MISSING {relative}")
            elif path.is_file():
                print(f"  data:   {relative} sha256={sha256(path)}")
            else:
                count = sum(1 for item in path.rglob("*") if item.is_file())
                minimum = requirement.get("minimum_file_count")
                if minimum is not None and count < minimum:
                    failed = True
                    print(f"  data:   INCOMPLETE {relative} ({count}/{minimum} files)")
                else:
                    print(f"  data:   {relative} ({count} files; hash captured by freeze manifest)")
    if failed:
        print("\nINCOMPLETE: run scripts/benchmarks/bootstrap_official_benchmarks.ps1 as needed.")
        return 1
    print("\nREADY: all pinned code and expected data paths are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
