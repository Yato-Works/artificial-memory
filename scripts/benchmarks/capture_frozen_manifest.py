"""Capture immutable provenance for one AM benchmark run.

The manifest never contains API keys. It hashes local data/config/result inputs and
records all pinned upstream checkout revisions, making score claims auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def checkout_head(relative: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT / relative), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--top-k", required=True, type=int)
    parser.add_argument("--context-token-budget", required=True, type=int)
    args = parser.parse_args()

    registry = json.loads((ROOT / "benchmark_config" / "official_benchmarks.json").read_text(encoding="utf-8"))
    spec = registry["benchmarks"].get(args.benchmark)
    if spec is None:
        raise SystemExit(f"Unknown benchmark: {args.benchmark}")
    files = {"dataset": args.dataset, "config": args.config, "answers": args.answers, "result": args.result}
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise SystemExit(f"Files must exist before freezing: {', '.join(missing)}")
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": args.benchmark,
        "official_repository": spec["official_repository"],
        "pinned_commit": spec["pinned_commit"],
        "checked_out_commit": checkout_head(spec["checkout"]),
        "model": args.model,
        "judge": args.judge,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "context_token_budget": args.context_token_budget,
        "sha256": {name: digest(path) for name, path in files.items()},
    }
    output = args.result.with_suffix(args.result.suffix + ".frozen-manifest.json")
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
