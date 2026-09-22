"""Download the official PERMA dataset snapshot (ustclsc/PERMA) into datasets/official/perma.

Uses the pinned perma evaluation venv's huggingface_hub so no brotli codec is involved
(the hermes hf.exe CLI hits an httpx brotli decoding error on this machine).

Usage:
    .venv-benchmarks/perma/Scripts/python.exe scripts/benchmarks/download_perma.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEST = REPO_ROOT / "datasets" / "official" / "perma"


def main() -> int:
    from huggingface_hub import snapshot_download

    DEST.mkdir(parents=True, exist_ok=True)
    print(f"[perma] downloading ustclsc/PERMA -> {DEST}", flush=True)
    path = snapshot_download(
        "ustclsc/PERMA",
        repo_type="dataset",
        local_dir=str(DEST),
        max_workers=4,
    )
    print(f"[perma] DONE: {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
