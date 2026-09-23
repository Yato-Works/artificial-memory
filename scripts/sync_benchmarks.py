#!/usr/bin/env python3
"""Sync benchmark datasets and third-party repos to Linux evaluation environment.

Usage:
    python scripts/sync_benchmarks.py --host night-engine-room --remote-path /home/eli/Projects/artificial_memory
    
Requires:
    - SSH access to target host
    - scp available (standard with OpenSSH on Windows 10+)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run command and return result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result


def sync_directory(local_dir: Path, host: str, remote_path: str, dry_run: bool = False) -> bool:
    """Sync a directory to remote using scp."""
    if not local_dir.exists():
        print(f"Local directory not found: {local_dir}")
        return False
    
    remote_dir = f"{host}:{remote_path}/{local_dir.relative_to(local_dir.anchor).as_posix()}"
    # Use scp -r for recursive copy
    cmd = ["scp", "-r"]
    if dry_run:
        print(f"[DRY RUN] Would copy {local_dir} to {remote_dir}")
        return True
    cmd.extend([str(local_dir), remote_dir])
    
    result = run_cmd(cmd)
    return result.returncode == 0


def sync_datasets(local_root: Path, host: str, remote_path: str, dry_run: bool = False) -> bool:
    """Sync datasets/external to remote."""
    local_datasets = local_root / "datasets" / "external"
    return sync_directory(local_datasets, host, remote_path, dry_run)


def sync_third_party(local_root: Path, host: str, remote_path: str, dry_run: bool = False) -> bool:
    """Sync third_party/benchmarks to remote."""
    local_tp = local_root / "third_party" / "benchmarks"
    return sync_directory(local_tp, host, remote_path, dry_run)


def sync_configs(local_root: Path, host: str, remote_path: str, dry_run: bool = False) -> bool:
    """Sync benchmark_config to remote."""
    local_config = local_root / "benchmark_config"
    return sync_directory(local_config, host, remote_path, dry_run)


def verify_remote(host: str, remote_path: str) -> bool:
    """Verify remote has required structure."""
    cmd = [
        "ssh", host,
        f"ls -la {remote_path}/datasets/external/ 2>/dev/null && "
        f"ls -la {remote_path}/third_party/benchmarks/ 2>/dev/null && "
        f"ls -la {remote_path}/benchmark_config/ 2>/dev/null"
    ]
    result = run_cmd(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Sync benchmarks to Linux eval environment")
    parser.add_argument("--host", type=str, default="night-engine-room",
                        help="SSH hostname (default: night-engine-room)")
    parser.add_argument("--remote-path", type=str, default="/home/eli/Projects/artificial_memory",
                        help="Remote project path")
    parser.add_argument("--local-root", type=str, default=".",
                        help="Local project root (default: current dir)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be synced without doing it")
    parser.add_argument("--skip-datasets", action="store_true",
                        help="Skip datasets/external sync")
    parser.add_argument("--skip-third-party", action="store_true",
                        help="Skip third_party/benchmarks sync")
    parser.add_argument("--skip-configs", action="store_true",
                        help="Skip benchmark_config sync")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify remote structure")
    
    args = parser.parse_args()
    
    local_root = Path(args.local_root).resolve()
    if not local_root.exists():
        print(f"Local root not found: {local_root}")
        return 1
    
    print(f"Local root: {local_root}")
    print(f"Remote: {args.host}:{args.remote_path}")
    print("=" * 60)
    
    if args.verify_only:
        print("Verifying remote structure...")
        if verify_remote(args.host, args.remote_path):
            print("✓ Remote structure verified")
            return 0
        else:
            print("✗ Remote verification failed")
            return 1
    
    success = True
    
    if not args.skip_datasets:
        print("\n>>> Syncing datasets/external...")
        if not sync_datasets(local_root, args.host, args.remote_path, args.dry_run):
            print("✗ datasets sync failed")
            success = False
        else:
            print("✓ datasets synced")
    
    if not args.skip_third_party:
        print("\n>>> Syncing third_party/benchmarks...")
        if not sync_third_party(local_root, args.host, args.remote_path, args.dry_run):
            print("✗ third_party sync failed")
            success = False
        else:
            print("✓ third_party synced")
    
    if not args.skip_configs:
        print("\n>>> Syncing benchmark_config...")
        if not sync_configs(local_root, args.host, args.remote_path, args.dry_run):
            print("✗ benchmark_config sync failed")
            success = False
        else:
            print("✓ benchmark_config synced")
    
    if success:
        print("\n" + "=" * 60)
        print("All sync operations completed successfully!")
        print("Run verification:")
        print(f"  python scripts/sync_benchmarks.py --verify-only --host {args.host} --remote-path {args.remote_path}")
    else:
        print("\n" + "=" * 60)
        print("Some sync operations failed. Check output above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())