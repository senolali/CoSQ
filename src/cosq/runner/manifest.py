"""Run provenance.

The manifest is what lets someone explain a discrepancy a year later: the commit, the
seed, the pinned model revision, the environment. A run whose working tree was dirty
is recorded as such, and by default refused outright — results that cannot be mapped
back to a commit are not results.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any

_TRACKED_PACKAGES = ("numpy", "scipy", "PyYAML", "torch", "transformers", "bitsandbytes")


def _run_git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_state() -> dict[str, Any]:
    commit = _run_git("rev-parse", "HEAD")
    status = _run_git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": _run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def build_manifest(
    *,
    run_id: str,
    config: dict[str, Any],
    config_hash: str,
    seed: int,
    dataset: dict[str, Any],
    backend: dict[str, str],
    prompts_digest: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "config_hash": config_hash,
        "prompts_digest": prompts_digest,
        "git": git_state(),
        "seed": seed,
        "dataset": dataset,
        "backend": backend,
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
    }
