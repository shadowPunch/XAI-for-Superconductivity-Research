"""
tracking.py
-----------
Weights & Biases run setup shared by every script that trains, evaluates or
explains a model. Only configs, metrics and derived tables/plots are ever
logged -- never raw data.

Set SCREENING_WANDB=0 to run fully offline (W&B calls become no-ops).
If W&B is enabled but unreachable or unauthenticated, wandb.init raises and
the script stops rather than running untracked.
"""

import os
import hashlib
import subprocess
from pathlib import Path

import wandb

PROJECT = "xai-superconductivity-screening"
ROOT = Path(__file__).parent


def start_run(name: str, job_type: str, config: dict, tags=(), notes: str = ""):
    """Start a W&B run (use as a context manager so failures mark the run failed)."""
    mode = "disabled" if os.environ.get("SCREENING_WANDB", "1") == "0" else None
    return wandb.init(
        project=PROJECT, name=name, job_type=job_type,
        config={**config, "code_version": code_version()},
        tags=list(tags), notes=notes, mode=mode,
    )


def code_version() -> str:
    """Short git SHA, suffixed '+dirty' if the working tree has uncommitted changes."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    sha = git("rev-parse", "--short", "HEAD") or "unknown"
    return sha + ("+dirty" if git("status", "--porcelain") else "")


def file_md5(path: Path) -> str:
    """Content hash used as a data/model version identifier."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()
