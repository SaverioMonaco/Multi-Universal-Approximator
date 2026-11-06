"""Shared bits of the study scripts: repo paths and the run record."""
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
RESULTS = REPO / "results"  # gitignored


def provenance() -> dict:
    try:
        commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                         text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain",
                                              "src", "scripts"], text=True).strip())
    except Exception:
        commit, dirty = "unknown", None
    return {"repo": "Multi-Universal-Approximator", "commit": commit, "dirty": dirty,
            "script": Path(sys.argv[0]).name, "argv": sys.argv[1:], "host": platform.node()}
