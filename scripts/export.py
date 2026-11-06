"""Copy the results JSON to another folder (e.g. a paper or thesis data folder).

    python scripts/export.py --dest DIR      # or set MUA_EXPORT_DIR
"""
import argparse
import os
import shutil
from pathlib import Path

import _common

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=os.environ.get("MUA_EXPORT_DIR"))
    args = ap.parse_args()
    if args.dest is None:
        ap.error("give --dest or set MUA_EXPORT_DIR")
    args.dest = Path(args.dest)
    args.dest.mkdir(parents=True, exist_ok=True)
    for f in sorted(_common.RESULTS.glob("*/*.json")):
        shutil.copy2(f, args.dest / f.name)
        print(f"{f.relative_to(_common.REPO)} -> {args.dest / f.name}")


if __name__ == "__main__":
    main()
