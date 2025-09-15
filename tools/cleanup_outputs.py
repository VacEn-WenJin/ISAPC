#!/usr/bin/env python3
"""
Cleanup utility for ISAPC outputs

Removes accidental nested galaxy folders (e.g., output/VCC1588_stack/VCC1588_stack)
and duplicate deliverables under output/FINAL_DELIVERABLES when a top-level
FINAL_DELIVERABLES exists.

Usage:
  python tools/cleanup_outputs.py           # dry run, prints actions
  python tools/cleanup_outputs.py --apply   # actually perform deletions/moves
"""
from pathlib import Path
import shutil
import argparse


def is_trivially_empty(d: Path) -> bool:
    """Return True if directory contains no files and <= 3 empty subdirs."""
    if not d.exists() or not d.is_dir():
        return True
    files = [p for p in d.rglob('*') if p.is_file()]
    if files:
        return False
    # Count subdirs
    subdirs = [p for p in d.rglob('*') if p.is_dir()]
    return len(subdirs) <= 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs', default='output', help='Outputs root')
    ap.add_argument('--apply', action='store_true', help='Perform changes')
    args = ap.parse_args()

    outputs = Path(args.outputs)
    if not outputs.exists():
        print(f"Outputs dir not found: {outputs}")
        return 1

    planned = []

    # 1) Remove nested duplicate galaxy folders: output/<g>_stack/<g>_stack
    for gdir in outputs.iterdir():
        if not gdir.is_dir() or not gdir.name.endswith('_stack'):
            continue
        nested = gdir / gdir.name
        if nested.exists() and nested.is_dir():
            # Only remove if trivially empty
            if is_trivially_empty(nested):
                planned.append(("rmdir", nested))

    # 2) Consolidate duplicate deliverables under output/FINAL_DELIVERABLES
    top_final = outputs.parent / 'FINAL_DELIVERABLES'
    inner_final = outputs / 'FINAL_DELIVERABLES'
    if inner_final.exists() and inner_final.is_dir():
        if top_final.exists() and top_final.is_dir():
            # Move contents up if any missing, then remove inner
            for item in inner_final.iterdir():
                dest = top_final / item.name
                if dest.exists():
                    continue
                planned.append(("move", item, dest))
            planned.append(("rmdir", inner_final))
        else:
            # Rename/move inner to top
            planned.append(("move", inner_final, top_final))

    # Execute or print
    if not planned:
        print("Nothing to clean.")
        return 0

    for action in planned:
        if action[0] == 'rmdir':
            target = action[1]
            print(f"Remove directory: {target}")
            if args.apply:
                shutil.rmtree(target, ignore_errors=True)
        elif action[0] == 'move':
            src, dst = action[1], action[2]
            print(f"Move: {src} -> {dst}")
            if args.apply:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))

    if args.apply:
        print("Cleanup applied.")
    else:
        print("Dry run. Re-run with --apply to perform changes.")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
