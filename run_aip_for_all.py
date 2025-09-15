#!/usr/bin/env python3
"""
Batch runner for AIP alpha/Fe on all completed ISAPC galaxy outputs.

- Scans ./output/*_stack/Data for ISAPC results
- Runs run_aip_alpha_fe.py per galaxy
- Collects deliverables into ./FINAL_DELIVERABLES/<galaxy>_stack

Usage:
  python run_aip_for_all.py [--outputs ./output] [--deliver ./FINAL_DELIVERABLES]
"""
import subprocess
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outputs', default='output', help='ISAPC outputs root (contains *_stack)')
    ap.add_argument('--deliver', default='FINAL_DELIVERABLES', help='Deliverables root')
    args = ap.parse_args()

    outputs_dir = Path(args.outputs)
    deliver_dir = Path(args.deliver)
    deliver_dir.mkdir(parents=True, exist_ok=True)

    if not outputs_dir.exists():
        log.error(f'Outputs dir not found: {outputs_dir}')
        return 1

    galaxies = [p for p in outputs_dir.iterdir() if p.is_dir() and p.name.endswith('_stack')]
    if not galaxies:
        log.warning('No galaxy outputs found to run AIP on.')
        return 0

    ok = 0
    for gdir in galaxies:
        galaxy = gdir.name.replace('_stack', '')
        log.info(f'Running AIP for {galaxy}…')
        # Use current Python to call the module
        res = subprocess.run([
            'python', 'run_aip_alpha_fe.py', '--galaxy', galaxy
        ], cwd=Path(__file__).parent)
        if res.returncode == 0:
            ok += 1
        else:
            log.error(f'AIP failed for {galaxy} (code {res.returncode})')

    # Collect deliverables
    log.info('Collecting deliverables…')
    subprocess.run([
        'python', 'aip_collect_results.py', '--outputs-dir', str(outputs_dir), '--output-root', str(deliver_dir)
    ], cwd=Path(__file__).parent)

    log.info(f'Finished AIP. Success: {ok}/{len(galaxies)}')
    return 0 if ok > 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
