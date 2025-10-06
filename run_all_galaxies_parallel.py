#!/usr/bin/env python3
"""
ISAPC Batch Processing - Parallel (ProcessPool)

Runs multiple galaxies concurrently using processes to avoid GIL/BLAS contention.
Configurable worker count and environment thread pinning.

Usage:
    python run_all_galaxies_parallel.py --workers 3 --mode ALL --args "--auto-reuse --save-error-maps"

You can pass additional CLI flags for main.py via --args string.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from galaxy_catalog import get_redshift


def build_cmd(fits_file: str, extra_args: str, mode: str) -> list[str]:
    galaxy_name = Path(fits_file).stem.replace('_stack', '')
    z = get_redshift(galaxy_name)
    # Default template path heuristic
    tmpl = 'templates/spectra_emiles_9.0.npz' if Path('templates/spectra_emiles_9.0.npz').exists() else 'data/templates/spectra_emiles_9.0.npz'
    base = [
        sys.executable, 'main.py',
        fits_file,
        '-z', str(z),
        '-t', tmpl,
        '-o', 'output',
        '-m', mode,
        '--auto-reuse',
    ]
    if extra_args:
        base += shlex.split(extra_args)
    return base


def run_one(
    fits_file: str,
    extra_args: str,
    mode: str,
    env_vars: dict[str, str],
    log_dir: str | None = None,
) -> tuple[str, bool, float, str | None]:
    name = Path(fits_file).stem.replace('_stack', '')
    cmd = build_cmd(fits_file, extra_args, mode)
    start = time.time()
    env = os.environ.copy()
    env.update(env_vars)
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        dur = time.time() - start
        if log_dir:
            (Path(log_dir) / f"{name}.out.log").write_text(res.stdout)
            (Path(log_dir) / f"{name}.err.log").write_text(res.stderr)
        return name, True, dur, None
    except subprocess.CalledProcessError as e:
        dur = time.time() - start
        if log_dir:
            (Path(log_dir) / f"{name}.out.log").write_text(e.stdout or '')
            (Path(log_dir) / f"{name}.err.log").write_text(e.stderr or '')
        return name, False, dur, e.stderr


def main():
    ap = argparse.ArgumentParser(description='Run ISAPC for all galaxies in parallel')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 2), help='Number of parallel worker processes')
    ap.add_argument('--mode', type=str, choices=['P2P', 'VNB', 'RDB', 'ALL'], default='ALL', help='Analysis mode for main.py')
    ap.add_argument('--args', type=str, default='', help='Extra args passed to main.py (quoted string)')
    ap.add_argument('--glob', type=str, default='data/*/*_stack.fits', help='Glob (or comma-separated globs) to select FITS files')
    ap.add_argument('--list', type=str, default='', help='Optional file with one FITS path per line')
    ap.add_argument('--max-galaxies', type=int, default=0, help='If >0, limit to first N files (smoke run)')
    ap.add_argument('--pin-blas', action='store_true', help='Pin BLAS threads per worker (recommended)')
    ap.add_argument('--log-dir', type=str, default='', help='Directory to store per-galaxy stdout/stderr logs')
    ap.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    args = ap.parse_args()

    def expand_globs(patterns: str) -> list[str]:
        files: list[str] = []
        for p in [s.strip() for s in patterns.split(',') if s.strip()]:
            files.extend(glob.glob(p))
        return files

    fits_files: list[str] = []
    if args.glob:
        fits_files.extend(expand_globs(args.glob))
    if args.list:
        list_path = Path(args.list)
        if list_path.exists():
            for line in list_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                fits_files.append(line)
        else:
            print('List file not found:', args.list)
    # De-duplicate and sort
    fits_files = sorted({f for f in fits_files})
    if not fits_files:
        print('No FITS files found. Patterns:', args.glob, 'List:', args.list)
        return 2
    if args.max_galaxies and args.max_galaxies > 0:
        fits_files = fits_files[: args.max_galaxies]

    env_vars: dict[str, str] = {}
    if args.pin_blas:
        # Keep BLAS usage modest per process to reduce oversubscription
        env_vars.update({
            'OMP_NUM_THREADS': '2',
            'OPENBLAS_NUM_THREADS': '2',
            'MKL_NUM_THREADS': '2',
            'NUMEXPR_NUM_THREADS': '2',
        })

    # Prepare logging dir
    log_dir = args.log_dir
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    else:
        # default time-stamped log dir under logs/parallel_runs
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = str(Path('logs') / 'parallel_runs' / ts)
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    # CSV summary path
    summary_csv = Path(log_dir) / 'runs.csv'

    # Info header
    cpu_count = os.cpu_count() or 1
    blas_threads = env_vars.get('OMP_NUM_THREADS', 'default') if args.pin_blas else 'inherit'
    print(f"Running {len(fits_files)} galaxies with {args.workers} workers; mode={args.mode}")
    print(f"Host CPU cores: {cpu_count}; BLAS threads per worker: {blas_threads}; Logs: {log_dir}")

    if args.dry_run:
        print('Dry run commands:')
        for f in fits_files:
            print(' ', ' '.join(build_cmd(f, args.args, args.mode)))
        return 0

    t0 = time.time()
    results: list[tuple[str, bool, float]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex, open(summary_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['galaxy', 'ok', 'duration_s'])
        futs = {ex.submit(run_one, f, args.args, args.mode, env_vars, log_dir): f for f in fits_files}
        with tqdm(total=len(fits_files), unit='gal', desc='Processing') as pbar:
            for fut in as_completed(futs):
                name, ok, dt, err = fut.result()
                status = 'ok' if ok else 'fail'
                pbar.set_postfix({status: name, 'last_s': f'{dt:.1f}'})
                pbar.update(1)
                if not ok and err:
                    tail = '\n'.join(err.splitlines()[-10:])
                    print(f"\n{name} error tail:\n{tail}\n")
                writer.writerow([name, ok, f"{dt:.3f}"])
                results.append((name, ok, dt))

    elapsed = time.time() - t0
    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"Done in {elapsed/60:.1f} min; {n_ok}/{len(results)} succeeded")
    return 0 if n_ok == len(results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
