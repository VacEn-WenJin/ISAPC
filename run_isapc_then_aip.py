#!/usr/bin/env python3
"""
Orchestrate ISAPC → AIP for a single galaxy:
1) Wait for full-IFU P2P results to exist in output/<galaxy>_stack/Data
2) Run VNB (auto-reuse P2P velocities)
3) Run RDB with flux-equalized inner bins (n=3, bin2 bias configurable)
4) Run AIP alpha/Fe and radial gradients

Usage:
  python run_isapc_then_aip.py --galaxy VCC_1049_1 \
    --redshift 0.0037 --template templates/spectra_emiles_9.0.npz \
    [--rdb-equalize-n-inner 3] [--rdb-bin2-bias 0.9]

This script assumes main.py and run_aip_alpha_fe.py are available in the repo root.
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
import sys
import os


def run(cmd: list[str], env: dict | None = None) -> int:
    print("→ Running:", " ".join(cmd), flush=True)
    try:
        r = subprocess.run(cmd, check=False, env=env)
        print(f"  Exit code: {r.returncode}")
        return r.returncode
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as e:
        print(f"Error running command: {e}")
        return 1


def wait_for_file(path: Path, poll_sec: int = 60, timeout_sec: int | None = None) -> bool:
    """Poll for a file to exist; return True when found, False on timeout."""
    print(f"⏳ Waiting for file: {path}")
    start = time.time()
    while True:
        if path.exists():
            print("✓ Found file.")
            return True
        if timeout_sec is not None and (time.time() - start) > timeout_sec:
            print("✗ Timeout waiting for file.")
            return False
        time.sleep(poll_sec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--galaxy", required=True, help="Galaxy base name (e.g., VCC_1049_1 or VCC1146_obstack)")
    ap.add_argument("--redshift", required=True, type=float)
    ap.add_argument("--template", required=True, help="Template npz path")
    ap.add_argument("--data", default=None, help="Input cube path (defaults to data/IFU/<galaxy>_stack.fits)")
    ap.add_argument("--output-stem", default=None, help="Override output stem (default: <galaxy> if it already ends with _stack/_obstack else <galaxy>_stack)")
    ap.add_argument("--poll", type=int, default=60, help="Poll interval in seconds")
    ap.add_argument("--timeout", type=int, default=None, help="Optional timeout in seconds")
    ap.add_argument("--rdb-equalize-n-inner", type=int, default=3)
    ap.add_argument("--rdb-bin2-bias", type=float, default=0.9)
    ap.add_argument("--snr", type=float, default=30.0, help="Target SNR for VNB")
    ap.add_argument("--min-snr", type=float, default=1.0)
    ap.add_argument("-j", "--n-jobs", type=int, default=4, help="Parallel jobs to pass to child runs")
    ap.add_argument("--blas-threads", type=int, default=1, help="Threads for BLAS/OpenMP backends in children")
    args = ap.parse_args()

    galaxy = args.galaxy
    # Determine output stem
    if args.output_stem:
        out_stem = args.output_stem
    else:
        if galaxy.endswith(("_stack", "_obstack", "_stacked")):
            out_stem = galaxy
        else:
            out_stem = f"{galaxy}_stack"
    cube = args.data or f"data/IFU/{out_stem}.fits"
    out_root = Path("output") / out_stem
    data_dir = out_root / "Data"
    p2p_file = data_dir / f"{out_stem}_P2P_results.npz"

    # Prepare environment to avoid oversubscription in child processes
    child_env = os.environ.copy()
    t = str(max(1, int(args.blas_threads)))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        child_env.setdefault(var, t)

    # 1) Wait for P2P to complete
    ok = wait_for_file(p2p_file, poll_sec=args.poll, timeout_sec=args.timeout)
    if not ok:
        sys.exit(1)

    # 2) VNB (auto-reuse P2P)
    vnb_cmd = [
        sys.executable, "main.py", cube,
        "-m", "VNB",
        "-z", str(args.redshift),
        "-t", args.template,
        "-o", "output",
        "-j", str(args.n_jobs),
        "--auto-reuse",
        "--target-snr", str(args.snr),
        "--min-snr", str(args.min_snr),
    ]
    if run(vnb_cmd, env=child_env) != 0:
        print("VNB failed or was interrupted; continuing to RDB may produce limited outputs.")

    # 3) RDB with requested binning strategy
    rdb_cmd = [
        sys.executable, "main.py", cube,
        "-m", "RDB",
        "-z", str(args.redshift),
        "-t", args.template,
        "-o", "output",
        "-j", str(args.n_jobs),
        "--auto-reuse",
        "--rdb-equalize-flux",
        "--rdb-equalize-n-inner", str(args.rdb_equalize_n_inner),
        "--rdb-bin2-bias", str(args.rdb_bin2_bias),
    ]
    run(rdb_cmd, env=child_env)

    # 4) AIP alpha/Fe
    aip_cmd = [sys.executable, "run_aip_alpha_fe.py", "--galaxy", out_stem]
    run(aip_cmd, env=child_env)

    print("All steps attempted.")


if __name__ == "__main__":
    main()
