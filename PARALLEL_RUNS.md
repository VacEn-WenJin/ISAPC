# Parallel Multi-Galaxy Runs

This guide shows how to run ISAPC for many galaxies concurrently using processes (recommended for CPU-bound pPXF). It includes resource pinning, per-galaxy logs, a CSV summary, and a dry-run mode.

## Script

- Entry point: `run_all_galaxies_parallel.py`
- Concurrency: `concurrent.futures.ProcessPoolExecutor`
- Progress: `tqdm` progress bar
- Logs: per-galaxy stdout/stderr logs under `logs/parallel_runs/<timestamp>/`
- Summary: `runs.csv` in the same log directory

## Key flags

- `--workers <N>`: number of concurrent processes
- `--mode {P2P,VNB,RDB,ALL}`: which pipeline to run in `main.py`
- `--args "<extra flags>"`: passed through to `main.py` (quote your string)
- `--glob "pattern1,pattern2"`: one or more glob patterns for FITS files
- `--list <file>`: optional list file with one FITS path per line
- `--max-galaxies <N>`: limit total files (smoke run)
- `--pin-blas`: pin OMP/OPENBLAS/MKL/NUMEXPR threads to 2 per worker
- `--log-dir <dir>`: store logs and summary in this directory (default timestamped under `logs/parallel_runs/`)
- `--dry-run`: print the commands without executing

## Examples

### 1) Dry-run a single galaxy

```bash
python run_all_galaxies_parallel.py \
  --glob 'data/IFU/VCC1588_stack.fits' \
  --mode RDB \
  --args "--rdb-equalize-flux --rdb-equalize-n-inner 3 --rdb-bin2-bias 0.9" \
  --workers 1 --max-galaxies 1 --pin-blas --dry-run
```

### 2) Run a small batch (2 processes)

```bash
python run_all_galaxies_parallel.py \
  --glob 'data/IFU/*_stack.fits' \
  --mode ALL \
  --args "--auto-reuse --save-error-maps --rdb-equalize-flux" \
  --workers 2 --pin-blas
```

### 3) Mixed sources via list file and globs

```bash
python run_all_galaxies_parallel.py \
  --glob 'data/IFU/*_stack.fits,data/MUSE/*_stack.fits' \
  --list filelists/more_targets.txt \
  --mode VNB \
  --args "--target-snr 20 --min-snr 2.0" \
  --workers 4 --pin-blas
```

## Tips

- Start with `--dry-run` to confirm commands, then remove it.
- Use modest `--workers` (2–4) plus `--pin-blas` to avoid oversubscription.
- Check `runs.csv` and per-galaxy logs in `logs/parallel_runs/<timestamp>/` for quick diagnostics.
