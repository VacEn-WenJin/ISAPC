# ISAPC + AIP Final Rerun Summary (2025-09-27)

This document summarizes the completed rerun, plots generated, velocity/rest-frame checks, and AIP gradient changes.

## What was produced

- Normalized spectra for the three innermost RDB bins per galaxy and overlays:
  - Collected here: `combined_rdb_plots/` (flat, galaxy-prefixed filenames)
  - Files per galaxy: `RDB0_norm.png`, `RDB1_norm.png`, `RDB2_norm.png`,
    `RDB_first3_norm_overlay.png`, `RDB_first3_norm_shaded_overlay.png`, `RDB_first3_norm_combined.png`
- AIP gradient summary and comparison to prior run:
  - New combined table: `FINAL_DELIVERABLES/tables/combined_gradient_summary.csv`
  - Old vs new diff: `FINAL_DELIVERABLES/tables/alpha_fe_gradient_changes.csv`
- Velocity/rest-frame diagnostic:
  - Full output: `FINAL_DELIVERABLES/velocity_alignment.txt`

## AIP gradient changes (old vs new)

- Overlap galaxies compared: 18
- Sign changes in slope: 6
  - VCC0308: 0.042 → -0.072 (Δ=-0.114)
  - VCC1146: 0.001 → -0.000 (Δ=-0.001)
  - VCC1368: -0.033 → 0.022 (Δ=+0.055)
  - VCC1486: 0.029 → -0.010 (Δ=-0.039)
  - VCC1588: -0.002 → 0.016 (Δ=+0.017)
  - VCC1902: -0.038 → 0.027 (Δ=+0.065)
- Mean absolute slope change: ~0.0532
- Top |Δslope|:
  - VCC0308 (-0.114), VCC1193 (-0.113), VCC1811 (-0.102), VCC1410 (-0.091), VCC1049 (-0.082)

See `tables/alpha_fe_gradient_changes.csv` for full details and metrics.

## Velocity/rest-frame alignment

- All RDB binned NPZ files advertise rest frame (wave_frame=rest, rest_frame_applied=True).
- Diagnostic median Δv summary (first 3 RDB bins), threshold=80 km/s:
  - PASS: VCC0688, VCC0990, VCC1193, VCC1368, VCC1549, VCC1588, VCC1695, VCC1902
  - FLAG: VCC0308, VCC0667, VCC1049, VCC1146, VCC1410, VCC1431, VCC1486, VCC1811, VCC1890, VCC1910, VCC1949
- Caveat: current diagnostic doesn’t subtract per-bin stellar velocity; sharp absorption minima can bias Δv. This can produce large |Δv| even with properly rest-framed products.

Raw output with per-galaxy medians and std is in `FINAL_DELIVERABLES/velocity_alignment.txt`.

## Plot collection

- Verified and collected for 20/20 galaxies into `combined_rdb_plots/` (flat layout).
- You can browse quickly by sorting filenames (e.g., `VCC1049_RDB_first3_norm_overlay.png`).

## Recreate/build helpers

- Verify or build all normalized plots:
  - `python -u tools/build_rdb_all_plots.py` (use `--force` to refresh)
- Collect plots to a folder:
  - `python -u tools/collect_rdb_plots.py --dest combined_rdb_plots --flat`
- Run velocity alignment summary:
  - `python -u tools/check_velocity_alignment.py --threshold 80 > FINAL_DELIVERABLES/velocity_alignment.txt`

## Next steps (optional)

- Refine Δv diagnostic by subtracting per-bin stellar velocities (velocity_binned) before measuring line centers.
- Add a one-pager with thumbnails of overlays for quick inspection.
- Confirm emission-subtracted spectra are consistently used in index measurements for the two-panel plots (audit script available on request).
