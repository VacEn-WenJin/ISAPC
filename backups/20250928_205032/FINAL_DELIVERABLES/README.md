# ISAPC Virgo α/Fe Analysis — Final Deliverables

This folder contains the final figures, tables, and reports produced by the ISAPC + AIP pipeline.

## Figures
- `virgo_cluster_final_gradients.png` — Corrected 2D Virgo cluster map
  - Triangles show gradient sign (up/down), filled vs hollow indicates emission.
  - Marker and arrow colors encode relative velocity Δv using the `cool` colormap (symmetric around Δv=0).
  - Substructures (M87/A, M49/B, M60/W, M86/C) are filled by the mean Δv of enclosed galaxies.
  - Δv colorbar ticks at [−vmax, 0, +vmax]; on-plot note “Δv = 0 at v = … km/s” near the scale bar.
  - Arrows start at the data point (triangle center); no “Preliminary” watermark.

- `virgo_cluster_final_with_panels.png` — Companion map + distance panels.

## Tables
- `../alpha_gradient_dual/combined_gradient_summary.csv`
  - Per-galaxy gradients (RDB 3 inner bins; VNB present where valid). Includes slope, error, p-value, R², N.

## Reports
- `FULL_SCIENCE_REPORT.md` — Aggregated AIP fit parameters and per-bin indices for the selected sample (e.g., VCC1146, VCC1588).
- `LITERATURE_FACTS_AUDIT.md` — Facts-to-paper mapping with verbatim supporting sentences.

## Method highlights
- Primary gradient definition: Δ[α/Fe]/Δ(R/Rₑ) using exactly the three innermost RDB bins (N=3; enforced and validated).
- AIP α/Fe maps computed from TMB03 index grid (Hβ, Mgb, Fe5015); out-of-grid pixels are skipped.
- VNB profile loader hardened to accept raw NPZ structure (nested `distance`/`binning`) as well as flattened dicts.

## Recreate artifacts
- Combined gradients CSV:
  ```zsh
  python alpha_gradient_dual/build_combined_gradient_summary.py
  ```
- Final cluster figures:
  ```zsh
  python create_virgo_cluster_final_corrected.py
  python create_virgo_cluster_map_with_vectors.py
  ```
- Full science report (example):
  ```zsh
  python generate_full_report.py --galaxies VCC1146 VCC1588
  ```

## Phase-space diagram
- Generate Virgo phase-space diagram (x=projected distance to M87; y=(v−v_sys)/σ):
  ```zsh
  python virgo_phase_diagram.py \
    --catalog /path/to/apjad3453t1_mrt.txt \
    --output FINAL_DELIVERABLES/virgo_phase_space_diagram.png
  ```
  Options: `--v-sys`, `--sigma`, `--distance-mpc` (default 16.5), `--center-ra/--center-dec` (default M87).


## Example reference values
- VCC1146 (RDB 3-bin): slope = −0.1080 ± 0.0110 dex/Re (N=3)
- VCC1588 (RDB 3-bin): slope = −0.0122 ± 0.0206 dex/Re (N=3)

If you want matched-range VNB fits (constrained to the same radial extent as RDB) or additional galaxies in the report, we can extend the scripts accordingly.
