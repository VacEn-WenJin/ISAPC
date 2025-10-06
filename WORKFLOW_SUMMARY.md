## ISAPC Workflow Summary (Math, Physics, and Pseudocode)

This document captures the full workflow used in this project: P2P → VNB → RDB → AIP. Equations use KaTeX formatting.

### 1) Data and Notation

- Data cube: intensity $I(\lambda, x, y)$, wavelength grid $\lambda$, spatial indices $(x,y)$
- Velocity field $v(x,y)$, dispersion $\sigma(x,y)$ from stellar kinematics
- Binning modes: P2P (per spaxel), VNB (Voronoi SNR), RDB (radial annuli)
- Spectral indices (e.g., Fe5015, Mgb, H\beta) measured per spectrum after rest-frame correction

### 2) Rest-frame correction

For each spaxel $s=(x,y)$, rest-frame shift is applied using line-of-sight velocity $v_s$:

$$ \lambda' = \lambda \left( 1 - \frac{v_s}{c} \right) $$

Spectra are interpolated on a common grid $\lambda'$ before binning/comparison.

### 3) P2P stellar kinematics and indices

Per spaxel, fit composite stellar templates (via pPXF) to extract $v_s,\, \sigma_s$ and continuum; fit emission lines separately if present.

- Kinematic fit objective (schematic):
  $$ \min_\theta \; \lVert I_s(\lambda) - \big( T(\lambda; \theta) * \mathcal{G}(v_s,\sigma_s) + P_m(\lambda) \big) - E(\lambda) \rVert_2^2 $$

where $T$ is the optimal template mix, $\mathcal{G}$ Gaussian broadening, $P_m$ low-order polynomial, and $E$ emission components.

Indices are measured on the rest-frame, emission-cleaned spectrum.

### 4) VNB (Voronoi) binning and fits

- Target SNR $S_\mathrm{target}$; accrete pixels until local SNR meets target.
- Fit binned spectra as in P2P to provide robust maps and an alternative radial profile.

### 5) RDB (Radial) binning with equalized inner bins

- Compute elliptical radius: $$ R = \sqrt{ x'^2 + \left( \frac{y'}{1-\epsilon} \right)^2 } \quad \text{with rotation by PA} $$
- Define annuli (10 bins nominal). Equalize flux for the inner $N=3$ bins with a small bias for bin 2 ($\beta\!<\!1$) so that
  $$ F_0 \approx F_1 \approx F_2, \quad F_1 \leftarrow \beta F_1 \; (\beta \approx 0.9) $$
- Combine per-bin spectra after velocity correction; fit kinematics, measure indices, build radial trends.

### 6) AIP: alpha/Fe derivation and gradient

- Using TMB03 index grid and measured indices, compute per-pixel $[\alpha/\mathrm{Fe}]$ via 2D interpolation over the model grid.
- Aggregate by RDB radial bins to obtain $\{(R_i/\mathrm{Re},\, [\alpha/\mathrm{Fe}]_i)\}$.
- Fit inner-3 gradient:
  $$ [\alpha/\mathrm{Fe}](R) = a + b\,R, \quad b = \arg\min_b \sum_i w_i\big( y_i - a - bR_i \big)^2 $$
  where $w_i$ are inverse-variance or uniform for inner 3 bins.

### 7) Deliverables

- Per galaxy:
  - P2P/VNB/RDB data NPZs under `output/<gal>/Data/`
  - Plots under `output/<gal>/Plots/` incl. RDB inner-3 overlays and AIP map/profile
- Project:
  - Combined galleries (inner3 overlays, alpha/Fe 2D), Virgo cluster RA/Dec map
  - CSV: `output/alpha_fe_gradient_summary.csv`

### 8) Pseudocode overview

```
for galaxy in galaxies:
  cube = load_cube(path)
  p2p = run_p2p(cube, template, z)
  v_field = p2p.velocity

  vnb = run_vnb(cube, v_field, target_snr)

  rdb = run_rdb(cube, v_field,
                equalize_flux=True, n_inner=3, bin2_bias=0.9,
                n_bins=10)

  aip = run_aip_alpha_fe(rdb, vnb, tmb03_model)
  save_plots_and_npz(galaxy, p2p, vnb, rdb, aip)

collect_aip_summary(output/*_AIP_alpha_fe_results.npz) -> output/alpha_fe_gradient_summary.csv
convert_to_cluster_summary() -> alpha_gradient_dual/combined_gradient_summary.csv
create_virgo_cluster_map_with_vectors.py -> FINAL_DELIVERABLES/*.png
```

### 9) Notes on performance and stability

- BLAS threads pinned to 1; use `-j` for safe parallel jobs.
- Rest-frame correction applied per spaxel prior to binning ensures consistent indices.
- RDB inner-3 equalization stabilizes inner gradient and comparison across galaxies.
