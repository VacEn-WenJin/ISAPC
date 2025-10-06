# ISAPC → AIP Short Workflow Summary

This document summarizes the end-to-end pipeline from raw data to results, with the essential plots and formulas.

## 1) Data inputs and preprocessing
- Inputs: Reduced IFU spectra cubes (per galaxy) and ancillary photometry (for R/Re and position angle).
- Preprocessing: mask sky lines/bad pixels, Voronoi binning to S/N target, wavelength calibration.
- Emission handling: pPXF stellar+gas fit; subtract gas component; export original vs. normalized (continuum-flattened) spectra.

Example output: see normalized-vs-original spectra in `FULL_SCIENCE_REPORT.md` for VCC1146/VCC1588.

## 2) Index measurement and α/Fe mapping (AIP)
- Measured Lick/TMB03 indices per spatial bin: Hβ, Mgb, Fe5015.
- Map to TMB03 model grid to infer [α/Fe] at each bin (out-of-grid masked).
- Produce α/Fe 2D map and annular/radial bin profiles.

Formula (indices → α/Fe via grid interpolation):
- Given indices vector I = (Hβ, Mgb, Fe5015), find model parameters θ = ([α/Fe], age, [Z/H]) minimizing Δ(I, I_model(θ)) on the TMB03 grid.

Illustration: see `FINAL_DELIVERABLES/alpha_fe_from_indices_VCC1146_bin1.png` (and `..._VCC1588_bin1.png`) showing the observed bin over the TMB03 Fe5015–Mgb plane colored by model [α/Fe], with Hβ context and the interpolated result.

## 3) Radial profiles and effective radius normalization
- Compute radial distance per bin, convert to normalized radius r = R/Rₑ.
- Two profile sources:
  - RDB: Radial Distance Binning of α/Fe map (primary source).
  - VNB: Voronoi (original bin) profile when valid.

## 4) Gradient definition and fitting
Primary definition (enforced): use exactly the three innermost RDB bins to fit a line:
- Model: [α/Fe](r) = a + b·r, with r = R/Rₑ.
- Gradient: b = Δ[α/Fe]/Δ(R/Rₑ).
- Fit: weighted least squares on the 3 inner bins; report slope b, σ_b, p-value, R², N.

Key outputs: `alpha_gradient_dual/combined_gradient_summary.csv` (includes RDB 3-bin and VNB when available).

## 5) Cluster-level visualization
- 2D Virgo cluster map with galaxies placed by sky position.
- Coloring by relative velocity Δv (symmetric, `cool` colormap), with substructure fills (M87/A, M49/B, M60/W, M86/C) colored by mean Δv.
- Triangle markers encode gradient sign and emission; arrows start at datapoints and length ∝ |b|.
- Δv=0 reference labeled near the scale bar; no watermark.

See `FINAL_DELIVERABLES/virgo_cluster_final_gradients.png` and `FINAL_DELIVERABLES/virgo_cluster_final_with_panels.png`.

## 6) Validation examples (VCC1146, VCC1588)
- VCC1146: RDB 3-bin b = −0.1080 ± 0.0110 dex/Re (N=3)
- VCC1588: RDB 3-bin b = −0.0122 ± 0.0206 dex/Re (N=3)

Details, per-bin indices, and spectra panels: `FINAL_DELIVERABLES/FULL_SCIENCE_REPORT.md`.

## 7) Core formulas (quick reference)
- Normalized radius: r = R/Rₑ
- Linear gradient: [α/Fe](r) = a + b·r
- Slope uncertainty (WLS): σ_b from covariance of (XᵀWX)⁻¹
- Relative velocity: Δv = v_gal − v_ref; symmetric normalization for colorbar

## 8) How to reproduce
```zsh
# Rebuild gradients CSV
python alpha_gradient_dual/build_combined_gradient_summary.py

# Regenerate final cluster figures
python create_virgo_cluster_final_corrected.py

# Regenerate full science report (example)
python generate_full_report.py --galaxies VCC1146 VCC1588

# Make the index→α/Fe mapping figure (example)
python plot_alpha_fe_from_indices.py --galaxy VCC1146 --bin 1
```

If you want matched-range VNB fits (same radial extent as the 3-bin RDB) or additional galaxies added to the report and figures, we can extend the scripts with that option.