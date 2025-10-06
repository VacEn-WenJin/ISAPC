# ISAPC Short Workflow Report

Date: 2025-09-18

This report summarizes the end-to-end ISAPC pipeline used to derive α/Fe radial gradients for Virgo Cluster galaxies and to visualize their spatial and kinematic context.

## Overview
- Inputs: IFU datacubes and metadata, TMB03 index models, galaxy list and emission flags, and configuration in `config.ini`.
- Core Outputs:
   - Per-galaxy radial [α/Fe] profiles and gradient fits with errors.
   - Combined gradient summary CSV (`alpha_gradient_dual/combined_gradient_summary.csv`).
   - Cluster map with gradient direction (triangles), magnitude (arrows), emission fill, and velocity coloring (this report links it below).

## Pipeline Steps and Methods
1) Data ingestion and coordinates
   - Read IFU FITS headers to obtain RA/DEC; fallback positions if headers are missing (`create_virgo_cluster_map_with_vectors.py`).
   - Central galaxy velocity used for color mapping is from our velocity dict; Δv is relative to the sample mean (shown on the colorbar).

2) Binning (radial and/or Voronoi)
   - We support S/N-aware Voronoi binning and elliptical radial binning by R/Re.
   - Target S/N and ring counts are configured (see `main.py` CLI flags `--target-snr`, `--n-rings`).
   - Edge cases handled: low-S/N spaxels, masked lines, minimum spaxel counts per bin.

3) Fitting (ppxf: stellar continuum + emission)
   - For each bin, we fit the stellar continuum and gas emission simultaneously using pPXF.
   - We produce two spectra for QA and downstream indices:
     - Original (observed) binned spectrum.
     - Normalized, emission-reduced spectrum (continuum-normalized; gas lines subtracted) used to stabilize index measurement.
   - These QA plots can be generated via the MUSE class plotting routines already wired in the codebase.

4) Indices and [α/Fe] inference (TMB03 framework)
   - We measure Lick-style absorption indices (e.g., Mg b, Fe5270, Fe5335) with error propagation.
   - Composite iron index: $\langle Fe \rangle = (Fe5270 + Fe5335)/2$.
   - From TMB03 model grids, we infer [α/Fe] by interpolating indices at the best-fit age/metallicity or via grid inversion.
   - Typical working relation (schematic):
     $$
     [\alpha/Fe] \;=\; f\big(\text{Mg}b, \langle Fe \rangle, \text{age}, Z\big)\,.
     $$

5) Radial profile and gradient fitting
   - Define galactocentric radius in units of $R/ R_e$ per bin; build radial [α/Fe] profile.
   - Fit a robust linear relation with uncertainties:
     $$
     [\alpha/Fe](R) \;=\; a + b\,\left(\frac{R}{R_e}\right),\quad \text{with}\;\sigma_b\;\text{from weighted least squares}
     $$
   - Methods: 3-bin RDB baseline (inner/mid/outer) and extended fits for cross-checks. We report the RDB slope (b) and its error.

6) Visualization
   - Per-galaxy gradient plot with error bars and best-fit line.
   - Virgo cluster map:
     - Triangle orientation encodes sign of slope (up: positive, down: negative).
     - Arrow length encodes |slope|, anchored at triangle center.
     - Marker fill indicates emission presence.
     - Color encodes relative velocity Δv using `cool` colormap; Δv=0 reference is annotated.
     - Substructure circles filled by mean Δv of enclosed galaxies; majors (M87/M86/M60/M49) colored by velocity.

## Figure
The main cluster map is saved as:
- `FINAL_DELIVERABLES/virgo_cluster_map_with_vectors.png`

![Virgo Cluster Map](./virgo_cluster_map_with_vectors.png)

## Example Per-Galaxy Outputs
- Normalized vs. original spectrum (post-fit QA) per bin are available via the existing plotting in the `MUSECube` class and analysis routines.
- Gradient plots with errors are generated in our gradient analysis stage and included in the per-galaxy figures directory (see analysis scripts).

Quick examples to (re)generate key figures:

Note: Unless otherwise noted, run these commands from the repository root (`ISAPC_Aug/ISAPC`).

```zsh
# Rebuild α/Fe gradients and summary (if needed)
python alpha_gradient_analysis.py

# Regenerate the cluster figure
python create_virgo_cluster_map_with_vectors.py

# Optional: rerun end-to-end for selected galaxies (e.g., VCC1588, VCC1146)
python run_complete_workflow.py --galaxies VCC1588,VCC1146 --threads 4
```

### Re-generate VCC1588 and VCC1146 deliverables

Normalized spectra after fitting (P2P) and AIP α/Fe products can be regenerated with:

```zsh
# 1) Run ISAPC (P2P only) to produce post-fit normalized spectra panels
#    Outputs go to: output/<GAL>_stack/Plots/P2P/
#    Example panel files: <GAL>_P2P_spectrum_norm_0.png, ...
python main.py data/MUSE/VCC1588_stack.fits \
   -z $(python -c 'import galaxy_catalog as g; print(g.get_redshift("VCC1588"))') \
   -t data/templates/spectra_emiles_9.0.npz \
   -o output -m P2P --n-jobs 4 --auto-reuse

python main.py data/MUSE/VCC1146_stack.fits \
   -z $(python -c 'import galaxy_catalog as g; print(g.get_redshift("VCC1146"))') \
   -t data/templates/spectra_emiles_9.0.npz \
   -o output -m P2P --n-jobs 4 --auto-reuse

# 2) Run AIP α/Fe and radial profile for each galaxy
#    Outputs go to: output/<GAL>_stack/Plots/
python run_aip_alpha_fe.py --galaxy VCC1588
python run_aip_alpha_fe.py --galaxy VCC1146

# 3) Optionally re-run gradient aggregation and cluster map
python alpha_gradient_analysis.py
python create_virgo_cluster_map_with_vectors.py
```

Expected output locations:
- `output/VCC1588_stack/Plots/P2P/`: `VCC1588_P2P_spectrum_norm_*.png` (emission-reduced normalized spectra)
- `output/VCC1146_stack/Plots/P2P/`: `VCC1146_P2P_spectrum_norm_*.png`
- `output/<GAL>_stack/Plots/`: `<GAL>_AIP_alpha_fe_map.png`, `<GAL>_AIP_alpha_fe_radial_profile.png`
- Cluster maps and tables under `FINAL_DELIVERABLES/`

## Key Files
- `alpha_gradient_dual/combined_gradient_summary.csv` — consolidated α/Fe gradients.
- `create_virgo_cluster_map_with_vectors.py` — original-style cluster visualization, velocity-coded.
- `alpha_gradient_analysis.py` — gradient extraction and aggregation.
- `ISAPC_Galaxy.py` — optional emission flags and galaxy catalog.

## Reproducibility
Run the main cluster figure generation:
```zsh
python create_virgo_cluster_map_with_vectors.py
```
Output location: `FINAL_DELIVERABLES/`.

## Notes and Next Steps
- Δv=0 currently uses the sample mean velocity; we can switch to a fixed systemic for Virgo A or per-subcluster means.
- Panels figure (`*_panels.png`) includes distance–slope trends for quick context.
- If required, we can export vector formats (`.pdf`, `.svg`) and package a minimal data/README bundle.

### References
- See `FINAL_DELIVERABLES/LITERATURE_FACTS_AUDIT.md` for a fact-by-fact mapping to papers with verbatim quotations (Voronoi binning, pPXF, TMB03, MUSE).
