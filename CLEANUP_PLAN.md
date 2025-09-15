# ISAPC rebuild and cleanup plan (non-destructive)

This plan inventories core pipeline code versus legacy/one-off scripts to enable a clean, reliable rebuild without changing scientific results. No files are moved or deleted yet; this document proposes what to keep and what to archive after your confirmation.

## Core to keep (v5.1.0 pipeline)
- main CLI and analysis modules
  - `main.py` (Version 5.1.0 CLI, error propagation)
  - `analysis/p2p.py`, `analysis/voronoi.py`, `analysis/radial.py`
  - `utils/` (`io.py`, `error_propagation.py`, `parallel.py`, `calc.py`)
  - `muse.py` (MUSECube)
- Configuration and models
  - `config.ini`, `config_manager.py`
  - `spectral_indices.py`, `stellar_population.py`, `galaxy_params.py`
  - `binning.py`, `visualization.py`, `physical_radius.py`
  - `templates/`, `TMB03/` (model data)
- Runners (to keep for convenience)
  - `run_complete_workflow.py`, `run_complete_pipeline.py`, `run_all_galaxies.py`, `run_all_galaxies_multithreaded.py`

## Candidates to archive (legacy/duplicates)
Safe to move into `legacy/` pending your confirmation. These appear to be superseded by the v5.1.0 pipeline and/or consolidated plotting.

- Older alpha/Fe analysis variants
  - `corrected_alpha_fe_analyzer.py`, `corrected_alpha_fe_analyzer_fixed.py`
  - `enhanced_alpha_fe_analyzer.py`, `final_corrected_alpha_fe_analyzer.py`
  - `enhanced_radial_plots_3bin_corrected.py`
  - `enhanced_plots_summary.py`, `enhanced_alpha_fe_analyzer.py`
- Plot/script generators replaced by standardized outputs
  - `create_final_alpha_fe_figures.py`, `create_galaxy_gradient_summary.py`
  - `create_isapc_workflow_latex_summary.py`
  - `create_virgo_cluster_*` (multiple variants)
  - Various `plot_*histogram.py`
- Older workflow wrappers or one-offs
  - `complete_virgo_alpha_fe_analysis*.py` (multiple)
  - `complete_physics_analysis.py`, `run_phy_visu_all_galaxies.py`, `run_p2p_only.sh`
  - `updated_complete_virgo_analysis.py`, `isapc_project_cleanup.py`, `final_organization.py`
- Debug/probing utilities (retain in legacy for reference)
  - `debug_alpha_fe_calculation.py`, `failure_analysis.py`, `quick_galaxy_summary.py`
  - `check_*`, `compare_gradient_methods.py`, `analyze_tmb03_velocity_dispersion.py`
- Empty or placeholder tests
  - `test_*.py` (currently mostly empty) → replace with a minimal synthetic-data test suite

If any of the above remain authoritative, we’ll keep them and annotate accordingly.

## Proposed actions (after your OK)
1. Create `legacy/` and move the “candidates to archive” there (git history preserved).
2. Add a minimal, fast test using synthetic spectra to validate core math paths (kinematics, indices, error maps) for regression safety.
3. Freeze the CLI contract and document inputs/outputs; tag as `v5.1.0`.
4. Optional: add CI to run the synthetic test and a style/syntax check on PRs.

## Safety and scientific parity
- This plan doesn’t touch numerical code yet.
- Any moves are reversible; we’ll re-run a smoke test post-archive.
- If you provide a known-good dataset + outputs, we’ll add a checksum-based regression check to guarantee parity.

---
Ownership note: Please mark any files in the archive list that you still use, and I’ll keep them in place.
