Redshift Consistency Check
==========================

Source of truth: `galaxy_catalog.py` (REDSHIFTS, TYPES)

Verified sample vs. code mappings:

- VCC1588: image table z=0.0042 → catalog 0.0042 → OK
- VCC1146: image table z=0.0023 → catalog 0.0023 → OK

Notes
- FITS headers in `data/IFU/*_stack.fits` do not embed a REDSHIFT card; the pipeline passes z explicitly via CLI and stores it in `muse.py` state and output headers.
- Runners `run_all_galaxies*.py` and physics scripts are updated to import the centralized catalog to prevent divergence.

How to extend
- Edit `galaxy_catalog.py` to add/update values; all scripts will pick up changes automatically.
