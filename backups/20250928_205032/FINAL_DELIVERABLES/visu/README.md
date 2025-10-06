# VCC1588 — Visu Mini Panels

This folder contains compact panels used in the short, science-only demonstration:

- `VCC1588_visu_norm.png` — Example post-fit normalized spectrum (emission-reduced) with index windows.
- `VCC1588_visu_index_grid.png` — Mgb vs Fe5015 index points for VCC1588 overlaid on TMB03 model points; labeled by bin order.
- `VCC1588_visu_radial_profile.png` — AIP-derived α/Fe radial profile with error bars and linear fit.

Note: The RDB vs VNB compact panel is omitted here for VCC1588 due to missing per-bin α/Fe extraction in `*_RDB_results.npz` under current outputs. Once available, re-run:

```zsh
python make_visu_vcc1588.py
```
