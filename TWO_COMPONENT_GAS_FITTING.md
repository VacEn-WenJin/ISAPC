# Two-Component Gas Fitting (pPXF)

This repository supports fitting one or two gas kinematic components in both pixel-by-pixel and binned modes using pPXF. The feature is fully config-driven and backward-compatible with the prior single-component default.

## How to enable

Edit `config.ini` and set under `[GasKinematics]`:

- `components = 2` to enable two gas components
- `mode = narrow_broad` or `duplicate` (informational only for now)
- Set initial sigmas and bounds if desired

Example:

```
[GasKinematics]
components = 2
mode = narrow_broad
narrow_sigma_init = 40.0
broad_sigma_init = 120.0
narrow_sigma_bounds = 10.0, 150.0
broad_sigma_bounds = 60.0, 300.0
velocity_window = 300.0
```

If omitted, the default is `components = 1` (original behavior).

## What it does

- Builds the gas template matrix with 1 or 2 kinematic components by duplicating all emission-line templates per component.
- Calls pPXF with component mapping:
	- 0: stellar, 1: gas comp1, 2: gas comp2 (if enabled)
- Initial guesses and bounds are set from the config. Velocity bounds are ±`velocity_window` km/s around the initial velocity of that spaxel/bin.
- Fluxes are aggregated per base line name by summing across gas components.
- Gas velocity/dispersion stored per line is taken from the component with the higher flux for that line in that spaxel/bin.

## Outputs and compatibility

- Existing result dictionaries remain intact: `emission_flux`, `emission_vel`, `emission_sig`, `gas_bestfit_field`, etc.
- When two components are enabled, the per-line kinematics pick the dominant component by flux. This keeps downstream maps and plots working without change.
- Internally, `muse.py` also exposes `gas_sol_list` during fitting for advanced debugging, but this is not required by other modules.

## Notes

- Two-component fitting increases runtime. Start with 1–2 test galaxies to validate results and performance.
- Consider constraining broad component sigmas with realistic bounds to avoid degeneracies.
- You can still restrict or expand the list of fitted emission lines via the `[EmissionLines]` section.

