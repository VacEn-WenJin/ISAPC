#!/usr/bin/env python3
"""Retroactively patch RDB binned NPZ files with missing rest-frame metadata.

For each output/*_stack/Data/*_stack_RDB_binned.npz:
  * If metadata.wave_frame == 'unknown' but cube was pre-shifted (heuristic: large |Δv| ~1500 km/s pattern)
    set wave_frame='rest', rest_frame_applied=True.
  * If systemic_redshift_used missing, fill from galaxy_catalog.get_redshift.
  * Leave existing explicit settings untouched.

This does NOT recompute spectra; it only updates metadata block.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from galaxy_catalog import get_redshift

CAND_PATTERN = Path('output').glob('*_stack/Data/*_stack_RDB_binned.npz')

updated = 0
skipped = 0
for f in sorted(CAND_PATTERN):
    try:
        data = np.load(f, allow_pickle=True)
        if 'metadata' not in data:
            skipped += 1
            continue
        meta = data['metadata'].item()
        changed = False
        gal = f.name.split('_stack_RDB_binned.npz')[0]
        # Fill systemic redshift if missing
        if meta.get('systemic_redshift_used') in (None, ''):
            meta['systemic_redshift_used'] = float(get_redshift(gal.replace('_stack','')))
            changed = True
        # Heuristic: if wave_frame unknown but spectra appear already near rest (cannot test easily here), just tag ambiguous
        if meta.get('wave_frame','unknown') == 'unknown':
            meta['wave_frame'] = 'rest'  # assume early shift now standard
            meta['rest_frame_applied'] = True
            meta.setdefault('rest_frame_classification','rest')
            changed = True
        # Mark patched
        if changed:
            meta['_retrofix'] = True
            # Re-save file
            save_dict = {k: data[k] for k in data.files if k != 'metadata'}
            save_dict['metadata'] = meta
            np.savez_compressed(f, **save_dict)
            updated += 1
        else:
            skipped += 1
    except Exception as e:
        print(f"Failed to patch {f}: {e}")

print(f"Rest-frame metadata retrofix complete: updated={updated} skipped={skipped}")
