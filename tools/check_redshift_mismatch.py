#!/usr/bin/env python3
"""
Scan scripts that might still embed redshift mappings and compare them
against galaxy_catalog.REDSHIFTS, reporting any mismatches.
"""
import re
import sys
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from galaxy_catalog import REDSHIFTS


CANDIDATE_FILES = [
    ROOT / 'run_complete_pipeline.py',
    ROOT / 'run_all_galaxies.py',
    ROOT / 'run_all_galaxies_multithreaded.py',
    ROOT / 'run_phy_visu_all_galaxies.py',
    ROOT / 'alpha_gradient_analysis.py',
]

pattern = re.compile(r"'VCC\d{3,4}'\s*:\s*([0-9]*\.[0-9]+)")

mismatches = []
for f in CANDIDATE_FILES:
    if not f.exists():
        continue
    txt = f.read_text()
    for m in re.finditer(r"'VCC(\d{3,4})'\s*:\s*([0-9]*\.[0-9]+)", txt):
        g = f"VCC{m.group(1)}"
        z = float(m.group(2))
        z0 = REDSHIFTS.get(g)
        if z0 is not None and abs(z - z0) > 1e-6:
            mismatches.append((str(f.relative_to(ROOT)), g, z, z0))

if mismatches:
    print('Found redshift mismatches:')
    for file, g, z, z0 in mismatches:
        print(f"  {file}: {g} -> {z} (catalog {z0})")
    sys.exit(1)
else:
    print('No mismatches found. All redshifts align with galaxy_catalog.py')
    sys.exit(0)
