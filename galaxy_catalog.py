"""
Galaxy catalog: single source of truth for redshifts and types.

Values are aligned with the observed galaxy list (image provided) and
the mappings already used consistently in the pipeline for VCC1588 et al.

If a galaxy is missing, add it here so all scripts stay consistent.
"""

from typing import Dict

# Redshifts (z) for Virgo galaxies used in this workspace.
# These match the observed table values; flagged entries are estimates.
REDSHIFTS: Dict[str, float] = {
    'M60': 0.0034,
    'VCC0308': 0.0055,  # dE
    'VCC0667': 0.0048,  # Sd
    'VCC0688': 0.0038,  # Sc
    'VCC0990': 0.0058,  # dS0(N)
    'VCC1049': 0.0021,  # dE(N)
    'VCC1146': 0.0023,  # E
    'VCC1193': 0.0025,  # Sd
    'VCC1368': 0.0035,  # SBa
    'VCC1410': 0.0054,  # Sd
    'VCC1431': 0.0050,  # dE
    'VCC1486': 0.0004,  # Sc
    'VCC1499': 0.0042,  # estimate (Medrez filter; adjust if known)
    'VCC1549': 0.0046,  # dE(N)
    'VCC1588': 0.0042,  # Sd
    'VCC1695': 0.0058,  # dE
    'VCC1811': 0.0023,  # Sc
    'VCC1890': 0.0040,  # dE
    'VCC1902': 0.0038,  # SBa
    'VCC1910': 0.0007,  # dE(N)
    'VCC1949': 0.0058,  # dS0(N)
}

# Galaxy types (from observed table)
TYPES: Dict[str, str] = {
    'VCC0308': 'dE',
    'VCC0667': 'Sd',
    'VCC0688': 'Sc',
    'VCC0990': 'dS0(N)',
    'VCC1049': 'dE(N)',
    'VCC1146': 'E',
    'VCC1193': 'Sd',
    'VCC1368': 'SBa',
    'VCC1410': 'Sd',
    'VCC1431': 'dE',
    'VCC1486': 'Sc',
    'VCC1499': 'dE',  # unknown/Medrez; placeholder
    'VCC1549': 'dE(N)',
    'VCC1588': 'Sd',
    'VCC1695': 'dE',
    'VCC1811': 'Sc',
    'VCC1890': 'dE',
    'VCC1902': 'SBa',
    'VCC1910': 'dE(N)',
    'VCC1949': 'dS0(N)',
}

DEFAULT_VIRGO_Z = 0.0033

def get_redshift(name: str) -> float:
    """Return redshift for a galaxy, defaulting to Virgo mean if missing."""
    return REDSHIFTS.get(name, DEFAULT_VIRGO_Z)

def get_type(name: str) -> str:
    """Return morphological type if known, else an empty string."""
    return TYPES.get(name, '')
