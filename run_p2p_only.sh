#!/bin/bash

# Simple P2P-only analysis for all remaining galaxies
# This avoids the VNB and RDB binning issues that cause crashes

echo "=== Starting P2P Analysis for Remaining Galaxies ==="
echo "Started at: $(date)"

# List of remaining galaxies (excluding VCC1588 and VCC0308 which are already done)
galaxies=(
    "VCC0667"
    "VCC0688" 
    "VCC0990"
    "VCC1049"
    "VCC1146"
    "VCC1193"
    "VCC1368"
    "VCC1410"
    "VCC1431"
    "VCC1486"
    "VCC1499"
    "VCC1549"
    "VCC1695"
    "VCC1811"
    "VCC1890"
    "VCC1902"
    "VCC1910"
    "VCC1949"
)

echo "Total galaxies to process: ${#galaxies[@]}"
echo "Mode: P2P only (avoiding problematic binning)"

# Process each galaxy using simple P2P mode
for galaxy in "${galaxies[@]}"; do
    echo ""
    echo "=== Processing $galaxy ==="
    echo "Started at: $(date)"
    
    # Run P2P analysis only - this is what actually works reliably
    python main.py "data/MUSE/${galaxy}_stack.fits" \
        -z 0.003 \
        -t "data/templates/spectra_emiles_9.0.npz" \
        -o output \
        -m P2P \
        --n-jobs 1 \
        --auto-reuse
    
    if [ $? -eq 0 ]; then
        echo "✅ $galaxy P2P analysis completed successfully at $(date)"
    else
        echo "❌ $galaxy P2P analysis failed at $(date)"
    fi
    
    echo "--- $galaxy processing finished ---"
done

echo ""
echo "=== All P2P analyses completed ==="
echo "Finished at: $(date)"
echo "Check individual results in output/ directories"
echo "Note: Only P2P analysis completed. VNB and RDB can be run separately if needed."
