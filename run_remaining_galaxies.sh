#!/bin/bash

# Simple command-based approach to run ISAPC for all remaining galaxies
# Using the exact same commands that worked for VCC1588 and VCC0308

echo "=== Starting ISAPC Analysis for Remaining Galaxies ==="
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

# Process each galaxy using the exact same command that worked before
for galaxy in "${galaxies[@]}"; do
    echo ""
    echo "=== Processing $galaxy ==="
    echo "Started at: $(date)"
    
    # Run the exact same command format that worked for VCC1588
    # Using direct python execution (no conda) with proper redshift values
    python main.py "data/MUSE/${galaxy}_stack.fits" \
        -z 0.003 \
        -t "data/templates/spectra_emiles_9.0.npz" \
        -o output \
        -m ALL \
        --n-jobs 1 \
        --auto-reuse
    
    if [ $? -eq 0 ]; then
        echo "✅ $galaxy completed successfully at $(date)"
    else
        echo "❌ $galaxy failed at $(date)"
    fi
    
    echo "--- $galaxy processing finished ---"
done

echo ""
echo "=== All galaxies processing completed ==="
echo "Finished at: $(date)"
echo "Check individual results in output/ and galaxy_summary/ directories"
