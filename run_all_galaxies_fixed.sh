#!/bin/bash

# Fixed script using the exact VCC1588 working parameters
# Each galaxy will use its specific redshift value

echo "=== Starting ISAPC ALL Analysis for Remaining Galaxies ==="
echo "Started at: $(date)"

# Galaxy list (redshift will be looked up from galaxy_catalog.py)
galaxies=(
    "VCC0667" "VCC0688" "VCC0990" "VCC1049" "VCC1146" "VCC1193" "VCC1368"
    "VCC1410" "VCC1431" "VCC1486" "VCC1499" "VCC1549" "VCC1695" "VCC1811"
    "VCC1890" "VCC1902" "VCC1910" "VCC1949"
)

echo "Total galaxies to process: ${#galaxies[@]}"

# Process each galaxy using the exact same parameters that worked for VCC1588
for galaxy in "${galaxies[@]}"; do
    redshift=$(python - <<PY
from galaxy_catalog import get_redshift
print(get_redshift("$galaxy"))
PY
)
    echo ""
    echo "=== Processing $galaxy (z=$redshift) ==="
    echo "Started at: $(date)"
    
    # Use the exact same command that worked for VCC1588
    python main.py "data/MUSE/${galaxy}_stack.fits" \
        -z "$redshift" \
        -t "data/templates/spectra_emiles_9.0.npz" \
        -o output \
        -m ALL \
        --target-snr 20.0 \
        --min-snr 1.0 \
        --n-rings 6 \
        --vel-init 0.0 \
        --sigma-init 50.0 \
        --poly-degree 3 \
        --n-jobs 1 \
        --save-error-maps \
        --auto-reuse \
        --cvt \
        --physical-radius
    
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
echo "Check individual results in output/ directories"
