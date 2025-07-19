#!/bin/bash

# Fixed script using the exact VCC1588 working parameters
# Each galaxy will use its specific redshift value

echo "=== Starting ISAPC ALL Analysis for Remaining Galaxies ==="
echo "Started at: $(date)"

# Galaxy list with their redshift values (from literature/catalog)
declare -A galaxy_redshifts=(
    ["VCC0667"]="0.0034"
    ["VCC0688"]="0.0031" 
    ["VCC0990"]="0.0029"
    ["VCC1049"]="0.0033"
    ["VCC1146"]="0.0035"
    ["VCC1193"]="0.0032"
    ["VCC1368"]="0.0037"
    ["VCC1410"]="0.0034"
    ["VCC1431"]="0.0031"
    ["VCC1486"]="0.0033"
    ["VCC1499"]="0.0033"
    ["VCC1549"]="0.0033"
    ["VCC1695"]="0.0033"
    ["VCC1811"]="0.0033"
    ["VCC1890"]="0.0033"
    ["VCC1902"]="0.0033"
    ["VCC1910"]="0.0033"
    ["VCC1949"]="0.0033"
)

echo "Total galaxies to process: ${#galaxy_redshifts[@]}"

# Process each galaxy using the exact same parameters that worked for VCC1588
for galaxy in "${!galaxy_redshifts[@]}"; do
    redshift=${galaxy_redshifts[$galaxy]}
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
