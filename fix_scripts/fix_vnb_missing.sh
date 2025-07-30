#!/bin/bash
# Script to run missing VNB analyses
# Generated automatically by failure analysis

# Galaxy redshifts
declare -A galaxy_redshifts=(
    ["VCC0667"]=0.00435
    ["VCC0688"]=0.00353
    ["VCC1049"]=0.00332
    ["VCC1193"]=0.00341
    ["VCC1368"]=0.00324
    ["VCC1410"]=0.00342
    ["VCC1486"]=0.00345
    ["VCC1695"]=0.00356
    ["VCC1890"]=0.00338
    ["VCC1949"]=0.00345
)

echo "Starting VNB analysis for missing galaxies"
echo "Total galaxies to process: ${#galaxy_redshifts[@]}"

# Process each galaxy
for galaxy in "${!galaxy_redshifts[@]}"; do
    redshift=${galaxy_redshifts[$galaxy]}
    echo ""
    echo "=== Processing $galaxy VNB analysis (z=$redshift) ==="
    echo "Started at: $(date)"
    
    python main.py "data/MUSE/${galaxy}_stack.fits" \
        -z "$redshift" \
        -t "data/templates/spectra_emiles_9.0.npz" \
        -o output \
        -m VNB \
        --target-snr 5.0 \
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
        echo "✅ $galaxy VNB completed successfully at $(date)"
    else
        echo "❌ $galaxy VNB failed at $(date)"
    fi
done

echo ""
echo "VNB analysis complete for all missing galaxies"
