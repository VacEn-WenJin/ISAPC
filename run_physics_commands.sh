#!/bin/bash
# Simple commands to run complete physics analysis
# Run these commands in a screen session

echo "=== Complete Physics Analysis for All Virgo Galaxies ==="
echo "Date: $(date)"
echo "Directory: $(pwd)"
echo ""

# Check if we're in the right directory
if [[ ! -f "run_complete_physics_analysis.py" ]]; then
    echo "Error: Not in ISAPC directory!"
    echo "Run: cd /home/siqi/WkpSpace/ISAPC_Jul/ISAPC"
    exit 1
fi

# Check available galaxies
echo "Checking available galaxies..."
ls -la output/*/Data/*_P2P_results.npz | wc -l | awk '{print "Found " $1 " galaxies with P2P results"}'
echo ""

# Run the analysis
echo "Starting complete physics analysis..."
echo "This will process all available galaxies with:"
echo "  - Alpha abundance gradient calculations"
echo "  - P2P velocity field analysis (corrected velocities)"
echo "  - Error propagation throughout"
echo "  - Statistical summaries and CSV output"
echo ""
echo "Results will be saved to: physics_analysis_results/"
echo "Logs will be saved to: logs/"
echo ""

# Start the analysis
python run_complete_physics_analysis.py

echo ""
echo "=== Analysis Complete ==="
echo "Check the physics_analysis_results/ directory for:"
echo "  - complete_physics_analysis_YYYYMMDD_HHMMSS.csv"
echo "  - analysis_summary_YYYYMMDD_HHMMSS.txt"
echo ""
echo "Check logs/ directory for detailed processing logs"
