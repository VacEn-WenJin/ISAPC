#!/bin/bash
# Screen commands for running complete ISAPC + Physics workflow
# 
# This will run the complete workflow for all 20 MUSE galaxies
# 1. Run ISAPC analysis for any incomplete galaxies
# 2. Run physics visualization analysis on all results

echo "Starting Complete ISAPC + Physics Workflow in Screen"
echo "This will process all 20 MUSE galaxies"

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "Error: main.py not found! Please run from ISAPC directory"
    exit 1
fi

# Create a new screen session for the complete workflow
screen -dmS isapc_complete_workflow

# Run the complete workflow
screen -S isapc_complete_workflow -X stuff "python run_all_galaxies.py$(printf \\r)"

echo ""
echo "Complete workflow started in screen session 'isapc_complete_workflow'"
echo ""
echo "To monitor progress:"
echo "  screen -r isapc_complete_workflow"
echo ""
echo "To detach from screen (without stopping):"
echo "  Ctrl+A, then D"
echo ""
echo "To check if it's still running:"
echo "  screen -ls"
echo ""
echo "Log files will be created in ./logs/ directory"
echo ""
echo "Expected workflow:"
echo "  1. Check existing ISAPC completions"
echo "  2. Run ISAPC for any incomplete galaxies (may take hours)"
echo "  3. Run physics visualization analysis"
echo "  4. Generate final summary"
