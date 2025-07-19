#!/bin/bash
# Screen script for running complete physics analysis on all Virgo galaxies
# Usage: bash run_physics_screen.sh

# Set script directory
SCRIPT_DIR="/home/siqi/WkpSpace/ISAPC_Jul/ISAPC"
cd "$SCRIPT_DIR" || exit 1

# Screen session name
SESSION_NAME="physics_analysis"

# Check if screen session already exists
if screen -list | grep -q "$SESSION_NAME"; then
    echo "Screen session '$SESSION_NAME' already exists!"
    echo "To attach: screen -r $SESSION_NAME"
    echo "To kill existing: screen -S $SESSION_NAME -X quit"
    exit 1
fi

# Create new screen session and run analysis
echo "Starting complete physics analysis in screen session: $SESSION_NAME"
echo "Directory: $SCRIPT_DIR"
echo ""
echo "Commands that will run:"
echo "1. cd $SCRIPT_DIR"
echo "2. python run_complete_physics_analysis.py"
echo ""
echo "To monitor progress:"
echo "  screen -r $SESSION_NAME     # Attach to session"
echo "  Ctrl+A, D                   # Detach from session"
echo "  screen -list                # List sessions"
echo ""

# Start screen session with physics analysis
screen -dmS "$SESSION_NAME" bash -c "
cd '$SCRIPT_DIR' || exit 1;
echo 'Starting complete physics analysis at: \$(date)';
echo 'Working directory: \$(pwd)';
echo 'Python version: \$(python --version)';
echo '';
echo 'Running: python run_complete_physics_analysis.py';
echo '================================================================';
python run_complete_physics_analysis.py;
echo '';
echo '================================================================';
echo 'Physics analysis completed at: \$(date)';
echo 'Check logs/ and physics_analysis_results/ directories for output';
echo '';
echo 'Press any key to exit screen session...';
read -n 1;
"

echo "Screen session '$SESSION_NAME' started successfully!"
echo ""
echo "To attach to the session and monitor progress:"
echo "  screen -r $SESSION_NAME"
echo ""
echo "To check if it's still running:"
echo "  screen -list"
