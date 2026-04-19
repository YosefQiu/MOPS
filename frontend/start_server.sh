#!/bin/bash
# Quick start script for MOPS Frontend Backend

echo "=========================================="
echo "MOPS Frontend Backend Server"
echo "=========================================="

# Change to script directory
cd "$(dirname "$0")"

# Check if env.sh exists in parent directory
if [ -f ../env.sh ]; then
    echo "[Setup] Loading environment variables from ../env.sh"
    source ../env.sh
else
    echo "[Warning] ../env.sh not found. Azure API credentials may not be configured."
fi

# Check Python version
PYTHON_CMD=$(which python3 || which python)
if [ -z "$PYTHON_CMD" ]; then
    echo "[Error] Python not found. Please install Python 3.6+"
    exit 1
fi

echo "[Setup] Python: $PYTHON_CMD"
$PYTHON_CMD --version

# Check if Flask is installed
if ! $PYTHON_CMD -c "import flask" 2>/dev/null; then
    echo "[Setup] Flask not found. Installing dependencies..."
    $PYTHON_CMD -m pip install -r requirements.txt
fi

# Create output directory if it doesn't exist
mkdir -p ../Agent/outputs/remapping
mkdir -p ../Agent/outputs/streamline
mkdir -p ../Agent/outputs/pathline

echo "[Setup] Output directories ready"
echo ""
echo "=========================================="
echo "Starting Flask server on http://localhost:5000"
echo "=========================================="
echo ""
echo "Open index.html in your browser to use the frontend."
echo "Press Ctrl+C to stop the server."
echo ""

# Start the server
$PYTHON_CMD backend_server.py
