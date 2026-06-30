#!/bin/bash
# Spider-Sense v2 — Build single executable
# Usage: bash build_scripts/build_exe.sh

set -e

echo "🕸️  Building Spider-Sense v2 executable..."
echo "=============================================="
echo ""

# Step 1: Build frontend
echo "[1/3] Building React frontend..."
cd web-frontend
if [ ! -d "node_modules" ]; then
    echo "    Installing dependencies..."
    npm install
fi
echo "    Building..."
npx react-scripts build
cd ..
echo "    ✓ Frontend built"

# Step 2: Install Python dependencies
echo ""
echo "[2/3] Installing Python dependencies..."
pip install -r requirements.txt
pip install pyinstaller
echo "    ✓ Dependencies installed"

# Step 3: Build executable
echo ""
echo "[3/3] Building executable..."
pyinstaller --clean build_scripts/spider-sense.spec

echo ""
echo "=============================================="
echo "✅  Build complete!"
echo "    Output: dist/spider-sense.exe"
echo ""
echo "To run: dist/spider-sense.exe"
echo "To customize port: dist/spider-sense.exe --port 8080"
