#!/usr/bin/env bash
# ==============================================================================
# run_cli.sh — Headless Command-Line Evaluation Runner
# Author: Ravikant | Institution: VIT Bhopal University
#
# This script executes the entire pipeline end-to-end in a pure terminal /
# non-GUI environment (suitable for headless servers, Docker containers, and CI).
# ==============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

echo "========================================================================"
echo "  Image Completion (Context Encoders) — Headless Evaluation Pipeline"
echo "  Author: Ravikant (VIT Bhopal University)"
echo "========================================================================"
echo ""

# 1. Environment Verification
echo "[Step 1/3] Verifying CLI Environment & Dependencies..."
python3 -c "
import sys
print(f'  Python Version: {sys.version.split()[0]}')
"

# 2. Architecture Inspection (Pure CLI)
echo ""
echo "[Step 2/3] Checking Model Architecture via Terminal CLI..."
python3 train.py --summary

# 3. Headless Benchmark & Evaluation
echo ""
echo "[Step 3/3] Running Headless Inpainting Benchmark..."
python3 demo.py --output demo_output.png

echo ""
echo "========================================================================"
echo "  [SUCCESS] All evaluation steps completed successfully via CLI!"
echo "  Output artifacts generated:"
echo "    - demo_output.png (Reconstruction & Error Heatmap)"
echo "========================================================================"
