#!/usr/bin/env bash
#
# Run omnirefactor tests on Mac (MPS), Threadripper (CUDA), and
# Threadripper with CUDA disabled (CPU-only), then combine coverage.
#
# Usage:  bash scripts/coverage_cross_device.sh
#
set -euo pipefail

MAC_ROOT="/Volumes/DataDrive/omnipose/omnirefactor"
REMOTE_ROOT="/home/kcutler/DataDrive/omnipose/omnirefactor"
REMOTE="kcutler@threadripper.local"
PYENV='export PATH="$HOME/.pyenv/shims:$HOME/.pyenv/bin:$PATH"'
COV_DIR="$MAC_ROOT/.coverage_combined"

rm -rf "$COV_DIR"
mkdir -p "$COV_DIR"

SRC="--source=src/omnirefactor"

# Notebook tests run on Mac only (need a Jupyter kernel).
SKIP_REMOTE="--ignore=tests/test_plot_notebook.py"

echo "=== Mac (MPS) ==="
cd "$MAC_ROOT"
python -m coverage run $SRC --data-file="$COV_DIR/.coverage.mac" -m pytest tests/ -q -k "not cli"

echo ""
echo "=== Threadripper (CUDA) ==="
ssh "$REMOTE" "$PYENV && cd $REMOTE_ROOT && python -m coverage run $SRC --data-file=/tmp/.coverage.omni_cuda -m pytest tests/ -q $SKIP_REMOTE -k 'not cli'"
scp "$REMOTE":/tmp/.coverage.omni_cuda "$COV_DIR/.coverage.cuda"

echo ""
echo "=== Threadripper (CPU-only) ==="
ssh "$REMOTE" "$PYENV && cd $REMOTE_ROOT && CUDA_VISIBLE_DEVICES='' python -m coverage run $SRC --data-file=/tmp/.coverage.omni_cpu -m pytest tests/ -q $SKIP_REMOTE -k 'not cli'"
scp "$REMOTE":/tmp/.coverage.omni_cpu "$COV_DIR/.coverage.cpu"

echo ""
echo "=== Combining coverage ==="
cd "$MAC_ROOT"
# Combine all: device runs + notebook kernel coverage files
python -m coverage combine --data-file="$COV_DIR/.coverage" "$COV_DIR"/.coverage.*
python -m coverage report --data-file="$COV_DIR/.coverage" --show-missing
echo ""
python -m coverage html --data-file="$COV_DIR/.coverage" -d "$COV_DIR/htmlcov"
echo "HTML report: file://$COV_DIR/htmlcov/index.html"
