#!/usr/bin/env bash
# =============================================================
#  HAD-FL  |  Environment Setup Script
#  Run once before your first experiment
# =============================================================
set -e

echo "=============================================="
echo "  HAD-FL Environment Setup"
echo "=============================================="

# ── 1. Check Python version ───────────────────────────────────
PYTHON=$(command -v python3 || command -v python)
PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
echo "[1/5] Python detected: $PY_VERSION"

MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 9 ]; }; then
    echo "ERROR: Python 3.9+ is required. Found $PY_VERSION"
    exit 1
fi

# ── 2. Create virtual environment ─────────────────────────────
echo "[2/5] Creating virtual environment (.venv)..."
$PYTHON -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet

# ── 3. Detect CUDA & install PyTorch ─────────────────────────
echo "[3/5] Detecting GPU / CUDA..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
    echo "    GPU found. CUDA $CUDA_VER detected."
    CUDA_TAG=$(echo "$CUDA_VER" | awk -F. '{printf "cu%d%d", $1, $2}')
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" --quiet
else
    echo "    No GPU found. Installing CPU-only PyTorch."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
fi

# ── 4. Install remaining dependencies ────────────────────────
echo "[4/5] Installing Python dependencies..."
pip install -r requirements.txt --quiet

# ── 5. Create result directories ─────────────────────────────
echo "[5/5] Creating results directories..."
mkdir -p results/cifar10 results/mnist results/fmnist

echo ""
echo "=============================================="
echo "  Setup complete!"
echo "  Activate with:  source .venv/bin/activate"
echo "=============================================="
