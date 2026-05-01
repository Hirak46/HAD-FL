#!/usr/bin/env bash
# =============================================================
#  HAD-FL  |  Run a Single Dataset Experiment
#  Usage:  bash scripts/run_single.sh [mnist|fmnist|cifar10]
# =============================================================
set -e

DATASET="${1:-mnist}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

source "$ROOT_DIR/.venv/bin/activate" 2>/dev/null || true

case "$DATASET" in
    mnist)   SCRIPT="fl_mnist.py"   ;;
    fmnist)  SCRIPT="fl_fmnist.py"  ;;
    cifar10) SCRIPT="fl_cifar10.py" ;;
    *)
        echo "Usage: bash scripts/run_single.sh [mnist|fmnist|cifar10]"
        exit 1
        ;;
esac

LOG_DIR="$ROOT_DIR/results/$DATASET"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run_${TIMESTAMP}.log"

echo "======================================================"
echo "  HAD-FL  |  Running: $DATASET"
echo "  Log: $LOG"
echo "======================================================"

cd "$ROOT_DIR/src"
python3 "$SCRIPT" 2>&1 | tee "$LOG"

echo ""
echo "[✓] Done! Results saved in: $LOG_DIR"
