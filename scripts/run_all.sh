#!/usr/bin/env bash
# =============================================================
#  HAD-FL  |  Run ALL Three Experiments Sequentially
#  Logs are saved to results/<dataset>/run_<timestamp>.log
# =============================================================
set -e

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT_DIR/src"

source "$ROOT_DIR/.venv/bin/activate" 2>/dev/null || true

run_experiment() {
    local NAME=$1
    local SCRIPT=$2
    local LOG_DIR="$ROOT_DIR/results/$NAME"
    mkdir -p "$LOG_DIR"
    local LOG="$LOG_DIR/run_${TIMESTAMP}.log"

    echo ""
    echo "======================================================"
    echo "  Starting: $NAME"
    echo "  Script  : $SCRIPT"
    echo "  Log     : $LOG"
    echo "======================================================"

    cd "$ROOT_DIR/src"
    python3 "$SCRIPT" 2>&1 | tee "$LOG"
    echo "[✓] $NAME finished. Log saved to $LOG"
}

run_experiment "mnist"   "$SRC/fl_mnist.py"
run_experiment "fmnist"  "$SRC/fl_fmnist.py"
run_experiment "cifar10" "$SRC/fl_cifar10.py"

echo ""
echo "======================================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Results saved in: $ROOT_DIR/results/"
echo "======================================================"
