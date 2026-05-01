# HAD-FL — Hierarchical Anomaly Detection for Federated Learning

A Byzantine-robust federated learning framework that defends against
poisoning attacks using hierarchical anomaly detection, HDBSCAN clustering,
and adaptive temporal reputation scoring.

---

## Project Structure

```
HAD-FL/
├── src/                          # Core experiment scripts
│   ├── fl_mnist.py               # FL experiment on MNIST
│   ├── fl_fmnist.py              # FL experiment on Fashion-MNIST
│   └── fl_cifar10.py             # FL experiment on CIFAR-10
│
├── configs/                      # Hyperparameter configuration files
│   ├── mnist_config.yaml
│   ├── fmnist_config.yaml
│   └── cifar10_config.yaml
│
├── scripts/                      # Shell helpers & utility scripts
│   ├── setup.sh                  # One-time environment setup
│   ├── run_all.sh                # Run all three experiments
│   ├── run_single.sh             # Run one dataset experiment
│   └── check_env.py              # Verify dependencies before running
│
├── results/                      # Output CSVs & logs (auto-created)
│   ├── mnist/
│   ├── fmnist/
│   └── cifar10/
│
├── notebooks/
│   └── analysis_template.py      # Starter script to plot results
│
├── tests/
│   └── test_imports.py           # Dependency smoke tests
│
├── docs/
│   ├── architecture.md           # System design notes
│   └── results_guide.md          # How to read output CSV files
│
├── requirements.txt              # Python package list
├── .gitignore
└── LICENSE
```

---

## Requirements

| Requirement | Minimum Version |
|-------------|----------------|
| Python | 3.9+ |
| PyTorch | 2.0+ |
| CUDA (optional) | 11.8+ (for GPU acceleration) |
| RAM | 8 GB minimum, 16 GB recommended |
| Disk | ~3 GB (datasets + results) |

---

## Quick Start

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/HAD-FL.git
cd HAD-FL
```

### Step 2 — Run environment setup

```bash
bash scripts/setup.sh
```

This will:
- Create a Python virtual environment (`.venv/`)
- Auto-detect your GPU / CUDA version and install the correct PyTorch build
- Install all dependencies from `requirements.txt`
- Create the `results/` directory tree

### Step 3 — Activate the virtual environment

```bash
source .venv/bin/activate
```

### Step 4 — Verify the environment

```bash
python3 scripts/check_env.py
```

Expected output:
```
=======================================================
  HAD-FL  |  Environment Check
=======================================================
  [OK]  PyTorch            2.x.x
  [OK]  TorchVision        0.x.x
  ...
  [GPU] NVIDIA GeForce RTX xxxx   ← or [CPU] if no GPU
=======================================================
  All dependencies satisfied. Ready to run!
```

---

## Running Experiments

### Run a single dataset

```bash
# MNIST (fastest — ~30 min on GPU)
bash scripts/run_single.sh mnist

# Fashion-MNIST
bash scripts/run_single.sh fmnist

# CIFAR-10 (slowest — ~2–4 hours on GPU)
bash scripts/run_single.sh cifar10
```

### Run all three datasets sequentially

```bash
bash scripts/run_all.sh
```

Logs are saved automatically to `results/<dataset>/run_<timestamp>.log`.

### Run directly with Python

```bash
cd src
python3 fl_mnist.py
python3 fl_fmnist.py
python3 fl_cifar10.py
```

---

## What Each Experiment Does

Each script runs a full federated learning simulation across:

- **7 aggregation methods**: Mean, Median, Trimmed-Mean, Krum, Multi-Krum, Bulyan, **HAD-FL**
- **4 Byzantine attacks**: ISA, MinMax, LIE, TRIM
- **4 malicious client ratios**: 10%, 20%, 30%, 40%
- **Plus a clean baseline**: No attack

Datasets are downloaded automatically on first run.

---

## Experiment Configuration

Hyperparameters are documented in `configs/`.
The scripts use their own internal constants — the YAML files
are provided for reference and future refactoring.

Key defaults (all datasets):

| Parameter | Value |
|-----------|-------|
| Total clients (N) | 100 |
| Clients per round (τ) | 10 |
| Global rounds | 150 |
| Local epochs | 2 |
| Batch size | 64 |
| Learning rate | 0.01 |
| Non-IID α (Dirichlet) | 0.5 |

---

## Output Files

Results are saved to `results/<dataset>/`:

```
Performance_<dataset>_<model>_<agg>_<attack>_<pct>pct.csv   ← accuracy, F1, loss per round
Robustness_<dataset>_<model>_<agg>_<attack>_<pct>pct.csv    ← accepted/rejected updates
System_<dataset>_<model>_<agg>_<attack>_<pct>pct.csv        ← timing overhead
Baseline_<dataset>_<model>_<agg>_NoAttack.csv               ← clean baseline
run_<timestamp>.log                                          ← full console output
```

See `docs/results_guide.md` for column descriptions and analysis examples.

---

## Analysing Results

```bash
python3 notebooks/analysis_template.py
```

This generates bar charts and training-curve plots from your output CSVs.

---

## Running Tests

```bash
pip install pytest
python3 -m pytest tests/ -v
```

---

## Manual Installation (no setup script)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# GPU (replace cu121 with your CUDA version, e.g. cu118)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
mkdir -p results/mnist results/fmnist results/cifar10
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: hdbscan` | `pip install hdbscan` |
| `CUDA out of memory` | Reduce `BATCH_SIZE` in the script (line ~50) |
| Script runs slowly | Confirm GPU is detected: `python3 scripts/check_env.py` |
| Datasets not downloading | Check internet connection; datasets download via `torchvision` |
| Permission denied on `.sh` files | `chmod +x scripts/*.sh` |

---

## License

This project is licensed under the terms in the `LICENSE` file.
