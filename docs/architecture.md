# HAD-FL — Architecture & Design Notes

## Overview

HAD-FL: Hierarchical Adaptive Aggregation for Defending Federated Learning Against Novel Heterogeneous Model Poisoning Attacks

## Core Components

### Attack Models (`src/fl_*.py`)

| Attack | Description |
|--------|-------------|
| **ISA** | Informed Statistical Attack — coordinate-wise importance-adaptive poisoning |
| **MinMax** | Scale malicious updates to maximise perturbation while evading norm-based defences |
| **LIE** | Little Is Enough — subtle updates within statistical bounds of honest clients |
| **TRIM** | Targeted at Trimmed-Mean — crafted to survive trimmed aggregation |

### Aggregation Methods

| Method | Category | Byzantine Tolerance |
|--------|----------|---------------------|
| Mean | Baseline | None |
| Median | Robust | Moderate |
| Trimmed-Mean | Robust | Moderate |
| Krum | Robust | High |
| Multi-Krum | Robust | High |
| Bulyan | Robust | High |
| **HADFL** | **Proposed** | **Highest** |

### HAD-FL Pipeline (5 Stages)

```
Stage 1: Adaptive Norm Clipping
         └─ Clip client updates to a dynamic norm budget

Stage 2: Directional Clustering (HDBSCAN + PCA)
         └─ Project updates to low-dim space, cluster by direction

Stage 3: MAD Layer-wise Scoring
         └─ Detect per-layer outliers via Median Absolute Deviation

Stage 4: Geometric Scoring
         └─ Measure geometric consistency of each update

Stage 5: Temporal Reputation Weighting
         └─ Maintain per-client reputation across rounds
             → Final weighted aggregation
```

## Dataset & Model Pairs

| Script | Dataset | Model | Image Size |
|--------|---------|-------|-----------|
| `fl_mnist.py` | MNIST | FourLayerCNN | 28×28 grayscale |
| `fl_fmnist.py` | Fashion-MNIST | FourLayerCNN | 28×28 grayscale |
| `fl_cifar10.py` | CIFAR-10 | MobileNetV2 | 32×32 RGB |

## Output Files (per experiment)

```
results/<dataset>/
├── Performance_<dataset>_<model>_<agg>_<attack>_<pct>pct.csv
├── Robustness_<dataset>_<model>_<agg>_<attack>_<pct>pct.csv
├── System_<dataset>_<model>_<agg>_<attack>_<pct>pct.csv
├── Baseline_<dataset>_<model>_<agg>_NoAttack.csv
└── run_<timestamp>.log
```

## Non-IID Data Partitioning

Federated data is split using a **Dirichlet distribution** (`α = 0.5`).
Lower `α` → more heterogeneous (realistic) distributions across clients.
