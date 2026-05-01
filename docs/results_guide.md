# HAD-FL — Reading Experiment Results

## Output CSV Files

After running an experiment, three CSV files are generated per
(aggregation method × attack × malicious ratio) combination.

### 1. `Performance_*.csv`

Round-by-round training metrics.

| Column | Description |
|--------|-------------|
| `round` | FL global round number |
| `test_accuracy` | Global model accuracy on test set |
| `test_loss` | Cross-entropy loss |
| `top5_accuracy` | Top-5 accuracy (CIFAR-10 only) |
| `precision` | Macro-averaged precision |
| `recall` | Macro-averaged recall |
| `f1_score` | Macro-averaged F1 |

### 2. `Robustness_*.csv`

Attack-specific robustness metrics.

| Column | Description |
|--------|-------------|
| `round` | FL global round number |
| `num_malicious` | Number of malicious clients selected |
| `updates_accepted` | Updates accepted by aggregator |
| `updates_rejected` | Updates rejected by aggregator |

### 3. `System_*.csv`

Computational overhead metrics.

| Column | Description |
|--------|-------------|
| `round` | FL global round number |
| `aggregation_time_s` | Wall-clock time for aggregation step |
| `training_time_s` | Wall-clock time for local training |

## Naming Convention

```
Performance_CIFAR-10_MobileNetV2_HADFL_ISA_20pct.csv
             │         │           │     │    └─ malicious ratio
             │         │           │     └─ attack type
             │         │           └─ aggregation method
             │         └─ model name
             └─ dataset name
```

## Quick Analysis (Python)

```python
import pandas as pd
import glob

# Load all HADFL results for CIFAR-10 under ISA attack
files = glob.glob("results/cifar10/Performance_CIFAR-10_MobileNetV2_HADFL_ISA_*.csv")
dfs = {f.split("_")[-1].replace(".csv",""):  pd.read_csv(f) for f in files}

for pct, df in sorted(dfs.items()):
    final_acc = df["test_accuracy"].iloc[-1]
    print(f"  ISA {pct} malicious → Final accuracy: {final_acc:.4f}")
```
