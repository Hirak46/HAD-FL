"""
HAD-FL | Result Analysis Template
Copy and adapt this script to analyse experiment output CSVs.
Run from the project root:  python3 notebooks/analysis_template.py
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), '..', 'results')

# ── Helper ────────────────────────────────────────────────────

def load_results(dataset: str, metric_type: str = "Performance") -> pd.DataFrame:
    """Load all CSVs of a given type for a dataset into one DataFrame."""
    pattern = os.path.join(RESULTS_ROOT, dataset, f"{metric_type}_*.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"[!] No files matched: {pattern}")
        return pd.DataFrame()

    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        # Parse metadata from filename
        parts = os.path.basename(fp).replace(".csv", "").split("_")
        df["aggregation"] = parts[3] if len(parts) > 3 else "?"
        df["attack"]      = parts[4] if len(parts) > 4 else "NoAttack"
        df["mal_pct"]     = parts[5] if len(parts) > 5 else "0pct"
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ── Example 1: Final accuracy by aggregation & attack ────────

def plot_final_accuracy(dataset: str):
    df = load_results(dataset, "Performance")
    if df.empty:
        return

    # Take last round per experiment
    final = df.groupby(["aggregation", "attack", "mal_pct"])["test_accuracy"].last().reset_index()

    plt.figure(figsize=(12, 5))
    sns.barplot(data=final, x="aggregation", y="test_accuracy", hue="attack")
    plt.title(f"{dataset.upper()} — Final Test Accuracy by Aggregation & Attack")
    plt.ylabel("Test Accuracy")
    plt.xlabel("Aggregation Method")
    plt.xticks(rotation=25)
    plt.tight_layout()
    out = os.path.join(RESULTS_ROOT, dataset, "final_accuracy_comparison.png")
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()


# ── Example 2: Accuracy over rounds for one experiment ───────

def plot_training_curve(dataset: str, agg: str, attack: str, mal_pct: str):
    df = load_results(dataset, "Performance")
    if df.empty:
        return

    subset = df[(df.aggregation == agg) & (df.attack == attack) & (df.mal_pct == mal_pct)]
    if subset.empty:
        print(f"[!] No data for {agg} / {attack} / {mal_pct}")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(subset["round"], subset["test_accuracy"], linewidth=2)
    plt.title(f"{dataset.upper()} | {agg} under {attack} ({mal_pct} malicious)")
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy")
    plt.tight_layout()
    plt.show()


# ── Run ───────────────────────────────────────────────────────

if __name__ == "__main__":
    for ds in ["mnist", "fmnist", "cifar10"]:
        plot_final_accuracy(ds)
